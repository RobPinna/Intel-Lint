from __future__ import annotations

import io
import logging
import os
import zipfile
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .api_setup import router as setup_router
from .core.engine import analyze_with_selected_engine
from .core.ollama import _ollama_processor_state
from .frontend_assets import locate_frontend_dist
from .io.outputs import write_latest_outputs
from .models.schemas import AnalyzeRequest, AnalyzeResponse
from .runtime import configure_file_logging, load_settings

ROOT_DIR = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
OUTPUTS_DIR = Path(SETTINGS["output_dir"])
FRONTEND_DIST_DIR = locate_frontend_dist(ROOT_DIR)
DEFAULT_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
DEFAULT_DEV_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
DEFAULT_OLLAMA_HOST = SETTINGS["ollama_url"]

app = FastAPI(title="Intel Lint API", version="0.1.0")
configure_file_logging()
logger = logging.getLogger(__name__)
logger.info(
    "api_start engine_default=%s output_dir=%s log_file=%s",
    SETTINGS["engine"],
    SETTINGS["output_dir"],
    SETTINGS["log_file"],
)


def _cors_origins_from_env() -> list[str] | None:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if not raw:
        return None
    values = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    return values or None


debug_mode = os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
configured_origins = _cors_origins_from_env()
cors_origins = ["*"] if debug_mode else (configured_origins or DEFAULT_DEV_ORIGINS)
cors_origin_regex = None if debug_mode or configured_origins else DEFAULT_DEV_ORIGIN_REGEX
cors_allow_credentials = not debug_mode

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_origin_regex,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(setup_router)

if FRONTEND_DIST_DIR:
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.exists() and assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ollama")
def health_ollama() -> dict[str, Any]:
    host = os.getenv("OLLAMA_URL", "").strip() or os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).strip() or DEFAULT_OLLAMA_HOST
    host = host.rstrip("/")
    ps_url = f"{host}/api/ps"
    tags_url = f"{host}/api/tags"

    try:
        with httpx.Client(timeout=5.0) as client:
            ps_resp = client.get(ps_url)
            tags_resp = client.get(tags_url)
        ps_resp.raise_for_status()
        tags_resp.raise_for_status()
    except Exception as exc:
        return {
            "reachable": False,
            "host": host,
            "gpu_active": False,
            "processor_summary": "unreachable",
            "error": str(exc),
        }

    ps_data = ps_resp.json() if ps_resp.content else {}
    models = ps_data.get("models", []) if isinstance(ps_data, dict) else []
    state = _ollama_processor_state()
    gpu_active = bool(state.get("gpu_active", False))
    processor_summary = str(state.get("processor_summary", "unknown"))

    return {
        "reachable": True,
        "host": host,
        "gpu_active": gpu_active,
        "processor_summary": processor_summary,
        "running_models": len(models),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        response = analyze_with_selected_engine(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    write_latest_outputs(response, OUTPUTS_DIR)
    logger.info(
        "api_analyze_complete engine=%s claims=%d output_dir=%s",
        response.claims.engine,
        len(response.claims.claims),
        OUTPUTS_DIR,
    )
    return response


@app.get("/download/latest")
def download_latest() -> StreamingResponse:
    required_files = ["claims.json", "annotated.md", "rewrite.md"]
    missing = [name for name in required_files if not (OUTPUTS_DIR / name).exists()]
    if missing:
        raise HTTPException(status_code=404, detail=f"Missing output files: {', '.join(missing)}")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in required_files:
            zf.write(OUTPUTS_DIR / name, arcname=name)

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=intel-lint-latest.zip"},
    )


def _resolve_frontend_file(path: str) -> Path | None:
    if not FRONTEND_DIST_DIR:
        return None
    candidate = (FRONTEND_DIST_DIR / path).resolve()
    if FRONTEND_DIST_DIR.resolve() not in candidate.parents and candidate != FRONTEND_DIST_DIR.resolve():
        return None
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


@app.get("/", include_in_schema=False)
def frontend_index() -> FileResponse | PlainTextResponse:
    if not FRONTEND_DIST_DIR:
        return PlainTextResponse("Frontend build not found. Use Vite dev server or build frontend/dist first.", status_code=404)
    index_file = FRONTEND_DIST_DIR / "index.html"
    if not index_file.exists():
        return PlainTextResponse("Frontend index.html not found in dist directory.", status_code=404)
    return FileResponse(index_file)


@app.get("/{full_path:path}", include_in_schema=False)
def frontend_spa_fallback(full_path: str):
    normalized = (full_path or "").lstrip("/")
    if normalized == "api" or normalized.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    if not FRONTEND_DIST_DIR:
        raise HTTPException(status_code=404, detail="Not found")

    target = _resolve_frontend_file(normalized)
    if target is not None:
        return FileResponse(target)

    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Not found")
