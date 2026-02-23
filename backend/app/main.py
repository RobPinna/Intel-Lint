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
from fastapi.responses import StreamingResponse

from .engine import analyze_with_selected_engine
from .engine_ollama import _ollama_processor_state
from .engine_placeholder import write_latest_outputs
from .models import AnalyzeRequest, AnalyzeResponse
from intel_lint.runtime import configure_file_logging, load_settings

ROOT_DIR = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
OUTPUTS_DIR = Path(SETTINGS["output_dir"])
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
