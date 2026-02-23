from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import __version__
from .runtime import DEFAULT_OLLAMA_URL, get_default_paths, load_settings


router = APIRouter(prefix="/api", tags=["setup"])

_PULL_JOBS: dict[str, dict[str, Any]] = {}
_PULL_JOBS_LOCK = threading.Lock()


class SetupSettingsUpdate(BaseModel):
    ENGINE: str | None = None
    OLLAMA_URL: str | None = None
    MODEL: str | None = None


class OllamaPullRequest(BaseModel):
    model: str | None = None
    ollama_url: str | None = None


def _effective_settings_payload() -> dict[str, Any]:
    settings = load_settings()
    return {
        "ENGINE": settings["engine"],
        "OLLAMA_URL": settings["ollama_url"],
        "MODEL": settings["model"],
        "OUTPUT_DIR": settings["output_dir"],
        "DATA_DIR": settings["data_dir"],
        "CONFIG_DIR": settings["config_dir"],
        "SETTINGS_FILE": settings["settings_file"],
    }


def _read_local_settings_file(settings_file: Path) -> dict[str, Any]:
    if not settings_file.exists():
        return {}
    try:
        payload = json.loads(settings_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def _write_local_settings(update: SetupSettingsUpdate) -> dict[str, Any]:
    paths = get_default_paths()
    settings_file = Path(paths["settings_file"])
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    payload = _read_local_settings_file(settings_file)

    if update.ENGINE is not None:
        engine = update.ENGINE.strip().lower()
        if engine not in {"ollama", "placeholder"}:
            raise HTTPException(status_code=400, detail="ENGINE must be one of: ollama, placeholder")
        payload["ENGINE"] = engine

    if update.OLLAMA_URL is not None:
        ollama_url = update.OLLAMA_URL.strip().rstrip("/")
        if not ollama_url:
            raise HTTPException(status_code=400, detail="OLLAMA_URL must not be empty")
        payload["OLLAMA_URL"] = ollama_url
        payload["OLLAMA_HOST"] = ollama_url

    if update.MODEL is not None:
        model = update.MODEL.strip()
        if not model:
            raise HTTPException(status_code=400, detail="MODEL must not be empty")
        payload["MODEL"] = model
        payload["OLLAMA_MODEL"] = model

    settings_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return _effective_settings_payload()


def _check_ollama_status(ollama_url: str, model: str) -> dict[str, Any]:
    tags_url = f"{ollama_url.rstrip('/')}/api/tags"
    try:
        with httpx.Client(timeout=4.0) as client:
            response = client.get(tags_url)
            response.raise_for_status()
            payload = response.json() if response.content else {}
    except Exception as exc:
        return {
            "reachable": False,
            "ollama_url": ollama_url,
            "model": model,
            "model_present": False,
            "details": f"Ollama unreachable: {exc}",
        }

    models = payload.get("models", []) if isinstance(payload, dict) else []
    model_present = False
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        if name == model or name.startswith(f"{model}:") or name.split(":", 1)[0] == model:
            model_present = True
            break

    details = "model found locally" if model_present else "model not found locally (pull required)"
    return {
        "reachable": True,
        "ollama_url": ollama_url,
        "model": model,
        "model_present": model_present,
        "details": details,
    }


def _set_pull_job(job_id: str, **updates: Any) -> None:
    with _PULL_JOBS_LOCK:
        job = _PULL_JOBS.setdefault(job_id, {})
        job.update(updates)


def _get_pull_job(job_id: str) -> dict[str, Any] | None:
    with _PULL_JOBS_LOCK:
        job = _PULL_JOBS.get(job_id)
        return dict(job) if job is not None else None


def _run_ollama_pull_job(job_id: str, model: str, ollama_url: str) -> None:
    _set_pull_job(job_id, state="running", last_line="starting pull", exit_code=None)
    env = os.environ.copy()
    env["OLLAMA_HOST"] = ollama_url
    env["OLLAMA_URL"] = ollama_url

    try:
        process = subprocess.Popen(
            ["ollama", "pull", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except Exception as exc:
        _set_pull_job(job_id, state="error", last_line=f"failed to start pull: {exc}", exit_code=None)
        return

    last_line = ""
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        last_line = line[-400:]
        _set_pull_job(job_id, last_line=last_line)

    exit_code = process.wait()
    if exit_code == 0:
        _set_pull_job(job_id, state="done", last_line=last_line or "pull complete", exit_code=0)
    else:
        _set_pull_job(
            job_id,
            state="error",
            last_line=last_line or f"ollama pull failed with exit code {exit_code}",
            exit_code=exit_code,
        )


@router.get("/health")
def api_health() -> dict[str, Any]:
    settings = load_settings()
    return {
        "ok": True,
        "version": __version__,
        "engine": settings["engine"],
        "model": settings["model"],
        "ollama_url": settings["ollama_url"],
        "data_dir": settings["data_dir"],
    }


@router.get("/setup/settings")
def get_setup_settings() -> dict[str, Any]:
    return _effective_settings_payload()


@router.post("/setup/settings")
def post_setup_settings(update: SetupSettingsUpdate) -> dict[str, Any]:
    return _write_local_settings(update)


@router.get("/setup/ollama/status")
def get_ollama_status() -> dict[str, Any]:
    settings = load_settings()
    ollama_url = settings.get("ollama_url", DEFAULT_OLLAMA_URL).strip() or DEFAULT_OLLAMA_URL
    model = settings.get("model", "").strip()
    return _check_ollama_status(ollama_url=ollama_url, model=model)


@router.post("/setup/ollama/pull")
def post_ollama_pull(request: OllamaPullRequest) -> dict[str, Any]:
    if shutil.which("ollama") is None:
        raise HTTPException(
            status_code=400,
            detail="Ollama CLI not found in PATH. Install Ollama and retry.",
        )

    settings = load_settings()
    ollama_url = (request.ollama_url or settings.get("ollama_url") or DEFAULT_OLLAMA_URL).strip().rstrip("/")
    model = (request.model or settings.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="Model is required.")

    job_id = uuid.uuid4().hex
    _set_pull_job(
        job_id,
        state="queued",
        last_line="queued",
        exit_code=None,
        model=model,
        ollama_url=ollama_url,
    )
    thread = threading.Thread(target=_run_ollama_pull_job, args=(job_id, model, ollama_url), daemon=True)
    thread.start()
    return {"job_id": job_id, "state": "queued", "last_line": "queued"}


@router.get("/setup/ollama/pull/{job_id}")
def get_ollama_pull(job_id: str) -> dict[str, Any]:
    job = _get_pull_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")

    response = {
        "state": str(job.get("state", "error")),
        "last_line": str(job.get("last_line", "")),
    }
    exit_code = job.get("exit_code")
    if exit_code is not None:
        response["exit_code"] = int(exit_code)
    return response
