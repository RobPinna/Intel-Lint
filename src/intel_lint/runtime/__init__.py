from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_ENGINE = "ollama"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "FenkoHQ/Foundation-Sec-8B"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _fallback_user_data_dir(app_name: str) -> Path:
    home = Path.home()
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or (home / "AppData" / "Local"))
        return base / app_name
    if sys_platform() == "darwin":
        return home / "Library" / "Application Support" / app_name
    base = Path(os.getenv("XDG_DATA_HOME", home / ".local" / "share"))
    return base / app_name


def sys_platform() -> str:
    # Keep tiny wrapper to simplify tests/mocking when needed.
    return os.sys.platform


def _ensure_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_data_dir() -> Path:
    override = os.getenv("INTEL_LINT_DATA_DIR", "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        if _ensure_dir(path):
            return path

    app_name = "intel-lint"
    try:
        from platformdirs import user_data_dir  # type: ignore

        preferred = Path(user_data_dir(app_name, "IntelLint")).expanduser().resolve()
    except Exception:
        preferred = _fallback_user_data_dir(app_name).expanduser().resolve()

    for candidate in (
        preferred,
        (Path.home() / f".{app_name}").expanduser().resolve(),
    ):
        if _ensure_dir(candidate):
            return candidate

    return (Path.home() / f".{app_name}").expanduser().resolve()


@lru_cache(maxsize=1)
def get_default_paths() -> dict[str, Path]:
    data_dir = get_data_dir()
    outputs_dir = data_dir / "outputs"
    logs_dir = data_dir / "logs"
    config_dir = data_dir / "config"
    latest_outputs_dir = outputs_dir / "latest"
    cache_dir = outputs_dir / "cache"
    settings_file = config_dir / "settings.json"
    log_file = logs_dir / "app.log"

    for path in (outputs_dir, logs_dir, config_dir, latest_outputs_dir, cache_dir):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    return {
        "data_dir": data_dir,
        "outputs_dir": outputs_dir,
        "latest_outputs_dir": latest_outputs_dir,
        "cache_dir": cache_dir,
        "logs_dir": logs_dir,
        "config_dir": config_dir,
        "settings_file": settings_file,
        "log_file": log_file,
    }


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and ((value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'"))):
            value = value[1:-1]
        values[key] = value
    return values


def _normalize_settings_payload(payload: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in payload.items():
        if value is None:
            continue
        normalized[str(key).strip().upper()] = str(value).strip()
    return normalized


def _apply_aliases(settings: dict[str, str], defaults: dict[str, str]) -> dict[str, str]:
    resolved = dict(defaults)
    resolved.update(settings)

    if resolved.get("OLLAMA_HOST", "").strip() and not resolved.get("OLLAMA_URL", "").strip():
        resolved["OLLAMA_URL"] = resolved["OLLAMA_HOST"].strip()
    if resolved.get("OLLAMA_URL", "").strip() and not resolved.get("OLLAMA_HOST", "").strip():
        resolved["OLLAMA_HOST"] = resolved["OLLAMA_URL"].strip()

    if resolved.get("OLLAMA_MODEL", "").strip() and not resolved.get("MODEL", "").strip():
        resolved["MODEL"] = resolved["OLLAMA_MODEL"].strip()
    if resolved.get("MODEL", "").strip() and not resolved.get("OLLAMA_MODEL", "").strip():
        resolved["OLLAMA_MODEL"] = resolved["MODEL"].strip()

    return resolved


def load_settings() -> dict[str, str]:
    paths = get_default_paths()
    defaults = {
        "ENGINE": DEFAULT_ENGINE,
        "OLLAMA_URL": DEFAULT_OLLAMA_URL,
        "MODEL": DEFAULT_MODEL,
        "OUTPUT_DIR": str(paths["latest_outputs_dir"]),
        "VERBOSE_LOGGING": "0",
    }

    merged: dict[str, str] = {}

    dotenv_values = _normalize_settings_payload(_read_dotenv(_REPO_ROOT / ".env"))
    merged.update(dotenv_values)

    settings_file = paths["settings_file"]
    if settings_file.exists():
        try:
            raw = json.loads(settings_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                merged.update(_normalize_settings_payload(raw))
        except Exception:
            # Keep runtime resilient even if local settings are malformed.
            pass

    merged.update(_normalize_settings_payload(dict(os.environ)))
    resolved = _apply_aliases(merged, defaults)

    output_dir = Path(resolved.get("OUTPUT_DIR", defaults["OUTPUT_DIR"])).expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    return {
        "engine": resolved.get("ENGINE", defaults["ENGINE"]).strip().lower() or DEFAULT_ENGINE,
        "ollama_url": resolved.get("OLLAMA_URL", defaults["OLLAMA_URL"]).strip() or DEFAULT_OLLAMA_URL,
        "model": resolved.get("MODEL", defaults["MODEL"]).strip() or DEFAULT_MODEL,
        "output_dir": str(output_dir),
        "verbose_logging": _is_truthy(resolved.get("VERBOSE_LOGGING", defaults["VERBOSE_LOGGING"])),
        "data_dir": str(paths["data_dir"]),
        "outputs_dir": str(paths["outputs_dir"]),
        "cache_dir": str(paths["cache_dir"]),
        "logs_dir": str(paths["logs_dir"]),
        "config_dir": str(paths["config_dir"]),
        "settings_file": str(settings_file),
        "log_file": str(paths["log_file"]),
    }


def configure_file_logging(level: int = logging.INFO) -> Path:
    settings = load_settings()
    effective_level = logging.DEBUG if settings.get("verbose_logging", False) else level
    log_path = Path(settings["log_file"])
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return log_path

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == log_path.resolve():
                    return log_path
            except Exception:
                continue

    try:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError:
        return log_path
    file_handler.setLevel(effective_level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root_logger.addHandler(file_handler)

    if root_logger.level == logging.NOTSET or root_logger.level > effective_level:
        root_logger.setLevel(effective_level)

    return log_path
