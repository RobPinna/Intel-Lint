from __future__ import annotations

import importlib.metadata
from pathlib import Path
import tomllib


def _version_from_pyproject() -> str | None:
    pyproject_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject_file.exists():
        return None
    try:
        payload = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    project = payload.get("project")
    if not isinstance(project, dict):
        return None
    version = project.get("version")
    if isinstance(version, str) and version.strip():
        return version.strip()
    return None


def _resolve_version() -> str:
    try:
        return importlib.metadata.version("intel-lint")
    except Exception:
        from_pyproject = _version_from_pyproject()
        if from_pyproject:
            return from_pyproject
        return "0.1.0"


__version__ = _resolve_version()

__all__ = ["__version__", "app"]


def __getattr__(name: str):
    if name == "app":
        from .api import app

        return app
    raise AttributeError(name)
