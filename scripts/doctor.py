from __future__ import annotations

import importlib
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intel_lint.runtime import get_data_dir, load_settings


def _check_python_version() -> tuple[bool, str]:
    ok = sys.version_info >= (3, 11)
    detail = f"Python {sys.version.split()[0]} (requires >= 3.11)"
    return ok, detail


def _check_dependencies() -> tuple[bool, str]:
    modules = ["fastapi", "httpx", "pydantic", "uvicorn"]
    missing: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)

    if missing:
        return False, f"missing imports: {', '.join(missing)}"
    return True, "required Python dependencies import successfully"


def _check_data_dir_writable() -> tuple[bool, str]:
    data_dir = get_data_dir()
    probe = data_dir / ".doctor_write_test"
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True, f"data dir writable: {data_dir}"
    except OSError as exc:
        return False, f"data dir is not writable: {data_dir} ({exc})"


def _check_ollama_if_enabled() -> tuple[bool, str]:
    settings = load_settings()
    engine = settings.get("engine", "ollama").strip().lower()
    if engine != "ollama":
        return True, f"ENGINE={engine} (ollama reachability check skipped)"

    ollama_url = settings.get("ollama_url", "http://localhost:11434").strip().rstrip("/")
    tags_url = f"{ollama_url}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=4) as response:
            ok = 200 <= int(response.status) < 300
        if ok:
            return True, f"Ollama reachable at {ollama_url}"
        return False, f"Ollama check returned non-success status at {ollama_url}"
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return False, f"Ollama unreachable at {ollama_url} ({exc})"


def main() -> int:
    checks = [
        ("python version", _check_python_version),
        ("dependency imports", _check_dependencies),
        ("data dir writable", _check_data_dir_writable),
        ("ollama reachability", _check_ollama_if_enabled),
    ]

    failures = 0
    for name, fn in checks:
        ok, detail = fn()
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
