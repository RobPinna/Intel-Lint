from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_PACKAGE = FRONTEND_DIR / "package.json"
PYPROJECT = ROOT_DIR / "pyproject.toml"
REQUIREMENTS = ROOT_DIR / "requirements.txt"

API_ENTRYPOINT = "intel_lint.api:app"
API_HOST_DEFAULT = "127.0.0.1"
API_PORT_DEFAULT = 8000
UI_HOST_DEFAULT = "127.0.0.1"
UI_PORT_DEFAULT = 5173


def _venv_python() -> Path:
    if os.name == "nt":
        return ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    return ROOT_DIR / ".venv" / "bin" / "python"


def _active_python() -> str:
    venv_python = _venv_python()
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _format_cmd(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    workdir = cwd or ROOT_DIR
    print(f"+ {_format_cmd(cmd)}")
    completed = subprocess.run(cmd, cwd=workdir, env=env)
    return completed.returncode


def _popen(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.Popen[str]:
    workdir = cwd or ROOT_DIR
    print(f"+ {_format_cmd(cmd)}")
    return subprocess.Popen(cmd, cwd=workdir, env=env)


def _terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _ensure_npm() -> bool:
    if shutil.which("npm"):
        return True
    print("error: npm is required for UI commands and was not found in PATH.", file=sys.stderr)
    return False


def _frontend_has_dev_script() -> bool:
    if not FRONTEND_PACKAGE.exists():
        return False
    try:
        data = json.loads(FRONTEND_PACKAGE.read_text(encoding="utf-8"))
    except Exception:
        return False
    scripts = data.get("scripts")
    return isinstance(scripts, dict) and "dev" in scripts


def _api_command(host: str, port: int, reload: bool) -> list[str]:
    cmd = [
        _active_python(),
        "-m",
        "uvicorn",
        API_ENTRYPOINT,
        "--app-dir",
        "src",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        cmd.append("--reload")
    return cmd


def _ui_command(host: str, port: int) -> list[str]:
    return [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        host,
        "--port",
        str(port),
    ]


def cmd_setup(args: argparse.Namespace) -> int:
    if args.venv and not _venv_python().exists():
        code = _run([sys.executable, "-m", "venv", ".venv"])
        if code != 0:
            return code

    python_exec = _active_python()
    if PYPROJECT.exists():
        code = _run([python_exec, "-m", "pip", "install", "-e", ".[dev]"])
    elif REQUIREMENTS.exists():
        code = _run([python_exec, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    else:
        print("error: no pyproject.toml or requirements.txt found for dependency install.", file=sys.stderr)
        return 2
    if code != 0:
        return code

    if args.frontend:
        if not FRONTEND_PACKAGE.exists():
            print("error: frontend/package.json not found.", file=sys.stderr)
            return 2
        if not _ensure_npm():
            return 2
        return _run(["npm", "install"], cwd=FRONTEND_DIR)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    doctor_script = ROOT_DIR / "scripts" / "doctor.py"
    if not doctor_script.exists():
        print("error: scripts/doctor.py not found.", file=sys.stderr)
        return 2
    extra = list(args.doctor_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    return _run([_active_python(), str(doctor_script), *extra], cwd=ROOT_DIR)


def cmd_api(args: argparse.Namespace) -> int:
    return _run(_api_command(args.host, args.port, args.reload), cwd=ROOT_DIR, env=os.environ.copy())


def cmd_ui(args: argparse.Namespace) -> int:
    if not FRONTEND_PACKAGE.exists():
        print("error: frontend/package.json not found.", file=sys.stderr)
        return 2
    if not _frontend_has_dev_script():
        print("error: frontend package does not define an npm 'dev' script.", file=sys.stderr)
        return 2
    if not _ensure_npm():
        return 2
    return _run(_ui_command(args.host, args.port), cwd=FRONTEND_DIR, env=os.environ.copy())


def cmd_app(args: argparse.Namespace) -> int:
    if args.no_ui:
        api_args = argparse.Namespace(host=args.api_host, port=args.api_port, reload=args.reload)
        return cmd_api(api_args)

    if not FRONTEND_PACKAGE.exists():
        print("error: frontend/package.json not found; use --no-ui to run API only.", file=sys.stderr)
        return 2
    if not _frontend_has_dev_script():
        print("error: frontend package does not define an npm 'dev' script.", file=sys.stderr)
        return 2
    if not _ensure_npm():
        return 2

    node_modules_dir = FRONTEND_DIR / "node_modules"
    if not node_modules_dir.exists():
        code = _run(["npm", "install"], cwd=FRONTEND_DIR, env=os.environ.copy())
        if code != 0:
            return code

    backend_cmd = _api_command(args.api_host, args.api_port, args.reload)
    frontend_cmd = _ui_command(args.ui_host, args.ui_port)

    print(f"API URL: http://{args.api_host}:{args.api_port}")
    print(f"UI URL:  http://{args.ui_host}:{args.ui_port}")
    backend_proc = _popen(backend_cmd, cwd=ROOT_DIR, env=os.environ.copy())
    frontend_proc = _popen(frontend_cmd, cwd=FRONTEND_DIR, env=os.environ.copy())

    try:
        while True:
            backend_code = backend_proc.poll()
            frontend_code = frontend_proc.poll()
            if backend_code is not None:
                _terminate(frontend_proc)
                return backend_code
            if frontend_code is not None:
                _terminate(backend_proc)
                return frontend_code
            time.sleep(0.5)
    except KeyboardInterrupt:
        _terminate(frontend_proc)
        _terminate(backend_proc)
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-platform local runner for Intel Lint.")
    sub = parser.add_subparsers(dest="command", required=True)

    setup_parser = sub.add_parser("setup", help="Install Python dependencies.")
    setup_parser.add_argument("--venv", action="store_true", help="Create .venv first if missing.")
    setup_parser.add_argument("--frontend", action="store_true", help="Also install frontend npm dependencies.")
    setup_parser.set_defaults(func=cmd_setup)

    doctor_parser = sub.add_parser("doctor", help="Run local environment diagnostics.")
    doctor_parser.add_argument("doctor_args", nargs=argparse.REMAINDER, help="Arguments passed through to scripts/doctor.py.")
    doctor_parser.set_defaults(func=cmd_doctor)

    api_parser = sub.add_parser("api", help="Run backend API (uvicorn).")
    api_parser.add_argument("--host", default=API_HOST_DEFAULT, help=f"API host (default: {API_HOST_DEFAULT}).")
    api_parser.add_argument("--port", default=API_PORT_DEFAULT, type=int, help=f"API port (default: {API_PORT_DEFAULT}).")
    api_parser.add_argument("--reload", dest="reload", action="store_true", default=True, help="Enable autoreload (default).")
    api_parser.add_argument("--no-reload", dest="reload", action="store_false", help="Disable autoreload.")
    api_parser.set_defaults(func=cmd_api)

    ui_parser = sub.add_parser("ui", help="Run frontend dev server.")
    ui_parser.add_argument("--host", default=UI_HOST_DEFAULT, help=f"UI host (default: {UI_HOST_DEFAULT}).")
    ui_parser.add_argument("--port", default=UI_PORT_DEFAULT, type=int, help=f"UI port (default: {UI_PORT_DEFAULT}).")
    ui_parser.set_defaults(func=cmd_ui)

    app_parser = sub.add_parser("app", help="Run API + UI together.")
    app_parser.add_argument("--api-host", default=API_HOST_DEFAULT, help=f"API host (default: {API_HOST_DEFAULT}).")
    app_parser.add_argument("--api-port", default=API_PORT_DEFAULT, type=int, help=f"API port (default: {API_PORT_DEFAULT}).")
    app_parser.add_argument("--ui-host", default=UI_HOST_DEFAULT, help=f"UI host (default: {UI_HOST_DEFAULT}).")
    app_parser.add_argument("--ui-port", default=UI_PORT_DEFAULT, type=int, help=f"UI port (default: {UI_PORT_DEFAULT}).")
    app_parser.add_argument("--reload", dest="reload", action="store_true", default=True, help="Enable API autoreload (default).")
    app_parser.add_argument("--no-reload", dest="reload", action="store_false", help="Disable API autoreload.")
    app_parser.add_argument("--no-ui", action="store_true", help="Run API only.")
    app_parser.set_defaults(func=cmd_app)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
