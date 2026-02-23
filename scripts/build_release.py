from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
FRONTEND_LOCKFILE = FRONTEND_DIR / "package-lock.json"
SPEC_FILE = ROOT_DIR / "IntelLint.spec"

DIST_RELEASE_DIR = ROOT_DIR / "dist_release"
PYI_WORK_DIR = ROOT_DIR / "build" / "pyinstaller-work"

NODE_BIN = ""
NPM_BIN = ""


def _format_cmd(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"+ {_format_cmd(cmd)}")
    try:
        completed = subprocess.run(cmd, cwd=cwd or ROOT_DIR)
    except FileNotFoundError:
        print(f"error: command not found: {cmd[0]}", file=sys.stderr)
        return 127
    return completed.returncode


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _ensure_node_npm() -> int:
    global NODE_BIN, NPM_BIN
    NODE_BIN = shutil.which("node") or ""
    NPM_BIN = shutil.which("npm") or ""
    if not NODE_BIN:
        return _fail("Node.js is required but 'node' was not found in PATH.")
    if not NPM_BIN:
        return _fail("npm is required but 'npm' was not found in PATH.")
    return 0


def _build_frontend() -> int:
    if not FRONTEND_DIR.exists():
        return _fail(f"frontend directory not found: {FRONTEND_DIR}")

    install_cmd = [NPM_BIN, "ci"] if FRONTEND_LOCKFILE.exists() else [NPM_BIN, "install"]
    code = _run(install_cmd, cwd=FRONTEND_DIR)
    if code != 0:
        return _fail("frontend dependency install failed.")

    code = _run([NPM_BIN, "run", "build"], cwd=FRONTEND_DIR)
    if code != 0:
        return _fail("frontend build failed.")

    index_file = FRONTEND_DIST_DIR / "index.html"
    if not index_file.exists():
        return _fail(f"frontend build output missing: {index_file}")
    return 0


def _ensure_pyinstaller() -> int:
    code = _run([sys.executable, "-m", "PyInstaller", "--version"])
    if code != 0:
        return _fail("PyInstaller is not available. Install it in your build environment.")
    return 0


def _run_pyinstaller() -> int:
    if not SPEC_FILE.exists():
        return _fail(f"spec file not found: {SPEC_FILE}")
    DIST_RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    PYI_WORK_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(DIST_RELEASE_DIR),
        "--workpath",
        str(PYI_WORK_DIR),
        str(SPEC_FILE),
    ]
    code = _run(cmd)
    if code != 0:
        return _fail("PyInstaller build failed.")
    return 0


def main() -> int:
    code = _ensure_node_npm()
    if code != 0:
        return code

    code = _build_frontend()
    if code != 0:
        return code

    code = _ensure_pyinstaller()
    if code != 0:
        return code

    code = _run_pyinstaller()
    if code != 0:
        return code

    exe_name = "IntelLint.exe" if os.name == "nt" else "IntelLint"
    artifact = DIST_RELEASE_DIR / exe_name
    if artifact.exists():
        print(f"build complete: {artifact}")
    else:
        print(f"build complete. artifact directory: {DIST_RELEASE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
