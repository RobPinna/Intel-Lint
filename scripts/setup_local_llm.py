from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intel_lint.runtime import DEFAULT_MODEL, DEFAULT_OLLAMA_URL, get_default_paths, load_settings


def _run(cmd: list[str]) -> int:
    print(f"+ {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=ROOT_DIR)
    return completed.returncode


def _check_ollama_cli() -> int:
    if shutil.which("ollama") is None:
        print("error: 'ollama' CLI not found in PATH.", file=sys.stderr)
        print("Install Ollama first: https://ollama.com/download", file=sys.stderr)
        return 1
    return _run(["ollama", "--version"])


def _check_ollama_service(ollama_url: str) -> bool:
    tags_url = f"{ollama_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=4) as response:
            return 200 <= int(response.status) < 300
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {}


def _write_local_settings(ollama_url: str, model: str) -> Path:
    paths = get_default_paths()
    settings_path = Path(paths["settings_file"])
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _read_json(settings_path)
    payload["ENGINE"] = "ollama"
    payload["OLLAMA_URL"] = ollama_url
    payload["MODEL"] = model

    settings_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return settings_path


def main() -> int:
    current = load_settings()
    parser = argparse.ArgumentParser(description="Set up Intel Lint for local-first Ollama usage.")
    parser.add_argument("--ollama-url", default=current.get("ollama_url") or DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", default=current.get("model") or DEFAULT_MODEL)
    args = parser.parse_args()

    ollama_url = (args.ollama_url or DEFAULT_OLLAMA_URL).strip()
    model = (args.model or DEFAULT_MODEL).strip()

    code = _check_ollama_cli()
    if code != 0:
        return code

    if not _check_ollama_service(ollama_url):
        print(f"error: Ollama service is not reachable at {ollama_url}", file=sys.stderr)
        print("Start Ollama and retry. Example: ollama serve", file=sys.stderr)
        return 1

    code = _run(["ollama", "pull", model])
    if code != 0:
        return code

    settings_path = _write_local_settings(ollama_url, model)
    settings = load_settings()
    print(f"updated local settings: {settings_path}")
    print("")
    print("Next steps:")
    print("1) Run diagnostics:")
    print("   python scripts/run.py doctor")
    print("2) CLI sample run:")
    print(f"   python -m intel_lint.cli.main samples/sample1.txt --out \"{settings['output_dir']}\"")
    print("3) UI run:")
    print("   python scripts/run.py setup --frontend")
    print("   python scripts/run.py app")
    print("   Frontend: http://127.0.0.1:5173")
    print("   API:      http://127.0.0.1:8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
