# Intel Lint

Intel Lint is a local-first analyzer for CTI report review (claims extraction, evidence checks, bias signals).
It is designed for single-user local processing of sensitive reports.

## For Users (Recommended)
1. Install Ollama: https://ollama.com/download
2. Download IntelLint from GitHub Releases for your platform.
3. Run IntelLint.
   - The app binds to `127.0.0.1` and opens the local web UI automatically.
4. In the Setup page, click **Pull `FenkoHQ/Foundation-Sec-8B`**.
5. Use the app from the local UI.

Local storage (user data dir):
- Windows: `%LOCALAPPDATA%\\intel-lint`
- macOS: `~/Library/Application Support/intel-lint`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/intel-lint`

The user data dir stores local config, outputs, logs, and cache.

## Privacy & local-first
- In `ENGINE=ollama` mode, model inference is local through your Ollama service.
- No telemetry is required for normal local operation.
- This is intended for sensitive CTI reporting workflows where data should remain local.

## For Developers
```bash
python -m pip install -e ".[dev]"
python scripts/run.py setup --frontend
python scripts/run.py app
python scripts/build_release.py
```

Notes:
- `npm install` is dev-only and is handled by `python scripts/run.py setup --frontend`.
- The dev UI is `http://127.0.0.1:5173` and proxies API calls to the local backend.

## Placeholder Mode (Tests/Smoke Only)
Placeholder engine is for CI/tests/smoke checks only; output quality is limited versus LLM mode.

```bash
python -m pytest
```

## Local Data Wipe
1. Close IntelLint.
2. Delete the `intel-lint` user data directory for your OS path above.

## Developer Notes
- `.env.example` is optional for development overrides.
