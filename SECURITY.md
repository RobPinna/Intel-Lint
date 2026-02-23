# Security

## Local-first intent
- Intel-Lint is designed for local analysis of sensitive CTI reports.
- Default API bind is loopback (`127.0.0.1`) via `python scripts/run.py api` or `python scripts/run.py app`.
- Recommended engine is local Ollama (`ENGINE=ollama`, `OLLAMA_URL=http://localhost:11434`).

## Data storage
- Runtime data is stored in the user data directory (not the repo by default):
  - Windows: `%LOCALAPPDATA%\\intel-lint`
  - macOS: `~/Library/Application Support/intel-lint`
  - Linux: `${XDG_DATA_HOME:-~/.local/share}/intel-lint`
- Subfolders:
  - `outputs/latest/` generated outputs (`claims.json`, `annotated.md`, `rewrite.md`)
  - `logs/app.log` metadata logs (timestamps, engine, output path)
  - `config/settings.json` local runtime settings
  - `outputs/cache/` deterministic cache artifacts
  - `debug/` verbose debug artifacts (only when `VERBOSE_LOGGING=1`)

## Wipe local data
- Delete the user data directory above to remove local outputs, logs, cache, and settings.
- Optional: remove repo-local `.env` if used for developer overrides.

## Verifying local-only behavior (Ollama mode)
- Set:
  - `ENGINE=ollama`
  - `OLLAMA_URL=http://localhost:11434`
- Run the app and confirm network activity is only to `localhost` (`127.0.0.1:11434` and local app ports).
- Keep `VERBOSE_LOGGING=0` (default) unless you explicitly need debug artifacts.
