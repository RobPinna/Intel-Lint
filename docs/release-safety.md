# Release Safety Notes

This public portfolio repository is code-first.

Do not redistribute:
- local Ollama model blobs (`llm_models/blobs`)
- local virtual environments (`.venv`, `backend/.venv`)
- build artifacts (`frontend/node_modules`, `frontend/dist`)
- runtime artifacts (`outputs/*`, local DB/cache files)

Before publishing:
1. Run `release-safety-check.cmd`.
2. Confirm `.env` is not tracked.
3. Keep `llm_models/` as placeholder-only.
