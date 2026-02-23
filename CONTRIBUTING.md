# Contributing

## Scope
- This repository is an educational/research CTI analysis project.
- It is not a production security product and comes with no guarantees.

## Before Push Checklist
- [ ] Run `python -m pip install -e ".[dev]"`.
- [ ] Run `python scripts/run.py doctor`.
- [ ] Run `python -m pytest`.
- [ ] Run `python scripts/release_check.py`.
- [ ] Confirm `.env` is not tracked.
- [ ] Confirm no `settings.json` file is tracked.
- [ ] Confirm no runtime artifacts are staged (`outputs/`, logs, caches, local data).
- [ ] Confirm no model blobs/vectorstores are staged.
- [ ] Keep changes focused and avoid core-logic redesign unless requested.
- [ ] Update README/docs only when behavior or commands changed.
