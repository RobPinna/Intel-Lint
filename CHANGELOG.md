# Changelog

## v0.1.0 - 2026-02-22
- Introduced modern Python packaging with `pyproject.toml`.
- Added `src/intel_lint` package layout with clear `core`, `io`, `models`, and `cli` boundaries.
- Kept backward-compatible `backend/app` wrappers to preserve existing behavior.
- Added quality tooling: Ruff, mypy, pytest, pre-commit, and GitHub Actions CI.
- Added deterministic offline samples, golden fixtures, and example scenarios.
- Reworked documentation for portfolio publication, assumptions, and limitations.
- Added repository governance and safety files (`LICENSE`, `CONTRIBUTING`, `SECURITY`, `CODE_OF_CONDUCT`).
