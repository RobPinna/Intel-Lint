# Contributing

## Scope
- This repository is an educational/research CTI analysis project.
- It is not a production security product and comes with no guarantees.

## Local setup
1. `py -3 -m pip install -e ".[dev]"`
2. `pre-commit install`
3. `pytest`

## Quality gates
- `ruff check .`
- `ruff format --check .`
- `mypy`
- `pytest`

## Pull requests
- Keep changes focused and behavior-preserving unless a clear bug is fixed.
- Add or update tests for any functional change.
- Do not introduce secrets, telemetry, or external API keys in code or docs.
