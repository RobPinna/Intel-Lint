# Intel Lint Overview

## Purpose
Intel Lint evaluates CTI narrative text and outputs:
- `claims.json` with claim/evidence/bias structure
- `annotated.md` with span-linked annotations
- `rewrite.md` with neutralized wording

This project is educational/research software. It is not a security product.

## Architecture
```mermaid
flowchart LR
  CLI[intel-lint CLI] --> CORE[core.engine]
  API[FastAPI /analyze] --> CORE
  CORE --> PH[core.placeholder]
  CORE --> OL[core.ollama]
  PH --> OUT[io.outputs]
  OL --> OUT
  OUT --> FILES[claims.json / annotated.md / rewrite.md]
```

## Operational behavior
- Default mode is `ENGINE=ollama` for local LLM analysis with Ollama.
- `ENGINE=placeholder` is available for tests/smoke-only deterministic checks.
- Dev entrypoint is cross-platform via `python scripts/run.py app`.
- API endpoint `/download/latest` bundles latest output artifacts.

## Limits and assumptions
- Claim extraction is text-bound only; no external fact validation is performed.
- Bias flags are heuristic and should be reviewed by an analyst.
- Evidence spans may miss nuanced context in highly formatted reports.
