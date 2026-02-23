# Intel Lint

Intel Lint is a local CTI text quality analyzer that extracts claims, evidence spans, and bias flags.

## Disclaimer
- Educational/research project for analyst workflow experiments.
- Not a security product.
- No guarantee of correctness, completeness, or fitness for operational security decisions.

## What / Why / Output
- What: analyze CTI narrative text and structure it into machine-readable claims.
- Why: support claim quality review, bias detection, and neutral rewrite generation.
- Output:
  - `claims.json`
  - `annotated.md`
  - `rewrite.md`

## Quickstart (3 commands)
```bash
py -3 -m pip install -e ".[dev]"
pytest
py -3 -m intel_lint.cli.main samples/sample1.txt --out outputs/latest
```

## Come usare il progetto (release GitHub)
1. Clona il repository.
2. Configura i placeholder:
```bat
copy .env.example .env
```
3. Sostituisci i valori dove serve:
- File: `.env`
  Placeholder: `ENGINE=placeholder`
  Cosa mettere: `placeholder` (offline) oppure `ollama` se usi LLM locale.
- File: `.env`
  Placeholder: `OLLAMA_MODEL=foundation-sec:latest`
  Cosa mettere: il modello Ollama che hai effettivamente installato.
- File: `.env`
  Placeholder: `OLLAMA_HOST=http://127.0.0.1:11434`
  Cosa mettere: host locale Ollama (in genere lasciare invariato).
4. Avvio:
```bat
start-dev.cmd
```
5. Verifica release-safety prima di pubblicare:
```bat
release-safety-check.cmd
```

## Full stack shortcut (backend + frontend)
From the repository root on Windows:
```bat
start-dev.cmd
```

Ollama GPU options:
```bat
start-dev.cmd ollama ipex
start-dev.cmd ollama nvidia
start-dev.cmd ollama vulkan
```

Notes:
- `start-dev.cmd` starts backend on `http://127.0.0.1:8000` and frontend on `http://127.0.0.1:5173`.
- Default mode is `placeholder` (offline by default).
- `nvidia` uses the standard Ollama runtime with CUDA autodetection.
- `llm_models/` in this repo is placeholder-only; local model blobs must not be committed.

## Repository tree (excerpt)
```text
CTIclaimsguard/
  start-dev.cmd
  start-ollama-gpu.cmd
  backend/
  frontend/
  src/intel_lint/
  examples/
  docs/
```

## Real example (local run)
Command:
```bash
py -3 -m intel_lint.cli.main examples/scenario_minimal/input.txt --out examples/scenario_minimal/out
```

Observed terminal output:
```text
wrote outputs to D:\Rob Pinna\Tech\Cybersec\CTIclaimsguard\examples\scenario_minimal\out
claims=2 engine=placeholder
```

Observed `claims.json` snippet:
```json
{
  "claim_id": "C002",
  "text": "The analyst report says this proves the platform will always fail under stress.",
  "score_label": "SUPPORTED",
  "bias_flags": [
    {
      "tag": "certainty"
    }
  ]
}
```

## Architecture
- Package: `src/intel_lint/`
- Core analysis: `src/intel_lint/core/`
- IO writing: `src/intel_lint/io/`
- Models/schema: `src/intel_lint/models/`
- CLI: `src/intel_lint/cli/`
- Backward-compatible wrappers: `backend/app/`

## Tests
- Unit and deterministic regression tests in `backend/tests/`.
- Golden fixtures in `golden/` generated from `samples/`.

## Limits and assumptions
- Text-only analysis; no external source verification.
- Bias detection is heuristic and should be analyst-reviewed.
- Placeholder engine is deterministic but intentionally simple.

## Security notes
- No telemetry.
- No embedded secrets or API keys.
- Offline operation by default (`ENGINE=placeholder`).

## Portfolio note
This repository demonstrates:
- CTI claim rigor through evidence-span grounding.
- Deterministic testing and reproducible outputs.
- Practical engineering controls (linting, typing boundaries, CI, and release hygiene).

## Commands
- Lint: `ruff check .`
- Format: `ruff format .`
- Typecheck: `mypy`
- Test: `pytest`
- API (dev): `uvicorn intel_lint.api:app --reload --app-dir src`
