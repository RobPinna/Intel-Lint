# Design Decisions

## Decision 1: `src/` package with compatibility wrappers
- Choice: new package code in `src/intel_lint`, while `backend/app` remains import-compatible wrappers.
- Why: preserve existing startup/testing behavior without forcing immediate caller migration.
- Tradeoff: temporary duplication in import paths until wrappers are retired.

## Decision 2: Deterministic default engine
- Choice: keep `placeholder` as default engine.
- Why: offline, reproducible outputs for CI and portfolio demos.
- Tradeoff: lower analytical depth than model-backed mode.

## Decision 3: Strong quality tooling, scoped mypy
- Choice: enforce Ruff and pytest broadly, run mypy on typed boundary modules.
- Why: maximize maintainability without blocking on full typing of large legacy internals.
- Tradeoff: some deep internals remain outside strict type guarantees.

## Decision 4: Golden fixtures for regression confidence
- Choice: add sample-driven golden artifacts under `golden/`.
- Why: stable output verification is critical for deterministic pipelines.
- Tradeoff: fixture updates are required when intentional output changes occur.
