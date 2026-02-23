Project: Intel Lint (LLM-first with guardrails)

Goal:
- Provide a simple modern SaaS-like UI + FastAPI backend that analyzes a text and returns:
  - claims.json
  - annotated.md
  - rewrite.md
- Also write the same files to outputs/latest/ on each run.
- Maintain samples/ and golden/ fixtures. Golden tests must run in placeholder mode.
- Support ENGINE=placeholder (default) and ENGINE=ollama (later).

Guardrails (for ollama mode):
- Output must be strict JSON conforming to Pydantic schema.
- Every claim and every bias flag must include evidence spans (start/end/quote) from input text.
- If evidence missing => score_label MUST be SPECULATIVE. 
- Deterministic settings (temperature 0).

LLM behavior:
- Act as a neutral auditor of Cyber Threat Intelligence reports: judge and extract claims/bias, never defend or "fix" the report.
- Base all claims and bias flags only on the provided text; do not add external facts or corrections.
- `rewrite.md` must restate the report in a neutral, evidence-grounded way without patching or justifying the source content.

UI:
- Left: textarea + sample dropdown + Analyze button
- Right: tabs: Annotated (markdown), Claims (table), Bias (chips), Rewrite (markdown)
- Download ZIP of the three output files
