from __future__ import annotations

import json
from pathlib import Path

from ..models.schemas import AnalyzeResponse


def write_latest_outputs(response: AnalyzeResponse, outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "claims.json").write_text(
        json.dumps(response.claims.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    (outputs_dir / "annotated.md").write_text(response.annotated_md, encoding="utf-8")
    (outputs_dir / "rewrite.md").write_text(response.rewrite_md, encoding="utf-8")
