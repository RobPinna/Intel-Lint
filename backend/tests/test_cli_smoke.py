from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from intel_lint.cli.main import main


LOCAL_TMP_ROOT = Path(__file__).resolve().parent / ".tmp_local"


def _make_local_tmp(prefix: str) -> Path:
    LOCAL_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOCAL_TMP_ROOT / f"{prefix}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_cli_generates_expected_files() -> None:
    tmp_path = _make_local_tmp("intel_lint_cli")
    input_file = tmp_path / "input.txt"
    out_dir = tmp_path / "out"
    input_file.write_text(
        "Threat report claims the actor always succeeds.\nObserved telemetry includes one malicious domain.\n",
        encoding="utf-8",
    )

    code = main([str(input_file), "--out", str(out_dir)])

    assert code == 0
    assert (out_dir / "claims.json").exists()
    assert (out_dir / "annotated.md").exists()
    assert (out_dir / "rewrite.md").exists()


def test_cli_returns_error_for_missing_input() -> None:
    tmp_path = _make_local_tmp("intel_lint_cli_missing")
    missing = tmp_path / "missing.txt"

    code = main([str(missing)])

    assert code == 2
