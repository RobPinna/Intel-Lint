from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from app.engine_placeholder import run_analysis
from app.main import app
from app.models import AnalyzeRequest
from intel_lint.runtime import load_settings


ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT / "samples"
GOLDEN_DIR = ROOT / "golden"
LATEST_DIR = Path(load_settings()["output_dir"])


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_placeholder_matches_golden_files() -> None:
    for sample in sorted(SAMPLES_DIR.glob("*.txt")):
        response = run_analysis(AnalyzeRequest(text=_read(sample), sample_name=sample.name))
        sample_golden = GOLDEN_DIR / sample.stem

        expected_claims = json.loads(_read(sample_golden / "claims.json"))
        assert response.claims.model_dump(mode="json") == expected_claims
        assert response.annotated_md == _read(sample_golden / "annotated.md")
        assert response.rewrite_md == _read(sample_golden / "rewrite.md")


def test_analyze_writes_latest_outputs_and_download_zip() -> None:
    os.environ["ENGINE"] = "placeholder"
    client = TestClient(app)
    sample = SAMPLES_DIR / "sample1.txt"
    payload = {"text": _read(sample), "sample_name": sample.name}

    analyze_resp = client.post("/analyze", json=payload)
    assert analyze_resp.status_code == 200

    for filename in ["claims.json", "annotated.md", "rewrite.md"]:
        assert (LATEST_DIR / filename).exists(), f"missing latest output {filename}"

    zip_resp = client.get("/download/latest")
    assert zip_resp.status_code == 200
    assert zip_resp.headers["content-type"].startswith("application/zip")
    assert len(zip_resp.content) > 100


def test_analyze_allows_vite_dev_origin_cors() -> None:
    os.environ["ENGINE"] = "placeholder"
    client = TestClient(app)
    sample = SAMPLES_DIR / "sample1.txt"
    payload = {"text": _read(sample), "sample_name": sample.name}
    origin = "http://127.0.0.1:5173"

    preflight_resp = client.options(
        "/analyze",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight_resp.status_code == 200
    assert preflight_resp.headers.get("access-control-allow-origin") == origin

    analyze_resp = client.post("/analyze", json=payload, headers={"Origin": origin})
    assert analyze_resp.status_code == 200
    assert analyze_resp.headers.get("access-control-allow-origin") == origin
