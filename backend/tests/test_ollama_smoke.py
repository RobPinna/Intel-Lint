from __future__ import annotations

import os

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app


DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def _ollama_reachable() -> bool:
    host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")
    health_url = f"{host}/api/tags"
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(health_url)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(os.getenv("ENGINE", "placeholder").lower() != "ollama", reason="ENGINE is not ollama")
@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama is unreachable")
def test_ollama_smoke_analysis() -> None:
    client = TestClient(app)
    payload = {
        "text": "The vendor claims this breakthrough tool always detects fraud and never misses incidents.",
        "sample_name": "ollama_smoke.txt",
    }

    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["claims"]["engine"] == "ollama"
    assert isinstance(data["claims"]["claims"], list)
    assert len(data["claims"]["claims"]) > 0
    for claim in data["claims"]["claims"]:
        assert "score_label" in claim
        assert claim["score_label"] in {"SUPPORTED", "PLAUSIBLE", "SPECULATIVE"}
        assert "evidence" in claim
        assert isinstance(claim["evidence"], list)
    assert "annotated_md" in data
    assert "rewrite_md" in data
