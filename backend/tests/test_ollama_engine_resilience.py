from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from app import engine_ollama
from app.models import AnalyzeRequest, AnalyzeResponse, BiasFlag, Claim, ClaimsDocument, EvidenceSpan, ScoreLabel


LOCAL_TMP_ROOT = Path(__file__).resolve().parent / ".tmp_local"


def _make_local_tmp_dir() -> Path:
    LOCAL_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOCAL_TMP_ROOT / f"ollama_cache_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_ollama_engine_coerces_noncompliant_json(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_DETERMINISTIC_CACHE", "0")
    bad_json = """{
      "title": "Hybrid Cyber Exposure",
      "date": "2026-02-07",
      "sections": [
        {"section_title": "Executive Summary", "content": "The gateway has multiple control gaps."}
      ]
    }"""

    monkeypatch.setattr(
        engine_ollama,
        "_call_ollama",
        lambda _prompt, _generate_rewrite, _runtime_profile=None: (
            bad_json,
            {
                "prompt_eval_duration_ns": 0,
                "eval_duration_ns": 0,
                "prompt_eval_count": 0,
                "eval_count": 0,
                "prompt_eval_duration_s": 0.0,
                "eval_duration_s": 0.0,
            },
        ),
    )
    response = engine_ollama.run_analysis(
        AnalyzeRequest(text="The gateway has multiple control gaps. The team must harden exposed systems.")
    )

    assert response.claims.engine == "ollama"
    assert len(response.claims.claims) >= 1
    assert response.annotated_md
    assert response.rewrite_md


def test_ollama_warmup_timeout_does_not_abort_analysis(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_DETERMINISTIC_CACHE", "0")
    text = "APT29 targeted Exchange servers."
    payload = {
        "claims": [
            {
                "id": "c1",
                "type": "threat",
                "text": text,
                "start": 0,
                "end": len(text),
                "assertiveness": "high",
                "score_label": "SUPPORTED",
                "score_reason": "Direct statement from source text.",
                "evidence": [{"start": 0, "end": len(text), "quote": text}],
            }
        ],
        "bias_flags": [
            {
                "bias_type": "formalism",
                "evidence": [{"start": 0, "end": 5, "quote": "APT29"}],
                "why": "No uncertainty marker is provided.",
                "suggested_fix": "Add confidence level and source quality.",
            }
        ],
    }

    monkeypatch.setenv("OLLAMA_REQUIRE_GPU", "1")
    monkeypatch.setattr(
        engine_ollama,
        "_ensure_gpu_ready_for_model",
        lambda: (_ for _ in ()).throw(
            ValueError(
                "Could not reach Ollama warm-up endpoint "
                "'http://localhost:11434/api/generate': timed out."
            )
        ),
    )
    monkeypatch.setattr(
        engine_ollama,
        "_call_ollama",
        lambda _prompt, _generate_rewrite, _runtime_profile=None: (
            json.dumps(payload),
            {
                "prompt_eval_duration_ns": 0,
                "eval_duration_ns": 0,
                "prompt_eval_count": 0,
                "eval_count": 0,
                "prompt_eval_duration_s": 0.0,
                "eval_duration_s": 0.0,
            },
        ),
    )

    response = engine_ollama.run_analysis(AnalyzeRequest(text=text, generate_rewrite=False))

    assert response.claims.engine == "ollama"
    assert response.claims.claims
    assert response.warning is not None
    assert "warm-up timed out" in response.warning.lower()


def test_ollama_deterministic_cache_returns_same_response(monkeypatch) -> None:
    tmp_path = _make_local_tmp_dir()
    monkeypatch.setenv("OLLAMA_DETERMINISTIC_CACHE", "1")
    monkeypatch.setenv("OLLAMA_STRICT_DETERMINISM", "0")
    monkeypatch.setenv("OLLAMA_DETERMINISTIC_CACHE_MAX_ENTRIES", "32")
    monkeypatch.setattr(engine_ollama, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(engine_ollama, "CACHE_FILE", tmp_path / "ollama_deterministic_cache.json")

    call_counter = {"count": 0}
    text = "The gateway has control gaps and weak hardening."

    def fake_call(_prompt, _generate_rewrite, _runtime_profile=None):
        call_counter["count"] += 1
        payload = {
            "claims": [
                {
                    "id": "c1",
                    "type": "statement",
                    "text": text,
                    "start": 0,
                    "end": len(text),
                    "assertiveness": "medium",
                    "score_label": "SUPPORTED",
                    "score_reason": f"call-{call_counter['count']}",
                    "evidence": [{"start": 0, "end": len(text), "quote": text}],
                }
            ],
            "bias_flags": [],
        }
        return (
            json.dumps(payload),
            {
                "prompt_eval_duration_ns": 0,
                "eval_duration_ns": 0,
                "prompt_eval_count": 0,
                "eval_count": 0,
                "prompt_eval_duration_s": 0.0,
                "eval_duration_s": 0.0,
            },
        )

    monkeypatch.setattr(engine_ollama, "_call_ollama", fake_call)
    request = AnalyzeRequest(text=text, generate_rewrite=False)

    first = engine_ollama.run_analysis(request)
    second = engine_ollama.run_analysis(request)

    assert call_counter["count"] == 1
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_ollama_deterministic_cache_normalizes_line_endings(monkeypatch) -> None:
    tmp_path = _make_local_tmp_dir()
    monkeypatch.setenv("OLLAMA_DETERMINISTIC_CACHE", "1")
    monkeypatch.setenv("OLLAMA_STRICT_DETERMINISM", "0")
    monkeypatch.setattr(engine_ollama, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(engine_ollama, "CACHE_FILE", tmp_path / "ollama_deterministic_cache.json")

    call_counter = {"count": 0}

    def fake_call(_prompt, _generate_rewrite, _runtime_profile=None):
        call_counter["count"] += 1
        text = "Line one"
        payload = {
            "claims": [
                {
                    "id": "c1",
                    "type": "statement",
                    "text": text,
                    "start": 0,
                    "end": 8,
                    "assertiveness": "medium",
                    "score_label": "SUPPORTED",
                    "score_reason": "deterministic",
                    "evidence": [{"start": 0, "end": 8, "quote": text}],
                }
            ],
            "bias_flags": [],
        }
        return (
            json.dumps(payload),
            {
                "prompt_eval_duration_ns": 0,
                "eval_duration_ns": 0,
                "prompt_eval_count": 0,
                "eval_count": 0,
                "prompt_eval_duration_s": 0.0,
                "eval_duration_s": 0.0,
            },
        )

    monkeypatch.setattr(engine_ollama, "_call_ollama", fake_call)

    req_unix = AnalyzeRequest(text="Line one.\nLine two.", generate_rewrite=False)
    req_win = AnalyzeRequest(text="Line one.\r\nLine two.\r\n", generate_rewrite=False)

    first = engine_ollama.run_analysis(req_unix)
    second = engine_ollama.run_analysis(req_win)

    assert call_counter["count"] == 1
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_ollama_claim_order_is_canonical(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_DETERMINISTIC_CACHE", "0")
    text = "A. B."
    payload = {
        "claims": [
            {
                "id": "c2",
                "type": "statement",
                "text": "B.",
                "start": 3,
                "end": 5,
                "assertiveness": "medium",
                "score_label": "SUPPORTED",
                "score_reason": "second",
                "evidence": [{"start": 3, "end": 5, "quote": "B."}],
            },
            {
                "id": "c1",
                "type": "statement",
                "text": "A.",
                "start": 0,
                "end": 2,
                "assertiveness": "medium",
                "score_label": "SUPPORTED",
                "score_reason": "first",
                "evidence": [{"start": 0, "end": 2, "quote": "A."}],
            },
        ],
        "bias_flags": [],
    }

    monkeypatch.setattr(
        engine_ollama,
        "_call_ollama",
        lambda _prompt, _generate_rewrite, _runtime_profile=None: (
            json.dumps(payload),
            {
                "prompt_eval_duration_ns": 0,
                "eval_duration_ns": 0,
                "prompt_eval_count": 0,
                "eval_count": 0,
                "prompt_eval_duration_s": 0.0,
                "eval_duration_s": 0.0,
            },
        ),
    )

    response = engine_ollama.run_analysis(AnalyzeRequest(text=text, generate_rewrite=False))
    assert [claim.text for claim in response.claims.claims] == ["A.", "B."]


def _make_response(text: str, reason: str) -> AnalyzeResponse:
    claim = Claim(
        claim_id="C001",
        text=text,
        category="statement",
        score_label=ScoreLabel.SUPPORTED,
        evidence=[EvidenceSpan(start=0, end=len(text), quote=text)],
        bias_flags=[],
    )
    return AnalyzeResponse(
        claims=ClaimsDocument(engine="ollama", claims=[claim]),
        annotated_md=f"# Annotated\n\n{text}\n",
        rewrite_md=f"# Rewrite\n\n{text}\n",
        warning=reason,
    )


def test_strict_determinism_accepts_matching_candidates(monkeypatch) -> None:
    tmp_path = _make_local_tmp_dir()
    monkeypatch.setenv("OLLAMA_DETERMINISTIC_CACHE", "1")
    monkeypatch.setenv("OLLAMA_STRICT_DETERMINISM", "1")
    monkeypatch.setenv("OLLAMA_STRICT_DETERMINISM_ATTEMPTS", "2")
    monkeypatch.setattr(engine_ollama, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(engine_ollama, "CACHE_FILE", tmp_path / "ollama_deterministic_cache.json")

    call_counter = {"count": 0}

    def fake_once(_request):
        call_counter["count"] += 1
        return _make_response("Stable output text.", f"run-{call_counter['count']}")

    monkeypatch.setattr(engine_ollama, "_run_analysis_once", fake_once)
    req = AnalyzeRequest(text="Same report", generate_rewrite=False)

    first = engine_ollama.run_analysis(req)
    second = engine_ollama.run_analysis(req)

    assert call_counter["count"] == 2
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_strict_determinism_falls_back_to_last_stable_cache(monkeypatch) -> None:
    tmp_path = _make_local_tmp_dir()
    monkeypatch.setenv("OLLAMA_DETERMINISTIC_CACHE", "1")
    monkeypatch.setenv("OLLAMA_STRICT_DETERMINISM", "1")
    monkeypatch.setenv("OLLAMA_STRICT_DETERMINISM_ATTEMPTS", "2")
    monkeypatch.setattr(engine_ollama, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(engine_ollama, "CACHE_FILE", tmp_path / "ollama_deterministic_cache.json")

    text = "Consistent report body."
    req = AnalyzeRequest(text=text, generate_rewrite=False)
    content_key = engine_ollama._deterministic_content_key(req)
    cache_key = engine_ollama._deterministic_cache_key(req)
    stable = _make_response("Stable cached response.", "stable")
    engine_ollama._write_cached_response(cache_key, stable, content_key)

    call_counter = {"count": 0}

    def fake_once(_request):
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return _make_response("Variant A", "a")
        return _make_response("Variant B", "b")

    monkeypatch.setattr(engine_ollama, "_run_analysis_once", fake_once)
    # force cache miss for strict path while preserving same content key for fallback
    monkeypatch.setattr(engine_ollama, "_deterministic_cache_key", lambda _req: "new-key-miss")

    response = engine_ollama.run_analysis(req)

    assert call_counter["count"] == 0
    assert response.rewrite_md == stable.rewrite_md
    assert response.warning == stable.warning


def test_strict_determinism_caches_canonical_candidate_on_mismatch(monkeypatch) -> None:
    tmp_path = _make_local_tmp_dir()
    monkeypatch.setenv("OLLAMA_DETERMINISTIC_CACHE", "1")
    monkeypatch.setenv("OLLAMA_STRICT_DETERMINISM", "1")
    monkeypatch.setenv("OLLAMA_STRICT_DETERMINISM_ATTEMPTS", "2")
    monkeypatch.setattr(engine_ollama, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(engine_ollama, "CACHE_FILE", tmp_path / "ollama_deterministic_cache.json")

    call_counter = {"count": 0}

    def fake_once(_request):
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return _make_response("Variant A", "a")
        return _make_response("Variant B", "b")

    monkeypatch.setattr(engine_ollama, "_run_analysis_once", fake_once)
    req = AnalyzeRequest(text="repeatable report", generate_rewrite=False)

    first = engine_ollama.run_analysis(req)
    second = engine_ollama.run_analysis(req)

    assert call_counter["count"] == 2
    assert first.claims.model_dump(mode="json") == second.claims.model_dump(mode="json")
    assert first.annotated_md == second.annotated_md
    assert first.rewrite_md == second.rewrite_md
    assert first.warning in {"a", "b"}
    assert second.warning == first.warning


def test_evidence_and_suggested_fix_are_human_readable(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_DETERMINISTIC_CACHE", "0")
    monkeypatch.setenv("OLLAMA_STRICT_DETERMINISM", "0")
    text = (
        "The actor enables opportunistic credential abuse seen in admin-panel activity. "
        "Investigators should avoid overconfident language."
    )
    start = text.index("enables") + 2
    end = text.index("credential") + 2
    payload = {
        "claims": [
            {
                "id": "c1",
                "type": "statement",
                "text": text,
                "start": 0,
                "end": len(text),
                "assertiveness": "medium",
                "score_label": "SUPPORTED",
                "score_reason": "Supported by quote.",
                "evidence": [{"start": start, "end": end, "quote": text[start:end]}],
            }
        ],
        "bias_flags": [
            {
                "bias_type": "certainty",
                "evidence": [{"start": start, "end": end, "quote": text[start:end]}],
                "why": "Uses absolute wording [4900:4921] in context.",
                "suggested_fix": "Replace [4944:4969] \" admin-panel abuse seen i\" with calibrated wording.",
            }
        ],
    }

    monkeypatch.setattr(
        engine_ollama,
        "_call_ollama",
        lambda _prompt, _generate_rewrite, _runtime_profile=None: (
            json.dumps(payload),
            {
                "prompt_eval_duration_ns": 0,
                "eval_duration_ns": 0,
                "prompt_eval_count": 0,
                "eval_count": 0,
                "prompt_eval_duration_s": 0.0,
                "eval_duration_s": 0.0,
            },
        ),
    )

    response = engine_ollama.run_analysis(AnalyzeRequest(text=text, generate_rewrite=False))
    claim = response.claims.claims[0]
    ev = claim.evidence[0]
    fix = claim.bias_flags[0].suggested_fix

    assert ev.quote.startswith("The actor enables opportunistic")
    assert "credential" in ev.quote
    assert claim.evidence_status == "anchored"
    assert "[4900:4921]" not in fix
    assert "[4944:4969]" not in fix
