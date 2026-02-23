from __future__ import annotations

import json

import pytest

from app import engine_ollama
from app.models import AnalyzeRequest
from app.models import (
    BiasFlag,
    Claim,
    EvidenceSpan,
    OllamaAnalysisResult,
    OllamaBiasMaskItem,
    OllamaBiasMaskResult,
    OllamaClaim,
    ScoreLabel,
)


def test_sentence_splitter_is_stable_with_offsets() -> None:
    text = "First sentence. Second one!\nThird line? Fourth stays."
    blocks = engine_ollama._parse_text_blocks(text)
    first = engine_ollama._split_sentences_with_offsets(text, blocks)
    second = engine_ollama._split_sentences_with_offsets(text, blocks)

    assert [(s.sentence_id, s.start, s.end, s.text) for s in first] == [
        (0, 0, 15, "First sentence."),
        (1, 16, 27, "Second one!"),
        (2, 28, 39, "Third line?"),
        (3, 40, 53, "Fourth stays."),
    ]
    assert [(s.sentence_id, s.start, s.end, s.text) for s in second] == [
        (s.sentence_id, s.start, s.end, s.text) for s in first
    ]


def test_parse_blocks_marks_recommendations_section() -> None:
    text = "# Intro\nA.\n## Recommendations\n- Do this.\n- Do that."
    blocks = engine_ollama._parse_text_blocks(text)
    classes = [b.section_class for b in blocks]
    assert "RECOMMENDATIONS" in classes


def test_parse_blocks_marks_analyst_notes_section() -> None:
    text = "# Intro\nA.\n## Analyst Notes\nThis is analyst commentary."
    blocks = engine_ollama._parse_text_blocks(text)
    classes = [b.section_class for b in blocks]
    assert "ANALYST_NOTES" in classes


def test_mask_recommendations_preserves_length() -> None:
    text = "## Context\nA.\n## Recommendations\nReplace all systems."
    blocks = engine_ollama._parse_text_blocks(text)
    masked = engine_ollama._mask_recommendation_text(text, blocks)
    assert len(masked) == len(text)
    assert "Replace all systems." not in masked


def test_mask_analyst_notes_preserves_length() -> None:
    text = "## Context\nA.\n## Analyst Notes\nAnalyst-only notes here."
    blocks = engine_ollama._parse_text_blocks(text)
    masked = engine_ollama._mask_recommendation_text(text, blocks)
    assert len(masked) == len(text)
    assert "Analyst-only notes here." not in masked


def test_bias_batch_validator_requires_exact_sentence_ids() -> None:
    payload = OllamaBiasMaskResult(results=[OllamaBiasMaskItem(id=0, flags=0)])
    with pytest.raises(ValueError, match="Missing sentence_id"):
        engine_ollama._validate_bias_batch_items(payload, [0, 1])


def test_build_bias_flags_uses_sentence_spans() -> None:
    text = "The region is always hostile."
    sentences = engine_ollama._split_sentences_with_offsets(text)
    sentence_flags = {
        0: engine_ollama.BIAS_BIT["certainty_overclaim"] | engine_ollama.BIAS_BIT["regional_generalization"]
    }
    flags = engine_ollama._build_bias_flags_from_sentence_flags(text, sentences, sentence_flags)

    assert len(flags) == 2
    assert {flag.bias_type for flag in flags} == {"certainty_overclaim", "regional_generalization"}
    for flag in flags:
        ev = flag.evidence[0]
        assert ev.start == 0
        assert ev.end == len(text)
        assert ev.quote == text
        assert ev.sentence_id == 0


def test_hard_trigger_cultural_regional_sets_bits() -> None:
    sentence = "Security posture in this region reflects low maturity and accountability versus Western peers."
    flags = engine_ollama._hard_trigger_flags(sentence, "ASSESSMENT")
    assert flags & engine_ollama.BIAS_BIT["cultural_essentialism"]
    assert flags & engine_ollama.BIAS_BIT["regional_generalization"]


def test_hard_trigger_prescriptive_overreach_ignored_in_recommendations() -> None:
    sentence = "Within 24 hours, mandate replace all endpoints."
    flags = engine_ollama._hard_trigger_flags(sentence, "RECOMMENDATIONS")
    assert flags == 0


def test_hard_trigger_source_imbalance_is_deterministic() -> None:
    sentence = "This conclusion relies on a single source without independent corroboration."
    flags = engine_ollama._hard_trigger_flags(sentence, "ASSESSMENT")
    assert flags & engine_ollama.BIAS_BIT["source_imbalance"]


def test_detect_bias_skips_recommendation_sentences_in_llm_batches(monkeypatch) -> None:
    text = (
        "Security posture in this region reflects low maturity and accountability versus Western peers.\n"
        "## Recommendations\n"
        "Within 24 hours, mandate replace all systems.\n"
    )
    blocks = engine_ollama._parse_text_blocks(text)
    sent = engine_ollama._split_sentences_with_offsets(text, blocks)
    rec_ids = [s.sentence_id for s in sent if s.section_class == "RECOMMENDATIONS"]
    non_rec_ids = [s.sentence_id for s in sent if s.section_class != "RECOMMENDATIONS"]
    seen_ids: list[int] = []

    def fake_bias_call(_prompt, schema, _runtime_profile=None, _num_predict=None):
        ids = schema["properties"]["results"]["items"]["properties"]["id"]["enum"]
        seen_ids.extend(ids)
        body = {"results": [{"id": sid, "flags": 0} for sid in ids]}
        return (
            json.dumps(body),
            {"prompt_eval_count": 0, "eval_count": 0, "prompt_eval_duration_ns": 0, "eval_duration_ns": 0},
            {"response": json.dumps(body)},
        )

    monkeypatch.setattr(engine_ollama, "_call_ollama_structured", fake_bias_call)

    flags = engine_ollama._detect_bias_flags_for_text(
        source_text=text,
        blocks=blocks,
        perf_samples=[],
        call_budget={"used": 0, "max": 50},
        runtime_profile={"gpu_safe_mode": False},
        debug_trace={"run_id": "t", "batches": [], "summary": {}},
    )

    assert all(item in non_rec_ids for item in seen_ids)
    assert all(item not in seen_ids for item in rec_ids)
    assert flags
    assert {flag.bias_type for flag in flags} >= {"cultural_essentialism", "regional_generalization"}


def test_llm_source_imbalance_bit_is_ignored_without_deterministic_trigger(monkeypatch) -> None:
    text = "Unrelated sentence without source language."
    blocks = engine_ollama._parse_text_blocks(text)
    sentences = engine_ollama._split_sentences_with_offsets(text, blocks)
    sid = sentences[0].sentence_id

    def fake_bias_call(_prompt, _schema, _runtime_profile=None, _num_predict=None):
        body = {"results": [{"id": sid, "flags": engine_ollama.BIAS_BIT["source_imbalance"]}]}
        return (
            json.dumps(body),
            {"prompt_eval_count": 0, "eval_count": 0, "prompt_eval_duration_ns": 0, "eval_duration_ns": 0},
            {"response": json.dumps(body)},
        )

    monkeypatch.setattr(engine_ollama, "_call_ollama_structured", fake_bias_call)
    flags = engine_ollama._detect_bias_flags_for_text(
        source_text=text,
        blocks=blocks,
        perf_samples=[],
        call_budget={"used": 0, "max": 10},
        runtime_profile={"gpu_safe_mode": False},
        debug_trace={"run_id": "x", "batches": [], "summary": {}},
    )
    assert all(flag.bias_type != "source_imbalance" for flag in flags)


def test_attribution_caveat_guard_clears_bias() -> None:
    sentences = [
        engine_ollama.SentenceSpan(
            sentence_id=0,
            start=0,
            end=80,
            text="Current evidence supports activity; attribution is not confirmed.",
            section_class="ASSESSMENT",
        )
    ]
    sentence_flags = {0: engine_ollama.BIAS_BIT["attribution_without_evidence"]}
    flags = engine_ollama._build_bias_flags_from_sentence_flags("dummy", sentences, sentence_flags)
    assert not any(flag.bias_type == "attribution_without_evidence" for flag in flags)

    sentences = [
        engine_ollama.SentenceSpan(
            sentence_id=1,
            start=0,
            end=120,
            text="Attribution: No reliable evidence supports assigning the activity to a specific APT at this time.",
            section_class="ATTRIBUTION",
        )
    ]
    sentence_flags = {1: engine_ollama.BIAS_BIT["attribution_without_evidence"]}
    flags = engine_ollama._build_bias_flags_from_sentence_flags("dummy", sentences, sentence_flags)
    assert not any(flag.bias_type == "attribution_without_evidence" for flag in flags)


def test_claim_evidence_links_to_evidence_section() -> None:
    evidence_sentences = [
        engine_ollama.SentenceSpan(
            sentence_id=0,
            start=0,
            end=40,
            text="Summary of incident with high confidence.",
            section_class="ASSESSMENT",
        ),
        engine_ollama.SentenceSpan(
            sentence_id=1,
            start=41,
            end=110,
            text="Observations: DNS query to malicious.example.com observed in logs.",
            section_class="EVIDENCE",
        ),
    ]
    pool = engine_ollama._build_evidence_pool(evidence_sentences)
    claim = engine_ollama.OllamaClaim(
        id="C1",
        type="statement",
        text="DNS query to malicious.example.com was detected.",
        start=0,
        end=10,
        assertiveness="medium",
        score_label=engine_ollama.ScoreLabel.PLAUSIBLE,
        score_reason="test",
        evidence=[],
    )
    linked = engine_ollama._link_claim_to_evidence(claim, pool, max_items=2)
    assert linked
    assert linked[0].sentence_id == 1


def test_claim_evidence_link_respects_context_section() -> None:
    evidence_sentences = [
        engine_ollama.SentenceSpan(
            sentence_id=0,
            start=0,
            end=70,
            text="Executive Summary: Organizational exposure depends on procurement maturity.",
            section_name="Executive Summary",
            section_class="EXEC_SUMMARY",
        ),
        engine_ollama.SentenceSpan(
            sentence_id=1,
            start=71,
            end=130,
            text="Observations: RTSP services were exposed on internet-facing hosts.",
            section_name="Observations",
            section_class="EVIDENCE",
        ),
    ]
    pool = engine_ollama._build_evidence_pool(evidence_sentences)
    claim = engine_ollama.OllamaClaim(
        id="C2",
        type="statement",
        text="Socio-technical context indicates procurement constraints.",
        start=0,
        end=40,
        assertiveness="medium",
        score_label=engine_ollama.ScoreLabel.PLAUSIBLE,
        score_reason="test",
        evidence=[],
    )
    linked = engine_ollama._link_claim_to_evidence(
        claim,
        pool,
        max_items=2,
        allowed_classes={"CONTEXT", "EXEC_SUMMARY"},
    )
    assert linked
    assert linked[0].sentence_id == 0


def test_run_bias_batch_check_retries_once_on_invalid_ids(monkeypatch) -> None:
    sentences = [
        engine_ollama.SentenceSpan(sentence_id=0, start=0, end=4, text="One."),
        engine_ollama.SentenceSpan(sentence_id=1, start=5, end=9, text="Two."),
    ]
    bad_payload = {"response": '{"results":[{"id":0,"flags":0}]}'}
    call_count = {"n": 0}

    def fake_call(*_args, **_kwargs):
        call_count["n"] += 1
        return (
            bad_payload["response"],
            {"prompt_eval_count": 0, "eval_count": 0, "prompt_eval_duration_ns": 0, "eval_duration_ns": 0},
            bad_payload,
        )

    monkeypatch.setattr(engine_ollama, "_call_ollama_structured", fake_call)
    with pytest.raises(ValueError, match="Bias checklist validation failed"):
        engine_ollama._run_bias_batch_check(
            batch_sentences=sentences,
            batch_index=1,
            batch_count=1,
            perf_samples=[],
            call_budget={"used": 0, "max": 10},
            runtime_profile={"gpu_safe_mode": False},
            debug_trace={"run_id": "test", "batches": [], "summary": {}},
        )
    assert call_count["n"] == 2


def test_bias_detection_is_repeatable_for_ten_runs(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_DETERMINISTIC_CACHE", "0")
    monkeypatch.setenv("OLLAMA_STRICT_DETERMINISM", "0")
    monkeypatch.setattr(engine_ollama, "_ollama_reachable", lambda: True)

    text = (
        "Security posture in this region reflects low maturity and accountability versus Western peers. "
        "## Recommendations\nWithin 24 hours, mandate replace all systems."
    )
    claim_payload = {
        "claims": [
            {
                "id": "c1",
                "type": "statement",
                "text": "Security posture in this region reflects low maturity and accountability versus Western peers.",
                "start": 0,
                "end": 92,
                "assertiveness": "high",
                "score_label": "SUPPORTED",
                "score_reason": "Directly stated.",
                "evidence": [{"start": 0, "end": 92, "quote": text[:92]}],
            }
        ],
        "bias_flags": [],
    }

    monkeypatch.setattr(
        engine_ollama,
        "_call_ollama",
        lambda _prompt, _generate_rewrite, _runtime_profile=None: (
            json.dumps(claim_payload),
            {"prompt_eval_count": 0, "eval_count": 0, "prompt_eval_duration_ns": 0, "eval_duration_ns": 0},
        ),
    )

    def fake_bias_call(_prompt, schema, _runtime_profile=None, _num_predict=None):
        ids = schema["properties"]["results"]["items"]["properties"]["id"]["enum"]
        results = [{"id": sid, "flags": 0} for sid in ids]
        body = {"results": results}
        return (
            json.dumps(body),
            {"prompt_eval_count": 0, "eval_count": 0, "prompt_eval_duration_ns": 0, "eval_duration_ns": 0},
            {"response": json.dumps(body)},
        )

    monkeypatch.setattr(engine_ollama, "_call_ollama_structured", fake_bias_call)

    baseline = None
    for _ in range(10):
        response = engine_ollama.run_analysis(AnalyzeRequest(text=text, generate_rewrite=False))
        payload = response.model_dump(mode="json")
        if baseline is None:
            baseline = payload
        else:
            assert payload == baseline


def test_max_ollama_calls_clamps_low_env_to_required_minimum(monkeypatch) -> None:
    text = "A. B. C. D. E. F. G. H. I. J."
    monkeypatch.setenv("OLLAMA_MAX_CALLS", "6")
    monkeypatch.setenv("OLLAMA_BIAS_SENTENCES_PER_BATCH", "4")
    value = engine_ollama._max_ollama_calls(chunk_count=2, text=text)
    assert value > 6


def test_bias_budget_error_falls_back_without_raising(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_DETERMINISTIC_CACHE", "0")
    monkeypatch.setenv("OLLAMA_STRICT_DETERMINISM", "0")
    monkeypatch.setattr(engine_ollama, "_ollama_reachable", lambda: True)

    def fake_chunk(*_args, **_kwargs):
        text = "Simple statement."
        return [
            OllamaAnalysisResult(
                claims=[
                    OllamaClaim(
                        id="1",
                        type="statement",
                        text=text,
                        start=0,
                        end=len(text),
                        assertiveness="medium",
                        score_label=ScoreLabel.SUPPORTED,
                        score_reason="ok",
                        evidence=[EvidenceSpan(start=0, end=len(text), quote=text)],
                    )
                ],
                bias_flags=[],
                annotated_md="",
                rewrite_md="",
            )
        ]

    monkeypatch.setattr(engine_ollama, "_run_chunk_with_timeout_fallback", fake_chunk)
    monkeypatch.setattr(
        engine_ollama,
        "_detect_bias_flags_for_text",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("Ollama call budget exceeded (6)")),
    )

    response = engine_ollama.run_analysis(AnalyzeRequest(text="Simple statement.", generate_rewrite=False))
    assert response.claims.claims
    assert response.warning is not None
    assert "Bias detection fallback" in response.warning


def test_canonicalize_caps_bias_flags_without_validation_error(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MAX_BIAS_FLAGS", "20")
    many_flags = []
    for idx in range(25):
        many_flags.append(
            engine_ollama.OllamaBiasFlag(
                bias_type=f"type_{idx}",
                evidence=[EvidenceSpan(start=0, end=5, quote="abcde", sentence_id=0)],
                why="w",
                suggested_fix="f",
            )
        )
    result = OllamaAnalysisResult(
        claims=[],
        bias_flags=many_flags,
        annotated_md="",
        rewrite_md="",
    )
    normalized = engine_ollama._canonicalize_analysis_result(result)
    assert len(normalized.bias_flags) == 20


def test_apply_guardrails_claim_without_evidence_gets_fallback_span() -> None:
    source = "Observed lateral movement through a compromised admin account."
    candidate = OllamaAnalysisResult(
        claims=[
            OllamaClaim(
                id="c1",
                type="statement",
                text="Observed lateral movement through a compromised admin account.",
                start=0,
                end=len(source),
                assertiveness="high",
                score_label=ScoreLabel.SUPPORTED,
                score_reason="model",
                evidence=[],
            )
        ],
        bias_flags=[],
        annotated_md="",
        rewrite_md="",
    )

    guarded = engine_ollama._apply_guardrails(candidate, source)
    assert len(guarded.claims[0].evidence) == 1
    assert guarded.claims[0].evidence[0].quote == source
    assert guarded.claims[0].score_label == ScoreLabel.SPECULATIVE


def test_apply_guardrails_supported_requires_hard_markers() -> None:
    source = "The actor activity was observed and assessed by analysts."
    candidate = OllamaAnalysisResult(
        claims=[
            OllamaClaim(
                id="c1",
                type="statement",
                text="The actor activity was observed and assessed by analysts.",
                start=0,
                end=len(source),
                assertiveness="high",
                score_label=ScoreLabel.SUPPORTED,
                score_reason="model",
                evidence=[EvidenceSpan(start=0, end=len(source), quote=source)],
            )
        ],
        bias_flags=[],
        annotated_md="",
        rewrite_md="",
    )
    guarded = engine_ollama._apply_guardrails(candidate, source)
    assert guarded.claims[0].score_label == ScoreLabel.SPECULATIVE


def test_claim_output_self_citation_sets_missing(monkeypatch) -> None:
    monkeypatch.setattr(engine_ollama, "_write_claim_missing_debug", lambda **_kwargs: None)
    source = "Within a hybrid exposure environment, such devices may plausibly serve as initial access vectors."
    claim = OllamaClaim(
        id="c1",
        type="statement",
        text=source,
        start=0,
        end=len(source),
        assertiveness="medium",
        score_label=ScoreLabel.SUPPORTED,
        score_reason="r",
        evidence=[EvidenceSpan(start=0, end=len(source), quote=source, sentence_id=0)],
    )
    result = OllamaAnalysisResult(claims=[claim], bias_flags=[], annotated_md="", rewrite_md="")
    response = engine_ollama._to_api_response(result, source_text=source, blocks=engine_ollama._parse_text_blocks(source))
    item = response.claims.claims[0]
    assert item.evidence_status == "missing"
    assert item.evidence == []
    assert item.evidence_texts == []
    assert item.evidence_sentence_ids == []
    assert item.evidence_note is not None


def test_claim_output_anchored_has_sentence_ids_and_full_text(monkeypatch) -> None:
    monkeypatch.setattr(engine_ollama, "_write_claim_missing_debug", lambda **_kwargs: None)
    source = (
        "Key judgement: Exposure exists across distributed nodes. "
        "Passive OSINT enumeration identified 42 exposed RTSP endpoints on port 554."
    )
    evidence_text = "Passive OSINT enumeration identified 42 exposed RTSP endpoints on port 554."
    evidence_start = source.index(evidence_text)
    evidence_end = evidence_start + len(evidence_text)
    claim = OllamaClaim(
        id="c1",
        type="statement",
        text="Exposure exists across distributed nodes.",
        start=0,
        end=55,
        assertiveness="medium",
        score_label=ScoreLabel.SUPPORTED,
        score_reason="r",
        evidence=[EvidenceSpan(start=evidence_start, end=evidence_end, quote=evidence_text)],
    )
    result = OllamaAnalysisResult(claims=[claim], bias_flags=[], annotated_md="", rewrite_md="")
    response = engine_ollama._to_api_response(result, source_text=source, blocks=engine_ollama._parse_text_blocks(source))
    item = response.claims.claims[0]
    assert item.evidence_status == "anchored"
    assert item.evidence_sentence_ids
    assert item.evidence_texts == [evidence_text]
    assert item.evidence[0].quote == evidence_text
    assert item.evidence[0].sentence_id == item.evidence_sentence_ids[0]


def test_claim_output_truncated_evidence_is_dropped(monkeypatch) -> None:
    monkeypatch.setattr(engine_ollama, "_write_claim_missing_debug", lambda **_kwargs: None)
    source = "Within a hybrid exposure environment, such devices may plausibly serve as initial access vectors."
    truncated_quote = "ithin a hy..."
    start = 1
    end = start + len(truncated_quote)
    claim = OllamaClaim(
        id="c1",
        type="statement",
        text="Such devices may plausibly serve as initial access vectors.",
        start=0,
        end=len(source),
        assertiveness="medium",
        score_label=ScoreLabel.PLAUSIBLE,
        score_reason="r",
        evidence=[EvidenceSpan(start=start, end=end, quote=truncated_quote)],
    )
    result = OllamaAnalysisResult(claims=[claim], bias_flags=[], annotated_md="", rewrite_md="")
    response = engine_ollama._to_api_response(result, source_text=source, blocks=engine_ollama._parse_text_blocks(source))
    item = response.claims.claims[0]
    assert item.evidence_status == "missing"
    assert item.evidence == []
    assert item.evidence_texts == []
    assert item.evidence_sentence_ids == []


def test_claim_output_uses_full_sentence_not_snippet(monkeypatch) -> None:
    monkeypatch.setattr(engine_ollama, "_write_claim_missing_debug", lambda **_kwargs: None)
    source = (
        "Context indicates procurement constraints in the sector. "
        "Observations showed DNS telemetry anomalies across three sensors."
    )
    partial_quote = "Observations showed DNS telemetry anomalies"
    start = source.index(partial_quote)
    end = start + len(partial_quote)
    full_sentence = "Observations showed DNS telemetry anomalies across three sensors."
    claim = OllamaClaim(
        id="c1",
        type="statement",
        text="Procurement constraints may influence exposure.",
        start=0,
        end=58,
        assertiveness="low",
        score_label=ScoreLabel.PLAUSIBLE,
        score_reason="r",
        evidence=[EvidenceSpan(start=start, end=end, quote=partial_quote)],
    )
    result = OllamaAnalysisResult(claims=[claim], bias_flags=[], annotated_md="", rewrite_md="")
    response = engine_ollama._to_api_response(result, source_text=source, blocks=engine_ollama._parse_text_blocks(source))
    item = response.claims.claims[0]
    assert item.evidence_status == "anchored"
    assert item.evidence_texts == [full_sentence]
    assert item.evidence[0].quote == full_sentence


def test_map_bias_flags_prefers_nearest_claim_when_no_overlap() -> None:
    claims = [
        OllamaClaim(
            id="c1",
            type="statement",
            text="first",
            start=0,
            end=10,
            assertiveness="medium",
            score_label=ScoreLabel.PLAUSIBLE,
            score_reason="r",
            evidence=[EvidenceSpan(start=0, end=10, quote="0123456789")],
        ),
        OllamaClaim(
            id="c2",
            type="statement",
            text="second",
            start=50,
            end=60,
            assertiveness="medium",
            score_label=ScoreLabel.PLAUSIBLE,
            score_reason="r",
            evidence=[EvidenceSpan(start=50, end=60, quote="abcdefghij")],
        ),
    ]
    flags = [
        engine_ollama.OllamaBiasFlag(
            bias_type="loaded_language",
            evidence=[EvidenceSpan(start=45, end=49, quote="near")],
            why="w",
            suggested_fix="f",
        )
    ]
    mapped = engine_ollama._map_bias_flags_to_claims(claims, flags)
    assert len(mapped["c1"]) == 0
    assert len(mapped["c2"]) == 1


def test_claim_in_analyst_notes_is_excluded() -> None:
    text = "## Analyst Notes\nThis is analyst commentary."
    blocks = engine_ollama._parse_text_blocks(text)
    claim = OllamaClaim(
        id="c1",
        type="statement",
        text="This is analyst commentary.",
        start=text.index("This"),
        end=len(text),
        assertiveness="low",
        score_label=ScoreLabel.SPECULATIVE,
        score_reason="r",
        evidence=[EvidenceSpan(start=text.index("This"), end=len(text), quote="This is analyst commentary.")],
    )
    assert engine_ollama._claim_in_recommendations(claim, blocks)


def test_sanitize_evidence_snaps_to_full_sentence_with_sentence_id() -> None:
    source = "alpha beta gamma delta epsilon."
    short = [EvidenceSpan(start=6, end=10, quote="beta")]
    sanitized = engine_ollama._sanitize_evidence_list(short, source)
    assert len(sanitized) == 1
    assert sanitized[0].quote == source
    assert sanitized[0].sentence_id == 0


def test_sanitize_evidence_discards_heading_fragment() -> None:
    source = "## 1. Executive Summary\nThe actor used valid credentials."
    heading_span = [EvidenceSpan(start=0, end=10, quote="## 1. Exec")]
    sanitized = engine_ollama._sanitize_evidence_list(heading_span, source)
    assert sanitized == []


def test_fallback_prefers_body_not_heading() -> None:
    source = "## 1. Executive Summary\nExecutive summary indicates account misuse in logs."
    ev = engine_ollama._derive_claim_evidence_fallback(
        source_text=source,
        claim_start=0,
        claim_end=12,
        claim_text="Executive summary indicates account misuse in logs.",
    )
    assert ev
    assert not ev[0].quote.strip().startswith("##")


def test_deterministic_annotated_markdown_includes_claim_codes() -> None:
    text = "Credential abuse was observed in the admin panel."
    span = EvidenceSpan(start=0, end=len(text), quote=text)
    claim = Claim(
        claim_id="C001",
        text=text,
        category="statement",
        score_label=ScoreLabel.SUPPORTED,
        evidence=[span],
        bias_flags=[
            BiasFlag(
                tag="loaded_language",
                suggested_fix="Use neutral wording.",
                evidence=[span],
                bias_code="B001",
            )
        ],
    )

    annotated = engine_ollama._build_deterministic_annotated_md(text, [claim])

    assert "[C001]" in annotated
    assert "[CLAIM:C001:SUP]" in annotated
    assert "[BIAS:B001:loaded_language]" in annotated


def test_choose_chunk_boundary_prefers_paragraph_break() -> None:
    paragraph_a = "A" * 900
    paragraph_b = "B" * 900
    text = f"{paragraph_a}\n\n{paragraph_b}"
    tentative_end = len(paragraph_a) + 350
    boundary = engine_ollama._choose_chunk_boundary(text, 0, tentative_end)
    assert boundary == len(paragraph_a) + 2


def test_choose_chunk_split_point_prefers_paragraph_boundary_near_middle() -> None:
    text = ("A" * 800) + "\n\n" + ("B" * 800) + "\n\n" + ("C" * 800)
    split = engine_ollama._choose_chunk_split_point(text)
    assert text[:split].endswith("\n\n")


def test_chunk_split_worthy_on_schema_json_error() -> None:
    exc = ValueError(
        "Failed to produce valid schema-compliant JSON after 2 attempts for chunk 8/10: "
        "Unterminated string starting at: line 77 column 23 (char 3398)"
    )
    assert engine_ollama._is_chunk_split_worthy_error(exc)
