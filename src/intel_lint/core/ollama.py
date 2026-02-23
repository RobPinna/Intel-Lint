from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from ..runtime import load_settings
from ..models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BiasFlag,
    Claim,
    ClaimsDocument,
    EvidenceSpan,
    OllamaAnalysisResult,
    OllamaBiasFlag,
    OllamaBiasMaskItem,
    OllamaBiasMaskResult,
    OllamaClaimExtractionResult,
    OllamaClaim,
    ScoreLabel,
)

_RUNTIME_SETTINGS = load_settings()
VERBOSE_LOGGING = bool(_RUNTIME_SETTINGS.get("verbose_logging", False))
DEFAULT_OLLAMA_HOST = _RUNTIME_SETTINGS["ollama_url"]
DEFAULT_OLLAMA_MODEL = _RUNTIME_SETTINGS["model"]
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 180.0
DEFAULT_OLLAMA_WARMUP_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_CHARS_PER_CHUNK = 5000
DEFAULT_CHUNK_OVERLAP_CHARS = 80
DEFAULT_MAX_ATTEMPTS = 1
DEFAULT_NUM_PREDICT = 800
DEFAULT_NUM_CTX = 3072
DEFAULT_NUM_BATCH = 128
DEFAULT_MAX_CLAIMS = 4
DEFAULT_MAX_BIAS_FLAGS = 80
DEFAULT_MAX_CALLS = 6
DEFAULT_MAX_SPLIT_DEPTH = 1
DEFAULT_BIAS_SENTENCES_PER_BATCH = 10
DEFAULT_BIAS_NUM_PREDICT = 384
DEFAULT_GPU_SAFE_MAX_CHARS_PER_CHUNK = 3500
DEFAULT_OLLAMA_SEED = 42
DEFAULT_DETERMINISTIC_CACHE_MAX_ENTRIES = 256
DEFAULT_STRICT_DETERMINISM_ATTEMPTS = 2
MIN_CHARS_PER_CHUNK = 700
MIN_EVIDENCE_CHARS = 32
MAX_SUGGESTED_FIX_CHARS = 600
MIN_INDEPENDENT_EVIDENCE_CHARS = 25
OLLAMA_CHAT_PATH = "/api/chat"
OLLAMA_GENERATE_PATH = "/api/generate"
ROOT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(_RUNTIME_SETTINGS["cache_dir"])
CACHE_FILE = CACHE_DIR / "ollama_deterministic_cache.json"
DEBUG_DIR = Path(_RUNTIME_SETTINGS["data_dir"]) / "debug"
DETERMINISTIC_CACHE_VERSION = "v7"
NO_INDEPENDENT_EVIDENCE_NOTE = (
    "No independent supporting evidence found in report text; manual review recommended."
)
TRUNCATED_PREFIX_PATTERNS = (
    r"^ithin\b",
    r"^ement\b",
    r"^[a-z]{2,5}\s+a\b",
)
PARAGRAPH_BREAK_PATTERN = re.compile(r"(?:\r?\n)[ \t]*(?:\r?\n)+")
BIASTYPE_FIELDS = [
    "certainty_overclaim",
    "speculation_as_fact",
    "stereotyping",
    "regional_generalization",
    "cultural_essentialism",
    "loaded_language",
    "fear_appeal",
    "single_cause_fallacy",
    "attribution_without_evidence",
    "source_imbalance",
    "normative_judgment",
    "prescriptive_overreach",
]
BIAS_BIT = {
    "certainty_overclaim": 1 << 0,
    "speculation_as_fact": 1 << 1,
    "stereotyping": 1 << 2,
    "regional_generalization": 1 << 3,
    "cultural_essentialism": 1 << 4,
    "loaded_language": 1 << 5,
    "fear_appeal": 1 << 6,
    "single_cause_fallacy": 1 << 7,
    "attribution_without_evidence": 1 << 8,
    "source_imbalance": 1 << 9,
    "normative_judgment": 1 << 10,
    "prescriptive_overreach": 1 << 11,
}
ALL_BIAS_FLAGS_MASK = sum(BIAS_BIT.values())
RECOMMENDATION_HEADERS = {
    "recommendations",
    "mitigations",
    "actions",
    "remediation",
    "immediate actions",
    "next steps",
    "containment",
    "eradication",
    "recovery",
}
ANALYST_NOTE_HEADERS = {
    "analyst notes",
    "analyst note",
    "notes",
    "analyst commentary",
    "analyst comments",
    "note dell analista",
    "note dell'analista",
    "note analista",
}
SECTION_CLASS_KEYWORDS = {
    "FINDINGS": ("key judgement", "key judgment", "key judgements", "key judgments", "finding", "findings"),
    "OSINT": ("osint", "passive osint", "shodan", "censys"),
    "EVIDENCE": ("evidence", "observation", "observations", "ioc", "indicator", "telemetry", "log", "logs"),
    "EXEC_SUMMARY": ("executive summary", "summary"),
    "ASSESSMENT": ("assessment", "analysis", "risk", "impact"),
    "ATTRIBUTION": ("attribution", "actor", "threat actor", "apt"),
    "CONTEXT": ("context", "background", "overview", "scope", "socio technical", "socio-technical"),
}
_MODEL_DIGEST_CACHE: dict[str, str] = {}
_CANDIDATE_MARKERS = (
    "should",
    "must",
    "always",
    "never",
    "therefore",
    "because",
    "attributed",
    "apt",
    "high confidence",
    "region",
    "culture",
    "maturity",
    "western",
    "inevitably",
    "guarantees",
    "accountability",
    "source",
    "according to",
    "corrobor",
    "unverified",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if VERBOSE_LOGGING else logging.INFO)


@dataclass(frozen=True)
class SentenceSpan:
    sentence_id: int
    start: int
    end: int
    text: str
    block_id: int = 0
    section_name: str = "Document"
    section_class: str = "OTHER"


@dataclass(frozen=True)
class EvidenceSentence:
    sentence_id: int
    text: str
    start: int
    end: int
    section_class: str


@dataclass(frozen=True)
class TextBlock:
    block_id: int
    section_name: str
    section_class: str
    start: int
    end: int
    text: str


def run_analysis(request: AnalyzeRequest) -> AnalyzeResponse:
    cache_key = _deterministic_cache_key(request)
    content_key = _deterministic_content_key(request)
    cache_enabled = _deterministic_cache_enabled()
    if cache_enabled:
        cached_response = _read_cached_response(cache_key)
        if cached_response is not None:
            logger.info("ollama_cache hit key=%s", cache_key[:12])
            return cached_response
        content_cached_response = _read_latest_cached_response_by_content_key(content_key)
        if content_cached_response is not None:
            logger.info("ollama_cache content-hit key=%s", content_key[:12])
            _write_cached_response(cache_key, content_cached_response, content_key)
            return content_cached_response

    if not _strict_determinism_enabled():
        response = _run_analysis_once(request)
        if cache_enabled:
            _write_cached_response(cache_key, response, content_key)
        return response

    attempts = _strict_determinism_attempts()
    candidates: list[AnalyzeResponse] = []
    fingerprints: dict[str, AnalyzeResponse] = {}

    for _ in range(attempts):
        candidate = _run_analysis_once(request)
        fp = _response_fingerprint(candidate)
        if fp in fingerprints:
            stable = fingerprints[fp]
            if cache_enabled:
                _write_cached_response(cache_key, stable, content_key)
            return stable
        fingerprints[fp] = candidate
        candidates.append(candidate)

    fallback = _read_latest_cached_response_by_content_key(content_key) if cache_enabled else None
    if fallback is not None:
        return fallback

    if candidates:
        canonical = min(candidates, key=_response_fingerprint)
        if cache_enabled:
            _write_cached_response(cache_key, canonical, content_key)
        return canonical

    unstable = _run_analysis_once(request)
    if cache_enabled:
        _write_cached_response(cache_key, unstable, content_key)
    return unstable


def _run_analysis_once(request: AnalyzeRequest) -> AnalyzeResponse:
    text = request.text
    generate_rewrite = request.generate_rewrite
    baseline_options = _ollama_options_snapshot()
    _assert_options_stable(baseline_options, context="run_start")
    blocks = _parse_text_blocks(text)
    claim_text = _mask_recommendation_text(text, blocks)
    sentences_all = _split_sentences_with_offsets(text, blocks)
    evidence_pool = _build_evidence_pool(sentences_all)
    excluded_sentences_skipped = sum(1 for sentence in sentences_all if _is_excluded_section_class(sentence.section_class))
    input_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    model = _ollama_model_name()
    options_snapshot = _ollama_options_snapshot()
    debug_trace = _start_debug_trace(
        input_hash=input_hash,
        model=model,
        generate_rewrite=generate_rewrite,
        options=options_snapshot,
    )
    chunks = _chunk_text(claim_text)
    chunk_results: list[OllamaAnalysisResult] = []
    warnings: list[str] = []
    perf_samples: list[dict[str, int]] = []
    chunk_count = len(chunks)
    call_budget = {"used": 0, "max": _max_ollama_calls(chunk_count, text)}
    runtime_profile: dict[str, Any] = {"gpu_safe_mode": _gpu_safe_mode_start(), "baseline_options": baseline_options}
    ollama_disabled = False
    started = time.perf_counter()
    logger.info(
        "ollama_run_start input_hash=%s model=%s digest=%s generate_rewrite=%s options=%s chunks=%d call_budget=%d",
        input_hash[:16],
        model,
        _ollama_model_digest(model),
        str(generate_rewrite).lower(),
        json.dumps(options_snapshot, ensure_ascii=False),
        chunk_count,
        call_budget["max"],
    )
    logger.info(
        "ollama_sections total_blocks=%d total_sentences=%d excluded_sentences=%d",
        len(blocks),
        len(sentences_all),
        excluded_sentences_skipped,
    )

    gpu_required = _require_gpu()
    if gpu_required:
        try:
            _ensure_gpu_ready_for_model()
        except ValueError as exc:
            error_text = str(exc).lower()
            if "warm-up endpoint" in error_text or "warm-up request failed" in error_text:
                if _is_timeout_error(exc):
                    warnings.append("Ollama warm-up timed out; continuing with direct analysis calls.")
                else:
                    warnings.append("Ollama warm-up failed; continuing with direct analysis calls.")
            else:
                raise
    gpu_state = _ollama_processor_state()
    if (
        not gpu_required
        and not _cpu_fallback_on_gpu_failure()
        and not gpu_state["gpu_active"]
        and _should_warn_for_gpu_state(str(gpu_state.get("processor_summary", "unknown")))
    ):
        warnings.append(f"Ollama processor is {gpu_state['processor_summary']} (GPU not active)")

    for index, (chunk_start, chunk_text) in enumerate(chunks, start=1):
        if ollama_disabled:
            fallback = _fallback_result_from_text(chunk_text)
            chunk_results.append(_shift_result_spans(fallback, chunk_start, index, chunk_count))
            continue
        try:
            chunk_results.extend(
                _run_chunk_with_timeout_fallback(
                    chunk_text=chunk_text,
                    full_text=text,
                    chunk_start=chunk_start,
                    chunk_index=index,
                    chunk_count=chunk_count,
                    blocks=blocks,
                    generate_rewrite=generate_rewrite,
                    perf_samples=perf_samples,
                    call_budget=call_budget,
                    runtime_profile=runtime_profile,
                    debug_trace=debug_trace,
                    evidence_pool=evidence_pool,
                )
            )
        except ValueError as exc:
            warnings.append(f"Chunk {index}/{chunk_count} fallback: {exc}")
            fallback = _fallback_result_from_text(chunk_text)
            chunk_results.append(_shift_result_spans(fallback, chunk_start, index, chunk_count))
            if _should_disable_ollama_after_error(exc):
                ollama_disabled = True
                warnings.append("Subsequent chunks used deterministic fallback after Ollama runner failure.")

    merged = _merge_chunk_results(chunk_results)
    filtered_claims = [claim for claim in merged.claims if not _claim_in_recommendations(claim, blocks)]
    filtered_bias = [flag for flag in merged.bias_flags if not _bias_flag_in_recommendations(flag, blocks)]
    if len(filtered_claims) != len(merged.claims):
        warnings.append("Claims in Recommendations/Actions sections were excluded.")
    merged = OllamaAnalysisResult(
        claims=filtered_claims,
        bias_flags=filtered_bias,
        annotated_md=merged.annotated_md,
        rewrite_md=merged.rewrite_md,
    )
    if ollama_disabled:
        warnings.append("Bias detection skipped after Ollama runner shutdown.")
        detected_bias_flags = merged.bias_flags
    elif not _ollama_reachable():
        warnings.append("Bias detection skipped because Ollama is unreachable; using extraction fallback.")
        detected_bias_flags = merged.bias_flags
    else:
        try:
            detected_bias_flags = _detect_bias_flags_for_text(
                source_text=text,
                blocks=blocks,
                perf_samples=perf_samples,
                call_budget=call_budget,
                runtime_profile=runtime_profile,
                debug_trace=debug_trace,
            )
        except ValueError as exc:
            if "Bias checklist validation failed" in str(exc):
                _write_debug_trace(debug_trace)
                raise
            warnings.append(f"Bias detection fallback: {exc}")
            detected_bias_flags = merged.bias_flags
    merged = OllamaAnalysisResult(
        claims=merged.claims,
        bias_flags=detected_bias_flags,
        annotated_md=merged.annotated_md,
        rewrite_md=merged.rewrite_md if generate_rewrite else "",
    )
    merged = _canonicalize_analysis_result(merged)
    response = _to_api_response(
        merged,
        source_text=text,
        blocks=blocks,
        debug_trace=debug_trace,
    )
    response.annotated_md = _build_deterministic_annotated_md(text, response.claims.claims)
    elapsed = time.perf_counter() - started
    perf_summary = _summarize_perf(perf_samples)
    debug_trace["summary"] = {
        "elapsed_seconds": round(elapsed, 6),
        "chunks": chunk_count,
        "num_sentences": len(sentences_all),
        "num_excluded_sentences_skipped": excluded_sentences_skipped,
        "calls": perf_summary["calls"],
        "prompt_eval_count": perf_summary["prompt_eval_count"],
        "eval_count": perf_summary["eval_count"],
        "prompt_eval_duration_s": perf_summary["prompt_eval_duration_s"],
        "eval_duration_s": perf_summary["eval_duration_s"],
        "load_duration_s": perf_summary["load_duration_s"],
        "total_duration_s": perf_summary["total_duration_s"],
        "dominant_lever": perf_summary["dominant_lever"],
    }
    logger.info(
        "ollama_perf total_elapsed_s=%.3f chunks=%d calls=%d prompt_eval_count=%d eval_count=%d "
        "prompt_eval_duration_s=%.3f eval_duration_s=%.3f load_duration_s=%.3f total_duration_s=%.3f dominant=%s",
        elapsed,
        chunk_count,
        perf_summary["calls"],
        perf_summary["prompt_eval_count"],
        perf_summary["eval_count"],
        perf_summary["prompt_eval_duration_s"],
        perf_summary["eval_duration_s"],
        perf_summary["load_duration_s"],
        perf_summary["total_duration_s"],
        perf_summary["dominant_lever"],
    )
    for item in perf_samples:
        logger.info(
            "ollama_perf_chunk chunk=%d phase=%s attempt=%d prompt_eval_count=%d eval_count=%d "
            "prompt_eval_duration_s=%.3f eval_duration_s=%.3f load_duration_s=%.3f total_duration_s=%.3f",
            item.get("chunk_index", 0),
            item.get("phase", "unknown"),
            item.get("attempt", 0),
            item.get("prompt_eval_count", 0),
            item.get("eval_count", 0),
            item.get("prompt_eval_duration_s", 0.0),
            item.get("eval_duration_s", 0.0),
            item.get("load_duration_s", 0.0),
            item.get("total_duration_s", 0.0),
        )
    if any(int(item.get("gpu_safe_mode_used", 0)) > 0 for item in perf_samples):
        warnings.append("GPU safe mode activated after runner instability (flash attention off, reduced ctx/batch).")
    if any(int(item.get("cpu_fallback_used", 0)) > 0 for item in perf_samples):
        warnings.append("GPU runner was unstable; Ollama completed using CPU fallback for reliability.")
    if warnings:
        response.warning = "; ".join(warnings)
    if call_budget["used"] >= call_budget["max"]:
        budget_note = f"Ollama call budget reached ({call_budget['used']}/{call_budget['max']})."
        response.warning = f"{response.warning}; {budget_note}" if response.warning else budget_note
    _write_debug_trace(debug_trace)
    return response


def _run_chunk_with_timeout_fallback(
    chunk_text: str,
    full_text: str,
    chunk_start: int,
    chunk_index: int,
    chunk_count: int,
    blocks: list[TextBlock],
    generate_rewrite: bool,
    perf_samples: list[dict[str, int]],
    call_budget: dict[str, int],
    runtime_profile: dict[str, Any],
    debug_trace: dict[str, Any],
    evidence_pool: list[EvidenceSentence],
    split_depth: int = 0,
) -> list[OllamaAnalysisResult]:
    try:
        return [
            _run_single_chunk_analysis(
                chunk_text=chunk_text,
                full_text=full_text,
                chunk_start=chunk_start,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                blocks=blocks,
                generate_rewrite=generate_rewrite,
                perf_samples=perf_samples,
                call_budget=call_budget,
                runtime_profile=runtime_profile,
                debug_trace=debug_trace,
                evidence_pool=evidence_pool,
            )
        ]
    except ValueError as exc:
        if (
            not _is_chunk_split_worthy_error(exc)
            or len(chunk_text) <= MIN_CHARS_PER_CHUNK
            or split_depth >= _max_split_depth()
        ):
            raise

    split_point = _choose_chunk_split_point(chunk_text)
    if split_point <= 0 or split_point >= len(chunk_text):
        raise ValueError(
            f"Ollama timed out and chunk could not be split safely (chunk {chunk_index}/{chunk_count})."
        )

    left_text = chunk_text[:split_point]
    right_text = chunk_text[split_point:]

    left_results = _run_chunk_with_timeout_fallback(
        chunk_text=left_text,
        full_text=full_text,
        chunk_start=chunk_start,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        blocks=blocks,
        generate_rewrite=generate_rewrite,
        perf_samples=perf_samples,
        call_budget=call_budget,
        runtime_profile=runtime_profile,
        debug_trace=debug_trace,
        evidence_pool=evidence_pool,
        split_depth=split_depth + 1,
    )
    right_results = _run_chunk_with_timeout_fallback(
        chunk_text=right_text,
        full_text=full_text,
        chunk_start=chunk_start + split_point,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        blocks=blocks,
        generate_rewrite=generate_rewrite,
        perf_samples=perf_samples,
        call_budget=call_budget,
        runtime_profile=runtime_profile,
        debug_trace=debug_trace,
        evidence_pool=evidence_pool,
        split_depth=split_depth + 1,
    )
    return left_results + right_results


def _run_single_chunk_analysis(
    chunk_text: str,
    full_text: str,
    chunk_start: int,
    chunk_index: int,
    chunk_count: int,
    blocks: list[TextBlock],
    generate_rewrite: bool,
    perf_samples: list[dict[str, int]],
    call_budget: dict[str, int],
    runtime_profile: dict[str, Any],
    debug_trace: dict[str, Any],
    evidence_pool: list[EvidenceSentence],
) -> OllamaAnalysisResult:
    claims_result = _extract_claims_for_chunk(
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        generate_rewrite=generate_rewrite,
        perf_samples=perf_samples,
        call_budget=call_budget,
        runtime_profile=runtime_profile,
        debug_trace=debug_trace,
    )
    claims_absolute = _shift_result_spans(claims_result, chunk_start, chunk_index, chunk_count)
    guarded_chunk = _apply_guardrails(
        claims_absolute,
        full_text,
        evidence_pool=evidence_pool,
        blocks=blocks,
        debug_trace=debug_trace,
    )
    guarded_claims = guarded_chunk.claims

    return OllamaAnalysisResult(
        claims=guarded_claims,
        bias_flags=guarded_chunk.bias_flags,
        annotated_md="",
        rewrite_md=claims_result.rewrite_md if generate_rewrite else "",
    )


def _extract_claims_for_chunk(
    chunk_text: str,
    chunk_index: int,
    chunk_count: int,
    generate_rewrite: bool,
    perf_samples: list[dict[str, int]],
    call_budget: dict[str, int],
    runtime_profile: dict[str, Any],
    debug_trace: dict[str, Any],
) -> OllamaAnalysisResult:
    prompt = _build_claims_prompt(chunk_text, generate_rewrite=generate_rewrite)
    last_error = "unknown error"
    previous_output = ""
    last_raw: dict[str, Any] | None = None
    attempts = _max_attempts()

    for attempt in range(1, attempts + 1):
        if call_budget["used"] >= call_budget["max"]:
            raise ValueError(f"Ollama call budget exceeded ({call_budget['max']})")
        call_budget["used"] += 1
        response_text, perf = _call_ollama(prompt, generate_rewrite, runtime_profile)
        raw_payload = {"response": response_text}
        perf["chunk_index"] = chunk_index
        perf["attempt"] = attempt
        perf["phase"] = "claims"
        perf_samples.append(perf)
        previous_output = response_text
        _record_debug_batch(
            debug_trace=debug_trace,
            phase="claims",
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            batch_index=1,
            batch_count=1,
            sentence_ids=[],
            request_summary={"chars": len(chunk_text), "generate_rewrite": generate_rewrite},
            raw_response=raw_payload,
            attempt=attempt,
            error=None,
        )
        try:
            raw = _extract_json(response_text)
            last_raw = raw
            extracted = OllamaClaimExtractionResult.model_validate(raw)
            return OllamaAnalysisResult(
                claims=extracted.claims,
                bias_flags=extracted.bias_flags,
                annotated_md="",
                rewrite_md=extracted.rewrite_md or "",
            )
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            last_error = str(exc)
            _record_debug_batch(
                debug_trace=debug_trace,
                phase="claims",
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                batch_index=1,
                batch_count=1,
                sentence_ids=[],
                request_summary={"chars": len(chunk_text), "generate_rewrite": generate_rewrite},
                raw_response=raw_payload,
                attempt=attempt,
                error=last_error,
            )
            if attempt < attempts:
                prompt = _build_claims_repair_prompt(
                    input_text=chunk_text,
                    broken_output=previous_output,
                    error=last_error,
                    generate_rewrite=generate_rewrite,
                )

    if last_raw is not None:
        coerced = _coerce_noncompliant_result(last_raw, chunk_text)
        if coerced is not None:
            return OllamaAnalysisResult(
                claims=coerced.claims,
                bias_flags=coerced.bias_flags,
                annotated_md="",
                rewrite_md=coerced.rewrite_md if generate_rewrite else "",
            )

    _write_debug_trace(debug_trace)
    raise ValueError(
        f"Failed to produce valid schema-compliant JSON after {attempts} attempts for chunk "
        f"{chunk_index}/{chunk_count}: {last_error}"
    )


def _detect_bias_flags_for_text(
    source_text: str,
    blocks: list[TextBlock],
    perf_samples: list[dict[str, int]],
    call_budget: dict[str, int],
    runtime_profile: dict[str, Any],
    debug_trace: dict[str, Any],
) -> list[OllamaBiasFlag]:
    started = time.perf_counter()
    sentences = _split_sentences_with_offsets(source_text, blocks)
    if not sentences:
        return []

    sentence_flags: dict[int, int] = {sentence.sentence_id: 0 for sentence in sentences}
    candidate_sentences: list[SentenceSpan] = []
    excluded_skipped = 0
    hard_trigger_hits = 0

    for sentence in sentences:
        if _is_excluded_section_class(sentence.section_class):
            excluded_skipped += 1
            sentence_flags[sentence.sentence_id] = 0
            continue
        hard_flags = _hard_trigger_flags(sentence.text, sentence.section_class)
        if hard_flags:
            hard_trigger_hits += 1
            sentence_flags[sentence.sentence_id] |= hard_flags
        if hard_flags or _is_candidate_sentence(sentence.text):
            candidate_sentences.append(sentence)

    batches = _sentence_batches(candidate_sentences, _bias_sentences_per_batch())
    total_batches = len(batches)
    llm_items: list[OllamaBiasMaskItem] = []
    for batch_index, batch_sentences in enumerate(batches, start=1):
        llm_items.extend(
            _run_bias_batch_check(
                batch_sentences=batch_sentences,
                batch_index=batch_index,
                batch_count=total_batches,
                perf_samples=perf_samples,
                call_budget=call_budget,
                runtime_profile=runtime_profile,
                debug_trace=debug_trace,
            )
        )
    for item in llm_items:
        sentence = sentences[item.id] if 0 <= item.id < len(sentences) else None
        if sentence is None or _is_excluded_section_class(sentence.section_class):
            continue
        llm_mask = _normalize_bias_mask(item.flags)
        llm_mask &= ~BIAS_BIT["source_imbalance"]
        sentence_flags[item.id] |= llm_mask

    logger.info(
        "ollama_bias_summary num_sentences=%d num_candidates=%d excluded_skipped=%d hard_triggers=%d num_batches=%d duration_s=%.3f",
        len(sentences),
        len(candidate_sentences),
        excluded_skipped,
        hard_trigger_hits,
        total_batches,
        time.perf_counter() - started,
    )
    debug_trace["bias_run"] = {
        "num_sentences": len(sentences),
        "num_candidates": len(candidate_sentences),
        "num_excluded_sentences_skipped": excluded_skipped,
        "num_batches": total_batches,
        "hard_trigger_hits": hard_trigger_hits,
        "skipped_sections": {
            "RECOMMENDATIONS": sum(1 for s in sentences if s.section_class == "RECOMMENDATIONS"),
            "ANALYST_NOTES": sum(1 for s in sentences if s.section_class == "ANALYST_NOTES"),
        },
    }
    return _build_bias_flags_from_sentence_flags(source_text, sentences, sentence_flags)


def _run_bias_batch_check(
    batch_sentences: list[SentenceSpan],
    batch_index: int,
    batch_count: int,
    perf_samples: list[dict[str, int]],
    call_budget: dict[str, int],
    runtime_profile: dict[str, Any],
    debug_trace: dict[str, Any],
) -> list[OllamaBiasMaskItem]:
    expected_ids = [item.sentence_id for item in batch_sentences]
    prompt = _build_bias_prompt(batch_sentences)
    schema = _bias_output_schema(expected_ids)
    previous_output = ""
    last_error = "unknown error"
    attempts = _max_attempts()

    for attempt in range(1, attempts + 1):
        if call_budget["used"] >= call_budget["max"]:
            raise ValueError(f"Ollama call budget exceeded ({call_budget['max']})")
        call_budget["used"] += 1
        response_text, perf, raw_payload = _call_ollama_structured(
            prompt,
            schema,
            runtime_profile,
            _ollama_bias_num_predict(len(batch_sentences)),
        )
        perf["chunk_index"] = 0
        perf["attempt"] = attempt
        perf["phase"] = "bias_batch"
        perf_samples.append(perf)
        previous_output = response_text

        _record_debug_batch(
            debug_trace=debug_trace,
            phase="bias_batch",
            chunk_index=0,
            chunk_count=1,
            batch_index=batch_index,
            batch_count=batch_count,
            sentence_ids=expected_ids,
            request_summary={"sentences": len(batch_sentences), "candidate_ids_count": len(expected_ids)},
            raw_response=raw_payload,
            attempt=attempt,
            error=None,
        )

        try:
            raw = _extract_json(response_text)
            parsed = OllamaBiasMaskResult.model_validate(raw)
            return _validate_bias_batch_items(parsed, expected_ids)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            last_error = str(exc)
            _record_debug_batch(
                debug_trace=debug_trace,
                phase="bias_batch",
                chunk_index=0,
                chunk_count=1,
                batch_index=batch_index,
                batch_count=batch_count,
                sentence_ids=expected_ids,
                request_summary={"sentences": len(batch_sentences), "candidate_ids_count": len(expected_ids)},
                raw_response=raw_payload,
                attempt=attempt,
                error=last_error,
            )
            if attempt < attempts:
                prompt = _build_bias_repair_prompt(
                    batch_sentences=batch_sentences,
                    broken_output=previous_output,
                    error=last_error,
                )

    _write_debug_trace(debug_trace)
    raise ValueError(
        f"Bias checklist validation failed after {attempts} attempts for batch {batch_index}/{batch_count}: {last_error}"
    )


def _validate_bias_batch_items(
    payload: OllamaBiasMaskResult, expected_ids: list[int]
) -> list[OllamaBiasMaskItem]:
    expected = set(expected_ids)
    seen: set[int] = set()
    validated: list[OllamaBiasMaskItem] = []
    for item in payload.results:
        sid = int(item.id)
        if sid not in expected:
            raise ValueError(f"Unexpected sentence_id {sid}; expected only {sorted(expected)}")
        if sid in seen:
            raise ValueError(f"Duplicate sentence_id {sid} in bias checklist output")
        seen.add(sid)
        validated.append(OllamaBiasMaskItem(id=sid, flags=int(item.flags)))

    missing = [sid for sid in expected_ids if sid not in seen]
    if missing:
        raise ValueError(f"Missing sentence_id entries in bias checklist output: {missing}")
    return sorted(validated, key=lambda item: item.id)


def _build_bias_flags_from_sentence_flags(
    source_text: str,
    sentences: list[SentenceSpan],
    sentence_flags: dict[int, int],
) -> list[OllamaBiasFlag]:
    flags: list[OllamaBiasFlag] = []
    seen: set[tuple[str, int, int]] = set()
    limit = _max_bias_flags()

    for sentence in sentences:
        raw_mask = sentence_flags.get(sentence.sentence_id, 0)
        active_types = _decode_bias_mask(raw_mask)
        active_types = _apply_attribution_caveat_guard(active_types, sentence.text, sentence.sentence_id)
        active_types = _prune_bias_types(active_types)
        if not active_types:
            continue

        quote = source_text[sentence.start : sentence.end]
        evidence = [
            EvidenceSpan(
                start=sentence.start,
                end=sentence.end,
                quote=quote,
                sentence_id=sentence.sentence_id,
            )
        ]
        for bias_type in active_types:
            key = (bias_type, sentence.start, sentence.end)
            if key in seen:
                continue
            seen.add(key)
            flags.append(
                OllamaBiasFlag(
                    bias_type=bias_type,
                    evidence=evidence,
                    why=_bias_why_message(bias_type, sentence.text),
                    suggested_fix=_bias_fix_message(bias_type),
                )
            )
            if len(flags) >= limit:
                logger.warning("Bias flags capped at %d (set OLLAMA_MAX_BIAS_FLAGS to adjust).", limit)
                return flags
    return flags


def _prune_bias_types(bias_types: list[str]) -> list[str]:
    """
    Drop overly-generic bias tags when more specific ones are present in the same sentence.
    Example: if cultural/geo biases are detected, avoid also returning normative_judgment.
    """
    if len(bias_types) <= 1:
        return bias_types
    pruned = [b for b in bias_types if b != "normative_judgment"]
    return pruned or bias_types


def _apply_attribution_caveat_guard(bias_types: list[str], sentence_text: str, sentence_id: int | None = None) -> list[str]:
    if not bias_types:
        return bias_types
    lowered = (sentence_text or "").lower()
    patterns = [
        "attribution is not confirmed",
        "not confirmed",
        "no reliable evidence",
        "insufficient evidence",
        "cannot attribute",
        "attribution is not supported",
        "unable to attribute",
        "attribution remains unconfirmed",
        "no evidence supports assigning",
    ]
    if any(pat in lowered for pat in patterns):
        if "attribution_without_evidence" in bias_types:
            bias_types = [b for b in bias_types if b != "attribution_without_evidence"]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "attribution_caveat_guard cleared attribution_without_evidence sentence_id=%s match=%s",
                    sentence_id,
                    next((pat for pat in patterns if pat in lowered), "unknown"),
                )
    return bias_types


def _decode_bias_mask(mask: int) -> list[str]:
    active: list[str] = []
    normalized = _normalize_bias_mask(mask)
    for bias_type in BIASTYPE_FIELDS:
        bit = BIAS_BIT.get(bias_type, 0)
        if bit and (normalized & bit):
            active.append(bias_type)
    return active


def _normalize_bias_mask(mask: int) -> int:
    value = int(mask) if isinstance(mask, (int, bool)) else 0
    if value < 0:
        return 0
    return value & ALL_BIAS_FLAGS_MASK


def _is_candidate_sentence(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _CANDIDATE_MARKERS)


def _hard_trigger_flags(text: str, section_class: str) -> int:
    if _is_excluded_section_class(section_class):
        return 0
    lowered = str(text or "").lower()
    if not lowered:
        return 0
    mask = 0

    region_hit = bool(re.search(r"\b(region|mena|emerging markets|this region)\b", lowered))
    culture_hit = bool(re.search(r"\b(culture|low maturity|accountability)\b", lowered))
    west_hit = bool(re.search(r"\b(western|europe|us)\b", lowered))
    if region_hit and culture_hit and west_hit:
        mask |= BIAS_BIT["cultural_essentialism"] | BIAS_BIT["regional_generalization"]

    timebox_hit = bool(re.search(r"\b(within\s+(24|48)\s+hours?|one\s+week)\b", lowered))
    mandate_hit = bool(re.search(r"\b(replace all|deploy across all|mandate)\b", lowered))
    if timebox_hit and mandate_hit and not _is_excluded_section_class(section_class):
        mask |= BIAS_BIT["prescriptive_overreach"]

    absolute_hit = bool(re.search(r"\b(always|inevitably|guarantees?|proves?|confirmed)\b", lowered))
    caveat_hit = bool(re.search(r"\b(likely|may|possible)\b", lowered))
    if absolute_hit and not caveat_hit:
        mask |= BIAS_BIT["certainty_overclaim"]

    if _deterministic_source_imbalance(lowered):
        mask |= BIAS_BIT["source_imbalance"]

    return mask


def _deterministic_source_imbalance(lowered_text: str) -> bool:
    text = str(lowered_text or "")
    if not text:
        return False
    patterns = [
        r"\b(single|one|sole|only)\s+source\b",
        r"\bwithout\s+(independent\s+)?corroboration\b",
        r"\b(not|never)\s+corroborated\b",
        r"\bunverified\s+source\b",
        r"\baccording to\s+(a|one|single)\s+\w+\s+source\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _bias_why_message(bias_type: str, sentence_text: str) -> str:
    clean_sentence = _normalize_human_text(sentence_text)
    if not clean_sentence:
        return f"Sentence matches checklist condition: {_humanize_bias_type(bias_type)}."
    return (
        f"Sentence shows {_humanize_bias_type(bias_type)} pattern based on wording: "
        f"\"{clean_sentence}\"."
    )


def _bias_fix_message(bias_type: str) -> str:
    return _default_fix_for_bias_type(bias_type)


def _humanize_bias_type(bias_type: str) -> str:
    return " ".join(str(bias_type).split("_")).strip()


def _parse_text_blocks(text: str) -> list[TextBlock]:
    source = text or ""
    if not source:
        return []

    header_regex = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*$", flags=re.IGNORECASE | re.MULTILINE)
    plain_regex = re.compile(
        r"^\s*(recommendations|mitigations|actions|remediation|immediate actions|next steps|containment|eradication|recovery|analyst notes|analyst note|notes|analyst commentary|analyst comments|note dell[' ]analista|note analista)\s*:?\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    header_entries: list[tuple[int, int, str]] = []
    for match in header_regex.finditer(source):
        header_entries.append((match.start(), match.end(), match.group(1).strip()))
    for match in plain_regex.finditer(source):
        header_entries.append((match.start(), match.end(), match.group(1).strip()))
    headers = sorted(header_entries, key=lambda item: item[0])
    deduped_headers: list[tuple[int, int, str]] = []
    seen_positions: set[tuple[int, int]] = set()
    for item in headers:
        key = (item[0], item[1])
        if key in seen_positions:
            continue
        seen_positions.add(key)
        deduped_headers.append(item)
    headers = deduped_headers
    blocks: list[TextBlock] = []
    cursor = 0
    block_id = 0

    if not headers:
        section_name = "Document"
        section_class = _section_class_for_name(section_name)
        return [
            TextBlock(
                block_id=0,
                section_name=section_name,
                section_class=section_class,
                start=0,
                end=len(source),
                text=source,
            )
        ]

    for index, header in enumerate(headers):
        header_start, header_end, header_name = header
        if header_start > cursor:
            pre_text = source[cursor:header_start]
            pre_name = "Document"
            pre_class = _section_class_for_name(pre_name)
            blocks.append(
                TextBlock(
                    block_id=block_id,
                    section_name=pre_name,
                    section_class=pre_class,
                    start=cursor,
                    end=header_start,
                    text=pre_text,
                )
            )
            block_id += 1

        section_name = header_name
        body_end = headers[index + 1][0] if index + 1 < len(headers) else len(source)
        section_class = _section_class_for_name(section_name)
        blocks.append(
            TextBlock(
                block_id=block_id,
                section_name=section_name or "Document",
                section_class=section_class,
                start=header_start,
                end=body_end,
                text=source[header_start:body_end],
            )
        )
        block_id += 1
        cursor = body_end

    if cursor < len(source):
        tail_name = "Document"
        tail_class = _section_class_for_name(tail_name)
        blocks.append(
            TextBlock(
                block_id=block_id,
                section_name=tail_name,
                section_class=tail_class,
                start=cursor,
                end=len(source),
                text=source[cursor:],
            )
        )

    return blocks


def _section_class_for_name(section_name: str) -> str:
    normalized = _normalize_section_name(section_name)
    if normalized in RECOMMENDATION_HEADERS:
        return "RECOMMENDATIONS"
    if normalized in ANALYST_NOTE_HEADERS:
        return "ANALYST_NOTES"
    for section_class, keywords in SECTION_CLASS_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return section_class
    return "OTHER"


def _normalize_section_name(section_name: str) -> str:
    normalized = " ".join(str(section_name or "").strip().lower().split())
    normalized = normalized.replace("'", " ").replace("`", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = " ".join(normalized.split())
    return normalized


def _is_excluded_section_class(section_class: str) -> bool:
    return section_class in {"RECOMMENDATIONS", "ANALYST_NOTES"}


def _mask_recommendation_text(source_text: str, blocks: list[TextBlock]) -> str:
    if not source_text:
        return source_text
    chars = list(source_text)
    for block in blocks:
        if not _is_excluded_section_class(block.section_class):
            continue
        for index in range(block.start, min(block.end, len(chars))):
            if chars[index] not in {"\n", "\r"}:
                chars[index] = " "
    return "".join(chars)


def _split_sentences_with_offsets(text: str, blocks: list[TextBlock] | None = None) -> list[SentenceSpan]:
    source = text or ""
    length = len(source)
    if length == 0:
        return []
    source_blocks = blocks if blocks is not None and blocks else _parse_text_blocks(source)

    sentences: list[SentenceSpan] = []
    sentence_start = 0
    sentence_id = 0
    idx = 0
    while idx < length:
        ch = source[idx]
        if ch in {"\n", "\r"}:
            sentence_id = _append_sentence_span(sentences, source, sentence_start, idx, sentence_id, source_blocks)
            if ch == "\r" and idx + 1 < length and source[idx + 1] == "\n":
                idx += 1
            sentence_start = idx + 1
            idx += 1
            continue

        if ch in {".", "?", "!"}:
            next_idx = idx + 1
            if next_idx >= length or source[next_idx].isspace():
                sentence_id = _append_sentence_span(
                    sentences, source, sentence_start, next_idx, sentence_id, source_blocks
                )
                sentence_start = next_idx
        idx += 1

    _append_sentence_span(sentences, source, sentence_start, length, sentence_id, source_blocks)
    return sentences


def _append_sentence_span(
    target: list[SentenceSpan],
    source: str,
    start: int,
    end: int,
    next_sentence_id: int,
    blocks: list[TextBlock],
) -> int:
    text_len = len(source)
    s = max(0, min(text_len, int(start)))
    e = max(0, min(text_len, int(end)))
    if e <= s:
        return next_sentence_id
    while s < e and source[s].isspace():
        s += 1
    while e > s and source[e - 1].isspace():
        e -= 1
    if e <= s:
        return next_sentence_id
    block = _block_for_span(blocks, s, e)
    target.append(
        SentenceSpan(
            sentence_id=next_sentence_id,
            start=s,
            end=e,
            text=source[s:e],
            block_id=block.block_id if block is not None else 0,
            section_name=block.section_name if block is not None else "Document",
            section_class=block.section_class if block is not None else "OTHER",
        )
    )
    return next_sentence_id + 1


EVIDENCE_SECTION_CLASSES = {"EVIDENCE", "FINDINGS", "OSINT", "CONTEXT", "EXEC_SUMMARY"}


def _build_evidence_pool(sentences: list[SentenceSpan]) -> list["EvidenceSentence"]:
    pool: list[EvidenceSentence] = []
    for s in sentences:
        if s.section_class not in EVIDENCE_SECTION_CLASSES:
            continue
        pool.append(
            EvidenceSentence(
                sentence_id=s.sentence_id,
                text=s.text,
                start=s.start,
                end=s.end,
                section_class=s.section_class,
            )
        )
    return pool


def _allowed_evidence_classes_for_claim(claim: OllamaClaim, block: TextBlock | None) -> set[str]:
    claim_text = (claim.text or "").lower()
    section_name = (block.section_name if block is not None else "").lower()
    section_class = (block.section_class if block is not None else "").upper()

    is_findings_claim = (
        section_class in {"FINDINGS", "OSINT"}
        or "key judgement" in section_name
        or "key judgment" in section_name
        or "finding" in section_name
    )
    is_context_claim = (
        section_class in {"CONTEXT", "EXEC_SUMMARY"}
        or "executive summary" in section_name
        or "context" in section_name
        or "socio-technical" in claim_text
        or "socio technical" in claim_text
    )

    if is_findings_claim:
        return {"FINDINGS", "OSINT", "EVIDENCE"}
    if is_context_claim:
        return {"CONTEXT", "EXEC_SUMMARY"}
    return {"EVIDENCE", "FINDINGS", "OSINT"}


_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "for",
    "on",
    "at",
    "with",
    "by",
    "from",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "it",
    "its",
    "their",
    "his",
    "her",
    "our",
    "your",
}


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", text.lower()) if t not in _STOPWORDS}


def _tech_bonus(text: str) -> int:
    bonus = 0
    lowered = text.lower()
    if "/" in text:
        bonus += 1
    keywords = ["dns", "ttl", "whois", "access log", "telemetry", "ioc", "indicator"]
    if any(k in lowered for k in keywords):
        bonus += 1
    if re.search(r"\b[a-f0-9]{32,64}\b", lowered):
        bonus += 1
    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", lowered):
        bonus += 1
    if re.search(r"https?://", lowered):
        bonus += 1
    return bonus


def _link_claim_to_evidence(
    claim: OllamaClaim,
    pool: list["EvidenceSentence"],
    max_items: int = 3,
    allowed_classes: set[str] | None = None,
) -> list[EvidenceSpan]:
    evidence, _scores = _link_claim_to_evidence_with_scores(
        claim,
        pool,
        max_items=max_items,
        allowed_classes=allowed_classes,
    )
    return evidence


def _link_claim_to_evidence_with_scores(
    claim: OllamaClaim,
    pool: list["EvidenceSentence"],
    max_items: int = 3,
    allowed_classes: set[str] | None = None,
) -> tuple[list[EvidenceSpan], list[tuple[int, float]]]:
    if not pool:
        return [], []
    claim_tokens = _tokenize(claim.text)
    scored: list[tuple[float, EvidenceSentence]] = []
    for ev in pool:
        if allowed_classes is not None and ev.section_class not in allowed_classes:
            continue
        ev_tokens = _tokenize(ev.text)
        inter = len(claim_tokens & ev_tokens) if ev_tokens else 0
        union = len(claim_tokens | ev_tokens) or max(1, len(claim_tokens))
        jaccard = inter / union
        score = jaccard + 0.1 * _tech_bonus(ev.text)
        scored.append((score, ev))
    scored.sort(key=lambda item: item[0], reverse=True)

    best = [(score, ev) for score, ev in scored if score > 0]
    selected = []
    scores = []
    for score, ev in best[:max_items]:
        selected.append(EvidenceSpan(start=ev.start, end=ev.end, quote=ev.text, sentence_id=ev.sentence_id))
    for score, ev in scored[:5]:
        scores.append((ev.sentence_id, round(score, 4)))
    return selected, scores


def _block_for_span(blocks: list[TextBlock], start: int, end: int) -> TextBlock | None:
    for block in blocks:
        if start >= block.start and end <= block.end:
            return block
    for block in blocks:
        if start < block.end and end > block.start:
            return block
    return blocks[0] if blocks else None


def _claim_in_recommendations(claim: OllamaClaim, blocks: list[TextBlock]) -> bool:
    spans: list[tuple[int, int]] = [(int(claim.start), int(claim.end))]
    spans.extend((int(ev.start), int(ev.end)) for ev in claim.evidence)
    for start, end in spans:
        block = _block_for_span(blocks, start, end)
        if block is not None and _is_excluded_section_class(block.section_class):
            return True
    return False


def _bias_flag_in_recommendations(flag: OllamaBiasFlag, blocks: list[TextBlock]) -> bool:
    for ev in flag.evidence:
        block = _block_for_span(blocks, int(ev.start), int(ev.end))
        if block is not None and _is_excluded_section_class(block.section_class):
            return True
    return False


def _sentence_batches(sentences: list[SentenceSpan], size: int) -> list[list[SentenceSpan]]:
    if not sentences:
        return []
    step = max(1, int(size))
    return [sentences[i : i + step] for i in range(0, len(sentences), step)]


def _claim_output_schema(generate_rewrite: bool) -> dict[str, Any]:
    evidence_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "start": {"type": "integer", "minimum": 0},
            "end": {"type": "integer", "minimum": 0},
            "quote": {"type": "string", "minLength": 1},
        },
        "required": ["start", "end", "quote"],
    }
    claim_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "type": {"type": "string", "minLength": 1},
            "text": {"type": "string", "minLength": 1},
            "start": {"type": "integer", "minimum": 0},
            "end": {"type": "integer", "minimum": 0},
            "assertiveness": {"type": "string", "minLength": 1},
            "score_label": {"type": "string", "enum": [item.value for item in ScoreLabel]},
            "score_reason": {"type": "string", "minLength": 1},
            "evidence": {"type": "array", "items": evidence_schema, "minItems": 1, "maxItems": 2},
        },
        "required": [
            "id",
            "type",
            "text",
            "start",
            "end",
            "assertiveness",
            "score_label",
            "score_reason",
            "evidence",
        ],
    }
    bias_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "bias_type": {"type": "string", "minLength": 1},
            "evidence": {"type": "array", "items": evidence_schema, "minItems": 1, "maxItems": 2},
            "why": {"type": "string", "minLength": 1},
            "suggested_fix": {"type": "string", "minLength": 1},
        },
        "required": ["bias_type", "evidence", "why", "suggested_fix"],
    }
    properties: dict[str, Any] = {
        "claims": {"type": "array", "items": claim_schema, "maxItems": _max_claims()},
        "bias_flags": {"type": "array", "items": bias_schema, "maxItems": _max_bias_flags()},
    }
    required = ["claims"]
    if generate_rewrite:
        properties["rewrite_md"] = {"type": "string"}
        required.append("rewrite_md")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _bias_output_schema(sentence_ids: list[int]) -> dict[str, Any]:
    item_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "integer", "enum": sentence_ids},
            "flags": {"type": "integer", "minimum": 0, "maximum": ALL_BIAS_FLAGS_MASK},
        },
        "required": ["id", "flags"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "items": item_schema,
                "minItems": len(sentence_ids),
                "maxItems": len(sentence_ids),
            }
        },
        "required": ["results"],
    }


def _build_claims_prompt(input_text: str, generate_rewrite: bool) -> str:
    schema_json = json.dumps(_claim_output_schema(generate_rewrite), ensure_ascii=False)
    rewrite_rule = (
        "- Include rewrite_md as polished markdown, preserving section structure from input when possible.\n"
        if generate_rewrite
        else "- Do not include rewrite_md.\n"
    )
    return (
        "Return strict JSON only, matching the schema exactly.\n"
        "Role: You are a senior cyber threat intelligence analyst auditing the report.\n"
        "Task: extract factual claims with evidence spans from input text and surface any that are weakly supported.\n"
        "Rules:\n"
        f"- At most {_max_claims()} claims.\n"
        "- Every claim must include at least one evidence span from the input text.\n"
        "- If evidence is weak or absent, set score_label to SPECULATIVE and explain briefly.\n"
        "- Be vigilant for cultural, social, or geographic bias implied in the claim wording or evidence; record these as bias_flags where applicable.\n"
        "- Do not defend, rewrite, or improve the source report; only audit it.\n"
        "- No extra fields.\n"
        f"{rewrite_rule}"
        f"JSON Schema: {schema_json}\n\n"
        "Input text:\n"
        "---BEGIN INPUT---\n"
        f"{input_text}\n"
        "---END INPUT---"
    )


def _build_claims_repair_prompt(
    input_text: str, broken_output: str, error: str, generate_rewrite: bool
) -> str:
    schema_json = json.dumps(_claim_output_schema(generate_rewrite), ensure_ascii=False)
    rewrite_rule = (
        "rewrite_md is required and must be a markdown string.\n"
        if generate_rewrite
        else "rewrite_md must not be present.\n"
    )
    return (
        "Repair your previous response.\n"
        "Return strict JSON only, no markdown fences.\n"
        f"Validation error:\n{error}\n\n"
        f"{rewrite_rule}"
        f"JSON Schema: {schema_json}\n\n"
        f"Previous output:\n{broken_output}\n\n"
        "Input text:\n"
        "---BEGIN INPUT---\n"
        f"{input_text}\n"
        "---END INPUT---"
    )


def _build_bias_prompt(sentences: list[SentenceSpan]) -> str:
    sentence_payload = [{"id": item.sentence_id, "text": item.text} for item in sentences]
    schema_json = json.dumps(_bias_output_schema([item.sentence_id for item in sentences]), ensure_ascii=False)
    return (
        "Return strict JSON only.\n"
        "Role: senior cyber threat intelligence analyst. Identify bias in each sentence, with attention to cultural, social, and geographic framing.\n"
        "Output must be minified JSON on a single line (no prose, no markdown).\n"
        "Evaluate each sentence ID independently.\n"
        "You must return exactly one result per sentence_id provided.\n"
        "Do not invent IDs and do not omit IDs.\n"
        f"Bit mapping: {json.dumps(BIAS_BIT, ensure_ascii=False)}.\n"
        "Return flags as integer OR of matching bits for each sentence.\n"
        f"JSON Schema: {schema_json}\n"
        f"Sentences JSON: {json.dumps(sentence_payload, ensure_ascii=False)}"
    )


def _build_bias_repair_prompt(
    batch_sentences: list[SentenceSpan], broken_output: str, error: str
) -> str:
    sentence_ids = [item.sentence_id for item in batch_sentences]
    schema_json = json.dumps(_bias_output_schema(sentence_ids), ensure_ascii=False)
    sentence_payload = [{"id": item.sentence_id, "text": item.text} for item in batch_sentences]
    return (
        "Repair your previous response.\n"
        "Return strict JSON only, matching schema exactly.\n"
        "Output must be minified JSON on a single line.\n"
        "Mandatory constraints: one record per sentence_id, no duplicates, no missing IDs.\n"
        f"Bit mapping: {json.dumps(BIAS_BIT, ensure_ascii=False)}.\n"
        f"Validation error: {error}\n"
        f"JSON Schema: {schema_json}\n"
        f"Sentence IDs: {sentence_ids}\n"
        f"Previous output: {broken_output}\n"
        f"Sentences JSON: {json.dumps(sentence_payload, ensure_ascii=False)}"
    )


def _ollama_base_url() -> str:
    host = os.getenv("OLLAMA_URL", "").strip() or os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).strip()
    if not host:
        host = DEFAULT_OLLAMA_HOST
    return host.rstrip("/")


def _ollama_model_name() -> str:
    model = os.getenv("MODEL", "").strip() or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
    if not model:
        model = DEFAULT_OLLAMA_MODEL
    return model


def _ollama_url(path: str) -> str:
    return f"{_ollama_base_url()}{path}"


def _ollama_generate_url() -> str:
    return _ollama_url(OLLAMA_GENERATE_PATH)


def _require_gpu() -> bool:
    raw = os.getenv("OLLAMA_REQUIRE_GPU", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _ensure_gpu_ready_for_model() -> None:
    state = _ollama_processor_state()
    summary = str(state.get("processor_summary", "unreachable"))
    if summary == "unreachable":
        raise ValueError(
            "Could not reach Ollama while checking GPU status. "
            "Verify OLLAMA_HOST and that Ollama is running."
        )

    # /api/ps reports only loaded models. If idle, warm the model once then re-check processor.
    if summary == "idle":
        _warmup_model_for_processor_detection()
        state = _ollama_processor_state()
        summary = str(state.get("processor_summary", "unreachable"))

    if not state.get("gpu_active", False):
        raise ValueError(
            f"Ollama is not using GPU (processor={summary}). "
            "Close 'ollama app', start 'ollama serve' with a GPU runtime profile "
            "(IPEX-LLM/XPU for Intel Arc or Vulkan where supported), "
            "and verify with `ollama ps`."
        )


def _warmup_model_for_processor_detection() -> None:
    model = _ollama_model_name()
    url = _ollama_generate_url()
    num_gpu = _ollama_num_gpu_override()
    options: dict[str, Any] = {
        "temperature": 0,
        "num_predict": 1,
        "num_ctx": 1024,
        "num_batch": min(_ollama_num_batch(), 32),
    }
    seed = _ollama_seed()
    if seed is not None:
        options["seed"] = seed
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    payload = {
        "model": model,
        "prompt": "ping",
        "stream": False,
        "options": options,
    }
    try:
        with httpx.Client(timeout=_httpx_timeout(_ollama_warmup_timeout_seconds())) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ValueError(
            f"Ollama warm-up request failed ({exc.response.status_code}) at '{url}'. "
            "Check OLLAMA_MODEL and pull it if missing."
        ) from exc
    except httpx.RequestError as exc:
        raise ValueError(
            f"Could not reach Ollama warm-up endpoint '{url}': {exc}. "
            "Verify OLLAMA_HOST and that Ollama is running."
        ) from exc


def _ollama_processor_state() -> dict[str, Any]:
    url = _ollama_url("/api/ps")
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return {"gpu_active": False, "processor_summary": "unreachable"}

    models = data.get("models", []) if isinstance(data, dict) else []
    processors: list[str] = []
    gpu_active = False
    total_vram_bytes = 0
    for model in models:
        if not isinstance(model, dict):
            continue
        proc = str(model.get("processor", "")).strip()
        if proc:
            processors.append(proc)
            if "gpu" in proc.lower():
                gpu_active = True
        try:
            size_vram = int(model.get("size_vram", 0) or 0)
        except (TypeError, ValueError):
            size_vram = 0
        if size_vram > 0:
            total_vram_bytes += size_vram
            gpu_active = True

    if processors:
        summary = ", ".join(processors)
        return {"gpu_active": gpu_active, "processor_summary": summary}

    # IPEX-Ollama builds may omit "processor" but report VRAM usage.
    if models:
        if total_vram_bytes > 0:
            return {
                "gpu_active": True,
                "processor_summary": f"GPU (size_vram={_format_binary_bytes(total_vram_bytes)})",
            }
        return {"gpu_active": False, "processor_summary": "100% CPU"}

    cli_state = _ollama_processor_state_from_cli()
    if cli_state is not None:
        return cli_state

    summary = "unknown" if models else "idle"
    return {"gpu_active": gpu_active, "processor_summary": summary}


def _ollama_reachable() -> bool:
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(_ollama_url("/api/tags"))
        return resp.status_code == 200
    except Exception:
        return False


def _ollama_processor_state_from_cli() -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        return None

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return {"gpu_active": False, "processor_summary": "idle"}

    processors: list[str] = []
    for line in lines[1:]:
        match = re.search(r"(\d+%\s+(?:CPU|GPU)|\bCPU\b|\bGPU\b)", line, flags=re.IGNORECASE)
        if match:
            processors.append(match.group(1).upper())

    if not processors:
        return None

    summary = ", ".join(processors)
    gpu_active = any("GPU" in item for item in processors)
    return {"gpu_active": gpu_active, "processor_summary": summary}


def _call_ollama(
    prompt: str, generate_rewrite: bool, runtime_profile: dict[str, Any] | None = None
) -> tuple[str, dict[str, int]]:
    schema = _claim_output_schema(generate_rewrite=generate_rewrite)
    response_text, perf, _raw = _call_ollama_structured(
        prompt=prompt,
        schema=schema,
        runtime_profile=runtime_profile,
        num_predict=_ollama_num_predict(),
    )
    return response_text, perf


def _call_ollama_generate_fallback(
    prompt: str, generate_rewrite: bool, runtime_profile: dict[str, Any] | None = None
) -> tuple[str, dict[str, int]]:
    # Backward-compatible wrapper used by tests and legacy call sites.
    return _call_ollama(prompt, generate_rewrite, runtime_profile)


def _call_ollama_structured(
    prompt: str,
    schema: dict[str, Any],
    runtime_profile: dict[str, Any] | None = None,
    num_predict: int | None = None,
) -> tuple[str, dict[str, int], dict[str, Any]]:
    model = _ollama_model_name()
    url = _ollama_generate_url()
    timeout_seconds = _ollama_timeout_seconds()
    predict_limit = num_predict if num_predict is not None else _ollama_num_predict()
    num_ctx = _ollama_num_ctx()
    num_batch = _ollama_num_batch()
    num_gpu = _ollama_num_gpu_override()
    baseline_options = (runtime_profile or {}).get("baseline_options")
    if baseline_options is not None:
        _assert_options_stable(baseline_options, context="ollama_call")
    options: dict[str, Any] = {
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "min_p": 0,
        "typical_p": 1,
        "repeat_penalty": 1,
        "repeat_last_n": 64,
        "mirostat": 0,
        "num_predict": predict_limit,
        "num_ctx": num_ctx,
        "num_batch": num_batch,
    }
    seed = _ollama_seed()
    if seed is not None:
        options["seed"] = seed
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    profile = runtime_profile or {"gpu_safe_mode": False}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "raw": True,
        "keep_alive": -1,
        "options": options,
    }
    if profile.get("gpu_safe_mode", False):
        _apply_gpu_safe_profile(payload, num_ctx=num_ctx, num_batch=num_batch, attempt_index=1)
    backoff_s = 1.0
    cpu_fallback_used = False
    gpu_safe_used = bool(profile.get("gpu_safe_mode", False))
    for transport_attempt in range(1, 4):
        try:
            with httpx.Client(timeout=_httpx_timeout(timeout_seconds)) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                break
        except httpx.HTTPStatusError as exc:
            detail = _http_error_detail(exc.response)
            detail_lower = detail.lower()
            transient_runner_error = (
                "runner has unexpectedly stopped" in detail_lower
                or "resource limitations" in detail_lower
                or "resource limits" in detail_lower
            )
            if transient_runner_error and transport_attempt < 3:
                if _gpu_safe_mode_on_error():
                    profile["gpu_safe_mode"] = True
                    _apply_gpu_safe_profile(
                        payload, num_ctx=num_ctx, num_batch=num_batch, attempt_index=transport_attempt
                    )
                    gpu_safe_used = True
                if _cpu_fallback_on_gpu_failure():
                    # Intel Arc GPU paths can intermittently fail on long/structured prompts.
                    # Fallback to CPU for this attempt to complete the request deterministically.
                    payload["options"]["num_gpu"] = 0
                    cpu_fallback_used = True
                time.sleep(backoff_s)
                backoff_s *= 2
                continue
            if transient_runner_error:
                raise ValueError(
                    f"Ollama runner stopped due to resource limits for model '{model}' at '{url}'. "
                    "Reduce OLLAMA_NUM_CTX / OLLAMA_NUM_PREDICT / OLLAMA_MAX_CHARS_PER_CHUNK, "
                    "or restart Ollama with a stable GPU profile."
                ) from exc
            raise ValueError(
                f"Ollama request failed ({exc.response.status_code}) for model '{model}' at '{url}'. "
                f"Detail: {detail}. "
                "Verify OLLAMA_HOST, OLLAMA_MODEL, and run: ollama pull <model>."
            ) from exc
        except httpx.RequestError as exc:
            if transport_attempt < 2:
                time.sleep(backoff_s)
                backoff_s *= 2
                continue
            raise ValueError(
                f"Could not reach Ollama at '{url}' for model '{model}': {exc}. "
                "Verify OLLAMA_HOST, that Ollama is running, and increase OLLAMA_TIMEOUT_SECONDS if model warm-up is slow."
            ) from exc

    response_text = data.get("response", "")
    if not response_text:
        raise ValueError("Ollama returned empty response body")
    perf = _extract_perf_from_ollama_payload(data)
    perf["cpu_fallback_used"] = 1 if cpu_fallback_used else 0
    perf["gpu_safe_mode_used"] = 1 if gpu_safe_used else 0
    return response_text, perf, data


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("Ollama output JSON must be an object")
    return data


def _http_error_detail(response: httpx.Response) -> str:
    try:
        response.read()
    except Exception:
        pass

    try:
        parsed = response.json()
        if isinstance(parsed, dict) and parsed.get("error"):
            return str(parsed["error"])
    except Exception:
        pass

    try:
        text = response.text.strip()
        if text:
            return text
    except Exception:
        pass

    return "No error detail returned by Ollama."


def _build_initial_prompt(input_text: str, generate_rewrite: bool) -> str:
    max_claims = _max_claims()
    schema_json = json.dumps(_structured_output_schema(generate_rewrite), ensure_ascii=False)
    rewrite_field_line = "- rewrite_md: string\n" if generate_rewrite else ""
    rewrite_rule_line = (
        "- Set rewrite_md to concise markdown.\n\n" if generate_rewrite else "- Do not include rewrite_md.\n\n"
    )
    return (
        "Return strict JSON only.\n"
        "Role: act as a senior cyber threat intelligence analyst. Audit the CTI report, find claims lacking sufficient evidence, and detect bias (especially cultural, social, or geographic).\n"
        "One JSON object with fields:\n"
        "- claims: array of objects with fields id, type, text, start, end, assertiveness, "
        "score_label(SUPPORTED|PLAUSIBLE|SPECULATIVE), score_reason, evidence[{start,end,quote}]\n"
        "- bias_flags: array of objects with fields bias_type, evidence[{start,end,quote}], why, suggested_fix\n"
        f"{rewrite_field_line}\n"
        "Rules:\n"
        f"- At most {max_claims} claims.\n"
        "- 1 evidence span per claim unless absolutely needed.\n"
        "- Keep quotes short and exact from input.\n"
        "- If evidence missing set score_label=SPECULATIVE.\n"
        "- Bias flags must highlight cultural/social/geographic framing when present; every bias flag requires an evidence span.\n"
        "- Do not add external facts; evaluate only the provided text.\n"
        "- No extra fields.\n"
        f"- Follow this JSON Schema exactly: {schema_json}\n"
        f"{rewrite_rule_line}"
        f"- Rewrite mode: {'ON' if generate_rewrite else 'OFF'}.\n"
        "Input text:\n"
        "---BEGIN INPUT---\n"
        f"{input_text}\n"
        "---END INPUT---"
    )


def _build_repair_prompt(input_text: str, broken_output: str, error: str, generate_rewrite: bool) -> str:
    schema_json = json.dumps(_structured_output_schema(generate_rewrite), ensure_ascii=False)
    rewrite_required_line = "- rewrite_md\n" if generate_rewrite else ""
    rewrite_mode_line = (
        "Rewrite mode is ON. rewrite_md is required.\n\n"
        if generate_rewrite
        else "Rewrite mode is OFF. rewrite_md must not be included.\n\n"
    )
    return (
        "Fix your JSON. Return strict JSON only.\n"
        f"Validation error:\n{error}\n\n"
        "Required fields:\n"
        "- claims[].id,type,text,start,end,assertiveness,score_label,score_reason,evidence[].start,end,quote\n"
        "- bias_flags[].bias_type,evidence[].start,end,quote,why,suggested_fix\n"
        f"{rewrite_required_line}\n"
        f"JSON Schema (must match exactly): {schema_json}\n\n"
        f"{rewrite_mode_line}"
        f"Previous output:\n{broken_output}\n"
        "Input text:\n"
        "---BEGIN INPUT---\n"
        f"{input_text}\n"
        "---END INPUT---"
    )


def _apply_guardrails(
    candidate: OllamaAnalysisResult,
    source_text: str,
    evidence_pool: list[EvidenceSentence] | None = None,
    blocks: list[TextBlock] | None = None,
    debug_trace: dict[str, Any] | None = None,
) -> OllamaAnalysisResult:
    guarded_claims: list[OllamaClaim] = []
    for claim in candidate.claims:
        claim_start = max(0, min(claim.start, len(source_text)))
        claim_end = max(0, min(claim.end, len(source_text)))
        if claim_end <= claim_start:
            claim_start = 0
            claim_end = min(len(source_text), len(claim.text))

        valid_claim_evidence = _sanitize_evidence_list(claim.evidence, source_text)
        if valid_claim_evidence and not _evidence_matches_claim_text(claim.text, valid_claim_evidence):
            valid_claim_evidence = []

        linked_evidence: list[EvidenceSpan] = []
        candidate_scores: list[tuple[int, float]] = []
        if evidence_pool:
            claim_block = (
                _block_for_span(blocks, claim_start, claim_end)
                if blocks
                else None
            )
            allowed_classes = _allowed_evidence_classes_for_claim(claim, claim_block)
            linked_evidence, candidate_scores = _link_claim_to_evidence_with_scores(
                claim,
                evidence_pool,
                allowed_classes=allowed_classes,
            )
            if not linked_evidence:
                _write_claim_evidence_debug(debug_trace, claim.id, claim.text, candidate_scores)
            if linked_evidence:
                valid_claim_evidence = linked_evidence
        score_label = claim.score_label
        score_reason = claim.score_reason.strip()
        if not valid_claim_evidence:
            fallback_claim_evidence = _derive_claim_evidence_fallback(
                source_text=source_text,
                claim_start=claim_start,
                claim_end=claim_end,
                claim_text=claim.text,
            )
            if fallback_claim_evidence:
                valid_claim_evidence = fallback_claim_evidence
            if not valid_claim_evidence:
                span = _normalize_span(claim_start, claim_end, len(source_text))
                if span is not None and not _is_heading_span(source_text, span[0], span[1]):
                    s, e = span
                    quote = source_text[s:e]
                    if quote.strip():
                        valid_claim_evidence = [EvidenceSpan(start=s, end=e, quote=quote)]
            score_label = ScoreLabel.SPECULATIVE
            reason_note = "Evidence missing or invalid; forced to SPECULATIVE."
            if valid_claim_evidence:
                reason_note = (
                    "Model evidence was missing or invalid; deterministic fallback span applied. "
                    "Forced to SPECULATIVE."
                )
            score_reason = f"{score_reason} {reason_note}".strip() if score_reason else reason_note
        elif _requires_speculative_due_to_strong_attribution(claim.text, valid_claim_evidence):
            score_label = ScoreLabel.SPECULATIVE
            reason_note = (
                "Strong attribution (APT/TTP/CVE) is not directly grounded in matching evidence quote; "
                "forced to SPECULATIVE."
            )
            score_reason = f"{score_reason} {reason_note}".strip() if score_reason else reason_note
        elif score_label == ScoreLabel.SUPPORTED and not _evidence_meets_supported_rule(valid_claim_evidence):
            score_label = ScoreLabel.SPECULATIVE
            reason_note = (
                "SUPPORTED requires numeric/OSINT/technical evidence markers; "
                "claim downgraded to SPECULATIVE."
            )
            score_reason = f"{score_reason} {reason_note}".strip() if score_reason else reason_note
        if _is_heading_like_text(claim.text):
            evidence_heading_only = not valid_claim_evidence or all(
                _is_heading_span(source_text, ev.start, ev.end) or _is_heading_like_text(ev.quote)
                for ev in valid_claim_evidence
            )
            if evidence_heading_only:
                continue

        guarded_claims.append(
            OllamaClaim(
                id=claim.id.strip() or "C001",
                type=claim.type.strip() or "statement",
                text=claim.text.strip() or source_text[claim_start:claim_end] or source_text.strip() or "N/A",
                start=claim_start,
                end=claim_end,
                assertiveness=claim.assertiveness.strip() or "medium",
                score_label=score_label,
                score_reason=score_reason or "No reason provided.",
                evidence=valid_claim_evidence,
            )
        )

    guarded_bias_flags: list[OllamaBiasFlag] = []
    for flag in candidate.bias_flags:
        valid_flag_evidence = _sanitize_evidence_list(flag.evidence, source_text)
        if not valid_flag_evidence:
            continue
        why = _sanitize_reason_text(flag.why, "Potentially biased wording.")
        suggested_fix = _sanitize_suggested_fix(
            bias_type=flag.bias_type,
            why=why,
            suggested_fix=flag.suggested_fix,
        )
        guarded_bias_flags.append(
            OllamaBiasFlag(
                bias_type=flag.bias_type.strip() or "unspecified",
                evidence=valid_flag_evidence,
                why=why,
                suggested_fix=suggested_fix,
            )
        )

    return OllamaAnalysisResult(
        claims=guarded_claims,
        bias_flags=guarded_bias_flags,
        annotated_md=candidate.annotated_md,
        rewrite_md=candidate.rewrite_md,
    )


def _evidence_matches_claim_text(claim_text: str, evidence: list[EvidenceSpan]) -> bool:
    claim_norm = _normalize_for_overlap(claim_text)
    if not claim_norm:
        return False
    claim_tokens = {token for token in claim_norm.split() if len(token) > 3}
    if not claim_tokens:
        return False
    for span in evidence:
        quote_norm = _normalize_for_overlap(span.quote)
        if not quote_norm:
            continue
        quote_tokens = {token for token in quote_norm.split() if len(token) > 3}
        if len(claim_tokens.intersection(quote_tokens)) >= 2:
            return True
    return False


def _normalize_for_overlap(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = " ".join(lowered.split())
    return lowered


def normalize_text(text: str) -> str:
    normalized = str(text or "").lower().strip()
    if not normalized:
        return ""
    normalized = re.sub(r"[*`]+", "", normalized)
    normalized = re.sub(r'^[\'"`]+|[\'"`]+$', "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"[.,;:]+$", "", normalized)
    return normalized.strip()


def _looks_truncated_fragment(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return True
    if len(candidate) < MIN_INDEPENDENT_EVIDENCE_CHARS:
        return True
    lowered = candidate.lower()
    return any(re.search(pattern, lowered) is not None for pattern in TRUNCATED_PREFIX_PATTERNS)


def _span_looks_cut(source_text: str, start: int, end: int) -> bool:
    text_len = len(source_text)
    s = max(0, min(text_len, int(start)))
    e = max(0, min(text_len, int(end)))
    if e <= s:
        return True
    if s > 0 and s < text_len and source_text[s - 1].isalnum() and source_text[s].isalnum():
        return True
    if e < text_len and e > 0 and source_text[e - 1].isalnum() and source_text[e].isalnum():
        return True
    return False


def _resolve_evidence_sentence(
    span: EvidenceSpan,
    sentence_by_id: dict[int, SentenceSpan],
    sentence_spans: list[SentenceSpan],
) -> SentenceSpan | None:
    if isinstance(span.sentence_id, int) and span.sentence_id >= 0:
        sentence = sentence_by_id.get(int(span.sentence_id))
        if sentence is not None:
            return sentence
    return _best_sentence_for_span(sentence_spans, int(span.start), int(span.end))


def _coerce_evidence_to_sentences(
    evidence: list[EvidenceSpan],
    source_text: str,
    sentence_spans: list[SentenceSpan],
    sentence_by_id: dict[int, SentenceSpan],
) -> list[EvidenceSpan]:
    sanitized = _sanitize_evidence_list(evidence, source_text)
    coerced: list[EvidenceSpan] = []
    seen_sentence_ids: set[int] = set()
    for span in sanitized:
        sentence = _resolve_evidence_sentence(span, sentence_by_id, sentence_spans)
        if sentence is None:
            continue
        sid = int(sentence.sentence_id)
        if sid in seen_sentence_ids:
            continue
        seen_sentence_ids.add(sid)
        coerced.append(
            EvidenceSpan(
                start=int(sentence.start),
                end=int(sentence.end),
                quote=sentence.text,
                sentence_id=sid,
            )
        )
    return coerced


def _requires_speculative_due_to_strong_attribution(text: str, evidence: list[EvidenceSpan]) -> bool:
    claim_text = str(text or "")
    strong_tokens = re.findall(r"\b(?:APT\d{1,4}|TTP|CVE-\d{4}-\d{4,7}|MITRE ATT&CK)\b", claim_text, flags=re.IGNORECASE)
    if not strong_tokens:
        return False
    quotes = " ".join(span.quote for span in evidence)
    for token in strong_tokens:
        if re.search(re.escape(token), quotes, flags=re.IGNORECASE):
            return False
    return True


def _evidence_meets_supported_rule(evidence: list[EvidenceSpan]) -> bool:
    if not evidence:
        return False
    merged = " ".join(ev.quote for ev in evidence).lower()
    if not merged.strip():
        return False

    has_numeric_or_table = bool(
        re.search(r"\b\d+(?:\.\d+)?%?\b", merged)
        or re.search(r"\b\d+\s*-\s*\d+\b", merged)
        or any(marker in merged for marker in ("firmware", "endpoints", "endpoint", "table", "range", "count"))
    )
    has_osint_marker = bool(
        "passive osint enumeration identified" in merged
        or "shodan" in merged
        or "censys" in merged
        or re.search(r"deduplicated\s+ip\s*\+\s*port", merged)
    )
    has_technical_artifact = bool(
        "rtsp" in merged
        or "cve-" in merged
        or "hardcoded cred" in merged
        or "hard-coded cred" in merged
        or "auth bypass" in merged
        or "authentication bypass" in merged
    )
    return has_numeric_or_table or has_osint_marker or has_technical_artifact


def _derive_claim_evidence_fallback(
    source_text: str,
    claim_start: int,
    claim_end: int,
    claim_text: str,
) -> list[EvidenceSpan]:
    fallback = _find_span_in_text(source_text, claim_text, anchor=claim_start)
    if fallback is not None:
        s, e = fallback
        quote = source_text[s:e]
        if quote.strip() and not _is_heading_span(source_text, s, e):
            evidence = [EvidenceSpan(start=s, end=e, quote=quote)]
            if _evidence_matches_claim_text(claim_text, evidence):
                return evidence

    text_len = len(source_text)
    start = max(0, min(text_len, int(claim_start)))
    end = max(0, min(text_len, int(claim_end)))
    if end > start:
        span = _normalize_span(start, end, text_len)
        if span is not None:
            s, e = span
            quote = source_text[s:e].strip()
            if quote and not _is_heading_span(source_text, s, e):
                adjusted_start = source_text.find(quote, s, e + 1)
                if adjusted_start != -1:
                    s = adjusted_start
                    e = adjusted_start + len(quote)
                evidence = [EvidenceSpan(start=s, end=e, quote=source_text[s:e])]
                if _evidence_matches_claim_text(claim_text, evidence):
                    return evidence
    return []


def _sanitize_evidence_list(evidence: list[EvidenceSpan], source_text: str) -> list[EvidenceSpan]:
    sanitized: list[EvidenceSpan] = []
    text_len = len(source_text)
    sentence_spans = _split_sentences_with_offsets(source_text) if source_text else []

    for span in evidence:
        start = max(0, int(span.start))
        end = max(0, int(span.end))
        if start >= end or start >= text_len:
            continue
        end = min(end, text_len)
        start, end = _expand_evidence_span(source_text, start, end)
        quote = source_text[start:end]
        if not quote:
            continue
        if _is_heading_span(source_text, start, end) or _is_heading_like_text(quote):
            continue
        sentence = _best_sentence_for_span(sentence_spans, start, end)
        if sentence is not None:
            start = sentence.start
            end = sentence.end
            quote = sentence.text
            sentence_id = sentence.sentence_id
        else:
            sentence_id = span.sentence_id if isinstance(span.sentence_id, int) and span.sentence_id >= 0 else None
        sanitized.append(EvidenceSpan(start=start, end=end, quote=quote, sentence_id=sentence_id))

    return sanitized


def _best_sentence_for_span(sentences: list[SentenceSpan], start: int, end: int) -> SentenceSpan | None:
    best: SentenceSpan | None = None
    best_overlap = 0
    for sentence in sentences:
        overlap = max(0, min(end, sentence.end) - max(start, sentence.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best = sentence
    return best


def _expand_evidence_span(source_text: str, start: int, end: int) -> tuple[int, int]:
    text_len = len(source_text)
    s = max(0, min(text_len, int(start)))
    e = max(0, min(text_len, int(end)))
    if e <= s:
        return s, e

    # Avoid clipped words at span edges: expand to nearest whitespace boundary.
    while s > 0 and not source_text[s - 1].isspace():
        s -= 1
    while e < text_len and not source_text[e:e + 1].isspace():
        e += 1

    while s < e and source_text[s].isspace():
        s += 1
    while e > s and source_text[e - 1].isspace():
        e -= 1

    return s, e


def _seek_word_boundary_left(text: str, start: int, max_steps: int) -> int:
    idx = start
    steps = 0
    while idx > 0 and steps < max_steps:
        if text[idx - 1].isspace() or text[idx - 1] in {".", ",", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}"}:
            break
        idx -= 1
        steps += 1
    return idx


def _seek_word_boundary_right(text: str, end: int, max_steps: int) -> int:
    idx = end
    text_len = len(text)
    steps = 0
    while idx < text_len and steps < max_steps:
        if text[idx].isspace() or text[idx] in {".", ",", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}"}:
            break
        idx += 1
        steps += 1
    return idx


def _sanitize_suggested_fix(bias_type: str, why: str, suggested_fix: str) -> str:
    why_clean = _normalize_human_text(why)
    fix_clean = _normalize_human_text(suggested_fix)

    why_clean = re.sub(r"\[\s*\d+\s*:\s*\d+\s*\]", "", why_clean)
    fix_clean = re.sub(r"\[\s*\d+\s*:\s*\d+\s*\]", "", fix_clean)
    why_clean = re.sub(r"\s+", " ", why_clean).strip(" -:;,")
    fix_clean = re.sub(r"\s+", " ", fix_clean).strip(" -:;,")

    parts: list[str] = []
    if why_clean:
        parts.append(_ensure_sentence(why_clean))
    if fix_clean and fix_clean.lower() not in why_clean.lower():
        parts.append(_ensure_sentence(fix_clean))

    if not parts:
        parts.append(_default_fix_for_bias_type(bias_type))

    merged = " ".join(parts).strip()
    if len(merged) > MAX_SUGGESTED_FIX_CHARS:
        merged = merged[:MAX_SUGGESTED_FIX_CHARS].rsplit(" ", 1)[0].strip()
    return merged or _default_fix_for_bias_type(bias_type)


def _sanitize_reason_text(text: str, fallback: str) -> str:
    clean = _normalize_human_text(text)
    clean = re.sub(r"\[\s*\d+\s*:\s*\d+\s*\]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip(" -:;,")
    if not clean:
        return fallback
    return _ensure_sentence(clean)


def _structured_output_schema(generate_rewrite: bool) -> dict[str, Any]:
    evidence_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "start": {"type": "integer", "minimum": 0},
            "end": {"type": "integer", "minimum": 0},
            "quote": {"type": "string", "minLength": 1},
        },
        "required": ["start", "end", "quote"],
    }
    claim_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "type": {"type": "string", "minLength": 1},
            "text": {"type": "string", "minLength": 1},
            "start": {"type": "integer", "minimum": 0},
            "end": {"type": "integer", "minimum": 0},
            "assertiveness": {"type": "string", "minLength": 1},
            "score_label": {"type": "string", "enum": [item.value for item in ScoreLabel]},
            "score_reason": {"type": "string", "minLength": 1},
            "evidence": {"type": "array", "items": evidence_schema, "maxItems": 2},
        },
        "required": [
            "id",
            "type",
            "text",
            "start",
            "end",
            "assertiveness",
            "score_label",
            "score_reason",
            "evidence",
        ],
    }
    bias_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "bias_type": {"type": "string", "minLength": 1},
            "evidence": {"type": "array", "items": evidence_schema, "minItems": 1, "maxItems": 2},
            "why": {"type": "string", "minLength": 1},
            "suggested_fix": {"type": "string", "minLength": 1},
        },
        "required": ["bias_type", "evidence", "why", "suggested_fix"],
    }
    properties: dict[str, Any] = {
        "claims": {"type": "array", "items": claim_schema, "maxItems": 6},
        "bias_flags": {"type": "array", "items": bias_schema, "maxItems": _max_bias_flags()},
    }
    required = ["claims", "bias_flags"]
    if generate_rewrite:
        properties["rewrite_md"] = {"type": "string"}
        required.append("rewrite_md")

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _normalize_human_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _ensure_sentence(text: str) -> str:
    if not text:
        return ""
    if text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def _default_fix_for_bias_type(bias_type: str) -> str:
    key = str(bias_type or "").strip().lower()
    if key in {"certainty_overclaim", "speculation_as_fact"} or "certainty" in key:
        return "Add confidence qualifiers and clearly distinguish confirmed facts from assumptions."
    if key in {"fear_appeal", "loaded_language"} or "alarm" in key or "hype" in key:
        return "Use neutral, evidence-grounded wording and remove sensational language."
    if key in {"stereotyping", "regional_generalization", "cultural_essentialism"}:
        return "Replace group-level generalizations with evidence tied to observed actors and concrete events."
    if key in {"single_cause_fallacy", "attribution_without_evidence", "source_imbalance"}:
        return "State alternative explanations and cite corroborating sources before assigning causality."
    if key in {"normative_judgment", "prescriptive_overreach"}:
        return "Separate objective observations from recommendations and justify actions with explicit evidence."
    if "formal" in key:
        return "Simplify phrasing and anchor each statement to explicit evidence."
    return "Rewrite using neutral language anchored to explicit evidence spans."


def _canonicalize_analysis_result(result: OllamaAnalysisResult) -> OllamaAnalysisResult:
    canonical_claims: list[OllamaClaim] = []
    for claim in result.claims:
        dedup_evidence = _dedupe_and_sort_evidence(claim.evidence)
        canonical_claims.append(
            OllamaClaim(
                id=str(claim.id).strip() or "claim",
                type=" ".join(str(claim.type or "statement").split()) or "statement",
                text=" ".join(str(claim.text or "").split()) or "No statement available.",
                start=int(claim.start),
                end=int(claim.end),
                assertiveness=" ".join(str(claim.assertiveness or "medium").split()) or "medium",
                score_label=claim.score_label,
                score_reason=" ".join(str(claim.score_reason or "").split())
                or "Evidence-based confidence score.",
                evidence=dedup_evidence,
            )
        )

    canonical_claims.sort(
        key=lambda c: (
            int(c.start),
            int(c.end),
            str(c.type).lower(),
            str(c.text).lower(),
            str(c.id).lower(),
        )
    )

    canonical_bias_flags: list[OllamaBiasFlag] = []
    bias_counter = 1
    for flag in result.bias_flags:
        dedup_evidence = _dedupe_and_sort_evidence(flag.evidence)
        bias_code = getattr(flag, "bias_code", None)
        if not bias_code:
            bias_code = f"B{bias_counter:03d}"
            bias_counter += 1
        canonical_bias_flags.append(
            OllamaBiasFlag(
                bias_type=" ".join(str(flag.bias_type or "unspecified").split()) or "unspecified",
                evidence=dedup_evidence,
                why=" ".join(str(flag.why or "").split()) or "Potentially biased wording.",
                suggested_fix=" ".join(str(flag.suggested_fix or "").split())
                or "Rewrite using neutral language.",
                bias_code=bias_code,
            )
        )

    canonical_bias_flags.sort(
        key=lambda f: (
            str(f.bias_type).lower(),
            f.evidence[0].start if f.evidence else 0,
            f.evidence[0].end if f.evidence else 0,
            str(f.why).lower(),
        )
    )
    max_bias_flags = _max_bias_flags()
    if len(canonical_bias_flags) > max_bias_flags:
        canonical_bias_flags = canonical_bias_flags[:max_bias_flags]

    return OllamaAnalysisResult(
        claims=canonical_claims,
        bias_flags=canonical_bias_flags,
        annotated_md=result.annotated_md,
        rewrite_md=result.rewrite_md,
    )


def _dedupe_and_sort_evidence(evidence: list[EvidenceSpan]) -> list[EvidenceSpan]:
    unique: dict[tuple[int, int, str, int | None], EvidenceSpan] = {}
    for span in evidence:
        sid = span.sentence_id if isinstance(span.sentence_id, int) and span.sentence_id >= 0 else None
        key = (int(span.start), int(span.end), str(span.quote), sid)
        if key not in unique:
            unique[key] = EvidenceSpan(start=key[0], end=key[1], quote=key[2], sentence_id=sid)
    ordered = sorted(
        unique.values(),
        key=lambda ev: (int(ev.start), int(ev.end), str(ev.quote), ev.sentence_id if ev.sentence_id is not None else -1),
    )
    return ordered


def _claim_candidate_scores_for_output(
    claim: OllamaClaim,
    evidence_pool: list[EvidenceSentence],
    claim_block: TextBlock | None,
) -> list[tuple[int, float]]:
    if not evidence_pool:
        return []
    allowed_classes = _allowed_evidence_classes_for_claim(claim, claim_block)
    _linked, scores = _link_claim_to_evidence_with_scores(
        claim,
        evidence_pool,
        max_items=5,
        allowed_classes=allowed_classes,
    )
    return scores[:5]


def _claim_output_evidence_payload(
    claim: OllamaClaim,
    claim_id: str,
    source_text: str,
    sentence_spans: list[SentenceSpan],
    sentence_by_id: dict[int, SentenceSpan],
    evidence_pool: list[EvidenceSentence],
    claim_block: TextBlock | None,
    debug_trace: dict[str, Any] | None,
) -> tuple[list[EvidenceSpan], list[int], list[str], str, str | None]:
    normalized_claim = normalize_text(claim.text)
    anchored_evidence: list[EvidenceSpan] = []
    evidence_sentence_ids: list[int] = []
    evidence_texts: list[str] = []
    seen_sentence_ids: set[int] = set()

    for raw_span in claim.evidence:
        sentence_for_log = _resolve_evidence_sentence(raw_span, sentence_by_id, sentence_spans)
        sentence_id_for_log = (
            int(sentence_for_log.sentence_id)
            if sentence_for_log is not None
            else (int(raw_span.sentence_id) if isinstance(raw_span.sentence_id, int) else -1)
        )
        if _looks_truncated_fragment(raw_span.quote) or _span_looks_cut(source_text, raw_span.start, raw_span.end):
            logger.debug(
                "claim_evidence_discard claim_id=%s sentence_id=%s reason=truncated",
                claim_id,
                sentence_id_for_log,
            )
            continue

        resolved_spans = _coerce_evidence_to_sentences([raw_span], source_text, sentence_spans, sentence_by_id)
        if not resolved_spans:
            continue
        resolved = resolved_spans[0]
        sid = int(resolved.sentence_id) if isinstance(resolved.sentence_id, int) else -1
        if sid < 0 or sid in seen_sentence_ids:
            continue

        normalized_evidence = normalize_text(resolved.quote)
        if normalized_evidence and normalized_evidence == normalized_claim:
            logger.debug(
                "claim_evidence_discard claim_id=%s sentence_id=%s reason=self_citation",
                claim_id,
                sid,
            )
            continue

        seen_sentence_ids.add(sid)
        anchored_evidence.append(resolved)
        evidence_sentence_ids.append(sid)
        evidence_texts.append(resolved.quote)

    if anchored_evidence:
        return anchored_evidence, evidence_sentence_ids, evidence_texts, "anchored", None

    candidate_scores = _claim_candidate_scores_for_output(claim, evidence_pool, claim_block)
    _write_claim_missing_debug(
        debug_trace=debug_trace,
        claim_id=claim_id,
        claim_text=claim.text,
        section_class=claim_block.section_class if claim_block is not None else "OTHER",
        candidate_scores=candidate_scores,
    )
    return [], [], [], "missing", NO_INDEPENDENT_EVIDENCE_NOTE


def _to_api_response(
    result: OllamaAnalysisResult,
    source_text: str,
    blocks: list[TextBlock] | None = None,
    debug_trace: dict[str, Any] | None = None,
) -> AnalyzeResponse:
    claim_bias_flags = _map_bias_flags_to_claims(result.claims, result.bias_flags)
    source_blocks = blocks if blocks else _parse_text_blocks(source_text)
    sentence_spans = _split_sentences_with_offsets(source_text, source_blocks)
    sentence_by_id = {sentence.sentence_id: sentence for sentence in sentence_spans}
    evidence_pool = _build_evidence_pool(sentence_spans)
    claims: list[Claim] = []

    for index, claim in enumerate(result.claims, start=1):
        claim_id = f"C{index:03d}"
        mapped_flags = claim_bias_flags.get(claim.id, [])
        claim_block = _block_for_span(source_blocks, int(claim.start), int(claim.end))
        evidence, evidence_sentence_ids, evidence_texts, evidence_status, evidence_note = _claim_output_evidence_payload(
            claim=claim,
            claim_id=claim_id,
            source_text=source_text,
            sentence_spans=sentence_spans,
            sentence_by_id=sentence_by_id,
            evidence_pool=evidence_pool,
            claim_block=claim_block,
            debug_trace=debug_trace,
        )
        claims.append(
            Claim(
                claim_id=claim_id,
                text=claim.text,
                category=claim.type,
                score_label=claim.score_label,
                evidence=evidence,
                evidence_sentence_ids=evidence_sentence_ids,
                evidence_texts=evidence_texts,
                evidence_status=evidence_status,
                evidence_note=evidence_note,
                bias_flags=mapped_flags,
            )
        )

    return AnalyzeResponse(
        claims=ClaimsDocument(engine="ollama", claims=claims),
        annotated_md=result.annotated_md,
        rewrite_md=result.rewrite_md,
    )


def _map_bias_flags_to_claims(
    claims: list[OllamaClaim], bias_flags: list[OllamaBiasFlag]
) -> dict[str, list[BiasFlag]]:
    mapped: dict[str, list[BiasFlag]] = {claim.id: [] for claim in claims}
    bias_counter = 1
    for flag in bias_flags:
        linked_ids = _best_claims_for_evidence(claims, flag.evidence)
        bias_code = getattr(flag, "bias_code", None)
        if not bias_code:
            bias_code = f"B{bias_counter:03d}"
            bias_counter += 1

        converted = BiasFlag(
            tag=flag.bias_type,
            suggested_fix=flag.suggested_fix,
            evidence=flag.evidence,
            bias_code=bias_code,
        )
        for claim_id in linked_ids:
            mapped[claim_id].append(converted)
    return mapped


def _spans_overlap_claim(claim: OllamaClaim, evidence: list[EvidenceSpan]) -> bool:
    for ev in evidence:
        if ev.start < claim.end and ev.end > claim.start:
            return True
    return False


def _nearest_claim_for_evidence(claims: list[OllamaClaim], evidence: list[EvidenceSpan]) -> OllamaClaim | None:
    if not claims or not evidence:
        return None
    starts = [int(ev.start) for ev in evidence if int(ev.end) > int(ev.start)]
    ends = [int(ev.end) for ev in evidence if int(ev.end) > int(ev.start)]
    if not starts or not ends:
        return claims[0]
    ev_center = (min(starts) + max(ends)) // 2

    def distance(claim: OllamaClaim) -> int:
        c_start = int(claim.start)
        c_end = int(claim.end)
        if c_start <= ev_center <= c_end:
            return 0
        if ev_center < c_start:
            return c_start - ev_center
        return ev_center - c_end

    return min(claims, key=distance)


def _best_claims_for_evidence(claims: list[OllamaClaim], evidence: list[EvidenceSpan]) -> list[str]:
    if not claims or not evidence:
        return []
    # Prefer maximal overlap; if none, fallback to nearest center.
    overlaps: list[tuple[int, int, str]] = []
    for claim in claims:
        overlap_len = 0
        for ev in evidence:
            start = max(int(ev.start), int(claim.start))
            end = min(int(ev.end), int(claim.end))
            if end > start:
                overlap_len = max(overlap_len, end - start)
        overlaps.append((overlap_len, int(claim.start), claim.id))
    max_overlap = max(item[0] for item in overlaps)
    if max_overlap > 0:
        winners = [item for item in overlaps if item[0] == max_overlap]
        winners.sort(key=lambda t: (t[1], t[2]))
        return [w[2] for w in winners[:1]]

    nearest = _nearest_claim_for_evidence(claims, evidence)
    return [nearest.id] if nearest is not None else []


def _ollama_timeout_seconds() -> float | None:
    raw = os.getenv("OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_OLLAMA_TIMEOUT_SECONDS)).strip()
    try:
        parsed = float(raw)
    except ValueError:
        return DEFAULT_OLLAMA_TIMEOUT_SECONDS
    if parsed <= 0:
        return None
    return parsed


def _ollama_warmup_timeout_seconds() -> float | None:
    raw = os.getenv("OLLAMA_WARMUP_TIMEOUT_SECONDS", str(DEFAULT_OLLAMA_WARMUP_TIMEOUT_SECONDS)).strip()
    try:
        parsed = float(raw)
    except ValueError:
        return DEFAULT_OLLAMA_WARMUP_TIMEOUT_SECONDS
    if parsed <= 0:
        return None
    return parsed


def _max_chars_per_chunk() -> int:
    raw = os.getenv("OLLAMA_MAX_CHARS_PER_CHUNK", str(DEFAULT_MAX_CHARS_PER_CHUNK)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_MAX_CHARS_PER_CHUNK
    if parsed < 2000:
        return 2000
    if _gpu_chunk_guard_enabled():
        parsed = min(parsed, _gpu_safe_max_chars_per_chunk())
    return parsed


def _chunk_overlap_chars() -> int:
    raw = os.getenv("OLLAMA_CHUNK_OVERLAP_CHARS", str(DEFAULT_CHUNK_OVERLAP_CHARS)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_CHUNK_OVERLAP_CHARS
    if parsed < 0:
        return 0
    return parsed


def _max_attempts() -> int:
    return 2


def _ollama_seed() -> int | None:
    raw = os.getenv("OLLAMA_SEED", str(DEFAULT_OLLAMA_SEED)).strip()
    if raw.lower() in {"none", "null", "off", "disable", "disabled"}:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_OLLAMA_SEED
    if parsed < 0:
        return 0
    if parsed > 2_147_483_647:
        return 2_147_483_647
    return parsed


def _deterministic_cache_enabled() -> bool:
    raw = os.getenv("OLLAMA_DETERMINISTIC_CACHE", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _strict_determinism_enabled() -> bool:
    raw = os.getenv("OLLAMA_STRICT_DETERMINISM", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _strict_determinism_attempts() -> int:
    raw = os.getenv("OLLAMA_STRICT_DETERMINISM_ATTEMPTS", str(DEFAULT_STRICT_DETERMINISM_ATTEMPTS)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_STRICT_DETERMINISM_ATTEMPTS
    if parsed < 2:
        return 2
    if parsed > 3:
        return 3
    return parsed


def _deterministic_cache_max_entries() -> int:
    raw = os.getenv(
        "OLLAMA_DETERMINISTIC_CACHE_MAX_ENTRIES",
        str(DEFAULT_DETERMINISTIC_CACHE_MAX_ENTRIES),
    ).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_DETERMINISTIC_CACHE_MAX_ENTRIES
    if parsed < 16:
        return 16
    if parsed > 5000:
        return 5000
    return parsed


def _deterministic_cache_key(request: AnalyzeRequest) -> str:
    normalized_text = _normalize_text_for_cache(request.text)
    payload = {
        "version": DETERMINISTIC_CACHE_VERSION,
        "text": normalized_text,
        "generate_rewrite": request.generate_rewrite,
        "model": _ollama_model_name(),
        "seed": _ollama_seed(),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _deterministic_content_key(request: AnalyzeRequest) -> str:
    normalized_text = _normalize_text_for_cache(request.text)
    payload = {
        "version": DETERMINISTIC_CACHE_VERSION,
        "text": normalized_text,
        "generate_rewrite": request.generate_rewrite,
        "model": _ollama_model_name(),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_text_for_cache(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    # Collapse runs of spaces/tabs for cache key stability only (offsets remain tied to raw text elsewhere).
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized.strip()


def _start_debug_trace(
    input_hash: str,
    model: str,
    generate_rewrite: bool,
    options: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": str(time.time_ns()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_hash": input_hash,
        "model": model,
        "model_digest": _ollama_model_digest(model),
        "generate_rewrite": generate_rewrite,
        "options": options,
        "batches": [],
        "summary": {},
    }


def _record_debug_batch(
    debug_trace: dict[str, Any],
    phase: str,
    chunk_index: int,
    chunk_count: int,
    batch_index: int,
    batch_count: int,
    sentence_ids: list[int],
    request_summary: dict[str, Any],
    raw_response: dict[str, Any],
    attempt: int,
    error: str | None,
) -> None:
    batches = debug_trace.setdefault("batches", [])
    batches.append(
        {
            "phase": phase,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "batch_index": batch_index,
            "batch_count": batch_count,
            "attempt": attempt,
            "sentence_ids": sentence_ids,
            "request_summary": request_summary,
            "error": error,
            "raw_response": raw_response,
        }
    )


def _write_debug_trace(debug_trace: dict[str, Any]) -> None:
    if not VERBOSE_LOGGING:
        return
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        run_id = str(debug_trace.get("run_id") or time.time_ns())
        target = DEBUG_DIR / f"bias_raw_{run_id}.json"
        target.write_text(json.dumps(debug_trace, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("ollama_debug write failed: %s", exc)


def _write_claim_evidence_debug(
    debug_trace: dict[str, Any] | None, claim_id: str, claim_text: str, scores: list[tuple[int, float]]
) -> None:
    if not VERBOSE_LOGGING:
        return
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        run_id = str((debug_trace or {}).get("run_id") or time.time_ns())
        target = DEBUG_DIR / "claim_evidence_debug.json"
        payload = []
        if target.exists():
            try:
                payload = json.loads(target.read_text(encoding="utf-8"))
            except Exception:
                payload = []
        payload.append(
            {
                "run_id": run_id,
                "claim_id": claim_id,
                "claim_text": claim_text,
                "scores": [{"sentence_id": sid, "score": score} for sid, score in scores[:5]],
            }
        )
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("claim_evidence_debug write failed: %s", exc)


def _write_claim_missing_debug(
    debug_trace: dict[str, Any] | None,
    claim_id: str,
    claim_text: str,
    section_class: str,
    candidate_scores: list[tuple[int, float]],
) -> None:
    if not VERBOSE_LOGGING:
        return
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        run_id = str((debug_trace or {}).get("run_id") or time.time_ns())
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        target = DEBUG_DIR / f"claim_missing_{timestamp}_{time.time_ns()}.json"
        payload = {
            "run_id": run_id,
            "claim_id": claim_id,
            "claim_text": claim_text,
            "section_class": section_class,
            "candidate_sentence_ids": [
                {"sentence_id": int(sentence_id), "score": float(score)} for sentence_id, score in candidate_scores[:5]
            ],
            "note": NO_INDEPENDENT_EVIDENCE_NOTE,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("claim_missing_debug write failed: %s", exc)


def _ollama_options_snapshot() -> dict[str, Any]:
    return {
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "min_p": 0,
        "seed": _ollama_seed(),
        "num_predict": _ollama_num_predict(),
        "bias_num_predict": _ollama_bias_num_predict(),
        "num_ctx": _ollama_num_ctx(),
        "num_batch": _ollama_num_batch(),
        "num_gpu": _ollama_num_gpu_override(),
        "max_bias_flags": _max_bias_flags(),
        "keep_alive": -1,
        "format": "json_schema",
        "retry_attempts": _max_attempts(),
        "bias_sentences_per_batch": _bias_sentences_per_batch(),
    }


def _assert_options_stable(baseline: dict[str, Any], context: str) -> None:
    current = _ollama_options_snapshot()
    if current != baseline:
        raise ValueError(f"Option drift detected during {context}: {baseline} != {current}")


def _ollama_model_digest(model: str) -> str:
    cached = _MODEL_DIGEST_CACHE.get(model)
    if cached:
        return cached
    url = _ollama_url("/api/tags")
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            payload = resp.json()
    except Exception:
        _MODEL_DIGEST_CACHE[model] = "unknown"
        return "unknown"

    models = payload.get("models", []) if isinstance(payload, dict) else []
    digest = "unknown"
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name in {model, f"{model}:latest"} or model in {name, name.replace(":latest", "")}:
            digest = str(item.get("digest", "")).strip() or "unknown"
            break
    _MODEL_DIGEST_CACHE[model] = digest
    return digest


def _load_cache_entries() -> dict[str, Any]:
    try:
        if not CACHE_FILE.exists():
            return {}
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            return {}
        return entries
    except Exception as exc:
        logger.warning("ollama_cache read failed: %s", exc)
        return {}


def _read_cached_response(cache_key: str) -> AnalyzeResponse | None:
    entries = _load_cache_entries()
    payload = entries.get(cache_key)
    if not isinstance(payload, dict):
        return None
    try:
        if "response" in payload and isinstance(payload["response"], dict):
            return AnalyzeResponse.model_validate(payload["response"])
        return AnalyzeResponse.model_validate(payload)
    except ValidationError:
        return None


def _read_latest_cached_response_by_content_key(content_key: str) -> AnalyzeResponse | None:
    entries = _load_cache_entries()
    best_payload: dict[str, Any] | None = None
    best_created_at = -1
    for value in entries.values():
        if not isinstance(value, dict):
            continue
        wrapped_response = value.get("response") if isinstance(value.get("response"), dict) else None
        wrapped_content_key = str(value.get("content_key", ""))
        created_at = int(value.get("created_at_ns", 0) or 0)
        if wrapped_response is None:
            continue
        if wrapped_content_key != content_key:
            continue
        if created_at > best_created_at:
            best_created_at = created_at
            best_payload = wrapped_response
    if best_payload is None:
        return None
    try:
        return AnalyzeResponse.model_validate(best_payload)
    except ValidationError:
        return None


def _response_fingerprint(response: AnalyzeResponse) -> str:
    payload = {
        "claims": response.claims.model_dump(mode="json"),
        "annotated_md": response.annotated_md,
        "rewrite_md": response.rewrite_md,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _append_warning(response: AnalyzeResponse, message: str) -> AnalyzeResponse:
    current = (response.warning or "").strip()
    if current:
        if message in current:
            return response
        new_warning = f"{current}; {message}"
    else:
        new_warning = message
    return response.model_copy(update={"warning": new_warning})


def _write_cached_response(cache_key: str, response: AnalyzeResponse, content_key: str) -> None:
    try:
        entries = _load_cache_entries()
        entries.pop(cache_key, None)
        entries[cache_key] = {
            "response": response.model_dump(mode="json"),
            "content_key": content_key,
            "created_at_ns": time.time_ns(),
        }

        max_entries = _deterministic_cache_max_entries()
        while len(entries) > max_entries:
            oldest_key = next(iter(entries))
            entries.pop(oldest_key, None)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"version": DETERMINISTIC_CACHE_VERSION, "entries": entries}
        tmp_file = CACHE_FILE.with_suffix(".tmp")
        serialized = json.dumps(payload, ensure_ascii=False)
        tmp_file.write_text(serialized, encoding="utf-8")
        try:
            tmp_file.replace(CACHE_FILE)
        except PermissionError:
            # Some locked-down Windows environments deny atomic replace operations.
            # Fall back to direct write so deterministic-cache behavior still works.
            CACHE_FILE.write_text(serialized, encoding="utf-8")
            try:
                tmp_file.unlink(missing_ok=True)
            except OSError:
                pass
    except Exception as exc:
        logger.warning("ollama_cache write failed: %s", exc)


def _max_ollama_calls(chunk_count: int, text: str = "") -> int:
    raw = os.getenv("OLLAMA_MAX_CALLS", "").strip()
    sentence_count = len(_split_sentences_with_offsets(text)) if text else 0
    sentence_batches = (sentence_count + _bias_sentences_per_batch() - 1) // _bias_sentences_per_batch()
    attempts = _max_attempts()
    required_budget = max(DEFAULT_MAX_CALLS, (chunk_count * attempts) + (sentence_batches * attempts) + 2)
    if raw == "":
        return required_budget
    try:
        parsed = int(raw)
    except ValueError:
        return required_budget
    if parsed < 1:
        parsed = 1
    if parsed > 120:
        parsed = 120
    if parsed < required_budget:
        logger.warning(
            "OLLAMA_MAX_CALLS=%d is below required minimum=%d for this input; using minimum.",
            parsed,
            required_budget,
        )
        return required_budget
    return parsed


def _max_split_depth() -> int:
    raw = os.getenv("OLLAMA_MAX_SPLIT_DEPTH", str(DEFAULT_MAX_SPLIT_DEPTH)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_MAX_SPLIT_DEPTH
    if parsed < 0:
        return 0
    if parsed > 4:
        return 4
    return parsed


def _ollama_num_predict() -> int:
    raw = os.getenv("OLLAMA_NUM_PREDICT", str(DEFAULT_NUM_PREDICT)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_NUM_PREDICT
    if parsed < 800:
        return 800
    if parsed > 1500:
        return 1500
    return parsed


def _ollama_bias_num_predict(batch_size: int | None = None) -> int:
    env_raw = os.getenv("OLLAMA_BIAS_NUM_PREDICT", "").strip()
    if env_raw:
        try:
            parsed = int(env_raw)
        except ValueError:
            parsed = DEFAULT_BIAS_NUM_PREDICT
    else:
        dynamic = 384
        if batch_size is not None:
            dynamic = 256 + (max(1, int(batch_size)) * 6)
        parsed = dynamic
    if parsed < 256:
        return 256
    if parsed > 512:
        return 512
    return parsed


def _bias_sentences_per_batch() -> int:
    raw = os.getenv("OLLAMA_BIAS_SENTENCES_PER_BATCH", str(DEFAULT_BIAS_SENTENCES_PER_BATCH)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_BIAS_SENTENCES_PER_BATCH
    if parsed < 4:
        return 4
    if parsed > 40:
        return 40
    return parsed


def _ollama_num_ctx() -> int:
    raw = os.getenv("OLLAMA_NUM_CTX", str(DEFAULT_NUM_CTX)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_NUM_CTX
    if parsed < 2048:
        return 2048
    if parsed > 8192:
        return 8192
    return parsed


def _ollama_num_batch() -> int:
    raw = os.getenv("OLLAMA_NUM_BATCH", str(DEFAULT_NUM_BATCH)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_NUM_BATCH
    if parsed < 32:
        return 32
    if parsed > 512:
        return 512
    return parsed


def _ollama_num_gpu_override() -> int | None:
    raw = os.getenv("OLLAMA_NUM_GPU", "").strip()
    if raw == "":
        return None
    try:
        parsed = int(raw)
    except ValueError:
        return None
    if parsed < 0:
        return 0
    return parsed


def _gpu_safe_mode_start() -> bool:
    raw = os.getenv("OLLAMA_GPU_SAFE_MODE_START", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _gpu_safe_mode_on_error() -> bool:
    raw = os.getenv("OLLAMA_GPU_SAFE_MODE_ON_ERROR", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _gpu_safe_num_ctx(base_ctx: int) -> int:
    raw = os.getenv("OLLAMA_GPU_SAFE_NUM_CTX", "2048").strip()
    try:
        parsed = int(raw)
    except ValueError:
        parsed = 2048
    parsed = max(1024, min(4096, parsed))
    return min(base_ctx, parsed)


def _gpu_safe_num_batch(base_batch: int) -> int:
    raw = os.getenv("OLLAMA_GPU_SAFE_NUM_BATCH", "64").strip()
    try:
        parsed = int(raw)
    except ValueError:
        parsed = 64
    parsed = max(16, min(256, parsed))
    return min(base_batch, parsed)


def _gpu_safe_num_gpu_layers() -> int:
    raw = os.getenv("OLLAMA_GPU_SAFE_NUM_GPU_LAYERS", "0").strip()
    try:
        parsed = int(raw)
    except ValueError:
        return 0
    if parsed < 0:
        return 0
    return parsed


def _gpu_chunk_guard_enabled() -> bool:
    # Guard applies only for GPU-first execution without CPU fallback.
    return _require_gpu() and not _cpu_fallback_on_gpu_failure() and _gpu_safe_mode_on_error()


def _gpu_safe_max_chars_per_chunk() -> int:
    raw = os.getenv("OLLAMA_GPU_SAFE_MAX_CHARS_PER_CHUNK", str(DEFAULT_GPU_SAFE_MAX_CHARS_PER_CHUNK)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_GPU_SAFE_MAX_CHARS_PER_CHUNK
    if parsed < 2000:
        return 2000
    if parsed > 6000:
        return 6000
    return parsed


def _apply_gpu_safe_profile(payload: dict[str, Any], num_ctx: int, num_batch: int, attempt_index: int) -> None:
    options = payload.setdefault("options", {})
    options["num_ctx"] = _gpu_safe_num_ctx(num_ctx)
    safe_batch = _gpu_safe_num_batch(num_batch)
    if attempt_index >= 2:
        safe_batch = max(16, safe_batch // 2)
    options["num_batch"] = safe_batch
    gpu_layers = _gpu_safe_num_gpu_layers()
    if gpu_layers > 0:
        options["num_gpu"] = gpu_layers


def _cpu_fallback_on_gpu_failure() -> bool:
    raw = os.getenv("OLLAMA_CPU_FALLBACK_ON_GPU_FAILURE", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _max_claims() -> int:
    raw = os.getenv("OLLAMA_MAX_CLAIMS", str(DEFAULT_MAX_CLAIMS)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_MAX_CLAIMS
    if parsed < 3:
        return 3
    if parsed > 20:
        return 20
    return parsed


def _max_bias_flags() -> int:
    raw = os.getenv("OLLAMA_MAX_BIAS_FLAGS", str(DEFAULT_MAX_BIAS_FLAGS)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_MAX_BIAS_FLAGS
    if parsed < 12:
        return 12
    if parsed > 200:
        return 200
    return parsed


def _chunk_text(text: str) -> list[tuple[int, str]]:
    max_chars = _max_chars_per_chunk()
    overlap = min(_chunk_overlap_chars(), max_chars // 3)
    text_len = len(text)

    if text_len <= max_chars:
        return [(0, text)]

    chunks: list[tuple[int, str]] = []
    start = 0
    while start < text_len:
        tentative_end = min(start + max_chars, text_len)
        end = _choose_chunk_boundary(text, start, tentative_end)
        if end <= start:
            end = tentative_end

        chunks.append((start, text[start:end]))
        if end >= text_len:
            break

        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _paragraph_boundaries_between(text: str, start: int, end: int) -> list[int]:
    s = max(0, int(start))
    e = min(len(text), int(end))
    if e <= s:
        return []
    segment = text[s:e]
    boundaries: list[int] = []
    for match in PARAGRAPH_BREAK_PATTERN.finditer(segment):
        boundary = s + match.end()
        if s < boundary < e:
            boundaries.append(boundary)
    return boundaries


def _choose_chunk_boundary(text: str, start: int, tentative_end: int) -> int:
    if tentative_end >= len(text):
        return len(text)

    span = max(1, tentative_end - start)
    minimum_useful = min(700, max(220, span // 6))
    paragraph_boundaries = _paragraph_boundaries_between(text, start + 1, tentative_end)
    if paragraph_boundaries:
        for boundary in reversed(paragraph_boundaries):
            if boundary - start >= minimum_useful:
                return boundary

    window_start = max(start, tentative_end - 900)
    for separator in ["\n", ". ", "! ", "? ", "; "]:
        idx = text.rfind(separator, window_start, tentative_end)
        if idx != -1:
            candidate = idx + len(separator)
            if candidate - start >= max(120, minimum_useful // 2):
                return candidate
    return tentative_end


def _choose_chunk_split_point(text: str) -> int:
    text_len = len(text)
    if text_len <= 1:
        return text_len

    target = text_len // 2
    guard = max(200, min(1200, text_len // 8))
    min_pos = guard
    max_pos = max(min_pos + 1, text_len - guard)

    paragraph_boundaries = _paragraph_boundaries_between(text, min_pos, max_pos)
    if paragraph_boundaries:
        return min(paragraph_boundaries, key=lambda boundary: abs(boundary - target))

    left = _choose_chunk_boundary(text, 0, target)
    if min_pos < left < max_pos:
        return left

    right_window_end = min(text_len, target + max(500, text_len // 6))
    right = _choose_chunk_boundary(text, target, right_window_end)
    if min_pos < right < max_pos:
        return right

    return target


def _shift_result_spans(
    result: OllamaAnalysisResult, chunk_start: int, chunk_index: int, chunk_count: int
) -> OllamaAnalysisResult:
    claims: list[OllamaClaim] = []
    for claim in result.claims:
        shifted_evidence = [
            EvidenceSpan(
                start=ev.start + chunk_start,
                end=ev.end + chunk_start,
                quote=ev.quote,
                sentence_id=ev.sentence_id,
            )
            for ev in claim.evidence
        ]
        claims.append(
            OllamaClaim(
                id=f"{chunk_index}-{claim.id}",
                type=claim.type,
                text=claim.text,
                start=claim.start + chunk_start,
                end=claim.end + chunk_start,
                assertiveness=claim.assertiveness,
                score_label=claim.score_label,
                score_reason=claim.score_reason,
                evidence=shifted_evidence,
            )
        )

    bias_flags: list[OllamaBiasFlag] = []
    for flag in result.bias_flags:
        shifted_evidence = [
            EvidenceSpan(
                start=ev.start + chunk_start,
                end=ev.end + chunk_start,
                quote=ev.quote,
                sentence_id=ev.sentence_id,
            )
            for ev in flag.evidence
        ]
        bias_flags.append(
            OllamaBiasFlag(
                bias_type=flag.bias_type,
                evidence=shifted_evidence,
                why=flag.why,
                suggested_fix=flag.suggested_fix,
            )
        )

    annotated = result.annotated_md
    rewrite = result.rewrite_md
    if chunk_count > 1:
        annotated = f"## Chunk {chunk_index}/{chunk_count}\n\n{annotated}"
        rewrite = f"## Chunk {chunk_index}/{chunk_count}\n\n{rewrite}"

    return OllamaAnalysisResult(
        claims=claims,
        bias_flags=bias_flags,
        annotated_md=annotated,
        rewrite_md=rewrite,
    )


def _merge_chunk_results(results: list[OllamaAnalysisResult]) -> OllamaAnalysisResult:
    claims: list[OllamaClaim] = []
    bias_flags: list[OllamaBiasFlag] = []
    annotated_parts: list[str] = []
    rewrite_parts: list[str] = []

    for result in results:
        claims.extend(result.claims)
        bias_flags.extend(result.bias_flags)
        annotated_parts.append(result.annotated_md.strip())
        rewrite_parts.append(result.rewrite_md.strip())

    max_claims = _max_claims()
    if len(claims) > max_claims:
        claims = claims[:max_claims]

    return OllamaAnalysisResult(
        claims=claims,
        bias_flags=bias_flags,
        annotated_md="\n\n".join(part for part in annotated_parts if part),
        rewrite_md="\n\n".join(part for part in rewrite_parts if part),
    )


def _httpx_timeout(timeout_seconds: float | None) -> httpx.Timeout:
    if timeout_seconds is None:
        return httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
    return httpx.Timeout(connect=10.0, read=timeout_seconds, write=timeout_seconds, pool=timeout_seconds)


def _is_timeout_error(exc: ValueError) -> bool:
    message = str(exc).lower()
    return "timed out" in message or "timeout" in message


def _is_chunk_split_worthy_error(exc: ValueError) -> bool:
    if _is_timeout_error(exc):
        return True
    message = str(exc).lower()
    patterns = [
        "runner has unexpectedly stopped",
        "resource limitations",
        "resource limits",
        "context window",
        "out of memory",
        "oom",
        "schema-compliant json",
        "unterminated string",
        "json decode",
        "expecting value",
        "extra data",
        "invalid control character",
    ]
    return any(pattern in message for pattern in patterns)


def _should_warn_for_gpu_state(summary: str) -> bool:
    state = summary.strip().lower()
    # "idle" means no model currently loaded; it's not a CPU fallback by itself.
    return state not in {"idle", "unknown"}


def _should_disable_ollama_after_error(exc: ValueError) -> bool:
    message = str(exc).lower()
    patterns = [
        "call budget exceeded",
    ]
    return any(pattern in message for pattern in patterns)


def _coerce_noncompliant_result(raw: dict[str, Any], source_text: str) -> OllamaAnalysisResult | None:
    if all(key in raw for key in ("claims", "bias_flags", "rewrite_md")):
        return None

    candidate_texts: list[str] = []
    sections = raw.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            for key in ("section_text", "content", "summary", "body"):
                value = section.get(key)
                if isinstance(value, str) and value.strip():
                    candidate_texts.append(value.strip())
                    break

    for key in ("summary", "content", "body", "analysis"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            candidate_texts.append(value.strip())

    if not candidate_texts:
        candidate_texts = _sentence_candidates(source_text)[:3]
    if not candidate_texts:
        candidate_texts = [source_text[:1200].strip() or "No claim extracted."]

    claims: list[OllamaClaim] = []
    for index, text in enumerate(candidate_texts[:5], start=1):
        span = _find_span_in_text(source_text, text)
        evidence: list[EvidenceSpan] = []
        if span is not None:
            start, end = span
            evidence.append(EvidenceSpan(start=start, end=end, quote=source_text[start:end]))
        claims.append(
            OllamaClaim(
                id=f"auto-{index}",
                type="statement",
                text=text,
                start=span[0] if span is not None else 0,
                end=span[1] if span is not None else min(len(source_text), max(1, len(text))),
                assertiveness="medium",
                score_label=ScoreLabel.SPECULATIVE if not evidence else ScoreLabel.PLAUSIBLE,
                score_reason="Auto-coerced from non-compliant model output.",
                evidence=evidence,
            )
        )

    return OllamaAnalysisResult(
        claims=claims,
        bias_flags=[],
        annotated_md="\n".join([f"- {claim.text}" for claim in claims]),
        rewrite_md=_format_rewrite_from_claims(claims, "Coerced Rewrite"),
    )


def _fallback_result_from_text(source_text: str) -> OllamaAnalysisResult:
    snippets = _sentence_candidates(source_text)[:3]
    if not snippets:
        snippets = [source_text[:1200].strip() or "No claim extracted."]
    claims: list[OllamaClaim] = []
    for index, snippet in enumerate(snippets, start=1):
        span = _find_span_in_text(source_text, snippet)
        evidence: list[EvidenceSpan] = []
        if span is not None:
            evidence.append(EvidenceSpan(start=span[0], end=span[1], quote=source_text[span[0] : span[1]]))
        claims.append(
            OllamaClaim(
                id=f"fallback-{index}",
                type="statement",
                text=snippet,
                start=span[0] if span else 0,
                end=span[1] if span else min(len(source_text), len(snippet) or 1),
                assertiveness="low",
                score_label=ScoreLabel.SPECULATIVE,
                score_reason="Fallback result due to repeated invalid model output.",
                evidence=evidence,
            )
        )
    return OllamaAnalysisResult(
        claims=claims,
        bias_flags=[],
        annotated_md="\n".join([f"- {claim.text}" for claim in claims]),
        rewrite_md=_format_rewrite_from_claims(claims, "Fallback Rewrite"),
    )


def _sentence_candidates(text: str) -> list[str]:
    parts = [part.strip() for part in text.replace("\n", " ").split(".")]
    return [part + "." for part in parts if len(part) > 20]


def _find_span_in_text(source_text: str, fragment: str, anchor: int | None = None) -> tuple[int, int] | None:
    if not source_text or not fragment:
        return None
    needle = fragment.strip()
    if not needle:
        return None
    text_len = len(source_text)
    anchor_pos = max(0, min(text_len, int(anchor))) if anchor is not None else 0

    def _candidates(full: str) -> list[int]:
        positions: list[int] = []
        idx = source_text.find(full)
        while idx != -1:
            positions.append(idx)
            idx = source_text.find(full, idx + 1)
        return positions

    def _choose(positions: list[int], length: int) -> tuple[int, int] | None:
        if not positions:
            return None
        # Prefer first occurrence at/after anchor, else nearest before anchor, tie-break by lower index.
        after = [p for p in positions if p >= anchor_pos]
        chosen = min(after) if after else max(positions)
        span = (chosen, chosen + length)
        if _is_heading_span(source_text, span[0], span[1]):
            # Try next non-heading in deterministic order.
            for pos in sorted(positions):
                candidate = (pos, pos + length)
                if not _is_heading_span(source_text, candidate[0], candidate[1]):
                    return candidate
            return None
        return span

    exact_positions = _candidates(needle)
    span = _choose(exact_positions, len(needle))
    if span:
        return span

    for size in [240, 180, 120, 80]:
        fallback = needle[:size]
        if not fallback:
            continue
        positions = _candidates(fallback)
        span = _choose(positions, len(fallback))
        if span:
            return span
    return None


def _is_heading_span(source_text: str, start: int, end: int) -> bool:
    text_len = len(source_text)
    s = max(0, min(text_len, int(start)))
    e = max(0, min(text_len, int(end)))
    if e <= s:
        return False
    line_start = source_text.rfind("\n", 0, s)
    line_start = 0 if line_start == -1 else line_start + 1
    line_end = source_text.find("\n", e)
    line_end = text_len if line_end == -1 else line_end
    line = source_text[line_start:line_end]
    return _is_heading_like_text(line)


def _is_heading_like_text(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    if re.match(r"^#{1,6}\s+", cleaned):
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 .:_-]{1,80}$", cleaned):
        return True
    if re.match(r"^[A-Z][A-Za-z0-9 /&-]{2,90}:\s*$", cleaned):
        return True
    if re.match(r"^(figure|table|appendix|supplement|exhibit)\b", cleaned, flags=re.IGNORECASE):
        return True
    letters = re.sub(r"[^A-Za-z]", "", cleaned)
    if letters and len(letters) >= 4 and cleaned == cleaned.upper() and len(cleaned) <= 90:
        return True
    return False


def _format_rewrite_from_claims(claims: list[OllamaClaim], title: str) -> str:
    lines = [f"# {title}", "", "## Executive Summary", ""]
    if not claims:
        lines.append("No validated claims were available for rewrite generation.")
        lines.append("")
    else:
        for claim in claims[:3]:
            lines.append(_normalize_sentence(claim.text))
            lines.append("")

    lines.append("## Scope")
    lines.append("")
    lines.append("This rewrite provides a readability-focused narrative based on validated extracted claims.")
    lines.append("Language is normalized to be precise, concise, and operationally actionable.")
    lines.append("")

    lines.append("## Assessment")
    lines.append("")
    if not claims:
        lines.append("No claims available.")
        lines.append("")
    else:
        for claim in claims:
            lines.append(_normalize_sentence(claim.text))
            lines.append("")

    lines.append("## Operational Impact")
    lines.append("")
    lines.append("The described activity can affect detection quality, triage speed, and incident response prioritization.")
    if claims:
        lines.append(_normalize_sentence(claims[0].text))
    lines.append("")

    lines.append("## Mitigations")
    lines.append("")
    lines.append("- Use neutral language anchored to explicit evidence spans.")
    lines.append("- Separate confirmed observations from assumptions.")
    lines.append("- Prioritize remediation tied directly to observed indicators.")
    lines.append("")

    lines.append("## Method Notes")
    lines.append("")
    lines.append(f"- Total claims reviewed: {len(claims)}.")
    lines.append("- This rewrite is deterministic and generated from extracted claims.")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def _normalize_sentence(text: str) -> str:
    cleaned = " ".join((text or "").split()).strip()
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned)
    cleaned = re.sub(r"^>\s+", "", cleaned)
    cleaned = re.sub(r"^[-*+]\s+", "", cleaned)
    cleaned = re.sub(r"^\d+[.)]\s+", "", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return "No statement available."
    if cleaned.endswith((".", "!", "?")):
        return cleaned
    return f"{cleaned}."


def _build_deterministic_annotated_md(source_text: str, claims: list[Claim]) -> str:
    if not source_text:
        return "# Annotated Report\n\n_No source text._\n"

    boundaries = {0, len(source_text)}
    claim_spans: list[tuple[int, int, str, str]] = []
    bias_spans: list[tuple[int, int, str, str]] = []
    bias_codes: list[str] = []
    bias_counter = 1

    for claim in claims:
        score_tag = _score_short_tag(str(claim.score_label))
        claim_code = str(claim.claim_id or "").strip() or "C000"
        for ev in claim.evidence:
            span = _normalize_span(ev.start, ev.end, len(source_text))
            if span is None:
                continue
            start, end = span
            boundaries.add(start)
            boundaries.add(end)
            claim_spans.append((start, end, claim_code, score_tag))

        for flag in claim.bias_flags:
            bias_code = getattr(flag, "bias_code", None)
            if not bias_code:
                bias_code = f"B{bias_counter:03d}"
                bias_counter += 1
            bias_codes.append(bias_code)
            bias_tag = _safe_bias_tag(flag.tag)
            for ev in flag.evidence:
                span = _normalize_span(ev.start, ev.end, len(source_text))
                if span is None:
                    continue
                start, end = span
                boundaries.add(start)
                boundaries.add(end)
                bias_spans.append((start, end, bias_code, bias_tag))

    points = sorted(boundaries)
    annotated_chunks: list[str] = []
    for idx in range(len(points) - 1):
        start = points[idx]
        end = points[idx + 1]
        if end <= start:
            continue
        segment = source_text[start:end]
        active_claim_tags = sorted({(claim_code, tag) for s, e, claim_code, tag in claim_spans if s <= start and end <= e})
        active_bias_tags = sorted({(bias_code, tag) for s, e, bias_code, tag in bias_spans if s <= start and end <= e})
        starting_claim_codes = sorted({claim_code for s, _e, claim_code, _tag in claim_spans if s == start})
        starting_bias_codes = sorted({bias_code for s, _e, bias_code, _tag in bias_spans if s == start})

        prefix_parts = [f"[{claim_code}] " for claim_code in starting_claim_codes]
        prefix_parts.extend(f"[{bias_code}] " for bias_code in starting_bias_codes)
        prefix_parts.extend([f"[CLAIM:{claim_code}:{tag}]" for claim_code, tag in active_claim_tags])
        prefix_parts.extend([f"[BIAS:{bias_code}:{tag}]" for bias_code, tag in active_bias_tags])
        suffix_parts = ["[/BIAS]" for _ in active_bias_tags] + ["[/CLAIM]" for _ in active_claim_tags]
        if prefix_parts:
            annotated_chunks.append("".join(prefix_parts) + segment + "".join(suffix_parts))
        else:
            annotated_chunks.append(segment)

    lines = ["# Annotated Report", "", "## Marked Text", "", "".join(annotated_chunks), "", "## Summary", ""]
    lines.append("| Claim ID | Score | Evidence Spans | Bias Tags |")
    lines.append("|---|---|---:|---|")
    for claim in claims:
        span_count = len(claim.evidence)
        tags = sorted({flag.tag for flag in claim.bias_flags})
        tag_text = ", ".join(tags) if tags else "-"
        lines.append(f"| {claim.claim_id} | {_score_label_text(str(claim.score_label))} | {span_count} | {tag_text} |")
    lines.append("")
    return "\n".join(lines)


def _normalize_span(start: int, end: int, text_len: int) -> tuple[int, int] | None:
    s = max(0, min(text_len, int(start)))
    e = max(0, min(text_len, int(end)))
    if e <= s:
        return None
    return (s, e)


def _score_short_tag(score: str) -> str:
    upper = _score_label_text(score).upper()
    if upper.startswith("SUP"):
        return "SUP"
    if upper.startswith("PLA"):
        return "PLA"
    return "S"


def _score_label_text(score: str) -> str:
    raw = str(score or "").strip()
    if not raw:
        return "SPECULATIVE"
    if "." in raw:
        tail = raw.split(".")[-1].strip()
        if tail:
            return tail
    return raw


def _safe_bias_tag(tag: str) -> str:
    clean = "".join(ch for ch in (tag or "bias") if ch.isalnum() or ch in "-_")
    return clean or "bias"


def _extract_perf_from_ollama_payload(payload: dict[str, Any]) -> dict[str, int]:
    load_duration_ns = int(payload.get("load_duration", 0) or 0)
    total_duration_ns = int(payload.get("total_duration", 0) or 0)
    prompt_eval_duration_ns = int(payload.get("prompt_eval_duration", 0) or 0)
    eval_duration_ns = int(payload.get("eval_duration", 0) or 0)
    prompt_eval_count = int(payload.get("prompt_eval_count", 0) or 0)
    eval_count = int(payload.get("eval_count", 0) or 0)
    return {
        "load_duration_ns": load_duration_ns,
        "total_duration_ns": total_duration_ns,
        "prompt_eval_duration_ns": prompt_eval_duration_ns,
        "eval_duration_ns": eval_duration_ns,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "load_duration_s": load_duration_ns / 1_000_000_000,
        "total_duration_s": total_duration_ns / 1_000_000_000,
        "prompt_eval_duration_s": prompt_eval_duration_ns / 1_000_000_000,
        "eval_duration_s": eval_duration_ns / 1_000_000_000,
    }


def _format_binary_bytes(value: int) -> str:
    size = float(max(0, value))
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.1f} {units[unit_index]}"


def _summarize_perf(samples: list[dict[str, int]]) -> dict[str, Any]:
    if not samples:
        return {
            "calls": 0,
            "prompt_eval_count": 0,
            "eval_count": 0,
            "prompt_eval_duration_s": 0.0,
            "eval_duration_s": 0.0,
            "load_duration_s": 0.0,
            "total_duration_s": 0.0,
            "dominant_lever": "unknown",
        }

    prompt_eval_count = sum(int(s.get("prompt_eval_count", 0)) for s in samples)
    eval_count = sum(int(s.get("eval_count", 0)) for s in samples)
    load_duration_ns = sum(int(s.get("load_duration_ns", 0)) for s in samples)
    total_duration_ns = sum(int(s.get("total_duration_ns", 0)) for s in samples)
    prompt_eval_duration_ns = sum(int(s.get("prompt_eval_duration_ns", 0)) for s in samples)
    eval_duration_ns = sum(int(s.get("eval_duration_ns", 0)) for s in samples)
    load_duration_s = load_duration_ns / 1_000_000_000
    total_duration_s = total_duration_ns / 1_000_000_000
    prompt_eval_duration_s = prompt_eval_duration_ns / 1_000_000_000
    eval_duration_s = eval_duration_ns / 1_000_000_000

    if total_duration_s > max(0.001, prompt_eval_duration_s + eval_duration_s) * 1.4:
        dominant_lever = "gpu_backend_or_runner_overhead"
    elif prompt_eval_duration_s > eval_duration_s * 1.2:
        dominant_lever = "context"
    elif eval_duration_s > prompt_eval_duration_s * 1.2:
        dominant_lever = "output"
    elif len(samples) > 2:
        dominant_lever = "retry_or_chunk_overhead"
    else:
        dominant_lever = "balanced_context_output"

    return {
        "calls": len(samples),
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "load_duration_s": load_duration_s,
        "total_duration_s": total_duration_s,
        "prompt_eval_duration_s": prompt_eval_duration_s,
        "eval_duration_s": eval_duration_s,
        "dominant_lever": dominant_lever,
    }
