from __future__ import annotations

import re

from ..io.outputs import write_latest_outputs
from ..models.schemas import AnalyzeRequest, AnalyzeResponse, BiasFlag, Claim, ClaimsDocument, EvidenceSpan, ScoreLabel

BIAS_PATTERNS = {
    "alarmist": ["catastrophic", "disaster", "devastating", "collapse", "panic"],
    "certainty": ["always", "never", "undeniable", "proves", "guaranteed"],
    "hype": ["revolutionary", "breakthrough", "game-changing", "unprecedented"],
}


def _normalize_quote(text: str) -> str:
    return " ".join(text.split())


def _find_all_spans(text: str, needle: str) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    pattern = re.compile(re.escape(needle), re.IGNORECASE)
    for match in pattern.finditer(text):
        spans.append(EvidenceSpan(start=match.start(), end=match.end(), quote=text[match.start() : match.end()]))
    return spans


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"[^.!?\n]+[.!?]?", text):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        start = match.start()
        end = match.end()
        spans.append((start, end, sentence))
    return spans


def run_analysis(request: AnalyzeRequest) -> AnalyzeResponse:
    text = request.text.strip()
    sentences = _sentence_spans(text)
    claims: list[Claim] = []

    for idx, (start, end, sentence) in enumerate(sentences, start=1):
        quote = text[start:end].strip()
        evidence = [EvidenceSpan(start=start, end=end, quote=quote)] if quote else []
        bias_flags: list[BiasFlag] = []

        for tag, terms in BIAS_PATTERNS.items():
            tag_evidence: list[EvidenceSpan] = []
            for term in terms:
                tag_evidence.extend(_find_all_spans(quote, term))
            if tag_evidence:
                adjusted = [
                    EvidenceSpan(
                        start=ev.start + start,
                        end=ev.end + start,
                        quote=text[ev.start + start : ev.end + start],
                    )
                    for ev in tag_evidence
                ]
                bias_flags.append(
                    BiasFlag(
                        tag=tag,
                        suggested_fix=f"Use neutral phrasing for {tag} language.",
                        evidence=adjusted,
                    )
                )

        score_label = ScoreLabel.SUPPORTED if evidence else ScoreLabel.SPECULATIVE

        claims.append(
            Claim(
                claim_id=f"C{idx:03d}",
                text=_normalize_quote(sentence),
                category="statement",
                score_label=score_label,
                evidence=evidence,
                bias_flags=bias_flags,
            )
        )

    doc = ClaimsDocument(engine="placeholder", claims=claims)
    annotated_md = _build_annotated_markdown(doc)
    rewrite_md = _build_rewrite_markdown(doc) if request.generate_rewrite else ""

    return AnalyzeResponse(claims=doc, annotated_md=annotated_md, rewrite_md=rewrite_md)


def _build_annotated_markdown(doc: ClaimsDocument) -> str:
    lines = ["# Annotated Claims", ""]
    for claim in doc.claims:
        lines.append(f"## {claim.claim_id}")
        lines.append(f"- Text: {claim.text}")
        lines.append(f"- Score: {claim.score_label}")
        if claim.evidence:
            lines.append("- Evidence:")
            for ev in claim.evidence:
                lines.append(f"  - [{ev.start}:{ev.end}] \"{ev.quote}\"")
        else:
            lines.append("- Evidence: none")
        if claim.bias_flags:
            lines.append("- Bias Flags:")
            for flag in claim.bias_flags:
                joined = ", ".join([f"[{ev.start}:{ev.end}] {ev.quote}" for ev in flag.evidence])
                lines.append(f"  - {flag.tag}: {joined} (fix: {flag.suggested_fix})")
        else:
            lines.append("- Bias Flags: none")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_rewrite_markdown(doc: ClaimsDocument) -> str:
    lines = ["# Neutral Rewrite", ""]
    for claim in doc.claims:
        sentence = claim.text
        for terms in BIAS_PATTERNS.values():
            for term in terms:
                sentence = re.sub(rf"\\b{re.escape(term)}\\b", "notable", sentence, flags=re.IGNORECASE)
        lines.append(f"- {sentence}")
    lines.append("")
    return "\n".join(lines)


