from __future__ import annotations

from enum import Enum
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer


class ScoreLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    PLAUSIBLE = "PLAUSIBLE"
    SPECULATIVE = "SPECULATIVE"


class EvidenceSpan(BaseModel):
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    quote: str = Field(..., min_length=1)
    sentence_id: int | None = Field(default=None, ge=0, exclude_if=lambda value: value is None)


class BiasFlag(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tag: str = Field(..., min_length=1)
    suggested_fix: str = Field(..., min_length=1)
    evidence: List[EvidenceSpan] = Field(..., min_length=1)
    bias_code: str | None = Field(default=None, min_length=1)

    @model_serializer
    def serialize(self) -> dict:
        data = {
            "tag": self.tag,
            "suggested_fix": self.suggested_fix,
            "evidence": self.evidence,
        }
        if self.bias_code is not None:
            data["bias_code"] = self.bias_code
        return data


class Claim(BaseModel):
    model_config = ConfigDict(exclude_none=True)

    claim_id: str
    text: str
    category: str
    score_label: ScoreLabel
    evidence: List[EvidenceSpan]
    evidence_sentence_ids: list[int] | None = Field(default=None, exclude_if=lambda value: value is None)
    evidence_texts: list[str] | None = Field(default=None, exclude_if=lambda value: value is None)
    evidence_status: Literal["anchored", "missing"] | None = Field(default=None, exclude_if=lambda value: value is None)
    evidence_note: str | None = Field(default=None, exclude_if=lambda value: value is None)
    bias_flags: List[BiasFlag]


class ClaimsDocument(BaseModel):
    model_config = ConfigDict(exclude_none=True)

    engine: str
    claims: List[Claim]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    sample_name: str | None = None
    generate_rewrite: bool = True


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(exclude_none=True)

    claims: ClaimsDocument
    annotated_md: str
    rewrite_md: str
    warning: str | None = None


class OllamaClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    assertiveness: str = Field(..., min_length=1)
    score_label: ScoreLabel
    score_reason: str = Field(..., min_length=1)
    evidence: List[EvidenceSpan] = Field(default_factory=list, max_length=2)


class OllamaBiasFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bias_type: str = Field(..., min_length=1)
    evidence: List[EvidenceSpan] = Field(..., min_length=1, max_length=2)
    why: str = Field(..., min_length=1)
    suggested_fix: str = Field(..., min_length=1)
    bias_code: str | None = Field(default=None, min_length=1)


class OllamaAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: List[OllamaClaim] = Field(default_factory=list, max_length=6)
    bias_flags: List[OllamaBiasFlag] = Field(default_factory=list, max_length=200)
    annotated_md: str
    rewrite_md: str


class OllamaExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: List[OllamaClaim] = Field(default_factory=list, max_length=6)
    bias_flags: List[OllamaBiasFlag] = Field(default_factory=list, max_length=200)
    rewrite_md: str


class OllamaExtractionNoRewriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: List[OllamaClaim] = Field(default_factory=list, max_length=6)
    bias_flags: List[OllamaBiasFlag] = Field(default_factory=list, max_length=200)


class OllamaClaimExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: List[OllamaClaim] = Field(default_factory=list, max_length=6)
    bias_flags: List[OllamaBiasFlag] = Field(default_factory=list, max_length=200)
    rewrite_md: str | None = None


class OllamaBiasChecklist(BaseModel):
    model_config = ConfigDict(extra="forbid")

    certainty_overclaim: bool
    speculation_as_fact: bool
    stereotyping: bool
    regional_generalization: bool
    cultural_essentialism: bool
    loaded_language: bool
    fear_appeal: bool
    single_cause_fallacy: bool
    attribution_without_evidence: bool
    source_imbalance: bool
    normative_judgment: bool
    prescriptive_overreach: bool


class OllamaBiasSentenceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence_id: int = Field(..., ge=0)
    checklist: OllamaBiasChecklist


class OllamaBiasBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[OllamaBiasSentenceCheck] = Field(default_factory=list)


class OllamaBiasMaskItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(..., ge=0)
    flags: int = Field(..., ge=0)


class OllamaBiasMaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: List[OllamaBiasMaskItem] = Field(default_factory=list)
