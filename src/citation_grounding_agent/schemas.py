"""Typed intermediate and public answer contracts for citation grounding."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceQuote(StrictModel):
    """The agent must quote an exact portion of a retrieved FAQ answer."""

    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class DraftClaim(StrictModel):
    """One factual claim, backed by one or more cited quotations."""

    statement: str = Field(min_length=1)
    evidence: list[EvidenceQuote] = Field(min_length=1)


class GroundedDraft(StrictModel):
    """LLM-proposed result. Never show directly to the user."""

    status: Literal["answered", "insufficient_evidence"]
    claims: list[DraftClaim] = Field(default_factory=list)
    reason: str = ""


class ClaimAssessment(StrictModel):
    claim_index: int = Field(ge=0)
    supported: bool
    explanation: str


class EvidenceAssessment(StrictModel):
    results: list[ClaimAssessment]


class SourceCitation(StrictModel):
    citation_id: str
    chunk_id: str
    faq_id: str
    document_id: str
    source_path: str
    source_page: int = Field(ge=1)
    source_version: str
    quote: str


class VerifiedClaim(StrictModel):
    statement: str
    citation_ids: list[str]


class GroundedAnswer(StrictModel):
    status: Literal["answered", "insufficient_evidence"]
    evidence_status: Literal["supported", "low"]
    answer: str
    claims: list[VerifiedClaim]
    citations: list[SourceCitation]
    reason: str = ""