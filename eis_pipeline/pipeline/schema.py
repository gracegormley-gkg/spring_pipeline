"""
Pydantic models for the EIS metadata record.
All fields always present; missing data uses null / "unknown" / "insufficient_information".
"""

from __future__ import annotations

from datetime import datetime, UTC
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Evidence pointer
# ---------------------------------------------------------------------------

class EvidencePointer(BaseModel):
    chunk_id: str
    pages: list[int]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class OCRInfo(BaseModel):
    median_confidence: float | None
    page_count: int
    unclear_document_flag: bool


class LocationInfo(BaseModel):
    name: str | None = None
    state: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    geocode_source: str | None = None


class LeadAgency(BaseModel):
    name: str | None = None
    abbreviation: str | None = None
    source: Literal["mets", "regex", "fuzzy_match", "nul", "unknown"] = "unknown"


class SectionInfo(BaseModel):
    title: str
    start_page: int
    end_page: int


class SummaryField(BaseModel):
    # `text` is the detailed summary — the longer, evidence-cited version
    # produced from the chunked document. The critic validates this version.
    text: str
    # `layman_text` is a plain-language rewrite of `text` (no new content).
    # Generated in a second pass that takes `text` as input — so it inherits
    # the factual grounding without re-reading the document.
    layman_text: str | None = None
    evidence: list[EvidencePointer] = Field(default_factory=list)
    status: Literal["populated", "insufficient_information"] = "populated"


class ThemesField(BaseModel):
    primary: list[str] = Field(default_factory=list)
    subthemes: list[str] = Field(default_factory=list)


class AlternativeItem(BaseModel):
    name: str
    description: str
    evidence: list[EvidencePointer] = Field(default_factory=list)


class QuoteItem(BaseModel):
    text: str
    chunk_id: str
    page: int
    substring_verified: bool


class KeyPersonOrGroup(BaseModel):
    name: str
    type: Literal["person", "organization", "unknown"] = "unknown"
    # For a person: their title/affiliation (e.g. "EPA Acting Director").
    # For a group/bureau: their role in the project (e.g. "lead agency",
    # "consulted agency", "objecting party", "contractor").
    role: str | None = None
    # 1–2 sentence summary of the entity's opinion/stance.
    opinion_summary: str | None = None
    first_appearance_chunk: str | None = None
    appearance_order: int | None = None
    stance: Literal["supportive", "opposed", "mixed", "neutral", "insufficient_information"] = "insufficient_information"
    stance_evidence: list[EvidencePointer] = Field(default_factory=list)
    quote: QuoteItem | None = None


class HistoricalContextClaim(BaseModel):
    sentence: str
    evidence: list[EvidencePointer] = Field(default_factory=list)


class HistoricalContextInternal(BaseModel):
    text: str | None = None
    claims: list[HistoricalContextClaim] = Field(default_factory=list)
    status: Literal["populated", "insufficient_information"] = "insufficient_information"


class HistoricalContextExternal(BaseModel):
    text: None = None
    sources: list = Field(default_factory=list)
    tier: None = None
    status: Literal["no_external_context_available"] = "no_external_context_available"
    deferred_to_v2: bool = True


class ProjectCurrentStatus(BaseModel):
    value: Literal["unknown"] = "unknown"
    source_url: None = None
    source_passage: None = None
    source_tier: None = None
    evidence_date: None = None
    as_of_date: None = None
    disambiguation_checks_passed: list = Field(default_factory=list)
    deferred_to_v2: bool = True


class NERResult(BaseModel):
    people: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    # Provenance per entity. Keys are entity names (matching the above lists);
    # values describe where the entity came from. Used by the key_people module
    # to decide whether the entity needs Haiku triage (spacy_*) or bypasses it
    # (dict_*, haiku_gapfill).
    sources: dict[str, str] = Field(default_factory=dict)
    raw_count_before_dedupe: int = 0
    deduped_count: int = 0


class ChunkRecord(BaseModel):
    chunk_id: str
    title: str
    description: str
    topic_tags: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    text: str = ""
    median_confidence: float | None = None
    used: bool = True


class PipelineMetadata(BaseModel):
    version: str = "v1.0"
    run_timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    models_used: dict[str, str] = Field(default_factory=dict)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level record
# ---------------------------------------------------------------------------

class EISRecord(BaseModel):
    doc_id: str
    project_id: str
    title: str | None = None
    ocr: OCRInfo
    eis_type: Literal["Draft", "Final", "Supplemental", "Unlabelled"] = "Unlabelled"
    length_category: Literal["short", "medium", "long"] = "medium"
    word_count: int = 0
    has_headings: bool = False
    has_toc: bool = False
    sections: list[SectionInfo] = Field(default_factory=list)
    lead_agency: LeadAgency = Field(default_factory=LeadAgency)
    date: str | None = None
    year: int | None = None
    location: LocationInfo = Field(default_factory=LocationInfo)
    themes: ThemesField = Field(default_factory=ThemesField)
    summary: SummaryField | None = None
    alternatives_proposed: list[AlternativeItem] = Field(default_factory=list)
    key_people_and_groups: list[KeyPersonOrGroup] = Field(default_factory=list)
    historical_context_internal: HistoricalContextInternal = Field(
        default_factory=HistoricalContextInternal
    )
    historical_context_external: HistoricalContextExternal = Field(
        default_factory=HistoricalContextExternal
    )
    project_current_status: ProjectCurrentStatus = Field(
        default_factory=ProjectCurrentStatus
    )
    ner: NERResult = Field(default_factory=NERResult)
    chunks: list[ChunkRecord] = Field(default_factory=list)
    pipeline_metadata: PipelineMetadata = Field(default_factory=PipelineMetadata)
