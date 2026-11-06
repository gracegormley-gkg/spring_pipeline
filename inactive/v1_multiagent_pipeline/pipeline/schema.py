"""
Pydantic v2 schema for the EIS metadata record.

Transcribed from inter_agent_plan.md §0 (canonical schema) + v1_multiagent_plan.md §4
(structural validators + comment_blocks addition). The schema is stable across
v1/v2/v3; v1 just leaves more fields with status="deferred_v1".

Status conventions:
- "ok"                            : populated and verified
- "needs_review"                  : populated but failed a check
- "deferred_v1"                   : not extracted in v1; field present for forward-compat
- "skipped_section_not_found"     : producer abstained because its source section is missing
- "skipped_summary_unavailable"   : themes-style abstention
- "partial_grounding"             : critic returned "partial"
- "partial_budget_cap"            : extraction stopped mid-field due to budget cap
- "asserted_by_corpus_filter"     : value forced by corpus filter (e.g. lead_agency=USFS)
- "extracted_from_mets"           : sourced from NUL/METS
- "extracted_from_cover"          : sourced from cover-page text
- "rejected_nonverbatim"          : quote failed substring gate
- "comment_response_split_failed" : Stage 3f couldn't split comment from response
- "no_comment_section_found"     : Stage 3f had no public_comments section
"""

from __future__ import annotations

from datetime import datetime, UTC
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

# ---------------------------------------------------------------------------
# Common literal types
# ---------------------------------------------------------------------------

PublicationType = Literal["Draft EIS", "Final EIS", "Supplemental EIS", "ROD", "NOI"]
EISTypeValue = Literal["Draft", "Final", "Supplemental", "ROD", "NOI", "Unlabelled"]
ComponentRole = Literal["main", "appendix", "response_to_comments", "errata", "combined_rod"]

ProvenanceSource = Literal[
    "nul_api",
    "docs_with_digits_json",
    "mets_xml",
    "regex",
    "fuzzy_match",
    "embedding_fallback",
    "haiku_classifier",
    "sonnet_extractor",
    "asserted_by_corpus_filter",
]

SectionName = Literal[
    "cover",
    "summary",
    "purpose_and_need",
    "alternatives",
    "affected_environment",
    "environmental_consequences",
    "public_comments",
    "response_to_comments",
    "rod",
]

QuoteSection = Literal["public_comments", "response_to_comments"]

StanceValue = Literal["supportive", "opposed", "mixed", "neutral"]
StanceTargetType = Literal[
    "proposed_action", "no_action", "specific_alternative", "mitigation", "process", "unknown"
]

StakeholderType = Literal[
    "person", "organization", "tribal_government", "coalition", "agency", "anonymous"
]
AuthorshipRole = Literal[
    "primary_author", "co_signatory", "spokesperson", "hearing_speaker", "form_letter_member"
]

GeometryRole = Literal["site_specific", "corridor", "regional", "programmatic", "unknown"]
GeometryStatus = Literal["named_feature", "unknown"]
PolygonUncertainty = Literal["low", "medium", "high", "unknown"]
CentroidStatus = Literal["valid_for_pin", "misleading_for_pin", "not_applicable"]
SourceGeometryType = Literal["Point", "LineString", "Polygon", "MultiPolygon"]

PlaceRole = Literal[
    "project_area",
    "agency_office",
    "commenter_address",
    "context_reference",
    "alternative_site",
    "comparison_site",
    "mitigation_location",
]

ReviewRouting = Literal["auto_approve", "partial_review", "full_review"]
ExtractionBudgetStatus = Literal["complete", "partial_budget_cap", "error"]
ValidationApproach = Literal["rule_threshold_v1", "calibrated_v2"]
DatePrecision = Literal["day", "month", "year"]


# ---------------------------------------------------------------------------
# Provenance + evidence
# ---------------------------------------------------------------------------

class Provenance(BaseModel):
    """Where a value came from. Required on every populated field."""
    model_config = ConfigDict(extra="forbid")

    source: ProvenanceSource
    source_text_hash: str | None = None  # sha256 of the source span; null for nul_api / asserted

    # Document-relative pointers (null if source is nul_api / asserted)
    page: int | None = None
    char_offset_raw: tuple[int, int] | None = None
    section: SectionName | None = None

    # Free-form note (e.g. "first matching pattern; line 42")
    note: str | None = None

    @model_validator(mode="after")
    def _validate_offsets(self) -> "Provenance":
        if self.char_offset_raw is not None:
            s, e = self.char_offset_raw
            if s < 0 or e < 0 or s >= e:
                raise ValueError(
                    f"char_offset_raw must be non-negative and start < end; got ({s}, {e})"
                )
        return self


# ---------------------------------------------------------------------------
# FieldWithStatus[T] — generic wrapper
# ---------------------------------------------------------------------------

T = TypeVar("T")


class FieldWithStatus(BaseModel, Generic[T]):
    """Generic wrapper: value + status + provenance."""
    model_config = ConfigDict(extra="forbid")

    value: T | None = None
    status: str = "ok"  # per-field status; producers use the appropriate literal
    provenance: Provenance | None = None
    confidence: float | None = None  # 0..1; populated by Stage 4


# ---------------------------------------------------------------------------
# Quote (verbatim-verified)
# ---------------------------------------------------------------------------

class Quote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_raw: str          # exact substring of raw OCR text
    text_display: str      # dehyphenated/normalized for reading
    char_offset_raw: tuple[int, int]
    page: int
    section: QuoteSection
    source_text_hash: str
    normalization_rules_applied: list[str] = Field(default_factory=list)
    quote_status: Literal["ok", "rejected_nonverbatim", "deferred_v1"] = "ok"

    @model_validator(mode="after")
    def _validate_offsets(self) -> "Quote":
        s, e = self.char_offset_raw
        if s < 0 or e <= s:
            raise ValueError(
                f"Quote.char_offset_raw must be non-negative, end > start; got ({s}, {e})"
            )
        return self


# ---------------------------------------------------------------------------
# Sections (Stage 2)
# ---------------------------------------------------------------------------

class SectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: SectionName
    char_span: tuple[int, int] | None = None  # in raw text; null if not_found
    pages: tuple[int, int] | None = None      # (start_page, end_page); 1-indexed
    confidence: float = 1.0
    status: Literal["ok", "not_found", "needs_review", "ambiguous"] = "ok"
    detection_method: Literal["regex", "embedding_fallback", "default_pages", "manual"] = "regex"

    @model_validator(mode="after")
    def _validate_status_span(self) -> "SectionRecord":
        if self.status == "ok" and self.char_span is None:
            raise ValueError(f"section {self.name!r} status=ok but char_span is None")
        if self.char_span is not None:
            s, e = self.char_span
            if s < 0 or e <= s:
                raise ValueError(f"section {self.name!r}: invalid char_span ({s}, {e})")
        return self


# ---------------------------------------------------------------------------
# Components / grouping (Stage 1.5)
# ---------------------------------------------------------------------------

class Component(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    role: ComponentRole = "main"
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Comment blocks (Stage 3f) — top-level for Stage 4 self-consistency
# ---------------------------------------------------------------------------

class CommentBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str
    char_span_raw: tuple[int, int]
    pages: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_offsets(self) -> "CommentBlock":
        s, e = self.char_span_raw
        if s < 0 or e <= s:
            raise ValueError(
                f"CommentBlock {self.block_id} char_span_raw invalid: ({s}, {e})"
            )
        return self


# ---------------------------------------------------------------------------
# Date subfields
# ---------------------------------------------------------------------------

class PublicationDate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | None = None
    precision: DatePrecision | None = None
    status: str = "ok"
    provenance: Provenance | None = None
    evidence: dict[str, Any] | None = None


class DeferredDateField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: None = None
    status: Literal["deferred_v1"] = "deferred_v1"


class DateGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    publication: PublicationDate = Field(default_factory=PublicationDate)
    filing: DeferredDateField = Field(default_factory=DeferredDateField)
    comment_deadline: DeferredDateField = Field(default_factory=DeferredDateField)
    rod_date: DeferredDateField = Field(default_factory=DeferredDateField)


# ---------------------------------------------------------------------------
# Agency
# ---------------------------------------------------------------------------

class CooperatingAgency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    status: str = "ok"
    provenance: Provenance | None = None


class AgencyGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lead_agency: FieldWithStatus[str] = Field(default_factory=FieldWithStatus[str])
    cooperating_agencies: list[CooperatingAgency] = Field(default_factory=list)
    office_or_region: FieldWithStatus[str] = Field(default_factory=FieldWithStatus[str])


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

class ThemeEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    confidence: float


class SubthemeEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    confidence: float
    parent: str  # must be in `themes.primary[].value`


class ThemesField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary: list[ThemeEntry] = Field(default_factory=list)
    subthemes: list[SubthemeEntry] = Field(default_factory=list)
    status: str = "ok"
    provenance: dict[str, Any] | None = None  # { derived_from: "summary", summary_status: "ok" }
    critic: dict[str, Any] | None = None      # populated by Stage 4b

    @model_validator(mode="after")
    def _subthemes_have_chosen_parent(self) -> "ThemesField":
        primaries = {p.value for p in self.primary}
        for s in self.subthemes:
            if s.parent not in primaries:
                raise ValueError(
                    f"subtheme {s.value!r}.parent={s.parent!r} not in chosen primaries {primaries}"
                )
        return self


# ---------------------------------------------------------------------------
# Alternatives
# ---------------------------------------------------------------------------

class Alternative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    description: str
    provenance: Provenance | None = None
    critic: dict[str, Any] | None = None
    status: str = "ok"


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

class NamedPlace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: PlaceRole
    feature_type: str | None = None
    source_dataset: str | None = None
    source_feature_id: str | None = None
    spatial_qualifier: str | None = None
    polygon: dict[str, Any] | None = None  # GeoJSON Feature


class SpatialSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    representative_point: tuple[float, float] | None = None  # lon, lat
    centroid_method: str | None = None
    centroid_status: CentroidStatus | None = None
    bbox: tuple[tuple[float, float], tuple[float, float]] | None = None


class LocationField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    named_places: list[NamedPlace] = Field(default_factory=list)
    project_area_polygon: dict[str, Any] | None = None
    context_polygons: list[dict[str, Any]] = Field(default_factory=list)
    source_geometry: SourceGeometryType | None = None
    geometry_role: GeometryRole = "unknown"
    geometry_status: GeometryStatus = "unknown"
    polygon_uncertainty: PolygonUncertainty = "unknown"
    spatial_summary: SpatialSummary = Field(default_factory=SpatialSummary)
    status: str = "ok"

    @model_validator(mode="after")
    def _polygon_requires_named_place(self) -> "LocationField":
        if self.project_area_polygon is not None:
            has_match = any(
                p.role == "project_area" and p.polygon is not None
                for p in self.named_places
            )
            if not has_match:
                raise ValueError(
                    "location.project_area_polygon set but no named_places entry has "
                    "role=project_area and polygon!=null"
                )
        return self


# ---------------------------------------------------------------------------
# Stakeholders
# ---------------------------------------------------------------------------

class CommentAuthor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: StakeholderType
    canonical_id: str | None = None  # cross-doc registry: v2
    aliases: list[str] = Field(default_factory=list)


class StanceTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: StanceTargetType
    reference_id: str | None = None  # v1 leaves null for specific_alternative
    label_hint: str | None = None


class StanceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stance: StanceValue
    stance_target: StanceTarget
    stance_confidence: float
    quote: Quote | None = None
    sequence_order: int = 1
    critic: dict[str, Any] | None = None


class Stakeholder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comment_block_id: str
    comment_author: CommentAuthor
    represented_entity: str | None = None
    affiliation: str | None = None
    signatories: list[str] = Field(default_factory=list)
    authorship_role: AuthorshipRole = "primary_author"
    stance_records: list[StanceRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class HardGates(BaseModel):
    # Allow 'schema' field name (Pydantic warns it shadows a parent attr).
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    verbatim_quotes: Literal["pass", "fail", "not_run"] = "not_run"
    year_range: Literal["pass", "fail", "not_run"] = "not_run"
    theme_vocab: Literal["pass", "fail", "not_run"] = "not_run"
    geocoding_centroid: Literal["pass", "fail", "not_run"] = "not_run"
    schema_validation: Literal["pass", "fail", "not_run"] = "not_run"


class ValidationField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approach: ValidationApproach = "rule_threshold_v1"
    field_level_confidence: dict[str, float] = Field(default_factory=dict)
    hard_gates: HardGates = Field(default_factory=HardGates)
    self_consistency: dict[str, Any] = Field(default_factory=dict)
    review_routing: ReviewRouting = "auto_approve"
    routing_reasons: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline metadata
# ---------------------------------------------------------------------------

class PipelineMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline_version: str = "v1.0.0"
    stage_versions: dict[str, str] = Field(default_factory=dict)
    model_ids: dict[str, str] = Field(default_factory=dict)
    gazetteer_versions: dict[str, str] = Field(default_factory=dict)
    calibration_model_id: str | None = None
    extracted_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Deferred-v1 fields (schema-enforced via Literal)
# ---------------------------------------------------------------------------

class DeferredField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: None = None
    status: Literal["deferred_v1"] = "deferred_v1"


class DeferredCrossLinks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: list = Field(default_factory=list)
    status: Literal["deferred_v1"] = "deferred_v1"


# ---------------------------------------------------------------------------
# Top-level EISRecord
# ---------------------------------------------------------------------------

class EISRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    publication_id: str
    publication_type: PublicationType | None = None
    is_supplemental: bool = False
    physical_record_ids: list[str] = Field(default_factory=list)
    components: list[Component] = Field(default_factory=list)

    title:    FieldWithStatus[str] = Field(default_factory=FieldWithStatus[str])
    summary:  FieldWithStatus[str] = Field(default_factory=FieldWithStatus[str])
    layman_summary: FieldWithStatus[str] = Field(default_factory=FieldWithStatus[str])

    date: DateGroup = Field(default_factory=DateGroup)
    year: FieldWithStatus[int] = Field(default_factory=FieldWithStatus[int])

    agency: AgencyGroup = Field(default_factory=AgencyGroup)
    eis_type: FieldWithStatus[EISTypeValue] = Field(default_factory=FieldWithStatus)

    sections: list[SectionRecord] = Field(default_factory=list)
    comment_blocks: list[CommentBlock] = Field(default_factory=list)

    alternatives: list[Alternative] = Field(default_factory=list)
    themes: ThemesField = Field(default_factory=ThemesField)
    location: LocationField = Field(default_factory=LocationField)

    stakeholders: list[Stakeholder] = Field(default_factory=list)
    stakeholder_extraction_scope: Literal["organizations_only", "all"] = "organizations_only"
    excluded_stakeholder_types: list[str] = Field(default_factory=lambda: ["person"])
    stakeholder_status: str = "ok"

    historical_context:    DeferredField = Field(default_factory=DeferredField)
    project_status:        DeferredField = Field(default_factory=DeferredField)
    cross_publication_links: DeferredCrossLinks = Field(default_factory=DeferredCrossLinks)

    extraction_budget_status: ExtractionBudgetStatus = "complete"
    coverage_estimate: float = 1.0
    unprocessed_sections: list[str] = Field(default_factory=list)
    per_field_status: dict[str, str] = Field(default_factory=dict)

    validation: ValidationField = Field(default_factory=ValidationField)
    pipeline: PipelineMeta = Field(default_factory=PipelineMeta)

    # ------------------------------------------------------------------
    # Cross-field validators (build brief §4)
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _cross_links_deferred(self) -> "EISRecord":
        if self.cross_publication_links.status != "deferred_v1":
            raise ValueError("cross_publication_links.status must be 'deferred_v1' in v1")
        return self

    @model_validator(mode="after")
    def _provenance_section_in_sections(self) -> "EISRecord":
        """Every Provenance.section reference must match a SectionRecord.name in this doc."""
        section_names = {s.name for s in self.sections}
        if not section_names:
            return self  # Stage 2 hasn't run yet — skip

        def check(prov: Provenance | None, where: str) -> None:
            if prov is None or prov.section is None:
                return
            if prov.section not in section_names:
                raise ValueError(
                    f"provenance.section={prov.section!r} (at {where}) not in "
                    f"document sections {sorted(section_names)}"
                )

        check(self.title.provenance, "title")
        check(self.summary.provenance, "summary")
        check(self.layman_summary.provenance, "layman_summary")
        check(self.date.publication.provenance, "date.publication")
        check(self.year.provenance, "year")
        check(self.agency.lead_agency.provenance, "agency.lead_agency")
        check(self.agency.office_or_region.provenance, "agency.office_or_region")
        for ca in self.agency.cooperating_agencies:
            check(ca.provenance, f"agency.cooperating_agencies[{ca.value!r}]")
        check(self.eis_type.provenance, "eis_type")
        for i, alt in enumerate(self.alternatives):
            check(alt.provenance, f"alternatives[{i}]")
        return self

    @model_validator(mode="after")
    def _stakeholder_block_ids_exist(self) -> "EISRecord":
        block_ids = {b.block_id for b in self.comment_blocks}
        if not block_ids:
            return self  # no blocks recorded yet
        for sh in self.stakeholders:
            if sh.comment_block_id not in block_ids:
                raise ValueError(
                    f"stakeholder.comment_block_id={sh.comment_block_id!r} not in "
                    f"comment_blocks {sorted(block_ids)}"
                )
        return self

    @model_validator(mode="after")
    def _stance_quotes_in_comment_section(self) -> "EISRecord":
        for sh in self.stakeholders:
            for sr in sh.stance_records:
                if sr.quote is None:
                    continue
                if sr.quote.section not in ("public_comments", "response_to_comments"):
                    raise ValueError(
                        f"stakeholder {sh.comment_author.name!r}: quote.section="
                        f"{sr.quote.section!r} must be public_comments or response_to_comments"
                    )
        return self
