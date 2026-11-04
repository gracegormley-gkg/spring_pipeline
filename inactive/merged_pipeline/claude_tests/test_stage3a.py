"""
Stage 3a tests — METS-anchored fields with extractive cross-check.

Coverage:
  - METS happy path (NUL has title + creator + date)
  - NUL empty title -> Haiku gap-fill (with stubbed LLM)
  - NUL empty creator -> cover-text vocab scan (no LLM needed)
  - NUL empty creator + cover scan miss -> Haiku gap-fill canonicalizes
  - NUL empty creator + cover scan miss + Haiku non-canonical -> low-conf
  - NUL bad year -> regex fallback on first 5 fake pages
  - Year out of NEPA range -> needs_review
  - office_or_region always deferred_v1
  - Provenance.section/char_offset_raw set on cover-anchored fields
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import config
from pipeline.ingest import FakePage, IngestArtifact
from pipeline.schema import EISRecord, SectionRecord
from pipeline.sections import DetectedSection, SectionsArtifact
from pipeline.stage3a_mets_fields import (
    _extract_year_fallback,
    _match_agency_in_strings,
    _parse_publication_date,
    _scan_cover_for_agency,
    run as stage3a_run,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_doc.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_artifact(raw: str, nul: dict | None = None, page_size: int = 1000) -> IngestArtifact:
    pages: list[FakePage] = []
    pos = 0
    page_num = 1
    while pos < len(raw):
        end = min(pos + page_size, len(raw))
        pages.append(FakePage(page_num=page_num, char_start_raw=pos, char_end_raw=end))
        pos = end
        page_num += 1
    return IngestArtifact(
        publication_id="test",
        physical_record_ids=["test"],
        raw_text=raw,
        text_normalized=raw,
        alignment_map=[],
        pages=pages,
        nul_metadata=nul or {},
        raw_text_hash="sha256:test",
    )


def _build_sections_artifact(ingest: IngestArtifact) -> SectionsArtifact:
    """Build a sections artifact with cover defaulted to first 3 fake pages."""
    end_page = min(3, len(ingest.pages))
    if end_page == 0:
        return SectionsArtifact(publication_id=ingest.publication_id, sections=[])
    end_char = ingest.pages[end_page - 1].char_end_raw
    cover = DetectedSection(
        name="cover",
        char_span=(0, end_char),
        pages=(1, end_page),
        confidence=1.0,
        status="ok",
        detection_method="default_pages",
    )
    return SectionsArtifact(publication_id=ingest.publication_id, sections=[cover])


def _new_record(sections_artifact: SectionsArtifact | None = None) -> EISRecord:
    """EISRecord seeded with sections so cross-field validators can resolve
    provenance.section references."""
    sections: list[SectionRecord] = []
    if sections_artifact is not None:
        sections = [s.to_schema_record() for s in sections_artifact.sections]
    return EISRecord(publication_id="test", sections=sections)


class FakeLLM:
    """Stub LLM client. Returns canned dicts in FIFO order."""

    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.dry_run = False
        self.models = {"haiku": "haiku", "sonnet": "sonnet", "opus": "opus"}

    def call_json(self, model, system, messages, max_tokens=2048, temperature=0.0, label=""):
        self.calls.append({"model": model, "label": label})
        if not self.responses:
            raise RuntimeError(f"FakeLLM exhausted; unexpected call (label={label!r})")
        return self.responses.pop(0)

    def call(self, model, system, messages, max_tokens=2048, temperature=0.0, label=""):
        self.calls.append({"model": model, "label": label})
        if not self.responses:
            raise RuntimeError(f"FakeLLM exhausted; unexpected call (label={label!r})")
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("2017", (2017, "2017", "year")),
    ("1971", (1971, "1971", "year")),
    ("2017-05", (2017, "2017-05", "month")),
    ("2017-05-21", (2017, "2017-05-21", "day")),
    ("issued in 1985", (1985, "1985", "year")),
    ("not a year", (None, None, None)),
    ("", (None, None, None)),
    (None, (None, None, None)),
])
def test_parse_publication_date(value, expected) -> None:
    assert _parse_publication_date(value) == expected


# ---------------------------------------------------------------------------
# Year fallback
# ---------------------------------------------------------------------------

def test_year_fallback_picks_most_frequent() -> None:
    raw = "1985 was a long time ago. The 1985 study reviewed and approved 1985 again."
    art = _build_artifact(raw)
    assert _extract_year_fallback(art) == 1985


def test_year_fallback_rejects_out_of_range() -> None:
    raw = "ancient document from 1850; the 2020 update revisited it. Future date: 3000."
    art = _build_artifact(raw)
    # 1850 doesn't match the regex (pre-1970); 3000 doesn't match either.
    # 2020 matches and falls in [NEPA_YEAR, MAX_YEAR].
    assert _extract_year_fallback(art) == 2020


def test_year_fallback_returns_none_when_only_post_max_year() -> None:
    """Regex matches 2050 (20\\d{2}) but range gate rejects it (post MAX_YEAR)."""
    # Pick a year guaranteed to be > MAX_YEAR for many years to come.
    raw = "the year 2099 is far in the future and shouldn't qualify"
    art = _build_artifact(raw)
    assert _extract_year_fallback(art) is None


def test_year_fallback_returns_none_if_no_year() -> None:
    raw = "no years here at all just words"
    art = _build_artifact(raw)
    assert _extract_year_fallback(art) is None


# ---------------------------------------------------------------------------
# Agency matching
# ---------------------------------------------------------------------------

def test_match_agency_in_strings_direct() -> None:
    abbr, label = _match_agency_in_strings(["U.S. Forest Service"])
    assert abbr == "USFS"
    assert label == "U.S. Forest Service"


def test_match_agency_in_strings_substring() -> None:
    abbr, label = _match_agency_in_strings(["United States Forest Service, Region 5"])
    assert abbr == "USFS"


def test_match_agency_in_strings_miss() -> None:
    abbr, label = _match_agency_in_strings(["Some Random Contractor LLC"])
    assert abbr is None and label is None


def test_scan_cover_for_agency_finds_canonical_in_text() -> None:
    cover = (
        "U. S. DEPARTMENT OF AGRICULTURE\n"
        "FINAL ENVIRONMENTAL STATEMENT\n"
        "Forest Service, Region 5\n"
    )
    abbr, matched, offset = _scan_cover_for_agency(cover)
    # The longest variant present is "U.S. Department of Agriculture" (canonical)
    # but with "U. S. " spacing, only "Forest Service" / "USDA Forest Service"
    # / "Department of Agriculture" should hit.
    assert abbr in {"USFS", "USDA"}
    assert offset is not None and offset >= 0


def test_scan_cover_for_agency_skips_short_abbreviations() -> None:
    """3-letter abbreviations like 'EPA' shouldn't match inside unrelated words."""
    # 'BIA' would match word 'biased' in a naive substring scan; we skip <=3 char vocab.
    cover = "This proposal is heavily biased toward existing infrastructure."
    abbr, matched, offset = _scan_cover_for_agency(cover)
    assert abbr is None


# ---------------------------------------------------------------------------
# Stage 3a happy path: NUL provides everything
# ---------------------------------------------------------------------------

def test_stage3a_mets_happy_path() -> None:
    raw = "U.S. Forest Service Final EIS, 2017."
    art = _build_artifact(raw, nul={
        "title": "Lake Tahoe Restoration Plan",
        "creator": ["U.S. Forest Service"],
        "contributor": [],
        "date_created": "2017",
    })
    sections_art = _build_sections_artifact(art)
    record = _new_record(sections_art)

    warnings = stage3a_run(record, art, sections_art)

    assert record.title.value == "Lake Tahoe Restoration Plan"
    assert record.title.status == "extracted_from_mets"
    assert record.title.provenance is not None
    assert record.title.provenance.source == "nul_api"

    assert record.year.value == 2017
    assert record.year.status == "extracted_from_mets"

    assert record.date.publication.value == "2017"
    assert record.date.publication.precision == "year"

    assert record.agency.lead_agency.value == "USFS"
    assert record.agency.lead_agency.status == "extracted_from_mets"

    assert record.agency.office_or_region.status == "deferred_v1"
    assert warnings == []


# ---------------------------------------------------------------------------
# Stage 3a: NUL empty title -> Haiku gap-fill
# ---------------------------------------------------------------------------

def test_stage3a_title_haiku_gapfill() -> None:
    raw = "ENVIRONMENTAL STATEMENT FOR PROPOSED CASTAIC-HASKELL POWER LINE\n"
    art = _build_artifact(raw, nul={
        "title": None,
        "creator": ["U.S. Forest Service"],
        "contributor": [],
        "date_created": "1971",
    })
    sections_art = _build_sections_artifact(art)
    record = _new_record(sections_art)
    fake_llm = FakeLLM([
        {"title": "Castaic-Haskell Junction Power Transmission Line", "found": True},
    ])

    warnings = stage3a_run(record, art, sections_art, llm=fake_llm)

    assert record.title.value == "Castaic-Haskell Junction Power Transmission Line"
    assert record.title.status == "needs_review"  # Haiku-extracted always cross-checked
    assert record.title.provenance.source == "haiku_gapfill"
    assert record.title.provenance.section == "cover"
    assert record.title.provenance.char_offset_raw is not None


def test_stage3a_title_no_llm_left_as_needs_review() -> None:
    raw = "Some doc text without obvious title patterns.\n"
    art = _build_artifact(raw, nul={"title": None, "creator": ["U.S. Forest Service"], "date_created": "1971"})
    sections_art = _build_sections_artifact(art)
    record = _new_record(sections_art)

    warnings = stage3a_run(record, art, sections_art, llm=None)

    assert record.title.value is None
    assert record.title.status == "needs_review"
    assert any("title" in w for w in warnings)


# ---------------------------------------------------------------------------
# Stage 3a: NUL empty creator -> cover-vocab scan
# ---------------------------------------------------------------------------

def test_stage3a_lead_agency_cover_scan() -> None:
    raw = (
        "U.S. Department of Agriculture\n"
        "Forest Service Region 5\n"
        "Final Environmental Statement for project\n"
    )
    art = _build_artifact(raw, nul={
        "title": "X",
        "creator": [],
        "contributor": [],
        "date_created": "1971",
    })
    sections_art = _build_sections_artifact(art)
    record = _new_record(sections_art)

    warnings = stage3a_run(record, art, sections_art, llm=None)

    # Cover scan should have matched the agency without any LLM call.
    assert record.agency.lead_agency.value in {"USFS", "USDA"}
    assert record.agency.lead_agency.status == "needs_review"
    assert record.agency.lead_agency.provenance.source == "controlled_vocab_match"
    assert record.agency.lead_agency.provenance.section == "cover"
    assert record.agency.lead_agency.provenance.char_offset_raw is not None


def test_stage3a_lead_agency_haiku_gapfill_when_cover_scan_misses() -> None:
    """Cover has no recognizable vocab -> Haiku is invoked."""
    raw = "Some abstract project document with no clear federal agency on cover.\n"
    art = _build_artifact(raw, nul={"title": "X", "creator": [], "date_created": "1971"})
    sections_art = _build_sections_artifact(art)
    record = _new_record(sections_art)

    fake_llm = FakeLLM([
        {"agency": "U.S. Forest Service", "found": True},
    ])
    warnings = stage3a_run(record, art, sections_art, llm=fake_llm)

    assert record.agency.lead_agency.value == "USFS"
    assert record.agency.lead_agency.provenance.source == "haiku_gapfill"
    assert record.agency.lead_agency.status == "needs_review"


def test_stage3a_lead_agency_haiku_returns_noncanonical() -> None:
    raw = "No agency names on cover.\n"
    art = _build_artifact(raw, nul={"title": "X", "creator": [], "date_created": "1971"})
    sections_art = _build_sections_artifact(art)
    record = _new_record(sections_art)

    fake_llm = FakeLLM([
        {"agency": "Some Made Up Agency", "found": True},
    ])
    warnings = stage3a_run(record, art, sections_art, llm=fake_llm)

    # Non-canonical Haiku output is preserved with low confidence.
    assert record.agency.lead_agency.value == "Some Made Up Agency"
    assert record.agency.lead_agency.confidence == 0.45
    assert any("did not canonicalize" in w for w in warnings)


# ---------------------------------------------------------------------------
# Year out-of-range gate
# ---------------------------------------------------------------------------

def test_stage3a_year_out_of_range_marks_needs_review() -> None:
    raw = "Project doc.\n"
    art = _build_artifact(raw, nul={
        "title": "X",
        "creator": ["U.S. Forest Service"],
        "date_created": "1850",  # before NEPA
    })
    sections_art = _build_sections_artifact(art)
    record = _new_record(sections_art)

    warnings = stage3a_run(record, art, sections_art)

    # 1850 was parsed (NUL -> precision=year), but before NEPA -> needs_review
    # via year-range gate. The _parse_publication_date regex only matches
    # 1970+, so 1850 returns None -> regex fallback also fails -> year=None.
    # In that case status stays needs_review and warnings include year info.
    assert record.year.status == "needs_review"


# ---------------------------------------------------------------------------
# Schema validator: cover provenance only allowed if cover in record.sections
# ---------------------------------------------------------------------------

def test_stage3a_provenance_section_validates_against_record_sections() -> None:
    """Stage 3a sets provenance.section='cover' on fallback fields. The
    schema's _provenance_section_in_sections validator enforces that the
    record's sections list contains 'cover'. This test would fail if we
    forgot to seed sections."""
    raw = "U.S. Department of Agriculture cover content.\n"
    art = _build_artifact(raw, nul={"title": None, "creator": [], "date_created": "1971"})
    sections_art = _build_sections_artifact(art)
    record = _new_record(sections_art)
    fake_llm = FakeLLM([
        {"title": "Project Name", "found": True},
    ])

    # Should not raise.
    stage3a_run(record, art, sections_art, llm=fake_llm)
    # Round-trip: re-validate by serializing+reparsing.
    EISRecord.model_validate(record.model_dump())


# ---------------------------------------------------------------------------
# Integration on real fixture
# ---------------------------------------------------------------------------

def test_stage3a_on_fixture_doc() -> None:
    """Castaic-Haskell fixture: NUL has title only, creator+date empty.
    Expected outcomes (no LLM provided):
      - title: extracted_from_mets (NUL provided it)
      - year:  regex fallback finds 1971 in cover text -> needs_review
      - lead_agency: cover scan finds 'Department of Agriculture' / 'Forest Service'
        -> USFS or USDA, status=needs_review, source=controlled_vocab_match
    """
    if not FIXTURE_PATH.exists():
        pytest.skip(f"fixture not present: {FIXTURE_PATH}")

    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    doc_key, raw_text = next(iter(payload.items()))

    # Hand-construct nul_metadata to match what we observed empirically:
    # NUL has title, no creator, no date.
    nul = {
        "title": "Environmental Impact Statement, Proposed Castaic-Haskell Junction Power Transmission Line",
        "creator": [],
        "contributor": [],
        "date_created": None,
    }
    art = _build_artifact(raw_text, nul=nul, page_size=config.CHARS_PER_FAKE_PAGE)
    sections_art = _build_sections_artifact(art)
    record = _new_record(sections_art)

    warnings = stage3a_run(record, art, sections_art, llm=None)

    # Title from NUL
    assert record.title.value is not None
    assert "Castaic-Haskell" in record.title.value
    assert record.title.status == "extracted_from_mets"

    # Year from regex fallback (cover contains "APR 14 1971")
    assert record.year.value == 1971
    assert record.year.status == "needs_review"
    assert record.year.provenance.source == "regex"

    # Lead agency from cover vocab scan (cover has "DEPARTMENT OF AGRICULTURE"
    # and "FOREST SERVICE")
    assert record.agency.lead_agency.value in {"USFS", "USDA"}
    assert record.agency.lead_agency.status == "needs_review"
    assert record.agency.lead_agency.provenance.source == "controlled_vocab_match"

    # office_or_region always deferred
    assert record.agency.office_or_region.status == "deferred_v1"
