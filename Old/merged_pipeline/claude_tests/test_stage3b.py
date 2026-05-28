"""
Stage 3b tests — EIS type detection.

Coverage:
  - Regex hits modern "Final Environmental Impact Statement"
  - Regex hits NEPA-era "FINAL ENVIRONMENTAL STATEMENT" (Impact optional)
  - Draft / Supplemental / Supplemental Final all canonicalize correctly
  - Cover missing -> Unlabelled / needs_review
  - Regex misses + no LLM -> Unlabelled / needs_review
  - Regex misses + Haiku returns valid value -> extracted_from_cover
  - Regex misses + Haiku returns garbage -> Unlabelled
  - Regex misses + Haiku raises -> Unlabelled
  - Integration on fixture (matches "FINAL ENVIRONMENTAL STATEMENT" via regex)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import config
from pipeline.ingest import FakePage, IngestArtifact
from pipeline.schema import EISRecord, SectionRecord
from pipeline.sections import DetectedSection, SectionsArtifact
from pipeline.stage3b_eis_type import _regex_match, run as stage3b_run


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_doc.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_artifact(raw: str, page_size: int = 1000) -> IngestArtifact:
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
        nul_metadata={},
        raw_text_hash="sha256:test",
    )


def _build_sections_with_cover(ingest: IngestArtifact) -> SectionsArtifact:
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


def _build_sections_no_cover(ingest: IngestArtifact) -> SectionsArtifact:
    return SectionsArtifact(publication_id=ingest.publication_id, sections=[])


def _new_record(sections_artifact: SectionsArtifact) -> EISRecord:
    sections: list[SectionRecord] = [s.to_schema_record() for s in sections_artifact.sections]
    return EISRecord(publication_id="test", sections=sections)


class FakeLLM:
    """Stub LLM. `responses` is a list of strings (returned by .call) or a list
    of objects that raise (use `_RaiseSentinel`)."""
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.dry_run = False
        self.models = {"haiku": "haiku", "sonnet": "sonnet", "opus": "opus"}

    def call(self, model, system, messages, max_tokens=2048, temperature=0.0, label=""):
        self.calls.append({"model": model, "label": label})
        if not self.responses:
            raise RuntimeError(f"FakeLLM exhausted; unexpected call (label={label!r})")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def call_json(self, model, system, messages, max_tokens=2048, temperature=0.0, label=""):
        # Stage 3b only uses .call() (raw text), not .call_json
        raise RuntimeError("3b should call .call(), not .call_json()")


# ---------------------------------------------------------------------------
# Regex matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Final Environmental Impact Statement", "Final"),
    ("FINAL ENVIRONMENTAL IMPACT STATEMENT", "Final"),
    ("Draft Environmental Impact Statement", "Draft"),
    ("DRAFT ENVIRONMENTAL IMPACT STATEMENT", "Draft"),
    ("Supplemental Environmental Impact Statement", "Supplemental"),
    ("Supplemental Final Environmental Impact Statement", "Supplemental"),
    ("Supplemental Draft Environmental Impact Statement", "Supplemental"),
    # NEPA-era 1970s shape: "Environmental Statement" (no "Impact")
    ("FINAL ENVIRONMENTAL STATEMENT", "Final"),
    ("Draft Environmental Statement", "Draft"),
    # No match
    ("Programmatic Statement", None),
    ("Environmental Assessment", None),
    ("just prose, no labels", None),
    ("", None),
])
def test_regex_match(text, expected) -> None:
    assert _regex_match(text) == expected


# ---------------------------------------------------------------------------
# End-to-end run() — regex paths
# ---------------------------------------------------------------------------

def test_stage3b_regex_final() -> None:
    raw = "PROJECT COVER PAGE\nFinal Environmental Impact Statement\nfor the proposed action.\n"
    art = _build_artifact(raw)
    sections_art = _build_sections_with_cover(art)
    record = _new_record(sections_art)

    warnings = stage3b_run(record, art, sections_art, llm=None)

    assert record.eis_type.value == "Final"
    assert record.eis_type.status == "extracted_from_cover"
    assert record.eis_type.provenance.source == "regex"
    assert record.eis_type.provenance.section == "cover"
    assert warnings == []


def test_stage3b_regex_handles_nepa_era_no_impact() -> None:
    """1970s shape: 'FINAL ENVIRONMENTAL STATEMENT' — Impact word absent."""
    raw = "U. S. DEPARTMENT OF AGRICULTURE\nFINAL ENVIRONMENTAL STATEMENT\nFOR PROJECT\n"
    art = _build_artifact(raw)
    sections_art = _build_sections_with_cover(art)
    record = _new_record(sections_art)

    warnings = stage3b_run(record, art, sections_art, llm=None)

    assert record.eis_type.value == "Final"
    assert record.eis_type.provenance.source == "regex"


def test_stage3b_regex_supplemental() -> None:
    raw = "Supplemental Environmental Impact Statement\n"
    art = _build_artifact(raw)
    sections_art = _build_sections_with_cover(art)
    record = _new_record(sections_art)

    stage3b_run(record, art, sections_art, llm=None)
    assert record.eis_type.value == "Supplemental"


# ---------------------------------------------------------------------------
# No cover -> unlabelled
# ---------------------------------------------------------------------------

def test_stage3b_missing_cover_returns_unlabelled() -> None:
    raw = "Some prose with no cover detected.\n"
    art = _build_artifact(raw)
    sections_art = _build_sections_no_cover(art)
    record = _new_record(sections_art)

    warnings = stage3b_run(record, art, sections_art, llm=None)

    assert record.eis_type.value == "Unlabelled"
    assert record.eis_type.status == "needs_review"
    assert any("cover" in w for w in warnings)


# ---------------------------------------------------------------------------
# Regex miss -> Haiku fallback paths
# ---------------------------------------------------------------------------

def test_stage3b_regex_miss_no_llm() -> None:
    """Regex finds nothing AND no LLM provided -> Unlabelled, needs_review."""
    raw = "Generic memo with no EIS-type wording on the cover at all.\n"
    art = _build_artifact(raw)
    sections_art = _build_sections_with_cover(art)
    record = _new_record(sections_art)

    warnings = stage3b_run(record, art, sections_art, llm=None)

    assert record.eis_type.value == "Unlabelled"
    assert record.eis_type.status == "needs_review"
    assert any("regex missed" in w for w in warnings)


def test_stage3b_haiku_returns_valid() -> None:
    """Regex misses, Haiku returns 'Final' -> extracted_from_cover via Haiku."""
    raw = "Cover text with no regex match for EIS type.\n"
    art = _build_artifact(raw)
    sections_art = _build_sections_with_cover(art)
    record = _new_record(sections_art)
    fake_llm = FakeLLM(["Final"])

    warnings = stage3b_run(record, art, sections_art, llm=fake_llm)

    assert record.eis_type.value == "Final"
    assert record.eis_type.status == "extracted_from_cover"
    assert record.eis_type.provenance.source == "haiku_classifier"


def test_stage3b_haiku_returns_garbage() -> None:
    """Haiku returns something not in VALID_EIS_TYPES -> Unlabelled."""
    raw = "Cover with no regex match.\n"
    art = _build_artifact(raw)
    sections_art = _build_sections_with_cover(art)
    record = _new_record(sections_art)
    fake_llm = FakeLLM(["NotARealType"])

    warnings = stage3b_run(record, art, sections_art, llm=fake_llm)

    assert record.eis_type.value == "Unlabelled"
    assert record.eis_type.status == "needs_review"
    assert any("unexpected value" in w for w in warnings)


def test_stage3b_haiku_raises_gracefully() -> None:
    """LLM raises -> stage3b sets Unlabelled (defensive degradation)."""
    raw = "Cover with no regex match.\n"
    art = _build_artifact(raw)
    sections_art = _build_sections_with_cover(art)
    record = _new_record(sections_art)
    fake_llm = FakeLLM([RuntimeError("simulated outage")])

    warnings = stage3b_run(record, art, sections_art, llm=fake_llm)

    # The exception is caught inside _haiku_fallback -> returns None ->
    # stage3b takes the "unexpected value" branch and sets Unlabelled.
    assert record.eis_type.value == "Unlabelled"
    assert record.eis_type.status == "needs_review"


def test_stage3b_haiku_returns_unlabelled() -> None:
    raw = "Cover with no regex match.\n"
    art = _build_artifact(raw)
    sections_art = _build_sections_with_cover(art)
    record = _new_record(sections_art)
    fake_llm = FakeLLM(["Unlabelled"])

    stage3b_run(record, art, sections_art, llm=fake_llm)

    assert record.eis_type.value == "Unlabelled"
    assert record.eis_type.status == "needs_review"
    assert record.eis_type.confidence == 0.40


# ---------------------------------------------------------------------------
# Integration on fixture
# ---------------------------------------------------------------------------

def test_stage3b_on_fixture_doc() -> None:
    """Castaic-Haskell fixture has 'FINAL ENVIRONMENTAL STATEMENT' on cover.
    Updated regex catches it (Impact word optional). Should produce Final
    via regex, no LLM call needed."""
    if not FIXTURE_PATH.exists():
        pytest.skip(f"fixture not present: {FIXTURE_PATH}")

    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    _, raw_text = next(iter(payload.items()))

    art = _build_artifact(raw_text, page_size=config.CHARS_PER_FAKE_PAGE)
    sections_art = _build_sections_with_cover(art)
    record = _new_record(sections_art)

    warnings = stage3b_run(record, art, sections_art, llm=None)

    assert record.eis_type.value == "Final"
    assert record.eis_type.provenance.source == "regex"
    assert warnings == []
