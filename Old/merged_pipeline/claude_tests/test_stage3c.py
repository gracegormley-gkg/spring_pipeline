"""
Stage 3c summary tests.
"""

from __future__ import annotations

from pipeline.ingest import FakePage, IngestArtifact
from pipeline.schema import EISRecord, FieldWithStatus, SectionRecord
from pipeline.sections import DetectedSection, SectionsArtifact
from pipeline.stage3c_summary import run as stage3c_run


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
        publication_id="t", physical_record_ids=["t"],
        raw_text=raw, text_normalized=raw, alignment_map=[],
        pages=pages, nul_metadata={}, raw_text_hash="sha256:t",
    )


def _sections_with_summary(start: int, end: int) -> SectionsArtifact:
    return SectionsArtifact(
        publication_id="t",
        sections=[DetectedSection(
            name="summary", char_span=(start, end), pages=(1, 1),
            confidence=0.9, status="ok", detection_method="regex",
        )],
    )


def _new_record(sections_artifact: SectionsArtifact, *, with_title: bool = True) -> EISRecord:
    sections = [s.to_schema_record() for s in sections_artifact.sections]
    rec = EISRecord(publication_id="t", sections=sections)
    if with_title:
        rec.title = FieldWithStatus[str](value="Test EIS", status="ok")
    return rec


class FakeLLM:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.dry_run = False
        self.models = {"haiku": "haiku", "sonnet": "sonnet", "opus": "opus"}

    def call_json(self, model, system, messages, max_tokens=2048, temperature=0.2, label=""):
        self.calls.append({"model": model, "label": label})
        if not self.responses:
            raise RuntimeError(f"FakeLLM exhausted (label={label!r})")
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# Happy path: section retrieval -> Opus -> Haiku layman
# ---------------------------------------------------------------------------

def test_summary_two_pass_happy_path() -> None:
    raw = "X" * 100 + " summary content here " + "Y" * 100
    art = _build_artifact(raw)
    sections = _sections_with_summary(0, len(raw))
    record = _new_record(sections)

    fake_llm = FakeLLM([
        {"summary": "Detailed summary of the project.", "sufficient_information": True},
        {"layman_summary": "Plain language version."},
    ])

    warnings = stage3c_run(record, art, sections, llm=fake_llm)

    assert record.summary.value == "Detailed summary of the project."
    assert record.summary.status == "ok"
    assert record.summary.provenance.source == "regex"  # section detected via regex
    assert record.summary.provenance.section == "summary"

    assert record.layman_summary.value == "Plain language version."
    assert record.layman_summary.status == "ok"
    assert record.layman_summary.provenance.source == "haiku_classifier"
    # Layman is a rewrite; section should not be set
    assert record.layman_summary.provenance.section is None

    assert len(fake_llm.calls) == 2
    assert warnings == []


# ---------------------------------------------------------------------------
# Degraded retrieval -> needs_review
# ---------------------------------------------------------------------------

def test_summary_degraded_retrieval_marks_needs_review() -> None:
    raw = "no summary section but mentioning purpose somewhere"
    art = _build_artifact(raw)
    # No sections detected
    sections = SectionsArtifact(publication_id="t", sections=[])
    record = _new_record(sections)

    fake_llm = FakeLLM([
        {"summary": "Summary from keyword fallback.", "sufficient_information": True},
        {"layman_summary": "Layman version."},
    ])

    warnings = stage3c_run(record, art, sections, llm=fake_llm)

    assert record.summary.value == "Summary from keyword fallback."
    assert record.summary.status == "needs_review"
    assert record.summary.provenance.source == "fallback_keyword_search"
    assert any("retrieval degraded" in w for w in warnings)


def test_summary_first_n_fallback_when_section_and_keywords_miss() -> None:
    raw = "tiny doc no keywords"
    art = _build_artifact(raw)
    sections = SectionsArtifact(publication_id="t", sections=[])
    record = _new_record(sections)

    # Cascade falls through: no section, "purpose"/"proposed" not present in raw.
    fake_llm = FakeLLM([
        {"summary": "First N chars summary.", "sufficient_information": True},
        {"layman_summary": "Layman."},
    ])

    warnings = stage3c_run(record, art, sections, llm=fake_llm)
    assert record.summary.provenance.source == "fallback_first_n_chunks"
    assert record.summary.status == "needs_review"


# ---------------------------------------------------------------------------
# Insufficient info / LLM failure paths
# ---------------------------------------------------------------------------

def test_summary_insufficient_information() -> None:
    raw = "stub content"
    art = _build_artifact(raw)
    sections = _sections_with_summary(0, len(raw))
    record = _new_record(sections)

    fake_llm = FakeLLM([
        {"summary": None, "sufficient_information": False},
    ])

    warnings = stage3c_run(record, art, sections, llm=fake_llm)
    assert record.summary.value is None
    assert record.summary.status == "needs_review"
    # Layman is skipped (no detailed text to rewrite)
    assert record.layman_summary.value is None


def test_summary_llm_call_raises() -> None:
    class RaisingLLM:
        dry_run = False
        models = {"haiku": "haiku", "sonnet": "sonnet", "opus": "opus"}
        def call_json(self, *a, **k):
            raise RuntimeError("simulated outage")

    raw = "content"
    art = _build_artifact(raw)
    sections = _sections_with_summary(0, len(raw))
    record = _new_record(sections)

    warnings = stage3c_run(record, art, sections, llm=RaisingLLM())
    assert record.summary.value is None
    assert record.summary.status == "needs_review"


def test_summary_no_llm_provided() -> None:
    raw = "content"
    art = _build_artifact(raw)
    sections = _sections_with_summary(0, len(raw))
    record = _new_record(sections)

    warnings = stage3c_run(record, art, sections, llm=None)
    assert record.summary.value is None
    assert record.summary.status == "needs_review"
    assert record.layman_summary.value is None
    assert record.layman_summary.status == "needs_review"


def test_summary_layman_failure_keeps_detailed() -> None:
    """If layman call fails, detailed summary should still ship."""
    raw = "content"
    art = _build_artifact(raw)
    sections = _sections_with_summary(0, len(raw))
    record = _new_record(sections)

    fake_llm = FakeLLM([
        {"summary": "Detailed.", "sufficient_information": True},
        # Second call (layman) fails because exhausted
    ])

    warnings = stage3c_run(record, art, sections, llm=fake_llm)
    assert record.summary.value == "Detailed."
    assert record.summary.status == "ok"
    assert record.layman_summary.value is None
    assert record.layman_summary.status == "needs_review"


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------

def test_summary_schema_round_trip() -> None:
    raw = "summary section content"
    art = _build_artifact(raw)
    sections = _sections_with_summary(0, len(raw))
    record = _new_record(sections)

    fake_llm = FakeLLM([
        {"summary": "Detailed summary.", "sufficient_information": True},
        {"layman_summary": "Layman."},
    ])
    stage3c_run(record, art, sections, llm=fake_llm)

    # Should validate cleanly
    EISRecord.model_validate(record.model_dump())
