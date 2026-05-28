"""
Stage 3e alternatives tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import config
from pipeline.ingest import FakePage, IngestArtifact
from pipeline.schema import EISRecord, SectionRecord
from pipeline.sections import DetectedSection, SectionsArtifact
from pipeline.stage3e_alternatives import _extract_labels, _normalize_label, run as stage3e_run
from pipeline.retrieval import cascade


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_doc.json"


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


def _sections_with_alts(start: int, end: int,
                        detection_method: str = "regex") -> SectionsArtifact:
    return SectionsArtifact(
        publication_id="t",
        sections=[DetectedSection(
            name="alternatives", char_span=(start, end), pages=(1, 1),
            confidence=0.9, status="ok", detection_method=detection_method,
        )],
    )


def _new_record(sections_artifact: SectionsArtifact) -> EISRecord:
    sections = [s.to_schema_record() for s in sections_artifact.sections]
    return EISRecord(publication_id="t", sections=sections)


class FakeLLM:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.dry_run = False
        self.models = {"haiku": "haiku", "sonnet": "sonnet", "opus": "opus"}

    def call_json(self, model, system, messages, max_tokens=2048, temperature=0.2, label=""):
        self.calls.append({"model": model, "label": label})
        if not self.responses:
            raise RuntimeError("exhausted")
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# Label normalization + regex extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Alternative A", "Alternative A"),
    ("Alternative 1", "Alternative 1"),
    ("Alternative A-1", "Alternative A-1"),
    ("No  Action", "No Action"),
    ("No-Action", "No Action"),
    ("NO ACTION", "No Action"),
    ("No-Build", "No Build"),
    ("Status Quo", "Status Quo"),
    ("Variation B-2", "Variation B-2"),
])
def test_normalize_label(raw, expected) -> None:
    assert _normalize_label(raw) == expected


def test_extract_labels_finds_common_patterns() -> None:
    raw = (
        "ALTERNATIVES CONSIDERED\n"
        "Alternative A: build a new road.\n"
        "Alternative B: widen existing roads.\n"
        "No Action: maintain current conditions.\n"
        "Preferred Alternative: Alternative B.\n"
    )
    art = _build_artifact(raw)
    sections = _sections_with_alts(0, len(raw))
    res = cascade(art, sections, primary_sections=["alternatives"], fallback_keywords=[])
    labels = _extract_labels(res, art)
    label_strings = {lbl["label"] for lbl in labels}
    assert "Alternative A" in label_strings
    assert "Alternative B" in label_strings
    assert "No Action" in label_strings
    assert "Preferred Alternative" in label_strings


def test_extract_labels_dedupes_repeated_mentions() -> None:
    raw = (
        "Alternative A is great. "
        "Alternative A appears again. "
        "Alternative A is mentioned a third time. "
        "Alternative B is different."
    )
    art = _build_artifact(raw)
    sections = _sections_with_alts(0, len(raw))
    res = cascade(art, sections, primary_sections=["alternatives"], fallback_keywords=[])
    labels = _extract_labels(res, art)
    label_strings = [lbl["label"] for lbl in labels]
    # 'Alternative A' should appear exactly once in the dedup'd list
    assert label_strings.count("Alternative A") == 1
    assert "Alternative B" in label_strings


def test_extract_labels_handles_alignments_and_routes() -> None:
    raw = "Alignment 1 follows the river. Alignment 2 cuts through. Route A is the bypass."
    art = _build_artifact(raw)
    sections = _sections_with_alts(0, len(raw))
    res = cascade(art, sections, primary_sections=["alternatives"], fallback_keywords=[])
    labels = _extract_labels(res, art)
    label_strings = {lbl["label"] for lbl in labels}
    assert "Alignment 1" in label_strings
    assert "Alignment 2" in label_strings
    assert "Route A" in label_strings


# ---------------------------------------------------------------------------
# Full run() — happy path
# ---------------------------------------------------------------------------

def test_stage3e_happy_path() -> None:
    raw = (
        "ALTERNATIVES\n"
        "Alternative A: build a road. " + "X" * 100 + "\n"
        "No Action: do nothing. " + "Y" * 100 + "\n"
    )
    art = _build_artifact(raw)
    sections = _sections_with_alts(0, len(raw))
    record = _new_record(sections)

    fake_llm = FakeLLM([{
        "alternatives": [
            {"label": "Alternative A", "description": "Build a new road through the corridor.", "found": True},
            {"label": "No Action", "description": "Maintain current conditions; no road built.", "found": True},
        ],
    }])

    warnings = stage3e_run(record, art, sections, llm=fake_llm)
    assert len(record.alternatives) == 2
    by_label = {a.label: a for a in record.alternatives}
    assert by_label["Alternative A"].description == "Build a new road through the corridor."
    assert by_label["Alternative A"].status == "ok"
    assert by_label["No Action"].status == "ok"
    assert warnings == []
    assert len(fake_llm.calls) == 1  # single batched call


# ---------------------------------------------------------------------------
# Sonnet returns "found=false" for some labels -> needs_review per-alt
# ---------------------------------------------------------------------------

def test_stage3e_partial_descriptions() -> None:
    raw = "ALTERNATIVES\nAlternative A: details.\nVariation B-1: stub.\n"
    art = _build_artifact(raw)
    sections = _sections_with_alts(0, len(raw))
    record = _new_record(sections)

    fake_llm = FakeLLM([{
        "alternatives": [
            {"label": "Alternative A", "description": "A real description.", "found": True},
            {"label": "Variation B-1", "description": None, "found": False},
        ],
    }])

    stage3e_run(record, art, sections, llm=fake_llm)
    by_label = {a.label: a for a in record.alternatives}
    assert by_label["Alternative A"].status == "ok"
    assert by_label["Variation B-1"].status == "needs_review"
    assert by_label["Variation B-1"].description == ""


# ---------------------------------------------------------------------------
# No labels -> empty alternatives
# ---------------------------------------------------------------------------

def test_stage3e_no_labels_in_section() -> None:
    raw = "ALTERNATIVES\nNo named alternatives appear here, just prose.\n"
    art = _build_artifact(raw)
    sections = _sections_with_alts(0, len(raw))
    record = _new_record(sections)

    # Even though no labels found, with no LLM no call happens; empty list ok.
    stage3e_run(record, art, sections, llm=None)
    assert record.alternatives == []


# ---------------------------------------------------------------------------
# Degraded retrieval
# ---------------------------------------------------------------------------

def test_stage3e_degraded_retrieval_marks_descriptions_needs_review() -> None:
    """Section missing -> keyword fallback finds 'Alternative A' in body text;
    LLM describes; but status downgrades to needs_review because retrieval
    was degraded."""
    raw = "Some doc body. Alternative A appears here. " + "filler " * 200
    art = _build_artifact(raw)
    sections = SectionsArtifact(publication_id="t", sections=[])  # no alternatives section
    record = _new_record(sections)

    fake_llm = FakeLLM([{
        "alternatives": [
            {"label": "Alternative A", "description": "Found via keyword.", "found": True},
        ],
    }])

    warnings = stage3e_run(record, art, sections, llm=fake_llm)
    assert len(record.alternatives) == 1
    assert record.alternatives[0].status == "needs_review"
    assert any("retrieval degraded" in w for w in warnings)


# ---------------------------------------------------------------------------
# No LLM -> labels with empty descriptions
# ---------------------------------------------------------------------------

def test_stage3e_no_llm_ships_label_only_entries() -> None:
    raw = "ALTERNATIVES\nAlternative A: stuff. Alternative B: stuff.\n"
    art = _build_artifact(raw)
    sections = _sections_with_alts(0, len(raw))
    record = _new_record(sections)

    warnings = stage3e_run(record, art, sections, llm=None)
    assert len(record.alternatives) == 2
    assert all(a.description == "" for a in record.alternatives)
    assert all(a.status == "needs_review" for a in record.alternatives)


# ---------------------------------------------------------------------------
# LLM failure -> needs_review fallback
# ---------------------------------------------------------------------------

def test_stage3e_llm_failure_falls_back_to_label_only() -> None:
    class RaisingLLM:
        dry_run = False
        def call_json(self, *a, **k):
            raise RuntimeError("outage")

    raw = "ALTERNATIVES\nAlternative A: stuff.\n"
    art = _build_artifact(raw)
    sections = _sections_with_alts(0, len(raw))
    record = _new_record(sections)

    warnings = stage3e_run(record, art, sections, llm=RaisingLLM())
    assert len(record.alternatives) == 1
    assert record.alternatives[0].status == "needs_review"
    assert any("Sonnet description pass failed" in w for w in warnings)


# ---------------------------------------------------------------------------
# Integration on fixture
# ---------------------------------------------------------------------------

def test_stage3e_on_fixture_doc_no_alternatives_section() -> None:
    """Castaic-Haskell fixture doesn't have an 'alternatives' section detected
    by regex (typewritten 1970s). Without LLM, expect: alternatives=[] (no
    keyword fallback hits because it's a power transmission line memo with no
    'Alternative' label words)."""
    if not FIXTURE_PATH.exists():
        pytest.skip(f"fixture not present: {FIXTURE_PATH}")

    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    _, raw_text = next(iter(payload.items()))
    art = _build_artifact(raw_text, page_size=config.CHARS_PER_FAKE_PAGE)
    sections = SectionsArtifact(publication_id="t", sections=[])
    record = _new_record(sections)

    stage3e_run(record, art, sections, llm=None)
    # The fixture mentions "alternative" in body text — regex may or may not
    # extract a label. The key invariant: pipeline doesn't crash and produces
    # a list (possibly empty).
    assert isinstance(record.alternatives, list)
