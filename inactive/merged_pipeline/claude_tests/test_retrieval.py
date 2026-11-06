"""
Retrieval primitives + cascade tests.
"""

from __future__ import annotations

from pipeline.ingest import FakePage, IngestArtifact
from pipeline.retrieval import (
    SOURCE_FALLBACK_FIRST_N,
    SOURCE_FALLBACK_KEYWORD,
    cascade,
    get_first_n_chunks,
    get_keyword_windows,
    get_section_text,
)
from pipeline.sections import DetectedSection, SectionsArtifact


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
        publication_id="t",
        physical_record_ids=["t"],
        raw_text=raw,
        text_normalized=raw,
        alignment_map=[],
        pages=pages,
        nul_metadata={},
        raw_text_hash="sha256:t",
    )


def _sections_with(name: str, char_span: tuple[int, int],
                   detection_method: str = "regex", status: str = "ok") -> SectionsArtifact:
    return SectionsArtifact(
        publication_id="t",
        sections=[DetectedSection(
            name=name, char_span=char_span, pages=(1, 1),
            confidence=0.9, status=status, detection_method=detection_method,
        )],
    )


# ---------------------------------------------------------------------------
# get_section_text
# ---------------------------------------------------------------------------

def test_get_section_text_returns_window_for_ok_section() -> None:
    raw = "0123456789ABCDEFGHIJ"
    art = _build_artifact(raw)
    sections = _sections_with("summary", (5, 15))
    win = get_section_text(art, sections, "summary")
    assert win is not None
    assert win.text == raw[5:15]
    assert win.char_offset_raw == (5, 15)


def test_get_section_text_returns_none_when_section_missing() -> None:
    art = _build_artifact("hello")
    sections = SectionsArtifact(publication_id="t", sections=[])
    assert get_section_text(art, sections, "summary") is None


def test_get_section_text_returns_none_for_not_found_status() -> None:
    art = _build_artifact("hello")
    sections = SectionsArtifact(
        publication_id="t",
        sections=[DetectedSection(name="summary", status="not_found")],
    )
    assert get_section_text(art, sections, "summary") is None


# ---------------------------------------------------------------------------
# get_keyword_windows
# ---------------------------------------------------------------------------

def test_keyword_windows_finds_hits_with_surrounding_context() -> None:
    raw = "X" * 500 + " environmental impact statement " + "Y" * 500
    art = _build_artifact(raw)
    wins = get_keyword_windows(art, ["environmental"], window_chars=200, max_windows=4)
    assert len(wins) == 1
    assert "environmental" in wins[0].text.lower()
    s, e = wins[0].char_offset_raw
    assert e - s <= 250  # window_chars/2 each side + the keyword


def test_keyword_windows_coalesces_overlapping_hits() -> None:
    """Two keyword hits within the same paragraph -> one merged window."""
    raw = "intro " + "filler " * 100 + " purpose and need " + "filler " * 5 + " proposed action " + "filler " * 100
    art = _build_artifact(raw)
    wins = get_keyword_windows(art, ["purpose", "proposed"], window_chars=4000, max_windows=8)
    # Both hits are within ~50 chars; window_chars=4000 ensures overlap -> coalesced.
    assert len(wins) == 1
    assert "purpose" in wins[0].text.lower()
    assert "proposed" in wins[0].text.lower()


def test_keyword_windows_no_hits_returns_empty() -> None:
    raw = "completely unrelated prose"
    art = _build_artifact(raw)
    assert get_keyword_windows(art, ["xyzzy"]) == []


def test_keyword_windows_respects_max_windows() -> None:
    raw = " ".join(["hit" + " filler" * 200 for _ in range(10)])
    art = _build_artifact(raw)
    wins = get_keyword_windows(art, ["hit"], window_chars=10, max_windows=3)
    assert len(wins) <= 3


# ---------------------------------------------------------------------------
# get_first_n_chunks
# ---------------------------------------------------------------------------

def test_first_n_chunks_caps_to_max_chars() -> None:
    raw = "X" * 30_000
    art = _build_artifact(raw)
    wins = get_first_n_chunks(art, max_chars=10_000)
    assert len(wins) == 1
    assert wins[0].char_offset_raw == (0, 10_000)
    assert len(wins[0]) == 10_000


def test_first_n_chunks_handles_empty_doc() -> None:
    art = _build_artifact("")
    assert get_first_n_chunks(art) == []


# ---------------------------------------------------------------------------
# cascade
# ---------------------------------------------------------------------------

def test_cascade_uses_primary_section_when_present() -> None:
    raw = "0123456789" + "PURPOSE: X" * 100
    art = _build_artifact(raw)
    sections = _sections_with("purpose_and_need", (0, 100), detection_method="regex")
    res = cascade(
        art, sections,
        primary_sections=["purpose_and_need"],
        fallback_keywords=["purpose"],
    )
    assert res.degraded is False
    assert res.section == "purpose_and_need"
    assert res.provenance_source == "regex"
    assert len(res.windows) == 1
    assert res.windows[0].char_offset_raw == (0, 100)


def test_cascade_falls_through_to_keyword_when_section_missing() -> None:
    raw = "no headings here at all but mentioning purpose somewhere in the middle"
    art = _build_artifact(raw)
    sections = SectionsArtifact(publication_id="t", sections=[])
    res = cascade(
        art, sections,
        primary_sections=["purpose_and_need"],
        fallback_keywords=["purpose"],
        keyword_window_chars=200,
    )
    assert res.degraded is True
    assert res.provenance_source == SOURCE_FALLBACK_KEYWORD
    assert res.section is None
    assert len(res.windows) >= 1


def test_cascade_falls_through_to_first_n_when_keyword_misses() -> None:
    raw = "absolutely nothing useful in this short doc"
    art = _build_artifact(raw)
    sections = SectionsArtifact(publication_id="t", sections=[])
    res = cascade(
        art, sections,
        primary_sections=["summary"],
        fallback_keywords=["xyzzy"],
        first_n_max_chars=20,
    )
    assert res.degraded is True
    assert res.provenance_source == SOURCE_FALLBACK_FIRST_N
    assert res.section is None
    assert len(res.windows) == 1
    assert res.windows[0].char_offset_raw == (0, 20)


def test_cascade_skips_not_found_sections() -> None:
    """A section with status='not_found' must not be used; cascade falls through."""
    raw = "doc with the word purpose somewhere in it"
    art = _build_artifact(raw)
    sections = SectionsArtifact(
        publication_id="t",
        sections=[DetectedSection(name="purpose_and_need", status="not_found")],
    )
    res = cascade(
        art, sections,
        primary_sections=["purpose_and_need"],
        fallback_keywords=["purpose"],
    )
    assert res.degraded is True


def test_cascade_section_detection_method_maps_to_provenance_source() -> None:
    """Section detected via AI-TOC -> provenance source 'haiku_classifier'."""
    raw = "section content"
    art = _build_artifact(raw)
    sections = _sections_with("summary", (0, len(raw)), detection_method="ai_toc")
    res = cascade(
        art, sections,
        primary_sections=["summary"],
        fallback_keywords=[],
    )
    assert res.degraded is False
    assert res.provenance_source == "haiku_classifier"


def test_cascade_combined_text_joins_windows() -> None:
    # Use real word boundaries (\b) for the keyword regex to bind. Padding
    # with periods + spaces creates valid non-word boundaries on each side
    # of "purpose" and "proposed".
    pad = ". " * 100
    raw = pad + "purpose" + pad + pad + "proposed" + pad
    # Place hits far enough apart that windows (20 chars) don't coalesce
    art = _build_artifact(raw)
    sections = SectionsArtifact(publication_id="t", sections=[])
    res = cascade(
        art, sections,
        primary_sections=["summary"],
        fallback_keywords=["purpose", "proposed"],
        keyword_window_chars=20,
        max_keyword_windows=4,
    )
    assert res.degraded is True
    # Should have 2 windows
    assert len(res.windows) == 2
    combined = res.combined_text
    assert "WINDOW BOUNDARY" in combined  # separator present
