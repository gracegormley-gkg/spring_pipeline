"""
Sections tests — regex pass on a fixture-style text plus reject patterns.

Run: pytest tests/test_sections.py
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.ingest import FakePage, IngestArtifact
from pipeline.sections import _is_rejected_match, _regex_detect, run as sections_run
from pipeline import config


def _build_artifact(raw: str, page_size: int = 1000) -> IngestArtifact:
    """Tiny IngestArtifact stub for section tests."""
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


def test_regex_pass_finds_basic_sections() -> None:
    raw = (
        "EIS Cover Page\n"
        "\n"
        "SUMMARY\n"
        "Some summary text...\n"
        "\n"
        "PURPOSE AND NEED\n"
        "More text here...\n"
        "\n"
        "ALTERNATIVES\n"
        "Alternative descriptions...\n"
        "\n"
        "ENVIRONMENTAL CONSEQUENCES\n"
        "Impacts...\n"
    )
    art = _build_artifact(raw)
    result = sections_run(art, allow_embedding_fallback=False)

    by_name = result.by_name()
    assert by_name["summary"].status == "ok"
    assert by_name["purpose_and_need"].status == "ok"
    assert by_name["alternatives"].status == "ok"
    assert by_name["environmental_consequences"].status == "ok"
    # Cover should always be populated by default-pages logic
    assert by_name["cover"].status == "ok"
    assert by_name["cover"].detection_method == "default_pages"


def test_reject_pattern_zip_code() -> None:
    """A 'SUMMARY' that's adjacent to a ZIP code shouldn't match (looks like address)."""
    raw = "60601\n\nSUMMARY\nthis is fake — address-context match\n"
    art = _build_artifact(raw)
    result = sections_run(art, allow_embedding_fallback=False)
    by_name = result.by_name()
    # The reject pattern triggers on the surrounding ±60 chars window.
    # Confirm we don't accept it.
    assert by_name["summary"].status == "not_found"


def test_reject_pattern_legal_citation() -> None:
    """Legal citations like 'Section 4(f)' must not be detected as headings."""
    raw = "...the Section 4(f) ALTERNATIVES discussion follows...\n"
    art = _build_artifact(raw)
    result = sections_run(art, allow_embedding_fallback=False)
    # Whether it matches at all depends on where the line break falls; key
    # invariant: ALTERNATIVES embedded in prose with 'Section 4(f)' nearby
    # should be rejected.
    by_name = result.by_name()
    if by_name["alternatives"].status == "ok":
        # If we did find it, it must NOT be on the line containing 'Section 4(f)'
        s = by_name["alternatives"].char_span[0]  # type: ignore[index]
        window = raw[max(0, s - 30):s + 30]
        assert "Section 4(f)" not in window, f"matched in legal citation window: {window!r}"


def test_not_found_for_missing_section() -> None:
    raw = "Just some prose. No section markers here at all.\n"
    art = _build_artifact(raw)
    result = sections_run(art, allow_embedding_fallback=False)
    by_name = result.by_name()
    # All non-cover sections should be not_found
    for name in config.CEQ_SECTIONS:
        if name == "cover":
            continue
        assert by_name[name].status == "not_found", f"{name} unexpectedly found"
