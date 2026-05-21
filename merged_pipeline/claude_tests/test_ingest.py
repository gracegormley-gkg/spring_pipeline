"""
Stage 1 ingest tests — normalization + alignment map + fake page splitting.

Run from merged_pipeline/:
    pytest tests/test_ingest.py -v

Ported verbatim from v1_multiagent_pipeline/tests/test_ingest.py. The
algorithms haven't changed in the merged port; if any of these regress it
means we accidentally diverged from the v1_multi alignment behavior that
the schema's verbatim-quote gate depends on.
"""

from __future__ import annotations

from pipeline.ingest import (
    AlignmentSpan,  # noqa: F401  (re-exported for downstream importers)
    _normalize_with_alignment,
    _split_into_fake_pages,
    char_to_page,
    page_range_for_span,
)


# ---------------------------------------------------------------------------
# Normalization + alignment
# ---------------------------------------------------------------------------

def test_identity_normalization() -> None:
    raw = "The quick brown fox jumps over the lazy dog."
    norm, spans = _normalize_with_alignment(raw)
    assert norm == raw
    # Whole string should be one identity span
    assert len(spans) == 1
    s = spans[0]
    assert s.norm_start == 0 and s.norm_end == len(raw)
    assert s.raw_start == 0 and s.raw_end == len(raw)


def test_smart_quote_substitution() -> None:
    raw = "He said \u201chello\u201d to her."   # curly quotes
    norm, spans = _normalize_with_alignment(raw)
    assert norm == 'He said "hello" to her.'
    # Each substitution should produce a single-char-to-single-char span
    sub_spans = [s for s in spans if s.raw_end - s.raw_start == 1 and s.norm_end - s.norm_start == 1]
    assert len(sub_spans) >= 2  # at least the two curly quotes


def test_soft_hyphen_dehyphenation() -> None:
    raw = "envir-\nonmental impact"
    norm, spans = _normalize_with_alignment(raw)
    assert norm == "environmental impact"
    # The 'env-\n' chunk got collapsed to 'envir' so there must be a discontinuity
    # between the prefix span and the suffix span.
    raw_starts = [s.raw_start for s in spans]
    raw_ends = [s.raw_end for s in spans]
    # Some span should END at position 5 (just before the '-'),
    # and the next should START past the newline.
    assert any(re == 5 for re in raw_ends), spans
    assert any(rs >= 7 for rs in raw_starts), spans  # past '-\n'


def test_alignment_map_round_trip() -> None:
    """
    For every char in normalized text, we should be able to reconstruct the raw
    char that produced it (or know it came from a substitution).
    """
    raw = "hello \u201cworld\u201d"
    norm, spans = _normalize_with_alignment(raw)

    # Walk through each span and check that norm[norm_start:norm_end] either
    # equals raw[raw_start:raw_end] (identity) or is a known substitution.
    for sp in spans:
        norm_chunk = norm[sp.norm_start:sp.norm_end]
        raw_chunk = raw[sp.raw_start:sp.raw_end]
        if norm_chunk == raw_chunk:
            continue
        # Else expect a substitution: 1 raw char → 1+ norm chars
        assert sp.raw_end - sp.raw_start == 1


# ---------------------------------------------------------------------------
# Fake page splitting
# ---------------------------------------------------------------------------

def test_fake_page_splitting_basic() -> None:
    # Build a text with known line lengths
    line = "x" * 100 + "\n"  # 101 chars
    raw = line * 30          # 3030 chars total
    pages = _split_into_fake_pages(raw, chars_per_page=500)
    # Every page should hit ~500 chars (rounded up by line)
    assert len(pages) >= 5
    assert pages[0].char_start_raw == 0
    # Pages should tile the raw text contiguously
    for i in range(1, len(pages)):
        assert pages[i].char_start_raw == pages[i - 1].char_end_raw
    # Final page should cover the end of the text
    assert pages[-1].char_end_raw == len(raw)


def test_char_to_page_lookup() -> None:
    raw = ("alpha line\n" * 100)  # 1100 chars
    pages = _split_into_fake_pages(raw, chars_per_page=200)
    # Char 0 → page 1
    assert char_to_page(pages, 0) == 1
    # Mid-doc → some page > 1
    mid = char_to_page(pages, 500)
    assert mid is not None and mid >= 1
    # Past end → None
    assert char_to_page(pages, len(raw) + 100) is None


def test_page_range_for_span() -> None:
    raw = ("X" * 50 + "\n") * 20  # 1020 chars, 20 short lines
    pages = _split_into_fake_pages(raw, chars_per_page=200)
    span = (10, 100)
    pr = page_range_for_span(pages, span)
    assert pr is not None
    assert pr[0] >= 1 and pr[1] >= pr[0]
