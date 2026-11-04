"""
Stage 2 tests — section detection cascade.

Coverage:
  - Regex pass: basic sections + reject patterns (ZIP, legal citation, address)
  - Cover defaults
  - AI-TOC dispatch (with a stubbed LLM client; no network)
  - CEQ-name normalizer (match_title_to_ceq)
  - Cascade orchestration: AI-TOC silently no-ops without llm_client; embedding
    silently no-ops when allow_embedding_fallback=False
  - Integration: real fixture doc -> cover only, rest not_found (typewritten
    1970s; no sections match the regex; no LLM provided)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import config
from pipeline import ingest as ingest_mod
from pipeline.ingest import FakePage, IngestArtifact
from pipeline.sections import (
    DetectedSection,
    SectionsArtifact,
    _is_rejected_match,
    _regex_detect,
    match_title_to_ceq,
    run as sections_run,
)
from pipeline.sections_ai_toc import _find_anchor, _sample_for_toc, ai_toc_detect


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_doc.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_artifact(raw: str, page_size: int = 1000) -> IngestArtifact:
    """Tiny IngestArtifact stub for section tests. Page size is small so we
    get multi-page artifacts even on short fixtures."""
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


class FakeLLM:
    """Stub LLM client. `call_json` returns canned data per call.

    Each entry in `responses` is consumed once, in FIFO order. If `responses`
    is empty when called, raises RuntimeError so tests catch unexpected calls.
    """

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.models = {"haiku": "haiku", "sonnet": "sonnet", "opus": "opus"}

    def call_json(self, model, system, messages, max_tokens=2048, temperature=0.2, label=""):
        self.calls.append({
            "model": model, "label": label, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
        })
        if not self.responses:
            raise RuntimeError(f"FakeLLM exhausted; unexpected call (label={label!r})")
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# Regex pass
# ---------------------------------------------------------------------------

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
    result = sections_run(art, allow_ai_toc=False, allow_embedding_fallback=False)

    by_name = result.by_name()
    assert by_name["summary"].status == "ok"
    assert by_name["summary"].detection_method == "regex"
    assert by_name["purpose_and_need"].status == "ok"
    assert by_name["alternatives"].status == "ok"
    assert by_name["environmental_consequences"].status == "ok"
    # Cover always populated by default-pages logic
    assert by_name["cover"].status == "ok"
    assert by_name["cover"].detection_method == "default_pages"


def test_reject_pattern_zip_code() -> None:
    """A 'SUMMARY' adjacent to a ZIP code shouldn't match (looks like address)."""
    raw = "60601\n\nSUMMARY\nthis is fake — address-context match\n"
    art = _build_artifact(raw)
    result = sections_run(art, allow_ai_toc=False, allow_embedding_fallback=False)
    assert result.by_name()["summary"].status == "not_found"


def test_reject_pattern_legal_citation() -> None:
    """Legal citations like 'Section 4(f)' must not be detected as headings."""
    raw = "...the Section 4(f) ALTERNATIVES discussion follows...\n"
    art = _build_artifact(raw)
    result = sections_run(art, allow_ai_toc=False, allow_embedding_fallback=False)
    by_name = result.by_name()
    if by_name["alternatives"].status == "ok":
        s = by_name["alternatives"].char_span[0]  # type: ignore[index]
        window = raw[max(0, s - 30):s + 30]
        assert "Section 4(f)" not in window, f"matched in legal citation window: {window!r}"


def test_not_found_for_missing_section() -> None:
    raw = "Just some prose. No section markers here at all.\n"
    art = _build_artifact(raw)
    result = sections_run(art, allow_ai_toc=False, allow_embedding_fallback=False)
    by_name = result.by_name()
    for name in config.CEQ_SECTIONS:
        if name == "cover":
            continue
        assert by_name[name].status == "not_found", f"{name} unexpectedly found"


# ---------------------------------------------------------------------------
# CEQ-name normalizer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title,expected", [
    ("PURPOSE AND NEED", "purpose_and_need"),
    ("Need for the Action", "purpose_and_need"),
    ("Alternatives Considered", "alternatives"),
    ("ALTERNATIVES TO THE PROPOSED ACTION", "alternatives"),
    ("Affected Environment", "affected_environment"),
    ("Environmental Consequences", "environmental_consequences"),
    ("Public Comments", "public_comments"),
    ("Comments and Responses", "response_to_comments"),
    ("Record of Decision", "rod"),
    ("Summary", "summary"),
    ("EXECUTIVE SUMMARY", "summary"),
    # Doesn't map: random heading
    ("Acknowledgments", None),
    # Doesn't map: empty
    ("", None),
])
def test_match_title_to_ceq(title, expected) -> None:
    assert match_title_to_ceq(title) == expected


# ---------------------------------------------------------------------------
# AI-TOC pass (with stubbed LLM)
# ---------------------------------------------------------------------------

def test_ai_toc_finds_typewritten_summary() -> None:
    """Regex misses 'SUMMARY SHEET' (no $ anchor); AI-TOC should locate it via
    anchor phrase and map title -> 'summary'."""
    raw = (
        "U.S. DEPARTMENT OF AGRICULTURE\n"
        "FINAL ENVIRONMENTAL STATEMENT\n"
        "SUMMARY SHEET\n"
        "This is an administrative summary describing the proposed project to "
        "build a power transmission line through the national forest.\n"
        "\n"
        "Body of the document follows here with lots of words to make it not\n"
        "tiny, ensuring the AI-TOC sample logic doesn't trim too aggressively.\n"
    )
    art = _build_artifact(raw)
    fake_llm = FakeLLM([{
        "sections": [
            {
                "title": "SUMMARY SHEET",
                "anchor_phrase": "This is an administrative summary describing the proposed",
            },
        ]
    }])
    result = sections_run(art, llm_client=fake_llm, allow_embedding_fallback=False)
    by_name = result.by_name()
    assert by_name["summary"].status == "ok"
    assert by_name["summary"].detection_method == "ai_toc"
    assert by_name["summary"].char_span is not None
    s, e = by_name["summary"].char_span
    # The anchor phrase (or a case-insensitive variant) should sit at that offset
    assert "administrative summary" in raw[s:s + 200].lower()
    # Exactly one LLM call was made
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["model"] == "haiku"


def test_ai_toc_silently_skipped_without_llm_client() -> None:
    """No llm_client -> AI-TOC pass not attempted; only regex + cover run."""
    raw = "SUMMARY SHEET\nNot regex-matchable.\n"
    art = _build_artifact(raw)
    # No FakeLLM passed; no exception expected.
    result = sections_run(art, allow_embedding_fallback=False)
    assert result.by_name()["summary"].status == "not_found"


def test_ai_toc_drops_unmappable_title() -> None:
    """If the LLM returns a title we can't map to a CEQ name, drop it."""
    raw = "Some prose with no detectable structure at all.\n" * 10
    art = _build_artifact(raw)
    fake_llm = FakeLLM([{
        "sections": [
            {"title": "Acknowledgments", "anchor_phrase": "Some prose with no"},
        ]
    }])
    result = sections_run(art, llm_client=fake_llm, allow_embedding_fallback=False)
    # Acknowledgments doesn't map; nothing should be detected.
    for name in config.CEQ_SECTIONS:
        if name == "cover":
            continue
        assert result.by_name()[name].status == "not_found"


def test_ai_toc_skipped_if_regex_already_found() -> None:
    """AI-TOC must not overwrite a regex hit — regex confidence is higher."""
    raw = (
        "SUMMARY\nThe regex catches this heading.\n"
        + "more text here so cascade considers it long enough\n" * 20
    )
    art = _build_artifact(raw)
    # FakeLLM has no responses queued. If AI-TOC tries to call it,
    # FakeLLM raises -> the pipeline catches and logs (graceful degradation).
    fake_llm = FakeLLM([])
    result = sections_run(art, llm_client=fake_llm, allow_embedding_fallback=False)
    summary = result.by_name()["summary"]
    assert summary.status == "ok"
    assert summary.detection_method == "regex"
    # AI-TOC is only invoked when there are required-by-downstream sections still
    # missing. If summary is the only one we found, AI-TOC may have been called
    # for the others — that's fine. What matters is summary stayed regex.


def test_ai_toc_handles_llm_failure_gracefully() -> None:
    """LLM raises -> pipeline catches; sections that nothing else can find
    end up not_found, but the run completes."""
    class RaisingLLM:
        def call_json(self, *args, **kwargs):
            raise RuntimeError("simulated LLM outage")

    raw = "Just some prose.\n" * 50
    art = _build_artifact(raw)
    result = sections_run(art, llm_client=RaisingLLM(), allow_embedding_fallback=False)
    # No exception propagates; result exists with cover + not_founds
    assert isinstance(result, SectionsArtifact)
    assert result.by_name()["cover"].status == "ok"


# ---------------------------------------------------------------------------
# AI-TOC helpers (sampling + anchor finder)
# ---------------------------------------------------------------------------

def test_sample_for_toc_short_doc_returns_full_text() -> None:
    raw = "small doc"
    assert _sample_for_toc(raw, total_chars=15_000) == raw


def test_sample_for_toc_long_doc_concatenates_three_segments() -> None:
    raw = "X" * 60_000
    out = _sample_for_toc(raw, total_chars=15_000)
    assert "BEGINNING OF DOCUMENT" in out
    assert "MIDDLE OF DOCUMENT" in out
    assert "END OF DOCUMENT" in out
    # Total chars roughly bounded by total_chars + separator headers
    assert len(out) < 15_000 + 500


def test_find_anchor_verbatim_case_insensitive_and_whitespace_normalized() -> None:
    text = "Hello\nworld\nfrom\nthe\nOCR engine."
    # Verbatim
    assert _find_anchor(text, "world") is not None
    # Case-insensitive
    assert _find_anchor(text, "WORLD") is not None
    # Whitespace-normalized: anchor crosses line breaks
    idx = _find_anchor(text, "world from the OCR")
    assert idx is not None and text[idx:idx + 20].lower().startswith("world")
    # Not present
    assert _find_anchor(text, "phrase that's not there") is None


# ---------------------------------------------------------------------------
# Integration on real fixture (typewritten 1970s doc, no LLM)
# ---------------------------------------------------------------------------

def test_fixture_typewritten_doc_falls_back_gracefully() -> None:
    """The Castaic-Haskell fixture has 'SUMMARY SHEET' / 'CONCLUSIONS' / etc.
    None match the strict CEQ regex. Without an LLM and without embeddings,
    the cascade should produce: cover ok, all other CEQ sections not_found.
    The pipeline does NOT abort; sections are a precision booster, not a gate."""
    if not FIXTURE_PATH.exists():
        pytest.skip(f"fixture not present: {FIXTURE_PATH}")

    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    doc_key, raw_text = next(iter(payload.items()))

    # Build IngestArtifact directly (avoid going through ingest.run() so we
    # don't need a NULClient in this test).
    pages = ingest_mod._split_into_fake_pages(raw_text, config.CHARS_PER_FAKE_PAGE)
    text_norm, alignment = ingest_mod._normalize_with_alignment(raw_text)
    art = IngestArtifact(
        publication_id=doc_key,
        physical_record_ids=[doc_key],
        raw_text=raw_text,
        text_normalized=text_norm,
        alignment_map=alignment,
        pages=pages,
        nul_metadata={},
        raw_text_hash="sha256:test",
    )

    result = sections_run(art, allow_ai_toc=False, allow_embedding_fallback=False)

    by_name = result.by_name()
    # Cover always wins
    assert by_name["cover"].status == "ok"
    assert by_name["cover"].detection_method == "default_pages"
    # All other CEQ sections fell through to not_found
    for name in config.CEQ_SECTIONS:
        if name == "cover":
            continue
        assert by_name[name].status == "not_found", f"{name} unexpectedly detected"


# ---------------------------------------------------------------------------
# DetectedSection JSON shape (for Stage 5 conversion to schema.SectionRecord)
# ---------------------------------------------------------------------------

def test_detected_section_json_shape() -> None:
    sec = DetectedSection(
        name="summary",
        char_span=(10, 20),
        pages=(1, 1),
        confidence=0.9,
        status="ok",
        detection_method="regex",
    )
    j = sec.to_json()
    assert j["name"] == "summary"
    assert j["char_span"] == [10, 20]
    assert j["pages"] == [1, 1]
    assert j["confidence"] == 0.9
    assert j["status"] == "ok"
    assert j["detection_method"] == "regex"
