"""
Stage 1 + 1.5 integration test — exercise the full ingest.run() entrypoint
against a real EIS doc baked from docs_with_digits.json, then chain into
Stage 1.5 grouping.

Fixture: tests/fixtures/sample_doc.json — a single {doc_key: raw_text} pair.
The doc was selected as the smallest non-trivial entry in docs_with_digits
(USDA Forest Service, "Final Environmental Statement, Proposed Castaic-Haskell
Junction Power Transmission Line", ~11k chars). Confirmed indexed in NUL with
accession_number=p1074_35556035057348.

NUL behavior:
  - Tests use a stubbed NULClient (FakeNUL) by default to keep the suite
    offline and deterministic.
  - A separate test (test_real_nul_lookup_if_creds) exercises the real client
    if NUL_INTEGRATION_TEST=1 is set in the env. This isn't gated on creds
    (the NUL search API is open) but is opt-in to keep CI hermetic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pipeline import ingest
from pipeline import grouping
from pipeline.nul_client import NULClient, NULRecord


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_doc.json"


@pytest.fixture(scope="module")
def fixture_doc() -> tuple[str, str, Path]:
    """Returns (doc_key, raw_text, fixture_path). Skips if fixture absent."""
    if not FIXTURE_PATH.exists():
        pytest.skip(f"fixture not present: {FIXTURE_PATH}")
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert len(payload) == 1, "fixture should hold exactly one doc"
    doc_key, raw_text = next(iter(payload.items()))
    return doc_key, raw_text, FIXTURE_PATH


@pytest.fixture(autouse=True)
def _clear_docs_cache():
    """ingest.load_docs_json() module-caches by resolved path; tests shouldn't
    leak state across files. Clear before each test."""
    ingest._DOCS_CACHE.clear()
    yield
    ingest._DOCS_CACHE.clear()


class FakeNUL:
    """Stub NULClient that returns NULRecord(found=False) without network."""

    def __init__(self, found: bool = False, **fields) -> None:
        self._found = found
        self._fields = fields

    def get(self, doc_key: str) -> NULRecord:  # noqa: D401
        return NULRecord(
            title=self._fields.get("title"),
            creator=self._fields.get("creator", []),
            contributor=self._fields.get("contributor", []),
            date_created=self._fields.get("date_created"),
            raw=self._fields.get("raw", {}),
            cache_hit=False,
            found=self._found,
        )


# ---------------------------------------------------------------------------
# Stage 1: ingest.run() against the fixture
# ---------------------------------------------------------------------------

def test_ingest_run_against_fixture(fixture_doc) -> None:
    doc_key, raw_text, fixture_path = fixture_doc
    art = ingest.run(fixture_path, doc_key, FakeNUL())

    # Identity of what we put in
    assert art.publication_id == doc_key
    assert art.physical_record_ids == [doc_key]
    assert art.raw_text == raw_text
    assert art.raw_text_hash.startswith("sha256:")

    # Normalization preserves length-or-shrinks (we never insert chars beyond
    # ellipsis substitution, which expands by 2; assert no silent expansion of
    # >5% to catch bugs that would break char_offset_raw assumptions).
    assert len(art.text_normalized) <= int(len(raw_text) * 1.05)

    # Alignment + pages populated
    assert len(art.alignment_map) >= 1
    assert len(art.pages) >= 1

    # Pages tile the raw text contiguously
    assert art.pages[0].char_start_raw == 0
    for i in range(1, len(art.pages)):
        assert art.pages[i].char_start_raw == art.pages[i - 1].char_end_raw
    assert art.pages[-1].char_end_raw == len(raw_text)

    # NUL stub: not found, but metadata present
    assert art.nul_metadata["found"] is False
    assert art.nul_metadata["cache_hit"] is False


def test_alignment_round_trip_on_fixture(fixture_doc) -> None:
    """Every alignment span is either identity (norm[..]==raw[..]) or a
    substitution (1 raw char → 1+ norm chars). This is the binding contract
    for the verbatim-quote gate — if it ever regresses, schema-level quote
    verification would silently accept paraphrased text."""
    doc_key, raw_text, fixture_path = fixture_doc
    art = ingest.run(fixture_path, doc_key, FakeNUL())

    for sp in art.alignment_map:
        norm_chunk = art.text_normalized[sp.norm_start:sp.norm_end]
        raw_chunk = art.raw_text[sp.raw_start:sp.raw_end]
        if norm_chunk == raw_chunk:
            continue
        # Substitution: must consume exactly 1 raw char.
        assert sp.raw_end - sp.raw_start == 1, (
            f"non-identity span did not collapse to 1 raw char: span={sp}, "
            f"norm={norm_chunk!r}, raw={raw_chunk!r}"
        )


def test_unknown_doc_key_raises(fixture_doc) -> None:
    _, _, fixture_path = fixture_doc
    with pytest.raises(KeyError):
        ingest.run(fixture_path, "p1074_does_not_exist", FakeNUL())


# ---------------------------------------------------------------------------
# Stage 1 → Stage 1.5 chain
# ---------------------------------------------------------------------------

def test_stage1_to_grouping_chain(fixture_doc) -> None:
    doc_key, _, fixture_path = fixture_doc
    art = ingest.run(fixture_path, doc_key, FakeNUL())
    grp = grouping.run(art)

    assert grp.publication_id == doc_key
    assert grp.physical_record_ids == [doc_key]
    assert len(grp.components) == 1
    assert grp.components[0].record_id == doc_key
    assert grp.components[0].role == "main"
    assert grp.is_supplemental is False


# ---------------------------------------------------------------------------
# Optional live-NUL test (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("NUL_INTEGRATION_TEST") != "1",
    reason="set NUL_INTEGRATION_TEST=1 to hit the live NUL API",
)
def test_real_nul_lookup_if_creds(fixture_doc, tmp_path) -> None:
    """Live-NUL smoke test. NUL search API is open (no creds), but kept opt-in
    to keep CI hermetic. Asserts the fixture key resolves to the expected title."""
    doc_key, _, fixture_path = fixture_doc
    client = NULClient(cache_dir=tmp_path / "nul_cache")
    art = ingest.run(fixture_path, doc_key, client)
    assert art.nul_metadata["found"] is True
    assert "Castaic-Haskell" in (art.nul_metadata["title"] or "")
