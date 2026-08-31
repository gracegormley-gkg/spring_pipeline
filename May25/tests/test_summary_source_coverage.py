"""
Summary-source coverage: m2 must report when it did not read the whole document.

Two gaps, both meaning the reduce step ran on a subset of the document, and
neither of which left a trace anywhere a human would look (PIPELINE_REVIEW 2.10,
2.11):

  * a failed map call reached `log.error` and nothing else. In the v1 batch
    `p1074_35556036854040` reduced from 11 partials against a cap of 12 -- one
    call died -- and its only recorded warning is the geocoder.
  * `chunks_truncated` was written into `chunking_meta` and read by nobody: not
    the Critic, not the gate, not the manifest. `max_chunks=12` x 50pp caps the
    summary at 600 pages, which bit the 775-page document in the same batch.

The consequence in both cases is that a claim absent because page 700 was never
read is indistinguishable from a claim the document does not make. These tests
cover the producer side; `test_gate.py` covers the gating and the manifest.
"""

from __future__ import annotations

import pytest

import m2 as m2_mod
from conftest import SYNTHETIC_PAGES


@pytest.fixture
def doc(doc_factory):
    return doc_factory(*SYNTHETIC_PAGES, doc_id="synthetic")



class TestSummarySourceCoverageIsReported:
    """
    A failed map call used to reach `log.error` and nothing else, and
    `chunks_truncated` was written into `chunking_meta` and read by nobody. Both
    mean the reduce step ran on less than the document.
    """

    @staticmethod
    def _run(monkeypatch, doc, n_chunks, max_chunks, fail_indices=()):
        from chunk import Chunk

        chunks = [
            Chunk(
                index=i,
                text=f"Chunk {i} text about the project.",
                start_page=i * 2 + 1,
                end_page=i * 2 + 2,
                ceq_chapter=None,
                label=None,
            )
            for i in range(n_chunks)
        ]

        def fake_map(chunk):
            if chunk.index in fail_indices:
                raise RuntimeError("simulated map failure")
            return {
                "chunk_index": chunk.index,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "ceq_chapter": chunk.ceq_chapter,
                "findings": {},
            }

        monkeypatch.setattr(m2_mod, "_summary_map_one", fake_map)
        monkeypatch.setattr(
            m2_mod,
            "opus",
            lambda **kw: {
                k: {"text": "t", "quotes": []} for k in m2_mod.SUMMARY_SCHEMA_KEYS
            },
        )
        monkeypatch.setattr(m2_mod, "_summary_of_interest", lambda *a, **k: [])
        return m2_mod.extract_summary_and_salience(
            chunks, doc, max_chunks=max_chunks, parallel=2
        )

    def test_a_failed_map_call_reaches_the_meta(self, monkeypatch, doc):
        _s, _soi, meta = self._run(monkeypatch, doc, 4, 12, fail_indices=(1,))
        assert meta["n_chunks_failed"] == 1
        assert meta["failed_chunk_indices"] == [1]
        assert meta["n_chunks_summarized"] == 3
        assert meta["summary_source_incomplete"] is True
        # Not the same condition as the cap, and must not be conflated with it.
        assert meta["chunks_truncated"] is False

    def test_truncation_is_reported_separately(self, monkeypatch, doc):
        _s, _soi, meta = self._run(monkeypatch, doc, 17, 12)
        assert meta["chunks_truncated"] is True
        assert meta["n_chunks_failed"] == 0
        assert meta["n_chunks_summarized"] == 12
        assert meta["n_chunks_available"] == 17
        assert meta["summary_source_incomplete"] is True

    def test_both_at_once(self, monkeypatch, doc):
        """The v1 batch's 775-page document, with one call also dying."""
        _s, _soi, meta = self._run(monkeypatch, doc, 17, 12, fail_indices=(3,))
        assert meta["chunks_truncated"] is True
        assert meta["n_chunks_failed"] == 1
        assert meta["n_chunks_summarized"] == 11
        assert meta["summary_source_incomplete"] is True

    def test_a_complete_run_is_flagged_complete(self, monkeypatch, doc):
        _s, _soi, meta = self._run(monkeypatch, doc, 5, 12)
        assert meta["summary_source_incomplete"] is False
        assert meta["n_chunks_failed"] == 0
        assert meta["chunks_truncated"] is False

    def test_no_chunks_still_returns_the_full_meta_shape(self, doc):
        """The gate reads these keys; an empty document must not omit them."""
        _s, _soi, meta = m2_mod.extract_summary_and_salience([], doc)
        for key in (
            "n_chunks_summarized", "n_chunks_available", "chunks_truncated",
            "n_chunks_failed", "failed_chunk_indices", "summary_source_incomplete",
        ):
            assert key in meta

    def test_the_operator_sees_a_warning_not_just_a_log_line(self):
        from segment_b.run import DocResult, _warn_on_incomplete_summary_source

        result = DocResult(doc_id="d")
        _warn_on_incomplete_summary_source(
            {"chunking_meta": {
                "n_chunks_available": 17, "n_chunks_summarized": 11,
                "chunks_truncated": True, "n_chunks_failed": 1,
                "failed_chunk_indices": [4],
            }},
            result,
        )
        assert len(result.warnings) == 2
        assert any("map call(s) failed" in w for w in result.warnings)
        assert any("truncated" in w for w in result.warnings)

    def test_no_warning_when_coverage_is_complete(self):
        from segment_b.run import DocResult, _warn_on_incomplete_summary_source

        result = DocResult(doc_id="d")
        _warn_on_incomplete_summary_source(
            {"chunking_meta": {"chunks_truncated": False, "n_chunks_failed": 0}},
            result,
        )
        assert result.warnings == []
