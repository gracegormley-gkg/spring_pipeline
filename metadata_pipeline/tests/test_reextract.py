"""
MCAL_PLAN 7 Q8 re-extraction, wired (PIPELINE_REVIEW 2.5).

`segment_b/run.py` called `run_gate` without a `reextract` callback, and gate.py
guards the whole retry path on that argument being present. So the path was dead:
in the v1 batch 59 fields across 12 documents returned RE_EXTRACT and every one
became a hand-reviewed HUMAN_REVIEW without a second attempt.

Two things had to be true before wiring it was worth doing:

  * 41 of those 59 are `summary.*` or `summary_of_interest`, and re-extracting
    those used to mean redoing up to 12 Opus map calls. Checkpointing the map
    partials (`m2.save_partials`) reduces it to re-running the reduce -- one call.
  * `location` and `key_people` (6 of 59) come from deterministic postproc
    pipelines, so resampling them cannot change the value. Those decline
    explicitly, and the decline is recorded in the manifest audit rather than
    looking like a retry that silently did nothing.

No LLM is called anywhere in this file: the extractors are stubbed and what is
asserted is the routing, the checkpoint reuse, and the audit trail.
"""

from __future__ import annotations

import json

import pytest

import llm
import m2 as m2_mod
from mcal import settings
from segment_b import gate as G
from segment_b import run as R

from conftest import SYNTHETIC_PAGES, build_m1, build_m2


@pytest.fixture
def doc(doc_factory):
    return doc_factory(*SYNTHETIC_PAGES, doc_id="synthetic")


@pytest.fixture
def chunked(doc):
    from chunk import chunks_for_doc

    return chunks_for_doc(doc)


# --- The temperature override ------------------------------------------------


class TestTemperatureOverride:
    """
    No extractor in m1/m2 takes a temperature, so the +0.2 resample MCAL_PLAN 7 Q8
    asks for needs a scoped override rather than a parameter threaded through
    every extractor to serve one caller.
    """

    def test_default_is_the_call_sites_own_value(self):
        assert llm.effective_temperature(0.2) == 0.2

    def test_override_applies_inside_the_block_only(self):
        with llm.temperature(0.4):
            assert llm.effective_temperature(0.2) == 0.4
        assert llm.effective_temperature(0.2) == 0.2

    def test_it_restores_on_exception(self):
        with pytest.raises(RuntimeError):
            with llm.temperature(0.4):
                raise RuntimeError("boom")
        assert llm.effective_temperature(0.2) == 0.2

    def test_nested_overrides_restore_the_outer_value(self):
        with llm.temperature(0.4):
            with llm.temperature(0.9):
                assert llm.effective_temperature(0.2) == 0.9
            assert llm.effective_temperature(0.2) == 0.4


# --- The partials checkpoint --------------------------------------------------


class TestPartialsCheckpoint:
    PARTIALS = [
        {"chunk_index": 0, "start_page": 1, "end_page": 2, "ceq_chapter": None,
         "findings": {"overview": {"text": "t", "quotes": []}}},
    ]

    def test_round_trip(self, tmp_path):
        m2_mod.save_partials(tmp_path, "d1", self.PARTIALS)
        assert m2_mod.load_partials(tmp_path, "d1") == self.PARTIALS

    def test_absent_checkpoint_is_none_not_an_error(self, tmp_path):
        assert m2_mod.load_partials(tmp_path, "nope") is None

    def test_empty_partials_are_not_written(self, tmp_path):
        assert m2_mod.save_partials(tmp_path, "d1", []) is None
        assert m2_mod.load_partials(tmp_path, "d1") is None

    def test_corrupt_checkpoint_is_rejected_not_raised(self, tmp_path):
        m2_mod.partials_path(tmp_path, "d1").write_text("{not json")
        assert m2_mod.load_partials(tmp_path, "d1") is None

    def test_wrong_shape_is_rejected(self, tmp_path):
        m2_mod.partials_path(tmp_path, "d1").write_text('{"a": 1}')
        assert m2_mod.load_partials(tmp_path, "d1") is None

    def test_an_unwritable_dir_does_not_lose_the_document(self, tmp_path):
        """Losing the checkpoint is bad; losing the doc over it is worse."""
        blocker = tmp_path / "blocked"
        blocker.write_text("i am a file, not a directory")
        assert m2_mod.save_partials(blocker, "d1", self.PARTIALS) is None

    def test_it_is_written_before_the_reduce_runs(self, monkeypatch, doc, tmp_path):
        """
        The point of the checkpoint. The reduce is what truncates and throttles,
        and by then every expensive call has already been made.
        """
        from chunk import Chunk

        chunks = [
            Chunk(index=i, text=f"chunk {i}", start_page=i * 2 + 1,
                  end_page=i * 2 + 2, ceq_chapter=None, label=None)
            for i in range(3)
        ]
        monkeypatch.setattr(
            m2_mod, "_summary_map_one",
            lambda c: {"chunk_index": c.index, "start_page": c.start_page,
                       "end_page": c.end_page, "ceq_chapter": None, "findings": {}},
        )

        seen = {}

        def exploding_reduce(partials, _doc):
            seen["checkpoint"] = m2_mod.load_partials(tmp_path, doc.doc_id)
            raise RuntimeError("reduce truncated")

        monkeypatch.setattr(m2_mod, "_summary_reduce", exploding_reduce)
        with pytest.raises(RuntimeError):
            m2_mod.extract_summary_and_salience(
                chunks, doc, partials_dir=tmp_path
            )
        assert seen["checkpoint"] is not None
        assert len(seen["checkpoint"]) == 3


# --- The callback ------------------------------------------------------------


class TestReextractorRouting:
    @staticmethod
    def _call(reextract, field):
        return reextract(
            field, attempt=1, temperature=0.4, entry=None, critic_result=None
        )

    def _build(self, doc, chunked, tmp_path, m2=None):
        return R._reextractor(
            doc, build_m1(), m2 or build_m2(), chunked, {}, tmp_path
        )

    def test_summary_field_reduces_from_the_checkpoint(
        self, monkeypatch, doc, chunked, tmp_path
    ):
        m2_mod.save_partials(tmp_path, doc.doc_id, [{"chunk_index": 0}])
        monkeypatch.setattr(
            m2_mod, "reduce_from_partials",
            lambda p, d: {"overview": {"text": "redone", "evidence": []}},
        )
        out = self._call(self._build(doc, chunked, tmp_path), "summary.overview")
        assert out["entry"]["text"] == "redone"

    def test_summary_field_declines_without_a_checkpoint(
        self, doc, chunked, tmp_path
    ):
        """
        Honest decline rather than silently redoing 12 Opus map calls behind the
        operator's back.
        """
        out = self._call(self._build(doc, chunked, tmp_path), "summary.overview")
        assert out["ok"] is False
        assert "no summary map checkpoint" in out["note"]

    def test_summary_of_interest_reuses_the_salience_reduce(
        self, monkeypatch, doc, chunked, tmp_path
    ):
        m2_mod.save_partials(tmp_path, doc.doc_id, [{"chunk_index": 0}])
        monkeypatch.setattr(
            m2_mod, "_summary_of_interest", lambda p, s, d: [{"claim": "x"}]
        )
        out = self._call(
            self._build(doc, chunked, tmp_path), settings.SUMMARY_OF_INTEREST
        )
        assert out["entry"] == [{"claim": "x"}]

    def test_alternatives_and_themes_rerun_their_extractor(
        self, monkeypatch, doc, chunked, tmp_path
    ):
        monkeypatch.setattr(
            m2_mod, "extract_alternatives", lambda d, c: {"value": ["A"]}
        )
        monkeypatch.setattr(m2_mod, "extract_themes", lambda s: {"value": {"themes": []}})
        r = self._build(doc, chunked, tmp_path)
        assert self._call(r, "alternatives")["entry"] == {"value": ["A"]}
        assert "value" in self._call(r, "themes")["entry"]

    @pytest.mark.parametrize("field", ["title", "year", "eis_type", "lead_agency"])
    def test_m1_fields_rerun_their_extractor(
        self, monkeypatch, doc, chunked, tmp_path, field
    ):
        import m1 as m1_mod

        monkeypatch.setattr(
            m1_mod, f"extract_{field}",
            (lambda *a, **k: {"value": "redone", "confidence": "medium"}),
        )
        out = self._call(self._build(doc, chunked, tmp_path), field)
        assert out["entry"]["value"] == "redone"

    @pytest.mark.parametrize("field", ["location", "key_people"])
    def test_deterministic_pipelines_decline_with_a_reason(
        self, doc, chunked, tmp_path, field
    ):
        out = self._call(self._build(doc, chunked, tmp_path), field)
        assert out["ok"] is False
        assert "deterministic postproc" in out["note"]

    def test_an_extractor_exception_becomes_a_decline(
        self, monkeypatch, doc, chunked, tmp_path
    ):
        """A failed retry is an optimization that did not pay off, not a lost doc."""
        def boom(*a, **k):
            raise RuntimeError("model refused")

        monkeypatch.setattr(m2_mod, "extract_alternatives", boom)
        out = self._call(self._build(doc, chunked, tmp_path), "alternatives")
        assert out["ok"] is False
        assert "RuntimeError" in out["note"]

    def test_the_requested_temperature_is_in_force_during_the_call(
        self, monkeypatch, doc, chunked, tmp_path
    ):
        seen = {}
        monkeypatch.setattr(
            m2_mod, "extract_alternatives",
            lambda d, c: seen.setdefault("t", llm.effective_temperature(0.2)) or {},
        )
        self._call(self._build(doc, chunked, tmp_path), "alternatives")
        assert seen["t"] == 0.4


# --- End to end through run_gate ---------------------------------------------


class TestItReachesTheGate:
    def test_a_re_extract_field_is_retried_and_audited(
        self, mcal_artifacts, doc, tmp_path
    ):
        from test_gate import all_results, cr, reload_artifacts

        reload_artifacts(mcal_artifacts, tau=0.4)
        results = all_results(themes=cr("themes", "RE_EXTRACT"))

        calls = []

        def reextract(field, *, attempt, temperature, entry, critic_result):
            calls.append((field, attempt, temperature))
            return {"ok": False, "note": "declined in test"}

        out = G.run_gate(
            "synthetic", build_m1(), build_m2(), results,
            stage="v1", doc=doc, out_dir=tmp_path / "out",
            monitor_file=tmp_path / "m.json", reextract=reextract,
        )
        assert calls == [("themes", 1, pytest.approx(0.4))]
        audit = out.fields["themes"].re_extract
        assert audit["attempted"] is True
        assert audit["note"] == "declined in test"

    def test_without_the_callback_nothing_is_retried(
        self, mcal_artifacts, doc, tmp_path
    ):
        """The state this fix is about: the path is guarded on the callback."""
        from test_gate import all_results, cr, reload_artifacts

        reload_artifacts(mcal_artifacts, tau=0.4)
        out = G.run_gate(
            "synthetic", build_m1(), build_m2(),
            all_results(themes=cr("themes", "RE_EXTRACT")),
            stage="v1", doc=doc, out_dir=tmp_path / "out",
            monitor_file=tmp_path / "m.json",
        )
        assert out.fields["themes"].re_extract is None
        assert out.fields["themes"].gate_reason == "critic_verdict"

    def test_batch_id_is_recorded_instead_of_unbatched(
        self, mcal_artifacts, doc, tmp_path
    ):
        from test_gate import all_results, reload_artifacts

        reload_artifacts(mcal_artifacts, tau=0.4)
        G.run_gate(
            "synthetic", build_m1(), build_m2(), all_results(),
            stage="v1", doc=doc, out_dir=tmp_path / "out",
            monitor_file=tmp_path / "m.json", batch_id="v1-test-batch",
        )
        monitor = json.loads((tmp_path / "m.json").read_text())
        assert any(
            b.get("batch_id") == "v1-test-batch" for b in monitor.get("batches", [])
        ) or "v1-test-batch" in json.dumps(monitor)
