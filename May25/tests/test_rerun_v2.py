"""
scripts/rerun_v2.py: the resumable runner behind `smoke` and `all`.

The runner used to hand the whole list to `process_batch(force=True)`, so a run
that died at document 12 re-ran (and re-paid for) documents 1-11 on restart, and
the --max-usd cap started again from zero. These tests pin the resume contract
without calling a model.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rerun_v2.py"


@pytest.fixture
def rerun(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("rerun_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "OUT", tmp_path)
    monkeypatch.setattr(mod, "PROGRESS_DIR", tmp_path / "_progress")
    monkeypatch.setattr(mod.segb.critic_mod, "resolve_stage", lambda s: "v1")
    mod.commit = "a" * 40
    monkeypatch.setattr(
        mod.gate_mod, "code_version",
        lambda: {"describe": "extractor-v2.2", "commit": mod.commit, "dirty": False},
    )
    mod.calls = []
    mod.fail = set()

    def fake_process_doc(doc_id, **kw):
        mod.calls.append(doc_id)
        if doc_id in mod.fail:
            raise RuntimeError("boom")
        return mod.segb.DocResult(
            doc_id=doc_id, stage="v1",
            usage={"total": {"cost_usd": 2.0}},
            rollup={"n_fields": 15, "n_gated": 5},
        )

    monkeypatch.setattr(mod.segb, "process_doc", fake_process_doc)
    return mod


def _records(rerun, batch_id):
    path = rerun.PROGRESS_DIR / f"{batch_id}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()]


class TestResume:
    def test_a_fresh_run_records_every_document(self, rerun):
        summary = rerun._run(["a", "b", "c"], "t", max_usd=None)
        assert rerun.calls == ["a", "b", "c"]
        assert [r["doc_id"] for r in _records(rerun, "t")] == ["a", "b", "c"]
        assert summary["n_ok"] == 3 and summary["total_cost_usd"] == 6.0
        assert summary["overall_gate_rate"] == round(15 / 45, 4)

    def test_a_rerun_skips_finished_documents_and_retries_failed_ones(self, rerun):
        rerun.fail = {"b"}
        first = rerun._run(["a", "b", "c"], "t", max_usd=None)
        assert first["n_failed"] == 1

        rerun.calls.clear()
        rerun.fail = set()
        second = rerun._run(["a", "b", "c"], "t", max_usd=None)
        assert rerun.calls == ["b"]
        assert second["n_ok"] == 3

    def test_the_cap_counts_what_earlier_attempts_spent(self, rerun):
        rerun._run(["a", "b"], "t", max_usd=None)   # $4 spent
        rerun.calls.clear()
        summary = rerun._run(["a", "b", "c", "d"], "t", max_usd=4.0)
        assert rerun.calls == []
        assert summary["n_ok"] == 2
        assert {d["doc_id"]: d.get("error") for d in summary["docs"]}["c"].startswith("not run")

    def test_a_document_in_flight_finishes_past_the_cap(self, rerun):
        rerun._run(["a", "b", "c"], "t", max_usd=3.0)
        assert rerun.calls == ["a", "b"]

    def test_resuming_on_different_code_is_refused(self, rerun):
        rerun._run(["a"], "t", max_usd=None)
        rerun.commit = "b" * 40
        with pytest.raises(SystemExit, match="Move it aside"):
            rerun._run(["a", "b"], "t", max_usd=None)

    def test_each_record_carries_the_commit(self, rerun):
        rerun._run(["a"], "t", max_usd=None)
        assert _records(rerun, "t")[0]["commit"] == "a" * 40
