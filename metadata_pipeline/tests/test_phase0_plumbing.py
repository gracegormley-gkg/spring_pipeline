"""
Phase 0 plumbing: calibration reads Segment B, held-out docs stay out of
calibration, second graders do not double-count, and reruns do not erase grades.

Before this, `mcal.build` read grades, M2 and Critic output only from
`segment_a/output`, so every grade entered on a Segment B blind sheet was
silently ignored, and `segment_b.run process` rewrote a graded sheet blank.
"""

from __future__ import annotations

import json

import pytest

from mcal import build, confidence, grades, settings
from mcal import grading_sheet as gsh


def _sheet(path, doc_id, rows):
    """Write a minimal grading sheet: rows are (field, grade[, extra cols])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# doc_id: {doc_id}", "", "field,your_grade"]
    lines += [f"{f},{g}" for f, g in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    a = tmp_path / "segment_a_sheets"
    a.mkdir()
    b = tmp_path / "segment_b_output"
    b.mkdir()
    monkeypatch.setattr(settings, "GRADING_SHEETS_DIR", a)
    monkeypatch.setattr(settings, "SEGMENT_B_OUTPUT_DIR", b)
    monkeypatch.setattr(settings, "HOLDOUT_FILE", tmp_path / "holdout.txt")
    return a, b


# --- Calibration reads Segment B -------------------------------------------


class TestSegmentBSources:
    def test_sheet_dirs_include_each_segment_b_blind_dir(self, dirs):
        a, b = dirs
        (b / "doc1" / "blind").mkdir(parents=True)
        (b / "doc1" / "reveal").mkdir(parents=True)
        got = settings.grading_sheet_dirs()
        assert got == [a, b / "doc1" / "blind"], "reveal/ must never be read"

    def test_segment_b_grades_load(self, dirs):
        _a, b = dirs
        _sheet(b / "doc1" / "blind" / "doc1.csv", "doc1", [("year", "wrong")])
        gs = grades.load_grading_sheets()
        assert [(i.doc_id, i.field, i.correct) for i in gs.items] == [
            ("doc1", "year", False)
        ]

    def test_segment_b_sheet_wins_over_segment_a(self, dirs):
        a, b = dirs
        _sheet(a / "doc1.csv", "doc1", [("year", "wrong"), ("title", "wrong")])
        _sheet(b / "doc1" / "blind" / "doc1.csv", "doc1", [("year", "correct")])
        gs = grades.load_grading_sheets()
        assert [(i.field, i.correct) for i in gs.items] == [("year", True)]
        assert any("graded sheets in both" in w for w in gs.warnings)

    def test_path_helpers_prefer_segment_b(self, dirs):
        _a, b = dirs
        assert settings.manifest_path_for("DOC1") is None
        assert settings.m2_path_for("doc1") == settings.M2_DIR / "doc1.json"
        (b / "doc1").mkdir()
        for name in ("m2.json", "critic.json", "run_manifest.json"):
            (b / "doc1" / name).write_text("{}", encoding="utf-8")
        assert settings.m2_path_for("doc1") == b / "doc1" / "m2.json"
        assert settings.critic_path_for("doc1") == b / "doc1" / "critic.json"
        assert settings.manifest_path_for("DOC1") == b / "doc1" / "run_manifest.json"

    def test_evaluation_grades_dropped_once_doc_has_segment_b_output(self, dirs):
        """The Evaluation sheet describes Segment A prose; never pair it with B."""
        _a, b = dirs
        before = grades.load_grades()
        doc = before.doc_ids[0]
        (b / doc).mkdir()
        (b / doc / "run_manifest.json").write_text("{}", encoding="utf-8")
        after = grades.load_grades()
        assert doc not in after.doc_ids
        assert after.n_docs == before.n_docs - 1
        assert any("Dropped Evaluation-sheet grades" in w for w in after.warnings)


class TestScoreFromManifest:
    def test_uses_the_verdicts_the_gate_used(self, dirs):
        _a, b = dirs
        (b / "doc1").mkdir()
        manifest = {
            "year": {"quote_verdict": None, "critic_verdict": "PASS",
                     "verdict": "HUMAN_REVIEW"},
            "summary.overview": {"quote_verdict": "no",
                                 "critic_verdict": "RE_EXTRACT"},
        }
        (b / "doc1" / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        gs = grades.GradeSet(
            items=[
                grades.GradeItem("doc1", "year", "M1", True, "correct"),
                grades.GradeItem(
                    "doc1", "summary.overview", "summary_narrative", False, "wrong"
                ),
            ]
        )
        items, diag = build.score_graded_corpus(gs, stage="v1")
        by_field = {i.field: i for i in items}
        assert by_field["year"].signals.s_critic == 1.0
        assert by_field["summary.overview"].signals.s_critic == 0.3
        assert by_field["summary.overview"].signals.s_quote == 0.0
        assert all(i.signals.s_citation is None for i in items)
        assert diag["n_docs_scored_from_segment_b"] == 1
        assert by_field["year"].composite == confidence.composite(
            by_field["year"].signals
        )


# --- Held-out set -----------------------------------------------------------


class TestHoldout:
    def test_holdout_file_parsing(self, tmp_path):
        p = tmp_path / "h.txt"
        p.write_text("# test set\nP0491_ABC  # caps ok\n\np1074_x\n", encoding="utf-8")
        assert settings.holdout_doc_ids(p) == {"p0491_abc", "p1074_x"}
        assert settings.holdout_doc_ids(tmp_path / "missing.txt") == set()

    def test_exclude_only_include(self, dirs):
        _a, b = dirs
        _sheet(b / "keep" / "blind" / "keep.csv", "keep", [("year", "wrong")])
        _sheet(b / "held" / "blind" / "held.csv", "held", [("year", "wrong")])
        settings.HOLDOUT_FILE.write_text("held\n", encoding="utf-8")
        cal = grades.load_grades()
        assert "held" not in cal.doc_ids and "keep" in cal.doc_ids
        assert cal.holdout_doc_ids == ["held"]
        test = grades.load_grades(holdout="only")
        assert test.doc_ids == ["held"]
        both = grades.load_grades(holdout="include")
        assert {"held", "keep"} <= set(both.doc_ids)

    def test_active_selection_never_proposes_a_holdout_doc(self, dirs, monkeypatch):
        from mcal import active_select

        monkeypatch.setattr(
            settings, "available_doc_ids", lambda: ["held", "free", "P0491_X"]
        )
        settings.HOLDOUT_FILE.write_text("held\np0491_x\n", encoding="utf-8")
        got = active_select.candidate_doc_ids(grades.GradeSet())
        assert got == ["free"]


# --- Second grader ----------------------------------------------------------


class TestSecondGrader:
    def test_grader_from_filename(self, tmp_path):
        assert grades.grader_from_path(tmp_path / "doc1.csv") == ("doc1", "primary")
        assert grades.grader_from_path(tmp_path / "doc1__JS.csv") == ("doc1", "js")

    def test_second_grader_is_kept_apart_and_scored_for_kappa(self, dirs):
        _a, b = dirs
        blind = b / "doc1" / "blind"
        _sheet(blind / "doc1.csv", "doc1",
               [("year", "wrong"), ("title", "correct"), ("eis_type", "correct")])
        _sheet(blind / "doc1__js.csv", "doc1",
               [("year", "wrong"), ("title", "wrong"), ("eis_type", "correct")])
        gs = grades.load_grades()
        assert len([i for i in gs.items if i.doc_id == "doc1"]) == 3
        assert {i.grader for i in gs.items} == {"primary"}
        assert len(gs.secondary_items) == 3
        k = grades.cohen_kappa(gs)["js"]
        assert k["n"] == 3
        assert k["agreement"] == pytest.approx(2 / 3)
        # primary: 1 wrong of 3, js: 2 wrong of 3 -> pe = 4/9, kappa = 0.4
        assert k["kappa"] == pytest.approx(0.4)


# --- Reruns do not erase grades --------------------------------------------


def _rows(value="1972", quote="dated 1972"):
    return [
        gsh.SheetRow(field="year", extracted_value=value, quote=quote,
                     source_pages="1"),
        gsh.SheetRow(field="title", extracted_value="A Title", quote="",
                     source_pages=""),
    ]


def _grade(path, field, grade):
    meta, rows = gsh.read_sheet(path)
    text = path.read_text(encoding="utf-8")
    header_at = text.index("\nfield,extracted_value") + 1
    preamble = text[:header_at]
    import csv
    import io

    for r in rows:
        if r["field"] == field:
            r["your_grade"] = grade
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    path.write_text(preamble + buf.getvalue(), encoding="utf-8")


class TestCarryForward:
    def test_rerun_keeps_grades_on_unchanged_rows(self, tmp_path):
        paths = gsh.write_sheets(tmp_path, "doc1", _rows())
        _grade(paths["blind"], "year", "wrong")
        gsh.write_sheets(tmp_path, "doc1", _rows())
        _m, rows = gsh.read_sheet(paths["blind"])
        assert {r["field"]: r["your_grade"] for r in rows}["year"] == "wrong"
        assert not (paths["blind"].parent / gsh.SUPERSEDED_DIRNAME).exists()

    def test_changed_row_is_not_carried_and_old_sheet_is_archived(self, tmp_path):
        paths = gsh.write_sheets(tmp_path, "doc1", _rows())
        _grade(paths["blind"], "year", "wrong")
        _grade(paths["blind"], "title", "correct")
        gsh.write_sheets(tmp_path, "doc1", _rows(value="1973"))
        _m, rows = gsh.read_sheet(paths["blind"])
        got = {r["field"]: r["your_grade"] for r in rows}
        assert got == {"year": "", "title": "correct"}
        archived = list((paths["blind"].parent / gsh.SUPERSEDED_DIRNAME).glob("*.csv"))
        assert len(archived) == 1
        _m, old = gsh.read_sheet(archived[0])
        assert {r["field"]: r["your_grade"] for r in old}["year"] == "wrong"

    def test_archive_is_invisible_to_the_grade_loader(self, tmp_path):
        paths = gsh.write_sheets(tmp_path, "doc1", _rows())
        _grade(paths["blind"], "year", "wrong")
        gsh.write_sheets(tmp_path, "doc1", _rows(value="1973"))
        gs = grades.load_grading_sheets(paths["blind"].parent)
        assert gs.items == []

    def test_second_grader_copy_is_rewritten_too(self, tmp_path):
        paths = gsh.write_sheets(tmp_path, "doc1", _rows())
        second = paths["blind"].parent / "doc1__js.csv"
        second.write_text(paths["blind"].read_text(encoding="utf-8"), encoding="utf-8")
        _grade(second, "year", "correct")
        gsh.write_sheets(tmp_path, "doc1", _rows(value="1973"))
        _m, rows = gsh.read_sheet(second)
        assert {r["field"]: r["extracted_value"] for r in rows}["year"] == "1973"
        assert {r["field"]: r["your_grade"] for r in rows}["year"] == ""


# --- Segment B checkpoints are invalidated by their inputs ------------------


class TestRunnerCheckpoints:
    """
    `segment_b.run.process_doc` reuses checkpoints only while their inputs are
    unchanged: M2 under the current extractor, location under the current
    geocoder stack, key_people under the current year, and the Critic only for
    the exact values it judged. Every model and network call is stubbed.
    """

    @pytest.fixture
    def runner(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from segment_b import run as R

        calls = {"m2": 0, "loc": 0, "kp": 0, "critic": 0}
        state = {"stack": "reduced", "map_model": "sonnet"}

        def run_m2(doc, **kw):
            calls["m2"] += 1
            return {
                "_prompt_version": settings.M2_PROMPT_VERSION_REQUIRED,
                "_summary_map_model": state["map_model"],
                "summary": {},
            }

        def run_loc(doc, chapters):
            calls["loc"] += 1
            return {"scope": "site", "stack_seen": state["stack"]}

        def run_kp(doc, chapters, **kw):
            calls["kp"] += 1
            return {"agency_preparers": []}

        def run_critic(doc, m1, m2, **kw):
            calls["critic"] += 1
            return {}

        doc = SimpleNamespace(n_pages=3, doc_id="doc1")
        monkeypatch.setattr(R, "load_doc", lambda d: doc)
        monkeypatch.setattr(R, "lookup_work", lambda d: {"title": "T"})
        monkeypatch.setattr(
            R, "chunks_for_doc", lambda d: {"chunks": [], "chapters": []}
        )
        monkeypatch.setattr(R.m1_mod, "run_m1", lambda w, d: {"year": {"value": 1972}})
        monkeypatch.setattr(
            R.year_adjudicator, "adjudicate",
            lambda d, m1_year=None: {"year": None, "source_type": None},
        )
        monkeypatch.setattr(R.m2_mod, "run_m2", run_m2)
        monkeypatch.setattr(R.location_pipeline, "run_location_pipeline", run_loc)
        monkeypatch.setattr(R.location_pipeline, "as_m2_location_field", lambda l: l)
        monkeypatch.setattr(
            R.key_people_pipeline, "run_key_people_pipeline", run_kp
        )
        monkeypatch.setattr(
            R.key_people_pipeline, "as_m2_key_people_field", lambda k: k
        )
        monkeypatch.setattr(
            R, "_apply_acronyms",
            lambda m2, d, s: {"n_glossary": 0, "n_rewritten": 0,
                              "undefined": {}, "per_field": {}},
        )
        monkeypatch.setattr(R.critic_mod, "run_critic", run_critic)
        monkeypatch.setattr(
            R.gate_mod, "run_gate",
            lambda *a, **kw: SimpleNamespace(rollup={}, manifest={}),
        )
        monkeypatch.setattr(
            settings, "geocoder_precheck",
            lambda: {"stack": state["stack"], "checklist": []},
        )
        import config as segment_a_config

        monkeypatch.setattr(segment_a_config, "SUMMARY_MAP_MODEL", "sonnet")

        def go(**kw):
            r = R.process_doc(
                "doc1", stage="v1", out_dir=tmp_path, write_sheets=False,
                reextract=False, **kw,
            )
            assert r.error is None, r.error
            return r

        return go, calls, state, tmp_path / "doc1"

    def test_second_run_reuses_everything(self, runner):
        go, calls, _state, base = runner
        go()
        go()
        assert calls == {"m2": 1, "loc": 1, "kp": 1, "critic": 1}
        assert (base / "m2_extract.json").exists()
        assert (base / "critic.inputs.json").exists()

    def test_force_recomputes_everything(self, runner):
        go, calls, _state, _base = runner
        go()
        go(force=True)
        assert calls == {"m2": 2, "loc": 2, "kp": 2, "critic": 2}

    def test_new_geocoder_stack_reruns_location_and_rejudges(self, runner):
        go, calls, state, _base = runner
        go()
        state["stack"] = "full"
        r = go()
        assert calls == {"m2": 1, "loc": 2, "kp": 1, "critic": 2}
        assert "critic checkpoint stale; re-judged" in r.warnings

    def test_m2_from_another_map_model_is_re_extracted(self, runner):
        go, calls, state, _base = runner
        state["map_model"] = "opus"   # checkpoint written by the old extractor
        go()
        state["map_model"] = "sonnet"
        r = go()
        assert calls["m2"] == 2
        assert any("M2 checkpoint is stale" in w for w in r.warnings)

    def test_legacy_critic_without_fingerprint_is_rejudged(self, runner):
        go, calls, _state, base = runner
        go()
        (base / "critic.inputs.json").unlink()
        go()
        assert calls["critic"] == 2

    def test_legacy_m2_json_is_still_read_as_the_checkpoint(self, runner):
        go, calls, _state, base = runner
        go()
        (base / "m2_extract.json").unlink()
        go()
        assert calls["m2"] == 1


class TestBudgetCap:
    def test_stops_starting_documents_once_the_cap_is_spent(self, tmp_path, monkeypatch):
        from segment_b import run as R

        started = []

        def fake_process_doc(doc_id, **kw):
            started.append(doc_id)
            return R.DocResult(
                doc_id=doc_id, stage="v1", usage={"total": {"cost_usd": 4.0}}
            )

        monkeypatch.setattr(R, "process_doc", fake_process_doc)
        monkeypatch.setattr(R.critic_mod, "resolve_stage", lambda s: "v1")
        summary = R.process_batch(["a", "b", "c", "d"], out_dir=tmp_path, max_usd=7.0)
        # a ($4) and b ($8 total) start; the cap is checked before c.
        assert started == ["a", "b"]
        assert summary["n_failed"] == 2
        assert summary["total_cost_usd"] == 8.0
        assert all("budget cap" in d["error"] for d in summary["docs"][2:])

    def test_no_cap_runs_everything(self, tmp_path, monkeypatch):
        from segment_b import run as R

        monkeypatch.setattr(
            R, "process_doc",
            lambda doc_id, **kw: R.DocResult(doc_id=doc_id, stage="v1"),
        )
        monkeypatch.setattr(R.critic_mod, "resolve_stage", lambda s: "v1")
        summary = R.process_batch(["a", "b"], out_dir=tmp_path)
        assert summary["n_ok"] == 2
