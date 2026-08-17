"""
Tests for mcal/grading_sheet.py (MCAL_PLAN 7 Q5, 6; build item #14).

No LLM, no network. Rows are built from the real segment_a JSONs where available
and from `conftest`'s synthetic M1/M2 payloads otherwise, and the manifest path is
exercised against a real `segment_b/gate.py` run over synthetic artifacts rather
than a hand-written manifest -- the whole point of that path is compatibility with
what gate.py actually emits.

The four defects this module fixes, one class each:
  * `TestFailureTagColumn`   -- the column MCAL_PLAN 7 Q5 requires and grades.py
                                already reads, which never existed in the file.
  * `TestSoiUseful`          -- the doc-level acceptance test for the new field.
  * `TestBlinding`           -- no critic_verdict and no pre-populated notes.
  * `TestNoTruncation`       -- MCAL_PLAN 6 acceptance item 4.
Plus `TestRoundTrip`, which asserts rather than assumes that
`mcal.grades.load_grading_sheets` can read what we write.
"""

from __future__ import annotations

import csv
import json

import pytest

from mcal import grades as grades_mod
from mcal import grading_sheet as gsh
from mcal import settings

from conftest import (
    FABRICATED_QUOTE,
    LINCOLN_HWY,
    SYNTHETIC_PAGES,
    VERIFIABLE_QUOTE,
    build_m1,
    build_m2,
)


# --- Helpers ----------------------------------------------------------------


def rewrite_rows(path, rows):
    """Rewrite a sheet's data rows, preserving its `#` preamble."""
    preamble = [
        line for line in path.read_text(encoding="utf-8").split("\n")
        if line.startswith("#")
    ]
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for line in preamble:
            fh.write(line + "\n")
        fh.write("\n")
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def synth():
    return build_m1(), build_m2()


@pytest.fixture
def synth_rows(synth):
    m1, m2 = synth
    return gsh.rows_from_extractions(m1, m2, {})


@pytest.fixture
def synth_sheets(tmp_path, synth):
    m1, m2 = synth
    critic = {
        "summary": {
            "verdict": "PASS_WITH_NOTE",
            "notes": "Acronyms are not glossed on first use.",
            "model_confidence": "medium",
        },
        "location": {"verdict": "HUMAN_REVIEW", "notes": "No geocode returned."},
    }
    paths = gsh.build_and_write(
        tmp_path / "sheets", "synthetic", m1=m1, m2=m2, critic=critic,
        title="Synthetic EIS", work_id="csv:1",
    )
    return tmp_path / "sheets", paths


# --- Columns ----------------------------------------------------------------


class TestColumns:
    def test_blind_columns(self):
        assert gsh.BLIND_COLUMNS == (
            "field", "extracted_value", "quote", "source_pages", "quote_verified",
            "model_confidence", "your_grade", "your_failure_tag", "your_notes",
            "soi_useful",
        )

    def test_reveal_adds_the_critic_block(self):
        assert set(gsh.REVEAL_COLUMNS) - set(gsh.BLIND_COLUMNS) == set(
            gsh.CRITIC_COLUMNS
        )

    def test_original_columns_are_all_retained(self):
        """
        `segment_a/grading.py`'s columns minus `critic_verdict` must survive, or
        anything reading the old sheets breaks.
        """
        original = {
            "field", "extracted_value", "quote", "source_pages",
            "quote_verified", "model_confidence", "your_grade", "your_notes",
        }
        assert original <= set(gsh.BLIND_COLUMNS)
        assert original | {"critic_verdict"} <= set(gsh.REVEAL_COLUMNS)

    def test_unknown_pass_rejected(self):
        with pytest.raises(ValueError, match="Unknown pass"):
            gsh.columns_for("half-blind")


# --- Defect 1: your_failure_tag ---------------------------------------------


class TestFailureTagColumn:
    def test_column_exists_in_both_passes(self, synth_sheets):
        _d, paths = synth_sheets
        for p in paths.values():
            _meta, rows = gsh.read_sheet(p)
            assert "your_failure_tag" in rows[0]

    def test_column_starts_empty(self, synth_sheets):
        _d, paths = synth_sheets
        for p in paths.values():
            _meta, rows = gsh.read_sheet(p)
            assert all(not (r["your_failure_tag"] or "").strip() for r in rows)

    def test_grades_py_already_reads_it(self):
        """
        The loader has always preferred this column over its own regex tag
        inference (grades.py:557), so the authoritative column simply never
        existed in the file.
        """
        src = (settings.MCAL_ROOT / "grades.py").read_text()
        assert 'row.get("your_failure_tag")' in src

    def test_preamble_lists_per_field_vocabulary(self, synth_sheets):
        _d, paths = synth_sheets
        meta, _rows = gsh.read_sheet(paths["blind"])
        assert "tags[year]" in meta
        assert meta["tags[year]"] == "T11_year_ocr_error"
        assert "T01_missing_citation" in meta["tags[summary.public_response]"]

    def test_vocabulary_is_field_scoped_not_global(self, synth_rows):
        """
        Offering a summary row `T06_geocode_missing` invites an off-field tag,
        which pollutes the null-tag monitor MCAL_PLAN 6 reads to decide when the
        taxonomy needs new codes.
        """
        lines = {
            line.split(":", 1)[0]: line.split(":", 1)[1]
            for line in gsh.tag_vocabulary_lines(synth_rows)
        }
        assert "T06_geocode_missing" in lines["# tags[location]"]
        assert "T06_geocode_missing" not in lines["# tags[summary.overview]"]

    def test_every_listed_tag_is_a_real_code(self, synth_rows):
        from mcal import taxonomy

        tax = taxonomy.seed_taxonomy("v1")
        for line in gsh.tag_vocabulary_lines(synth_rows, tax):
            names = line.split(":", 1)[1].strip()
            if names == "(none)":
                continue
            for name in names.split(", "):
                assert tax.by_name(name) is not None, name

    def test_works_before_any_artifact_exists(self, tmp_path, monkeypatch, synth_rows):
        """
        `mcal/artifacts/` does not exist until build.py runs and a human ratifies
        the draft. A sheet that could not be written before the first build would
        make the first build ungradable.
        """
        monkeypatch.setattr(settings, "ARTIFACTS_DIR", tmp_path / "no_artifacts")
        lines = gsh.tag_vocabulary_lines(synth_rows)
        assert any("T01_missing_citation" in line for line in lines)


# --- Defect 2: soi_useful ---------------------------------------------------


class TestSoiUseful:
    def test_column_exists_in_both_passes(self, synth_sheets):
        _d, paths = synth_sheets
        for p in paths.values():
            _meta, rows = gsh.read_sheet(p)
            assert "soi_useful" in rows[0]

    def test_preamble_explains_the_question(self, synth_sheets):
        _d, paths = synth_sheets
        text = paths["blind"].read_text()
        assert "soi_useful (yes|no)" in text
        assert "standard summary" in text

    def test_preamble_says_an_empty_list_is_fine(self, synth_sheets):
        """
        MCAL_PLAN 3.15 rule 2. Without this the reviewer reads an empty
        summary_of_interest as a failure and grades the field down for being
        correct.
        """
        text = paths_text = synth_sheets[1]["blind"].read_text()
        assert "correct result for a routine" in paths_text

    def test_a_summary_of_interest_row_exists_even_when_empty(self, synth_rows):
        """You cannot answer soi_useful on a row that is not in the file."""
        fields = [r.field for r in synth_rows]
        assert settings.SUMMARY_OF_INTEREST in fields

    def test_read_back(self, synth_sheets):
        _d, paths = synth_sheets
        meta, rows = gsh.read_sheet(paths["blind"])
        assert gsh.read_soi_useful(paths["blind"]) is None
        for r in rows:
            if r["field"] == settings.SUMMARY_OF_INTEREST:
                r["soi_useful"] = "yes"
        rewrite_rows(paths["blind"], rows)
        assert gsh.read_soi_useful(paths["blind"]) == "yes"

    def test_conflicting_answers_are_warned_not_silently_resolved(
        self, synth_sheets, caplog
    ):
        _d, paths = synth_sheets
        _meta, rows = gsh.read_sheet(paths["blind"])
        rows[0]["soi_useful"] = "yes"
        rows[1]["soi_useful"] = "no"
        rewrite_rows(paths["blind"], rows)
        with caplog.at_level("WARNING"):
            assert gsh.read_soi_useful(paths["blind"]) == "yes"
        assert "conflicting" in caplog.text

    def test_extra_column_does_not_confuse_the_loader(self, synth_sheets):
        """`grades.load_grading_sheets` uses DictReader, so extras are ignored."""
        d, paths = synth_sheets
        _meta, rows = gsh.read_sheet(paths["blind"])
        for r in rows:
            if r["field"] == "year":
                r["your_grade"] = "wrong"
            if r["field"] == settings.SUMMARY_OF_INTEREST:
                r["soi_useful"] = "no"
        rewrite_rows(paths["blind"], rows)
        gs = grades_mod.load_grading_sheets(paths["blind"].parent)
        assert [i.field for i in gs.items] == ["year"]


# --- Defect 3: blinding -----------------------------------------------------


class TestBlinding:
    def test_blind_sheet_has_no_critic_verdict(self, synth_sheets):
        _d, paths = synth_sheets
        _meta, rows = gsh.read_sheet(paths["blind"])
        assert "critic_verdict" not in rows[0]
        body = "\n".join(",".join(r.values()) for r in rows)
        for verdict in ("PASS", "PASS_WITH_NOTE", "RE_EXTRACT", "HUMAN_REVIEW"):
            assert verdict not in body

    def test_blind_sheet_has_no_critic_notes_anywhere(self, synth_sheets):
        _d, paths = synth_sheets
        assert "Acronyms are not glossed" not in paths["blind"].read_text()
        assert "No geocode returned" not in paths["blind"].read_text()

    def test_your_notes_is_empty_in_the_blind_sheet(self, synth_sheets):
        """
        The defect that mattered most: `grading.py` pre-populated `your_notes`
        with the Critic's own notes. A reviewer whose notes field already argues a
        position is not an independent labeler.
        """
        _d, paths = synth_sheets
        _meta, rows = gsh.read_sheet(paths["blind"])
        assert all(not (r["your_notes"] or "").strip() for r in rows)

    def test_your_notes_is_empty_in_the_reveal_sheet_too(self, synth_sheets):
        """Critic notes live in `critic_notes`, never in the reviewer's column."""
        _d, paths = synth_sheets
        _meta, rows = gsh.read_sheet(paths["reveal"])
        assert all(not (r["your_notes"] or "").strip() for r in rows)
        assert any((r["critic_notes"] or "").strip() for r in rows)

    def test_reveal_sheet_does_carry_the_verdict(self, synth_sheets):
        _d, paths = synth_sheets
        _meta, rows = gsh.read_sheet(paths["reveal"])
        assert any(r["critic_verdict"] == "PASS_WITH_NOTE" for r in rows)

    def test_audit_passes_the_blind_sheet(self, synth_sheets):
        _d, paths = synth_sheets
        assert gsh.audit_blinding(paths["blind"]) == []

    def test_audit_fails_the_reveal_sheet(self, synth_sheets):
        _d, paths = synth_sheets
        problems = gsh.audit_blinding(paths["reveal"])
        assert "critic_column_present:critic_verdict" in problems

    def test_audit_catches_a_prepopulated_cell(self, synth_sheets):
        d, paths = synth_sheets
        _meta, rows = gsh.read_sheet(paths["blind"])
        rows[0]["your_notes"] = "the Critic thinks this is fine"
        rewrite_rows(paths["blind"], rows)
        assert "prepopulated_reviewer_cell:your_notes" in gsh.audit_blinding(
            paths["blind"]
        )

    def test_audit_catches_the_old_grading_py_sheet(self, tmp_path, synth):
        """
        End-to-end proof that the defect was real: run `segment_a/grading.py`'s
        own writer and audit its output.
        """
        from grading import write_grading_sheet

        m1, m2 = synth
        critic = {
            "summary": {"verdict": "PASS", "notes": "Critic reasoning goes here."}
        }
        path = write_grading_sheet(
            tmp_path, "synthetic", "csv:1", "t", m1, m2, critic
        )
        problems = gsh.audit_blinding(path)
        assert "critic_column_present:critic_verdict" in problems
        assert "prepopulated_reviewer_cell:your_notes" in problems

    def test_two_directories_not_two_filenames(self, synth_sheets):
        """
        `grades.load_grading_sheets` does a non-recursive `glob("*.csv")`. Two
        sheets for one document in one directory would load every grade twice and
        silently double `n_wrong_items` in each bucket.
        """
        d, paths = synth_sheets
        assert paths["blind"].parent.name == "blind"
        assert paths["reveal"].parent.name == "reveal"
        assert paths["blind"].name == paths["reveal"].name
        assert len(list(paths["blind"].parent.glob("*.csv"))) == 1

    def test_double_counting_would_have_happened_in_one_directory(self, tmp_path):
        """
        Demonstrates the hazard the directory split avoids, so the reason for the
        layout does not get refactored away.
        """
        rows = [gsh.SheetRow(field="year", extracted_value="1972")]
        flat = tmp_path / "flat"
        flat.mkdir()
        for name in ("doc.blind.csv", "doc.reveal.csv"):
            p = flat / name
            p.write_text(
                "# doc_id: doc\n\nfield,your_grade\nyear,wrong\n", encoding="utf-8"
            )
        gs = grades_mod.load_grading_sheets(flat)
        assert len(gs.items) == 2, "two sheets in one dir double-count"

    def test_preamble_states_the_pass(self, synth_sheets):
        _d, paths = synth_sheets
        for name, p in paths.items():
            meta, _rows = gsh.read_sheet(p)
            assert meta["pass"] == name

    def test_preamble_tells_the_reviewer_not_to_peek(self, synth_sheets):
        _d, paths = synth_sheets
        text = paths["blind"].read_text()
        assert "Do not open the reveal sheet" in text


class TestMergeBlindIntoReveal:
    def test_answers_are_copied(self, synth_sheets):
        _d, paths = synth_sheets
        _meta, rows = gsh.read_sheet(paths["blind"])
        for r in rows:
            if r["field"] == "year":
                r["your_grade"] = "wrong"
                r["your_failure_tag"] = "T11_year_ocr_error"
                r["your_notes"] = "cover says 1976"
        rewrite_rows(paths["blind"], rows)

        gsh.merge_blind_into_reveal(paths["blind"], paths["reveal"])
        _m, merged = gsh.read_sheet(paths["reveal"])
        year = next(r for r in merged if r["field"] == "year")
        assert year["your_grade"] == "wrong"
        assert year["your_failure_tag"] == "T11_year_ocr_error"
        assert year["your_notes"] == "cover says 1976"

    def test_critic_columns_survive_the_merge(self, synth_sheets):
        _d, paths = synth_sheets
        gsh.merge_blind_into_reveal(paths["blind"], paths["reveal"])
        _m, merged = gsh.read_sheet(paths["reveal"])
        assert list(merged[0].keys()) == list(gsh.REVEAL_COLUMNS)
        assert any(r["critic_verdict"] for r in merged)

    def test_merge_is_recorded_in_the_preamble(self, synth_sheets):
        _d, paths = synth_sheets
        gsh.merge_blind_into_reveal(paths["blind"], paths["reveal"])
        meta, _m = gsh.read_sheet(paths["reveal"])
        assert "merged_from_blind" in meta

    def test_orphan_rows_are_reported_not_appended(self, synth_sheets, caplog):
        _d, paths = synth_sheets
        _meta, rows = gsh.read_sheet(paths["blind"])
        rows.append({**{k: "" for k in rows[0]}, "field": "not_a_real_field",
                     "your_grade": "wrong"})
        rewrite_rows(paths["blind"], rows)
        with caplog.at_level("WARNING"):
            gsh.merge_blind_into_reveal(paths["blind"], paths["reveal"])
        assert "no reveal counterpart" in caplog.text
        _m, merged = gsh.read_sheet(paths["reveal"])
        assert "not_a_real_field" not in [r["field"] for r in merged]


# --- Defect 4: truncation ---------------------------------------------------


class TestNoTruncation:
    LONG_TEXT = "Sagebrush habitat displacement. " * 60  # ~1,900 chars

    def test_long_extracted_value_survives(self):
        rows = gsh.rows_from_extractions(
            build_m1(),
            build_m2(
                themes={
                    "value": {"themes": [self.LONG_TEXT], "subthemes": []},
                    "evidence": [],
                }
            ),
            {},
        )
        themes = next(r for r in rows if r.field == "themes")
        assert len(themes.extracted_value) > 1500
        assert "\u2026" not in themes.extracted_value

    def test_long_quote_survives(self):
        long_quote = VERIFIABLE_QUOTE + " " + ("and further impacts " * 40)
        rows = gsh.rows_from_extractions(
            build_m1(),
            build_m2(
                themes={
                    "value": {"themes": ["x"], "subthemes": []},
                    "evidence": [
                        {"quote": long_quote, "source_pages": ["4"],
                         "quote_verified": True}
                    ],
                }
            ),
            {},
        )
        themes = next(r for r in rows if r.field == "themes")
        assert len(themes.quote) > 700
        assert "\u2026" not in themes.quote

    def test_all_quotes_are_kept(self):
        """`grading.py` kept only the first 10."""
        evidence = [
            {"quote": f"Impact statement number {i} of the corridor study.",
             "source_pages": ["4"], "quote_verified": True}
            for i in range(15)
        ]
        rows = gsh.rows_from_extractions(
            build_m1(),
            build_m2(
                themes={"value": {"themes": ["x"]}, "evidence": evidence}
            ),
            {},
        )
        themes = next(r for r in rows if r.field == "themes")
        assert themes.quote.count(" | ") == 14

    def test_whitespace_is_still_collapsed(self):
        """
        Untruncated, not unnormalized: a newline in a cell would break the row.
        """
        assert "\n" not in gsh.render_value("a\nb\n\nc")
        assert gsh.render_value("a\n  b") == "a b"

    def test_grading_py_did_truncate(self):
        """Pins the defect so the fix is not mistaken for a no-op."""
        from grading import _short

        assert _short(self.LONG_TEXT, 400).endswith("\u2026")
        assert len(_short(self.LONG_TEXT, 400)) == 400

    def test_evidence_is_not_duplicated_into_extracted_value(self):
        rows = gsh.rows_from_extractions(build_m1(), build_m2(), {})
        loc = next(r for r in rows if r.field.startswith("location.places"))
        assert "evidence" not in loc.extracted_value
        assert loc.quote, "the quote belongs in its own column"


# --- Field expansion / row keys ---------------------------------------------


class TestExpansion:
    def test_alternatives_indexed(self):
        got = gsh.expand_field(
            "alternatives", [{"name": "A"}, {"name": "B"}]
        )
        assert [k for k, _v in got] == ["alternatives[0]", "alternatives[1]"]

    def test_empty_alternatives_still_gets_a_row(self):
        """
        MCAL_PLAN 1(8): `alternatives[0]` empty is the Buffalo failure. A row that
        is absent cannot be graded, and "extractor returned empty" must stay
        distinguishable from "not graded on that field".
        """
        assert gsh.expand_field("alternatives", []) == [("alternatives", [])]

    def test_location_places_plus_rollup(self):
        got = gsh.expand_field(
            "location",
            {"places": [{"name": "A"}, {"name": "B"}], "is_multi_site": True,
             "geocoded": []},
        )
        keys = [k for k, _v in got]
        assert keys == ["location.places[0]", "location.places[1]",
                        "location.summary"]
        rollup = dict(got)["location.summary"]
        assert rollup["is_multi_site"] is True

    def test_key_people_buckets(self):
        got = gsh.expand_field(
            "key_people",
            {"agency_preparers": [{"name": "P"}], "cooperating_agencies": [],
             "public_commenters": [{"name": "C"}, {"name": "D"}]},
        )
        keys = [k for k, _v in got]
        assert "key_people.agency_preparers[0]" in keys
        assert "key_people.cooperating_agencies" in keys  # empty bucket, one row
        assert "key_people.public_commenters[1]" in keys

    def test_key_people_scalar_flags_go_to_a_summary_row(self):
        """
        `key_people_pipeline.py` attaches `comment_response_present: bool`.
        Rendering it as an entry row invites a grade on a row with nothing in it.
        """
        got = gsh.expand_field(
            "key_people",
            {"agency_preparers": [{"name": "P"}], "comment_response_present": False},
        )
        keys = [k for k, _v in got]
        assert "key_people.comment_response_present" not in keys
        assert dict(got)["key_people.summary"] == {"comment_response_present": False}

    def test_unknown_list_bucket_still_gets_rows(self):
        got = gsh.expand_field(
            "key_people", {"tribal_governments": [{"name": "Nation"}]}
        )
        assert "key_people.tribal_governments[0]" in [k for k, _v in got]

    def test_summary_of_interest_indexed(self):
        got = gsh.expand_field(
            settings.SUMMARY_OF_INTEREST, [{"claim": "a"}, {"claim": "b"}]
        )
        assert [k for k, _v in got] == [
            "summary_of_interest[0]", "summary_of_interest[1]",
        ]

    def test_empty_summary_of_interest_gets_a_row(self):
        """MCAL_PLAN 3.15: an empty list is a substantive result."""
        assert gsh.expand_field(settings.SUMMARY_OF_INTEREST, []) == [
            (settings.SUMMARY_OF_INTEREST, [])
        ]

    def test_scalar_fields_pass_through(self):
        assert gsh.expand_field("year", 1972) == [("year", 1972)]


class TestRowKeysParse:
    def test_every_row_key_is_canonical(self, synth_rows):
        assert gsh.unparsable_fields(synth_rows) == []

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("year", "year"),
            ("summary.public_response", "summary.public_response"),
            ("alternatives[0]", "alternatives"),
            ("location.places[0]", "location"),
            ("location.summary", "location"),
            ("key_people.cooperating_agencies[2]", "key_people"),
            ("key_people.summary", "key_people"),
            ("summary_of_interest[0]", settings.SUMMARY_OF_INTEREST),
        ],
    )
    def test_canonical_field_mapping(self, key, expected):
        assert grades_mod.canonical_field(key) == expected

    def test_all_canonical_fields_are_represented(self, synth_rows):
        seen = {r.canonical_field for r in synth_rows}
        assert seen == set(settings.ALL_FIELDS)


# --- Round trip -------------------------------------------------------------


class TestRoundTrip:
    def test_written_sheet_is_readable_by_load_grading_sheets(self, synth_sheets):
        d, paths = synth_sheets
        _meta, rows = gsh.read_sheet(paths["blind"])
        filled = {
            "year": ("wrong", "T11_year_ocr_error"),
            "summary.environmental_impact": (
                "wrong", "T03_outside_text_fabrication",
            ),
            "location.places[0]": ("correct", ""),
            "key_people.public_commenters[0]": (
                "wrong", "T05_commenter_mislabeled_as_cooperator",
            ),
            "alternatives[0]": ("correct", ""),
            "summary_of_interest": ("correct", ""),
        }
        for r in rows:
            if r["field"] in filled:
                grade, tag = filled[r["field"]]
                r["your_grade"] = grade
                r["your_failure_tag"] = tag
        rewrite_rows(paths["blind"], rows)

        gs = grades_mod.load_grading_sheets(paths["blind"].parent)
        assert gs.warnings == []
        assert len(gs.items) == len(filled)
        by_field = {i.item_key: i for i in gs.items}
        assert by_field["year"].correct is False
        assert by_field["year"].failure_tags == ["T11_year_ocr_error"]
        assert by_field["year"].field == "year"
        assert by_field["location.places[0]"].field == "location"
        assert by_field["location.places[0]"].correct is True
        assert by_field["summary_of_interest"].field == settings.SUMMARY_OF_INTEREST

    def test_doc_id_is_recovered_from_the_preamble(self, synth_sheets):
        _d, paths = synth_sheets
        _meta, rows = gsh.read_sheet(paths["blind"])
        rows[0]["your_grade"] = "correct"
        rewrite_rows(paths["blind"], rows)
        gs = grades_mod.load_grading_sheets(paths["blind"].parent)
        assert gs.items[0].doc_id == "synthetic"

    def test_explicit_tag_wins_over_regex_inference(self, synth_sheets):
        _d, paths = synth_sheets
        _meta, rows = gsh.read_sheet(paths["blind"])
        for r in rows:
            if r["field"] == "year":
                r["your_grade"] = "wrong"
                r["your_failure_tag"] = "T19_scope_qualifier_dropped"
        rewrite_rows(paths["blind"], rows)
        gs = grades_mod.load_grading_sheets(paths["blind"].parent)
        assert gs.items[0].failure_tags[0] == "T19_scope_qualifier_dropped"

    def test_bucket_is_assigned(self, synth_sheets):
        _d, paths = synth_sheets
        _meta, rows = gsh.read_sheet(paths["blind"])
        for r in rows:
            if r["field"] == "summary.environmental_impact":
                r["your_grade"] = "wrong"
        rewrite_rows(paths["blind"], rows)
        gs = grades_mod.load_grading_sheets(paths["blind"].parent)
        assert gs.items[0].bucket == "summary_numeric"

    def test_unfilled_sheet_yields_nothing(self, synth_sheets):
        """A blank grade must never be read as 'correct'."""
        _d, paths = synth_sheets
        gs = grades_mod.load_grading_sheets(paths["blind"].parent)
        assert gs.items == []
        assert "synthetic" in gs.ungraded_doc_ids


# --- Quote verification -----------------------------------------------------


class TestQuoteVerification:
    def test_recomputed_with_quote_check_when_a_doc_is_supplied(
        self, doc_factory
    ):
        doc = doc_factory(*SYNTHETIC_PAGES)
        rows = gsh.rows_from_extractions(
            build_m1(),
            build_m2(
                themes={
                    "value": {"themes": ["x"]},
                    # Claims verified, but it is a fabrication.
                    "evidence": [
                        {"quote": FABRICATED_QUOTE, "source_pages": ["4"],
                         "quote_verified": True}
                    ],
                }
            ),
            {},
            doc=doc,
        )
        themes = next(r for r in rows if r.field == "themes")
        assert themes.quote_verified == "no", (
            "the stored quote_verified flag claimed True; quote_check must "
            "override it"
        )
        assert "[NO]" in themes.quote

    def test_stored_flag_used_when_no_doc(self):
        rows = gsh.rows_from_extractions(build_m1(), build_m2(), {})
        s = next(r for r in rows if r.field == "summary.overview")
        assert s.quote_verified == "yes"

    def test_verified_quote_still_verifies(self, doc_factory):
        doc = doc_factory(*SYNTHETIC_PAGES)
        rows = gsh.rows_from_extractions(
            build_m1(), build_m2(), {}, doc=doc
        )
        s = next(r for r in rows if r.field == "summary.overview")
        assert s.quote_verified == "yes"

    def test_page_tags_are_present(self):
        rows = gsh.rows_from_extractions(build_m1(), build_m2(), {})
        s = next(r for r in rows if r.field == "summary.overview")
        assert s.quote.startswith("[p.4] ")

    def test_uncited_quote_is_labelled(self):
        rows = gsh.rows_from_extractions(
            build_m1(),
            build_m2(
                themes={
                    "value": {"themes": ["x"]},
                    "evidence": [{"quote": "an uncited claim", "source_pages": []}],
                }
            ),
            {},
        )
        themes = next(r for r in rows if r.field == "themes")
        assert "[no page cited]" in themes.quote

    def test_m1_shows_provenance_instead_of_a_page(self):
        rows = gsh.rows_from_extractions(build_m1(), build_m2(), {})
        year = next(r for r in rows if r.field == "year")
        assert "NUL" in year.source_pages


# --- Manifest path ----------------------------------------------------------


class TestFromManifest:
    @pytest.fixture
    def manifest(self, mcal_artifacts, doc_factory, synth):
        """
        A real `run_manifest.json` from `segment_b/gate.py`, not a hand-written
        one -- the reason this path exists is compatibility with what gate.py
        actually emits.
        """
        from segment_b import gate

        m1, m2 = synth
        doc = doc_factory(*SYNTHETIC_PAGES, doc_id="synthetic")
        out = gate.run_gate(
            "synthetic", m1, m2, {}, stage="v1", doc=doc, write=False,
            update_monitor=False,
        )
        return out.manifest

    def test_reserved_keys_are_skipped(self, manifest):
        fields = gsh.manifest_fields(manifest)
        assert fields
        assert not any(f.startswith("_") for f in fields)
        assert "_meta" in manifest

    def test_all_canonical_fields_present(self, manifest):
        assert set(gsh.manifest_fields(manifest)) == set(settings.ALL_FIELDS)

    def test_rows_are_built_and_parse(self, manifest):
        rows = gsh.rows_from_manifest(manifest)
        assert rows
        assert gsh.unparsable_fields(rows) == []
        assert {r.canonical_field for r in rows} == set(settings.ALL_FIELDS)

    def test_gate_verdict_lands_in_critic_verdict(self, manifest):
        rows = gsh.rows_from_manifest(manifest)
        assert any(r.critic_verdict for r in rows)

    def test_gate_reason_is_in_the_notes(self, manifest):
        rows = gsh.rows_from_manifest(manifest)
        assert any("gate_reason=" in r.critic_notes for r in rows)

    def test_composite_is_labelled_not_passed_off_as_confidence(self, manifest):
        """
        `model_confidence` has no equivalent in the 3.12 schema. Composite is the
        nearest thing and is what the confidence machinery acts on, but it must be
        labelled so it is never read as the extractor's self-report.
        """
        rows = gsh.rows_from_manifest(manifest)
        confs = [r.model_confidence for r in rows if r.model_confidence]
        assert confs
        assert all(c.startswith("composite=") for c in confs)

    def test_item_evidence_recovered_from_nested_blocks(self, manifest):
        """
        The manifest stores one field-level quote, but `location.places[i]` and
        friends keep their own `evidence` inside `extracted_value`.
        """
        rows = gsh.rows_from_manifest(manifest)
        place = next(r for r in rows if r.field.startswith("location.places"))
        assert "Cook County, Illinois" in place.quote

    def test_field_level_cite_is_marked_as_such(self):
        manifest = {
            "key_people": {
                "extracted_value": {
                    "agency_preparers": [{"name": "A"}, {"name": "B"}],
                },
                "evidence_quote": "the preparers of this statement were",
                "source_pages": [7],
                "verdict": "HUMAN_REVIEW",
                "composite": 0.0,
            },
            "_meta": {"artifact_stage": "v1"},
        }
        rows = gsh.rows_from_manifest(manifest)
        assert all("(field-level)" in r.source_pages for r in rows)

    def test_empty_summary_of_interest_row_reads_as_a_list(self, manifest):
        rows = gsh.rows_from_manifest(manifest)
        soi = next(r for r in rows if r.field == settings.SUMMARY_OF_INTEREST)
        assert soi.extracted_value == "[]"

    def test_sheets_written_from_a_manifest(self, tmp_path, manifest):
        paths = gsh.build_and_write(
            tmp_path / "sheets", "synthetic", manifest=manifest
        )
        meta, rows = gsh.read_sheet(paths["blind"])
        assert meta["source"] == "run_manifest.json"
        assert meta["artifact_stage"] == "v1"
        assert gsh.audit_blinding(paths["blind"]) == []
        assert rows

    def test_manifest_wins_over_extractions(self, tmp_path, manifest, synth):
        m1, m2 = synth
        paths = gsh.build_and_write(
            tmp_path / "s", "synthetic", m1=m1, m2=m2, manifest=manifest
        )
        meta, _rows = gsh.read_sheet(paths["blind"])
        assert meta["source"] == "run_manifest.json"


# --- Refusals ---------------------------------------------------------------


class TestRefusals:
    def test_no_source_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Refusing to write an empty sheet"):
            gsh.build_and_write(tmp_path, "doc")

    def test_zero_row_source_raises(self, tmp_path):
        with pytest.raises(ValueError, match="zero rows"):
            gsh.build_and_write(tmp_path, "doc", manifest={"_meta": {}})

    def test_unknown_pass_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown pass"):
            gsh.pass_dir(tmp_path, "peek")

    def test_missing_sheet_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No grading sheet"):
            gsh.read_sheet(tmp_path / "absent.csv")


# --- Real document ----------------------------------------------------------


class TestAgainstRealDoc:
    @pytest.fixture
    def real(self):
        paths = {
            "m1": settings.M1_DIR / f"{LINCOLN_HWY}.json",
            "m2": settings.M2_DIR / f"{LINCOLN_HWY}.json",
            "critic": settings.CRITIC_DIR / f"{LINCOLN_HWY}.json",
        }
        if not all(p.exists() for p in paths.values()):
            pytest.skip("segment_a output for LINCOLN_HWY not available")
        return {k: json.loads(p.read_text()) for k, p in paths.items()}

    def test_rows_build_and_parse(self, real):
        rows = gsh.rows_from_extractions(real["m1"], real["m2"], real["critic"])
        assert len(rows) > 40
        assert gsh.unparsable_fields(rows) == []

    def test_sheets_write_and_round_trip(self, tmp_path, real):
        paths = gsh.build_and_write(
            tmp_path / "sheets", LINCOLN_HWY,
            m1=real["m1"], m2=real["m2"], critic=real["critic"],
            title="Lincoln Hwy",
        )
        assert gsh.audit_blinding(paths["blind"]) == []
        _meta, rows = gsh.read_sheet(paths["blind"])
        for r in rows:
            if r["field"] == "summary.environmental_impact":
                r["your_grade"] = "wrong"
                r["your_failure_tag"] = "T03_outside_text_fabrication"
        rewrite_rows(paths["blind"], rows)
        gs = grades_mod.load_grading_sheets(paths["blind"].parent)
        assert len(gs.items) == 1
        assert gs.items[0].doc_id == LINCOLN_HWY
        assert gs.items[0].bucket == "summary_numeric"

    def test_the_fabricated_clause_is_visible_untruncated(self, real):
        """
        MCAL_PLAN 6 acceptance item 4, on the case that matters: the reviewer must
        be able to see the fabricated clause without opening the M2 JSON. It is at
        the very END of the subfield text, which is exactly where a 400-char cap
        removes it.
        """
        rows = gsh.rows_from_extractions(real["m1"], real["m2"], real["critic"])
        env = next(r for r in rows if r.field == "summary.environmental_impact")
        assert "important wildlife habitats are affected" in env.extracted_value

    def test_grading_py_would_have_hidden_it(self, real):
        """
        Two mechanisms, both live on this document: `grading.py` caps the quote
        column at 10 quotes x 300 chars, and caps `extracted_value` at 300-400
        chars for `themes`, `location.places[i]` and `key_people.*[i]`.
        `summary.environmental_impact` here cites 12 quotes, so two of them --
        including the p.52 National Register cite, the only evidence bearing on
        the fabricated clause -- are dropped from the old sheet entirely.
        """
        from grading import _short, build_rows

        old = build_rows(LINCOLN_HWY, "csv:1", real["m1"], real["m2"], real["critic"])
        new = gsh.rows_from_extractions(real["m1"], real["m2"], real["critic"])
        old_env = next(
            r for r in old if r["field"] == "summary.environmental_impact"
        )
        new_env = next(
            r for r in new if r.field == "summary.environmental_impact"
        )
        n_evidence = len(
            real["m2"]["summary"]["environmental_impact"]["evidence"]
        )
        assert n_evidence > 10, "fixture requires more quotes than grading.py keeps"
        assert old_env["quote"].count(" | ") == 9        # capped at 10 quotes
        assert new_env.quote.count(" | ") == n_evidence - 1
        assert len(new_env.quote) > len(old_env["quote"])
        assert "National Register" in new_env.quote
        assert "National Register" not in old_env["quote"]

        # The value cap bites on the structured fields.
        old_kp = next(
            r for r in old
            if r["field"].startswith("key_people.cooperating_agencies[")
        )
        assert len(old_kp["extracted_value"]) <= 300
        assert _short("x" * 500, 300).endswith("\u2026")

    def test_key_people_rows_are_all_individually_gradable(self, real):
        """
        MCAL_PLAN 1(10): 5/8 docs labeled every commenter a cooperator. Grading
        that requires seeing each cooperating-agency entry on its own row.
        """
        rows = gsh.rows_from_extractions(real["m1"], real["m2"], real["critic"])
        coop = [
            r for r in rows
            if r.field.startswith("key_people.cooperating_agencies[")
        ]
        assert len(coop) > 5
        assert all(r.extracted_value for r in coop)
