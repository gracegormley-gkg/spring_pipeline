"""
Tests for mcal/grades.py -- the Evaluation-sheet adapter.

These pin the two normalization decisions documented in that module (missing
citations count as wrong; doc-level acronym notes do not), the ungraded-cell
handling, and the per-bucket N_wrong_docs counts that drive CP degeneracy.

The expected counts are cross-checked against MCAL_PLAN 1's own failure-mode
tally, which is what makes this a meaningful test rather than a snapshot.
"""

from __future__ import annotations

import pytest

from mcal import grades, settings


# --- Field-key normalization ------------------------------------------------


class TestCanonicalField:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            # Evaluation-sheet labels
            ("EIS_type", "eis_type"),
            ("key people", "key_people"),
            ("alternatives[0]", "alternatives"),
            ("summary.public_response", "summary.public_response"),
            ("title", "title"),
            # Grading-sheet item paths collapse to the coarse field
            ("location.places[0]", "location"),
            ("location.places[12]", "location"),
            ("location.summary", "location"),
            ("key_people.cooperating_agencies[2]", "key_people"),
            ("key_people.public_commenters[0]", "key_people"),
            ("alternatives[7]", "alternatives"),
            # Whitespace tolerance
            ("  year  ", "year"),
        ],
    )
    def test_maps(self, raw, expected):
        assert grades.canonical_field(raw) == expected

    @pytest.mark.parametrize("raw", ["ID", "slug", "notes", "", None, "nonsense_field"])
    def test_metadata_and_unknown_return_none(self, raw):
        assert grades.canonical_field(raw) is None

    def test_every_canonical_target_is_a_known_field(self):
        """The map must never produce a key that has no CP bucket."""
        for label in grades._EVAL_ROW_TO_FIELD:
            fld = grades.canonical_field(label)
            assert fld in settings.FIELD_TO_BUCKET, f"{label} -> {fld} has no bucket"


# --- Grade text classification ----------------------------------------------


class TestClassifyGrade:
    def test_bare_ok_is_correct(self):
        assert grades.classify_grade("title", "ok") == (True, [])

    def test_trailing_whitespace_tolerated(self):
        """The Evaluation sheet contains 'ok ' with a trailing space."""
        assert grades.classify_grade("title", "ok ")[0] is True

    def test_blank_is_ungraded_not_correct(self):
        """
        The Lincoln Hwy `key people` cell is blank. Treating a blank as a pass
        is how a calibration set gets quietly optimistic.
        """
        for blank in ["", "   ", None, "n/a", "cant_tell"]:
            correct, _ = grades.classify_grade("key_people", blank)
            assert correct is None, f"{blank!r} should be ungraded"

    def test_missing_citation_counts_as_wrong(self):
        """
        Decision (A) in the module docstring. MCAL_PLAN 3.5 sends Q1=no to
        RE_EXTRACT and MCAL_PLAN 6 makes this a gating target, so "ok, missing
        citation" cannot be scored as a pass.
        """
        correct, tags = grades.classify_grade(
            "summary.public_response", "ok, missing citation - pg 190"
        )
        assert correct is False
        assert "T01_missing_citation" in tags

    @pytest.mark.parametrize(
        "field,raw,tag",
        [
            ("year", 'wrong: "1980" correct: "1979"', "T11_year_ocr_error"),
            ("eis_type", 'wrong: "ROD", correct: "Final"',
             "T12_eis_type_confused_with_rod"),
            ("summary.environmental_impact",
             'hallucination: "or important wildlife habitats are affected."',
             "T03_outside_text_fabrication"),
            ("summary.environmental_impact",
             'wrong: "Magnitude 7.5" correct: "Magnitude 7.0"',
             "T02_numeric_hallucination"),
            ("summary.project_description",
             'wrong: "from $659 million (Alt. V)" correct: "from $369 million (Alt. XI)"',
             "T02_numeric_hallucination"),
            ("location", "no geocode", "T06_geocode_missing"),
            ("location", "geocodes milwaukee (wish better specificity)",
             "T07_geocode_wrong_specificity"),
            ("location", "no location", "T08_scope_misclassified_national"),
            ("location", "has 3 locations (all listed in the doc) one geocoded",
             "T09_multi_site_partial_geocode"),
            ("key_people", "all commenters = cooperators",
             "T05_commenter_mislabeled_as_cooperator"),
            ("alternatives", "empty", "T10_alternatives_chapter_missed"),
        ],
    )
    def test_seed_tag_inference(self, field, raw, tag):
        correct, tags = grades.classify_grade(field, raw)
        assert correct is False
        assert tag in tags, f"expected {tag} in {tags}"

    def test_untaggable_failure_is_still_wrong(self):
        """
        'nearly empty' key_people has no matching T01-T18 code. It must still
        count as wrong -- left untagged so taxonomy induction can name it,
        rather than force-fit to an ill-matching seed code.
        """
        correct, tags = grades.classify_grade("key_people", "nearly empty")
        assert correct is False
        assert tags == []

    def test_grading_sheet_vocabulary(self):
        assert grades.classify_grade("title", "correct") == (True, [])
        assert grades.classify_grade("title", "wrong")[0] is False
        assert grades.classify_grade("title", "minor_issue")[0] is False
        assert grades.classify_grade("title", "cant_tell")[0] is None


# --- Loading the real sheet -------------------------------------------------


@pytest.fixture(scope="module")
def gs():
    if not settings.EVALUATION_CSV.exists():
        pytest.skip("Evaluation sheet not present")
    return grades.load_grades()


class TestLoadRealSheet:
    def test_eight_docs_graded_not_nine(self, gs):
        """
        MCAL_PLAN 0 and 6 assume n=9 at seed v1. Only 8 docs have an
        Evaluation-sheet column; p0491_35556036091957 has a grading sheet but no
        grades. The loader must surface this rather than silently reporting 9.
        """
        assert gs.n_docs == 8
        assert any("n=9" in w for w in gs.warnings)

    def test_no_ungraded_doc_counted_as_graded(self, gs):
        assert "p0491_35556036091957" not in gs.doc_ids

    def test_lincoln_hwy_blank_key_people_excluded(self, gs):
        """That specific cell is blank and must produce no GradeItem."""
        assert gs.get("p1074_35556039563135", "key_people") is None

    @pytest.mark.parametrize(
        "field,n_wrong",
        [
            # Cross-checked against MCAL_PLAN 1's own tallies.
            ("year", 3),                            # 1(1) "3/8 wrong"
            ("eis_type", 1),                        # 1(2) "1/8 wrong"
            ("summary.public_response", 4),         # 1(5) "4/8 missing citations"
            ("summary.affected_community", 2),      # 1(6) "2/8 missing cites"
            ("summary.alternatives_overview", 1),   # 1(6)-(7) "1/8 missing cite"
            ("alternatives", 1),                    # 1(8) Buffalo empty
            ("title", 0),                           # 1 "no observed failures"
            ("lead_agency", 0),
            ("themes", 0),
            ("summary.overview", 0),
        ],
    )
    def test_failure_counts_match_plan(self, gs, field, n_wrong):
        items = gs.for_field(field)
        assert sum(1 for i in items if not i.correct) == n_wrong

    def test_location_is_six_of_eight_not_five(self, gs):
        """
        MCAL_PLAN 1(9) says "5/8 issues" but enumerates six distinct docs:
        Randolph + LA Transit (no geocode), Airport Spur (specificity),
        Buffalo + Lincoln Hwy (partial multi-site), Fuel Economy (national).
        Only Operation Breakthrough and Bad Creek are clean. The plan miscounts.
        """
        wrong = [i for i in gs.for_field("location") if not i.correct]
        assert len(wrong) == 6

    def test_key_people_five_cooperator_mislabels(self, gs):
        """MCAL_PLAN 1(10) "5/8 all commenters = cooperators"."""
        tagged = [
            i
            for i in gs.for_field("key_people")
            if "T05_commenter_mislabeled_as_cooperator" in i.failure_tags
        ]
        assert len(tagged) == 5

    def test_acronym_note_does_not_mark_fields_wrong(self, gs):
        """
        Decision (B): the doc-level "includes undefined acronyms" note is
        recorded as a flag, not folded into y_i. If it were, every summary field
        on all 8 docs would be wrong and the distinction between an unglossed
        acronym and a fabricated magnitude would vanish.
        """
        flagged = [i for i in gs.items if i.acronym_issue]
        assert len(flagged) > 0, "the flag should be set somewhere"
        # summary.overview was graded ok on 8/8 despite every doc having the note
        assert all(i.correct for i in gs.for_field("summary.overview"))

    def test_all_items_have_a_bucket(self, gs):
        for i in gs.items:
            assert i.bucket in settings.BUCKETS
            assert i.field in settings.BUCKETS[i.bucket]

    def test_raw_grade_preserved_for_induction(self, gs):
        """taxonomy.py induction must cluster the human's words, not my regexes."""
        rows = gs.induction_rows()
        assert rows
        assert all("raw_grade" in r for r in rows)
        assert any("commenters" in r["raw_grade"] for r in rows)


class TestConformalViews:
    def test_wrong_docs_is_the_calibration_set(self, gs):
        """MCAL_PLAN 3.3 restricts tau_raw and LOO to docs with >=1 wrong item."""
        for bucket in settings.BUCKET_ORDER:
            wd = gs.wrong_docs(bucket)
            assert wd == sorted(set(wd)), "must be deduped and sorted"
            for doc_id in wd:
                assert gs.wrong_items(bucket, doc_id), (
                    f"{doc_id} listed as wrong in {bucket} but has no wrong items"
                )

    def test_n_wrong_docs_never_exceeds_n_docs(self, gs):
        for bucket in settings.BUCKET_ORDER:
            assert gs.n_wrong_docs(bucket) <= gs.n_docs

    def test_summary_of_interest_starts_empty(self, gs):
        """
        MCAL_PLAN 3.3: the field is new, so it has zero graded examples and must
        land in degenerate_severe -> gate_all_to_human at seed v1.
        """
        assert gs.for_bucket("summary_of_interest") == []
        assert gs.n_wrong_docs("summary_of_interest") == 0

    def test_expected_degeneracy_at_seed_v1(self, gs):
        """
        Documents the seed-v1 starting position. location and key_people are the
        only buckets with enough wrong docs to avoid the degenerate flag, which
        is a better position than MCAL_PLAN 7.5 predicted ("most buckets will be
        degenerate_severe").
        """
        n = {b: gs.n_wrong_docs(b) for b in settings.BUCKET_ORDER}
        severe = {
            b for b, v in n.items() if v < settings.DEGENERATE_SEVERE_MIN_WRONG_DOCS
        }
        degenerate = {
            b
            for b, v in n.items()
            if v < settings.DEGENERATE_MIN_WRONG_DOCS and b not in severe
        }
        healthy = set(n) - severe - degenerate

        assert severe == {"summary_of_interest", "alternatives+themes"}
        assert degenerate == {"M1", "summary_narrative", "summary_numeric"}
        assert healthy == {"location", "key_people"}
