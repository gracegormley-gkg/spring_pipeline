"""
Tests for mcal/active_select.py (MCAL_PLAN 3.6, build item #13).

No LLM by construction -- the module makes no model calls at all -- and no network.
Feature extraction is exercised against synthetic `Doc`s and, where the corpus is
available, against the real 21-document pool.

The three properties that matter operationally, each with its own class:
  * determinism (`TestDeterminism`) -- `next_batch.csv` is an artifact a reviewer
    works from, and a batch that reshuffles between runs is a batch nobody trusts;
  * graded-doc exclusion (`TestCandidatePool`) -- re-grading a graded document
    buys nothing and the pool is small;
  * honesty of the scoring model (`TestPriors`, `TestScoring`) -- the smoothing and
    the log-odds shifts are prior beliefs, and the tests state which.
"""

from __future__ import annotations

import csv

import pytest

from mcal import active_select as sel
from mcal import grades as grades_mod
from mcal import settings

from conftest import LA_TRANSIT, LINCOLN_HWY


# --- Helpers ----------------------------------------------------------------


def make_features(doc_id: str = "d1", **kw) -> sel.DocFeatures:
    """A neutral feature vector: nothing risky, everything present."""
    base = dict(
        doc_id=doc_id,
        normalized_doc_id=settings.normalize_doc_id(doc_id),
        n_pages=120,
        n_chars=200_000,
        year=1985,
        year_source="inventory",
        title="A Test EIS",
        ceq_chapters=["Alternatives", "Environmental Consequences"],
        has_comment_response_chapter=True,
        has_glossary=True,
        n_distinct_states=1,
        states=["Illinois"],
        national_scope_hits=0,
        rod_mentions=0,
    )
    base.update(kw)
    return sel.DocFeatures(**base)


class _FakeItem:
    def __init__(self, field, correct, tags=(), acronym_issue=False, doc_id="d"):
        self.field = field
        self.correct = correct
        self.failure_tags = list(tags)
        self.acronym_issue = acronym_issue
        self.doc_id = doc_id


class FakeGradeSet:
    def __init__(self, items=(), doc_ids=()):
        self.items = list(items)
        self._doc_ids = list(doc_ids)

    @property
    def doc_ids(self):
        return list(self._doc_ids)

    @property
    def n_docs(self):
        return len(self._doc_ids)

    def for_field(self, field):
        return [i for i in self.items if i.field == field]

    def tag_counts(self):
        counts: dict[str, int] = {}
        for i in self.items:
            for t in i.failure_tags:
                counts[t] = counts.get(t, 0) + 1
        return counts


# --- Feature extraction -----------------------------------------------------


class TestFeatureExtraction:
    def test_detects_a_comment_response_chapter(self, doc_factory):
        doc = doc_factory(
            "COVER",
            "CHAPTER 6 COMMENTS AND RESPONSES\nLetters received are reproduced "
            "below with agency responses.",
        )
        f = sel.extract_features("synthetic", doc)
        assert f.has_comment_response_chapter

    def test_absence_of_comment_response_chapter(self, doc_factory):
        doc = doc_factory("COVER", "CHAPTER 4 ENVIRONMENTAL CONSEQUENCES")
        assert not sel.extract_features("synthetic", doc).has_comment_response_chapter

    @pytest.mark.parametrize(
        "text",
        [
            "GLOSSARY OF TERMS",
            "LIST OF ABBREVIATIONS",
            "List of Acronyms",
            "ABBREVIATIONS AND ACRONYMS",
            "Definition of Terms",
        ],
    )
    def test_glossary_variants(self, doc_factory, text):
        assert sel.extract_features("s", doc_factory("COVER", text)).has_glossary

    def test_no_glossary(self, doc_factory):
        assert not sel.extract_features(
            "s", doc_factory("COVER", "body text with no front matter list")
        ).has_glossary

    def test_distinct_states_not_mention_counts(self, doc_factory):
        """
        Illinois repeated 50 times is one site, not fifty. Counting mentions would
        make every single-state corridor project look multi-site.
        """
        doc = doc_factory("Illinois " * 50 + " and Indiana and Ohio")
        f = sel.extract_features("s", doc)
        assert f.n_distinct_states == 3
        assert f.states == ["Illinois", "Indiana", "Ohio"]

    def test_multi_word_state_names(self, doc_factory):
        doc = doc_factory("North Carolina, South Carolina and New York")
        f = sel.extract_features("s", doc)
        assert set(f.states) == {"North Carolina", "South Carolina", "New York"}

    def test_national_scope_excludes_the_bare_word_national(self, doc_factory):
        """
        "National Register of Historic Places" and "National Park Service" appear
        in nearly every document in this corpus. Matching bare "national" would
        make the scope signal fire everywhere and discriminate nothing.
        """
        doc = doc_factory(
            "No National Register of Historic Places sites are affected. The "
            "National Park Service was consulted under the National "
            "Environmental Policy Act."
        )
        assert sel.extract_features("s", doc).national_scope_hits == 0

    def test_national_scope_detects_a_rulemaking(self, doc_factory):
        doc = doc_factory(
            "This programmatic statement supports a nationwide rulemaking "
            "published in the Federal Register."
        )
        f = sel.extract_features("s", doc)
        assert f.national_scope_hits >= 4
        assert f.looks_national

    def test_national_scope_is_length_normalized(self):
        """A 5-hit 800-page document is not a national rulemaking."""
        short = make_features(n_pages=50, national_scope_hits=5)
        long = make_features(n_pages=800, national_scope_hits=5)
        assert short.looks_national
        assert not long.looks_national

    def test_year_from_front_matter_when_inventory_has_none(self, doc_factory):
        doc = doc_factory(
            "Project 1234 DRAFT ENVIRONMENTAL IMPACT STATEMENT March 1974",
            "Approved 1974 by the Regional Administrator",
        )
        f = sel.extract_features("no_such_doc_in_inventory", doc)
        assert f.year == 1974
        assert f.year_source == "front_matter_regex"

    def test_modal_year_wins_over_first_seen(self, doc_factory):
        """
        Cover pages carry project numbers that look like years; the real date
        repeats.
        """
        doc = doc_factory("Contract 1968 award", "Dated 1976. Signed 1976.")
        assert sel.extract_features("nope", doc).year == 1976

    def test_era_flags(self):
        assert make_features(year=1973).is_pre_regulations is True
        assert make_features(year=1981).is_pre_regulations is False
        assert make_features(year=None).is_pre_regulations is None, (
            "an unknown year must not be guessed into an era"
        )

    def test_alternatives_chapter_flag(self):
        assert make_features(ceq_chapters=["Alternatives"]).has_alternatives_chapter
        assert not make_features(ceq_chapters=["Mitigation"]).has_alternatives_chapter

    def test_missing_corpus_raises_actionably(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "PAGES_DATA_DIR", tmp_path / "nothing")
        with pytest.raises(FileNotFoundError, match="No materialized OCR"):
            sel.extract_features("p1074_nonexistent")


# --- Priors -----------------------------------------------------------------


class TestPriors:
    def test_laplace_smoothing(self):
        gs = FakeGradeSet(
            [_FakeItem("year", correct=False)] * 3
            + [_FakeItem("year", correct=True)] * 5
        )
        priors = sel.field_error_priors(gs)
        assert priors["year"] == pytest.approx(4 / 10)

    def test_zero_observed_failures_is_not_zero_probability(self):
        """
        MCAL_PLAN 1: title/themes/lead_agency/summary.overview are 8/8 correct,
        and the plan itself notes the Wilson interval still admits ~30%. An
        unsmoothed 0.0 would give those fields zero uncertainty forever, so the
        sampler would stop exploring them.
        """
        gs = FakeGradeSet([_FakeItem("title", correct=True)] * 8)
        assert sel.field_error_priors(gs)["title"] == pytest.approx(1 / 10)

    def test_ungraded_field_gets_max_uncertainty(self):
        """
        `summary_of_interest` has zero graded examples by construction
        (MCAL_PLAN 3.15). 0.5 maximizes its variance contribution, which is the
        honest encoding of "never observed".
        """
        priors = sel.field_error_priors(FakeGradeSet([]))
        assert priors[settings.SUMMARY_OF_INTEREST] == sel.UNKNOWN_FIELD_PRIOR
        assert sel.UNKNOWN_FIELD_PRIOR == 0.5

    def test_all_canonical_fields_present(self):
        priors = sel.field_error_priors(FakeGradeSet([]))
        assert set(priors) == set(settings.ALL_FIELDS)

    def test_real_corpus_priors_are_in_range(self):
        gs = grades_mod.load_grades()
        priors = sel.field_error_priors(gs)
        assert all(0.0 < p < 1.0 for p in priors.values())
        # key_people was 5/8 wrong; title was 8/8 right.
        assert priors["key_people"] > priors["title"]


# --- Feature adjustments ----------------------------------------------------


class TestAdjustments:
    def test_every_adjustment_names_a_real_field(self):
        for _feat, field, _delta, _why in sel.FIELD_ADJUSTMENTS:
            assert field in settings.ALL_FIELDS, field

    def test_every_adjustment_has_a_rationale(self):
        for feat, field, delta, why in sel.FIELD_ADJUSTMENTS:
            assert delta > 0, (feat, field)
            assert "MCAL_PLAN" in why or len(why) > 40, (feat, field)

    def test_every_adjustment_feature_is_derivable(self):
        """No adjustment may key off a feature `active_feature_names` never emits."""
        emitted = set()
        for f in (
            make_features(year=1970, ceq_chapters=[], has_glossary=False,
                          has_comment_response_chapter=False, n_pages=900,
                          n_distinct_states=9, national_scope_hits=99,
                          rod_mentions=3),
            make_features(ceq_chapters=["Consultation"]),
        ):
            emitted |= set(sel.active_feature_names(f))
        referenced = {feat for feat, _f, _d, _w in sel.FIELD_ADJUSTMENTS}
        assert referenced <= emitted, referenced - emitted

    def test_pre_1978_raises_key_people_risk(self):
        priors = {f: 0.2 for f in settings.ALL_FIELDS}
        old = sel.predicted_error_rates(make_features(year=1973), priors)
        new = sel.predicted_error_rates(make_features(year=1985), priors)
        assert old["key_people"] > new["key_people"]

    def test_missing_alternatives_chapter_raises_alternatives_risk(self):
        priors = {f: 0.2 for f in settings.ALL_FIELDS}
        without = sel.predicted_error_rates(
            make_features(ceq_chapters=["Mitigation"]), priors
        )
        with_ = sel.predicted_error_rates(
            make_features(ceq_chapters=["Alternatives"]), priors
        )
        assert without["alternatives"] > with_["alternatives"]

    def test_missing_comment_response_raises_public_response_risk(self):
        priors = {f: 0.2 for f in settings.ALL_FIELDS}
        without = sel.predicted_error_rates(
            make_features(has_comment_response_chapter=False), priors
        )
        with_ = sel.predicted_error_rates(
            make_features(has_comment_response_chapter=True), priors
        )
        assert without["summary.public_response"] > with_["summary.public_response"]

    def test_national_scope_raises_location_risk(self):
        priors = {f: 0.2 for f in settings.ALL_FIELDS}
        natl = sel.predicted_error_rates(
            make_features(n_pages=100, national_scope_hits=50), priors
        )
        local = sel.predicted_error_rates(make_features(), priors)
        assert natl["location"] > local["location"]

    def test_rates_stay_probabilities(self):
        """Log-odds arithmetic cannot leave [0, 1]; assert it on the worst case."""
        priors = {f: 0.9 for f in settings.ALL_FIELDS}
        rates = sel.predicted_error_rates(
            make_features(
                year=1969, ceq_chapters=[], has_glossary=False,
                has_comment_response_chapter=False, n_pages=1500,
                n_distinct_states=40, national_scope_hits=500, rod_mentions=9,
            ),
            priors,
        )
        assert all(0.0 < p < 1.0 for p in rates.values())

    def test_adjustments_never_lower_a_prior(self):
        priors = {f: 0.2 for f in settings.ALL_FIELDS}
        risky = sel.predicted_error_rates(
            make_features(year=1970, ceq_chapters=[], has_glossary=False,
                          has_comment_response_chapter=False, n_pages=900,
                          n_distinct_states=9, rod_mentions=1),
            priors,
        )
        for field, p in risky.items():
            assert p >= priors[field] - 1e-9, field


# --- Tag prediction ---------------------------------------------------------


class TestTagPrediction:
    def test_every_predicted_tag_is_a_real_taxonomy_code(self):
        from mcal import taxonomy

        tax = taxonomy.seed_taxonomy("v1", include_proposed=True)
        for _feat, tag, _w in sel.TAG_PREDICTIONS:
            assert tax.by_name(tag) is not None, tag

    def test_weights_are_probabilities(self):
        for _feat, tag, w in sel.TAG_PREDICTIONS:
            assert 0.0 < w <= 1.0, tag

    def test_pre_1978_predicts_t13(self):
        assert "T13_pre_1978_nepa_format" in sel.predicted_tags(
            make_features(year=1973)
        )

    def test_post_1978_does_not(self):
        assert "T13_pre_1978_nepa_format" not in sel.predicted_tags(
            make_features(year=1985)
        )

    def test_no_alternatives_chapter_predicts_t10(self):
        assert "T10_alternatives_chapter_missed" in sel.predicted_tags(
            make_features(ceq_chapters=["Mitigation"])
        )

    def test_multi_site_predicts_t09(self):
        assert "T09_multi_site_partial_geocode" in sel.predicted_tags(
            make_features(n_distinct_states=4)
        )

    def test_exactly_two_states_predicts_t14_not_t09(self):
        """
        T14 is "scope is regional but fewer than two primary sites"; T09 is
        multi-site partial geocoding. Two states is the regional case, and the two
        tags must not both fire on it.
        """
        tags = sel.predicted_tags(make_features(n_distinct_states=2))
        assert "T14_regional_scope_underspecified" in tags
        assert "T09_multi_site_partial_geocode" not in tags

    def test_national_scope_predicts_t08(self):
        assert "T08_scope_misclassified_national" in sel.predicted_tags(
            make_features(n_pages=100, national_scope_hits=50)
        )

    def test_no_glossary_predicts_acronym_tags(self):
        tags = sel.predicted_tags(make_features(has_glossary=False))
        assert "T04_undefined_acronym" in tags
        assert "T15_jargon_without_gloss" in tags


class TestTagRarity:
    def test_unobserved_tag_is_maximally_rare(self):
        assert sel.tag_rarity("T13_pre_1978_nepa_format", {}) == 1.0

    def test_rarity_decreases_with_observations(self):
        counts = {"T05_commenter_mislabeled_as_cooperator": 5}
        assert sel.tag_rarity("T05_commenter_mislabeled_as_cooperator", counts) < 0.2

    def test_acronym_issue_flag_is_folded_into_t04(self):
        """
        `grades.py` decision (B) records "includes undefined acronyms" as a
        per-item `acronym_issue` flag, never as a tag, so T04 would otherwise look
        like a zero-exemplar tag with maximum rarity -- when it is in fact the most
        thoroughly observed failure in the corpus (8/8, MCAL_PLAN 1(11)).

        Counted per DOCUMENT: the flag is a doc-level note copied onto every item
        of the document, so an item-level count would report 111 against
        T01_missing_citation's 10 on the real corpus.
        """
        gs = FakeGradeSet(
            [
                _FakeItem(
                    "summary.overview",
                    correct=True,
                    acronym_issue=True,
                    doc_id=f"doc{i}",
                )
                for i in range(8)
            ]
        )
        assert gs.tag_counts().get("T04_undefined_acronym") is None
        counts = sel.tag_counts_in_grades(gs)
        assert counts["T04_undefined_acronym"] == 8

    def test_acronym_flag_counted_per_doc_not_per_item(self):
        """Many items on one document are one observation, not many."""
        gs = FakeGradeSet(
            [
                _FakeItem("summary.overview", correct=True, acronym_issue=True, doc_id="d")
                for _ in range(20)
            ]
        )
        assert sel.tag_counts_in_grades(gs)["T04_undefined_acronym"] == 1

    def test_t15_is_not_credited_with_acronym_observations(self):
        """
        T15_jargon_without_gloss is introduced by the plain-language clause
        (build item #4), which has never run, so it genuinely has zero exemplars.
        Unglossed domain jargon is a different defect from an undefined acronym --
        MCAL_PLAN 3.14 separates them. Crediting T15 here would drive its rarity
        to near zero and suppress selection of the documents needed to observe it.
        """
        gs = FakeGradeSet(
            [
                _FakeItem("summary.overview", correct=True, acronym_issue=True, doc_id=f"d{i}")
                for i in range(8)
            ]
        )
        counts = sel.tag_counts_in_grades(gs)
        assert "T15_jargon_without_gloss" not in counts
        assert sel.tag_rarity("T15_jargon_without_gloss", counts) == 1.0
        assert sel.tag_rarity("T04_undefined_acronym", counts) < 0.2

    def test_real_corpus_t04_is_not_treated_as_unobserved(self):
        counts = sel.tag_counts_in_grades(grades_mod.load_grades())
        assert counts.get("T04_undefined_acronym", 0) > 0


class TestPoolPrevalence:
    def test_universal_tag_is_discounted_to_nothing(self):
        """
        A tag predicted for every candidate carries no information about WHICH
        candidate to grade, however rare it is in the graded set.
        """
        feats = [make_features(f"d{i}", has_glossary=False) for i in range(5)]
        prev = sel.pool_tag_prevalence(feats)
        assert prev["T04_undefined_acronym"] == 1.0
        assert sel.effective_rarity("T04_undefined_acronym", {}, prev) == 0.0

    def test_distinctive_tag_keeps_its_rarity(self):
        feats = [make_features(f"d{i}", has_glossary=True) for i in range(4)]
        feats.append(make_features("d9", ceq_chapters=[], has_glossary=True))
        prev = sel.pool_tag_prevalence(feats)
        assert prev["T10_alternatives_chapter_missed"] == pytest.approx(0.2)
        assert sel.effective_rarity("T10_alternatives_chapter_missed", {}, prev) > 0.7

    def test_correction_skipped_for_a_tiny_pool(self):
        """With one candidate every tag is 'universal' by arithmetic."""
        assert sel.pool_tag_prevalence([make_features()]) == {}

    def test_universal_tags_are_dropped_from_the_dominant_list(self):
        feats = [make_features(f"d{i}", has_glossary=False) for i in range(4)]
        feats[0] = make_features("d0", has_glossary=False, ceq_chapters=[])
        cands = sel.rank_candidates(grade_set=FakeGradeSet([]), features=feats)
        top = cands[0]
        assert "T04_undefined_acronym" not in top.dominant_tags
        assert "T10_alternatives_chapter_missed" in top.dominant_tags


# --- Scoring ----------------------------------------------------------------


class TestScoring:
    def test_variance_peaks_at_one_half(self):
        """
        Uncertainty sampling: a document we are confident will FAIL teaches almost
        as little as one we are confident will pass.
        """
        priors_mid = {f: 0.5 for f in settings.ALL_FIELDS}
        priors_low = {f: 0.02 for f in settings.ALL_FIELDS}
        priors_high = {f: 0.98 for f in settings.ALL_FIELDS}
        f = make_features()
        counts: dict[str, int] = {}
        mid = sel.score_candidate(f, priors_mid, counts).variance_term
        low = sel.score_candidate(f, priors_low, counts).variance_term
        high = sel.score_candidate(f, priors_high, counts).variance_term
        assert mid > low
        assert mid > high
        assert mid == pytest.approx(1.0)

    def test_terms_are_in_unit_interval(self):
        c = sel.score_candidate(
            make_features(year=1969, ceq_chapters=[], has_glossary=False,
                          has_comment_response_chapter=False, n_pages=1500,
                          n_distinct_states=40, national_scope_hits=500),
            {f: 0.5 for f in settings.ALL_FIELDS},
            {},
        )
        assert 0.0 <= c.variance_term <= 1.0
        assert 0.0 <= c.rarity_term <= 1.0
        assert 0.0 <= c.uncertainty_score <= 1.0

    def test_weights_sum_to_one(self):
        assert sel.VARIANCE_WEIGHT + sel.RARITY_WEIGHT == pytest.approx(1.0)

    def test_rarity_lifts_a_doc_with_unobserved_tags(self):
        priors = {f: 0.3 for f in settings.ALL_FIELDS}
        f = make_features(ceq_chapters=["Mitigation"])
        fresh = sel.score_candidate(f, priors, {})
        saturated = sel.score_candidate(
            f, priors, {"T10_alternatives_chapter_missed": 40}
        )
        assert fresh.rarity_term > saturated.rarity_term
        assert fresh.uncertainty_score > saturated.uncertainty_score

    def test_no_predicted_tags_means_zero_rarity(self):
        f = make_features()  # nothing risky at all
        assert sel.predicted_tags(f) == {}
        assert sel.score_candidate(f, {f2: 0.3 for f2 in settings.ALL_FIELDS}, {}) \
            .rarity_term == 0.0

    def test_explain_quotes_the_rationale(self):
        c = sel.score_candidate(
            make_features(ceq_chapters=["Mitigation"]),
            {f: 0.3 for f in settings.ALL_FIELDS},
            {},
        )
        text = "\n".join(c.explain())
        assert "no_alternatives_chapter" in text
        assert "MCAL_PLAN 1(8)" in text

    def test_to_dict_is_json_shaped(self):
        import json

        c = sel.score_candidate(
            make_features(), {f: 0.3 for f in settings.ALL_FIELDS}, {}
        )
        json.dumps(c.to_dict())


# --- Determinism ------------------------------------------------------------


class TestDeterminism:
    def test_identical_input_gives_identical_ranking(self):
        feats = [
            make_features("a", year=1973),
            make_features("b", ceq_chapters=[]),
            make_features("c", n_distinct_states=5),
        ]
        gs = FakeGradeSet([])
        first = sel.rank_candidates(grade_set=gs, features=feats)
        second = sel.rank_candidates(grade_set=gs, features=list(reversed(feats)))
        assert [c.doc_id for c in first] == [c.doc_id for c in second]
        assert [c.uncertainty_score for c in first] == [
            c.uncertainty_score for c in second
        ]

    def test_ties_break_on_doc_id(self):
        feats = [make_features("zebra"), make_features("apple"),
                 make_features("mango")]
        ranked = sel.rank_candidates(grade_set=FakeGradeSet([]), features=feats)
        assert len({c.uncertainty_score for c in ranked}) == 1
        assert [c.doc_id for c in ranked] == ["apple", "mango", "zebra"]

    def test_csv_is_byte_identical_across_runs(self, tmp_path):
        feats = [make_features("a", year=1973), make_features("b", ceq_chapters=[])]
        gs = FakeGradeSet([])
        p1 = tmp_path / "one.csv"
        p2 = tmp_path / "two.csv"
        sel.write_next_batch(sel.rank_candidates(grade_set=gs, features=feats),
                             n=2, path=p1)
        sel.write_next_batch(sel.rank_candidates(grade_set=gs, features=feats),
                             n=2, path=p2)
        assert p1.read_bytes() == p2.read_bytes()

    def test_no_llm_import_in_the_module(self):
        """
        MCAL_PLAN 3.6 requires this to be cheap enough to run between every
        round. Asserted structurally rather than by trusting the docstring.
        """
        src = (settings.MCAL_ROOT / "active_select.py").read_text()
        for forbidden in ("import llm", "from llm ", "call_json", "MODEL_OPUS",
                          "MODEL_SONNET"):
            assert forbidden not in src, forbidden


# --- Candidate pool ---------------------------------------------------------


class TestCandidatePool:
    def test_graded_docs_are_excluded(self, monkeypatch):
        monkeypatch.setattr(
            settings, "available_doc_ids",
            lambda: ["p1074_aaa", "p1074_bbb", "P0491_ccc"],
        )
        gs = FakeGradeSet(doc_ids=["p1074_aaa"])
        assert sel.candidate_doc_ids(gs) == ["P0491_ccc", "p1074_bbb"]

    def test_case_insensitive_exclusion(self, monkeypatch):
        """
        Casing on disk is inconsistent (`P0491_...` and `p0491_...` both exist),
        so exclusion must go through `settings.normalize_doc_id`.
        """
        monkeypatch.setattr(
            settings, "available_doc_ids", lambda: ["P0491_35556036854362"]
        )
        gs = FakeGradeSet(doc_ids=["p0491_35556036854362"])
        assert sel.candidate_doc_ids(gs) == []

    def test_on_disk_casing_is_preserved(self, monkeypatch):
        """
        `pages.load_doc` does a case-SENSITIVE directory lookup, so the emitted
        doc_id has to be the on-disk name to be directly usable.
        """
        monkeypatch.setattr(
            settings, "available_doc_ids", lambda: ["P0491_35556036854362"]
        )
        assert sel.candidate_doc_ids(FakeGradeSet()) == ["P0491_35556036854362"]

    def test_sorted_case_insensitively(self, monkeypatch):
        monkeypatch.setattr(
            settings, "available_doc_ids",
            lambda: ["p1074_zzz", "P0491_aaa", "p0491_bbb"],
        )
        assert sel.candidate_doc_ids(FakeGradeSet()) == [
            "P0491_aaa", "p0491_bbb", "p1074_zzz",
        ]

    def test_empty_corpus(self, monkeypatch):
        monkeypatch.setattr(settings, "available_doc_ids", lambda: [])
        assert sel.candidate_doc_ids(FakeGradeSet()) == []


# --- Artifact ---------------------------------------------------------------


class TestNextBatchCsv:
    def _cands(self):
        return sel.rank_candidates(
            grade_set=FakeGradeSet([]),
            features=[
                make_features("a", year=1973),
                make_features("b", ceq_chapters=[]),
                make_features("c", n_distinct_states=6),
                make_features("d"),
            ],
        )

    def test_columns_match_the_plan_exactly(self, tmp_path):
        p = sel.write_next_batch(self._cands(), n=2, path=tmp_path / "nb.csv")
        with open(p, newline="") as fh:
            header = next(csv.reader(fh))
        assert header == [
            "doc_id", "uncertainty_score", "dominant_predicted_failure_tags",
        ]
        assert tuple(header) == sel.NEXT_BATCH_COLUMNS

    def test_no_comment_preamble(self, tmp_path):
        """
        Unlike the grading sheets. MCAL_PLAN 2 specifies this file's schema
        verbatim, and a reviewer or UI reading it must not have to know M-Cal's
        commenting conventions.
        """
        p = sel.write_next_batch(self._cands(), n=2, path=tmp_path / "nb.csv")
        assert not p.read_text().startswith("#")

    def test_respects_n(self, tmp_path):
        p = sel.write_next_batch(self._cands(), n=2, path=tmp_path / "nb.csv")
        assert len(sel.read_next_batch(p)) == 2

    def test_n_larger_than_pool_is_fine(self, tmp_path):
        p = sel.write_next_batch(self._cands(), n=99, path=tmp_path / "nb.csv")
        assert len(sel.read_next_batch(p)) == 4

    def test_round_trip(self, tmp_path):
        cands = self._cands()
        p = sel.write_next_batch(cands, n=3, path=tmp_path / "nb.csv")
        rows = sel.read_next_batch(p)
        assert [r["doc_id"] for r in rows] == [c.doc_id for c in cands[:3]]
        assert rows[0]["dominant_predicted_failure_tags"] == cands[0].dominant_tags
        assert rows[0]["uncertainty_score"] == pytest.approx(
            cands[0].uncertainty_score, abs=1e-4
        )

    def test_tags_are_pipe_joined(self, tmp_path):
        """
        A comma-joined cell survives csv quoting but not a round trip through a
        spreadsheet, which is where this file gets opened.
        """
        p = sel.write_next_batch(self._cands(), n=4, path=tmp_path / "nb.csv")
        text = p.read_text()
        assert sel.TAG_SEPARATOR == "|"
        assert '"' not in text

    def test_default_path_is_unversioned(self):
        """MCAL_PLAN 2 lists next_batch.csv without a stage suffix."""
        assert settings.NEXT_BATCH_PATH.name == "next_batch.csv"
        assert sel.write_next_batch.__doc__ and "rolling" in \
            sel.write_next_batch.__doc__

    def test_missing_file_error_is_actionable(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="python -m mcal.active_select"):
            sel.read_next_batch(tmp_path / "absent.csv")

    def test_default_n_is_from_settings(self):
        assert settings.NEXT_BATCH_SIZE == 10

    def test_report_flags_uncalibrated(self):
        cands = self._cands()
        report = sel.selection_report(cands, n=2, grade_set=FakeGradeSet([]))
        assert report["calibrated"] is False
        assert "not calibrated" in report["caveat"].lower()
        assert report["n_selected"] == 2
        assert report["n_pool"] == 4

    def test_report_names_zero_exemplar_tags(self):
        cands = self._cands()
        report = sel.selection_report(cands, n=4, grade_set=FakeGradeSet([]))
        assert report["zero_exemplar_tags_covered"]


# --- CLI --------------------------------------------------------------------


class TestCli:
    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch, capsys):
        out = tmp_path / "nb.csv"
        monkeypatch.setattr(settings, "NEXT_BATCH_PATH", out)
        rc = sel.main(["--n", "2", "--dry-run"])
        assert rc == 0
        assert not out.exists()
        assert "dry-run" in capsys.readouterr().out

    def test_writes_the_batch(self, tmp_path, capsys):
        out = tmp_path / "nb.csv"
        rc = sel.main(["--n", "3", "--out", str(out)])
        assert rc == 0
        assert out.exists()
        assert len(sel.read_next_batch(out)) <= 3

    def test_empty_pool_exits_nonzero(self, monkeypatch, capsys):
        monkeypatch.setattr(settings, "available_doc_ids", lambda: [])
        assert sel.main(["--n", "10", "--dry-run"]) == 1
        assert "No candidate documents" in capsys.readouterr().err

    def test_restricting_the_pool(self, tmp_path, capsys):
        out = tmp_path / "nb.csv"
        rc = sel.main(["--doc", LINCOLN_HWY, "--n", "1", "--out", str(out)])
        assert rc == 0
        assert sel.read_next_batch(out)[0]["doc_id"] == LINCOLN_HWY

    def test_verbose_shows_reasoning(self, capsys):
        sel.main(["--doc", LA_TRANSIT, "--n", "1", "--dry-run", "--verbose"])
        assert "variance term" in capsys.readouterr().out


# --- Real corpus ------------------------------------------------------------


class TestAgainstCorpus:
    @staticmethod
    @pytest.fixture(scope="class")
    def pool():
        gs = grades_mod.load_grades()
        ids = sel.candidate_doc_ids(gs)
        if not ids:
            pytest.skip("no ungraded materialized docs on this machine")
        return gs, ids

    def test_the_graded_eight_are_excluded(self, pool):
        gs, ids = pool
        graded = {settings.normalize_doc_id(d) for d in gs.doc_ids}
        assert graded
        assert not ({settings.normalize_doc_id(i) for i in ids} & graded)

    def test_the_ungraded_sheet_doc_is_a_candidate(self, pool):
        """
        `p0491_35556036091957` has OCR and a grading sheet but no grades
        (grades.py docstring), so it belongs in the pool.
        """
        _gs, ids = pool
        norm = {settings.normalize_doc_id(i) for i in ids}
        if "p0491_35556036091957" not in {
            settings.normalize_doc_id(d) for d in settings.available_doc_ids()
        }:
            pytest.skip("that doc is not materialized here")
        assert "p0491_35556036091957" in norm

    def test_the_capitalized_doc_is_a_candidate_with_its_disk_casing(self, pool):
        _gs, ids = pool
        on_disk = settings.available_doc_ids()
        cap = [d for d in on_disk if d != d.lower()]
        if not cap:
            pytest.skip("no capitalized doc directory here")
        for d in cap:
            assert d in ids, f"{d} lost its on-disk casing"

    def test_ranking_is_reproducible(self, pool):
        gs, ids = pool
        a = sel.rank_candidates(grade_set=gs, doc_ids=ids)
        b = sel.rank_candidates(grade_set=gs, doc_ids=ids)
        assert [c.doc_id for c in a] == [c.doc_id for c in b]
        assert [round(c.uncertainty_score, 9) for c in a] == [
            round(c.uncertainty_score, 9) for c in b
        ]

    def test_every_candidate_gets_features(self, pool):
        gs, ids = pool
        cands = sel.rank_candidates(grade_set=gs, doc_ids=ids)
        assert len(cands) == len(ids)
        for c in cands:
            assert c.features.n_pages > 0
            assert c.features.n_chars > 0

    def test_batch_covers_thin_tags(self, pool):
        """
        The point of the rarity term. `critic_prompt.select_few_shots` does greedy
        set-cover over observed tags, so a batch that exercises only
        already-covered tags cannot improve any prompt.
        """
        gs, ids = pool
        cands = sel.rank_candidates(grade_set=gs, doc_ids=ids)
        report = sel.selection_report(cands, n=settings.NEXT_BATCH_SIZE, grade_set=gs)
        assert report["zero_exemplar_tags_covered"], (
            "a batch that covers no zero-exemplar tag is not doing its job"
        )

    def test_dominant_tags_are_not_all_identical(self, pool):
        """
        Regression on the prevalence correction. Before it, T04/T15 occupied two
        of the three slots on 9 of the top 10 rows and the column was useless.
        """
        gs, ids = pool
        cands = sel.rank_candidates(grade_set=gs, doc_ids=ids)
        top = cands[: settings.NEXT_BATCH_SIZE]
        distinct = {tuple(c.dominant_tags) for c in top}
        assert len(distinct) > 1

    def test_pool_size_matches_the_corpus(self, pool):
        gs, ids = pool
        assert len(ids) == len(settings.available_doc_ids()) - gs.n_docs
