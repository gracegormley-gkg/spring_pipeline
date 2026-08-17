"""
Tests for mcal/quote_check.py (MCAL_PLAN 3.2, build item #1).

Covers OCR normalization, page coercion, the dual ratio+coverage gate, the
numeric channel, and the two regression fixtures MCAL_PLAN 1(4) names by file:
the Lincoln Hwy wildlife clause and the LA Transit magnitude figure.
"""

from __future__ import annotations

import pytest

from mcal import quote_check as qc
from mcal import settings

from conftest import LA_TRANSIT, LINCOLN_HWY


# --- OCR normalization ------------------------------------------------------


class TestNormalize:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("Modoc National Forest", "M0doc Nati0nal F0rest"),   # O <-> 0
            ("commercial", "comrnercial"),                        # rn -> m
            ("would", "vvould"),                                  # vv -> w
            ("Illinois", "I11inois"),                             # l <-> 1 <-> I
            ("Sacramento", "5acramento"),                         # S <-> 5
            ("BLM", "8LM"),                                       # B <-> 8
            ("  spaced   out  ", "spaced out"),                   # whitespace
            ("re-\nsponse", "response"),                          # hyphenation
            ("\u201cquoted\u201d", '"quoted"'),                   # smart quotes
            ("fi\u2010nal", "fi-nal"),                            # unicode hyphen
        ],
    )
    def test_confusables_collapse(self, a, b):
        assert qc.normalize(a) == qc.normalize(b)

    def test_distinct_words_stay_distinct(self):
        assert qc.normalize("transmission") != qc.normalize("transportation")

    def test_empty(self):
        assert qc.normalize("") == ""
        assert qc.normalize(None) == ""

    def test_punctuation_stripped(self):
        assert qc.normalize("cost: $1,200.") == qc.normalize("cost 1 200")


# --- Numeric channel --------------------------------------------------------


class TestNumericTokens:
    def test_magnitude_precision_preserved(self):
        """The whole point of a separate numeric channel: 7.5 != 7.0."""
        assert qc.numeric_tokens("Magnitude 7.5") == ["7.5"]
        assert qc.numeric_tokens("Magnitude 7.0") == ["7.0"]
        assert qc.numeric_tokens("Magnitude 7.5") != qc.numeric_tokens("Magnitude 7.0")

    def test_confusable_folding_would_have_destroyed_this(self):
        """Normalization maps 5 -> 5 and S -> 5, so 7.5 and 7.S collide there."""
        assert qc.normalize("7.5") == qc.normalize("7.S")
        # ...which is exactly why numeric_tokens does not use normalize().
        assert qc.numeric_tokens("7.5") != qc.numeric_tokens("7.S")

    def test_scale_words_attached(self):
        assert qc.numeric_tokens("$369 million") == ["369million"]
        assert qc.numeric_tokens("$659 million") == ["659million"]
        assert qc.numeric_tokens("369 million") != qc.numeric_tokens("369")

    def test_thousands_separators(self):
        assert qc.numeric_tokens("1,200 acres") == ["1200"]
        assert qc.numeric_tokens("1,450 million") == ["1450million"]

    def test_multiple(self):
        assert qc.numeric_tokens("a 47-mile 500-kV line") == ["47", "500"]

    def test_none_present(self):
        assert qc.numeric_tokens("no figures here") == []


# --- Page coercion ----------------------------------------------------------


class TestCoercePages:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (["27"], [27]),                    # segment_a shape: list[str]
            ([12, 47], [12, 47]),              # MCAL_PLAN shape: list[int]
            ("12-14", [12, 13, 14]),           # grading-sheet span
            (["1-3"], [1, 2, 3]),
            (["331", "365"], [331, 365]),
            (27, [27]),
            (None, []),
            ([], []),
            (["", None, "5"], [5]),
        ],
    )
    def test_shapes(self, raw, expected):
        assert qc.coerce_pages(raw) == expected

    def test_reversed_span_is_normalized(self):
        assert qc.coerce_pages("14-12") == [12, 13, 14]

    def test_absurd_span_does_not_expand(self):
        """A malformed span must not silently swallow the whole document."""
        got = qc.coerce_pages("1-9999")
        assert got == [1, 9999]

    def test_tolerance_expansion(self):
        assert qc.expand_with_tolerance([10], 2) == [8, 9, 10, 11, 12]

    def test_tolerance_never_goes_below_page_one(self):
        assert qc.expand_with_tolerance([1], 2) == [1, 2, 3]

    def test_tolerance_merges_overlaps(self):
        assert qc.expand_with_tolerance([10, 12], 2) == [8, 9, 10, 11, 12, 13, 14]


# --- Content coverage -------------------------------------------------------


class TestContentCoverage:
    def test_full_coverage(self):
        q = qc.normalize("sagebrush habitat displacement")
        p = qc.normalize("the sagebrush habitat displacement would be permanent")
        assert qc.content_coverage(q, p) == 1.0

    def test_zero_coverage(self):
        q = qc.normalize("sagebrush habitat displacement acreage")
        p = qc.normalize("comment letters received during the review period")
        assert qc.content_coverage(q, p) == 0.0

    def test_too_few_tokens_returns_none(self):
        """Falls back to the ratio gate rather than scoring noise."""
        assert qc.content_coverage(qc.normalize("the a of"), qc.normalize("x")) is None

    def test_nepa_boilerplate_is_not_evidence(self):
        """
        'environmental impact statement' matches every page of every EIS, so it
        must not count toward coverage.
        """
        assert qc.content_tokens(qc.normalize("environmental impact statement")) == []


# --- Gate behaviour ---------------------------------------------------------


class TestGate:
    def test_empty_quote_is_never_verified(self):
        """
        MCAL_PLAN 3.5 requires a >=20-char evidence_quote or a null + RE_EXTRACT.
        Returning 'yes' for an empty string would silently defeat that.
        """
        r = qc.check_quote("", ["1"], None)
        assert r.verified == "no"
        assert r.s_quote == 0.0
        assert r.reason == "empty_quote"

    def test_no_cited_pages_is_not_verified(self):
        r = qc.check_quote("some claim about acreage", None, None)
        assert r.verified == "no"
        assert r.reason == "no_source_pages_cited"

    def test_s_quote_mapping_matches_plan(self):
        assert settings.QUOTE_VERDICT_SCORES == {"yes": 1.0, "mixed": 0.5, "no": 0.0}

    def test_aggregate_verdict(self):
        mk = lambda v: qc.QuoteCheck(verified=v, score=0.0)
        assert qc.aggregate_verdict([mk("yes"), mk("yes")]) == "yes"
        assert qc.aggregate_verdict([mk("no"), mk("no")]) == "no"
        assert qc.aggregate_verdict([mk("yes"), mk("no")]) == "mixed"
        assert qc.aggregate_verdict([mk("mixed")]) == "mixed"

    def test_empty_evidence_list_scores_zero_not_one(self):
        """No evidence is not the same as verified evidence."""
        assert qc.aggregate_verdict([]) == "no"
        assert qc.s_quote_for([]) == 0.0

    def test_s_quote_is_averaged(self):
        mk = lambda v: qc.QuoteCheck(verified=v, score=0.0)
        assert qc.s_quote_for([mk("yes"), mk("no")]) == 0.5
        assert qc.s_quote_for([mk("yes"), mk("yes"), mk("yes"), mk("no")]) == 0.75


# --- Real-corpus behaviour --------------------------------------------------


class TestAgainstCorpus:
    def test_all_segment_a_verified_quotes_still_verify(self, doc_loader, m2_loader):
        """
        No regression against the existing exact-match verifier: every quote
        segment_a marked quote_verified must come back 'yes'.
        """
        doc = doc_loader(LINCOLN_HWY)
        m2 = m2_loader(LINCOLN_HWY)
        checked = 0
        for sub in m2.get("summary", {}).values():
            for ev in sub.get("evidence", []) or []:
                if not (ev.get("quote_verified") and ev.get("source_pages")):
                    continue
                r = qc.check_quote(ev["quote"], ev["source_pages"], doc)
                assert r.verified == "yes", (
                    f"regression: previously-verified quote now {r.verified} "
                    f"({r.reason}): {ev['quote'][:80]!r}"
                )
                checked += 1
        assert checked > 20, "fixture should exercise a meaningful number of quotes"

    def test_foreign_quote_is_rejected(self, doc_loader, m2_loader):
        """A quote from another document must not verify against this one."""
        target = doc_loader(LINCOLN_HWY)
        other = m2_loader(LA_TRANSIT)
        rejected = accepted = 0
        for sub in other.get("summary", {}).values():
            for ev in sub.get("evidence", []) or []:
                if not (ev.get("quote_verified") and ev.get("source_pages")):
                    continue
                r = qc.check_quote(ev["quote"], ev["source_pages"], target)
                if r.verified == "yes":
                    accepted += 1
                else:
                    rejected += 1
        assert rejected > 0
        # Some NEPA boilerplate genuinely recurs across documents, so allow a
        # small rate -- but never a majority.
        assert accepted / (accepted + rejected) < 0.05


class TestRegressionLincolnHwyWildlifeClause:
    """
    MCAL_PLAN 1(4) / 3.4: 'or important wildlife habitats are affected' is
    pure prior-injection -- Opus completing a plausible NEPA sentence. It must
    fail verification against the pages the summary actually cited.

    This is the case that motivated the coverage gate: somewhere in the document
    the clause scores 62.8 on `partial_ratio` alone, which clears the plan's 60.0
    'mixed' floor, while its content-token coverage is 0.00, which does not.

    Two corrections to what this docstring used to say, both from re-measuring
    after the MCAL_PLAN build-#4/#5 rerun:

      * Coverage is 0.00, not 0.20. It is not "low", it is EMPTY -- none of the
        clause's content tokens are anywhere in the searched window. (0.22 is the
        coverage of the whole three-item SENTENCE, whose two true items do carry
        tokens; see tests/test_lincoln_hwy_wildlife_clause.py.)
      * 62.8 is a property of the DOCUMENT, not of the cited-page window. The
        rerun changed which pages this subfield cites (10 distinct -> 15, and it
        dropped p.308 and p.52), and the best partial_ratio inside the cited
        window fell 62.8 -> 58.1 as a result. So on the new page set the char
        ratio alone would have rejected the clause -- by luck, not by design.
        `test_char_ratio_alone_would_have_passed_it` therefore measures the whole
        document, which is the page-set-independent form of the same claim, and
        `test_the_cited_window_ratio_moved_with_the_rerun` records the movement.

    Worth looking at what the 62.8 match actually is. On p.308 the matched span
    normalizes to "d 1mpr0vement w111 have 11tt1e affect 0n ex" -- OCR-mangled
    text with no relationship to wildlife habitats at all. That is the strongest
    possible argument for the coverage gate: `partial_ratio` in the 55-65 band
    against 1970s OCR is pure noise, and coverage is the channel that knows it.
    """

    CLAUSE = "or important wildlife habitats are affected"

    def test_rejected_on_cited_pages(self, doc_loader, graded_pages):
        doc = doc_loader(LINCOLN_HWY)
        pages = graded_pages(LINCOLN_HWY, "environmental_impact")
        assert pages, "fixture requires cited pages"
        r = qc.check_quote(self.CLAUSE, pages, doc)
        assert r.verified == "no", f"fabrication accepted as {r.verified}: {r.reason}"
        assert r.s_quote == 0.0

    def test_char_ratio_alone_would_have_passed_it(self, doc_loader):
        """
        Documents why the second gate exists. If this ever fails, the coverage
        gate has become redundant and can be reconsidered.

        Measured over the WHOLE document rather than the cited-page window. That
        is a strengthening, not a loosening: the whole document is the maximal
        search window, so its `partial_ratio` is the highest score the clause can
        achieve anywhere, and showing that the best possible char-ratio clears the
        mixed floor while coverage stays at 0.00 is the strongest form of the
        claim. It is also stable under re-runs, which the cited-window version was
        not.
        """
        doc = doc_loader(LINCOLN_HWY)
        r = qc.check_quote(
            self.CLAUSE, None, doc, search_whole_doc_if_no_pages=True
        )
        assert r.score == pytest.approx(62.79, abs=0.05)
        assert r.score >= settings.QUOTE_RATIO_MIXED
        assert r.coverage == 0.0
        assert r.coverage < settings.QUOTE_COVERAGE_MIXED
        assert r.matched_page == 308

    def test_the_cited_window_ratio_moved_with_the_rerun(
        self, doc_loader, graded_pages, graded_pages_pre_amendment
    ):
        """
        The measurement that used to be asserted, and what happened to it.

            cited-window best partial_ratio   coverage   clears 60.0 floor?
          pre-amendment (10 pages, incl 308)         62.79       0.00   yes
          post-amendment (15 pages, no 308)          58.14       0.00   no

        Both verdicts are 'no' either way, so nothing about the fabrication's
        rejection changed. What changed is WHICH gate does the rejecting inside
        the cited window: pre-amendment it took the coverage gate, post-amendment
        the char ratio would have sufficed on its own. That is an accident of the
        rerun citing different pages, not evidence that the coverage gate is
        redundant -- hence the whole-document form of the claim above.
        """
        doc = doc_loader(LINCOLN_HWY)
        old_pages = graded_pages_pre_amendment(LINCOLN_HWY, "environmental_impact")
        new_pages = graded_pages(LINCOLN_HWY, "environmental_impact")
        assert len(set(qc.coerce_pages(old_pages))) == 10
        assert len(set(qc.coerce_pages(new_pages))) == 15
        assert 308 in qc.coerce_pages(old_pages)
        assert 308 not in qc.coerce_pages(new_pages)

        old = qc.check_quote(self.CLAUSE, old_pages, doc)
        new = qc.check_quote(self.CLAUSE, new_pages, doc)
        assert old.score == pytest.approx(62.79, abs=0.05)
        assert new.score == pytest.approx(58.14, abs=0.05)
        assert old.score >= settings.QUOTE_RATIO_MIXED
        assert new.score < settings.QUOTE_RATIO_MIXED
        # The channel that rejects it regardless of the page set.
        assert old.coverage == new.coverage == 0.0
        assert old.verified == new.verified == "no"

    def test_the_top_scoring_match_is_ocr_noise(self, doc_loader):
        """
        Why a 62.8 `partial_ratio` must never be allowed to verify anything. The
        span on p.308 that scores 62.8 against "or important wildlife habitats are
        affected" is OCR-mangled boilerplate about an improvement having little
        effect, sharing not one content token with the clause.
        """
        doc = doc_loader(LINCOLN_HWY)
        r = qc.check_quote(
            self.CLAUSE, None, doc, search_whole_doc_if_no_pages=True
        )
        assert r.normalized_match_span is not None
        page, start, end = r.normalized_match_span
        assert page == 308
        page_text = next(p.text for p in doc.pages if p.page_num == page)
        matched = qc.normalize(page_text)[start:end]
        assert "w111 have 11tt1e affect" in matched
        assert "habitat" not in matched
        assert "wildlife" not in matched

    def test_rejected_even_against_whole_document(self, doc_loader):
        doc = doc_loader(LINCOLN_HWY)
        r = qc.check_quote(self.CLAUSE, None, doc, search_whole_doc_if_no_pages=True)
        assert r.verified == "no"


class TestRegressionEnvImpactMagnitude:
    """
    MCAL_PLAN 1(4) / 3.4: 'Magnitude 7.5' vs 'Magnitude 7.0' on the
    Newport-Inglewood fault (LA Transit).

    IMPORTANT -- this documents a plan defect rather than asserting the plan's
    expected behaviour. MCAL_PLAN 1(4) diagnoses this as a numeric
    hallucination caused by map-reduce 'decoupling' of
    (alternative_label <-> figure), fixable by atomic decomposition plus
    substring verification.

    That diagnosis does not survive contact with the document. Page 146 states
    verbatim: 'The most severe ground shaking ... would be generated by a
    Magnitude 7.5 earthquake occurring on the Newport-Inglewood Fault'. The
    model's pairing of 7.5 with Newport-Inglewood is therefore substring-TRUE
    and correctly coupled. The human's '7.0' comes from page 145, which frames
    7.0 as the *maximum credible* event.

    So the real defect is scope/qualifier loss, not decoupling -- and no
    substring-based verifier can catch it, because the claim is present in the
    source. These tests pin the actual, honest behaviour so the limitation
    stays visible instead of being mistaken for a passing gate.
    """

    CLAIM_75 = "a Magnitude 7.5 earthquake occurring on the Newport-Inglewood Fault"
    CLAIM_70 = "a Magnitude 7.0 earthquake occurring on the Newport-Inglewood Fault"

    def test_numeric_channel_separates_the_two_figures(self, doc_loader, graded_pages):
        doc = doc_loader(LA_TRANSIT)
        pages = graded_pages(LA_TRANSIT, "environmental_impact")
        r75 = qc.check_quote(self.CLAIM_75, pages, doc, require_numeric=True)
        r70 = qc.check_quote(self.CLAIM_70, pages, doc, require_numeric=True)
        assert r75.quote_numbers == ["7.5"]
        assert r70.quote_numbers == ["7.0"]
        assert r75.verified != r70.verified, (
            "the numeric channel must distinguish these two claims even though "
            "their prose is identical"
        )

    def test_the_graded_wrong_figure_is_substring_true(self, doc_loader, graded_pages):
        """
        The figure the human marked WRONG verifies, because the document says
        it. This is the uncatchable case; see the class docstring.
        """
        doc = doc_loader(LA_TRANSIT)
        pages = graded_pages(LA_TRANSIT, "environmental_impact")
        r = qc.check_quote(self.CLAIM_75, pages, doc, require_numeric=True)
        assert r.verified == "yes"
        assert r.missing_numbers == []

    def test_substituted_figure_is_flagged_numerically(self, doc_loader, graded_pages):
        doc = doc_loader(LA_TRANSIT)
        pages = graded_pages(LA_TRANSIT, "environmental_impact")
        r = qc.check_quote(self.CLAIM_70, pages, doc, require_numeric=True)
        assert r.missing_numbers == ["7.0"]
        assert r.verified == "mixed"
