"""
Tests for segment_b/year_adjudicator.py (MCAL_PLAN 3.13, 1(1), build item #15).

Covers OCR digit repair, short-date handling, signature/transmittal detection,
the priority ordering, the deterministic fallback, LLM-output validation, and
the real behaviour on the three docs MCAL_PLAN 1(1) says Segment A got wrong.

NO test here makes an LLM call: `adjudicate(..., call=fake)` injects the single
Sonnet call, and `TestNoNetwork` asserts that the injection point is the only one
by checking that the default callable is never reached without credentials.
"""

from __future__ import annotations

import json

import pytest

from mcal import settings
from segment_b import year_adjudicator as ya

from conftest import BUFFALO, LA_TRANSIT, LINCOLN_HWY, OPERATION_BREAKTHROUGH

FUEL_ECONOMY = "p1074_35556036861797"
AIRPORT_SPUR = "p1074_35556036546182"
RANDOLPH = "p1074_35556036105336"
BAD_CREEK = "p1074_35556036806586"


def _answer(**kw) -> dict:
    """A well-formed LLM response, overridable per test."""
    base = {
        "year": 1971,
        "source_type": "signature",
        "confidence": "high",
        "evidence_quote": "Date: June 1, 1971",
        "note": None,
    }
    base.update(kw)
    return base


def _recorder(response):
    """A fake `call` that records its arguments and returns `response`."""
    calls: list[dict] = []

    def _fn(system, user, **kw):
        calls.append({"system": system, "user": user, "kw": kw})
        if isinstance(response, Exception):
            raise response
        return response

    _fn.calls = calls
    return _fn


# --- OCR digit repair -------------------------------------------------------


class TestDigitRepair:
    @pytest.mark.parametrize("token,expected", [
        ("l972", 1972),
        ("197O", 1970),
        ("I97I", 1971),
        ("l97Z", 1972),
        ("197S", 1975),
        ("l980", 1980),
        ("2O05", 2005),
        ("1972", 1972),
    ])
    def test_repairs_named_in_the_plan(self, token, expected):
        assert ya.repair_year_token(token) == expected

    @pytest.mark.parametrize("token", ["1855", "1066", "2099", "18O5", "abcd", "", "197"])
    def test_out_of_range_is_not_guessed_at(self, token):
        assert ya.repair_year_token(token) is None

    def test_repair_is_length_preserving(self):
        text = "signed l97l on the l4th"
        assert len(ya.repair_ocr_years(text)) == len(text)

    def test_ordinary_words_are_untouched(self):
        """
        The reason this module does not reuse `quote_check.normalize`: that
        normalizer folds letters onto digits globally and would corrupt prose.
        """
        for word in ("loss", "Ross", "Illinois", "single", "goose", "IOWA"):
            assert ya.repair_ocr_years(word) == word

    def test_accession_numbers_cannot_contribute(self):
        assert ya.years_in("35556036861797") == []
        assert ya.years_in("p1074_35556039563135") == []

    def test_years_in_finds_repaired_years(self):
        assert ya.years_in("Draft l972, Final 197S, revised I97I") == [1972, 1975, 1971]

    def test_decade_plurals_are_excluded(self):
        assert ya.years_in("during the 1950s and 1960s") == []


class TestShortDates:
    @pytest.mark.parametrize("text,expected", [
        ("Date: 3/3/77", [1977]),
        ("Dated 12-5-75", [1975]),
        ("signed 1/2/05", [2005]),
    ])
    def test_two_digit_dates(self, text, expected):
        assert ya.years_in(text) == expected

    def test_unmappable_two_digit_year_is_dropped(self):
        """"40" would be 1940 or 2040; neither is in NEPA's range."""
        assert ya.years_in("3/3/40") == []

    def test_bare_two_digit_numbers_are_not_years(self):
        assert ya.years_in("77 comments were received") == []

    def test_citation_keys_are_not_dates(self):
        """"Duke Power Company. 1976c." is a bibliography entry."""
        assert ya.years_in("Duke Power Company. 1976c. Transmittal Letter") == []
        assert ya.years_in("published in 1976. Transmittal Letter") == [1976]


# --- Signature / transmittal detection --------------------------------------


class TestSignatureDetection:
    def test_signature_page_found(self, doc_factory):
        doc = doc_factory(
            "FINAL ENVIRONMENTAL IMPACT STATEMENT",
            "cover text with no date",
            "Approved: John W. Snow, Administrator\nDate: 3/3/77",
        )
        sig, trans = ya.find_signature_pages(doc)
        assert sig == [3]
        assert trans == []

    def test_transmittal_page_found(self, doc_factory):
        doc = doc_factory(
            "cover",
            "LETTER OF TRANSMITTAL\nJanuary 5, 1975\nDear Mr. Chairman:",
        )
        sig, trans = ya.find_signature_pages(doc)
        assert trans == [2]

    def test_prose_participle_is_not_a_signature(self, doc_factory):
        """
        MCAL_PLAN 1(1) lists `Approved` as a keyword; taken literally it fires on
        ordinary body prose and promotes body years above the cover date.
        """
        doc = doc_factory(
            "cover JUNE 1977",
            "A federal grant application was approved by UMTA in October 1976.",
        )
        sig, trans = ya.find_signature_pages(doc)
        assert sig == [] and trans == []
        assert ya.fallback_choice(ya.collect_candidates(doc)).year == 1977

    def test_keyword_far_from_the_year_does_not_count(self, doc_factory):
        filler = "x" * 400
        doc = doc_factory("cover 1975", f"Date: today\n{filler}\nin 1990 the volumes")
        sig, _ = ya.find_signature_pages(doc)
        assert sig == []

    def test_bare_date_needs_a_strong_keyword_on_the_page(self, doc_factory):
        """
        OCR eats the colon ("Date DCT 2 7 1975"), so a bare "date" is honoured --
        but only on a page a strong keyword already nominated.
        """
        weak_only = doc_factory("cover", "revised drawings bearing the date January 5, 1976")
        assert ya.find_signature_pages(weak_only) == ([], [])
        with_strong = doc_factory(
            "cover", "APPROVED AND ADOPTED BY\nDate DCT 2 7 1975\nState Highway Engineer"
        )
        assert with_strong and ya.find_signature_pages(with_strong)[0] == [2]


# --- Priority ordering ------------------------------------------------------


class TestPriority:
    def test_plan_ordering(self):
        """MCAL_PLAN 1(1): signature > transmittal > cover > body."""
        p = ya.SOURCE_PRIORITY
        assert p["signature"] > p["transmittal"] > p["cover"] > p["body"]

    def test_source_type_enum_matches_plan(self):
        assert set(ya.SOURCE_TYPES) == {
            "signature", "transmittal", "cover", "body", "adjudicated"
        }

    def test_signature_outranks_a_more_frequent_body_year(self, doc_factory):
        doc = doc_factory(
            "COVER 1972",
            "Approved: A. Smith\nDate: June 1, 1971",
            "1972 traffic. 1972 volumes. 1972 again. 1972 once more.",
        )
        chosen = ya.fallback_choice(ya.collect_candidates(doc))
        assert (chosen.year, chosen.source_type) == (1971, "signature")
        # ...and this is exactly where a plain mode goes wrong.
        assert ya.modal_year(ya.collect_candidates(doc)) == 1972

    def test_cover_outranks_body(self, doc_factory):
        doc = doc_factory("Prepared April, 1979", "in 1974 the county voted; 1974 again")
        chosen = ya.fallback_choice(ya.collect_candidates(doc))
        assert (chosen.year, chosen.source_type) == (1979, "cover")

    def test_closest_keyword_wins_inside_a_signature_tier(self, doc_factory):
        """
        The Fuel Economy case: "MODEL YEAR 1979" is printed all over the cover,
        but the publication date is "Date: 3/3/77" on the signature line.
        """
        doc = doc_factory(
            "AVERAGE FUEL ECONOMY STANDARD FOR MODEL YEAR 1979\n"
            "John W. Snow, Administrator\nDate: 3/3/77\n"
            "MODEL YEAR 1979 standards for MODEL YEAR 1979 vehicles",
        )
        chosen = ya.fallback_choice(ya.collect_candidates(doc))
        assert chosen.year == 1977

    def test_outlier_candidate_cannot_win_the_fallback(self, doc_factory):
        """"12/5/15" is OCR damage, but 2015 is in range and 1915 is not."""
        doc = doc_factory(
            "APPROVED AND ADOPTED BY\n12/5/15 Date State Highway Engineer",
            "prepared 1975; revised 1976; adopted 1975",
        )
        chosen = ya.fallback_choice(ya.collect_candidates(doc))
        assert chosen.year != 2015


# --- Candidate aggregation --------------------------------------------------


class TestCandidates:
    def test_aggregation_shape(self, doc_factory):
        doc = doc_factory("cover 1975", "Approved:\nDate: 1975\n1975 and 1976")
        rows = ya.aggregate_candidates(ya.collect_candidates(doc))
        assert rows and set(rows[0]) == {
            "year", "count", "pages", "source_types", "best_source"
        }
        assert rows[0]["best_source"] == "signature"

    def test_deduplicated_per_year_and_page(self, doc_factory):
        doc = doc_factory("1975 1975 1975 1975")
        assert len(ya.collect_candidates(doc)) == 1

    def test_empty_doc(self, doc_factory):
        doc = doc_factory("no dates at all here")
        assert ya.collect_candidates(doc) == []
        assert ya.fallback_choice([]) is None
        assert ya.modal_year([]) is None


# --- adjudicate() -----------------------------------------------------------


class TestAdjudicate:
    def test_output_schema_is_exactly_the_plan_s(self, doc_factory):
        doc = doc_factory("Approved:\nDate: June 1, 1971")
        out = ya.adjudicate(doc, call=_recorder(_answer()))
        assert set(out) == {
            "year", "source_type", "confidence", "candidates", "evidence_quote", "note"
        }
        assert out["year"] == 1971
        assert out["source_type"] == "signature"
        assert out["confidence"] == "high"

    def test_exactly_one_llm_call(self, doc_factory):
        doc = doc_factory("Approved:\nDate: June 1, 1971", "also 1971 and 1972 here")
        fake = _recorder(_answer())
        ya.adjudicate(doc, call=fake)
        assert len(fake.calls) == 1

    def test_prompt_states_the_priority_rule(self, doc_factory):
        doc = doc_factory("Approved:\nDate: June 1, 1971")
        fake = _recorder(_answer())
        ya.adjudicate(doc, call=fake)
        user = fake.calls[0]["user"]
        assert "OUTRANKS" in user
        for word in ("signature", "transmittal", "cover", "body"):
            assert word in user
        assert user.index("1. signature") < user.index("2. transmittal")
        assert user.index("2. transmittal") < user.index("3. cover")
        assert user.index("3. cover") < user.index("4. body")

    def test_prompt_is_token_bounded(self, doc_factory):
        """Many mentions must not turn into an unbounded prompt."""
        body = " ".join(f"in {1970 + (i % 30)} the volume rose" for i in range(400))
        doc = doc_factory("cover 1975", body, body, body, body)
        fake = _recorder(_answer(year=1975, source_type="cover"))
        ya.adjudicate(doc, call=fake)
        user = fake.calls[0]["user"]
        assert user.count("CONTEXT:") <= ya.MAX_CANDIDATES_IN_PROMPT
        assert len(user) < 20_000

    def test_m1_year_is_not_leaked_into_the_prompt(self, doc_factory):
        """An independent read is the whole point (MCAL_PLAN 3.13)."""
        doc = doc_factory("Approved:\nDate: June 1, 1971")
        fake = _recorder(_answer())
        out = ya.adjudicate(doc, m1_year=1999, call=fake)
        assert "1999" not in fake.calls[0]["user"]
        assert "disagrees_with_m1:1999" in out["note"]

    def test_m1_agreement_is_noted(self, doc_factory):
        doc = doc_factory("Approved:\nDate: June 1, 1971")
        out = ya.adjudicate(doc, m1_year=1971, call=_recorder(_answer()))
        assert "agrees_with_m1:1971" in out["note"]

    def test_no_candidates_skips_the_call(self, doc_factory):
        doc = doc_factory("this document carries no dates whatsoever")
        fake = _recorder(_answer())
        out = ya.adjudicate(doc, call=fake)
        assert fake.calls == []
        assert out["year"] is None
        assert out["source_type"] == "adjudicated"
        assert out["confidence"] == "low"
        assert out["candidates"] == []
        assert "no year candidates" in out["note"]


class TestRobustness:
    """MCAL_PLAN 3.13 output must survive any LLM behaviour. Nothing may raise."""

    @pytest.fixture
    def doc(self, doc_factory):
        return doc_factory(
            "COVER 1972", "Approved: A. Smith\nDate: June 1, 1971", "body 1972 and 1972"
        )

    @pytest.mark.parametrize("response", [
        {"year": "nineteen seventy one"},
        {"year": 1066},
        {"year": 2200},
        {"year": None},
        {},
        {"year": [1971]},
        "not a dict",
        None,
        42,
    ])
    def test_junk_falls_back_to_the_regex_candidate(self, doc, response):
        out = ya.adjudicate(doc, call=_recorder(response))
        assert out["year"] == 1971
        assert out["source_type"] == "signature"
        assert out["confidence"] == "low"
        assert "fell_back_to_regex_candidate" in out["note"]

    def test_exception_falls_back(self, doc):
        out = ya.adjudicate(doc, call=_recorder(RuntimeError("bedrock down")))
        assert out["year"] == 1971
        assert "llm_call_failed:RuntimeError" in out["note"]

    def test_string_year_is_coerced(self, doc):
        out = ya.adjudicate(doc, call=_recorder(_answer(year="1971")))
        assert out["year"] == 1971
        assert "fell_back" not in (out["note"] or "")

    def test_invalid_source_type_is_repaired_from_the_evidence(self, doc):
        out = ya.adjudicate(doc, call=_recorder(_answer(source_type="letterhead")))
        assert out["source_type"] == "signature"
        assert "invalid_source_type:letterhead" in out["note"]

    def test_invalid_confidence_is_downgraded(self, doc):
        out = ya.adjudicate(doc, call=_recorder(_answer(confidence="certain")))
        assert out["confidence"] == "low"
        assert "invalid_confidence:certain" in out["note"]

    def test_off_candidate_year_is_kept_but_flagged(self, doc):
        """The model may read text our repair missed -- but never confidently."""
        out = ya.adjudicate(doc, call=_recorder(_answer(year=1969)))
        assert out["year"] == 1969
        assert out["confidence"] == "low"
        assert "llm_year_not_in_regex_candidates" in out["note"]

    def test_ungrounded_evidence_quote_is_replaced(self, doc):
        out = ya.adjudicate(
            doc,
            call=_recorder(_answer(evidence_quote="the Secretary signed this on May 3, 1971")),
        )
        assert "evidence_quote_not_grounded_replaced" in out["note"]
        assert "Date" in out["evidence_quote"]

    def test_grounded_evidence_quote_is_kept(self, doc):
        quote = "Approved: A. Smith Date: June 1, 1971"
        out = ya.adjudicate(doc, call=_recorder(_answer(evidence_quote=quote)))
        assert out["evidence_quote"] == quote
        assert "not_grounded" not in (out["note"] or "")

    def test_missing_quote_is_backfilled_at_low_confidence(self, doc):
        out = ya.adjudicate(doc, call=_recorder(_answer(evidence_quote=None)))
        assert out["evidence_quote"]
        assert out["confidence"] == "low"
        assert "no_evidence_quote_returned" in out["note"]

    def test_body_only_evidence_cannot_be_high_confidence(self, doc_factory):
        doc = doc_factory("no date on the cover", "the 1974 election and the 1974 vote")
        out = ya.adjudicate(
            doc,
            call=_recorder(
                _answer(
                    year=1974,
                    source_type="body",
                    confidence="high",
                    evidence_quote="the 1974 election and the 1974 vote",
                )
            ),
        )
        assert out["confidence"] == "medium"
        assert "downgraded_high_to_medium" in out["note"]

    def test_llm_note_is_preserved(self, doc):
        out = ya.adjudicate(doc, call=_recorder(_answer(note="cover carried no date")))
        assert "cover carried no date" in out["note"]

    def test_modal_disagreement_is_recorded(self, doc):
        out = ya.adjudicate(doc, call=_recorder(_answer()))
        assert "modal_candidate_was_1972" in out["note"]

    def test_result_is_json_serialisable(self, doc):
        out = ya.adjudicate(doc, call=_recorder(_answer()))
        assert json.loads(json.dumps(out)) == out


class TestNoNetwork:
    def test_default_call_is_only_reached_when_not_injected(self, doc_factory, monkeypatch):
        """
        Guard against a future edit that ignores `call=`. We replace the module's
        default with a sentinel that raises, then confirm injection bypasses it.
        """
        def _boom(system, user, **kw):
            raise AssertionError("default LLM path must not be used when call= is given")

        monkeypatch.setattr(ya, "_default_call", _boom)
        doc = doc_factory("Approved:\nDate: June 1, 1971")
        assert ya.adjudicate(doc, call=_recorder(_answer()))["year"] == 1971


# --- Real corpus ------------------------------------------------------------


class TestAgainstCorpus:
    """
    MCAL_PLAN 1(1): 3/8 docs wrong, all pre-1980. These tests pin what the
    deterministic half of the adjudicator achieves BEFORE the LLM weighs in,
    which is what the fallback path is judged on.
    """

    # doc -> the year the human grade (or, where the grade was "ok", Segment A's
    # accepted M1 answer) says is correct.
    TARGETS = {
        RANDOLPH: 1973,
        AIRPORT_SPUR: 1976,
        BAD_CREEK: 1977,
        BUFFALO: 1977,
        FUEL_ECONOMY: 1977,
        LA_TRANSIT: 1979,      # human: wrong "1980", correct "1979"
    }
    # Docs whose graded-correct year does not occur anywhere in the OCR, so no
    # extractor working from this text can reach it. Operation Breakthrough is
    # graded "correct: latest date I could find was 1976" but the scan's highest
    # year is 1973; Lincoln Hwy is graded "I believe 1971" but the document's
    # only approval date is the FHWA memorandum "DATE: DEC 3 1976".
    UNREACHABLE = {OPERATION_BREAKTHROUGH: 1976, LINCOLN_HWY: 1971}

    def test_deterministic_fallback_hits_every_reachable_target(self, doc_loader):
        misses = {}
        for doc_id, target in self.TARGETS.items():
            chosen = ya.fallback_choice(ya.collect_candidates(doc_loader(doc_id)))
            got = chosen.year if chosen else None
            if got != target:
                misses[doc_id] = (got, target)
        assert misses == {}

    def test_tiered_fallback_beats_the_plain_mode(self, doc_loader):
        """
        Justifies the documented deviation in `fallback_choice`. MCAL_PLAN 3.13
        says "modal regex candidate"; measured on the graded docs the plain mode
        is right 3/8 and the priority-tiered rule 6/8.
        """
        tiered = modal = 0
        for doc_id, target in {**self.TARGETS, **self.UNREACHABLE}.items():
            mentions = ya.collect_candidates(doc_loader(doc_id))
            chosen = ya.fallback_choice(mentions)
            tiered += (chosen.year if chosen else None) == target
            modal += ya.modal_year(mentions) == target
        assert tiered > modal
        assert tiered >= 6

    def test_unreachable_targets_are_absent_from_the_candidate_set(self, doc_loader):
        """
        Documents why 2 of 8 cannot be fixed here, and how the two cases differ.

        Operation Breakthrough's graded year is absent from the OCR entirely.
        Lincoln Hwy's does occur, but only in body prose outside every window the
        plan specifies -- and the document's own approval block says 1976, so no
        priority rule faithful to MCAL_PLAN 1(1) can prefer 1971. If either
        assertion starts failing, the adjudicator should be re-measured.
        """
        ob_years = {
            y
            for p in doc_loader(OPERATION_BREAKTHROUGH).pages
            for y in ya.years_in(p.text)
        }
        assert self.UNREACHABLE[OPERATION_BREAKTHROUGH] not in ob_years

        for doc_id, target in self.UNREACHABLE.items():
            mentions = ya.collect_candidates(doc_loader(doc_id))
            assert target not in {m.year for m in mentions}, f"{doc_id}: re-tune"

    def test_la_transit_regression(self, doc_loader):
        """
        MCAL_PLAN 1(1): graded wrong "1980", correct "1979". The cover reads
        "April, 1979"; M1's first-3pp regex picked up a later in-body mention.
        """
        mentions = ya.collect_candidates(doc_loader(LA_TRANSIT))
        chosen = ya.fallback_choice(mentions)
        assert chosen.year == 1979
        assert chosen.source_type == "cover"
        assert "1979" in chosen.context

    def test_fuel_economy_short_date_regression(self, doc_loader):
        """
        The cover's big 4-digit year is "MODEL YEAR 1979"; the real date is the
        signature line "Date: 3/3/77". Without short-date support the module
        would answer 1979.
        """
        mentions = ya.collect_candidates(doc_loader(FUEL_ECONOMY))
        chosen = ya.fallback_choice(mentions)
        assert chosen.year == 1977
        assert chosen.source_type == "signature"
        assert "3/3/77" in chosen.context

    def test_signature_page_search_covers_the_whole_document(self, doc_loader):
        """
        MCAL_PLAN 1(1) diagnoses M1's failure as "regex on first-3pp only", so
        the keyword sweep must be able to look past the front matter.
        """
        found_late = False
        for doc_id in (*self.TARGETS, *self.UNREACHABLE):
            doc = doc_loader(doc_id)
            sig, trans = ya.find_signature_pages(doc)
            if any(p > 5 for p in (*sig, *trans)):
                found_late = True
        assert found_late

    def test_every_doc_produces_a_valid_result_with_a_stubbed_llm(self, doc_loader):
        for doc_id in (*self.TARGETS, *self.UNREACHABLE):
            doc = doc_loader(doc_id)
            out = ya.adjudicate(doc, call=_recorder("garbage"))
            assert out["source_type"] in ya.SOURCE_TYPES
            assert out["confidence"] in ya.CONFIDENCE_LEVELS
            assert out["year"] is None or (
                settings_year_min() <= out["year"] <= settings_year_max()
            )

    def test_s_source_agreement_is_a_fraction(self, doc_loader):
        doc = doc_loader(LA_TRANSIT)
        out = ya.adjudicate(doc, call=_recorder(_answer(year=1979, source_type="cover")))
        assert 0.0 <= ya.s_source_agreement(out, 1979) <= 1.0
        assert ya.s_source_agreement(out, 1979) > ya.s_source_agreement(out, 1985)
        assert ya.s_source_agreement({"year": None}, 1979) == 0.0
        # Weight 0 through stage v3 (MCAL_PLAN 3.3); logged, not acted on.
        assert settings.SIGNAL_WEIGHTS["s_source"] == 0.0


def settings_year_min() -> int:
    return ya.YEAR_MIN


def settings_year_max() -> int:
    return ya.YEAR_MAX


def test_year_bounds_come_from_segment_a_config():
    """One source of truth for the range (MCAL_PLAN 1(1): 1969-2026)."""
    assert (ya.YEAR_MIN, ya.YEAR_MAX) == (1969, 2026)
