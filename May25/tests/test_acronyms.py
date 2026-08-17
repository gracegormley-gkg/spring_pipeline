"""
Tests for segment_b/postproc/acronyms.py (MCAL_PLAN 3.8, 4 Q1, build item #3).

Covers the initials validator, the denylist, both parenthetical directions, the
glossary-section parser, first-use rewriting, idempotency, the T04 tagging path,
the s_acronym signal, and the commons artifact round-trip.

`TestAgainstCorpus` recomputes the empirical figures quoted in the module
docstring against the real per-page OCR, so those claims cannot rot silently.
No test in this file makes an LLM call, because the module cannot: MCAL_PLAN 4 Q1
requires this post-processor to be deterministic.
"""

from __future__ import annotations

import json

import pytest

from mcal import settings
from segment_b.postproc import acronyms as ac

from conftest import BUFFALO, LA_TRANSIT, LINCOLN_HWY, OPERATION_BREAKTHROUGH

# "Fuel Economy" -- the Evaluation CSV singles this one out as "heavy on
# undefined acronyms", so it is the module's primary regression fixture.
FUEL_ECONOMY = "p1074_35556036861797"
AIRPORT_SPUR = "p1074_35556036546182"


def _gloss(**entries) -> ac.Glossary:
    """A minimal Glossary with no commons, for post-pass unit tests."""
    return ac.Glossary(
        doc_id="test",
        entries={
            tok: ac.AcronymEntry(token=tok, expansion=exp, source=ac.SOURCE_PARENTHETICAL)
            for tok, exp in entries.items()
        },
        commons={},
    )


# --- Candidate detection ----------------------------------------------------


class TestCandidateDetection:
    def test_plan_regex_is_verbatim(self):
        """MCAL_PLAN 3.8 specifies the pattern; it must not drift."""
        assert ac.ACRONYM_RE.pattern == r"\b([A-Z][A-Z0-9]{1,}[A-Z0-9])\b"

    @pytest.mark.parametrize("text,expected", [
        ("the EIS says", ["EIS"]),
        ("NEPA and CEQ", ["NEPA", "CEQ"]),
        ("USACE issued a ROD", ["USACE", "ROD"]),
        ("NPA's were counted", ["NPA"]),          # possessive plural, 1970s style
        ("lowercase eis", []),
        ("GM and Ford", []),                       # 2 letters, not in commons
    ])
    def test_tokens_found(self, text, expected):
        assert [ac.canonical_token(o.token) for o in ac.iter_occurrences(text)] == expected

    def test_special_forms_are_not_split(self):
        """PM2.5 must not be detected as the meaningless token 'PM2'."""
        got = [o.token for o in ac.iter_occurrences("PM2.5 and NOx and SO2")]
        assert got == ["PM2.5", "NOx", "SO2"]

    def test_two_letter_tokens_only_when_curated(self):
        """EA is in the commons seed, so it is detectable; XY is not."""
        assert [o.token for o in ac.iter_occurrences("an EA was prepared")] == ["EA"]
        assert [o.token for o in ac.iter_occurrences("an XY was prepared")] == []


class TestDenylist:
    @pytest.mark.parametrize("token", ["III", "VII", "VIII", "XIV", "XX"])
    def test_roman_numerals(self, token):
        assert ac.is_denylisted(token)

    @pytest.mark.parametrize("token", ["SEC", "FIG", "TBL", "APPENDIX", "TABLE", "VOL"])
    def test_section_markers(self, token):
        assert ac.is_denylisted(token)

    @pytest.mark.parametrize(
        "token", ["THE", "AND", "NOT", "ALL", "ANY", "MAY", "SHALL", "FOR"]
    )
    def test_ordinary_english_named_in_the_task(self, token):
        assert ac.is_denylisted(token)

    def test_state_code_only_denied_with_place_context(self):
        assert ac.is_denylisted("OR", "Portland, ")
        assert not ac.is_denylisted("OR", "either this ")

    def test_real_acronyms_survive(self):
        for token in ("EIS", "NEPA", "LEDPA", "SHPO", "GVWR", "SEWRPC"):
            assert not ac.is_denylisted(token)


# --- Initials validation ----------------------------------------------------


class TestInitialsMatch:
    @pytest.mark.parametrize("token,expansion", [
        ("EIS", "Environmental Impact Statement"),
        ("BLM", "Bureau of Land Management"),
        ("LOS", "level of service"),                     # O comes from a stopword
        ("ROW", "right-of-way"),                         # hyphens split
        ("NPRM", "Notice of Proposed Rulemaking"),       # M is intra-word
        ("GHG", "greenhouse gases"),                     # two letters from one word
        ("USACE", "U.S. Army Corps of Engineers"),       # U.S. supplies U and S
        ("FONSI", "Finding of No Significant Impact"),
        ("SEWRPC", "Southeastern Wisconsin Regional Planning Commission"),
        ("NOx", "nitrogen oxides"),
    ])
    def test_accepts_real_pairs(self, token, expansion):
        assert ac.initials_match(token, expansion)

    @pytest.mark.parametrize("token,expansion", [
        ("EIS", "Bureau of Land Management"),
        ("BLM", "Environmental Impact Statement"),
        ("EIS", "statement of the case"),
        ("ADT", "and daily"),                    # leading stopword
        ("XYZ", "some entirely unrelated prose"),
    ])
    def test_rejects_implausible_pairs(self, token, expansion):
        assert not ac.plausible_expansion(token, expansion)

    def test_expansion_may_not_start_with_a_function_word(self):
        """Otherwise OPC harvests 'of Planning Coordination'."""
        assert not ac.plausible_expansion("OPC", "of Planning Coordination")
        assert ac.plausible_expansion("OPC", "Office of Planning Coordination")

    def test_one_letter_per_word_is_stricter_than_intra_word(self):
        """The mode `_expansion_before` tries first must reject a dropped word."""
        assert ac.initials_match("AADT", "average daily traffic")
        assert not ac.initials_match("AADT", "average daily traffic", max_letters_per_word=1)
        assert ac.initials_match("AADT", "annual average daily traffic", max_letters_per_word=1)

    def test_coverage_is_a_fraction(self):
        assert ac.initials_coverage("EIS", "Environmental Impact Statement") == 1.0
        assert ac.initials_coverage("SO2", "sulfur dioxide") == pytest.approx(0.5)
        assert ac.initials_coverage("EIS", "totally unrelated words here") == 0.0

    def test_cross_reference_is_not_a_definition(self):
        assert not ac.plausible_expansion("EIS", "FEIS")


# --- Parenthetical harvest --------------------------------------------------


class TestHarvestParentheticals:
    def test_direction_a_expansion_then_token(self):
        got, _ = ac.harvest_parentheticals(
            "prepared under the National Environmental Policy Act (NEPA) in 1975"
        )
        assert [(e.token, e.expansion) for e in got] == [
            ("NEPA", "National Environmental Policy Act")
        ]

    def test_direction_b_token_then_expansion(self):
        got, _ = ac.harvest_parentheticals(
            "issued a LEDPA (Least Environmentally Damaging Practicable Alternative) finding"
        )
        assert [(e.token, e.expansion) for e in got] == [
            ("LEDPA", "Least Environmentally Damaging Practicable Alternative")
        ]

    def test_leading_article_is_not_annexed(self):
        got, _ = ac.harvest_parentheticals("the Bureau of Land Management (BLM) manages")
        assert got[0].expansion == "Bureau of Land Management"

    def test_all_caps_definition_is_down_cased(self):
        """1970s cover pages are set in full caps; summaries should not shout."""
        got, _ = ac.harvest_parentheticals("BUREAU OF LAND MANAGEMENT (BLM)")
        assert got[0].expansion == "Bureau of Land Management"

    def test_possessive_is_stripped_from_the_expansion(self):
        got, _ = ac.harvest_parentheticals(
            "described in FEDERAL HIGHWAY ADMINISTRATION'S (FHWA) manual"
        )
        assert got[0].expansion == "Federal Highway Administration"

    def test_implausible_pair_is_rejected_and_logged(self):
        got, bad = ac.harvest_parentheticals(
            "comments were received during the review period (XQZ) last spring"
        )
        assert got == []
        assert bad and bad[0]["token"] == "XQZ"

    def test_sentence_boundary_stops_the_expansion(self):
        got, _ = ac.harvest_parentheticals(
            "That concluded the review. Environmental Impact Statement (EIS)"
        )
        assert got[0].expansion == "Environmental Impact Statement"

    def test_abbreviation_period_does_not_stop_the_expansion(self):
        got, _ = ac.harvest_parentheticals("filed by the U.S. Army Corps of Engineers (USACE)")
        assert got[0].expansion == "U.S. Army Corps of Engineers"

    def test_denylisted_token_is_never_harvested(self):
        got, _ = ac.harvest_parentheticals("Chapter Three, Alternatives Considered (III)")
        assert got == []


# --- Glossary section -------------------------------------------------------


class TestGlossarySection:
    @pytest.mark.parametrize("line", [
        "ACRONYMS AND ABBREVIATIONS",
        "Glossary",
        "LIST OF ACRONYMS",
        "3.1  Abbreviations",
        "LIST OF ACRONYNS",            # OCR damage
    ])
    def test_headings_matched(self, line):
        assert ac.looks_like_glossary_heading(line)

    @pytest.mark.parametrize("line", [
        "LIST OF FIGURES",
        "LIST OF TABLES",
        "List of Commenters",
        "PURPOSE AND NEED",
        "",
    ])
    def test_non_headings_rejected(self, line):
        assert not ac.looks_like_glossary_heading(line)

    def test_table_rows_parsed(self):
        rows, _ = ac.parse_glossary_rows(
            [
                "EIS     Environmental Impact Statement",
                "ROW ..... right-of-way",
                "LOS  -  level of service",
                "SHPO\tState Historic Preservation Officer",
            ]
        )
        assert {r.token: r.expansion for r in rows} == {
            "EIS": "Environmental Impact Statement",
            "ROW": "right-of-way",
            "LOS": "level of service",
            "SHPO": "State Historic Preservation Officer",
        }
        assert all(r.source == ac.SOURCE_GLOSSARY_SECTION for r in rows)

    def test_prose_lines_are_not_entries(self):
        rows, _ = ac.parse_glossary_rows(
            [
                "EIS     Environmental Impact Statement",
                "SHALL BE construed as meaning that the agency must act",
                "III     Chapter three of this document",
            ]
        )
        assert [r.token for r in rows] == ["EIS"]

    def test_relaxed_gate_allows_trailing_qualifiers(self):
        """Table layout is strong structural evidence; a strict match is too much."""
        rows, _ = ac.parse_glossary_rows(["ADT     average daily traffic count, 1975"])
        assert rows and rows[0].token == "ADT"

    def test_glossary_section_beats_parenthetical(self, doc_factory):
        doc = doc_factory(
            "cover page",
            "ACRONYMS\n\nSEWRPC   Southeastern Wisconsin Regional Planning Commission\n",
            "the Southeastern Wisconsin Regional Planning Council (SEWRPC) met",
        )
        g = ac.build_glossary(doc, commons={})
        assert g.source_for("SEWRPC") == ac.SOURCE_GLOSSARY_SECTION
        assert g.expand("SEWRPC").endswith("Commission")


# --- Commons ----------------------------------------------------------------


class TestCommons:
    def test_all_plan_tokens_present(self):
        """The ~40 entries MCAL_PLAN 3.8 names, verbatim."""
        required = """EIS NEPA CEQ EPA USACE NOAA USFWS USFS BLM DOT FHWA FAA ROD
            FONSI DEIS FEIS SEIS EA LEDPA NHPA ESA CWA CAA NAAQS PM2.5 VOC NOx
            SO2 MSAT GHG CO2 VMT HOV LOS ADT ROW DBE MBE SHPO THPO""".split()
        assert set(required) <= set(ac.COMMONS_SEED)
        assert len(ac.COMMONS_SEED) >= 40

    def test_every_commons_expansion_is_plausible(self):
        """Self-consistency: the curated seed must pass our own validator."""
        bad = [
            (t, e)
            for t, e in ac.COMMONS_SEED.items()
            if not ac.plausible_expansion(t, e, strict=False)
        ]
        assert bad == []

    def test_seed_artifact_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "ARTIFACTS_DIR", tmp_path)
        path = ac.write_commons_seed("v1")
        assert path.name == "acronym_commons.v1.json"
        payload = json.loads(path.read_text())
        assert set(payload) == {"acronyms"}
        assert all(set(r) == {"token", "expansion", "sources"} for r in payload["acronyms"])
        assert all(r["sources"] == [] for r in payload["acronyms"])
        loaded = ac.load_commons("v1")
        assert loaded["NEPA"].expansion == ac.COMMONS_SEED["NEPA"]
        assert loaded["NEPA"].source == ac.SOURCE_COMMONS

    def test_missing_artifact_falls_back_to_builtin(self, tmp_path, monkeypatch):
        """A missing commons file must degrade output, never halt a run."""
        monkeypatch.setattr(settings, "ARTIFACTS_DIR", tmp_path)
        assert ac.load_commons("v9")["EIS"].expansion == ac.COMMONS_SEED["EIS"]

    def test_corrupt_artifact_falls_back_to_builtin(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "ARTIFACTS_DIR", tmp_path)
        p = settings.artifact_path(ac.COMMONS_ARTIFACT_NAME, "v1")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not json")
        assert ac.load_commons("v1")["EIS"].expansion == ac.COMMONS_SEED["EIS"]

    def test_doc_glossary_takes_priority_over_commons(self, doc_factory):
        doc = doc_factory("the Environmental Impact Study (EIS) for this project")
        g = ac.build_glossary(doc, commons=ac.commons_entries())
        assert g.expand("EIS") == "Environmental Impact Study"
        assert g.source_for("EIS") == ac.SOURCE_PARENTHETICAL

    def test_commons_used_when_doc_is_silent(self, doc_factory):
        doc = doc_factory("the BLM administers the area")
        g = ac.build_glossary(doc, commons=ac.commons_entries())
        assert g.expand("BLM") == "Bureau of Land Management"
        assert g.source_for("BLM") == ac.SOURCE_COMMONS


# --- Post-pass --------------------------------------------------------------


class TestAnnotateField:
    def test_first_use_is_rewritten(self):
        g = _gloss(LEDPA="Least Environmentally Damaging Practicable Alternative")
        out, tags, st = ac.annotate_field("The LEDPA was selected.", g)
        assert out == (
            "The Least Environmentally Damaging Practicable Alternative (LEDPA) "
            "was selected."
        )
        assert tags == []
        assert st["rewritten"] == ["LEDPA"]

    def test_later_uses_are_left_alone(self):
        g = _gloss(EIS="Environmental Impact Statement")
        out, _, st = ac.annotate_field("This EIS supersedes the earlier EIS.", g)
        assert out == (
            "This Environmental Impact Statement (EIS) supersedes the earlier EIS."
        )
        assert out.count("Environmental Impact Statement") == 1
        assert st["n_occurrences"] == 2

    def test_already_defined_text_is_untouched(self):
        g = _gloss(EIS="Environmental Impact Statement")
        text = "The Environmental Impact Statement (EIS) evaluates the EIS scope."
        out, tags, st = ac.annotate_field(text, g)
        assert out == text
        assert st["already_defined"] == ["EIS"]
        assert st["rewritten"] == []

    def test_reverse_order_definition_is_recognised(self):
        g = _gloss(EIS="Environmental Impact Statement")
        text = "The EIS (Environmental Impact Statement) is attached."
        out, _, st = ac.annotate_field(text, g)
        assert out == text
        assert st["already_defined"] == ["EIS"]

    def test_ocr_damaged_existing_definition_still_counts(self):
        """`quote_check.normalize` reuse: 'Envir0nmenta1' must still match."""
        g = _gloss(EIS="Environmental Impact Statement")
        text = "The Envir0nmenta1 Impact 5tatement (EIS) is attached."
        out, _, st = ac.annotate_field(text, g)
        assert out == text
        assert st["already_defined"] == ["EIS"]

    def test_idempotent(self):
        g = _gloss(
            EIS="Environmental Impact Statement",
            NEPA="National Environmental Policy Act",
        )
        text = "Under NEPA, this EIS evaluates alternatives; the EIS is final."
        once, _, _ = ac.annotate_field(text, g)
        twice, _, st2 = ac.annotate_field(once, g)
        assert once == twice
        assert st2["rewritten"] == []
        assert sorted(st2["already_defined"]) == ["EIS", "NEPA"]

    def test_parenthesised_occurrence_does_not_nest_parens(self):
        g = _gloss(EIS="Environmental Impact Statement")
        out, _, _ = ac.annotate_field("the statement (EIS) says", g)
        assert out == "the statement (Environmental Impact Statement, EIS) says"
        assert "((" not in out
        again, _, _ = ac.annotate_field(out, g)
        assert again == out

    def test_unknown_acronym_is_tagged_never_rewritten(self):
        g = _gloss(EIS="Environmental Impact Statement")
        text = "The SCRTD and the EIS disagree."
        out, tags, st = ac.annotate_field(text, g)
        assert tags == [ac.TAG_UNDEFINED_ACRONYM]
        assert st["undefined"] == ["SCRTD"]
        assert "SCRTD" in out and "Southern" not in out   # no fabricated expansion
        assert ac.suggested_verdict(tags) == "PASS_WITH_NOTE"

    def test_tag_is_emitted_once_per_field(self):
        g = _gloss()
        _, tags, st = ac.annotate_field("SCRTD, LACTC and OCTD all commented.", g)
        assert tags == [ac.TAG_UNDEFINED_ACRONYM]
        assert st["undefined"] == ["SCRTD", "LACTC", "OCTD"]

    def test_clean_field_suggests_pass(self):
        g = _gloss(EIS="Environmental Impact Statement")
        _, tags, st = ac.annotate_field("This EIS is final.", g)
        assert st["suggested_verdict"] == "PASS"
        assert ac.suggested_verdict(tags) == "PASS"

    def test_ordinary_capitals_are_not_tagged(self):
        """
        A doc that uses 'BREAKTHROUGH' in a heading has not used an acronym.

        This is the evidence-based half of the ordinary-word filter, for words
        too document-specific to appear in the static denylist -- "Operation
        BREAKTHROUGH" is the real case.
        """
        g = ac.Glossary(
            doc_id="t", entries={}, commons={}, ordinary_words=frozenset({"breakthrough"})
        )
        _, tags, st = ac.annotate_field("BREAKTHROUGH housing prototypes", g)
        assert tags == []
        assert st["skipped_ordinary"] == ["BREAKTHROUGH"]
        assert st["n_occurrences"] == 0

    def test_statically_denylisted_words_never_reach_the_stats(self):
        _, tags, st = ac.annotate_field("PROPERTY values AND land use", _gloss())
        assert tags == []
        assert st["n_occurrences"] == 0
        assert st["skipped_ordinary"] == []

    def test_known_token_is_never_treated_as_ordinary(self):
        """LOS must survive in a document full of 'Los Angeles'."""
        g = ac.Glossary(
            doc_id="t",
            entries={},
            commons=ac.commons_entries(),
            ordinary_words=frozenset({"los"}),
        )
        out, tags, _ = ac.annotate_field("volumes at LOS D", g)
        assert out == "volumes at level of service (LOS) D"
        assert tags == []

    def test_rewrite_false_scores_without_editing(self):
        g = _gloss(EIS="Environmental Impact Statement")
        out, _, st = ac.annotate_field("This EIS is final.", g, rewrite=False)
        assert out == "This EIS is final."
        assert st["rewritten"] == ["EIS"]

    def test_empty_text(self):
        out, tags, st = ac.annotate_field("", _gloss())
        assert (out, tags) == ("", [])
        assert st["defined_first_use_rate"] == 1.0

    def test_annotate_record_is_per_field(self):
        """MCAL_PLAN 3.8 defines first use per output field, not per document."""
        g = _gloss(EIS="Environmental Impact Statement")
        texts, tags, stats = ac.annotate_record(
            {"a": "The EIS is final.", "b": "The EIS is long."}, g
        )
        assert texts["a"].startswith("The Environmental Impact Statement (EIS)")
        assert texts["b"].startswith("The Environmental Impact Statement (EIS)")
        assert stats["a"]["rewritten"] == ["EIS"]
        assert tags == {"a": [], "b": []}


# --- s_acronym --------------------------------------------------------------


class TestDefinedFirstUseRate:
    def test_all_defined(self):
        g = _gloss(EIS="Environmental Impact Statement")
        _, _, st = ac.annotate_field("This EIS is final.", g)
        assert ac.defined_first_use_rate(st) == 1.0

    def test_none_defined(self):
        _, _, st = ac.annotate_field("SCRTD commented.", _gloss())
        assert ac.defined_first_use_rate(st) == 0.0

    def test_partial(self):
        g = _gloss(EIS="Environmental Impact Statement")
        _, _, st = ac.annotate_field("The EIS and the SCRTD and the LACTC.", g)
        assert ac.defined_first_use_rate(st) == pytest.approx(1 / 3)

    def test_no_acronyms_scores_one_not_zero(self):
        """Plain-language prose must not be penalised (MCAL_PLAN 3.14)."""
        _, _, st = ac.annotate_field("The project would widen the road.", _gloss())
        assert ac.defined_first_use_rate(st) == 1.0

    def test_pooled_across_fields_not_averaged(self):
        g = _gloss(EIS="Environmental Impact Statement")
        _, _, a = ac.annotate_field("The EIS and the SCRTD and the LACTC.", g)
        _, _, b = ac.annotate_field("This EIS is final.", g)
        # pooled: 2 defined of 4 distinct; a mean of per-field rates would be 0.67
        assert ac.defined_first_use_rate([a, b]) == pytest.approx(0.5)

    def test_in_unit_interval(self):
        g = _gloss(EIS="Environmental Impact Statement")
        for text in ("", "no acronyms", "EIS EIS EIS", "SCRTD", "EIS and SCRTD"):
            r = ac.defined_first_use_rate(ac.annotate_field(text, g)[2])
            assert 0.0 <= r <= 1.0

    def test_signal_is_registered_at_weight_zero(self):
        """MCAL_PLAN 3.3 computes and logs s_acronym at weight 0 through v3."""
        assert settings.SIGNAL_WEIGHTS["s_acronym"] == 0.0


# --- Real corpus ------------------------------------------------------------


class TestAgainstCorpus:
    """
    Recomputes the empirical claims in the module docstring. These are the
    measurements MCAL_PLAN 6 expects to be reported, so they are asserted rather
    than trusted.
    """

    ALL_GRADED = (
        OPERATION_BREAKTHROUGH,
        "p1074_35556036105336",     # Randolph Urban Renewal
        AIRPORT_SPUR,
        "p1074_35556036806586",     # Bad Creek
        BUFFALO,
        FUEL_ECONOMY,
        LA_TRANSIT,
        LINCOLN_HWY,
    )

    def test_every_graded_doc_yields_a_glossary(self, doc_loader):
        """MCAL_PLAN 1(11): undefined acronyms in 8/8 docs. All 8 are fixable."""
        for doc_id in self.ALL_GRADED:
            g = ac.build_glossary(doc_loader(doc_id), commons=ac.commons_entries())
            assert len(g) >= 3, f"{doc_id} harvested only {len(g)} entries"

    def test_no_graded_doc_has_a_glossary_section(self, doc_loader):
        """
        Documented finding: these 1969-1980 statements predate the
        Acronyms/Glossary-section convention. If this ever fails, the corpus has
        grown and the section parser is now load-bearing.
        """
        for doc_id in self.ALL_GRADED:
            g = ac.build_glossary(doc_loader(doc_id), commons={})
            assert g.sections_found == [], f"{doc_id} unexpectedly has {g.sections_found}"

    def test_fuel_economy_specifics(self, doc_loader):
        """
        The Evaluation CSV calls this doc "heavy on undefined acronyms". Spot
        checks against the actual page text.
        """
        g = ac.build_glossary(doc_loader(FUEL_ECONOMY), commons=ac.commons_entries())
        expected = {
            "NHTSA": "National Highway Traffic Safety Administration",
            "GVWR": "gross vehicle weight rating",
            "AFES": "average fuel economy standard",
            "NPRM": "Notice of Proposed Rulemaking",   # M is intra-word
            "NPA": "nonpassenger automobiles",
            "AMC": "American Motors Corporation",
            "FEA": "Federal Energy Administration",
            "VMT": "vehicle miles traveled",
            "EGR": "exhaust gas recirculation",
            "CID": "cubic inch displacement",
        }
        for token, expansion in expected.items():
            assert g.expand(token) == expansion, f"{token} -> {g.expand(token)!r}"
            assert g.source_for(token) == ac.SOURCE_PARENTHETICAL

    def test_fuel_economy_summary_now_leaves_exactly_one_undefined_token(
        self, doc_loader, m2_loader
    ):
        """
        REGRESSION, recorded not hidden. Was `defined_first_use_rate == 1.0` with
        no tags; is now 16/18 = 0.889 with `T04_undefined_acronym` on two fields.

            metric                     pre-amendment   post-amendment
            defined_first_use_rate       1.0 (21/21)     0.889 (16/18)
            distinct undefined tokens    none            {'LUV'}
            fields tagged T04            0               2 of 6
            first-use rewrites           19              15

        The cause is entirely on the prose side -- the glossary is harvested from
        the document and the document did not change, so `build_glossary` returns
        exactly what it did before. What changed is that the amended, more
        concrete summary now names specific vehicles: "captive imports
        (foreign-built pickups like the Chevrolet LUV and Ford Courier ...)".

        "LUV" is a General Motors model designation. It occurs 5 times in the
        source and the source never expands it, so there is nothing for the
        parenthetical harvester to find and `g.expand("LUV")` is None. By the
        rule's own definition this is a correct detection: an ALL-CAPS token used
        in the summary that is not defined at first use. By intent it is a false
        positive -- LUV is a product name whose defining context is the word
        "Chevrolet" right before it, not an acronym a reader needs expanded.

        This test therefore asserts the honest split rather than either number
        alone: every token the glossary CAN expand is still defined at first use
        (the property the pipeline is responsible for), and the only shortfall is
        a token no glossary built from this document could ever expand.
        `test_fuel_economy_summary_was_clean_pre_amendment` keeps the before-state.
        """
        doc = doc_loader(FUEL_ECONOMY)
        g = ac.build_glossary(doc, commons=ac.commons_entries())
        fields = _summary_fields(m2_loader(FUEL_ECONOMY))
        assert fields, "fixture requires M2 summary output"
        texts, tags, stats = ac.annotate_record(fields, g)

        rate = ac.defined_first_use_rate(stats.values())
        assert rate == pytest.approx(16 / 18)
        assert rate < 1.0

        undefined = {t for s in stats.values() for t in s["undefined"]}
        assert undefined == {"LUV"}
        # The shortfall is unexpandable, not merely un-rewritten.
        assert g.expand("LUV") is None

        tagged = [f for f, t in tags.items() if t == [ac.TAG_UNDEFINED_ACRONYM]]
        assert sorted(tagged) == ["summary.affected_community", "summary.overview"]
        assert all(
            t in ([], [ac.TAG_UNDEFINED_ACRONYM]) for t in tags.values()
        )

        # Every acronym the glossary knows is still defined at first use.
        expandable = {
            t for s in stats.values()
            for t in (s["rewritten"] + s["already_defined"] + s["undefined"])
            if g.expand(t)
        }
        assert expandable
        assert not (expandable & undefined)

        n_rewritten = sum(len(s["rewritten"]) for s in stats.values())
        assert n_rewritten == 15, "was 19 pre-amendment; still a heavy rewrite"
        assert "National Highway Traffic Safety Administration (NHTSA)" in texts[
            "summary.overview"
        ]

    def test_fuel_economy_summary_was_clean_pre_amendment(
        self, doc_loader, m2_pre_amendment_loader
    ):
        """
        History preserved: the 1.0 the test above used to assert, against the
        corpus it was true of. Same document, same glossary, different prose.
        """
        doc = doc_loader(FUEL_ECONOMY)
        g = ac.build_glossary(doc, commons=ac.commons_entries())
        fields = _summary_fields(m2_pre_amendment_loader(FUEL_ECONOMY))
        assert fields, "fixture requires the pre-amendment M2 archive"
        texts, tags, stats = ac.annotate_record(fields, g)
        assert ac.defined_first_use_rate(stats.values()) == 1.0
        assert all(t == [] for t in tags.values())
        assert sum(len(s["rewritten"]) for s in stats.values()) == 19
        assert "National Highway Traffic Safety Administration (NHTSA)" in texts[
            "summary.overview"
        ]

    def test_luv_is_a_model_name_the_document_never_expands(self, doc_loader):
        """
        Pins the diagnosis so the regression above is actionable without re-reading
        the document. LUV appears 5 times in the source, always as a bare model
        name ("the Chevrolet LUV", "their LUV vehicle"), and never with a
        parenthetical or an inline expansion. No harvester reading only this
        document can define it, so the shortfall is not fixable in
        `segment_b/postproc/acronyms.py` -- it needs either a model-name stoplist
        or a commons entry, which is a judgement call for a human.
        """
        doc = doc_loader(FUEL_ECONOMY)
        text = doc.full_text or ""
        assert text.count("LUV") == 5
        g = ac.build_glossary(doc, commons=ac.commons_entries())
        assert g.expand("LUV") is None
        assert g.source_for("LUV") is None
        # It is not filtered as an ordinary English word either, which is why it
        # reaches the undefined bucket rather than being ignored.
        assert not g.is_probably_ordinary("LUV")

    def test_harvest_is_correct_on_hand_checked_examples(self, doc_loader):
        """
        Cases that caught real bugs, each verified against the page text.

        AADT/AASHO/EES exercise the one-letter-per-word-first search: with
        intra-word matching enabled from the start, each silently lost its
        leading word. OPC exercises the leading-function-word rule.
        """
        cases = {
            AIRPORT_SPUR: {
                "AADT": "annual average daily traffic",
                "SEWRPC": "Southeastern Wisconsin Regional Planning Commission",
            },
            "p1074_35556036806586": {"EES": "Energy Efficient Structure"},
            LINCOLN_HWY: {
                "AASHO": "American Association of State Highway Officials",
                "FHWA": "Federal Highway Administration",
                "NIPC": "Northeastern Illinois Planning Commission",
            },
            BUFFALO: {
                "OPC": "Office of Planning Coordination",
                "SUNYAB": "State University of New York at Buffalo",
                "SMSA": "Standard Metropolitan Statistical Area",
            },
            LA_TRANSIT: {
                "SCAG": "Southern California Association of Governments",
                "SCRTD": "Southern California Rapid Transit District",
                "LACBD": "Los Angeles Central Business District",
            },
        }
        for doc_id, expected in cases.items():
            g = ac.build_glossary(doc_loader(doc_id), commons={})
            for token, expansion in expected.items():
                assert g.expand(token) == expansion, f"{doc_id}:{token} -> {g.expand(token)!r}"

    def test_lowercase_evidence_separates_words_from_acronyms(self, doc_loader):
        """
        The measurement the ordinary-word filter rests on: real acronyms are
        never written in lowercase, ordinary words in ALL-CAPS headings are.
        """
        g = ac.build_glossary(doc_loader(BUFFALO), commons=ac.commons_entries())
        for token in ("EIS", "NEPA", "NFTA", "UMTA", "SMSA"):
            assert not g.is_probably_ordinary(token)
        assert g.ordinary_words, "expected some ALL-CAPS ordinary words in this doc"

    def test_postpass_is_idempotent_on_real_output(self, doc_loader, m2_loader):
        for doc_id in (FUEL_ECONOMY, LA_TRANSIT, BUFFALO, LINCOLN_HWY):
            g = ac.build_glossary(doc_loader(doc_id), commons=ac.commons_entries())
            for text in _summary_fields(m2_loader(doc_id)).values():
                once, _, _ = ac.annotate_field(text, g)
                twice, _, _ = ac.annotate_field(once, g)
                assert once == twice, f"{doc_id} post-pass not idempotent"

    def test_rewrite_only_inserts_text_and_preserves_the_rest(self, doc_loader, m2_loader):
        """A post-processor must never lose extracted prose."""
        doc_id = LA_TRANSIT
        g = ac.build_glossary(doc_loader(doc_id), commons=ac.commons_entries())
        for text in _summary_fields(m2_loader(doc_id)).values():
            out, _, st = ac.annotate_field(text, g)
            assert len(out) >= len(text)
            for token in st["rewritten"]:
                assert f"({token})" in out or f", {token}" in out

    def test_undefined_acronyms_are_reported_not_invented(self, doc_loader, m2_loader):
        """
        Where a doc genuinely never defines an acronym, we must tag rather than
        guess. Lincoln Hwy's summary uses L10 and PPM with no definition
        anywhere in the document.
        """
        g = ac.build_glossary(doc_loader(LINCOLN_HWY), commons=ac.commons_entries())
        fields = _summary_fields(m2_loader(LINCOLN_HWY))
        _, tags, stats = ac.annotate_record(fields, g)
        undefined = {t for s in stats.values() for t in s["undefined"]}
        assert undefined, "expected undefined acronyms in this doc"
        assert any(t == [ac.TAG_UNDEFINED_ACRONYM] for t in tags.values())
        assert ac.defined_first_use_rate(stats.values()) < 1.0


def _summary_fields(m2: dict) -> dict[str, str]:
    """The summary.* prose fields of an M2 output, as {field_key: text}."""
    out: dict[str, str] = {}
    for key, value in (m2.get("summary") or {}).items():
        text = value.get("text") if isinstance(value, dict) else value
        if isinstance(text, str) and text.strip():
            out[f"summary.{key}"] = text
    return out
