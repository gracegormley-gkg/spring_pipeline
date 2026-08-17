"""
Tests for mcal/atomic_verify.py (MCAL_PLAN 3.4, build item #6).

No LLM, no network: `decompose`'s `call=` parameter is injected everywhere, and
every deterministic validator is exercised directly.

Three groups worth calling out:

  * `TestCorpusValidators` pins the measured behaviour of the coreference and
     coordination validators over the real graded corpus. Those numbers are quoted
     in the module's own comments, so a loosening of the rules shows up here
     rather than as a silently noisier failure log.
  * `TestScopeQualifier` covers the documented deviation from MCAL_PLAN 1(3)/1(4).
     The end-to-end regressions for it live in the two files the plan names:
     tests/test_lincoln_hwy_wildlife_clause.py and
     tests/test_env_impact_magnitude_75_vs_70.py.
  * `TestFalseNegativeAudit` covers the advisory-at-v1 / gating-from-v2 switch,
     which is the only part of this module a build orchestrator must branch on.

CORPUS PROVENANCE. Every real-corpus number in this file was re-measured after
MCAL_PLAN build items #4/#5 (plain-language + concreteness clause, plus the new
`summary_of_interest` field) regenerated all 8 graded docs. No rule in
`mcal/atomic_verify.py` changed; the prose the rules run over is ~1.7x longer and
carries 37% more citations. The pre-amendment corpus is archived at
`segment_a/output/m2_pre_amendment/`, and this file measures BOTH, because:

  * the human grades in `Evaluation - Sheet1.csv` describe the pre-amendment
    prose, so any label-conditioned statistic over the current artifacts pairs
    labels with text they do not describe (see TestT01Invisibility's docstring);
  * two properties genuinely regressed (coordination-splitter precision 1/1 ->
    1/3, and the splitter no longer firing on the wildlife-clause sentence at
    all), and a before/after pair is the only way to state that honestly.

Classes reading the archive: `TestCorpusValidatorsPreAmendment`,
`TestCoordinationSplitterNoLongerFiresOnTheWildlifeSentence`,
`TestT01PreAmendmentIsTheConsistentPairing`, and the paired assertions inside
`TestT01Invisibility`.
"""

from __future__ import annotations

import json

import pytest

from mcal import atomic_verify as av
from mcal import grades as grades_mod
from mcal import quote_check as qc
from mcal import settings

from conftest import LA_TRANSIT, LINCOLN_HWY


# --- Template ---------------------------------------------------------------


class TestTemplate:
    def test_body_is_after_the_rule(self):
        """Same header/`---`/body convention as templates/m2_plain_language.md."""
        body = av.load_prompt_template()
        assert body
        assert "documented DEVIATION" not in body, (
            "the human-facing provenance header leaked into the model prompt"
        )
        assert body.startswith("You decompose")

    def test_mandatory_rules_present_verbatim(self):
        body = av.load_prompt_template()
        for fragment in av.REQUIRED_RULE_FRAGMENTS:
            assert fragment in body, f"MCAL_PLAN 3.4 requires {fragment!r} verbatim"

    def test_missing_template_fails_loudly(self, tmp_path):
        with pytest.raises(av.TemplateError, match="Missing atomic decomposition"):
            av.load_prompt_template(tmp_path / "nope.md")

    def test_gutted_template_fails_loudly(self, tmp_path):
        p = tmp_path / "atomic_decomposition.md"
        p.write_text("# header\n\n---\n\nSplit stuff up.\n", encoding="utf-8")
        with pytest.raises(av.TemplateError, match="missing mandatory rule text"):
            av.load_prompt_template(p)


# --- Coreference validator --------------------------------------------------


class TestCoreference:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("It comprises 1,086 homeowners.", ["it"]),
            ("They would fail to eliminate blight.", ["they"]),
            ("The project would include a 318-acre reservoir.", ["the project"]),
            ("The agency revised the EIS.", ["the agency"]),
            ("This would displace 78 structures.", ["this"]),
            ("Those are the preferred routes.", ["those"]),
        ],
    )
    def test_flags_unresolved(self, text, expected):
        assert av.unresolved_references(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            # Determiner uses, not anaphora.
            "This Draft EIS evaluates eleven transit alternatives.",
            "Those alternatives were rejected.",
            "Alternative V has a capital cost of $659 million.",
            # Relative-clause `that`.
            "The alignment that crosses the fault was dropped.",
            # Expletive `it`.
            "It is expected that noise levels would exceed the standard.",
            # Compound noun, not anaphora.
            "The alternatives section was praised by the Commission.",
            "The project area covers 55 square miles.",
            # Self-resolving: the antecedent is named in the same text.
            "The Department of Natural Resources found the department's review "
            "adequate.",
            "The Bad Creek Project would operate the project at full load.",
        ],
    )
    def test_does_not_flag_resolved(self, text):
        assert av.unresolved_references(text) == []

    def test_empty(self):
        assert av.unresolved_references("") == []
        assert av.unresolved_references(None) == []

    def test_validator_overrides_the_model(self):
        """
        MCAL_PLAN 3.4 makes `coreference_resolved` model-supplied. A model that
        forgets the rule reports True, so the deterministic check has to be able
        to contradict it.
        """
        atom = av.Atom(
            id="a", text="They would be displaced.", coreference_resolved=True,
            page=1, evidence_quote="They would be displaced.",
        )
        problems = av.validate_atom(atom)
        assert any(p.startswith("unresolved_reference:they") for p in problems)

    def test_model_flag_alone_also_fails(self):
        atom = av.Atom(
            id="a", text="Alternative V costs $369 million.",
            coreference_resolved=False, page=1, evidence_quote="x" * 25,
            claim_type="numeric",
        )
        assert "coreference_unresolved_by_decomposer" in av.validate_atom(atom)


# --- Negation validator -----------------------------------------------------


class TestNegation:
    def test_plan_cue_list_is_exact(self):
        """MCAL_PLAN 3.4 enumerates these; the schema rule quotes them."""
        assert av.NEGATION_CUES == (
            "not", "no", "neither", "never", "without", "except", "unless",
            "fails to", "does not", "would not",
        )

    @pytest.mark.parametrize(
        "text,cue",
        [
            ("No wetlands are affected.", "no"),
            ("The project does not affect wetlands.", "does not"),
            ("Groundwater would not be degraded.", "would not"),
            ("Neither alternative was selected.", "neither"),
            ("Construction proceeds without mitigation.", "without"),
            ("The plan fails to address noise.", "fails to"),
        ],
    )
    def test_detects_cues(self, text, cue):
        assert cue in av.negation_cues(text)

    def test_multiword_cue_reported_once(self):
        """'does not' must not also report the bare 'not' inside it."""
        assert av.negation_cues("It does not apply.") == ["does not"]

    def test_expected_polarity(self):
        assert av.expected_polarity("No wetlands are affected.") == "negative"
        assert av.expected_polarity("Wetlands are affected.") == "affirmative"

    def test_affirmative_atom_for_negated_claim_is_rejected(self):
        """MCAL_PLAN 3.4: 'Do NOT emit an affirmative-polarity atom for a
        negated claim.'"""
        atom = av.Atom(
            id="a", text="No wetlands are affected.", polarity="affirmative",
            page=1, evidence_quote="No wetlands are affected here.",
        )
        problems = av.validate_atom(atom)
        assert any(p.startswith("negation_dropped_from_polarity") for p in problems)

    def test_negative_atom_evidence_must_carry_the_cue(self):
        atom = av.Atom(
            id="a", text="No wetlands are affected.", polarity="negative",
            page=1, evidence_quote="Wetlands in the corridor were surveyed.",
        )
        assert "negation_cue_absent_from_evidence_quote" in av.validate_atom(atom)

    def test_negative_atom_needs_evidence_at_all(self):
        atom = av.Atom(
            id="a", text="No wetlands are affected.", polarity="negative", page=1,
        )
        assert "negative_atom_without_evidence_quote" in av.validate_atom(atom)

    def test_negation_cue_must_appear_in_the_source(self, doc_factory):
        """
        MCAL_PLAN 3.4 step 2: for negative polarity the cue must be present in
        the matched text. A page that asserts the positive must not verify the
        negative, even though almost every word matches.
        """
        doc = doc_factory(
            "Unique wetlands in the corridor are affected by the improvement "
            "and require a permit under Section 404."
        )
        atom = av.Atom(
            id="a",
            text="No unique wetlands in the corridor are affected.",
            polarity="negative",
            page=1,
            evidence_quote="No unique wetlands in the corridor are affected",
        )
        r = av.verify_atom(atom, doc)
        assert r.status == av.STATUS_FAIL
        assert "negation_cue_absent_from_source" in r.reasons


# --- Coordination splitting -------------------------------------------------


class TestCoordination:
    """
    Unit tests of the rule itself, on hand-written sentences.

    NOTE: `WILDLIFE` below is the PRE-AMENDMENT Lincoln Hwy sentence. It is kept
    as the rule's canonical positive case because it is the shape the rule was
    written for, but it is no longer the sentence in
    `segment_a/output/m2/p1074_35556039563135.json` -- the rerun appended a second
    clause to it and the rule now declines it. Do not read a pass here as "the
    corpus still splits". See
    `TestCoordinationSplitterNoLongerFiresOnTheWildlifeSentence`.
    """

    WILDLIFE = (
        "No National Register sites, unique wetlands, or important wildlife "
        "habitats are affected."
    )

    def test_the_wildlife_sentence_splits_into_three(self):
        got = av.coordinated_claims(self.WILDLIFE)
        assert got == [
            "No National Register sites are affected.",
            "No unique wetlands are affected.",
            "No important wildlife habitats are affected.",
        ]

    def test_shared_negator_is_preserved_on_every_split(self):
        for claim in av.coordinated_claims(self.WILDLIFE):
            assert av.expected_polarity(claim) == "negative"

    def test_and_list(self):
        got = av.coordinated_claims(
            "Conservation, rate revision, and power purchase were determined "
            "not to be reasonable."
        )
        assert len(got) == 3
        assert got[0].startswith("Conservation were determined")

    def test_two_item_list(self):
        assert av.coordinated_claims("Wetlands and habitats are affected.") == [
            "Wetlands are affected.",
            "Habitats are affected.",
        ]

    @pytest.mark.parametrize(
        "sentence,why",
        [
            ("The project would affect wetlands and habitats.",
             "coordinated OBJECTS are out of scope"),
            ("Rare and endangered species could be affected.",
             "coordinated ADJECTIVES share one head noun"),
            ("Jackson and Transylvania in NC) had a 1970 population of 140,990.",
             "unbalanced bracket means a mid-parenthetical fragment"),
            ("The Department of Natural Resources found no impact on fish or "
             "wildlife habitat and no rare species would be affected.",
             "a finite verb in the subject span means multiple clauses"),
            ("Species including the green salamander and Oconee bells could be "
             "affected.",
             "'including' introduces an appositive, not a coordination"),
            ("Wetlands are affected, and noise would increase.",
             "the predicate joins a second independent clause"),
            ("The alignment that crosses the fault or the ridge was dropped.",
             "a subordinator in the subject span"),
            ("Wetlands are affected.", "no coordinator at all"),
            ("", "empty"),
        ],
    )
    def test_refuses_out_of_scope_shapes(self, sentence, why):
        assert av.coordinated_claims(sentence) == [], why

    def test_gap_detection_finds_the_unsplit_clause(self):
        """
        The failure mode the validator exists for: a decomposer that emits the
        whole sentence as one atom hides the fabrication behind its two true
        neighbours.
        """
        atoms = [av.Atom(id="a", text=self.WILDLIFE)]
        gaps = av.coordination_gaps(self.WILDLIFE, atoms)
        assert any("wildlife habitats" in g for g in gaps)
        assert len(gaps) == 3

    def test_no_gap_when_properly_split(self):
        atoms = [
            av.Atom(id=f"a{i}", text=c)
            for i, c in enumerate(av.coordinated_claims(self.WILDLIFE))
        ]
        assert av.coordination_gaps(self.WILDLIFE, atoms) == []


# --- Scope qualifier (documented deviation) ---------------------------------


class TestScopeQualifier:
    @pytest.mark.parametrize(
        "text",
        [
            "Capital costs range from $659 million to $1,450 million.",
            "The project would displace up to 723 structures.",
            "The most severe ground shaking would reach Magnitude 7.5.",
            "Rail alternatives save more than 29,000 barrels per year.",
            "Costs are at least $369 million.",
            "The largest impact is on MacArthur Park.",
        ],
    )
    def test_scoped_claims_require_a_qualifier(self, text):
        atom = av.Atom(id="a", text=text, claim_type="numeric")
        assert av.requires_scope_qualifier(atom)
        problems = av.validate_atom(atom)
        assert any(p.startswith("scope_qualifier_missing") for p in problems)

    @pytest.mark.parametrize(
        "text",
        [
            "Alternative V has a capital cost of $659 million.",
            "The project requires 16.2 acres of right-of-way.",
            "The corridor crosses Cook County, Illinois.",
        ],
    )
    def test_unscoped_claims_do_not(self, text):
        atom = av.Atom(id="a", text=text, claim_type="numeric")
        assert not av.requires_scope_qualifier(atom)

    def test_comparative_type_always_requires_one(self):
        atom = av.Atom(id="a", text="Noise levels exceed the standard.",
                       claim_type="comparative")
        assert av.requires_scope_qualifier(atom)

    def test_mistyping_as_numeric_does_not_evade_the_check(self):
        """A model calling a range `numeric` must not opt out of the check."""
        atom = av.Atom(
            id="a", text="Costs range from $659 million to $1,450 million.",
            claim_type="numeric",
        )
        assert av.requires_scope_qualifier(atom)

    def test_page_qualifiers_detected(self):
        found = dict(
            (n, p)
            for n, p in av.page_qualifiers(
                "For the Rail/Bus Alternatives I-V, the capital costs in 1977 "
                "dollars range from 659 million and 1.120 billion dollars."
            )
        )
        assert "scoped_to_alternatives" in found
        assert "constant_dollars" in found

    def test_bare_in_the_area_is_not_a_qualifier(self):
        """
        Boilerplate guard. "nor in the area of this improvement" appears in
        almost every Lincoln Hwy impact sentence; treating it as a restriction
        made a TRUE atom fail.
        """
        names = [n for n, _p in av.page_qualifiers("nor in the area of this improvement")]
        assert "geographic_scope" not in names

    def test_qualified_geographic_scope_is_a_qualifier(self):
        names = [
            n for n, _p in av.page_qualifiers("felt in the starter line area")
        ]
        assert "geographic_scope" in names

    def test_carried_qualifier_accepted_when_rephrased(self):
        assert av.qualifier_is_carried(
            "the maximum credible earthquake",
            "The maximum credible event on the fault is Magnitude 7.0.",
        )

    def test_dropped_qualifier_detected(self):
        assert not av.qualifier_is_carried(
            "most severe ground shaking that would be felt in the starter line area",
            "The Newport-Inglewood Fault has a Magnitude of 7.5.",
        )

    def test_qualifier_verification_is_localized(self):
        ok, why = av.qualifier_verifies(
            "For the Rail/Bus Alternatives I-V",
            "(In Millions of 1977 Dollars) TOTAL BORED SUBWAY RAPID TRANSIT SYSTEM",
        )
        assert not ok
        assert why.startswith("qualifier_token_overlap")

    def test_qualifier_verification_accepts_a_real_hit(self):
        ok, _why = av.qualifier_verifies(
            "For the Rail/Bus Alternatives I-V",
            "For the Rail/Bus Alternatives I-V, the capital costs range from "
            "659 million and 1.120 billion dollars.",
        )
        assert ok

    def test_unverifiable_qualifier_fails_the_atom(self, doc_factory):
        doc = doc_factory(
            "(In Millions of 1977 Dollars) TOTAL BORED SUBWAY RAPID TRANSIT "
            "SYSTEM 1,035 1,120 923 849 659"
        )
        atom = av.Atom(
            id="a",
            text="Capital costs range from $659 million to $1,450 million.",
            page=1,
            evidence_quote="TOTAL BORED SUBWAY RAPID TRANSIT SYSTEM 1,035 1,120 "
                           "923 849 659",
            claim_type="comparative",
            scope_qualifier="For the Rail/Bus Alternatives I-V",
        )
        r = av.verify_atom(atom, doc)
        assert r.status == av.STATUS_FAIL
        assert any(p.startswith("scope_qualifier_unverified") for p in r.reasons)
        assert "T19_scope_qualifier_dropped" in r.tags

    def test_t19_exists_in_the_taxonomy(self):
        """The tag this module emits must be a real code, not a string literal."""
        from mcal import taxonomy

        tax = taxonomy.seed_taxonomy("v1", include_proposed=True)
        assert tax.by_name("T19_scope_qualifier_dropped") is not None

    def test_no_spurious_t19_on_an_atom_whose_quote_is_absent(self, doc_factory):
        """
        Locality gate. When the evidence quote is not on the page at all, the
        "located source sentences" are whatever shared a token or two, so any
        superlative found in them is unrelated. The atom already fails; a
        spurious `scope_qualifier_dropped` would inflate T19 in
        `reason_code_counts`, which is the diagnostic the next prompt revision is
        read from.
        """
        doc = doc_factory(
            "The most severe congestion would occur at the Wilshire and La Brea "
            "intersection during the afternoon peak hour."
        )
        atom = av.Atom(
            id="a",
            text="Tribal consultation was completed with the Ho-Chunk Nation.",
            page=1,
            evidence_quote="Tribal consultation was completed with the Ho-Chunk "
                           "Nation under Section 106.",
            claim_type="prose",
        )
        r = av.verify_atom(atom, doc)
        assert r.status == av.STATUS_FAIL
        assert r.dropped_qualifiers == []
        assert not any(p.startswith("scope_qualifier") for p in r.reasons)
        assert r.competing_values == []


# --- Type-specific checks ---------------------------------------------------


class TestTypeSpecificChecks:
    def test_numeric_mismatch_fails(self, doc_factory):
        doc = doc_factory(
            "The maximum credible earthquake assigned to the Newport-Inglewood "
            "zone is a Magnitude 7.0 event."
        )
        atom = av.Atom(
            id="a", text="The Newport-Inglewood zone has a Magnitude 7.5 event.",
            page=1, claim_type="numeric",
            evidence_quote="the Newport-Inglewood zone is a Magnitude 7.5 event",
        )
        r = av.verify_atom(atom, doc)
        assert r.status == av.STATUS_FAIL
        assert any(p.startswith("numeric_mismatch") for p in r.reasons)
        assert "T02_numeric_hallucination" in r.tags

    def test_numeric_match_passes(self, doc_factory):
        doc = doc_factory(
            "The total number of acres of land to be taken is approximately "
            "16.2 acres of right-of-way."
        )
        atom = av.Atom(
            id="a", text="Approximately 16.2 acres of right-of-way are taken.",
            page=1, claim_type="numeric",
            evidence_quote="The total number of acres of land to be taken is "
                           "approximately 16.2 acres",
        )
        assert av.verify_atom(atom, doc).status == av.STATUS_PASS

    def test_numeric_claim_without_a_figure_is_mistyped(self):
        atom = av.Atom(
            id="a", text="Noise levels are elevated.", claim_type="numeric",
            page=1, evidence_quote="Noise levels are elevated near the corridor.",
        )
        assert "claim_type_numeric_without_figure" in av.validate_atom(atom)

    def test_units_in(self):
        assert av.units_in("16.2 acres of right-of-way") == ["acres"]
        assert av.units_in("$659 million") == ["million"]
        assert "%" in av.units_in("only 0.1% improvement")
        assert av.units_in("no figures") == []

    def test_missing_unit_downgrades_but_does_not_fail(self, doc_factory):
        """
        Soft channel. OCR'd cost tables carry bare integers whose unit header is
        on another page, so a hard failure here would be a false-negative
        factory.
        """
        doc = doc_factory("TOTAL BORED SUBWAY RAPID TRANSIT SYSTEM 1,035 1,120 659")
        atom = av.Atom(
            id="a", text="The cost is 659 million.", page=1, claim_type="numeric",
            evidence_quote="TOTAL BORED SUBWAY RAPID TRANSIT SYSTEM 1,035 1,120 659",
        )
        r = av.verify_atom(atom, doc)
        assert r.status == av.STATUS_PARTIAL
        assert any(p.startswith("unit_absent") for p in r.reasons)
        assert r.score == 0.5

    def test_temporal_within_one_year_passes(self, doc_factory):
        doc = doc_factory(
            "Predicted 1990 design-year noise levels exceed the FHWA standard "
            "at forty-six of forty-nine sites."
        )
        atom = av.Atom(
            id="a", text="Design-year noise levels are predicted for 1989.",
            page=1, claim_type="temporal",
            evidence_quote="Predicted 1990 design-year noise levels exceed the "
                           "FHWA standard",
        )
        assert av.verify_atom(atom, doc).status == av.STATUS_PASS

    def test_temporal_outside_tolerance_fails(self, doc_factory):
        doc = doc_factory(
            "Predicted 1990 design-year noise levels exceed the FHWA standard "
            "at forty-six of forty-nine sites."
        )
        atom = av.Atom(
            id="a", text="Design-year noise levels are predicted for 1975.",
            page=1, claim_type="temporal",
            evidence_quote="Predicted 1990 design-year noise levels exceed the "
                           "FHWA standard",
        )
        r = av.verify_atom(atom, doc)
        assert r.status == av.STATUS_FAIL
        assert any(p.startswith("temporal_out_of_tolerance") for p in r.reasons)

    def test_temporal_without_a_year_is_judged_on_the_quote(self, doc_factory):
        doc = doc_factory(
            "The reservoir would operate continuously at full load for "
            "thirty-two hours over a thirty year licence term."
        )
        atom = av.Atom(
            id="a", text="The licence term is thirty years.", page=1,
            claim_type="temporal",
            evidence_quote="over a thirty year licence term",
        )
        ok, why = av._temporal_within_tolerance(atom, doc.pages[0].text)
        assert ok and why == ""


# --- Aggregation ------------------------------------------------------------


class TestAggregation:
    def _res(self, status, claim_type="prose"):
        return av.AtomResult(
            atom=av.Atom(id="x", text="t", claim_type=claim_type), status=status
        )

    def test_all_pass(self):
        assert av.aggregate_score([self._res("pass")] * 3) == 1.0

    def test_all_fail(self):
        assert av.aggregate_score([self._res("fail")] * 3) == 0.0

    def test_partial_scores_half(self):
        assert av.aggregate_score([self._res("partial")]) == 0.5

    def test_plain_mean_when_no_penalized_failures(self):
        got = av.aggregate_score([self._res("pass"), self._res("fail")])
        assert got == 0.5

    def test_numeric_failure_carries_double_weight(self):
        """MCAL_PLAN 3.4: 2x penalty on numeric and geospatial failures."""
        plain = av.aggregate_score([self._res("pass"), self._res("fail", "prose")])
        penalized = av.aggregate_score(
            [self._res("pass"), self._res("fail", "numeric")]
        )
        assert penalized < plain
        # 1*1.0 + 2*0.0 over weight 3
        assert penalized == pytest.approx(1 / 3)

    def test_geospatial_failure_also_penalized(self):
        assert av.aggregate_score(
            [self._res("pass"), self._res("fail", "geospatial")]
        ) == pytest.approx(1 / 3)

    def test_penalty_does_not_apply_to_passes(self):
        assert av.aggregate_score([self._res("pass", "numeric")]) == 1.0

    def test_score_stays_in_unit_interval(self):
        for n in range(1, 8):
            got = av.aggregate_score([self._res("fail", "numeric")] * n)
            assert 0.0 <= got <= 1.0

    def test_no_atoms_scores_zero(self):
        """An undecomposable subfield is not a verified one."""
        assert av.aggregate_score([]) == 0.0


# --- s_citation -------------------------------------------------------------


class TestSCitation:
    def test_fraction_with_a_page_cite(self):
        atoms = [
            av.Atom(id="a", text="x", page=12),
            av.Atom(id="b", text="y", page=None),
            av.Atom(id="c", text="z", page=14),
            av.Atom(id="d", text="w", page=None),
        ]
        assert av.s_citation(atoms) == 0.5

    def test_empty_is_none_not_zero(self):
        """
        None means 'not measured', 0.0 means 'measured, nothing cited'.
        `confidence.compute_signals(citation_rate=None)` relies on that.
        """
        assert av.s_citation([]) is None

    def test_accepts_results_as_well_as_atoms(self):
        r = av.AtomResult(atom=av.Atom(id="a", text="x", page=1), status="pass")
        assert av.s_citation([r]) == 1.0

    def test_s_quote_cannot_see_a_missing_citation(self):
        """
        The finding this signal exists for. `quote_check.s_quote_for` averages
        over the evidence that EXISTS, so a subfield whose two cited claims both
        verify scores 1.0 whether or not two further claims were left uncited.
        Atom-level citation coverage is what separates them.
        """
        two_good = [qc.QuoteCheck(verified="yes", score=100.0)] * 2
        assert qc.s_quote_for(two_good) == 1.0

        cited = [av.Atom(id=f"a{i}", text="x", page=1) for i in range(2)]
        uncited = [av.Atom(id=f"b{i}", text="y", page=None) for i in range(2)]
        assert av.s_citation(cited) == 1.0
        assert av.s_citation(cited + uncited) == 0.5

    def test_verified_variant_is_stricter(self, doc_factory):
        doc = doc_factory("The corridor crosses Cook County, Illinois.")
        good = av.verify_atom(
            av.Atom(id="a", text="The corridor crosses Cook County, Illinois.",
                    page=1,
                    evidence_quote="The corridor crosses Cook County, Illinois."),
            doc,
        )
        bad = av.verify_atom(
            av.Atom(
                id="b",
                text="Comment letters raised concerns about tribal consultation.",
                page=1,
                evidence_quote="Comment letters raised concerns about tribal "
                               "consultation and treaty fishing rights.",
            ),
            doc,
        )
        assert bad.check is not None and bad.check.verified == "no"
        assert av.s_citation([good, bad]) == 1.0
        assert av.s_citation_verified([good, bad]) == 0.5

    def test_uncited_atom_fails_and_is_tagged_t01(self, doc_factory):
        doc = doc_factory("Some page text about the corridor and its impacts.")
        r = av.verify_atom(av.Atom(id="a", text="A claim.", page=None), doc)
        assert r.status == av.STATUS_FAIL
        assert "uncited_no_page" in r.reasons
        assert "T01_missing_citation" in r.tags


# --- Decomposition (injected call) -----------------------------------------


def _atom_payload(**kw) -> dict:
    base = {
        "text": "Alternative V has a capital cost of $659 million.",
        "subject": "Alternative V",
        "predicate": "has a capital cost of",
        "object": "$659 million",
        "page": 214,
        "evidence_quote": "TOTALS - Alternate #5 9.0 11 $ 615.0 $44.0 $659.0 M",
        "claim_type": "numeric",
        "polarity": "affirmative",
        "coreference_resolved": True,
        "scope_qualifier": None,
    }
    base.update(kw)
    return base


class TestDecompose:
    def test_injected_call_is_used(self, doc_factory):
        seen = {}

        def call(system, user, **kw):
            seen["system"] = system
            seen["user"] = user
            return {"atoms": [_atom_payload()]}

        atoms, raws, warnings = av.decompose(
            "summary.project_description",
            "Capital costs range from $659 million.",
            [{"quote": "x", "source_pages": ["214"]}],
            doc_factory("page one text"),
            call=call,
        )
        assert warnings == []
        assert len(atoms) == 1 and len(raws) == 1
        assert "Split this passage into minimal factual claims" in seen["system"]
        assert "PASSAGE TO DECOMPOSE" in seen["user"]

    def test_ids_are_assigned_by_us_not_the_model(self, doc_factory):
        def call(system, user, **kw):
            return {"atoms": [_atom_payload(id="whatever-the-model-said")] * 2}

        atoms, _r, _w = av.decompose(
            "summary.environmental_impact", "text", [], doc_factory("p"), call=call
        )
        assert [a.id for a in atoms] == [
            "summary_environmental_impact.a01",
            "summary_environmental_impact.a02",
        ]

    def test_ids_are_stable_across_reruns(self, doc_factory):
        def call(system, user, **kw):
            return {"atoms": [_atom_payload(), _atom_payload()]}

        first = av.decompose("summary.overview", "t", [], doc_factory("p"), call=call)[0]
        second = av.decompose("summary.overview", "t", [], doc_factory("p"), call=call)[0]
        assert [a.id for a in first] == [a.id for a in second]

    def test_bare_list_response_accepted(self, doc_factory):
        atoms, _r, w = av.decompose(
            "summary.overview", "t", [], doc_factory("p"),
            call=lambda s, u, **kw: [_atom_payload()],
        )
        assert len(atoms) == 1 and w == []

    def test_call_failure_is_a_warning_not_an_exception(self, doc_factory):
        def boom(system, user, **kw):
            raise RuntimeError("bedrock throttled")

        atoms, raws, warnings = av.decompose(
            "summary.overview", "text", [], doc_factory("p"), call=boom
        )
        assert atoms == [] and raws == []
        assert warnings == ["decomposition_call_failed:RuntimeError"]

    def test_prose_response_is_a_warning(self, doc_factory):
        _a, _r, w = av.decompose(
            "summary.overview", "t", [], doc_factory("p"),
            call=lambda s, u, **kw: {"commentary": "I could not comply"},
        )
        assert w == ["decomposition_returned_NoneType_not_list"]

    def test_empty_passage_short_circuits(self):
        called = []
        av.decompose("summary.overview", "   ", [], None,
                     call=lambda *a, **k: called.append(1))
        assert called == []

    def test_bad_enums_fall_back_to_defaults_and_are_recorded(self):
        raw = _atom_payload(claim_type="quantitative", polarity="neg")
        atom = av.atom_from_dict(raw, atom_id="a")
        assert atom.claim_type == av.DEFAULT_CLAIM_TYPE
        assert atom.polarity == av.DEFAULT_POLARITY
        problems = av.validate_atom(atom, raw)
        assert "invalid_claim_type:quantitative" in problems
        assert "invalid_polarity:neg" in problems

    def test_page_shapes_coerced(self):
        assert av.atom_from_dict(_atom_payload(page="214"), atom_id="a").page == 214
        assert av.atom_from_dict(_atom_payload(page="214-215"), atom_id="a").page == 214
        assert av.atom_from_dict(_atom_payload(page="n/a"), atom_id="a").page is None
        assert av.atom_from_dict(_atom_payload(page=None), atom_id="a").page is None

    def test_missing_coreference_key_defaults_true(self):
        raw = _atom_payload()
        del raw["coreference_resolved"]
        assert av.atom_from_dict(raw, atom_id="a").coreference_resolved is True

    def test_evidence_window_is_in_the_prompt(self, doc_factory):
        doc = doc_factory(*[f"page {i} body text" for i in range(1, 9)])
        user = av.build_decomposition_user_prompt(
            "summary.overview", "some prose",
            [{"quote": "q", "source_pages": [4]}], doc,
        )
        for p in (2, 3, 4, 5, 6):
            assert f"[[PAGE {p}]]" in user
        assert "[[PAGE 1]]" not in user
        assert "[[PAGE 7]]" not in user

    def test_uncited_passage_tells_the_model_not_to_guess(self, doc_factory):
        user = av.build_decomposition_user_prompt(
            "summary.public_response", "prose", [], doc_factory("p")
        )
        assert "Do not\nguess page numbers." in user or "guess page numbers" in user


# --- Subfield / document orchestration --------------------------------------


class TestVerifySubfield:
    def test_injected_atoms_skip_the_llm(self, doc_factory):
        doc = doc_factory("The corridor crosses Cook County, Illinois.")
        atoms = [
            av.Atom(id="a", text="The corridor crosses Cook County, Illinois.",
                    page=1,
                    evidence_quote="The corridor crosses Cook County, Illinois.")
        ]
        fv = av.verify_subfield(
            "doc", "summary.affected_community",
            {"text": "prose", "evidence": []}, doc, atoms=atoms,
        )
        assert fv.n_atoms == 1
        assert fv.score == 1.0
        assert fv.bucket == "summary_narrative"

    def test_bucket_comes_from_settings(self, doc_factory):
        fv = av.verify_subfield(
            "doc", "summary.environmental_impact", {"text": "", "evidence": []},
            doc_factory("p"), atoms=[],
        )
        assert fv.bucket == settings.bucket_for_field("summary.environmental_impact")
        assert fv.bucket == "summary_numeric"

    def test_doc_id_is_normalized(self, doc_factory):
        fv = av.verify_subfield(
            "P0491_ABC", "summary.overview", {"text": "", "evidence": []},
            doc_factory("p"), atoms=[],
        )
        assert fv.doc_id == "p0491_abc"

    def test_under_split_passage_reports_a_warning(self, doc_factory):
        passage = (
            "No National Register sites, unique wetlands, or important wildlife "
            "habitats are affected."
        )
        fv = av.verify_subfield(
            "doc", "summary.environmental_impact",
            {"text": passage, "evidence": []}, doc_factory("some page"),
            atoms=[av.Atom(id="a", text=passage, page=1, evidence_quote=passage)],
        )
        assert any(w.startswith("coordination_under_split") for w in fv.warnings)
        assert len(fv.coordination_gaps) == 3

    def test_missing_doc_fails_every_atom(self):
        fv = av.verify_subfield(
            "doc", "summary.overview", {"text": "t", "evidence": []}, None,
            atoms=[av.Atom(id="a", text="x", page=1, evidence_quote="y" * 25)],
        )
        assert fv.score == 0.0
        assert "no_document_supplied" in fv.results[0].reasons

    def test_verify_document_covers_the_plan_scope(self, doc_factory):
        doc = doc_factory("page text about the corridor")
        m2 = {
            "summary": {
                sub.split(".", 1)[1]: {"text": "prose", "evidence": []}
                for sub in settings.SUMMARY_SUBFIELDS
            },
            "summary_of_interest": [],
        }
        dv = av.verify_document(
            "doc", m2, doc, call=lambda s, u, **kw: {"atoms": []}
        )
        assert set(dv.fields) == set(settings.ATOMIC_VERIFY_FIELDS)
        assert "summary.overview" not in dv.fields, (
            "overview is excluded to avoid double-counting the subfields' atoms"
        )

    def test_citation_rates_shape_matches_confidence_api(self, doc_factory):
        from mcal import confidence

        doc = doc_factory("The corridor crosses Cook County, Illinois.")
        dv = av.DocumentVerification(doc_id="d")
        dv.fields["summary.overview"] = av.verify_subfield(
            "d", "summary.overview", {"text": "t", "evidence": []}, doc,
            atoms=[
                av.Atom(id="a", text="x", page=1, evidence_quote="y" * 25),
                av.Atom(id="b", text="z", page=None),
            ],
        )
        rates = dv.citation_rates()
        assert rates["summary.overview"] == 0.5
        sig = confidence.compute_signals(
            "summary.overview",
            quote_verdict="yes",
            critic_verdict="PASS",
            citation_rate=rates["summary.overview"],
        )
        assert sig.s_citation == 0.5
        # Weight 0 today (MCAL_PLAN 3.3), so it must not move the composite.
        assert confidence.composite(sig) == 1.0


# --- summary_of_interest ----------------------------------------------------


class TestSummaryOfInterest:
    ENTRY = {
        "claim": "The document records disagreement between the Department of "
                 "the Interior and Duke Power over mitigation.",
        "salience_criterion": "contested",
        "page": 1,
        "evidence_quote": "The Department of the Interior could not fully endorse "
                          "the project until adverse impacts are mitigated.",
        "why_notable": "Interior withheld endorsement pending mitigation.",
    }

    @pytest.fixture
    def soi_doc(self, doc_factory):
        return doc_factory(
            "The Department of the Interior could not fully endorse the Bad Creek "
            "Project until adverse impacts are mitigated, Duke Power having "
            "declined to expand the mitigation package. Interior withheld "
            "endorsement pending mitigation of the identified adverse impacts."
        )

    def test_claim_decomposition_is_a_no_op(self, soi_doc):
        """
        MCAL_PLAN 3.4: each entry is already claim-shaped, so decomposition is a
        no-op for `claim` -- it goes straight to per-atom verification. Asserted
        by counting LLM calls: the claim must cost none.
        """
        calls = []

        def call(system, user, **kw):
            calls.append(user)
            return {"atoms": []}

        av.verify_summary_of_interest("d", [self.ENTRY], soi_doc, call=call)
        assert len(calls) == 1, "exactly one call, for why_notable"
        assert "Interior withheld endorsement" in calls[0]
        assert self.ENTRY["claim"] not in calls[0]

    def test_claim_atom_is_built_without_a_model(self):
        atom = av.claim_atom(self.ENTRY, index=0)
        assert atom.source == "claim_verbatim"
        assert atom.text == self.ENTRY["claim"]
        assert atom.page == 1
        assert atom.id.endswith("c")

    def test_claim_type_inferred_for_a_scoped_claim(self):
        atom = av.claim_atom(
            {"claim": "The largest impact is 1,200 acres.", "page": 1}, index=0
        )
        assert atom.claim_type == "comparative"

    def test_claim_type_inferred_numeric(self):
        atom = av.claim_atom({"claim": "Costs are $369 million.", "page": 1}, index=0)
        assert atom.claim_type == "numeric"

    def test_t17_when_claim_verifies_but_why_notable_does_not(self, soi_doc):
        """MCAL_PLAN 3.4: that exact asymmetry is T17_manufactured_salience."""
        bad_why = [
            av.Atom(
                id="w1",
                text="Interior objections of this kind are unusual in NEPA practice.",
                page=1,
                evidence_quote="Interior objections of this kind are unusual in "
                               "NEPA practice.",
            )
        ]
        fv, entries = av.verify_summary_of_interest(
            "d", [self.ENTRY], soi_doc, why_atoms_by_entry={0: bad_why}
        )
        e = entries[0]
        assert e.claim_ok
        assert not e.why_ok
        assert e.manufactured_salience
        assert av.T17 in fv.tags

    def test_no_t17_when_both_verify(self, soi_doc):
        good_why = [
            av.Atom(
                id="w1",
                text="Interior withheld endorsement pending mitigation.",
                page=1,
                evidence_quote="Interior withheld endorsement pending mitigation "
                               "of the identified adverse impacts.",
            )
        ]
        _fv, entries = av.verify_summary_of_interest(
            "d", [self.ENTRY], soi_doc, why_atoms_by_entry={0: good_why}
        )
        assert entries[0].claim_ok and entries[0].why_ok
        assert not entries[0].manufactured_salience

    def test_no_t17_when_the_claim_itself_fails(self, soi_doc):
        """
        Both failing is a plain unsupported entry, not manufactured salience.
        Collapsing the two would make T17's rate meaningless.
        """
        entry = dict(self.ENTRY, evidence_quote="A wholly invented sentence "
                                                "about tribal consultation.")
        _fv, entries = av.verify_summary_of_interest(
            "d", [entry], soi_doc, why_atoms_by_entry={0: []}
        )
        assert not entries[0].claim_ok
        assert not entries[0].manufactured_salience

    def test_empty_why_notable_is_not_ok(self, soi_doc):
        """MCAL_PLAN 3.15 rule 1 requires why_notable to be grounded."""
        entry = dict(self.ENTRY, why_notable="")
        _fv, entries = av.verify_summary_of_interest(
            "d", [entry], soi_doc, call=lambda s, u, **kw: {"atoms": []}
        )
        assert entries[0].manufactured_salience
        assert "why_notable_empty" in entries[0].warnings

    def test_empty_list_is_annotated_not_penalized(self, soi_doc):
        """
        MCAL_PLAN 3.15 rule 2: an empty list is a CORRECT result for a routine
        document. The score is 0.0 because there is nothing to verify, and the
        warning is what stops a reader treating that as a defect.
        """
        fv, entries = av.verify_summary_of_interest("d", [], soi_doc)
        assert entries == []
        assert fv.n_atoms == 0
        assert fv.score == 0.0
        assert any("CORRECT result for a routine document" in w for w in fv.warnings)

    def test_non_dict_entry_is_reported(self, soi_doc):
        fv, entries = av.verify_summary_of_interest(
            "d", ["just a string"], soi_doc
        )
        assert entries == []
        assert "soi_entry_0_not_an_object" in fv.warnings

    def test_page_recovered_from_nested_evidence(self):
        entry = {
            "claim": "A claim.",
            "evidence": [{"quote": "q", "source_pages": ["147"]}],
        }
        assert av.claim_atom(entry, index=0).page == 147


# --- False-negative audit ---------------------------------------------------


class _FakeGrade:
    def __init__(self, correct, raw="ok"):
        self.correct = correct
        self.raw_grade = raw


class _FakeGradeSet:
    def __init__(self, mapping):
        self._m = mapping

    def get(self, doc_id, field):
        return self._m.get((doc_id, field))


def _dv_with(doc_id, field, statuses):
    dv = av.DocumentVerification(doc_id=doc_id)
    dv.fields[field] = av.FieldVerification(
        doc_id=doc_id,
        field=field,
        bucket=settings.bucket_for_field(field),
        results=[
            av.AtomResult(atom=av.Atom(id=f"a{i}", text=f"t{i}", page=1), status=s)
            for i, s in enumerate(statuses)
        ],
    )
    return dv


class TestFalseNegativeAudit:
    FIELD = "summary.public_response"

    def test_only_correctly_graded_subfields_count(self):
        dv_ok = _dv_with("d1", self.FIELD, ["pass", "fail"])
        dv_wrong = _dv_with("d2", self.FIELD, ["fail", "fail"])
        gs = _FakeGradeSet(
            {
                ("d1", self.FIELD): _FakeGrade(True),
                ("d2", self.FIELD): _FakeGrade(False, "wrong: ..."),
            }
        )
        audit = av.false_negative_audit([dv_ok, dv_wrong], gs, stage="v1")
        assert audit["n_atoms_on_correct_subfields"] == 2
        assert audit["n_false_negatives"] == 1
        assert audit["false_negative_rate"] == 0.5

    def test_advisory_at_v1(self):
        dv = _dv_with("d1", self.FIELD, ["fail"] * 5)
        gs = _FakeGradeSet({("d1", self.FIELD): _FakeGrade(True)})
        audit = av.false_negative_audit([dv], gs, stage="v1")
        assert audit["gating"] is False
        assert audit["exceeds_ceiling"] is True
        assert audit["flagged"] is False, "v1 is advisory (MCAL_PLAN 3.4)"

    def test_gating_from_v2(self):
        dv = _dv_with("d1", self.FIELD, ["fail"] * 5)
        gs = _FakeGradeSet({("d1", self.FIELD): _FakeGrade(True)})
        audit = av.false_negative_audit([dv], gs, stage="v2")
        assert audit["gating"] is True
        assert audit["flagged"] is True

    def test_under_ceiling_is_not_flagged_at_v2(self):
        dv = _dv_with("d1", self.FIELD, ["pass"] * 19 + ["fail"])
        gs = _FakeGradeSet({("d1", self.FIELD): _FakeGrade(True)})
        audit = av.false_negative_audit([dv], gs, stage="v2")
        assert audit["false_negative_rate"] == 0.05
        assert audit["flagged"] is False

    def test_ceiling_matches_settings(self):
        assert settings.ATOMIC_FALSE_NEGATIVE_CEILING == 0.10
        assert settings.ATOMIC_FALSE_NEGATIVE_GATING_FROM_STAGE == 2

    def test_small_sample_gets_a_note(self):
        dv = _dv_with("d1", self.FIELD, ["pass"] * 8)
        gs = _FakeGradeSet({("d1", self.FIELD): _FakeGrade(True)})
        audit = av.false_negative_audit([dv], gs, stage="v1")
        assert any("advisory" in n for n in audit["notes"])

    def test_no_grades_is_reported_not_treated_as_clean(self):
        dv = _dv_with("d1", self.FIELD, ["fail"])
        audit = av.false_negative_audit([dv], None, stage="v2")
        assert audit["false_negative_rate"] is None
        assert audit["flagged"] is False
        assert any("No atoms from correctly-graded" in n for n in audit["notes"])

    def test_partial_counts_as_a_failure(self):
        """
        Conservative by choice: over-reporting candidate false negatives triggers
        the prompt review the log exists for; under-reporting hides them.
        """
        dv = _dv_with("d1", self.FIELD, ["partial"])
        gs = _FakeGradeSet({("d1", self.FIELD): _FakeGrade(True)})
        audit = av.false_negative_audit([dv], gs, stage="v1")
        assert audit["n_false_negatives"] == 1


# --- Artifacts --------------------------------------------------------------


class TestArtifacts:
    def test_failure_log_entry_keys_match_the_plan(self):
        r = av.AtomResult(
            atom=av.Atom(id="a1", text="claim", page=52, evidence_quote="q"),
            status="fail",
            reasons=["quote_unverified:coverage=0.00"],
        )
        entry = r.log_entry("ok, missing citation - pg 35")
        assert set(entry) == {
            "atom_id", "atom_text", "evidence_quote", "page",
            "failure_reason", "subfield_human_grade",
        }
        assert entry["subfield_human_grade"] == "ok, missing citation - pg 35"

    def test_schema_is_written_to_the_stage_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "ARTIFACTS_DIR", tmp_path / "artifacts")
        p = av.save_atomic_schema("v1", draft=True)
        assert p.name == "atomic_schema.v1.json"
        assert p.parent.name == "v1-draft"
        payload = json.loads(p.read_text())
        assert payload["version"] == "v1"
        assert payload["properties"]["claim_type"]["enum"] == list(av.CLAIM_TYPES)
        assert payload["properties"]["polarity"]["enum"] == list(av.POLARITIES)
        assert "scope_qualifier" in payload["required"]
        assert payload["x-mcal"]["scope_qualifier_is_a_deviation"] is True

    def test_failure_log_written(self, tmp_path, monkeypatch, doc_factory):
        monkeypatch.setattr(settings, "ARTIFACTS_DIR", tmp_path / "artifacts")
        dv = _dv_with("d1", "summary.public_response", ["pass", "fail"])
        p = av.save_failure_log([dv], None, stage="v1", draft=True)
        assert p.name == "atomic_verify_failure_log.v1.json"
        payload = json.loads(p.read_text())
        assert payload["n_atoms"] == 2
        assert payload["n_failures"] == 1
        assert payload["failures"][0]["field"] == "summary.public_response"
        assert "false_negative_audit" in payload

    def test_load_failure_log_gives_an_actionable_error(self, tmp_path, monkeypatch):
        """`mcal/artifacts/` may not exist -- build.py is not written yet."""
        monkeypatch.setattr(settings, "ARTIFACTS_DIR", tmp_path / "nothing_here")
        with pytest.raises(FileNotFoundError, match="python -m mcal.build"):
            av.load_failure_log("v1")

    def test_reason_prefix_counts(self):
        dv = av.DocumentVerification(doc_id="d")
        dv.fields["summary.overview"] = av.FieldVerification(
            doc_id="d", field="summary.overview", bucket="summary_narrative",
            results=[
                av.AtomResult(
                    atom=av.Atom(id="a", text="t"), status="fail",
                    reasons=["uncited_no_page", "evidence_quote_missing"],
                ),
                av.AtomResult(
                    atom=av.Atom(id="b", text="t"), status="fail",
                    reasons=["uncited_no_page"],
                ),
            ],
        )
        payload = av.build_failure_log([dv], None, stage="v1")
        assert payload["reason_code_counts"]["uncited_no_page"] == 2


# --- Real-corpus measurements -----------------------------------------------


def _collect_summary_sentences(m2_dir):
    """(doc_id, field, sentence) for every graded doc's summary.* subfields."""
    gs = grades_mod.load_grades()
    out: list[tuple[str, str, str]] = []
    for doc_id in gs.doc_ids:
        p = m2_dir / f"{doc_id}.json"
        if not p.exists():
            continue
        m2 = json.loads(p.read_text())
        for field in settings.SUMMARY_SUBFIELDS:
            sub = field.split(".", 1)[1]
            text = ((m2.get("summary") or {}).get(sub) or {}).get("text") or ""
            for s in av.sentences(text):
                out.append((doc_id, field, s))
    return out


class TestCorpusValidators:
    """
    Deterministic validators over the real graded corpus. No LLM.

    These pin the numbers quoted in atomic_verify.py's own comments. If a rule is
    loosened the counts move and these fail, which is the point: the validators'
    value is entirely in their precision, and precision regressions are otherwise
    invisible.

    ALL NUMBERS IN THIS CLASS WERE RE-MEASURED after the MCAL_PLAN build-#4/#5
    prompt amendment (plain-language + concreteness clause) regenerated the 8
    graded docs. The amendment did not change any rule in atomic_verify.py; it
    changed the PROSE the rules run over, which is ~1.7x longer. The
    pre-amendment corpus is still on disk at
    `segment_a/output/m2_pre_amendment/`, so every count below is stated as
    old -> new and the before-state is asserted in
    `TestCorpusValidatorsPreAmendment` rather than being lost.

    Summary of the movement:

      sentences                 165  -> 286   (+73%)
      coreference-flagged        30  ->  41   (18.2% -> 14.3%)   rate improved
      coordination-splitter hits  1  ->   3   (1/1 correct -> 1/3 correct)  WORSE
      negation cues              29  ->  50   (17.6% -> 17.5%)   flat
      scope cues                 25  ->  45   (15.2% -> 15.7%)   flat
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def summary_sentences():
        out = _collect_summary_sentences(settings.M2_DIR)
        if not out:
            pytest.skip("no M2 output for the graded docs on this machine")
        return out

    def test_sentence_count(self, summary_sentences):
        """
        286 sentences, was 165 pre-amendment (+73%).

        The plain-language clause traded a few long clause-stacked sentences for
        many shorter concrete ones, so this is the denominator every rate below
        is taken over and it is not the same denominator as before. Pinned
        exactly, because if it drifts the rates below are being compared against
        the wrong base.
        """
        assert len(summary_sentences) == 286

    def test_coreference_hit_rate(self, summary_sentences):
        """
        41 of 286 (14.3%). Was 30 of 165 (18.2%).

        The absolute workload went UP (+11 sentences to resolve) but the rate
        went DOWN by 3.9 points, which is the direction the plain-language clause
        predicts: shorter sentences repeat the referent instead of pronominalizing
        it. Still the WORKLOAD the coreference rule imposes, not a
        false-positive rate.
        """
        flagged = [
            (d, f, s) for d, f, s in summary_sentences
            if av.unresolved_references(s)
        ]
        assert len(flagged) == 41
        assert len(flagged) / len(summary_sentences) == pytest.approx(0.143, abs=0.002)

    def test_coreference_cue_breakdown(self, summary_sentences):
        """
        Re-measured. Old -> new, per cue:

            the project          20 -> 19
            it                    8 -> 15
            that                  0 ->  4   (new)
            the alternative       0 ->  2   (new)
            the proposed action   0 ->  2   (new)
            the applicant         2 ->  1
            they                  1 ->  0
            the agency            1 ->  1

        Two things worth noting rather than smoothing over. `it` nearly doubled,
        which is the one place the "shorter sentences" story does not hold -- the
        new prose uses more em-dash and semicolon continuations, and `it` inside
        those still needs resolving. And three cue types appear for the first
        time (`that`, `the alternative`, `the proposed action`); they are new
        because the amended prose discusses named alternatives explicitly, not
        because the cue list changed -- `av.GENERIC_ANAPHORA` is untouched.
        """
        counts: dict[str, int] = {}
        for _d, _f, s in summary_sentences:
            for cue in av.unresolved_references(s):
                counts[cue] = counts.get(cue, 0) + 1
        assert counts == {
            "the project": 19,
            "it": 15,
            "that": 4,
            "the alternative": 2,
            "the proposed action": 2,
            "the applicant": 1,
            "the agency": 1,
        }

    def test_coordination_splitter_precision_regressed_to_1_of_3(
        self, summary_sentences
    ):
        """
        REGRESSION, recorded not hidden. Precision 1/1 -> 1/3.

        Pre-amendment the deterministic splitter fired on exactly 1 of 165
        sentences and that one hit was correct: the wildlife-clause sentence,
        the only genuine outside-text fabrication in the graded corpus
        (MCAL_PLAN 1(4)). "High precision, low recall, by design" was true.

        On the regenerated corpus it fires on 3 of 286 sentences and only ONE of
        the three is a correct split:

          1. p1074_35556039563135 summary.public_response -- CORRECT.
             "HUD, Commerce, HEW, OEO, and the Cook County Highway Department had
             no objections" -> 5 genuine coordinated-subject claims.
          2. p1074_35556036806586 summary.environmental_impact -- FALSE POSITIVE.
             "Highly erodible soils, already gullying from past logging and
             exploratory drilling, could push ..." The comma-delimited span is a
             participial APPOSITIVE modifying the head noun, not a subject list,
             so the splitter emits "Already gullying from past logging could push
             total suspended solids ... to 44,000 mg/L", which is not a claim the
             prose makes.
          3. p1074_35556036811230 summary.alternatives_overview -- FALSE
             POSITIVE, and the worse of the two. "A two-stage screening (first
             economic, then service and social/environmental factors) ranked the
             Minimum LRRT System first, and it was selected as best overall ..."
             splits INSIDE the parenthetical and emits items with unbalanced
             brackets, e.g. "A two-stage screening (first economic was selected
             as best overall ...".

        And the sentence the rule exists for no longer fires at all: see
        `TestCoordinationSplitterNoLongerFiresOnTheWildlifeSentence` and
        tests/test_lincoln_hwy_wildlife_clause.py.

        Blast radius: `coordinated_claims` feeds `coordination_gaps`, which
        `verify_subfield` reports as a WARNING, so these false positives add
        review noise but cannot by themselves fail an atom. The two guard gaps
        they expose are reported as module findings; this test only pins the
        measured state so a later fix shows up here as a change.
        """
        hits = [
            (d, f, s, av.coordinated_claims(s))
            for d, f, s in summary_sentences
            if av.coordinated_claims(s)
        ]
        assert len(hits) == 3
        by_field = {(d, f): claims for d, f, _s, claims in hits}

        # (1) the one correct split.
        good = by_field[(LINCOLN_HWY, "summary.public_response")]
        assert len(good) == 5
        assert good[0].startswith("HUD had no objections")
        assert good[-1].startswith("The Cook County Highway Department")

        # (2) appositive mistaken for a subject list.
        appositive = by_field[
            ("p1074_35556036806586", "summary.environmental_impact")
        ]
        assert len(appositive) == 3
        assert appositive[1].startswith("Already gullying from past logging")

        # (3) split inside a parenthetical -> unbalanced brackets in the items.
        paren = by_field[
            ("p1074_35556036811230", "summary.alternatives_overview")
        ]
        assert len(paren) == 4
        assert any(
            c.count("(") != c.count(")") for c in paren
        ), "the parenthetical-splitting defect is the point of this row"

        # No hit is the wildlife sentence any more.
        assert not any("wildlife habitats" in s for _d, _f, s, _c in hits)

    def test_negation_rate(self, summary_sentences):
        """
        50 of 286 sentences carry a negation cue (17.5%). Was 29 of 165 (17.6%).

        Essentially unchanged as a rate -- the amendment did not make the prose
        more or less negated, only longer. Every one of these is still an atom
        whose polarity the decomposer must get right, which is why the plan makes
        the cue a hard schema requirement rather than advice.
        """
        n = sum(1 for _d, _f, s in summary_sentences if av.negation_cues(s))
        assert n == 50
        assert n / len(summary_sentences) == pytest.approx(0.175, abs=0.002)

    def test_scope_cue_rate(self, summary_sentences):
        """
        45 of 286 sentences (15.7%) carry a range, bound, superlative or
        comparison cue. Was 25 of 165 (15.2%).

        Still ~1 sentence in 6-7, i.e. the surface MCAL_PLAN 1(3)/1(4)'s two
        failures live on did not shrink. Mildly notable that it grew: the
        concreteness clause makes the prose quote ranges ("574,000-642,000 daily
        riders", "Magnitude 7.0-7.5") where the old prose gave a bare point
        value, so more atoms now REQUIRE a scope_qualifier than before. That is
        more work for the qualifier check but it is also the honest shape of the
        source.
        """
        n = sum(1 for _d, _f, s in summary_sentences if av.scope_cues(s))
        assert n == 45
        assert n / len(summary_sentences) == pytest.approx(0.157, abs=0.002)


class TestCorpusValidatorsPreAmendment:
    """
    The same validators over the ARCHIVED pre-amendment corpus.

    Kept so the before-state of every number in `TestCorpusValidators` is
    asserted rather than merely described in a docstring, and so that a claim
    like "splitter precision was 1/1" can be re-checked instead of trusted. If
    `segment_a/output/m2_pre_amendment/` is ever pruned these skip.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def old_summary_sentences():
        out = _collect_summary_sentences(settings.M2_PRE_AMENDMENT_DIR)
        if not out:
            pytest.skip("pre-amendment M2 archive not available on this machine")
        return out

    def test_the_archive_is_the_corpus_the_old_numbers_came_from(
        self, old_summary_sentences
    ):
        """Reproduces every count this file used to assert, exactly."""
        sents = old_summary_sentences
        assert len(sents) == 165
        assert sum(1 for _d, _f, s in sents if av.unresolved_references(s)) == 30
        assert sum(1 for _d, _f, s in sents if av.negation_cues(s)) == 29
        assert sum(1 for _d, _f, s in sents if av.scope_cues(s)) == 25

    def test_the_old_coreference_breakdown(self, old_summary_sentences):
        counts: dict[str, int] = {}
        for _d, _f, s in old_summary_sentences:
            for cue in av.unresolved_references(s):
                counts[cue] = counts.get(cue, 0) + 1
        assert counts == {
            "the project": 20,
            "it": 8,
            "the applicant": 2,
            "they": 1,
            "the agency": 1,
        }

    def test_the_splitter_used_to_be_1_for_1(self, old_summary_sentences):
        """
        The precision claim that no longer holds, asserted against the corpus it
        was true of. Exactly one hit, three claims, the wildlife sentence.
        """
        hits = [
            (d, f, s, av.coordinated_claims(s))
            for d, f, s in old_summary_sentences
            if av.coordinated_claims(s)
        ]
        assert len(hits) == 1
        doc_id, field, _sentence, claims = hits[0]
        assert doc_id == LINCOLN_HWY
        assert field == "summary.environmental_impact"
        assert len(claims) == 3
        assert "wildlife habitats" in claims[-1]


class TestCoordinationSplitterNoLongerFiresOnTheWildlifeSentence:
    """
    The single most consequential corpus change, isolated.

    MCAL_PLAN 1(4)'s prescribed mechanism for the Lincoln Hwy fabrication is
    "atomic decomposition with coordination splitting". The fabricated clause
    "important wildlife habitats" is STILL in the regenerated output, but the
    rerun appended a second independent clause to its sentence, and
    `coordinated_claims` -- correctly, by its own guards -- refuses multi-clause
    sentences. So the deterministic mechanism the plan names no longer fires on
    the one corpus item it was written for.

    This is a coverage regression in the rule, NOT a fabrication that got fixed,
    and not something the tests should paper over. It is reported as a module
    finding; the tests here only pin why it happens so the diagnosis survives.
    """

    OLD_SENTENCE = (
        "No National Register sites, unique wetlands, or important wildlife "
        "habitats are affected."
    )
    NEW_SENTENCE = (
        "No National Register historic sites, wetlands, floodplains, or "
        "important wildlife habitats are affected, and the State Historic "
        "Preservation Officer determined that no archaeological resources "
        "subject to Section 106 protection would be affected."
    )
    # The new sentence's first clause on its own.
    NEW_FIRST_CLAUSE = (
        "No National Register historic sites, wetlands, floodplains, or "
        "important wildlife habitats are affected."
    )

    def test_the_old_sentence_split_into_three(self):
        assert av.coordinated_claims(self.OLD_SENTENCE) == [
            "No National Register sites are affected.",
            "No unique wetlands are affected.",
            "No important wildlife habitats are affected.",
        ]

    def test_the_new_sentence_does_not_split_at_all(self):
        assert av.coordinated_claims(self.NEW_SENTENCE) == []

    def test_the_cause_is_the_appended_second_clause_not_the_list(self):
        """
        Diagnosis. The coordinated subject list is intact and would still split
        into FOUR (the rerun added "floodplains"); what kills it is the
        ", and the State Historic Preservation Officer determined ..." clause,
        which trips both the `_MAX_COORD_PREDICATE_WORDS` (21 > 20) and
        `_CLAUSE_JOIN` guards. Both guards are right in general -- a shared
        predicate that contains a second independent clause is not shared -- so
        the gap is that nothing splits a compound sentence into clauses BEFORE
        the subject-list rule runs.
        """
        claims = av.coordinated_claims(self.NEW_FIRST_CLAUSE)
        assert len(claims) == 4
        assert claims[-1] == "No important wildlife habitats are affected."
        assert "No floodplains are affected." in claims

    def test_av_sentences_does_not_break_the_compound_sentence_up(self):
        """
        And the sentence splitter will not do it for us: it cuts on
        `[.!?:;]`, and the clause boundary here is a comma.
        """
        assert av.sentences(self.NEW_SENTENCE) == [self.NEW_SENTENCE]

    def test_the_shared_negation_would_still_survive_the_split(self):
        """Unchanged property: whatever it splits, polarity is preserved."""
        for claim in av.coordinated_claims(self.NEW_FIRST_CLAUSE):
            assert av.expected_polarity(claim) == "negative"


def _subfield_stats(m2_dir):
    """
    Per-(doc, summary subfield) stats pairing an M2 artifact with its human grade.

    `m2_dir` is a parameter, not a constant, because the two things this must be
    computed over are the CURRENT artifacts and the ARCHIVED pre-amendment ones,
    and the difference between the two answers is itself the finding
    (see TestT01Invisibility's docstring).
    """
    from pages import load_doc

    gs = grades_mod.load_grades()
    stats = []
    for doc_id in gs.doc_ids:
        p = m2_dir / f"{doc_id}.json"
        if not p.exists() or settings.resolve_doc_dir(doc_id) is None:
            continue
        doc = load_doc(settings.resolve_doc_dir(doc_id).name, settings.PAGES_DATA_DIR)
        m2 = json.loads(p.read_text())
        for field in settings.SUMMARY_SUBFIELDS:
            sub = field.split(".", 1)[1]
            value = ((m2.get("summary") or {}).get(sub) or {})
            evidence = value.get("evidence") or []
            item = gs.get(doc_id, field)
            if item is None:
                continue
            checks = qc.check_evidence_list(evidence, doc)
            n_sent = len(av.sentences(value.get("text") or ""))
            stats.append(
                {
                    "doc_id": doc_id,
                    "field": field,
                    "text": value.get("text") or "",
                    "grade": item.raw_grade,
                    "correct": item.correct,
                    "s_quote": qc.s_quote_for(checks),
                    "n_evidence": len(evidence),
                    "n_sentences": n_sent,
                    "ev_per_sentence": (len(evidence) / n_sent) if n_sent else 0.0,
                    "missing_citation": "T01_missing_citation" in item.failure_tags,
                }
            )
    return stats


def _split_populations(stats):
    """(missing-citation subfields, cleanly-graded subfields)."""
    missing = [s for s in stats if s["missing_citation"]]
    clean = [s for s in stats if s["correct"] and not s["missing_citation"]]
    return missing, clean


def _mean(stats, key):
    return sum(s[key] for s in stats) / len(stats)


class TestT01Invisibility:
    """
    The measurement behind `s_citation`, on the real corpus. No LLM.

    T01_missing_citation is the most common observed failure -- 10 of the 30 wrong
    items in the Evaluation sheet. These tests show that field-level `s_quote`
    cannot see it, and that the obvious field-level proxy does not see it either,
    which is the entire case for computing citation coverage at the atom level.

    ------------------------------------------------------------------------
    VALIDITY CAVEAT -- read before trusting any number here.
    ------------------------------------------------------------------------
    The human grades in `May25/Evaluation - Sheet1.csv` were written by reading
    the PRE-AMENDMENT M2 prose. The MCAL_PLAN build-#4/#5 rerun rewrote all 8
    graded docs, so the labels no longer describe the artifacts in
    `segment_a/output/m2/`. A statistic computed by pairing the new artifacts
    with the old labels is therefore of questionable validity: the label says
    "ok, missing citation - pg 41" about a sentence that has since been
    rewritten, and possibly cited.

    Both pairings are computed here and neither is presented as the other:

      (a) NEW artifacts x OLD labels -- STALE. Reported for continuity with the
          rest of the suite, which reads `settings.M2_DIR`. Not evidence about
          the current output's citation behaviour, because nothing re-graded it.
      (b) OLD artifacts x OLD labels -- INTERNALLY CONSISTENT. This is the one
          that supports a conclusion, and it is the pairing the original numbers
          in this file were measured under. `TestT01PreAmendmentIsTheConsistentPairing`
          shows it reproduces them exactly.

    The honest reading of the two together is in
    `test_s_quote_does_not_separate_them`: the +37% citation increase did NOT
    rescue field-level `s_quote`. On (b) it had a 0.028 edge and AUROC 0.598; on
    (a) the edge is -0.007 and AUROC 0.476. Whichever pairing you take, the
    conclusion that atom-level `s_citation` is required is unchanged, and on (a)
    it is if anything stronger. Re-grading the new corpus is the only thing that
    would turn (a) into evidence.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def subfield_stats():
        """(a) NEW artifacts x OLD labels. Stale -- see the class docstring."""
        stats = _subfield_stats(settings.M2_DIR)
        if not stats:
            pytest.skip("graded corpus + M2 output not available on this machine")
        return stats

    @staticmethod
    @pytest.fixture(scope="class")
    def subfield_stats_pre_amendment():
        """(b) OLD artifacts x OLD labels. The internally-consistent pairing."""
        stats = _subfield_stats(settings.M2_PRE_AMENDMENT_DIR)
        if not stats:
            pytest.skip("pre-amendment M2 archive not available on this machine")
        return stats

    def test_the_populations_exist(self, subfield_stats):
        missing, clean = _split_populations(subfield_stats)
        assert len(missing) == 10, "MCAL_PLAN's most common failure, 10 of 30"
        assert len(clean) == 27

    def test_the_two_pairings_cover_the_same_labelled_items(
        self, subfield_stats, subfield_stats_pre_amendment
    ):
        """
        Makes the comparison below legitimate: the 10/27 split is a property of
        the LABELS, so it is identical under both pairings and every difference
        in the statistics comes from the artifacts alone.
        """
        keys = {(s["doc_id"], s["field"]) for s in subfield_stats}
        old_keys = {(s["doc_id"], s["field"]) for s in subfield_stats_pre_amendment}
        assert keys == old_keys
        assert len(keys) == 40
        assert [len(p) for p in _split_populations(subfield_stats)] == [10, 27]
        assert [
            len(p) for p in _split_populations(subfield_stats_pre_amendment)
        ] == [10, 27]

    def test_the_stale_label_problem_is_real_not_hypothetical(
        self, subfield_stats, subfield_stats_pre_amendment
    ):
        """
        Quantifies the caveat. Every one of the 40 labelled subfields was
        rewritten by the rerun, so there is no subset of the corpus for which
        pairing (a) is safe.
        """
        new_text = {(s["doc_id"], s["field"]): s["text"] for s in subfield_stats}
        old_text = {
            (s["doc_id"], s["field"]): s["text"]
            for s in subfield_stats_pre_amendment
        }
        unchanged = [k for k in new_text if new_text[k] == old_text[k]]
        assert unchanged == []
        # And the citation count over exactly these 40 subfields grew 351 -> 501.
        assert sum(s["n_evidence"] for s in subfield_stats_pre_amendment) == 351
        assert sum(s["n_evidence"] for s in subfield_stats) == 501

    def test_s_quote_does_not_separate_them(
        self, subfield_stats, subfield_stats_pre_amendment
    ):
        """
        The core finding, re-measured under both pairings after the +37% citation
        increase (452 -> 619 evidence entries across the 8 docs; 351 -> 501 over
        just these 40 graded subfields).

            pairing                       missing   clean    clean-missing
            (b) old artifacts x old lbls   0.960     0.988      +0.028
            (a) new artifacts x old lbls   0.974     0.967      -0.007

        (b) reproduces the numbers this test used to assert, which is the check
        that the archive really is the corpus they came from.

        Under (a) the gap does not merely shrink, it INVERTS: the
        missing-citation subfields now score marginally HIGHER on `s_quote` than
        the cleanly-graded ones. Do not read that as "missing citations are now
        detectable with the sign flipped" -- 0.007 on n=10 vs n=27 is noise, and
        the labels are stale. Read it as: more citations did not make `s_quote`
        informative about uncited claims, because `s_quote` averages over the
        citations the extractor CHOSE to supply and a missing citation is by
        definition absent from that average. Adding 150 more citations adds 150
        more terms to an average that still cannot contain the defect.

        The assertion is the same one as before -- no usable separation -- and it
        is now satisfied more emphatically, not by loosening the bound.
        """
        old_missing, old_clean = _split_populations(subfield_stats_pre_amendment)
        omq = _mean(old_missing, "s_quote")
        ocq = _mean(old_clean, "s_quote")
        assert omq == pytest.approx(0.960, abs=0.005)
        assert ocq == pytest.approx(0.988, abs=0.005)
        assert abs(ocq - omq) < 0.05

        missing, clean = _split_populations(subfield_stats)
        mq = _mean(missing, "s_quote")
        cq = _mean(clean, "s_quote")
        assert mq == pytest.approx(0.974, abs=0.005)
        assert cq == pytest.approx(0.966, abs=0.005)
        assert abs(cq - mq) < 0.05, (
            "s_quote is not discriminating between missing-citation and clean "
            "subfields; that is the finding, not a bug"
        )
        # The separation got WORSE, not better, when citations went up 37%.
        assert abs(cq - mq) < abs(ocq - omq)

    def test_no_threshold_on_s_quote_separates_them(
        self, subfield_stats, subfield_stats_pre_amendment
    ):
        """
        Stronger than a mean comparison: no cut on `s_quote` classifies the two
        populations at all, because most of both sit at exactly 1.0.

        AUROC moved 0.598 (pre-amendment) -> 0.476 (new artifacts, stale labels),
        i.e. from weakly-better-than-chance to indistinguishable from chance. Both
        are inside the near-chance band this test has always asserted.
        """
        from mcal import confidence

        for stats, expected in (
            (subfield_stats_pre_amendment, 0.598),
            (subfield_stats, 0.476),
        ):
            missing, clean = _split_populations(stats)
            mv = [s["s_quote"] for s in missing]
            cv = [s["s_quote"] for s in clean]
            assert max(mv) == 1.0
            assert min(cv) < max(mv)
            # AUROC via the same estimator confidence.py uses.
            auc = confidence.auroc(mv + cv, [0] * len(mv) + [1] * len(cv))
            assert auc is not None
            assert auc == pytest.approx(expected, abs=0.005)
            assert 0.35 < auc < 0.65, f"s_quote AUROC {auc:.3f} is near chance"

    def test_the_naive_field_level_proxy_still_does_not_work(
        self, subfield_stats, subfield_stats_pre_amendment
    ):
        """
        Re-measured; the CONCLUSION holds but the REASON changed, so the old test
        name ("points the wrong way") is no longer accurate and has been retired.

            pairing                       missing   clean   verdict
            (b) old artifacts x old lbls   2.503     2.098   proxy ranks the
                                                             DEFECTIVE subfields
                                                             higher -- wrong way
            (a) new artifacts x old lbls   1.805     1.841   proxy is flat: a 2%
                                                             difference on n=10
                                                             vs n=27 -- no signal

        Evidence-per-sentence fell in absolute terms under (a) (2.50 -> 1.80 for
        the missing population) even though the absolute citation count rose 37%,
        because the sentence count rose faster: the plain-language clause produced
        +73% sentences against +43% citations over these subfields. So the amended
        prose cites MORE in total and LESS densely per sentence.

        Either way "did it cite enough?" measured at field level is not the cheap
        fix: pre-amendment it was actively anti-correlated, and now it carries no
        information at all. Only enumerating the claims -- decomposition -- puts
        an uncited claim into the arithmetic.
        """
        old_missing, old_clean = _split_populations(subfield_stats_pre_amendment)
        omm = _mean(old_missing, "ev_per_sentence")
        omc = _mean(old_clean, "ev_per_sentence")
        assert omm == pytest.approx(2.50, abs=0.05)
        assert omc == pytest.approx(2.10, abs=0.05)
        assert omm > omc, "pre-amendment the proxy ranked the defective ones higher"

        missing, clean = _split_populations(subfield_stats)
        mm = _mean(missing, "ev_per_sentence")
        mc = _mean(clean, "ev_per_sentence")
        assert mm == pytest.approx(1.805, abs=0.01)
        assert mc == pytest.approx(1.841, abs=0.01)
        # No longer the wrong way -- but no way at all. Under 5% relative gap.
        assert abs(mm - mc) / mc < 0.05

        # The mechanism behind the drop: sentences grew faster than citations.
        assert sum(s["n_sentences"] for s in subfield_stats_pre_amendment) == 165
        assert sum(s["n_sentences"] for s in subfield_stats) == 286

    def test_every_citation_that_exists_verifies(
        self, subfield_stats, subfield_stats_pre_amendment
    ):
        """
        Root cause in one line. The extractor's citations are good; the problem is
        the claims it made without one, and only decomposition enumerates those.

        Still true under both pairings, and slightly MORE true under the new
        artifacts: the worst missing-citation subfield's `s_quote` rose from 0.80
        to 0.90. So the +37% citation increase did not come at the cost of
        citation quality within the graded subfields (97.3% vs 98.0% verified
        overall is a 0.7-point dip driven by the extra volume, not by the
        defective population).
        """
        for stats in (subfield_stats_pre_amendment, subfield_stats):
            missing, _clean = _split_populations(stats)
            assert all(s["s_quote"] >= 0.8 for s in missing)
        old_missing, _ = _split_populations(subfield_stats_pre_amendment)
        new_missing, _ = _split_populations(subfield_stats)
        assert min(s["s_quote"] for s in old_missing) == pytest.approx(0.80, abs=0.01)
        assert min(s["s_quote"] for s in new_missing) == pytest.approx(0.90, abs=0.01)

    def test_atom_level_citation_coverage_can_see_it(self, doc_factory):
        """
        The mechanism, on a minimal case. Two cited claims that verify plus two
        uncited claims: `s_quote` over the supplied evidence is 1.0, `s_citation`
        over the atoms is 0.5.

        Unaffected by the rerun -- it is synthetic on purpose. It is the answer to
        the question the two corpus pairings above leave open: whatever the prose
        looks like, the only unit of account in which an uncited claim is
        countable is the atom.
        """
        doc = doc_factory(
            "The corridor crosses Cook County, Illinois, and requires 16.2 acres "
            "of right-of-way."
        )
        cited = [
            av.Atom(
                id="a1", text="The corridor crosses Cook County, Illinois.",
                page=1, evidence_quote="The corridor crosses Cook County, Illinois",
            ),
            av.Atom(
                id="a2", text="The corridor requires 16.2 acres of right-of-way.",
                page=1, claim_type="numeric",
                evidence_quote="requires 16.2 acres of right-of-way",
            ),
        ]
        uncited = [
            av.Atom(id="a3", text="Eight structures would be demolished."),
            av.Atom(id="a4", text="Noise levels would exceed FHWA standards."),
        ]
        results = [av.verify_atom(a, doc) for a in cited + uncited]
        supplied = [r.check for r in results if r.check is not None]
        assert qc.s_quote_for(supplied) == 1.0
        assert av.s_citation(results) == 0.5
        assert all("T01_missing_citation" in r.tags for r in results[2:])


class TestT01PreAmendmentIsTheConsistentPairing:
    """
    Proves the archive is what the original T01 numbers were measured on.

    Separate class so that the "old artifacts x old labels" reproduction is
    findable on its own, and so that if the archive is pruned the loss is a
    single obvious skip rather than a silent weakening of TestT01Invisibility.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def old_stats():
        stats = _subfield_stats(settings.M2_PRE_AMENDMENT_DIR)
        if not stats:
            pytest.skip("pre-amendment M2 archive not available on this machine")
        return stats

    def test_it_reproduces_every_number_this_file_used_to_assert(self, old_stats):
        missing, clean = _split_populations(old_stats)
        assert (len(missing), len(clean)) == (10, 27)
        assert _mean(missing, "s_quote") == pytest.approx(0.960, abs=0.005)
        assert _mean(clean, "s_quote") == pytest.approx(0.988, abs=0.005)
        assert _mean(missing, "ev_per_sentence") == pytest.approx(2.50, abs=0.05)
        assert _mean(clean, "ev_per_sentence") == pytest.approx(2.10, abs=0.05)

    def test_the_archive_predates_summary_of_interest(self):
        """
        Sanity check that the archive is genuinely the pre-amendment build and
        not a copy of the current one: build item #5 added `summary_of_interest`
        and a `_prompt_version` marker, and neither is in the archive.
        """
        gs = grades_mod.load_grades()
        checked = 0
        for doc_id in gs.doc_ids:
            old = settings.M2_PRE_AMENDMENT_DIR / f"{doc_id}.json"
            new = settings.M2_DIR / f"{doc_id}.json"
            if not (old.exists() and new.exists()):
                continue
            o = json.loads(old.read_text())
            n = json.loads(new.read_text())
            assert "summary_of_interest" not in o
            assert "_prompt_version" not in o
            assert isinstance(n.get("summary_of_interest"), list)
            assert n.get("_prompt_version") == settings.M2_PROMPT_VERSION_REQUIRED
            checked += 1
        if not checked:
            pytest.skip("pre-amendment M2 archive not available on this machine")
        assert checked == 8


class TestAgainstRealDoc:
    def test_verified_m2_quotes_survive_atomization(self, doc_loader, m2_loader):
        """
        No regression against the existing verifier: an atom whose evidence_quote
        is one segment_a already marked verified must not fail on the quote check.
        """
        doc = doc_loader(LINCOLN_HWY)
        m2 = m2_loader(LINCOLN_HWY)
        checked = 0
        for field in settings.SUMMARY_SUBFIELDS:
            sub = field.split(".", 1)[1]
            ev = ((m2.get("summary") or {}).get(sub) or {}).get("evidence") or []
            for i, e in enumerate(ev):
                if not (e.get("quote_verified") and e.get("source_pages")):
                    continue
                pages = qc.coerce_pages(e["source_pages"])
                atom = av.Atom(
                    id=f"{sub}.{i}",
                    text=e["quote"],
                    page=pages[0],
                    evidence_quote=e["quote"],
                    claim_type="prose",
                    polarity=av.expected_polarity(e["quote"]),
                )
                r = av.verify_atom(atom, doc)
                assert r.check is not None
                assert r.check.verified == "yes", (
                    f"previously-verified quote now {r.check.verified}: "
                    f"{e['quote'][:70]!r}"
                )
                checked += 1
        assert checked > 20

    def test_competing_values_is_advisory_only(self, doc_loader):
        """
        It surfaces the p.145/p.146 magnitude contradiction for a human but never
        changes an atom's status -- on OCR'd cost tables its precision collapses.
        """
        doc = doc_loader(LA_TRANSIT)
        quote = (
            "The most severe ground shaking that would be felt in the starter "
            "line area would be generated by a Magnitude 7.5 earthquake "
            "occurring on the Newport-Inglewood Fault"
        )
        atom = av.Atom(
            id="a",
            text="The most severe ground shaking felt in the starter line area "
                 "would be generated by a Magnitude 7.5 earthquake on the "
                 "Newport-Inglewood Fault.",
            page=146,
            evidence_quote=quote,
            claim_type="comparative",
            scope_qualifier="that would be felt in the starter line area",
        )
        r = av.verify_atom(atom, doc)
        assert r.status == av.STATUS_PASS
        assert r.competing_values, "the contradiction should still be surfaced"
