"""
Regression: Lincoln Hwy wildlife-clause fabrication.

MCAL_PLAN 1(4) names this file explicitly. It is the one failure in the graded
corpus that the plan's stated mechanism -- atomic decomposition with coordination
splitting, then substring verification -- genuinely catches, and the only genuine
outside-text fabrication in the corpus.

The failure: `summary.environmental_impact` for LINCOLN_HWY
(p1074_35556039563135) ends with a sentence asserting that no important wildlife
habitats are affected. The document supports the National Register item (p.52:
"There are no known National Register of Historic Places or Landmarks involved
with, nor in the area of this improvement"). The word "habitat" does not occur
anywhere in the 347-page document. "important wildlife habitats are affected" is
Opus completing a plausible NEPA sentence -- prior injection.

------------------------------------------------------------------------------
POST-RERUN STATUS (MCAL_PLAN build items #4/#5, plain-language + concreteness).
------------------------------------------------------------------------------
The fabrication SURVIVED the rerun; the mechanism that caught it did not.

  pre-amendment sentence (archived in segment_a/output/m2_pre_amendment/):
      "No National Register sites, unique wetlands, or important wildlife
       habitats are affected."
  post-amendment sentence (segment_a/output/m2/):
      "No National Register historic sites, wetlands, floodplains, or important
       wildlife habitats are affected, and the State Historic Preservation
       Officer determined that no archaeological resources subject to Section 106
       protection would be affected."

The coordinated subject list is still there and even grew (3 items -> 4, adding
"floodplains"), but the rerun welded a second independent clause onto the
sentence. `av.coordinated_claims` refuses multi-clause sentences -- correctly, a
"shared" predicate containing its own clause is not shared -- so on the actual
output on disk it now returns []. Coordination splitting, the mechanism the plan
names for this exact item, no longer fires on this exact item.

`TestCoordinationSplit` records that as a coverage regression rather than
re-pointing the assertion at a hand-typed sentence and pretending nothing moved.
Everything downstream of the split (the fabricated atom fails, is tagged T03,
coverage is exactly 0) is unchanged and still asserted, because it only needs the
atom, not the splitter that would have produced it.

Why the coordination-splitting rule is load-bearing rather than incidental: the
extractor supplied 17 evidence quotes for this subfield (12 pre-amendment) and
every one of them verifies, so field-level `s_quote` is 1.0 -- a perfect score for
a subfield the human graded `hallucination`. `quote_check.s_quote_for` averages
over the citations the extractor CHOSE to make, and the wildlife clause has none,
so it never enters the arithmetic. Decomposition changes the unit of account from
"citations supplied" to "claims made", which is the only way the clause can be
scored at all; splitting then localizes the defect to one of the items, which is
what the reviewer and the failure log need. Measured in
`TestFieldLevelSQuoteCannotSeeIt`.

No LLM: the split is produced by `atomic_verify.coordinated_claims`, and the
decomposer is either bypassed (`atoms=`) or injected (`call=`).
"""

from __future__ import annotations

import pytest

from mcal import atomic_verify as av
from mcal import quote_check as qc
from mcal import settings

from conftest import LINCOLN_HWY

FIELD = "summary.environmental_impact"
SUBFIELD = "environmental_impact"

# The pre-amendment sentence. Retained as a constant because several assertions
# below are ABOUT the pre-amendment behaviour and must keep exercising the string
# they were measured on.
SENTENCE = (
    "No National Register sites, unique wetlands, or important wildlife "
    "habitats are affected."
)
# The post-amendment sentence, verbatim from segment_a/output/m2/.
SENTENCE_POST_AMENDMENT = (
    "No National Register historic sites, wetlands, floodplains, or important "
    "wildlife habitats are affected, and the State Historic Preservation "
    "Officer determined that no archaeological resources subject to Section 106 "
    "protection would be affected."
)
FABRICATED = "No important wildlife habitats are affected."
SUPPORTED = "No National Register sites are affected."

# The p.52 sentence segment_a itself cited for the National Register clause.
P52_QUOTE = (
    "There are no known National Register of Historic Places or Landmarks "
    "involved with, nor in the area of this improvement."
)
P52 = 52


@pytest.fixture(scope="module")
def lincoln(doc_loader):
    return doc_loader(LINCOLN_HWY)


@pytest.fixture(scope="module")
def env_impact(m2_loader):
    m2 = m2_loader(LINCOLN_HWY)
    value = (m2.get("summary") or {}).get(SUBFIELD)
    if not value:
        pytest.skip("no summary.environmental_impact in M2 output")
    return value


@pytest.fixture(scope="module")
def env_impact_pre_amendment(m2_pre_amendment_loader):
    m2 = m2_pre_amendment_loader(LINCOLN_HWY)
    value = (m2.get("summary") or {}).get(SUBFIELD)
    if not value:
        pytest.skip("no summary.environmental_impact in the pre-amendment archive")
    return value


def _wildlife_sentence(value) -> str:
    matches = [
        s for s in av.sentences(value.get("text") or "")
        if "wildlife habitats" in s
    ]
    assert matches, "fixture requires the coordinated sentence"
    return matches[0]


class TestTheSentenceIsStillThere:
    """If M2 is re-run and the fabrication disappears, this file must say so
    rather than silently keep passing on a sentence that no longer exists."""

    def test_present_in_m2_output(self, env_impact):
        text = " ".join((env_impact.get("text") or "").split())
        assert "important wildlife habitats are affected" in text, (
            "the MCAL_PLAN 1(4) fabrication is no longer in the M2 output; "
            "re-point or retire this regression rather than deleting the "
            "assertion"
        )

    def test_the_rerun_rewrote_the_sentence_around_it(
        self, env_impact, env_impact_pre_amendment
    ):
        """
        The precise scope of what the rerun changed here: the fabricated clause is
        byte-identical, the sentence containing it is not. Asserted so that the
        splitter regression below cannot be misread as "the fabrication changed".
        """
        old = _wildlife_sentence(env_impact_pre_amendment)
        new = _wildlife_sentence(env_impact)
        assert " ".join(old.split()) == SENTENCE
        assert " ".join(new.split()) == SENTENCE_POST_AMENDMENT
        assert old != new
        assert "important wildlife habitats are affected" in old
        assert "important wildlife habitats are affected" in new

    def test_habitat_appears_nowhere_in_the_document(self, lincoln):
        """The clause is not merely miscited -- it is absent from the corpus."""
        assert "habitat" not in (lincoln.full_text or "").lower()


class TestCoordinationSplit:
    """
    REGRESSION, recorded not repaired.

    Pre-amendment the deterministic splitter turned the wildlife sentence into 3
    checkable claims. Post-amendment it turns it into none. The rule did not
    change; the prose did.
    """

    def test_the_sentence_no_longer_splits_at_all(self, env_impact):
        """
        Was: 3 atoms, the last of them the fabrication. Now: 0.

        This is the assertion that used to be
        `test_the_sentence_splits_into_three_atoms`. It is not relaxed into
        `len(claims) >= 0`; it pins the new value exactly, so a later fix to the
        splitter (or another rewrite of the prose) fails here and gets looked at.
        """
        sentence = _wildlife_sentence(env_impact)
        assert av.coordinated_claims(sentence) == []

    def test_it_did_split_into_three_before_the_rerun(
        self, env_impact_pre_amendment
    ):
        """The property this file was built on, asserted against the archive so
        the history is preserved rather than described."""
        sentence = _wildlife_sentence(env_impact_pre_amendment)
        claims = av.coordinated_claims(sentence)
        assert len(claims) == 3
        assert claims[-1] == FABRICATED

    def test_the_cause_is_the_appended_clause_not_the_list(self, env_impact):
        """
        Diagnosis, so the finding is actionable. Take the same sentence and cut it
        at the ", and the State Historic Preservation Officer ..." boundary and the
        subject-list rule fires again, now yielding FOUR claims because the rerun
        added "floodplains". The list is fine; what defeats the rule is that
        nothing decomposes a compound sentence into clauses before it runs, and
        `av.sentences` will not do it because it cuts on `[.!?:;]` and this
        boundary is a comma.
        """
        sentence = _wildlife_sentence(env_impact)
        assert av.sentences(sentence) == [sentence]

        first_clause = sentence.split(", and the State Historic")[0].strip() + "."
        claims = av.coordinated_claims(first_clause)
        assert len(claims) == 4
        assert claims[-1] == FABRICATED
        assert "No floodplains are affected." in claims

    def test_the_shared_negation_survives_the_split(self):
        """
        Unchanged property, asserted on both sentences: whatever the splitter does
        emit, it never drops the shared "No".
        """
        for source in (
            SENTENCE,
            SENTENCE_POST_AMENDMENT.split(", and the State Historic")[0] + ".",
        ):
            claims = av.coordinated_claims(source)
            assert claims
            for claim in claims:
                assert av.expected_polarity(claim) == "negative", (
                    "MCAL_PLAN 3.4: never emit an affirmative atom for a negated "
                    "claim. Dropping the shared 'No' would invert all of them."
                )



class TestFabricatedAtomFailsVerification:
    """The assertion MCAL_PLAN 1(4) actually asks this file for."""

    def _atom(self, text, quote=None, page=P52):
        return av.Atom(
            id="wildlife",
            text=text,
            page=page,
            evidence_quote=quote if quote is not None else text,
            claim_type="prose",
            polarity=av.expected_polarity(text),
            field=FIELD,
        )

    def test_fabricated_atom_fails(self, lincoln):
        r = av.verify_atom(self._atom(FABRICATED), lincoln)
        assert r.status == av.STATUS_FAIL
        assert r.score == 0.0
        assert any(p.startswith("quote_unverified") for p in r.reasons)

    def test_fabricated_atom_is_tagged_as_fabrication(self, lincoln):
        r = av.verify_atom(self._atom(FABRICATED), lincoln)
        assert "T03_outside_text_fabrication" in r.tags

    def test_coverage_is_exactly_zero(self, lincoln):
        """
        Not merely below threshold -- NONE of the clause's content tokens are on
        the cited page. That is what distinguishes prior injection from a
        miscitation.
        """
        r = av.verify_atom(self._atom(FABRICATED), lincoln)
        assert r.check is not None
        assert r.check.coverage == 0.0

    def test_the_supported_sibling_passes(self, lincoln):
        """
        The split must not turn a true claim into a false alarm. If this fails,
        the splitter is producing garbage and the failing wildlife atom above
        proves nothing.
        """
        r = av.verify_atom(self._atom(SUPPORTED, quote=P52_QUOTE), lincoln)
        assert r.status == av.STATUS_PASS, r.reasons

    def test_middle_sibling_also_fails(self, lincoln):
        """
        "unique wetlands" is cited to the same page and is not there either --
        3 wetland mentions exist in the document but none on p.52 +/-2. Recorded
        so the split's outcome is fully pinned, not just its interesting row.
        """
        r = av.verify_atom(self._atom("No unique wetlands are affected."), lincoln)
        assert r.status == av.STATUS_FAIL


class TestFieldLevelSQuoteCannotSeeIt:
    """
    Why atomic decomposition is needed at all, measured.

    The escape route is not that the fabrication scores well -- it does not. It is
    that field-level `s_quote` never looks at it. `quote_check.s_quote_for`
    averages over the evidence entries the extractor SUPPLIED, and the extractor
    supplied no evidence entry for the wildlife clause. All 17 of this subfield's
    evidence quotes verify (12 pre-amendment; the rerun added 5 and all of them
    verify too), so `s_quote = 1.0` for a subfield the human graded
    `hallucination`. Adding citations did not help, which is the point: they are
    added where the model was already willing to cite.

    Decomposition changes the unit of account from "citations the extractor chose
    to make" to "claims the prose actually makes", which is the only way an
    unsupported clause can enter the arithmetic at all. It is also exactly the
    same mechanism that makes T01_missing_citation invisible.
    """

    def test_every_supplied_quote_verifies(self, lincoln, env_impact):
        checks = qc.check_evidence_list(env_impact.get("evidence") or [], lincoln)
        assert len(checks) == 17, "was 12 pre-amendment; the rerun cites more"
        assert {c.verified for c in checks} == {"yes"}

    def test_the_extra_citations_all_verify_too(
        self, lincoln, env_impact, env_impact_pre_amendment
    ):
        """
        The +42% citation growth on this subfield did not dilute quality, so the
        1.0 below is not an artifact of a laxer verifier -- it is 17 genuine hits.
        """
        old = qc.check_evidence_list(
            env_impact_pre_amendment.get("evidence") or [], lincoln
        )
        new = qc.check_evidence_list(env_impact.get("evidence") or [], lincoln)
        assert len(old) == 12
        assert len(new) == 17
        assert {c.verified for c in old} == {c.verified for c in new} == {"yes"}

    def test_field_level_s_quote_is_perfect_despite_the_fabrication(
        self, lincoln, env_impact
    ):
        checks = qc.check_evidence_list(env_impact.get("evidence") or [], lincoln)
        assert qc.s_quote_for(checks) == 1.0, (
            "if this is no longer 1.0 the extractor started citing the fabricated "
            "clause, which would be a different (and better) situation"
        )

    def test_no_supplied_quote_mentions_the_fabricated_clause(self, env_impact):
        joined = " ".join(
            (e.get("quote") or "") for e in (env_impact.get("evidence") or [])
        ).lower()
        assert "wildlife" not in joined
        assert "habitat" not in joined

    def test_the_clause_is_rejected_once_it_becomes_an_atom(self, lincoln):
        """
        The contrast. Given a citation of its own -- which is what decomposition
        forces -- the clause is rejected outright, not merely downgraded.
        """
        r = qc.check_quote(FABRICATED, [P52], lincoln)
        assert r.verified == "no"
        assert r.coverage == 0.0
        assert r.s_quote == 0.0

    def test_the_unsplit_sentence_is_also_rejected(self, lincoln):
        """
        Recorded because it bounds the claim in this file's docstring. Scored as
        its own quote, the whole sentence reaches only coverage 0.22 -- its two
        true items are not enough to carry it. So splitting is not the ONLY thing
        that could reject this clause; what splitting adds is attribution, i.e.
        knowing WHICH of the three items is unsupported, which is what the
        reviewer and the failure log need.
        """
        whole = qc.check_quote(SENTENCE, [P52], lincoln)
        assert whole.verified == "no"
        assert whole.coverage is not None and whole.coverage < 0.4
        # And the split localizes the defect to one of the three.
        results = [
            av.verify_atom(
                av.Atom(
                    id=f"c{i}", text=c, page=P52,
                    evidence_quote=P52_QUOTE if "National Register" in c else c,
                    polarity="negative",
                ),
                lincoln,
            )
            for i, c in enumerate(av.coordinated_claims(SENTENCE))
        ]
        assert [r.status for r in results] == [
            av.STATUS_PASS, av.STATUS_FAIL, av.STATUS_FAIL,
        ]

    def test_gap_detector_reports_the_under_split(self):
        atoms = [av.Atom(id="whole", text=SENTENCE, page=P52,
                         evidence_quote=SENTENCE)]
        gaps = av.coordination_gaps(SENTENCE, atoms)
        assert len(gaps) == 3
        assert FABRICATED in gaps


class TestEndToEndSubfield:
    """
    The full path with an injected decomposer, so the aggregation and the failure
    log are exercised too.
    """

    def _call(self, atoms_payload):
        def call(system, user, **kw):
            return {"atoms": atoms_payload}

        return call

    def test_subfield_score_is_reduced_and_the_atom_is_logged(
        self, lincoln, env_impact
    ):
        payload = [
            {
                "text": SUPPORTED, "subject": "National Register sites",
                "predicate": "are affected", "object": "", "page": P52,
                "evidence_quote": P52_QUOTE, "claim_type": "prose",
                "polarity": "negative", "coreference_resolved": True,
                "scope_qualifier": None,
            },
            {
                "text": FABRICATED, "subject": "important wildlife habitats",
                "predicate": "are affected", "object": "", "page": P52,
                "evidence_quote": FABRICATED, "claim_type": "prose",
                "polarity": "negative", "coreference_resolved": True,
                "scope_qualifier": None,
            },
        ]
        fv = av.verify_subfield(
            LINCOLN_HWY, FIELD, env_impact, lincoln, call=self._call(payload)
        )
        assert fv.n_atoms == 2
        assert fv.n_failed == 1
        assert fv.score == 0.5
        assert "T03_outside_text_fabrication" in fv.tags

        failures = fv.failures()
        assert len(failures) == 1
        entry = failures[0].log_entry("hallucination: \"or important wildlife "
                                     "habitats are affected.\"")
        assert entry["atom_text"] == FABRICATED
        assert entry["page"] == P52
        assert "coverage=0.00" in entry["failure_reason"]

    def test_the_human_graded_this_subfield_wrong(self):
        """
        Ties the machine result to the human label. The Evaluation sheet records
        `hallucination: "or important wildlife habitats are affected."` for this
        (doc, field), so a failing atom here is a TRUE positive and must not be
        counted against the false-negative ceiling.
        """
        from mcal import grades as grades_mod

        gs = grades_mod.load_grades()
        item = gs.get(LINCOLN_HWY, FIELD)
        if item is None:
            pytest.skip("Evaluation sheet not available")
        assert not item.correct
        assert "T03_outside_text_fabrication" in item.failure_tags

    def test_not_counted_as_a_false_negative(self, lincoln, env_impact):
        """
        MCAL_PLAN 3.4's audit measures failures on CORRECTLY-graded subfields.
        This subfield is graded wrong, so catching it must contribute nothing to
        the false-negative rate.
        """
        from mcal import grades as grades_mod

        payload = [
            {
                "text": FABRICATED, "subject": "important wildlife habitats",
                "predicate": "are affected", "object": "", "page": P52,
                "evidence_quote": FABRICATED, "claim_type": "prose",
                "polarity": "negative", "coreference_resolved": True,
                "scope_qualifier": None,
            }
        ]
        dv = av.DocumentVerification(doc_id=settings.normalize_doc_id(LINCOLN_HWY))
        dv.fields[FIELD] = av.verify_subfield(
            LINCOLN_HWY, FIELD, env_impact, lincoln, call=self._call(payload)
        )
        gs = grades_mod.load_grades()
        if gs.get(LINCOLN_HWY, FIELD) is None:
            pytest.skip("Evaluation sheet not available")
        audit = av.false_negative_audit([dv], gs, stage="v2")
        assert audit["n_atoms_on_correct_subfields"] == 0
        assert audit["flagged"] is False
