"""
Carried evidence: `summary.overview` and `themes` must not be credited for
citations they borrowed from the summary subfields.

The defect these tests pin (PIPELINE_REVIEW 2.3): `_summary_reduce` REPLACED the
overview's own evidence with up to three verified quotes taken from the five
subfields whenever none of the overview's own quotes verified, and
`extract_themes` populated the themes row entirely from the same source. Both
lists were untagged, so:

  * the grading sheet rendered `quote_verified: yes` with real page numbers for a
    field whose own claims had just failed verification -- and all 8 graded
    documents grade `summary.overview` "ok", a grade that may therefore be
    measuring the subfields;
  * `s_quote` averaged the borrowed quotes and scored 1.0, which is the signal
    the conformal thresholds are fitted on.

The fix is provenance, not removal: the quotes stay on the row so a reviewer has
pages to look at, tagged `evidence_source="carried_from_subfields"`, and every
scoring path counts a field's OWN evidence only.

`test_untagged_evidence_is_treated_as_own` is the compatibility guard: artifacts
written before the tag existed must score exactly as they did, or this change
would retroactively move the numbers the current thresholds were fitted on.
"""

from __future__ import annotations

import pytest

import evidence as ev_mod
import m2 as m2_mod
from mcal import confidence, grading_sheet as gsh, settings
from segment_b import critic as C
from segment_b import gate as G

from conftest import (
    FABRICATED_QUOTE,
    SYNTHETIC_PAGES,
    VERIFIABLE_QUOTE,
    build_m2,
)

FOUND = VERIFIABLE_QUOTE
FABRICATED = FABRICATED_QUOTE


@pytest.fixture
def doc(doc_factory):
    return doc_factory(*SYNTHETIC_PAGES, doc_id="synthetic")


def cr(field: str, verdict: str = "PASS") -> C.CriticResult:
    return C.CriticResult(
        field=field,
        bucket=settings.bucket_for_field(field),
        verdict=verdict,
        evidence_quote=VERIFIABLE_QUOTE,
        source_pages=[4],
        judge_model=C.judge_model_for(field)[1],
    )


def _ev(quote=FOUND, page="4", verified=True):
    return {"quote": quote, "source_pages": [page], "quote_verified": verified}


def _carried(quote=FOUND, page="4", source="summary.public_response"):
    return ev_mod.mark_carried(_ev(quote, page), source)


# --- The evidence helpers ----------------------------------------------------


class TestProvenanceHelpers:
    def test_untagged_is_own(self):
        assert ev_mod.own_evidence([_ev()]) == [_ev()]
        assert ev_mod.carried_evidence([_ev()]) == []

    def test_tagged_is_carried(self):
        c = _carried()
        assert ev_mod.is_carried(c)
        assert ev_mod.own_evidence([c]) == []
        assert ev_mod.carried_evidence([c]) == [c]

    def test_mark_carried_does_not_mutate_the_source(self):
        """The subfield's own evidence list must not acquire the tag."""
        original = _ev()
        tagged = ev_mod.mark_carried(original, "summary.environmental_impact")
        assert "evidence_source" not in original
        assert tagged["carried_from"] == "summary.environmental_impact"


# --- m2: what gets written ---------------------------------------------------


class TestSummaryReduceCarry:
    @staticmethod
    def _reduce(monkeypatch, doc, overview_quotes):
        """Run `_summary_reduce` with a stubbed Opus call."""
        payload = {
            "overview": {"text": "Overview.", "quotes": overview_quotes},
            "project_description": {"text": "PD.", "quotes": [FOUND]},
            "affected_community": {"text": "AC.", "quotes": [FOUND]},
            "alternatives_overview": {"text": "AO.", "quotes": [FOUND]},
            "environmental_impact": {"text": "EI.", "quotes": [FOUND]},
            "public_response": {"text": "PR.", "quotes": [FOUND]},
        }
        monkeypatch.setattr(m2_mod, "opus", lambda **kw: payload)
        return m2_mod._summary_reduce([], doc)

    def test_own_evidence_is_kept_when_the_carry_fires(self, monkeypatch, doc):
        """
        The old code assigned over the list, discarding the unverified entries --
        exactly when they were the most informative thing on the row.
        """
        final = self._reduce(monkeypatch, doc, [FABRICATED])
        overview = final["overview"]["evidence"]
        own = ev_mod.own_evidence(overview)
        assert len(own) == 1
        assert own[0]["quote"] == FABRICATED
        assert own[0]["quote_verified"] is False
        assert "not found verbatim" in own[0]["note"]

    def test_carried_entries_are_tagged_with_their_source(self, monkeypatch, doc):
        final = self._reduce(monkeypatch, doc, [FABRICATED])
        carried = ev_mod.carried_evidence(final["overview"]["evidence"])
        assert carried
        assert all(c["quote_verified"] for c in carried)
        assert all(
            c["carried_from"] in settings.SUMMARY_SUBFIELDS for c in carried
        )

    def test_no_carry_when_the_overview_cites_itself(self, monkeypatch, doc):
        final = self._reduce(monkeypatch, doc, [FOUND])
        assert ev_mod.carried_evidence(final["overview"]["evidence"]) == []

    def test_the_carry_is_still_capped_at_three(self, monkeypatch, doc):
        final = self._reduce(monkeypatch, doc, [FABRICATED])
        assert len(ev_mod.carried_evidence(final["overview"]["evidence"])) <= 3

    def test_subfields_keep_their_own_evidence_untagged(self, monkeypatch, doc):
        """Only the roll-up borrows; the lenders are unaffected."""
        final = self._reduce(monkeypatch, doc, [FABRICATED])
        for field in settings.SUMMARY_SUBFIELDS:
            sub = field.split(".", 1)[1]
            assert ev_mod.carried_evidence(final[sub]["evidence"]) == []


class TestThemesCarry:
    def test_all_themes_evidence_is_tagged(self, monkeypatch, doc):
        """Themes are never asked for quotes, so 100% of the row is borrowed."""
        monkeypatch.setattr(
            m2_mod,
            "sonnet",
            lambda **kw: {
                "themes": ["Transportation Infrastructure"],
                "subthemes": [],
                "justification": "From alternatives_overview.",
                "self_confidence": "high",
            },
        )
        # m2 uses bare subfield names internally; settings uses dotted ones.
        summary = {
            f: {"text": "t", "evidence": [_ev()]}
            for f in m2_mod.SUMMARY_SUBFIELDS
        }
        out = m2_mod.extract_themes(summary)
        assert out["evidence"]
        assert ev_mod.own_evidence(out["evidence"]) == []
        assert all(ev_mod.is_carried(e) for e in out["evidence"])


# --- Scoring -----------------------------------------------------------------


class TestSQuoteUsesOwnEvidenceOnly:
    def test_overview_scores_zero_when_only_borrowed_quotes_verify(self, doc):
        """
        The laundering, executed. Own quote fabricated, three borrowed quotes
        verified: the field must score 0.0, not 1.0.
        """
        entry = {
            "text": "Overview.",
            "evidence": [_ev(FABRICATED, "4", verified=False), _carried(), _carried()],
        }
        sig, verdict = G.signals_for_field(
            "summary.overview", entry, cr("summary.overview"), doc
        )
        assert verdict == "no"
        assert sig.s_quote == 0.0

    def test_the_old_behaviour_would_have_scored_one(self, doc):
        """
        Control: the same row with the tags stripped -- i.e. what the pipeline
        used to write -- still scores 1.0. Keeps the magnitude of the defect
        visible rather than asserted.
        """
        untagged = [
            {k: v for k, v in _carried().items()
             if k not in ("evidence_source", "carried_from")}
            for _ in range(2)
        ]
        entry = {"text": "Overview.", "evidence": untagged}
        sig, verdict = G.signals_for_field(
            "summary.overview", entry, cr("summary.overview"), doc
        )
        assert verdict == "yes"
        assert sig.s_quote == 1.0

    def test_overview_still_scores_its_own_verified_quotes(self, doc):
        entry = {"text": "Overview.", "evidence": [_ev(), _carried()]}
        sig, verdict = G.signals_for_field(
            "summary.overview", entry, cr("summary.overview"), doc
        )
        assert verdict == "yes"
        assert sig.s_quote == 1.0

    def test_themes_s_quote_is_excluded_not_zeroed(self, doc):
        """
        Themes has no first-party quotes by design, so `s_quote` is undefined:
        None, which `composite` renormalizes away. Not 1.0 (a credit it did not
        earn) and not 0.0 (a fault for a citation it was never asked to make).
        """
        entry = {"value": {"themes": ["X"], "subthemes": []},
                 "evidence": [_carried(), _carried()]}
        sig, verdict = G.signals_for_field("themes", entry, cr("themes"), doc)
        assert verdict is None
        assert sig.s_quote is None
        # Composite collapses to s_critic alone.
        assert confidence.composite(sig) == sig.s_critic

    def test_themes_with_a_note_verdict_still_loses_the_free_floor(self, doc):
        """
        The practical consequence: under the old laundering a PASS_WITH_NOTE
        themes row scored 0.5*1.0 + 0.5*0.7 = 0.85. Excluding s_quote gives 0.70.
        """
        entry = {"value": {"themes": ["X"], "subthemes": []},
                 "evidence": [_carried()]}
        sig, _ = G.signals_for_field(
            "themes", entry, cr("themes", "PASS_WITH_NOTE"), doc
        )
        assert confidence.composite(sig) == pytest.approx(0.70)

    def test_a_field_with_no_evidence_at_all_still_scores_zero(self, doc):
        """MCAL_PLAN 1(5)-(7): absence of evidence is a failure, not n/a."""
        sig, verdict = G.signals_for_field(
            "summary.overview", {"text": "x", "evidence": []},
            cr("summary.overview"), doc,
        )
        assert verdict == "no"
        assert sig.s_quote == 0.0

    def test_untagged_evidence_is_treated_as_own(self, doc):
        """
        Compatibility guard. Every m2.json on disk predates the tag, so all of
        their evidence must count as first-party and reproduce the existing score.
        """
        m2 = build_m2()
        entry = m2["summary"]["overview"]
        assert not any(ev_mod.is_carried(e) for e in entry["evidence"])
        sig, verdict = G.signals_for_field(
            "summary.overview", entry, cr("summary.overview"), doc
        )
        assert verdict == "yes"
        assert sig.s_quote == 1.0

    def test_citation_rate_ignores_borrowed_cites(self, doc):
        """A borrowed quote's page cite says the SUBFIELD was cited."""
        entry = {"text": "x", "evidence": [
            {"quote": FABRICATED, "source_pages": []}, _carried(), _carried(),
        ]}
        assert G.citation_rate("summary.overview", entry) == 0.0
        assert G.citation_rate("themes", {"evidence": [_carried()]}) is None


class TestCalibrationAgreesWithTheGate:
    def test_build_uses_the_same_own_evidence_helper(self):
        """
        `build.py` computes its own quote verdict for calibration. If it counted
        carried evidence while the gate did not, tau would be fitted on one
        definition of s_quote and applied against another.
        """
        import inspect

        from mcal import build

        src = inspect.getsource(build.score_graded_corpus)
        assert "own_evidence(evidence)" in src


# --- The reviewer's view -----------------------------------------------------


class TestGradingSheetShowsProvenance:
    def test_borrowed_quotes_are_labelled(self, doc):
        quote, pages, verified = gsh.render_evidence(
            [_ev(FABRICATED, "4"), _carried()], doc=doc
        )
        assert "[CARRIED FROM summary.public_response]" in quote
        assert quote.count("[CARRIED FROM") == 1

    def test_verified_column_reports_own_quotes_only(self, doc):
        """
        The column the reviewer trusts. Own quote fabricated + borrowed quote
        verified used to read "mixed"; it must read "no".
        """
        _q, _p, verified = gsh.render_evidence(
            [_ev(FABRICATED, "4"), _carried()], doc=doc
        )
        assert verified == "no"

    def test_carried_only_row_is_marked_not_applicable(self, doc):
        _q, _p, verified = gsh.render_evidence([_carried(), _carried()], doc=doc)
        assert verified == "n/a (carried)"

    def test_pages_are_still_offered_to_the_reviewer(self, doc):
        """The carry exists so the row points at real pages. That must survive."""
        quote, pages, _v = gsh.render_evidence([_carried(page="4")], doc=doc)
        assert "4" in pages
        assert FOUND in quote
