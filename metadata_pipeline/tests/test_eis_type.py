"""
EIS type: position-scored regex, and disagreement routed to a human
(PIPELINE_REVIEW 2.6).

Two defects, both in `m1.extract_eis_type`:

1. The regex loop `break`s on the first pattern that matched, so
   `config.EIS_TYPE_PATTERNS`' insertion order decided the label rather than
   where the phrase appears on the page. The order is
   `Supplemental, ROD, Draft, Final`, so a Final EIS whose cover page mentioned
   "record of decision" anywhere became a ROD, and one that mentioned a
   "supplement" became Supplemental.

2. On regex/Sonnet disagreement the code returned the REGEX value with
   `confidence: low`. In the one observed failure -- Lincoln Hwy, graded
   `wrong: "ROD", correct: "Final"` -- the regex was the source that was wrong,
   so the tie-break preferred the loser.

`TestDictOrderVsPosition` is the measurement: on seven realistic cover-page
phrasings, dict order gets 3 wrong and position gets 0.

The specificity the config's "order matters" comment protected is NOT lost,
because the patterns are phrase-anchored. That is asserted rather than assumed in
`test_final_supplemental_still_resolves_to_supplemental`.
"""

from __future__ import annotations

import pytest

import m1 as m1_mod
from mcal import settings
from segment_b import gate as G

from conftest import SYNTHETIC_PAGES, build_m1, build_m2
from test_gate import all_results, reload_artifacts


@pytest.fixture
def doc(doc_factory):
    return doc_factory(*SYNTHETIC_PAGES, doc_id="synthetic")


# (cover page text, correct label, label dict-order scoring returns)
CASES = [
    (
        "FINAL SUPPLEMENTAL ENVIRONMENTAL IMPACT STATEMENT for the project",
        "Supplemental",
        "Supplemental",
    ),
    (
        "SUPPLEMENTAL DRAFT ENVIRONMENTAL IMPACT STATEMENT",
        "Supplemental",
        "Supplemental",
    ),
    (
        "FINAL ENVIRONMENTAL IMPACT STATEMENT ... a supplement to the 1972 report",
        "Final",
        "Supplemental",
    ),
    (
        "FINAL ENVIRONMENTAL IMPACT STATEMENT ... and record of decision follows",
        "Final",
        "ROD",
    ),
    (
        "RECORD OF DECISION for the final environmental impact statement",
        "ROD",
        "ROD",
    ),
    (
        "FINAL SUPPLEMENT TO THE ENVIRONMENTAL IMPACT STATEMENT",
        "Supplemental",
        "Supplemental",
    ),
    (
        "DRAFT ENVIRONMENTAL IMPACT STATEMENT with no supplement",
        "Draft",
        "Supplemental",
    ),
]


def _dict_order_label(head: str):
    """The old rule: first pattern in config order that matches, then break."""
    import re

    from config import EIS_TYPE_PATTERNS

    for label, pat in EIS_TYPE_PATTERNS.items():
        if re.search(pat, head, re.IGNORECASE):
            return label
    return None


class TestDictOrderVsPosition:
    @pytest.mark.parametrize("head,correct,_old", CASES)
    def test_position_scoring_is_right(self, head, correct, _old):
        hits = m1_mod.eis_type_regex_hits(head)
        assert hits, head
        assert hits[0]["label"] == correct

    @pytest.mark.parametrize("head,correct,old", CASES)
    def test_dict_order_scoring_was_wrong_where_claimed(self, head, correct, old):
        """Pins the old behaviour so the improvement is measured, not asserted."""
        assert _dict_order_label(head) == old

    def test_the_tally(self):
        n_wrong_old = sum(1 for h, c, _ in CASES if _dict_order_label(h) != c)
        n_wrong_new = sum(
            1 for h, c, _ in CASES if m1_mod.eis_type_regex_hits(h)[0]["label"] != c
        )
        assert (n_wrong_old, n_wrong_new) == (3, 0)

    def test_hits_are_ordered_by_position_and_carry_offsets(self):
        hits = m1_mod.eis_type_regex_hits(
            "FINAL ENVIRONMENTAL IMPACT STATEMENT ... and record of decision follows"
        )
        assert [h["label"] for h in hits] == ["Final", "ROD"]
        assert hits[0]["offset"] < hits[1]["offset"]
        assert "matched" in hits[0]

    def test_final_supplemental_still_resolves_to_supplemental(self):
        """
        The case `config.EIS_TYPE_PATTERNS`' ordering comment existed to protect.
        It survives on position because the Final pattern needs "final" adjacent to
        "environmental", which "SUPPLEMENTAL" interrupts -- so Final never matches
        and there is nothing for the position rule to get wrong.
        """
        head = "FINAL SUPPLEMENTAL ENVIRONMENTAL IMPACT STATEMENT"
        labels = [h["label"] for h in m1_mod.eis_type_regex_hits(head)]
        assert labels == ["Supplemental"]
        assert "Final" not in labels

    def test_no_match_returns_empty(self):
        assert m1_mod.eis_type_regex_hits("ANNUAL REPORT OF THE COMMISSION") == []


class TestOnTheRealCorpus:
    def test_the_one_document_where_the_rules_disagree(self):
        """
        `p1074_35556036535615`: "Final" at offset 163, "Draft" at 520. Dict order
        returns Draft. 1 of 21 materialized documents, and it would have been
        labelled wrong.
        """
        from chunk import first_pages
        from config import FIRST_PAGE
        from pages import load_doc

        resolved = settings.resolve_doc_dir("p1074_35556036535615")
        if resolved is None:
            pytest.skip("document not materialized on this machine")
        doc = load_doc(resolved.name, settings.PAGES_DATA_DIR)
        head = first_pages(doc, FIRST_PAGE)

        hits = m1_mod.eis_type_regex_hits(head)
        assert [h["label"] for h in hits] == ["Final", "Draft"]
        assert hits[0]["label"] == "Final"
        assert _dict_order_label(head) == "Draft"


class TestDisagreementGoesToAHuman:
    @staticmethod
    def _entry(**kw):
        base = {
            "value": "ROD",
            "confidence": "low",
            "sources": ["regex (first page)", "Sonnet (first 2 pages)"],
            "note": "regex=ROD disagrees with Sonnet=Final",
            "review_required": True,
            "candidates": [
                {"value": "ROD", "source": "regex (first page)"},
                {"value": "Final", "source": "Sonnet (first 2 pages)"},
            ],
        }
        base.update(kw)
        return base

    @staticmethod
    def _gate(doc, tmp_path, m1):
        return G.run_gate(
            "synthetic", m1, build_m2(), all_results(),
            stage="v1", doc=doc, out_dir=tmp_path / "out",
            monitor_file=tmp_path / "null_tag_monitor.json",
        )

    def test_review_required_is_detected(self):
        assert G.review_required(self._entry()) is True
        assert G.review_required({"value": "Final", "confidence": "high"}) is False
        assert G.review_required(None) is False
        assert G.review_required("Final") is False

    def test_it_gates_the_field(self, mcal_artifacts, doc, tmp_path):
        reload_artifacts(mcal_artifacts, tau=0.4)
        m1 = build_m1()
        m1["eis_type"] = self._entry()
        g = self._gate(doc, tmp_path, m1).fields["eis_type"]
        assert g.gated_to_human is True
        assert g.gate_reason == "m1_source_disagreement"

    def test_both_candidates_survive_into_the_manifest(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        MCAL_PLAN 7 Q8: the reviewer grades from the manifest, so both readings have
        to be there. A gate that dropped the loser would leave them re-deriving it.
        """
        reload_artifacts(mcal_artifacts, tau=0.4)
        m1 = build_m1()
        m1["eis_type"] = self._entry()
        g = self._gate(doc, tmp_path, m1).fields["eis_type"]
        # `extracted_value` unwraps an M1 entry to its scalar, so the losing
        # candidate has to be carried explicitly or it never reaches the reviewer.
        assert g.extracted_value == "ROD"
        review = g.extraction_review
        assert [c["value"] for c in review["candidates"]] == ["ROD", "Final"]
        assert "disagrees with" in review["note"]
        assert review["review_required"] is True
        assert g.to_manifest_entry()["extraction_review"] == review

    def test_agreement_does_not_gate(self, mcal_artifacts, doc, tmp_path):
        reload_artifacts(mcal_artifacts, tau=0.4)
        out = self._gate(doc, tmp_path, build_m1())
        assert "m1_source_disagreement" not in out.fields["eis_type"].gate_reasons

    def test_a_disagreement_null_tag_is_not_a_taxonomy_gap(
        self, mcal_artifacts, doc, tmp_path
    ):
        reload_artifacts(mcal_artifacts, tau=0.4)
        m1 = build_m1()
        m1["eis_type"] = self._entry()
        out = self._gate(doc, tmp_path, m1)
        assert out.rollup["null_tag"]["M1"]["n_excluded_pre_judgement"] >= 1
