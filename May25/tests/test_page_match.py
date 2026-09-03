"""
Cited page vs matched page (PIPELINE_REVIEW 2.7).

`check_quote` searches `cited +/- 2` and records `matched_page`, but
`aggregate_verdict` returns a bare `yes|mixed|no`, so a quote cited to p.36 and
found on p.34 has always scored identically to an exact hit. The review called
this "mis-citation is live and unmeasured" and asked for a signal.

Measuring it first changes the conclusion. On the extraction path the invariant
holds absolutely -- 0 divergences over 1985 located quotes across the 12-doc v1
batch and the 8 graded documents -- because `evidence.verify_and_locate` writes
the page `doc.find_quote` FOUND the quote on. The cite is derived from the match,
so it cannot disagree with it, and a cited-vs-matched signal would be a constant.

The Evaluation sheet notes the review cited as evidence ("missing citation - pg
34, 36, 146") are a reviewer naming pages that SHOULD have been cited: T01
omission, which atom-level `s_citation` addresses, not a wrong page.

So this is instrumented as a REGRESSION GUARD rather than scored. The invariant
depends on the cite being derived from an exact substring match, and would break
the day `find_quote` is swapped for the OCR-normalized fuzzy matcher or a prompt
starts asserting page numbers itself. `test_the_corpus_invariant_holds` is what
would notice.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from mcal import quote_check as qc, settings
from segment_b import critic as C
from segment_b import gate as G

from conftest import SYNTHETIC_PAGES, VERIFIABLE_QUOTE


@pytest.fixture
def doc(doc_factory):
    return doc_factory(*SYNTHETIC_PAGES, doc_id="synthetic")


# --- The helper --------------------------------------------------------------


class TestPageMatchSummary:
    def test_exact_hit_is_exact(self, doc):
        ev = [{"quote": VERIFIABLE_QUOTE, "source_pages": ["4"]}]
        out = qc.page_match_summary(ev, qc.check_evidence_list(ev, doc))
        assert out["n_quotes"] == 1
        assert out["n_located"] == 1
        assert out["n_exact_page"] == 1
        assert out["exact_page_rate"] == 1.0
        assert out["divergences"] == []

    def test_a_neighbouring_page_is_recorded_as_divergent(self, doc):
        """
        The case the review is about: located, verified, and NOT on the cited page.
        The verdict stays "yes" -- the tolerance is deliberate -- but the fact is
        no longer discarded.
        """
        ev = [{"quote": VERIFIABLE_QUOTE, "source_pages": ["6"]}]
        checks = qc.check_evidence_list(ev, doc)
        assert qc.aggregate_verdict(checks) == "yes"

        out = qc.page_match_summary(ev, checks)
        assert out["n_located"] == 1
        assert out["n_exact_page"] == 0
        assert out["exact_page_rate"] == 0.0
        assert len(out["divergences"]) == 1
        d = out["divergences"][0]
        assert d["cited_pages"] == [6]
        assert d["matched_page"] == 4
        assert d["delta"] == -2
        assert d["reason"] == "found_within_tolerance_on_another_page"

    def test_an_unlocatable_quote_is_excluded_from_the_rate(self, doc):
        """
        Already fully accounted for by `s_quote`. Counting it here would mix two
        different failures into one number.
        """
        ev = [{"quote": "wholly absent sentence about nothing", "source_pages": ["4"]}]
        out = qc.page_match_summary(ev, qc.check_evidence_list(ev, doc))
        assert out["n_quotes"] == 1
        assert out["n_located"] == 0
        assert out["exact_page_rate"] is None
        assert out["divergences"] == []

    def test_an_uncited_quote_is_purely_an_s_quote_matter(self, doc):
        """
        `check_quote` searches `cited +/- tolerance`, so with nothing cited it
        searches nothing and cannot locate the quote. There is no page question to
        answer, and the missing citation is already what `s_quote` scores 0.
        """
        ev = [{"quote": VERIFIABLE_QUOTE, "source_pages": []}]
        checks = qc.check_evidence_list(ev, doc)
        assert checks[0].verified == "no"
        assert checks[0].pages_searched == []
        out = qc.page_match_summary(ev, checks)
        assert out["n_located"] == 0
        assert out["divergences"] == []

    def test_a_failed_matchs_best_guess_page_is_not_a_mis_citation(self, doc):
        """
        `check_quote` reports `matched_page` even when the verdict is "no" -- it is
        where the quote came closest. Counting that as located would report an
        ABSENT quote as one found on the wrong page.
        """
        ev = [{"quote": "wholly absent sentence about nothing", "source_pages": ["4"]}]
        checks = qc.check_evidence_list(ev, doc)
        assert checks[0].verified == "no"
        assert checks[0].matched_page is not None
        out = qc.page_match_summary(ev, checks)
        assert out["n_located"] == 0
        assert out["divergences"] == []

    def test_empty_input(self):
        out = qc.page_match_summary([], [])
        assert out["n_quotes"] == 0
        assert out["exact_page_rate"] is None


# --- Reaching the manifest ---------------------------------------------------


class TestItReachesTheManifest:
    def test_field_entry_carries_page_match(self, doc):
        entry = {
            "text": "x",
            "evidence": [{"quote": VERIFIABLE_QUOTE, "source_pages": ["6"]}],
        }
        audit = G.extraction_quote_audit("summary.overview", entry, doc)
        assert audit["verdict"] == "yes"
        assert audit["page_match"]["n_exact_page"] == 0

    def test_it_is_the_extractions_not_the_critics(self, doc):
        """
        `FieldGate.quote_check` is the CRITIC's check of a different quote.
        `quote_page_match` must describe the extraction, or the two get conflated.
        """
        entry = {
            "text": "x",
            "evidence": [{"quote": VERIFIABLE_QUOTE, "source_pages": ["6"]}],
        }
        gate = G.gate_field(
            "summary.overview",
            entry,
            C.CriticResult(
                field="summary.overview",
                bucket="summary_narrative",
                verdict="PASS",
                quote_check={"verified": "yes", "score": 100.0},
            ),
            _threshold(),
            stage="v1",
            doc=doc,
        )
        assert gate.quote_check == {"verified": "yes", "score": 100.0}
        assert gate.quote_page_match["divergences"][0]["matched_page"] == 4

    def test_m1_fields_have_no_page_match(self, doc):
        """
        M1 values are not quotes, and `critic.source_pages_for_field` defaults them
        to the `[1, 2, 3]` search WINDOW. Measuring divergence against a window is
        meaningless, and doing it produced a spurious 5.4% on the Critic path.
        """
        audit = G.extraction_quote_audit("year", {"value": 1977}, doc)
        assert audit["page_match"] is None

    def test_document_less_call_records_nothing(self):
        audit = G.extraction_quote_audit("summary.overview", {"evidence": []}, None)
        assert audit["page_match"] is None


def _threshold():
    from mcal.confidence import BucketThreshold

    return BucketThreshold(
        bucket="summary_narrative",
        alpha=0.15,
        alpha_effective=0.15,
        n_wrong_docs=6,
        n_items=12,
        n_wrong_items=6,
        tau_raw=0.4,
        curation_slack=0.0,
        tau_deployed=0.4,
        saturated=False,
        degenerate=False,
        degenerate_severe=False,
        gate_all_to_human=False,
        guarantee_conditioning="test",
    )


# --- The corpus invariant ----------------------------------------------------


class TestCorpusInvariant:
    """
    The measurement that turned this from a signal into a guard.
    """

    @staticmethod
    def _iter_docs():
        from pages import load_doc

        # Anchored to settings, not the CWD. A relative "segment_b/output" glob
        # silently matched nothing when pytest ran from the repo root, which
        # dropped the 12-doc batch and made the corpus size -- and therefore this
        # test's own threshold -- depend on where it was invoked from.
        segment_b_output = settings.SEGMENT_B_DIR / "output"
        roots = [
            (p.parent.name, p, p.parent / "m1.json")
            for p in sorted(segment_b_output.glob("*/m2.json"))
        ]
        roots += [
            (p.stem, p, settings.M1_DIR / f"{p.stem}.json")
            for p in sorted(settings.M2_DIR.glob("*.json"))
        ]
        for doc_id, m2_path, m1_path in roots:
            resolved = settings.resolve_doc_dir(doc_id)
            if resolved is None:
                continue
            doc = load_doc(resolved.name, settings.PAGES_DATA_DIR)
            m2 = json.loads(m2_path.read_text())
            m1 = json.loads(m1_path.read_text()) if m1_path.exists() else {}
            yield doc_id, doc, m1, m2

    def test_the_corpus_invariant_holds(self):
        """
        Every page cite in every extraction, on every document on this machine, is
        the page the quote was found on. If this fails, something started asserting
        page numbers instead of deriving them, and `exact_page_rate` has become a
        real signal worth weighting.
        """
        import evidence as ev_mod

        n_located = n_exact = 0
        divergences = []
        for doc_id, doc, m1, m2 in self._iter_docs():
            for field in settings.ALL_FIELDS:
                if field in settings.M1_FIELDS:
                    continue
                entry = C.extracted_entry(field, m1, m2)
                own = ev_mod.own_evidence(C.evidence_dicts(field, entry))
                if not own:
                    continue
                out = qc.page_match_summary(own, qc.check_evidence_list(own, doc))
                n_located += out["n_located"]
                n_exact += out["n_exact_page"]
                for d in out["divergences"]:
                    divergences.append((doc_id, field, d))

        if n_located == 0:
            pytest.skip("no materialized documents on this machine")
        assert n_located > 1900, f"only {n_located} quotes checked; corpus shrank?"
        assert divergences == [], (
            f"{len(divergences)} cited/matched divergences appeared. The "
            f"extraction path used to be exact by construction; see "
            f"quote_check.page_match_summary. First: {divergences[:3]}"
        )
        assert n_exact == n_located
