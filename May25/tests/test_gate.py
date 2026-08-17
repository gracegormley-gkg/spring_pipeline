"""
Tests for segment_b/gate.py (MCAL_PLAN 3.12, 6, 7 Q8, build item #12).

Three things are pinned hard, because the multi-round protocol (MCAL_PLAN 7.5)
breaks without them:

  1. `run_manifest.json` carries every key of the 3.12 schema for EVERY field,
     gated or not, with its raw extraction. At seed v1 most fields are gated and
     the reviewer grades from this file alone.
  2. `gate_reason` distinguishes WHY, so "the gate is too conservative" and "the
     Critic is the binding constraint" never collapse into one value.
  3. `summary_of_interest`'s `[]` (routine document) stays distinguishable from
     `null` (generation failure).

NO network, NO LLM calls. Critic results are constructed directly where the test
is about the gate, and produced by `segment_b/critic.py` with an injected fake
where the test is about the two modules fitting together. Thresholds and configs
are synthetic (`mcal_artifacts`), because `mcal/build.py` has not run.
"""

from __future__ import annotations

import json

import pytest

from mcal import settings
from segment_b import critic as C
from segment_b import gate as G

from conftest import (
    FABRICATED_QUOTE,
    SYNTHETIC_PAGES,
    VERIFIABLE_QUOTE,
    build_m1,
    build_m2,
)

RUBRIC = {
    "Q1": "yes", "Q2": "yes", "Q3": "yes", "Q4": "yes",
    "Q5": "no", "Q6a": "yes", "Q6b": "no",
}


# --- Helpers ----------------------------------------------------------------


@pytest.fixture
def doc(doc_factory):
    return doc_factory(*SYNTHETIC_PAGES, doc_id="synthetic")


def cr(field: str, verdict: str = "PASS", **kw) -> C.CriticResult:
    """A Critic result for one field, without touching an LLM."""
    kw.setdefault("evidence_quote", VERIFIABLE_QUOTE)
    kw.setdefault("rubric_answers", dict(RUBRIC))
    kw.setdefault("source_pages", [4])
    kw.setdefault("quote_check", {"verified": "yes", "score": 100.0})
    kw.setdefault("judge_model", C.judge_model_for(field)[1])
    return C.CriticResult(
        field=field,
        bucket=settings.bucket_for_field(field),
        verdict=verdict,
        **kw,
    )


def all_results(verdict: str = "PASS", **per_field) -> dict[str, C.CriticResult]:
    """Critic results for all 15 canonical fields, overridable per field."""
    out = {f: cr(f, verdict) for f in settings.ALL_FIELDS}
    out.update(per_field)
    return out


def reload_artifacts(artifacts, **kw):
    """Rewrite thresholds/config and drop the per-process artifact cache."""
    if "config" in kw:
        artifacts.write_config(**kw.pop("config"))
    artifacts.write_thresholds(**kw)
    C.clear_artifact_cache()
    return artifacts


def gate(doc, results, artifacts, tmp_path, **kw):
    kw.setdefault("stage", "v1")
    kw.setdefault("doc", doc)
    kw.setdefault("out_dir", tmp_path / "out")
    kw.setdefault("monitor_file", tmp_path / "null_tag_monitor.json")
    return G.run_gate(
        "synthetic", build_m1(), kw.pop("m2", None) or build_m2(), results, **kw
    )


# --- Threshold loading ------------------------------------------------------


class TestThresholdLoading:
    def test_missing_thresholds_is_an_actionable_error(self, mcal_artifacts):
        settings.artifact_path("thresholds.json", "v1").unlink()
        C.clear_artifact_cache()
        with pytest.raises(G.MissingArtifactError) as e:
            G.load_bucket_thresholds("v1")
        assert "python -m mcal.build" in str(e.value)

    def test_all_seven_buckets_are_required(self, mcal_artifacts):
        """
        A partial thresholds file would mean some fields silently had no gate.
        Bucket definitions are frozen (MCAL_PLAN 7.5), so a missing one is a
        build error, not a configuration choice.
        """
        path = settings.artifact_path("thresholds.json", "v1")
        payload = json.loads(path.read_text())
        del payload["buckets"]["location"]
        path.write_text(json.dumps(payload))
        C.clear_artifact_cache()
        with pytest.raises(G.MissingArtifactError) as e:
            G.load_bucket_thresholds("v1")
        assert "location" in str(e.value)

    def test_no_buckets_object_at_all(self, mcal_artifacts):
        with pytest.raises(G.MissingArtifactError):
            G.thresholds_from_payload({"version": "v1"})

    def test_null_tau_is_treated_as_gate_everything(self, mcal_artifacts):
        """Defaulting the other way would accept a whole bucket on a missing value."""
        payload = json.loads(
            settings.artifact_path("thresholds.json", "v1").read_text()
        )
        payload["buckets"]["M1"]["tau_deployed"] = None
        ths = G.thresholds_from_payload(payload)
        assert ths["M1"].gate_all_to_human is True
        assert ths["M1"].accepts(1.0) is False
        assert any("forced to gate_all_to_human" in n for n in ths["M1"].notes)

    def test_infinite_tau_round_trips(self, mcal_artifacts):
        """`confidence.save_thresholds` writes +inf for a gated bucket."""
        reload_artifacts(mcal_artifacts, degenerate_severe=("summary_of_interest",))
        ths = G.load_bucket_thresholds("v1")
        assert ths["summary_of_interest"].tau_deployed == float("inf")
        assert ths["summary_of_interest"].accepts(1.0) is False

    def test_thresholds_use_the_same_accept_rule_as_the_simulation(self, mcal_artifacts):
        """
        MCAL_PLAN 3.3: accept iff composite > tau_deployed. Rehydrating into the
        real `BucketThreshold` means gate_simulation.v(N).json and the production
        gate cannot drift apart.
        """
        ths = G.load_bucket_thresholds("v1")
        th = ths["alternatives+themes"]
        assert th.accepts(th.tau_deployed) is False
        assert th.accepts(th.tau_deployed + 0.01) is True


# --- Signals ----------------------------------------------------------------


class TestSignals:
    def test_m1_s_quote_defaults_to_one(self, doc):
        """
        MCAL_PLAN 3.3: M1 values carry no verbatim quote by design, so the M1
        composite is `0.5*s_critic + 0.5` -- a 0.5 floor.
        """
        m1 = build_m1()
        sig, verdict = G.signals_for_field("year", m1["year"], cr("year"), doc)
        assert verdict is None
        assert sig.s_quote == 1.0
        from mcal import confidence

        assert confidence.composite(sig) == 1.0

    def test_s_quote_comes_from_the_extractions_own_quotes(self, doc):
        """
        Not from the Critic's `evidence_quote`. That one is already accounted for:
        an unverifiable Critic quote forces HUMAN_REVIEW, which drives s_critic to
        0. Reusing it would double-count one measurement as two signals.
        """
        m2 = build_m2()
        entry = m2["summary"]["overview"]
        sig, verdict = G.signals_for_field(
            "summary.overview", entry, cr("summary.overview"), doc
        )
        assert verdict == "yes"
        assert sig.s_quote == 1.0

        entry["evidence"] = [{"quote": FABRICATED_QUOTE, "source_pages": ["4"]}]
        sig, verdict = G.signals_for_field(
            "summary.overview", entry, cr("summary.overview"), doc
        )
        assert verdict == "no"
        assert sig.s_quote == 0.0

    def test_no_evidence_scores_zero_not_not_applicable(self, doc):
        """MCAL_PLAN 1(5)-(7) is a family of missing-citation failures."""
        sig, verdict = G.signals_for_field(
            "themes", {"value": {"themes": []}}, cr("themes"), doc
        )
        assert verdict == "no"
        assert sig.s_quote == 0.0

    def test_zero_weight_signals_are_still_recorded(self, doc):
        """
        Weight validation at n>=60 cannot retro-fit a signal nobody recorded
        (MCAL_PLAN 3.3).
        """
        m1, m2 = build_m1(), build_m2()
        sig, _ = G.signals_for_field(
            "summary.overview", m2["summary"]["overview"], cr("summary.overview"), doc
        )
        assert sig.s_citation == 1.0
        sig, _ = G.signals_for_field("year", m1["year"], cr("year"), doc)
        # m1's note says NUL disagrees with the regex, so agreement is 1 of 2.
        assert sig.s_source == 0.5

    def test_document_less_call_does_not_penalize_the_field(self):
        """
        Manifest regeneration from persisted Critic results can verify nothing;
        asserting "no" would penalize every field for the caller's convenience.
        """
        sig, verdict = G.signals_for_field(
            "themes", build_m2()["themes"], cr("themes"), None
        )
        assert verdict is None


# --- Gate decision + gate_reason --------------------------------------------


class TestGateReasons:
    def test_clean_field_is_not_gated(self, mcal_artifacts, doc, tmp_path):
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        g = out.fields["summary.overview"]
        assert g.gated_to_human is False
        assert g.gate_reason is None
        assert g.verdict == "PASS"
        assert g.composite == 1.0

    def test_composite_below_tau_overrides_a_critic_pass(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        MCAL_PLAN 3.12: "if composite <= tau_deployed, emit HUMAN_REVIEW
        regardless of Critic verdict." This is the genuine "gate too
        conservative" signal -- the Critic passed and the score still failed.
        """
        reload_artifacts(mcal_artifacts, tau=0.6)
        m2 = build_m2()
        m2["summary"]["overview"]["evidence"] = [
            {"quote": FABRICATED_QUOTE, "source_pages": ["4"]}
        ]
        out = gate(doc, all_results(), mcal_artifacts, tmp_path, m2=m2)
        g = out.fields["summary.overview"]
        assert g.critic_verdict == "PASS"
        assert g.composite == 0.5
        assert g.applied_tau == 0.6
        assert g.gated_to_human is True
        assert g.verdict == "HUMAN_REVIEW"
        assert g.gate_reason == "composite_below_tau"

    def test_accept_rule_is_strictly_greater_than(self, mcal_artifacts, doc, tmp_path):
        reload_artifacts(mcal_artifacts, tau=1.0)
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        g = out.fields["summary.overview"]
        assert g.composite == 1.0 and g.applied_tau == 1.0
        assert g.gated_to_human is True
        assert g.gate_reason == "composite_below_tau"

    def test_bucket_degenerate_severe_gates_unconditionally(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        MCAL_PLAN 0/3.3: at seed v1 most buckets have N_wrong_docs < 3 and
        `gate_all_to_human=true`. Nearly every field routing to HUMAN_REVIEW is
        the intended behaviour.
        """
        reload_artifacts(mcal_artifacts, degenerate_severe=("summary_of_interest",))
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        g = out.fields[settings.SUMMARY_OF_INTEREST]
        assert g.gated_to_human is True
        assert g.gate_reason == "bucket_degenerate_severe"
        assert g.applied_tau is None  # +inf is not representable in strict JSON
        assert g.bucket_flags["tau_deployed_infinite"] is True

    def test_reduced_geocoder_stack_is_not_reported_as_degeneracy(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        MCAL_PLAN 3.9a forces the location bucket to gate_all when PAD-US / GNIS /
        MAPBOX_TOKEN are missing. Reported as `bucket_degenerate_severe` it would
        look solvable by grading more documents; it is solvable by two downloads.
        """
        mcal_artifacts.write_config(geocoder_stack="reduced")
        reload_artifacts(mcal_artifacts, gate_all=("location",))
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        g = out.fields["location"]
        assert g.gated_to_human is True
        assert g.gate_reason == "reduced_geocoder_stack"
        assert out.rollup["geocoder_stack"] == "reduced"

    def test_critic_verdict_outranks_composite_below_tau(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        When the Critic says HUMAN_REVIEW, s_critic = 0 so the composite is at
        most 0.5*s_quote and `composite_below_tau` is usually ALSO true. Reporting
        it would make the gate look like the binding constraint in every case
        where the judge was.
        """
        reload_artifacts(mcal_artifacts, tau=0.6)
        out = gate(
            doc,
            all_results(themes=cr("themes", "HUMAN_REVIEW")),
            mcal_artifacts, tmp_path,
        )
        g = out.fields["themes"]
        assert g.composite == 0.5 and g.applied_tau == 0.6
        assert "composite_below_tau" in g.gate_reasons
        assert g.gate_reason == "critic_verdict"

    def test_re_extract_verdict_gates_the_field(self, mcal_artifacts, doc, tmp_path):
        """A field still RE_EXTRACT after its one retry goes to a human."""
        out = gate(
            doc,
            all_results(alternatives=cr("alternatives", "RE_EXTRACT")),
            mcal_artifacts, tmp_path,
        )
        assert out.fields["alternatives"].gate_reason == "critic_verdict"
        assert out.fields["alternatives"].verdict == "HUMAN_REVIEW"

    def test_pass_with_note_is_not_gated_by_itself(self, mcal_artifacts, doc, tmp_path):
        out = gate(
            doc,
            all_results(
                **{
                    "summary.public_response": cr(
                        "summary.public_response", "PASS_WITH_NOTE",
                        failure_tag="T04_undefined_acronym",
                    )
                }
            ),
            mcal_artifacts, tmp_path,
        )
        g = out.fields["summary.public_response"]
        assert g.gated_to_human is False
        assert g.verdict == "PASS_WITH_NOTE"
        assert g.failure_tag == "T04_undefined_acronym"

    def test_policy_private_individual_reason(self, mcal_artifacts, doc, tmp_path):
        kp = cr("key_people", "HUMAN_REVIEW")
        kp.overrides = [C.NOTE_PRIVATE_INDIVIDUAL]
        kp.verdict_before_override = "PASS"
        out = gate(doc, all_results(key_people=kp), mcal_artifacts, tmp_path)
        g = out.fields["key_people"]
        assert g.gate_reason == "policy_private_individual"
        assert g.verdict_before_override == "PASS"

    def test_ambiguous_capacity_maps_to_the_policy_reason(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        MCAL_PLAN 3.12's enum has one policy value; the dual-capacity case in 3.5
        is part of the same private-individual provision, so it maps there and the
        specific override is kept in `critic_overrides`.
        """
        kp = cr("key_people", "HUMAN_REVIEW")
        kp.overrides = [C.NOTE_AMBIGUOUS_CAPACITY]
        out = gate(doc, all_results(key_people=kp), mcal_artifacts, tmp_path)
        assert out.fields["key_people"].gate_reason == "policy_private_individual"
        assert C.NOTE_AMBIGUOUS_CAPACITY in out.fields["key_people"].critic_overrides

    def test_dependent_field_cascade_reason(self, mcal_artifacts, doc, tmp_path):
        """
        Folding this into `critic_verdict` would make key_people look like a
        Critic problem when the actual problem is `year` (MCAL_PLAN 3.10 step 2).
        """
        kp = cr("key_people", "HUMAN_REVIEW")
        kp.overrides = [C.NOTE_DEPENDENT_CASCADE]
        out = gate(
            doc,
            all_results(year=cr("year", "RE_EXTRACT"), key_people=kp),
            mcal_artifacts, tmp_path,
        )
        assert out.fields["key_people"].gate_reason == "dependent_field_cascade"

    def test_policy_outranks_cascade(self, mcal_artifacts, doc, tmp_path):
        kp = cr("key_people", "HUMAN_REVIEW")
        kp.overrides = [C.NOTE_PRIVATE_INDIVIDUAL, C.NOTE_DEPENDENT_CASCADE]
        out = gate(doc, all_results(key_people=kp), mcal_artifacts, tmp_path)
        assert out.fields["key_people"].gate_reason == "policy_private_individual"
        assert "dependent_field_cascade" in out.fields["key_people"].gate_reasons

    def test_extraction_missing_reason(self, mcal_artifacts, doc, tmp_path):
        m2 = build_m2()
        del m2["themes"]
        themes = cr("themes", "HUMAN_REVIEW")
        themes.overrides = [C.NOTE_EXTRACTION_MISSING]
        out = gate(
            doc, all_results(themes=themes), mcal_artifacts, tmp_path, m2=m2
        )
        g = out.fields["themes"]
        assert g.gate_reason == "extraction_missing"
        assert g.extracted_value is None
        assert g.extraction_missing is True

    def test_critic_missing_field_is_emitted_not_skipped(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        MCAL_PLAN 7 Q8 / 6 acceptance item 4 both depend on the manifest being
        complete, so a field the Critic never reached is emitted with a reason.
        """
        results = all_results()
        del results["location"]
        out = gate(doc, results, mcal_artifacts, tmp_path)
        assert len(out.fields) == 15
        g = out.fields["location"]
        assert g.gate_reason == "critic_missing"
        assert g.gated_to_human is True
        assert g.extracted_value is not None  # raw extraction still emitted

    def test_every_reason_used_is_in_the_documented_vocabulary(
        self, mcal_artifacts, doc, tmp_path
    ):
        reload_artifacts(
            mcal_artifacts, tau=0.6, degenerate_severe=("summary_of_interest",)
        )
        kp = cr("key_people", "HUMAN_REVIEW")
        kp.overrides = [C.NOTE_PRIVATE_INDIVIDUAL]
        out = gate(
            doc,
            all_results(
                key_people=kp,
                themes=cr("themes", "HUMAN_REVIEW"),
                year=cr("year", "RE_EXTRACT"),
            ),
            mcal_artifacts, tmp_path,
        )
        for g in out.fields.values():
            assert g.gate_reason is None or g.gate_reason in G.GATE_REASONS
            for r in g.gate_reasons:
                assert r in G.GATE_REASONS


# --- Raw extraction is always emitted ---------------------------------------


class TestRawExtractionAlwaysEmitted:
    def test_gated_fields_keep_their_extraction(self, mcal_artifacts, doc, tmp_path):
        """
        MCAL_PLAN 7 Q8: "Gated fields (composite <= tau_deployed) also emit their
        raw extraction alongside the HUMAN_REVIEW flag. This is critical for the
        multi-round protocol." Without it, v2 has no calibration data.
        """
        reload_artifacts(mcal_artifacts, gate_all=tuple(settings.BUCKET_ORDER))
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        assert out.n_gated == 15
        for field, g in out.fields.items():
            if field == settings.SUMMARY_OF_INTEREST:
                assert g.extracted_value == []
            else:
                assert g.extracted_value is not None, field
            assert g.evidence_quote == VERIFIABLE_QUOTE
            assert g.source_pages == [4]
            assert g.rubric_answers["Q1"] == "yes"

    def test_no_field_is_ever_dropped(self, mcal_artifacts, doc, tmp_path):
        out = gate(doc, {}, mcal_artifacts, tmp_path)
        assert set(out.fields) == set(settings.ALL_FIELDS)
        assert len(out.fields) == 15

    def test_summary_values_are_unwrapped_for_the_reviewer(
        self, mcal_artifacts, doc, tmp_path
    ):
        """The envelope metadata is plumbing; a grader needs the answer."""
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        value = out.fields["summary.project_description"].extracted_value
        assert isinstance(value, str)
        assert "project description" in value


# --- Manifest ---------------------------------------------------------------


class TestManifest:
    def test_every_field_carries_every_schema_key(self, mcal_artifacts, doc, tmp_path):
        """MCAL_PLAN 3.12's per-field schema, verbatim."""
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        for field in settings.ALL_FIELDS:
            entry = out.manifest[field]
            for key in G.MANIFEST_REQUIRED_KEYS:
                assert key in entry, f"{field} missing {key}"

    def test_schema_keys_match_the_plan_exactly(self):
        assert G.MANIFEST_REQUIRED_KEYS == (
            "extracted_value", "evidence_quote", "source_pages", "verdict",
            "rubric_answers", "composite", "applied_tau", "gated_to_human",
            "gate_reason", "failure_tag", "bucket", "artifact_stage", "judge_model",
        )

    def test_rubric_answers_include_q6b_for_the_offline_audit(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        MCAL_PLAN 3.5/3.12: Q6(b) is logged-only at v1 and the manifest "is where
        the offline audit reads it from".
        """
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        assert out.manifest["summary.affected_community"]["rubric_answers"]["Q6b"] == "no"

    def test_soi_rubric_answers_include_q7(self, mcal_artifacts, doc, tmp_path):
        soi = cr(settings.SUMMARY_OF_INTEREST)
        soi.rubric_answers = dict(RUBRIC, Q7a="yes", Q7b="no", Q7c="yes")
        out = gate(doc, all_results(**{settings.SUMMARY_OF_INTEREST: soi}),
                   mcal_artifacts, tmp_path)
        answers = out.manifest[settings.SUMMARY_OF_INTEREST]["rubric_answers"]
        for key in ("Q7a", "Q7b", "Q7c"):
            assert key in answers

    def test_bucket_and_stage_are_recorded_per_field(self, mcal_artifacts, doc, tmp_path):
        """
        `artifact_stage` lets grades collected under v1 be told apart from v2's
        during later recalibration (MCAL_PLAN 3.12).
        """
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        assert out.manifest["location"]["bucket"] == "location"
        assert out.manifest["location"]["artifact_stage"] == "v1"
        assert out.manifest["summary.environmental_impact"]["bucket"] == "summary_numeric"

    def test_judge_model_is_reported_as_opus_or_sonnet(
        self, mcal_artifacts, doc, tmp_path
    ):
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        assert out.manifest["summary.public_response"]["judge_model"] == "opus"
        assert out.manifest["title"]["judge_model"] == "sonnet"

    def test_reserved_keys_cannot_collide_with_a_field_name(self):
        for key in G.RESERVED_KEYS:
            assert key.startswith("_")
            assert key not in settings.ALL_FIELDS

    def test_manifest_is_strict_json(self, mcal_artifacts, doc, tmp_path):
        """
        `allow_nan=False`: an inf tau or NaN composite must fail loudly rather
        than produce a file other JSON parsers reject.
        """
        reload_artifacts(mcal_artifacts, gate_all=("location",))
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        text = json.dumps(out.manifest, allow_nan=False)
        assert "Infinity" not in text and "NaN" not in text

    def test_manifest_is_written_per_doc(self, mcal_artifacts, doc, tmp_path):
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        assert out.manifest_path == tmp_path / "out" / "synthetic" / "run_manifest.json"
        on_disk = json.loads(out.manifest_path.read_text())
        assert set(settings.ALL_FIELDS) <= set(on_disk)

    def test_write_can_be_suppressed(self, mcal_artifacts, doc, tmp_path):
        out = gate(doc, all_results(), mcal_artifacts, tmp_path, write=False)
        assert out.manifest_path is None
        assert not (tmp_path / "out").exists()

    def test_meta_tells_the_reviewer_how_to_read_it(self, mcal_artifacts, doc, tmp_path):
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        meta = out.manifest[G.META_KEY]
        assert meta["doc_id"] == "synthetic"
        assert meta["artifact_stage"] == "v1"
        assert meta["n_fields"] == 15
        assert "routine" in meta["reviewer_note"]

    def test_audit_extras_do_not_displace_the_schema(self, mcal_artifacts, doc, tmp_path):
        """Extras are additive; a reviewer UI written against 3.12 still works."""
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        entry = out.manifest["themes"]
        keys = list(entry)
        assert keys[: len(G.MANIFEST_REQUIRED_KEYS)] == list(G.MANIFEST_REQUIRED_KEYS)
        assert "signals" in entry and "gate_reasons" in entry


# --- summary_of_interest ----------------------------------------------------


class TestSummaryOfInterest:
    def test_empty_list_is_emitted_as_an_empty_list(self, mcal_artifacts, doc, tmp_path):
        """
        MCAL_PLAN 3.12: `extracted_value: []` is a legitimate empty result. An
        empty list is a substantive result -- the document is routine -- not a
        missing value.
        """
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        entry = out.manifest[settings.SUMMARY_OF_INTEREST]
        assert entry["extracted_value"] == []
        assert entry["extracted_value"] is not None

    def test_generation_failure_is_null_plus_a_gate_reason(
        self, mcal_artifacts, doc, tmp_path
    ):
        m2 = build_m2()
        del m2["summary_of_interest"]
        soi = cr(settings.SUMMARY_OF_INTEREST, "HUMAN_REVIEW")
        soi.overrides = [C.NOTE_EXTRACTION_MISSING]
        out = gate(
            doc, all_results(**{settings.SUMMARY_OF_INTEREST: soi}),
            mcal_artifacts, tmp_path, m2=m2,
        )
        entry = out.manifest[settings.SUMMARY_OF_INTEREST]
        assert entry["extracted_value"] is None
        assert entry["gate_reason"] == "extraction_missing"

    def test_empty_and_null_are_distinguishable(self, mcal_artifacts, doc, tmp_path):
        """The two cases must never be conflated (MCAL_PLAN 3.12)."""
        empty = gate(doc, all_results(), mcal_artifacts, tmp_path)
        m2 = build_m2(summary_of_interest=None)
        missing = gate(doc, all_results(), mcal_artifacts, tmp_path, m2=m2)
        a = empty.manifest[settings.SUMMARY_OF_INTEREST]
        b = missing.manifest[settings.SUMMARY_OF_INTEREST]
        assert a["extracted_value"] == [] and b["extracted_value"] is None
        assert a["gate_reason"] != b["gate_reason"]

    def test_always_emitted_even_when_gated(self, mcal_artifacts, doc, tmp_path):
        reload_artifacts(mcal_artifacts, degenerate_severe=("summary_of_interest",))
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        assert settings.SUMMARY_OF_INTEREST in out.manifest
        assert out.manifest[settings.SUMMARY_OF_INTEREST]["extracted_value"] == []

    def test_diagnostics_cover_the_plan_6_list(self, mcal_artifacts, doc, tmp_path):
        m2 = build_m2(
            summary_of_interest=[
                {
                    "claim": "1,200 acres of sagebrush habitat would be affected",
                    "salience_criterion": "large_magnitude",
                    "page": 4,
                    "evidence_quote": VERIFIABLE_QUOTE,
                    "why_notable": "Largest quantified impact in the document.",
                    "evidence": [{"quote": VERIFIABLE_QUOTE, "source_pages": ["4"]}],
                },
                {
                    "claim": "Residents of three census tracts objected",
                    "salience_criterion": "community_pushback",
                    "page": 5,
                    "evidence_quote": "Residents of three census tracts objected",
                    "why_notable": "Comment volume changed the alignment.",
                    "evidence": [{"quote": "Residents of three census tracts objected",
                                  "source_pages": ["5"]}],
                },
            ]
        )
        out = gate(doc, all_results(), mcal_artifacts, tmp_path, m2=m2)
        d = out.rollup["summary_of_interest"]
        assert d["non_empty"] is True and d["n_entries"] == 2
        assert d["salience_criterion_counts"]["large_magnitude"] == 1
        assert d["salience_criterion_counts"]["community_pushback"] == 1
        assert d["overlap_with_standard_summary"]["jaccard"] is not None
        assert d["n_t17_manufactured_salience"] == 0
        assert d["soi_useful"] is None  # reviewer-filled (MCAL_PLAN 7 Q5)

    def test_off_vocabulary_criterion_is_reported(self, mcal_artifacts, doc, tmp_path):
        m2 = build_m2(
            summary_of_interest=[
                {"claim": "c", "salience_criterion": "interesting", "page": 4,
                 "evidence_quote": "q", "why_notable": "w"}
            ]
        )
        out = gate(doc, all_results(), mcal_artifacts, tmp_path, m2=m2)
        assert out.rollup["summary_of_interest"]["off_vocabulary_criteria"] == [
            "interesting"
        ]

    def test_t17_and_t18_are_counted(self, mcal_artifacts, doc, tmp_path):
        soi = cr(
            settings.SUMMARY_OF_INTEREST, "PASS_WITH_NOTE",
            failure_tag="T18_salience_duplicates_summary",
        )
        out = gate(doc, all_results(**{settings.SUMMARY_OF_INTEREST: soi}),
                   mcal_artifacts, tmp_path)
        assert out.rollup["summary_of_interest"]["n_t18_duplicates_summary"] == 1
        assert out.rollup["summary_of_interest"]["n_t17_manufactured_salience"] == 0

    def test_duplicated_claim_shows_high_jaccard(self):
        """
        MCAL_PLAN 6: high overlap means the field is duplicating rather than
        complementing -- the T18 signal at the corpus level.
        """
        summary = {
            "overview": {
                "text": "The reconstruction would affect 1,200 acres of sagebrush "
                        "habitat in Cook County over thirty years."
            }
        }
        dup = [
            {
                "claim": "The reconstruction would affect 1,200 acres of sagebrush "
                         "habitat in Cook County over thirty years.",
                "why_notable": "",
            }
        ]
        novel = [{"claim": "Tribal consultation letters were returned unopened.",
                  "why_notable": ""}]
        assert G.soi_summary_overlap(dup, summary)["jaccard"] > 0.8
        assert G.soi_summary_overlap(novel, summary)["jaccard"] < 0.2

    def test_jaccard_ignores_nepa_boilerplate(self):
        """
        Sharing "environmental impact statement project alternatives" is not
        evidence of duplication -- it is every page of every EIS. Reusing
        `quote_check.content_tokens` is what filters it.
        """
        boilerplate = (
            "The environmental impact statement of the proposed project alternatives."
        )
        summary = {"overview": {"text": boilerplate}}
        other = [{"claim": boilerplate, "why_notable": ""}]
        overlap = G.soi_summary_overlap(other, summary)
        # Every word is a stopword or NEPA filler, so neither side contributes a
        # single content token and the metric reports no overlap rather than 1.0.
        assert overlap["n_soi_tokens"] == 0
        assert overlap["n_summary_tokens"] == 0
        assert overlap["jaccard"] is None

    def test_batch_aggregate_flags_a_manufactured_salience_rate(self):
        """
        MCAL_PLAN 3.15: above ~60% non-empty, treat the field as manufacturing
        salience rather than detecting it.
        """
        per_doc = [{"present": True, "non_empty": True, "salience_criterion_counts": {}}] * 8
        per_doc += [{"present": True, "non_empty": False, "salience_criterion_counts": {}}] * 2
        agg = G.aggregate_soi_diagnostics(per_doc)
        assert agg["non_empty_rate"] == 0.8
        assert agg["exceeds_nonempty_ceiling"] is True
        assert agg["gating"] is False

    def test_batch_aggregate_flags_a_too_strict_rubric(self):
        per_doc = [{"present": True, "non_empty": False, "salience_criterion_counts": {}}] * 10
        agg = G.aggregate_soi_diagnostics(per_doc)
        assert agg["non_empty_rate"] == 0.0
        assert agg["near_zero_nonempty_rate"] is True

    def test_batch_aggregate_flags_a_collapsed_criterion_distribution(self):
        per_doc = [
            {
                "present": True,
                "non_empty": True,
                "salience_criterion_counts": {"contested": 2},
            }
        ] * 6
        agg = G.aggregate_soi_diagnostics(per_doc)
        assert agg["criterion_distribution_collapsed"] is True
        assert agg["n_criteria_used"] == 1


# --- Null-tag monitor (MCAL_PLAN 6) -----------------------------------------


class TestNullTagMonitor:
    def test_human_review_with_null_tag_is_counted(self, mcal_artifacts, doc, tmp_path):
        out = gate(
            doc,
            all_results(themes=cr("themes", "HUMAN_REVIEW", failure_tag=None)),
            mcal_artifacts, tmp_path,
        )
        row = out.rollup["null_tag"]["alternatives+themes"]
        assert row["n_human_review"] == 1
        assert row["n_null_tag"] == 1

    def test_human_review_with_a_tag_is_not_counted(self, mcal_artifacts, doc, tmp_path):
        out = gate(
            doc,
            all_results(
                themes=cr("themes", "HUMAN_REVIEW",
                          failure_tag="T03_outside_text_fabrication")
            ),
            mcal_artifacts, tmp_path,
        )
        row = out.rollup["null_tag"]["alternatives+themes"]
        assert row["n_human_review"] == 1
        assert row["n_null_tag"] == 0

    def test_threshold_is_flagged_when_exceeded(self, mcal_artifacts, doc, tmp_path):
        """
        MCAL_PLAN 6: above 15% in any bucket, the taxonomy needs a v(N+1) refresh
        with new T19+ codes.
        """
        results = all_results()
        for f in settings.M1_FIELDS:
            results[f] = cr(f, "HUMAN_REVIEW", failure_tag=None)
        out = gate(doc, results, mcal_artifacts, tmp_path)
        mon = out.null_tag_monitor
        assert mon["threshold"] == settings.NULL_TAG_REFRESH_THRESHOLD == 0.15
        assert mon["per_bucket"]["M1"]["null_tag_rate"] == 1.0
        assert mon["per_bucket"]["M1"]["exceeds_threshold"] is True
        assert "M1" in mon["buckets_needing_refresh"]
        assert mon["taxonomy_refresh_recommended"] is True

    def test_below_threshold_does_not_recommend_a_refresh(
        self, mcal_artifacts, doc, tmp_path
    ):
        results = all_results()
        for f in settings.M1_FIELDS:
            results[f] = cr(f, "HUMAN_REVIEW", failure_tag="T11_year_ocr_error")
        out = gate(doc, results, mcal_artifacts, tmp_path)
        assert out.null_tag_monitor["per_bucket"]["M1"]["null_tag_rate"] == 0.0
        assert out.null_tag_monitor["taxonomy_refresh_recommended"] is False

    def test_it_is_not_a_halt_condition(self, mcal_artifacts, doc, tmp_path):
        """MCAL_PLAN 6: "not a Segment B halt condition"."""
        results = all_results()
        for f in settings.M1_FIELDS:
            results[f] = cr(f, "HUMAN_REVIEW", failure_tag=None)
        out = gate(doc, results, mcal_artifacts, tmp_path)
        assert out.null_tag_monitor["halt_condition"] is False
        assert out.manifest_path.exists()  # the run completed

    def test_policy_routes_are_excluded_from_the_numerator(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        `_base.md` decision rule 1 mandates `failure_tag = null` on a policy
        route, so counting it would send the next taxonomy revision looking for a
        code that should not exist.
        """
        kp = cr("key_people", "HUMAN_REVIEW", failure_tag=None)
        kp.overrides = [C.NOTE_PRIVATE_INDIVIDUAL]
        out = gate(doc, all_results(key_people=kp), mcal_artifacts, tmp_path)
        row = out.rollup["null_tag"]["key_people"]
        assert row["n_human_review"] == 0
        assert row["n_null_tag"] == 0
        assert row["n_excluded_policy"] == 1

    def test_pre_judgement_gates_are_excluded_from_the_numerator(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        At seed v1 most fields are gated before any judgement is made; counting
        them would put every bucket near 1.0 and destroy the signal exactly when
        it is supposed to be read.
        """
        reload_artifacts(mcal_artifacts, degenerate_severe=tuple(settings.BUCKET_ORDER))
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        for row in out.rollup["null_tag"].values():
            assert row["n_human_review"] == 0
            assert row["n_excluded_pre_judgement"] == row["n_fields"]
        assert out.null_tag_monitor["taxonomy_refresh_recommended"] is False

    def test_fields_with_no_value_are_excluded_from_the_numerator(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        A field with no extraction gives the taxonomy nothing to categorize, so
        its null tag is not evidence of a missing category. Real Segment A M2
        output predates `summary_of_interest`, so this is the common case.
        """
        m2 = build_m2()
        del m2["summary_of_interest"]
        soi = cr(settings.SUMMARY_OF_INTEREST, "HUMAN_REVIEW", failure_tag=None)
        soi.overrides = [C.NOTE_EXTRACTION_MISSING]
        out = gate(
            doc, all_results(**{settings.SUMMARY_OF_INTEREST: soi}),
            mcal_artifacts, tmp_path, m2=m2,
        )
        row = out.rollup["null_tag"]["summary_of_interest"]
        assert row["n_human_review"] == 0
        assert row["n_excluded_pre_judgement"] == 1
        assert out.null_tag_monitor["taxonomy_refresh_recommended"] is False

    def test_every_excluded_reason_is_documented(self):
        for reason in G.NULL_TAG_PRE_JUDGEMENT_REASONS:
            assert reason in G.GATE_REASONS

    def test_off_vocabulary_tags_are_counted_separately(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        A judge inventing codes is the other explanation for a null tag;
        conflating the two would send a prompt problem to the taxonomy for repair.
        """
        themes = cr(
            "themes", "HUMAN_REVIEW", failure_tag=None,
            off_vocabulary_failure_tag="T77_invented",
        )
        out = gate(doc, all_results(themes=themes), mcal_artifacts, tmp_path)
        row = out.rollup["null_tag"]["alternatives+themes"]
        assert row["n_off_vocabulary"] == 1
        assert row["n_null_tag"] == 1

    def test_monitor_is_rolling_across_documents(self, mcal_artifacts, doc, tmp_path):
        """
        MCAL_PLAN 2 lists null_tag_monitor.json as the one artifact that is not
        stage-versioned and is maintained at batch level: a single document's rate
        is computed over a handful of items and is noise.
        """
        results = all_results()
        for f in settings.M1_FIELDS:
            results[f] = cr(f, "HUMAN_REVIEW", failure_tag=None)
        first = gate(doc, results, mcal_artifacts, tmp_path, batch_id="b1")
        second = gate(doc, results, mcal_artifacts, tmp_path, batch_id="b1")
        assert first.null_tag_monitor["n_docs"] == 1
        assert second.null_tag_monitor["n_docs"] == 2
        assert second.null_tag_monitor["per_bucket"]["M1"]["n_human_review"] == 8
        assert [b["batch_id"] for b in second.null_tag_monitor["batches"]] == ["b1", "b1"]

    def test_monitor_defaults_to_the_settings_path(self, mcal_artifacts, doc, tmp_path):
        out = gate(doc, all_results(), mcal_artifacts, tmp_path, monitor_file=None)
        assert settings.NULL_TAG_MONITOR_PATH.exists()
        assert out.null_tag_monitor["path"] == str(settings.NULL_TAG_MONITOR_PATH)

    def test_corrupt_monitor_file_is_recovered_from(self, mcal_artifacts, doc, tmp_path):
        path = tmp_path / "null_tag_monitor.json"
        path.write_text("{not json")
        out = gate(doc, all_results(), mcal_artifacts, tmp_path, monitor_file=path)
        assert out.null_tag_monitor["n_docs"] == 1
        assert json.loads(path.read_text())["threshold"] == 0.15

    def test_threshold_comes_from_settings_not_the_stale_file(
        self, mcal_artifacts, doc, tmp_path
    ):
        path = tmp_path / "null_tag_monitor.json"
        path.write_text(json.dumps({"threshold": 0.99, "per_bucket": {}}))
        out = gate(doc, all_results(), mcal_artifacts, tmp_path, monitor_file=path)
        assert out.null_tag_monitor["threshold"] == 0.15

    def test_monitor_can_be_disabled(self, mcal_artifacts, doc, tmp_path):
        out = gate(doc, all_results(), mcal_artifacts, tmp_path, update_monitor=False)
        assert out.null_tag_monitor == {}
        assert G.MONITOR_KEY not in out.manifest

    def test_manifest_surfaces_the_monitor_verdict(self, mcal_artifacts, doc, tmp_path):
        results = all_results()
        for f in settings.M1_FIELDS:
            results[f] = cr(f, "HUMAN_REVIEW", failure_tag=None)
        out = gate(doc, results, mcal_artifacts, tmp_path)
        block = out.manifest[G.MONITOR_KEY]
        assert block["taxonomy_refresh_recommended"] is True
        assert block["buckets_needing_refresh"] == ["M1"]
        assert block["halt_condition"] is False


# --- Doc-level rollup -------------------------------------------------------


class TestRollup:
    def test_counts(self, mcal_artifacts, doc, tmp_path):
        out = gate(
            doc,
            all_results(
                themes=cr("themes", "HUMAN_REVIEW"),
                location=cr("location", "RE_EXTRACT"),
            ),
            mcal_artifacts, tmp_path,
        )
        r = out.rollup
        assert r["n_fields"] == 15
        assert r["n_gated"] == 2
        assert r["n_passed"] == 13
        assert r["gate_rate"] == round(2 / 15, 4)
        assert r["gate_reasons"]["critic_verdict"] == 2
        assert out.gated_fields() == ["location", "themes"]

    def test_per_bucket_gate_counts(self, mcal_artifacts, doc, tmp_path):
        reload_artifacts(mcal_artifacts, degenerate_severe=("summary_narrative",))
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        bucket = out.rollup["per_bucket"]["summary_narrative"]
        assert bucket["n_fields"] == 5
        assert bucket["n_gated"] == 5
        assert bucket["gate_all_to_human"] is True
        assert bucket["degenerate_severe"] is True
        assert out.rollup["per_bucket"]["M1"]["n_gated"] == 0

    def test_every_bucket_appears_even_when_empty_of_gated_fields(
        self, mcal_artifacts, doc, tmp_path
    ):
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        assert set(out.rollup["per_bucket"]) == set(settings.BUCKET_ORDER)

    def test_critic_diagnostics_are_folded_in(self, mcal_artifacts, doc, tmp_path):
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        assert out.rollup["critic"]["n_fields"] == 15
        assert out.rollup["critic"]["verdicts"]["PASS"] == 15

    def test_rollup_is_json_serializable(self, mcal_artifacts, doc, tmp_path):
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        json.dumps(out.rollup, allow_nan=False)


# --- RE_EXTRACT retry (MCAL_PLAN 7 Q8) --------------------------------------


class TestReExtract:
    def test_retry_is_offered_at_temperature_plus_point_two(
        self, mcal_artifacts, doc, tmp_path
    ):
        seen: list[dict] = []

        def retry(field, *, attempt, temperature, entry, critic_result):
            seen.append(
                {"field": field, "attempt": attempt, "temperature": temperature}
            )
            return None

        gate(
            doc,
            all_results(themes=cr("themes", "RE_EXTRACT")),
            mcal_artifacts, tmp_path, reextract=retry,
        )
        assert len(seen) == 1
        assert seen[0]["field"] == "themes"
        assert seen[0]["attempt"] == 1
        assert seen[0]["temperature"] == pytest.approx(
            G.BASE_EXTRACTION_TEMPERATURE + G.RE_EXTRACT_TEMPERATURE_DELTA
        )
        assert seen[0]["temperature"] == pytest.approx(0.4)

    def test_only_re_extract_fields_are_retried(self, mcal_artifacts, doc, tmp_path):
        seen: list[str] = []

        def retry(field, **kw):
            seen.append(field)
            return None

        gate(
            doc,
            all_results(
                themes=cr("themes", "RE_EXTRACT"),
                location=cr("location", "HUMAN_REVIEW"),
            ),
            mcal_artifacts, tmp_path, reextract=retry,
        )
        assert seen == ["themes"]

    def test_exactly_one_attempt(self, mcal_artifacts, doc, tmp_path):
        """MCAL_PLAN 7 Q8 says one. A loop would burn tokens to the same answer."""
        n = {"calls": 0}

        def retry(field, **kw):
            n["calls"] += 1
            return G.ReExtraction(critic_result=cr(field, "RE_EXTRACT"))

        out = gate(
            doc, all_results(themes=cr("themes", "RE_EXTRACT")),
            mcal_artifacts, tmp_path, reextract=retry,
        )
        assert n["calls"] == 1
        assert out.fields["themes"].re_extract["attempt"] == 1
        assert out.fields["themes"].re_extract["max_attempts"] == 1
        assert out.fields["themes"].gate_reason == "critic_verdict"

    def test_successful_retry_replaces_the_verdict(self, mcal_artifacts, doc, tmp_path):
        def retry(field, **kw):
            return G.ReExtraction(critic_result=cr(field, "PASS"), note="fixed")

        out = gate(
            doc, all_results(themes=cr("themes", "RE_EXTRACT")),
            mcal_artifacts, tmp_path, reextract=retry,
        )
        g = out.fields["themes"]
        assert g.gated_to_human is False
        assert g.verdict == "PASS"
        assert g.re_extract["verdict_before"] == "RE_EXTRACT"
        assert g.re_extract["verdict_after"] == "PASS"
        assert g.re_extract["replaced"] is True

    def test_a_worse_retry_is_kept_anyway(self, mcal_artifacts, doc, tmp_path):
        """
        Cherry-picking the better of two samples would bias s_critic upward
        relative to the calibration set, where each item was scored once.
        """
        def retry(field, **kw):
            return G.ReExtraction(critic_result=cr(field, "HUMAN_REVIEW"))

        out = gate(
            doc, all_results(themes=cr("themes", "RE_EXTRACT")),
            mcal_artifacts, tmp_path, reextract=retry,
        )
        assert out.fields["themes"].critic_verdict == "HUMAN_REVIEW"

    def test_callback_returning_only_an_entry_triggers_a_re_critique(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        The "re-run through Critic" half of MCAL_PLAN 7 Q8: the caller re-extracts,
        the gate re-judges.
        """
        judged: list[str] = []

        def fake_call(model, system, user, *, max_tokens, temperature):
            judged.append(model)
            return {
                "evidence_quote": VERIFIABLE_QUOTE,
                "rubric_answers": dict(RUBRIC),
                "verdict": "PASS",
                "failure_tag": None,
                "note": None,
            }

        def retry(field, **kw):
            return {
                "entry": {
                    "value": {"themes": ["Transportation Infrastructure"]},
                    "evidence": [{"quote": VERIFIABLE_QUOTE, "source_pages": ["4"]}],
                }
            }

        out = gate(
            doc, all_results(themes=cr("themes", "RE_EXTRACT")),
            mcal_artifacts, tmp_path, reextract=retry, call=fake_call,
        )
        assert judged  # the Critic was re-run
        assert out.fields["themes"].verdict == "PASS"
        assert "re_extracted_at_temperature_+0.2" in out.fields["themes"].note

    def test_replacement_extraction_reaches_the_manifest(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        A reviewer grading the manifest in isolation must see the value the
        surviving verdict was passed on, not the superseded one.
        """
        replacement = {
            "value": {"themes": ["Land Use and Planning"]},
            "evidence": [{"quote": VERIFIABLE_QUOTE, "source_pages": ["4"]}],
        }

        def retry(field, **kw):
            return G.ReExtraction(
                entry=replacement, critic_result=cr(field, "PASS"), note="second try"
            )

        out = gate(
            doc, all_results(themes=cr("themes", "RE_EXTRACT")),
            mcal_artifacts, tmp_path, reextract=retry,
        )
        entry = out.manifest["themes"]
        assert entry["extracted_value"] == {"themes": ["Land Use and Planning"]}
        assert entry["verdict"] == "PASS"
        assert entry["re_extract"]["entry_replaced"] is True

    def test_new_value_without_a_new_verdict_is_not_adopted(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        Without a document there is nothing to re-judge against, so adopting the
        value would put the retry's extraction under the original verdict -- an
        inconsistency a reviewer grading the manifest in isolation could not see.
        """
        out = G.run_gate(
            "synthetic", build_m1(), build_m2(),
            all_results(themes=cr("themes", "RE_EXTRACT")),
            stage="v1", doc=None, out_dir=tmp_path / "out",
            monitor_file=tmp_path / "m.json",
            reextract=lambda field, **kw: {"entry": {"value": {"themes": ["Other"]}}},
        )
        rec = out.fields["themes"].re_extract
        assert rec["replaced"] is False
        assert "not adopted" in rec["note"]
        assert out.fields["themes"].extracted_value != {"themes": ["Other"]}

    def test_callback_may_decline(self, mcal_artifacts, doc, tmp_path):
        out = gate(
            doc, all_results(themes=cr("themes", "RE_EXTRACT")),
            mcal_artifacts, tmp_path, reextract=lambda field, **kw: None,
        )
        rec = out.fields["themes"].re_extract
        assert rec["attempted"] is True and rec["replaced"] is False
        assert out.fields["themes"].verdict == "HUMAN_REVIEW"

    def test_callback_exception_does_not_lose_the_document(
        self, mcal_artifacts, doc, tmp_path
    ):
        def boom(field, **kw):
            raise RuntimeError("extractor exploded")

        out = gate(
            doc, all_results(themes=cr("themes", "RE_EXTRACT")),
            mcal_artifacts, tmp_path, reextract=boom,
        )
        assert "extractor exploded" in out.fields["themes"].re_extract["error"]
        assert len(out.fields) == 15
        assert out.manifest_path.exists()

    def test_no_callback_means_no_retry(self, mcal_artifacts, doc, tmp_path):
        out = gate(
            doc, all_results(themes=cr("themes", "RE_EXTRACT")),
            mcal_artifacts, tmp_path,
        )
        assert out.re_extract == {}
        assert out.fields["themes"].re_extract is None

    def test_retry_audit_reaches_the_manifest(self, mcal_artifacts, doc, tmp_path):
        def retry(field, **kw):
            return G.ReExtraction(critic_result=cr(field, "PASS"))

        out = gate(
            doc, all_results(themes=cr("themes", "RE_EXTRACT")),
            mcal_artifacts, tmp_path, reextract=retry,
        )
        assert out.manifest["themes"]["re_extract"]["temperature"] == 0.4
        assert out.rollup["re_extract"]["themes"]["replaced"] is True


# --- End to end with the Critic ---------------------------------------------


class TestCriticToGate:
    def _call(self, **overrides):
        base = {
            "evidence_quote": VERIFIABLE_QUOTE,
            "rubric_answers": dict(RUBRIC),
            "verdict": "PASS",
            "failure_tag": None,
            "note": None,
        }
        base.update(overrides)

        def fn(model, system, user, *, max_tokens, temperature):
            return base

        return fn

    def test_full_pass_over_one_document(self, mcal_artifacts, doc, tmp_path):
        results = C.run_critic(
            doc, build_m1(), build_m2(), stage="v1", call=self._call()
        )
        out = gate(doc, results, mcal_artifacts, tmp_path)
        assert len(out.fields) == 15
        assert out.n_gated == 0
        for field in settings.ALL_FIELDS:
            for key in G.MANIFEST_REQUIRED_KEYS:
                assert key in out.manifest[field]

    def test_quote_verify_override_propagates_into_the_manifest(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        The full anti-hallucination path: fabricated Critic quote -> quote check
        fails -> HUMAN_REVIEW -> gated with `critic_verdict`, and the original
        verdict is still on the record.
        """
        results = C.run_critic(
            doc, build_m1(), build_m2(), stage="v1",
            call=self._call(evidence_quote=FABRICATED_QUOTE),
        )
        out = gate(doc, results, mcal_artifacts, tmp_path)
        entry = out.manifest["summary.environmental_impact"]
        assert entry["verdict"] == "HUMAN_REVIEW"
        assert entry["gate_reason"] == "critic_verdict"
        assert entry["verdict_before_override"] == "PASS"
        assert C.NOTE_EVIDENCE_UNVERIFIABLE in entry["note"]
        assert entry["quote_check"]["verified"] == "no"

    def test_seed_v1_shape_most_fields_gated_all_still_gradable(
        self, mcal_artifacts, doc, tmp_path
    ):
        """
        MCAL_PLAN 6 seed-v1 acceptance item 4: Segment B runs end to end and
        emits run_manifest.json with the full per-field schema INCLUDING
        extracted_value / evidence_quote / source_pages for gated fields, "so
        they can actually be graded".
        """
        reload_artifacts(
            mcal_artifacts, degenerate_severe=tuple(settings.BUCKET_ORDER)
        )
        results = C.run_critic(
            doc, build_m1(), build_m2(), stage="v1", call=self._call()
        )
        out = gate(doc, results, mcal_artifacts, tmp_path)
        assert out.n_gated == 15
        on_disk = json.loads(out.manifest_path.read_text())
        for field in settings.ALL_FIELDS:
            entry = on_disk[field]
            assert entry["gated_to_human"] is True
            assert entry["gate_reason"] == "bucket_degenerate_severe"
            assert entry["evidence_quote"]
            assert "extracted_value" in entry
            assert entry["rubric_answers"]
            if field != settings.SUMMARY_OF_INTEREST:
                assert entry["source_pages"]
            else:
                # An empty summary_of_interest cites nothing, by construction.
                assert entry["source_pages"] == []
                assert entry["extracted_value"] == []

    def test_stage_is_resolved_from_disk_when_not_given(
        self, mcal_artifacts, doc, tmp_path
    ):
        results = C.run_critic(doc, build_m1(), build_m2(), call=self._call())
        out = G.run_gate(
            "synthetic", build_m1(), build_m2(), results,
            doc=doc, out_dir=tmp_path / "out",
            monitor_file=tmp_path / "m.json",
        )
        assert out.stage == "v1"


# --- No network -------------------------------------------------------------


class TestNoNetwork:
    def test_gate_never_calls_an_llm(self, mcal_artifacts, doc, tmp_path, monkeypatch):
        import llm

        def boom(*a, **kw):
            raise AssertionError("real LLM client reached in a test")

        monkeypatch.setattr(llm, "call_with_usage", boom)
        monkeypatch.setattr(llm, "call_json_with_usage", boom)
        monkeypatch.setattr(C, "call_json_with_usage", boom)
        out = gate(doc, all_results(), mcal_artifacts, tmp_path)
        assert len(out.fields) == 15
