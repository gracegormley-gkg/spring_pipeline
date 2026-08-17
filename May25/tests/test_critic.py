"""
Tests for segment_b/critic.py (MCAL_PLAN 3.11, 3.5, 7 Q3, build items #2, #14).

Covers the four things the rewrite exists to fix, in the order the module applies
them: per-field granularity (15 verdicts, not segment_a's 9), the evidence-first
schema and its validation, the deterministic quote-verify override, and the
policy / dependent-field overrides.

NO test here makes a network or LLM call. Every path takes an injected `call`,
and `TestNoNetwork` pins that the injection point is the only one by making the
real client raise if it is ever reached.

Artifacts are constructed, not built: `mcal/artifacts/` does not exist until
`mcal/build.py` runs and a human ratifies the draft (MCAL_PLAN 3.7), so the
`mcal_artifacts` fixture writes a promoted stage v1 into a tmp dir and repoints
`settings.ARTIFACTS_DIR` at it.
"""

from __future__ import annotations

import json

import pytest

from mcal import settings
from segment_b import critic as C

from conftest import (
    FABRICATED_QUOTE,
    SYNTHETIC_PAGES,
    VERIFIABLE_QUOTE,
    build_m1,
    build_m2,
)


# --- Helpers ----------------------------------------------------------------


def response(**overrides) -> dict:
    """
    A well-formed Critic response, in MCAL_PLAN 3.5's key order.

    Built as an ordered literal on purpose: `_check_key_order` reads the key
    order `json.loads` preserved, so a test that wants to exercise it has to
    control it.
    """
    base = {
        "evidence_quote": VERIFIABLE_QUOTE,
        "rubric_answers": {
            "Q1": "yes", "Q2": "yes", "Q3": "yes", "Q4": "yes",
            "Q5": "no", "Q6a": "yes", "Q6b": "no",
        },
        "verdict": "PASS",
        "failure_tag": None,
        "note": None,
    }
    base.update(overrides)
    return base


def recorder(default=None, *, by_field=None, exc=None):
    """
    A fake `call`, recording its arguments.

    `by_field` maps a field name to the response for that field only; every other
    field gets `default` (or a clean PASS). The field is recovered from the user
    message, which names it in a `## FIELD UNDER REVIEW` section -- deliberately
    not from the model id, since 6 fields share the Opus judge.
    """
    calls: list[dict] = []

    def _fn(model, system, user, *, max_tokens, temperature):
        calls.append(
            {
                "model": model,
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if exc is not None:
            raise exc
        for field, resp in (by_field or {}).items():
            if f"`{field}`" in user:
                return resp
        return response() if default is None else default

    _fn.calls = calls
    return _fn


@pytest.fixture
def doc(doc_factory):
    return doc_factory(*SYNTHETIC_PAGES, doc_id="synthetic")


def critique(field, doc, call, *, m1=None, m2=None, **kw):
    return C.critique_field(
        field, doc, m1 if m1 is not None else build_m1(),
        m2 if m2 is not None else build_m2(),
        stage="v1", call=call, **kw,
    )


# --- Artifact loading -------------------------------------------------------


class TestArtifacts:
    def test_no_promoted_stage_is_an_actionable_error(self, tmp_path, monkeypatch):
        """`mcal/artifacts/` may not exist yet; the message must say what to run."""
        monkeypatch.setattr(settings, "ARTIFACTS_DIR", tmp_path / "artifacts")
        C.clear_artifact_cache()
        with pytest.raises(C.MissingArtifactError) as e:
            C.resolve_stage()
        msg = str(e.value)
        assert "python -m mcal.build" in msg
        assert "--stage v1" in msg

    def test_unpromoted_draft_is_named_in_the_error(self, tmp_path, monkeypatch):
        """A built-but-unratified draft is the likeliest real cause."""
        art = tmp_path / "artifacts"
        (art / "v1-draft").mkdir(parents=True)
        monkeypatch.setattr(settings, "ARTIFACTS_DIR", art)
        C.clear_artifact_cache()
        with pytest.raises(C.MissingArtifactError) as e:
            C.resolve_stage()
        assert "v1-draft" in str(e.value)
        assert "Ratify" in str(e.value)

    def test_missing_confidence_config(self, mcal_artifacts):
        settings.artifact_path("confidence_config.json", "v1").unlink()
        C.clear_artifact_cache()
        with pytest.raises(C.MissingArtifactError) as e:
            C.load_confidence_config("v1")
        assert "confidence_config" in str(e.value)

    def test_missing_taxonomy_explains_the_null_tag_monitor_stake(self, mcal_artifacts):
        settings.artifact_path("taxonomy.json", "v1").unlink()
        C.clear_artifact_cache()
        with pytest.raises(C.MissingArtifactError) as e:
            C.load_taxonomy("v1")
        assert "null-tag monitor" in str(e.value)

    def test_missing_prompt_is_a_build_error_not_a_document_error(
        self, mcal_artifacts, doc
    ):
        """
        A missing prompt affects every document identically, so it raises rather
        than producing 2,000 HUMAN_REVIEW routes with an obscure note.
        """
        from mcal import critic_prompt

        critic_prompt.prompt_path("themes", "v1", draft=False).unlink()
        C.clear_artifact_cache()
        with pytest.raises(C.MissingArtifactError) as e:
            critique("themes", doc, recorder())
        assert "themes" in str(e.value)

    def test_prompts_and_config_are_cached_per_process(self, mcal_artifacts):
        C.clear_artifact_cache()
        first = C.prompt_for("year", "v1")
        from mcal import critic_prompt

        critic_prompt.prompt_path("year", "v1", draft=False).write_text("CHANGED")
        assert C.prompt_for("year", "v1") == first
        C.clear_artifact_cache()
        assert C.prompt_for("year", "v1") == "CHANGED"


# --- Per-field granularity (the headline change) -----------------------------


class TestPerFieldGranularity:
    def test_fifteen_verdicts_not_nine(self, mcal_artifacts, doc):
        """
        segment_a emitted ONE verdict for all six summary subfields, one for all
        alternatives and one for all three key_people buckets: 9 total. Per-bucket
        conformal thresholds cannot separate what the Critic never separated, so
        MCAL_PLAN 3.11 pays for one verdict per field.
        """
        call = recorder(response())
        results = C.run_critic(doc, build_m1(), build_m2(), stage="v1", call=call)
        assert len(results) == 15
        assert set(results) == set(settings.ALL_FIELDS)
        assert len(call.calls) == 15

    def test_every_summary_subfield_is_judged_separately(self, mcal_artifacts, doc):
        """The 6 summary.* keys segment_a collapsed into one `summary` verdict."""
        results = C.run_critic(doc, build_m1(), build_m2(), stage="v1",
                               call=recorder(response()))
        for field in settings.SUMMARY_FIELDS:
            assert field in results
        assert len(settings.SUMMARY_FIELDS) == 6

    def test_one_bad_subfield_does_not_taint_the_others(self, mcal_artifacts, doc):
        """
        The exact failure segment_a's blended verdict caused: MCAL_PLAN 1(5) has
        public_response missing citations on 4/8 docs while project_description
        was fine on the same documents.
        """
        call = recorder(
            by_field={
                "summary.public_response": response(
                    verdict="RE_EXTRACT",
                    evidence_quote=None,
                    failure_tag="T01_missing_citation",
                ),
            }
        )
        results = C.run_critic(doc, build_m1(), build_m2(), stage="v1", call=call)
        assert results["summary.public_response"].verdict == "RE_EXTRACT"
        assert results["summary.public_response"].failure_tag == "T01_missing_citation"
        assert results["summary.project_description"].verdict == "PASS"
        assert results["summary.overview"].verdict == "PASS"

    def test_judge_model_routing(self, mcal_artifacts, doc):
        """
        Opus for the five summary.* subfields plus summary_of_interest; Sonnet
        elsewhere (MCAL_PLAN 3.11, 7 Q2). summary.overview stays on Sonnet -- it
        is a roll-up of already-Opus-judged subfields.
        """
        results = C.run_critic(doc, build_m1(), build_m2(), stage="v1",
                               call=recorder(response()))
        opus = {f for f, r in results.items() if r.judge_model == "opus"}
        assert opus == set(settings.SUMMARY_SUBFIELDS) | {settings.SUMMARY_OF_INTEREST}
        assert len(opus) == 6
        assert results["summary.overview"].judge_model == "sonnet"
        assert results["key_people"].judge_model == "sonnet"

    def test_config_can_override_judge_routing(self, mcal_artifacts, doc):
        cfg = C.load_confidence_config("v1")
        cfg = dict(cfg)
        cfg["judge_model_by_field"] = {"key_people": "opus", "summary.overview": "opus"}
        model, label = C.judge_model_for("key_people", cfg)
        assert (model, label) == (settings.MODEL_OPUS, "opus")
        # An explicit model id is honoured verbatim: the config is hand-editable.
        cfg["judge_model_by_field"]["themes"] = "us.anthropic.claude-opus-9-9"
        assert C.judge_model_for("themes", cfg) == (
            "us.anthropic.claude-opus-9-9", "opus",
        )

    def test_every_field_has_a_bucket(self, mcal_artifacts, doc):
        results = C.run_critic(doc, build_m1(), build_m2(), stage="v1",
                               call=recorder(response()))
        for field, r in results.items():
            assert r.bucket == settings.bucket_for_field(field)


# --- EVIDENCE section (MCAL_PLAN 7 Q3) --------------------------------------


class TestEvidenceSection:
    def test_window_is_min_minus_2_to_max_plus_2(self):
        pages, mode = C.evidence_pages([10, 12])
        assert pages == [8, 9, 10, 11, 12, 13, 14]
        assert mode == "contiguous"

    def test_window_never_goes_below_page_1(self):
        pages, _ = C.evidence_pages([1])
        assert pages[0] == 1

    def test_scattered_citations_fall_back_to_clustered_windows(self):
        """
        MCAL_PLAN 7 Q3 taken literally makes the window 187 pages for LA
        Transit's `[31, 214, 215]`, of which ~180 are irrelevant and would evict
        the pages that mattered under the char cap.
        """
        pages, mode = C.evidence_pages([31, 214, 215])
        assert mode == "clustered"
        assert 31 in pages and 214 in pages and 215 in pages
        assert 100 not in pages
        assert len(pages) < 20

    def test_clustered_window_still_covers_what_quote_check_searches(self):
        """
        Both modes must be supersets of `cited +/- tolerance`, or the Critic would
        be shown less than the deterministic checker looks at.
        """
        from mcal import quote_check

        cited = [31, 214, 215]
        pages, _ = C.evidence_pages(cited)
        assert set(quote_check.expand_with_tolerance(cited)) <= set(pages)

    def test_page_markers_are_interleaved(self, doc):
        block = C.build_evidence_section(doc, [4])
        assert "[[PAGE 2]]" in block.text
        assert "[[PAGE 4]]" in block.text
        assert "[[PAGE 6]]" in block.text
        assert VERIFIABLE_QUOTE in block.text
        assert block.window_mode == "contiguous"
        assert not block.truncated

    def test_no_cited_pages_yields_an_empty_block(self, doc):
        block = C.build_evidence_section(doc, [])
        assert block.text == ""
        assert block.window_mode == "none"

    def test_cap_is_explicit_and_reported(self, doc):
        """
        segment_a truncated at a bare `[:80_000]` with nothing in the output to
        say it had happened.
        """
        block = C.build_evidence_section(doc, [4], max_chars=200)
        assert block.truncated is True
        assert block.pages_omitted
        assert block.n_chars <= 200

    def test_cap_drops_context_pages_before_cited_ones(self, doc):
        """A cited page is only ever dropped after every context page has been."""
        block = C.build_evidence_section(doc, [4], max_chars=200)
        assert 4 in block.pages_included
        assert 4 not in block.pages_omitted

    def test_cap_logs_when_it_bites(self, doc, caplog):
        with caplog.at_level("WARNING"):
            C.build_evidence_section(doc, [4], max_chars=200)
        assert "EVIDENCE cap" in caplog.text

    def test_single_oversized_page_is_truncated_not_dropped(self, doc_factory):
        big = doc_factory("x" * 50_000, doc_id="big")
        block = C.build_evidence_section(big, [1], max_chars=100)
        assert block.pages_included == [1]
        assert block.truncated is True

    def test_evidence_is_a_dedicated_prompt_section(self, mcal_artifacts, doc):
        call = recorder(response())
        critique("summary.project_description", doc, call)
        user = call.calls[0]["user"]
        assert "## EVIDENCE" in user
        assert "[[PAGE 4]]" in user
        assert "## CITED PAGES" in user
        # The system half is the per-field prompt built by mcal/critic_prompt.py.
        assert "quote-anchored verifier" in call.calls[0]["system"]

    def test_evidence_meta_is_recorded_on_the_result(self, mcal_artifacts, doc):
        r = critique("summary.project_description", doc, recorder(response()))
        assert r.evidence_meta["cited_pages"] == [4]
        assert r.evidence_meta["window_mode"] == "contiguous"
        assert r.evidence_meta["truncated"] is False


# --- Source pages -----------------------------------------------------------


class TestSourcePages:
    def test_m1_defaults_to_front_matter(self):
        pages = C.source_pages_for_field("title", build_m1(), build_m2())
        assert pages == [1, 2, 3]

    def test_m1_entry_evidence_overrides_the_front_matter_default(self):
        """
        MCAL_PLAN 1(1): the real date is often on a signature page -- the Lincoln
        Hwy grade points at page 70. Handing the year Critic pages 1-3 would ask
        it to verify a transmittal-letter quote against the cover.
        """
        m1 = build_m1(
            year={
                "value": 1971,
                "sources": ["adjudicator"],
                "evidence": [{"quote": "Date: June 1, 1971", "source_pages": ["70"]}],
            }
        )
        assert C.source_pages_for_field("year", m1, build_m2()) == [70]

    def test_pages_are_walked_out_of_every_m2_shape(self):
        m1, m2 = build_m1(), build_m2()
        assert C.source_pages_for_field("summary.project_description", m1, m2) == [4]
        assert C.source_pages_for_field("alternatives", m1, m2) == [6]
        assert C.source_pages_for_field("location", m1, m2) == [4]
        assert C.source_pages_for_field("key_people", m1, m2) == [5]

    def test_consulted_entities_bucket_is_picked_up_without_extra_code(self):
        """
        `consulted_entities` postdates segment_a/critic.py's hand-written
        per-field page walker; the recursive walk needed no change for it.
        """
        m2 = build_m2()
        m2["key_people"]["value"]["consulted_entities"] = [
            {"name": "State Library", "evidence": [{"quote": "x", "source_pages": ["99"]}]}
        ]
        assert 99 in C.source_pages_for_field("key_people", build_m1(), m2)

    def test_soi_scalar_page_is_collected(self):
        m2 = build_m2(
            summary_of_interest=[
                {
                    "claim": "c",
                    "salience_criterion": "contested",
                    "page": 42,
                    "evidence_quote": "q",
                    "why_notable": "w",
                }
            ]
        )
        assert C.source_pages_for_field(
            settings.SUMMARY_OF_INTEREST, build_m1(), m2
        ) == [42]

    def test_source_pages_are_not_capped(self):
        """
        segment_a capped the span list at 6-10 per field, which silently removed
        pages from the set the quote-verify override searches. The prompt-size
        problem that cap solved is handled in build_evidence_section instead.
        """
        m2 = build_m2()
        m2["key_people"]["value"]["consulted_entities"] = [
            {"name": f"e{i}", "evidence": [{"quote": "x", "source_pages": [str(i)]}]}
            for i in range(20, 60)
        ]
        pages = C.source_pages_for_field("key_people", build_m1(), m2)
        assert len(pages) > 20


# --- Schema validation ------------------------------------------------------


class TestSchemaValidation:
    def test_unknown_verdict_is_coerced_to_human_review(self, mcal_artifacts, doc):
        """Same coercion segment_a/critic.py did, for the same reason."""
        r = critique("themes", doc, recorder(response(verdict="LOOKS_FINE_TO_ME")))
        assert r.verdict == "HUMAN_REVIEW"
        assert any(
            v.startswith(C.NOTE_UNKNOWN_VERDICT) for v in r.schema_violations
        )
        assert "LOOKS_FINE_TO_ME" in " ".join(r.schema_violations)

    def test_missing_verdict_is_human_review(self, mcal_artifacts, doc):
        raw = response()
        del raw["verdict"]
        assert critique("themes", doc, recorder(raw)).verdict == "HUMAN_REVIEW"

    def test_off_vocabulary_failure_tag_becomes_null_and_is_counted(
        self, mcal_artifacts, doc
    ):
        """
        MCAL_PLAN 6 reads "HUMAN_REVIEW with failure_tag = null" as evidence the
        taxonomy needs new T19+ codes, so a hallucinated tag must neither survive
        (diluting the tag distribution) nor vanish silently (making a prompt
        problem look like a taxonomy gap).
        """
        r = critique(
            "location",
            doc,
            recorder(response(verdict="RE_EXTRACT", failure_tag="T99_invented_code")),
        )
        assert r.failure_tag is None
        assert r.off_vocabulary_failure_tag == "T99_invented_code"
        assert C.NOTE_OFF_VOCABULARY_TAG in " ".join(r.schema_violations)

    def test_tag_from_another_fields_vocabulary_is_off_vocabulary(
        self, mcal_artifacts, doc
    ):
        """T06_geocode_missing is a `location` code; on `themes` it is noise."""
        r = critique(
            "themes", doc,
            recorder(response(verdict="RE_EXTRACT", failure_tag="T06_geocode_missing")),
        )
        assert r.failure_tag is None
        assert r.off_vocabulary_failure_tag == "T06_geocode_missing"

    def test_in_vocabulary_tag_survives(self, mcal_artifacts, doc):
        r = critique(
            "location", doc,
            recorder(response(verdict="RE_EXTRACT", failure_tag="T06_geocode_missing")),
        )
        assert r.failure_tag == "T06_geocode_missing"
        assert r.off_vocabulary_failure_tag is None

    def test_bare_tag_id_is_normalized_not_rejected(self, mcal_artifacts, doc):
        """MCAL_PLAN 3.12's own schema example writes `"failure_tag": "T01|null"`."""
        r = critique(
            "summary.public_response", doc,
            recorder(response(verdict="RE_EXTRACT", failure_tag="t01")),
        )
        assert r.failure_tag == "T01_missing_citation"
        assert r.off_vocabulary_failure_tag is None

    @pytest.mark.parametrize("raw", [None, "null", "none", "", "  ", "n/a"])
    def test_absent_tag_variants_all_mean_null(self, mcal_artifacts, doc, raw):
        r = critique("themes", doc, recorder(response(failure_tag=raw)))
        assert r.failure_tag is None
        assert r.off_vocabulary_failure_tag is None

    def test_q6b_is_always_present_even_though_logged_only(self, mcal_artifacts, doc):
        """
        MCAL_PLAN 3.5/3.12: Q6(b) is logged-only at v1 and run_manifest.json is
        where the offline concreteness audit reads it. A missing key would make an
        unanswered question indistinguishable from an unasked one.
        """
        raw = response(rubric_answers={"Q1": "yes"})
        r = critique("summary.affected_community", doc, recorder(raw))
        for key in C.BASE_RUBRIC_KEYS:
            assert key in r.rubric_answers
        assert r.rubric_answers["Q6b"] == "n/a"

    def test_soi_carries_q7abc(self, mcal_artifacts, doc):
        r = critique(settings.SUMMARY_OF_INTEREST, doc, recorder(response()))
        for key in ("Q7a", "Q7b", "Q7c"):
            assert key in r.rubric_answers

    def test_non_soi_fields_do_not_invent_q7(self, mcal_artifacts, doc):
        r = critique("themes", doc, recorder(response()))
        assert "Q7a" not in r.rubric_answers

    def test_rubric_keys_are_case_normalized(self, mcal_artifacts, doc):
        r = critique(
            "themes", doc,
            recorder(response(rubric_answers={"q6B": "no", "Q1": "YES"})),
        )
        assert r.rubric_answers["Q6b"] == "no"
        assert r.rubric_answers["Q1"] == "yes"

    def test_unrecognized_rubric_answer_is_preserved_not_rewritten(
        self, mcal_artifacts, doc
    ):
        """Rewriting 'partially' to 'no' would invent a defect."""
        r = critique(
            "themes", doc, recorder(response(rubric_answers={"Q2": "partially"}))
        )
        assert r.rubric_answers["Q2"] == "partially"
        assert any("rubric_answer_unrecognized" in v for v in r.schema_violations)

    def test_verdict_before_evidence_quote_is_a_recorded_violation(
        self, mcal_artifacts, doc
    ):
        """
        MCAL_PLAN 3.11 says "schema field order enforced"; json.loads preserves
        object order so the violation is observable. Recorded, not fatal -- the
        enforcement that matters is the quote check.
        """
        out_of_order = {
            "verdict": "PASS",
            "evidence_quote": VERIFIABLE_QUOTE,
            "rubric_answers": {"Q1": "yes"},
            "failure_tag": None,
            "note": None,
        }
        r = critique("themes", doc, recorder(out_of_order))
        assert C.NOTE_SCHEMA_ORDER in r.schema_violations
        assert r.verdict == "PASS"

    def test_correct_order_is_not_flagged(self, mcal_artifacts, doc):
        r = critique("themes", doc, recorder(response()))
        assert C.NOTE_SCHEMA_ORDER not in r.schema_violations

    def test_non_object_response_is_human_review(self, mcal_artifacts, doc):
        r = critique("themes", doc, recorder(["not", "an", "object"]))
        assert r.verdict == "HUMAN_REVIEW"
        assert any("response_not_an_object" in v for v in r.schema_violations)


# --- The quote-verify override (the load-bearing layer) ----------------------


class TestQuoteVerifyOverride:
    def test_fabricated_quote_flips_pass_to_human_review(self, mcal_artifacts, doc):
        """
        MCAL_PLAN 4 Q2 layer 3. The fabricated clause is the Lincoln Hwy failure
        from MCAL_PLAN 1(4), which survived a prompt that already forbade
        fabrication -- prompt words alone are known-insufficient here.
        """
        r = critique(
            "summary.environmental_impact", doc,
            recorder(response(evidence_quote=FABRICATED_QUOTE, verdict="PASS")),
        )
        assert r.verdict == "HUMAN_REVIEW"
        assert C.NOTE_EVIDENCE_UNVERIFIABLE in r.overrides
        assert C.NOTE_EVIDENCE_UNVERIFIABLE in r.note
        assert r.quote_check["verified"] == "no"

    def test_the_original_verdict_is_preserved_for_audit(self, mcal_artifacts, doc):
        """
        MCAL_PLAN 6 tracks the Critic's evidence_quote verifiable rate as a
        diagnostic; that number is unrecoverable if the override overwrites its
        own input.
        """
        r = critique(
            "summary.environmental_impact", doc,
            recorder(response(evidence_quote=FABRICATED_QUOTE, verdict="PASS")),
        )
        assert r.verdict_before_override == "PASS"
        assert r.verdict == "HUMAN_REVIEW"

    def test_verified_quote_leaves_the_verdict_alone(self, mcal_artifacts, doc):
        r = critique("summary.environmental_impact", doc, recorder(response()))
        assert r.verdict == "PASS"
        assert r.overrides == []
        assert r.verdict_before_override is None
        assert r.quote_check["verified"] == "yes"

    def test_quote_from_an_uncited_page_does_not_verify(self, mcal_artifacts, doc):
        """
        `search_whole_doc_if_no_pages=False`: the question is whether the quote is
        on the pages the extraction CITED. Finding it elsewhere confirms a
        mis-citation; reporting that as support would be worse than useless.
        """
        m2 = build_m2()
        m2["summary"]["environmental_impact"]["evidence"] = [
            {"quote": "x", "source_pages": ["1"], "quote_verified": True}
        ]
        r = critique(
            "summary.environmental_impact", doc,
            recorder(response(evidence_quote="Chapter 3 Alternatives considered")),
            m2=m2,
        )
        assert r.source_pages == [1]
        assert r.verdict == "HUMAN_REVIEW"
        assert C.NOTE_EVIDENCE_UNVERIFIABLE in r.overrides

    def test_mixed_verdict_also_overrides(self, mcal_artifacts, doc_factory):
        """
        A half-matching quote is what paraphrase-from-memory looks like, and a
        judge that cannot copy 20 characters has not shown it read the page.
        """
        d = doc_factory(
            "one two three",
            "The reconstruction would affect sagebrush habitat and displace "
            "residents of three census tracts near the alignment.",
            "filler",
        )
        partial = (
            "The reconstruction would affect coastal wetland habitat and displace "
            "shorebirds of three colonies near the estuary."
        )
        m2 = build_m2()
        m2["summary"]["overview"]["evidence"] = [
            {"quote": "x", "source_pages": ["2"], "quote_verified": True}
        ]
        r = critique(
            "summary.overview", d,
            recorder(response(evidence_quote=partial)), m2=m2,
        )
        assert r.quote_check["verified"] in ("mixed", "no")
        assert r.verdict == "HUMAN_REVIEW"

    def test_pass_with_no_quote_at_all_is_overridden(self, mcal_artifacts, doc):
        """A PASS claims support exists; failing to produce it is the failure."""
        r = critique("themes", doc, recorder(response(evidence_quote=None)))
        assert r.verdict == "HUMAN_REVIEW"
        assert r.quote_check["reason"] == "no_evidence_quote_returned"

    def test_null_quote_with_re_extract_is_the_prescribed_response(
        self, mcal_artifacts, doc
    ):
        """
        MCAL_PLAN 3.5 prescribes `evidence_quote = null` + `RE_EXTRACT` when no
        supporting substring exists. Overriding that to HUMAN_REVIEW would
        suppress the automated retry (7 Q8) and send a mechanically fixable field
        to a human.
        """
        r = critique(
            "summary.public_response", doc,
            recorder(
                response(
                    evidence_quote=None, verdict="RE_EXTRACT",
                    failure_tag="T01_missing_citation",
                )
            ),
        )
        assert r.verdict == "RE_EXTRACT"
        assert C.NOTE_EVIDENCE_UNVERIFIABLE not in r.overrides
        assert r.quote_check is None

    def test_override_fires_for_every_field_type(self, mcal_artifacts, doc):
        """Not just the summary fields -- the layer is unconditional."""
        results = C.run_critic(
            doc, build_m1(), build_m2(), stage="v1",
            call=recorder(response(evidence_quote=FABRICATED_QUOTE)),
            apply_cascade=False,
        )
        # summary_of_interest is exempt (legitimately empty), the rest are not.
        checked = {f: r for f, r in results.items() if f != settings.SUMMARY_OF_INTEREST}
        assert all(r.verdict == "HUMAN_REVIEW" for r in checked.values())
        assert all(C.NOTE_EVIDENCE_UNVERIFIABLE in r.overrides for r in checked.values())

    def test_ocr_damaged_quote_still_verifies(self, mcal_artifacts, doc_factory):
        """
        OCR damage is not disagreement (MCAL_PLAN 3.2). `quote_check` folds
        rn/m, l/1/I, O/0, S/5; re-implementing the match here instead would have
        rejected this.
        """
        d = doc_factory(
            "filler",
            "The proposed improvement extends along Lincoln Highway through "
            "Cook County to the Chicago Road centerline.",
            "filler",
        )
        m2 = build_m2()
        m2["themes"]["evidence"] = [
            {"quote": "x", "source_pages": ["2"], "quote_verified": True}
        ]
        damaged = "The proposed irnprovernent extends along Linco1n Highway through Cook County"
        r = critique("themes", d, recorder(response(evidence_quote=damaged)), m2=m2)
        assert r.quote_check["verified"] == "yes"
        assert r.verdict == "PASS"


# --- Private-individual policy ----------------------------------------------


class TestPrivateIndividualPolicy:
    def test_legacy_kind_private_stance_forces_human_review(self, mcal_artifacts, doc):
        """
        Ported from segment_a's `_apply_private_commenter_override`. MCAL_PLAN 5
        lists removing this under deferred/skipped with "(policy call,
        permanent)", so it is unconditional and has no threshold.
        """
        m2 = build_m2()
        m2["key_people"]["value"]["public_commenters"] = [
            {
                "name": "Johnson",
                "kind": "private",
                "stance": "oppose",
                "capacity": {"capacity": "private", "basis": "test"},
                "evidence": [{"quote": "Residents of three census tracts objected",
                              "source_pages": ["5"]}],
            }
        ]
        r = critique("key_people", doc, recorder(response()), m2=m2)
        assert r.verdict == "HUMAN_REVIEW"
        assert C.NOTE_PRIVATE_INDIVIDUAL in r.overrides
        assert r.verdict_before_override == "PASS"

    def test_policy_route_carries_no_failure_tag(self, mcal_artifacts, doc):
        """
        `_base.md` decision rule 1: a policy review is not evidence of a defect,
        and a tag on it would pollute the distribution the next taxonomy revision
        is fitted on.
        """
        m2 = build_m2()
        m2["key_people"]["value"]["public_commenters"] = [
            {"name": "Johnson", "stance": "oppose",
             "capacity": {"capacity": "private", "basis": "test"}, "evidence": []}
        ]
        r = critique(
            "key_people", doc,
            recorder(response(failure_tag="T05_commenter_mislabeled_as_cooperator")),
            m2=m2,
        )
        assert r.failure_tag is None

    def test_non_private_capacity_does_not_trigger(self, mcal_artifacts, doc):
        """Mayor Alice Chen is identified with an elected office on page 5."""
        r = critique("key_people", doc, recorder(response()), m2=build_m2())
        assert r.verdict == "PASS"
        assert C.NOTE_PRIVATE_INDIVIDUAL not in r.overrides

    def test_ambiguous_capacity_also_routes_to_human_review(self, mcal_artifacts, doc):
        """
        MCAL_PLAN 3.5: "If the passage is ambiguous about which capacity is being
        expressed, route to HUMAN_REVIEW regardless of Critic verdict."
        """
        m2 = build_m2()
        m2["key_people"]["value"]["public_commenters"] = [
            {
                "name": "Alice Chen",
                "stance": "support",
                "capacity": {
                    "capacity": "ambiguous",
                    "basis": "dual_capacity_cues_in_cited_passage",
                },
                "evidence": [],
            }
        ]
        r = critique("key_people", doc, recorder(response()), m2=m2)
        assert r.verdict == "HUMAN_REVIEW"
        assert C.NOTE_AMBIGUOUS_CAPACITY in r.overrides

    def test_pipeline_human_review_flag_is_honoured(self, mcal_artifacts, doc):
        m2 = build_m2()
        m2["key_people"]["value"]["public_commenters"] = [
            {"name": "Someone", "stance": "oppose", "human_review": True, "evidence": []}
        ]
        r = critique("key_people", doc, recorder(response()), m2=m2)
        assert r.verdict == "HUMAN_REVIEW"

    def test_legacy_extraction_is_classified_here(self, mcal_artifacts, doc):
        """
        segment_a M2 output carries no `capacity` block, so `classify_capacity` is
        run against the cited passage -- otherwise the policy would silently not
        apply to documents extracted before key_people_pipeline.py existed.
        """
        m2 = build_m2()
        m2["key_people"]["value"]["public_commenters"] = [
            {
                "name": "Mayor Alice Chen",
                "kind": "official",
                "stance": "support",
                "evidence": [{"quote": "Mayor Alice Chen of Chicago Heights supported",
                              "source_pages": ["5"]}],
            }
        ]
        r = critique("key_people", doc, recorder(response()), m2=m2)
        finding = r.capacity_findings[0]
        assert finding["source"] == "classified"
        assert finding["capacity"] == "non_private"
        assert r.verdict == "PASS"

    def test_stanceless_entities_are_not_policy_triggers(self, mcal_artifacts, doc):
        """MCAL_PLAN 3.5 Q5 is about a STANCE, not about being named."""
        m2 = build_m2()
        m2["key_people"]["value"]["public_commenters"] = [
            {
                "name": "Johnson",
                "kind": "private",
                "stance": None,
                "evidence": [{"quote": "Residents of three census tracts objected",
                              "source_pages": ["5"]}],
            }
        ]
        r = critique("key_people", doc, recorder(response()), m2=m2)
        assert r.capacity_findings == []
        assert r.verdict == "PASS"

    def test_rubric_q5_generalizes_the_policy_beyond_key_people(
        self, mcal_artifacts, doc
    ):
        """
        segment_a only checked `key_people`. `_base.md` asks Q5 of every field
        because `summary.public_response` can attribute a stance to a private
        individual just as `key_people` can.
        """
        raw = response(
            rubric_answers={
                "Q1": "yes", "Q2": "yes", "Q3": "yes", "Q4": "yes",
                "Q5": "yes", "Q6a": "yes", "Q6b": "yes",
            },
            verdict="PASS",
        )
        r = critique("summary.public_response", doc, recorder(raw))
        assert r.verdict == "HUMAN_REVIEW"
        assert C.NOTE_PRIVATE_INDIVIDUAL in r.overrides

    def test_capacity_findings_are_shown_to_the_critic(self, mcal_artifacts, doc):
        call = recorder(response())
        critique("key_people", doc, call, m2=build_m2())
        assert "DETERMINISTIC CAPACITY FINDINGS" in call.calls[0]["user"]
        assert "Alice Chen" in call.calls[0]["user"]


# --- Dependent-field cascade -------------------------------------------------


class TestDependentCascade:
    @pytest.mark.parametrize("year_verdict", ["RE_EXTRACT", "HUMAN_REVIEW"])
    def test_untrustworthy_year_cascades_to_key_people(
        self, mcal_artifacts, doc, year_verdict
    ):
        """
        MCAL_PLAN 3.10 step 2: if `year` is not trustworthy, key_people cannot be
        era-gated -- the pre-1978 branch would turn on a year we have just said we
        do not believe.
        """
        call = recorder(
            by_field={
                "year": response(
                    verdict=year_verdict, failure_tag="T11_year_ocr_error",
                    evidence_quote=(None if year_verdict == "RE_EXTRACT"
                                    else VERIFIABLE_QUOTE),
                ),
            }
        )
        results = C.run_critic(doc, build_m1(), build_m2(), stage="v1", call=call)
        assert results["year"].verdict == year_verdict
        kp = results["key_people"]
        assert kp.verdict == "HUMAN_REVIEW"
        assert C.NOTE_DEPENDENT_CASCADE in kp.overrides
        assert f"{C.NOTE_DEPENDENT_CASCADE}:year={year_verdict}" in kp.note

    @pytest.mark.parametrize("year_verdict", ["PASS", "PASS_WITH_NOTE"])
    def test_trustworthy_year_does_not_cascade(self, mcal_artifacts, doc, year_verdict):
        call = recorder(by_field={"year": response(verdict=year_verdict)})
        results = C.run_critic(doc, build_m1(), build_m2(), stage="v1", call=call)
        assert results["year"].verdict == year_verdict
        assert results["key_people"].verdict == "PASS"
        assert C.NOTE_DEPENDENT_CASCADE not in results["key_people"].overrides

    def test_cascade_preserves_the_dependents_own_verdict_for_audit(
        self, mcal_artifacts, doc
    ):
        call = recorder(by_field={"year": response(verdict="HUMAN_REVIEW")})
        results = C.run_critic(doc, build_m1(), build_m2(), stage="v1", call=call)
        assert results["key_people"].verdict_before_override == "PASS"

    def test_cascade_is_recorded_even_when_the_target_is_already_gated(
        self, mcal_artifacts, doc
    ):
        """
        The more specific reason stays first in `overrides` so gate.py can still
        prioritize it, but the cascade is recorded so it is not invisible.
        """
        call = recorder(
            by_field={
                "year": response(verdict="HUMAN_REVIEW"),
                "key_people": response(evidence_quote=FABRICATED_QUOTE),
            }
        )
        results = C.run_critic(doc, build_m1(), build_m2(), stage="v1", call=call)
        kp = results["key_people"]
        assert kp.overrides[0] == C.NOTE_EVIDENCE_UNVERIFIABLE
        assert C.NOTE_DEPENDENT_CASCADE in kp.overrides

    def test_cascade_runs_after_every_field_is_judged(self, mcal_artifacts, doc):
        """
        Fields are judged in arbitrary order, so a cascade applied during judging
        would depend on iteration order. Judging key_people BEFORE year must give
        the same answer.
        """
        call = recorder(by_field={"year": response(verdict="HUMAN_REVIEW")})
        results = C.run_critic(
            doc, build_m1(), build_m2(), stage="v1", call=call,
            fields=("key_people", "year"),
        )
        assert results["key_people"].verdict == "HUMAN_REVIEW"
        assert C.NOTE_DEPENDENT_CASCADE in results["key_people"].overrides

    def test_config_supplies_the_dependency_map(self, mcal_artifacts, doc):
        """A recalibration can add a dependency without a code change."""
        results = {
            "eis_type": C.CriticResult(field="eis_type", bucket="M1", verdict="RE_EXTRACT"),
            "themes": C.CriticResult(
                field="themes", bucket="alternatives+themes", verdict="PASS"
            ),
        }
        C.apply_dependent_cascade(results, {"dependent_fields": {"eis_type": ["themes"]}})
        assert results["themes"].verdict == "HUMAN_REVIEW"

    def test_cascade_can_be_disabled(self, mcal_artifacts, doc):
        call = recorder(by_field={"year": response(verdict="HUMAN_REVIEW")})
        results = C.run_critic(
            doc, build_m1(), build_m2(), stage="v1", call=call, apply_cascade=False
        )
        assert results["key_people"].verdict == "PASS"


# --- summary_of_interest ----------------------------------------------------


class TestSummaryOfInterest:
    def test_empty_list_can_pass(self, mcal_artifacts, doc):
        """
        MCAL_PLAN 3.15 rule 2: an empty summary_of_interest is a CORRECT and
        expected output for an unremarkable document. It is the load-bearing
        anti-hallucination provision for the field, so nothing in this module may
        coerce empty -> failure.
        """
        r = critique(
            settings.SUMMARY_OF_INTEREST, doc,
            recorder(
                response(
                    evidence_quote=None,
                    verdict="PASS",
                    rubric_answers={"Q7c": "yes"},
                )
            ),
        )
        assert r.empty_but_valid is True
        assert r.extraction_missing is False
        assert r.verdict == "PASS"
        assert r.overrides == []

    def test_empty_list_skips_the_quote_check_rather_than_failing_it(
        self, mcal_artifacts, doc
    ):
        r = critique(settings.SUMMARY_OF_INTEREST, doc, recorder(response()))
        assert r.quote_check is None
        assert "quote_verify_skipped" in r.note
        assert r.verdict == "PASS"

    def test_the_critic_is_told_that_emptiness_is_legitimate(self, mcal_artifacts, doc):
        call = recorder(response())
        critique(settings.SUMMARY_OF_INTEREST, doc, call)
        user = call.calls[0]["user"]
        assert "NOTE ON EMPTINESS" in user
        assert "MCAL_PLAN 3.15 rule 2" in user

    def test_missing_field_is_distinguishable_from_empty(self, mcal_artifacts, doc):
        """
        `[]` (routine document) vs `null` (generation failure) -- MCAL_PLAN 3.12
        requires these to stay distinct all the way into the manifest.
        """
        m2 = build_m2()
        del m2["summary_of_interest"]
        r = critique(settings.SUMMARY_OF_INTEREST, doc, recorder(response()), m2=m2)
        assert r.extraction_missing is True
        assert r.empty_but_valid is False
        assert r.verdict == "HUMAN_REVIEW"
        assert C.NOTE_EXTRACTION_MISSING in r.overrides

    def test_non_empty_soi_is_quote_checked_normally(self, mcal_artifacts, doc):
        m2 = build_m2(
            summary_of_interest=[
                {
                    "claim": "1,200 acres of sagebrush habitat affected",
                    "salience_criterion": "large_magnitude",
                    "page": 4,
                    "evidence_quote": VERIFIABLE_QUOTE,
                    "why_notable": "Largest quantified impact in the document.",
                    "evidence": [{"quote": VERIFIABLE_QUOTE, "source_pages": ["4"]}],
                }
            ]
        )
        r = critique(settings.SUMMARY_OF_INTEREST, doc, recorder(response()), m2=m2)
        assert r.empty_but_valid is False
        assert r.quote_check["verified"] == "yes"

    def test_empty_alternatives_is_NOT_treated_as_legitimately_empty(
        self, mcal_artifacts, doc
    ):
        """
        MCAL_PLAN 1(8): the fix for Buffalo's empty alternatives is a
        `{status: "alternatives_chapter_not_found"}` object, explicitly "never
        return empty silently". So emptiness is only excused for
        summary_of_interest.
        """
        m2 = build_m2(alternatives={"value": [], "confidence": "low"})
        r = critique("alternatives", doc, recorder(response()), m2=m2)
        assert r.empty_but_valid is False


# --- Failure handling -------------------------------------------------------


class TestNeverCrashes:
    def test_llm_exception_becomes_human_review(self, mcal_artifacts, doc):
        r = critique("themes", doc, recorder(exc=RuntimeError("bedrock 500")))
        assert r.verdict == "HUMAN_REVIEW"
        assert "bedrock 500" in r.note
        assert any(o.startswith(C.NOTE_LLM_FAILED) for o in r.overrides)
        # No model verdict existed, so there is no "before" verdict to claim.
        assert r.verdict_before_override is None

    def test_one_failing_field_does_not_lose_the_document(self, mcal_artifacts, doc):
        calls = {"n": 0}

        def flaky(model, system, user, *, max_tokens, temperature):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("throttled")
            return response()

        results = C.run_critic(doc, build_m1(), build_m2(), stage="v1", call=flaky)
        assert len(results) == 15
        assert sum(1 for r in results.values() if r.verdict == "HUMAN_REVIEW") >= 1

    def test_missing_extraction_skips_the_call_entirely(self, mcal_artifacts, doc):
        """Paying a judge to be told there is nothing to judge is waste."""
        call = recorder(response())
        r = critique("themes", doc, call, m2=build_m2(themes=None))
        assert r.verdict == "HUMAN_REVIEW"
        assert C.NOTE_EXTRACTION_MISSING in r.overrides
        assert call.calls == []

    def test_empty_m1_and_m2_still_yields_15_results(self, mcal_artifacts, doc):
        results = C.run_critic(doc, {}, {}, stage="v1", call=recorder(response()))
        assert len(results) == 15
        assert all(r.verdict == "HUMAN_REVIEW" for r in results.values())
        assert all(r.extraction_missing for r in results.values())

    def test_document_with_no_pages_does_not_crash(self, mcal_artifacts, doc_factory):
        empty = doc_factory("", doc_id="blank")
        results = C.run_critic(
            empty, build_m1(), build_m2(), stage="v1", call=recorder(response())
        )
        assert len(results) == 15


# --- Serialization + diagnostics --------------------------------------------


class TestResultContract:
    def test_result_carries_every_documented_key(self, mcal_artifacts, doc):
        r = critique("summary.project_description", doc, recorder(response())).to_dict()
        for key in (
            "evidence_quote", "rubric_answers", "verdict", "verdict_before_override",
            "failure_tag", "note", "judge_model", "source_pages", "quote_check",
        ):
            assert key in r

    def test_evidence_quote_precedes_verdict_in_our_own_output(
        self, mcal_artifacts, doc
    ):
        """The evidence-first discipline applies to what we emit, too."""
        keys = list(critique("themes", doc, recorder(response())).to_dict())
        assert keys.index("evidence_quote") < keys.index("verdict")

    def test_results_are_json_serializable(self, mcal_artifacts, doc):
        results = C.run_critic(doc, build_m1(), build_m2(), stage="v1",
                               call=recorder(response()))
        json.dumps(C.as_dict(results), allow_nan=False)

    def test_diagnostics_report_the_verifiable_rate(self, mcal_artifacts, doc):
        call = recorder(by_field={"themes": response(evidence_quote=FABRICATED_QUOTE)})
        results = C.run_critic(doc, build_m1(), build_m2(), stage="v1", call=call)
        diag = C.critic_diagnostics(results)
        assert diag["n_fields"] == 15
        assert 0.0 < diag["evidence_quote_verifiable_rate"] < 1.0
        assert diag["overrides"][C.NOTE_EVIDENCE_UNVERIFIABLE] == 1

    def test_diagnostics_count_off_vocabulary_tags(self, mcal_artifacts, doc):
        call = recorder(
            by_field={"themes": response(verdict="RE_EXTRACT", failure_tag="T77_nonsense")}
        )
        results = C.run_critic(doc, build_m1(), build_m2(), stage="v1", call=call)
        assert C.critic_diagnostics(results)["n_off_vocabulary_tags"] == 1

    def test_default_path_goes_through_llm_call_json_with_usage(
        self, mcal_artifacts, doc, monkeypatch
    ):
        """
        Usage-tracking wiring (MCAL_PLAN 2, Cost Summary): the default call is
        `llm.call_json_with_usage`, which records into the process-wide collector,
        so a caller that wraps a document in `llm.start_usage_session()` gets the
        Critic slice for free.
        """
        seen: list[dict] = []

        def stub(model, system, user, *, max_tokens, temperature):
            seen.append({"model": model})
            return response(), {"model": model, "input_tokens": 7, "output_tokens": 3}

        monkeypatch.setattr(C, "call_json_with_usage", stub)
        r = C.critique_field("themes", doc, build_m1(), build_m2(), stage="v1")
        assert seen and seen[0]["model"] == settings.MODEL_SONNET
        assert r.usage["output_tokens"] == 3

    def test_usage_is_captured_when_the_caller_returns_it(self, mcal_artifacts, doc):
        def with_usage(model, system, user, *, max_tokens, temperature):
            return response(), {"model": model, "input_tokens": 10, "output_tokens": 2}

        r = critique("themes", doc, with_usage)
        assert r.usage["input_tokens"] == 10


# --- No network -------------------------------------------------------------


class TestNoNetwork:
    def test_injected_call_is_the_only_llm_entry_point(
        self, mcal_artifacts, doc, monkeypatch
    ):
        import llm

        def boom(*a, **kw):
            raise AssertionError("real LLM client reached in a test")

        monkeypatch.setattr(llm, "call_json_with_usage", boom)
        monkeypatch.setattr(llm, "call_with_usage", boom)
        monkeypatch.setattr(C, "call_json_with_usage", boom)
        results = C.run_critic(doc, build_m1(), build_m2(), stage="v1",
                               call=recorder(response()))
        assert len(results) == 15

    def test_judging_is_deterministic_at_temperature_zero(self, mcal_artifacts, doc):
        """
        MCAL_PLAN 7 Q8's "+0.2" is for the EXTRACTOR retry. A non-deterministic
        judge would make s_critic -- and every threshold fitted on it --
        irreproducible.
        """
        call = recorder(response())
        critique("themes", doc, call)
        assert call.calls[0]["temperature"] == 0.0
