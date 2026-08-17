"""
M-Cal build orchestrator (MCAL_PLAN 3.7).

    Seed build:      python -m mcal.build --stage v1
    Recalibration:   python -m mcal.build --stage v2 --prior v1
    Ratify a draft:  python -m mcal.build --stage v1 --ratify
    Status:          python -m mcal.build --status

Everything is written to `artifacts/v(N)-draft/` first. Ratifying the taxonomy
promotes the whole draft directory to `artifacts/v(N)/`, which is the only thing
Segment B ever reads. That two-phase flow exists because MCAL_PLAN 6 makes
human ratification of the taxonomy a seed-v1 acceptance item, and because a
half-written artifact set that Segment B could pick up would be worse than none.

Two prechecks gate the run:

1. **M2 prompt version.** Build item #4 amends the M2 summary prompts, so every
   graded doc's M2 output must have been produced under the amended prompts
   before calibration touches it. Otherwise tau is fitted to prose Segment B
   will never emit, and the frozen thresholds encode an untested distribution
   shift. This precheck HALTS -- there is no safe way to continue.

2. **Geocoder assets.** PAD-US / GNIS / MAPBOX_TOKEN are user-supplied. Missing
   assets do NOT halt: the build continues in reduced mode, marks
   `confidence_config.geocoder_stack = "reduced"`, and forces the location
   bucket to `gate_all_to_human`. That lets the whole M-Cal loop be exercised
   before geocoder setup is finished, at the cost of routing every location
   field to a human (MCAL_PLAN 3.9a).

Stage ordering inside a build follows MCAL_PLAN 3.7: taxonomy (induction or
add-only carry-forward) -> atomic verification -> confidence + CP calibration ->
prompt build -> artifact emission -> report.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import active_select, confidence, critic_prompt, grades, settings, taxonomy

log = logging.getLogger("mcal.build")


# --- Precheck results -------------------------------------------------------


class PrecheckFailure(RuntimeError):
    """A halting precheck failed. Message is user-facing and actionable."""


@dataclass
class Precheck:
    ok: bool
    name: str
    detail: str = ""
    checklist: list[str] = dc_field(default_factory=list)


def precheck_m2_prompt_version() -> Precheck:
    """
    MCAL_PLAN 3.7 step 0: verify M2 output was produced under the amended prompts.

    Halting. `segment_a/run.py` stamps the marker only when every doc with M2
    output on disk was regenerated in the same run, so a present marker really
    does mean the whole calibration set is prose-consistent.
    """
    marker = settings.M2_PROMPT_VERSION_MARKER
    required = settings.M2_PROMPT_VERSION_REQUIRED

    if not marker.exists():
        return Precheck(
            ok=False,
            name="m2_prompt_version",
            detail=f"marker missing: {marker}",
            checklist=[
                "Build item #4 amends the M2 summary prompts (plain language +",
                "concreteness) and item #5 adds summary_of_interest. Calibration",
                "must run on the prose Segment B will actually ship, so every",
                "graded doc needs its M2 output regenerated first.",
                "",
                "1. Archive the current output so the amendment is reversible:",
                f"     cp -r {settings.M2_DIR} {settings.M2_PRE_AMENDMENT_DIR}",
                "2. Re-run M2 over the graded docs:",
                "     cd May25/segment_a && python run.py process --force",
                "   (or --doc <doc_id> per doc; the marker is only stamped once",
                "    every doc with M2 output on disk has been regenerated)",
                "3. Re-run this build.",
                "",
                "Cost note: observed ~$5.62/doc, ~91% Opus. Items #4/#5 add one",
                "Opus reduce call per doc on top of that.",
            ],
        )

    got = marker.read_text(encoding="utf-8").strip()
    if got != required:
        return Precheck(
            ok=False,
            name="m2_prompt_version",
            detail=f"marker is {got!r}, need {required!r}",
            checklist=[
                f"{marker} says {got!r} but this build requires {required!r}.",
                "The M2 prompt templates changed since that output was written.",
                "Re-run: cd May25/segment_a && python run.py process --force",
            ],
        )

    return Precheck(ok=True, name="m2_prompt_version", detail=got)


def precheck_geocoder() -> Precheck:
    """
    MCAL_PLAN 3.7 / 3.9a: check the three user-supplied geocoder assets.

    Non-halting by design. Returns ok=False to signal reduced mode, not failure.
    """
    result = settings.geocoder_precheck()
    return Precheck(
        ok=result["stack"] == "full",
        name="geocoder_assets",
        detail=result["stack"],
        checklist=result["checklist"],
    )


def precheck_grades(grade_set: grades.GradeSet) -> Precheck:
    """Non-halting: report what the calibration set actually contains."""
    if not grade_set.items:
        return Precheck(
            ok=False,
            name="grades",
            detail="no grades found",
            checklist=[
                "No human grades were loaded. Expected either:",
                f"  - {settings.EVALUATION_CSV} (transposed seed-v1 source), or",
                f"  - filled `your_grade` cells in {settings.GRADING_SHEETS_DIR}/*.csv",
                "Calibration cannot proceed without labels.",
            ],
        )
    return Precheck(
        ok=True,
        name="grades",
        detail=(
            f"{grade_set.n_docs} docs, {len(grade_set.items)} items, "
            f"{sum(1 for i in grade_set.items if not i.correct)} wrong"
        ),
        checklist=list(grade_set.warnings),
    )


# --- Scoring the graded corpus ---------------------------------------------


def score_graded_corpus(
    grade_set: grades.GradeSet,
    *,
    stage: str,
    atomic_citation_rates: Optional[dict[tuple[str, str], float]] = None,
) -> tuple[list[confidence.ScoredItem], dict]:
    """
    Build `ScoredItem`s for every graded (doc, field) from artifacts on disk.

    Signals come from the Segment A run: `s_critic` from `output/critic/`,
    `s_quote` from live `quote_check` over `output/m2/` evidence, and
    `s_citation` from atomic verification when available.

    A caveat that matters for interpreting tau: Segment A's Critic is COARSER
    than the buckets. It emits one `summary` verdict covering all six subfields,
    one `alternatives` verdict, and one `key_people` verdict. Until Segment B's
    per-field Critic has been run over the graded docs, every subfield in a
    document inherits the same `s_critic`, which compresses within-bucket score
    variance and makes tau look tighter than it will be in production. Recorded
    in the diagnostics as `critic_granularity`.
    """
    from pages import load_doc  # segment_a bridge

    from . import quote_check as qc

    items: list[confidence.ScoredItem] = []
    per_doc_notes: dict[str, str] = {}
    n_coarse = 0
    missing: list[str] = []

    for doc_id in grade_set.doc_ids:
        m2_path = settings.M2_DIR / f"{doc_id}.json"
        critic_path = settings.CRITIC_DIR / f"{doc_id}.json"
        if not m2_path.exists():
            missing.append(f"{doc_id}: no M2 output")
            continue

        m2 = json.loads(m2_path.read_text(encoding="utf-8"))
        critic = (
            json.loads(critic_path.read_text(encoding="utf-8"))
            if critic_path.exists()
            else {}
        )
        if not critic_path.exists():
            missing.append(f"{doc_id}: no Critic output")

        try:
            doc = load_doc(doc_id)
        except FileNotFoundError as e:
            missing.append(f"{doc_id}: {e}")
            continue

        for g in grade_set.for_doc(doc_id):
            field = g.field
            entry, evidence = _extract_entry_and_evidence(m2, field)

            # Segment A critic keys: coarse for summary.*/alternatives/key_people.
            ckey = "summary" if field.startswith("summary.") else field
            centry = critic.get(ckey) or critic.get(field) or {}
            verdict = centry.get("verdict") or "HUMAN_REVIEW"
            if field.startswith("summary.") and "summary" in critic:
                n_coarse += 1

            quote_verdict = None
            if evidence:
                checks = [
                    qc.check_quote(
                        e.get("quote", ""), e.get("source_pages"), doc
                    )
                    for e in evidence
                    if e.get("quote")
                ]
                if checks:
                    quote_verdict = qc.aggregate_verdict(checks)

            cite_rate = None
            if atomic_citation_rates:
                cite_rate = atomic_citation_rates.get((doc_id, field))
            if cite_rate is None and evidence:
                cite_rate = sum(
                    1 for e in evidence if e.get("source_pages")
                ) / len(evidence)

            signals = confidence.compute_signals(
                field,
                quote_verdict=quote_verdict,
                critic_verdict=verdict,
                citation_rate=cite_rate,
                acronym_rate=0.0 if g.acronym_issue else 1.0,
            )
            items.append(
                confidence.ScoredItem(
                    doc_id=doc_id,
                    field=field,
                    bucket=g.bucket,
                    signals=signals,
                    composite=confidence.composite(signals),
                    y=g.y,
                    failure_tags=list(g.failure_tags),
                )
            )
        per_doc_notes[doc_id] = grade_set.doc_notes.get(doc_id, "")

    diagnostics = {
        "n_items": len(items),
        "n_docs": len({i.doc_id for i in items}),
        "missing_inputs": missing,
        "critic_granularity": (
            "coarse (Segment A): summary.* subfields share one verdict"
            if n_coarse
            else "per-field"
        ),
        "n_items_with_inherited_critic_verdict": n_coarse,
        "atomic_citation_rates_used": bool(atomic_citation_rates),
    }
    return items, diagnostics


def _extract_entry_and_evidence(m2: dict, field: str) -> tuple[object, list[dict]]:
    """Pull a field's M2 entry and its evidence list, tolerating shape variation."""
    if field == settings.SUMMARY_OF_INTEREST:
        entries = m2.get("summary_of_interest")
        if not isinstance(entries, list):
            return None, []
        ev = [e for item in entries for e in (item.get("evidence") or [])]
        return entries, ev
    if field.startswith("summary."):
        sub = field.split(".", 1)[1]
        entry = (m2.get("summary") or {}).get(sub) or {}
        return entry, list(entry.get("evidence") or [])
    entry = m2.get(field)
    if not isinstance(entry, dict):
        return entry, []
    ev = list(entry.get("evidence") or [])
    if not ev:
        ev = _collect_nested_evidence(entry.get("value"))
    return entry, ev


def _collect_nested_evidence(value) -> list[dict]:
    """Recursively harvest `evidence` lists from a nested M2 value."""
    out: list[dict] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "evidence" and isinstance(v, list):
                out.extend(e for e in v if isinstance(e, dict))
            else:
                out.extend(_collect_nested_evidence(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(_collect_nested_evidence(v))
    return out


# --- Report -----------------------------------------------------------------


def cost_summary() -> dict:
    """
    Token/cost roll-up for `calibration_report.v(N).md` (MCAL_PLAN 2).

    Reads `segment_a/output/run_summary.json`, which `run.py` overwrites each
    invocation, so this reflects the most recent M2 rerun only. Projections to
    2000 docs are linear in the observed per-doc average -- crude, but the
    go/no-go decision only needs an order of magnitude.
    """
    path = settings.SEGMENT_A_OUTPUT / "run_summary.json"
    if not path.exists():
        return {"available": False, "reason": f"{path} not found"}

    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = [r for r in payload.get("runs", []) if not r.get("error")]
    if not runs:
        return {"available": False, "reason": "no successful runs recorded"}

    by_model: dict[str, dict] = {}
    total_cost = 0.0
    for r in runs:
        agg = (r.get("usage") or {}).get("by_model") or []
        for m in agg:
            slot = by_model.setdefault(
                m["model"],
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            slot["calls"] += m.get("calls", 0)
            slot["input_tokens"] += m.get("input_tokens", 0)
            slot["output_tokens"] += m.get("output_tokens", 0)
            slot["cost_usd"] += m.get("cost_usd", 0.0) or 0.0
        total_cost += ((r.get("usage") or {}).get("total") or {}).get("cost_usd", 0) or 0

    n = len(runs)
    per_doc = total_cost / n if n else 0.0
    for slot in by_model.values():
        slot["cost_usd"] = round(slot["cost_usd"], 4)

    return {
        "available": True,
        "n_docs_in_run": n,
        "by_model": by_model,
        "total_cost_usd": round(total_cost, 4),
        "per_doc_avg_usd": round(per_doc, 4),
        "projected_2000_docs_usd": round(per_doc * 2000, 2),
        "caveat": (
            "Linear projection from the most recent run.py invocation. Excludes "
            "the M-Cal Opus atomic-verify pass and Segment B's per-field Critic "
            "split, both of which add Opus calls not present in this run."
        ),
    }


def write_report(
    stage: str,
    *,
    draft: bool,
    grade_set: grades.GradeSet,
    thresholds: dict[str, confidence.BucketThreshold],
    items: list[confidence.ScoredItem],
    tax_diag: dict,
    prompt_diag: dict,
    score_diag: dict,
    prechecks: list[Precheck],
    geocoder_stack: str,
) -> Path:
    """Human-readable roll-up (MCAL_PLAN 2, `calibration_report.v(N).md`)."""
    s = settings.normalize_stage(stage)
    summary = confidence.summarize(thresholds, items)
    sim = confidence.simulate_gate(items, thresholds)
    cov = confidence.empirical_coverage(items, thresholds)
    weights = confidence.validate_weights(items)
    costs = cost_summary()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    L: list[str] = []
    A = L.append

    A(f"# M-Cal calibration report — stage `{s}`")
    A("")
    A(f"Generated {now}. Draft: `{draft}`.")
    A("")

    A("## Prechecks")
    A("")
    A("| check | status | detail |")
    A("|---|---|---|")
    for p in prechecks:
        A(f"| `{p.name}` | {'PASS' if p.ok else 'ATTENTION'} | {p.detail} |")
    A("")
    for p in prechecks:
        if p.checklist:
            A(f"**`{p.name}` notes**")
            A("")
            for line in p.checklist:
                A(f"> {line}" if line else ">")
            A("")

    A("## Calibration set")
    A("")
    A(f"- Documents graded: **{grade_set.n_docs}**")
    A(f"- Graded items: **{len(grade_set.items)}**")
    A(f"- Wrong items: **{sum(1 for i in grade_set.items if not i.correct)}**")
    A(f"- Grade granularity: {sorted({i.granularity for i in grade_set.items})}")
    A(f"- Grade sources: {sorted({i.source for i in grade_set.items})}")
    A(f"- Critic granularity: {score_diag.get('critic_granularity')}")
    A("")
    if grade_set.warnings:
        A("**Warnings**")
        A("")
        for w in grade_set.warnings:
            A(f"- {w}")
        A("")

    A("## Thresholds")
    A("")
    A("Accept in Segment B iff `composite > tau_deployed`.")
    A("")
    A("| bucket | N_wrong_docs | tau_raw | curation_slack | tau_deployed | flags |")
    A("|---|---|---|---|---|---|")
    for b in settings.BUCKET_ORDER:
        t = thresholds[b]
        flags = []
        if t.gate_all_to_human:
            flags.append("**gate_all_to_human**")
        if t.degenerate_severe:
            flags.append("degenerate_severe")
        elif t.degenerate:
            flags.append("degenerate")
        if t.saturated:
            flags.append("saturated")
        tr = "—" if t.tau_raw is None else f"{t.tau_raw:.4f}"
        td = "∞" if t.gate_all_to_human else f"{t.tau_deployed:.4f}"
        A(
            f"| `{b}` | {t.n_wrong_docs} | {tr} | {t.curation_slack:.4f} | "
            f"{td} | {', '.join(flags) or '—'} |"
        )
    A("")

    A("### Leave-one-doc-out curation slack")
    A("")
    A(
        "MCAL_PLAN 3.3 uses `max(delta)` rather than the 95th percentile, "
        "because at these sample sizes the percentile is dominated by the "
        "discreteness of the empirical quantile. Full distributions:"
    )
    A("")
    for b in settings.BUCKET_ORDER:
        t = thresholds[b]
        if t.loo_deltas:
            ds = ", ".join(f"{d:.4f}" for d in t.loo_deltas)
            A(f"- `{b}`: [{ds}] → max = {t.curation_slack:.4f}")
        else:
            A(f"- `{b}`: not computable (n < 2)")
    A("")

    A("### Per-bucket notes")
    A("")
    for b in settings.BUCKET_ORDER:
        t = thresholds[b]
        if t.notes:
            A(f"**`{b}`**")
            A("")
            for n in t.notes:
                A(f"- {n}")
            A("")

    A("## Acceptance status")
    A("")
    A(f"- Buckets gated entirely: **{summary['n_gate_all_to_human']}/7**")
    A(f"- Degenerate buckets: **{summary['n_degenerate']}/7**")
    A(
        f"- Original (pre-SOI) buckets non-degenerate: "
        f"**{summary['n_original_non_degenerate']}/6** "
        f"({summary['original_buckets_non_degenerate']})"
    )
    A(
        f"- MCAL_PLAN 6 v2+ criterion 3 (>=4 of 6 non-degenerate): "
        f"**{'MET' if summary['meets_v2_criterion_4_of_6'] else 'NOT MET'}**"
    )
    A(
        f"- Smallest non-empty bucket N_wrong_docs: "
        f"**{summary['smallest_non_empty_n_wrong_docs']}** "
        f"(full-scale needs >= {summary['full_scale_threshold']})"
    )
    A(
        f"- Full-scale Segment B unlocked: "
        f"**{'YES' if summary['full_scale_unlocked'] else 'NO'}**"
    )
    A("")
    if summary["warnings"]:
        for w in summary["warnings"]:
            A(f"> {w}")
        A("")

    A("## Gate simulation")
    A("")
    A(f"> {sim['caveat']}")
    A("")
    A(f"Overall gate rate: **{sim['overall_gate_rate']}**")
    A("")
    A("| bucket | graded items | gate rate | caught-error rate | false-defer rate |")
    A("|---|---|---|---|---|")
    for b, v in sim["per_bucket"].items():
        A(
            f"| `{b}` | {v.get('n_graded_items')} | {v.get('gate_rate')} | "
            f"{v.get('caught_error_rate')} | {v.get('false_defer_rate')} |"
        )
    A("")

    A("## Empirical CP coverage")
    A("")
    A("Should meet target by construction; reported per MCAL_PLAN 6.")
    A("")
    A("| bucket | coverage | target | meets |")
    A("|---|---|---|---|")
    for b, v in cov.items():
        A(
            f"| `{b}` | {v.get('coverage')} | {v.get('target')} | "
            f"{v.get('meets_target')} |"
        )
    A("")

    A("## Taxonomy")
    A("")
    A(f"- Tags: **{tax_diag['n_tags']}** "
      f"(seed {tax_diag['n_seed']}, empirical {tax_diag['n_proposed']}, "
      f"induced {tax_diag['n_induced']})")
    A(f"- Induction run: **{not tax_diag['induction_skipped']}**")
    cvg = tax_diag["coverage"]
    A(f"- Tags with exemplars: **{cvg['n_with_exemplars']}/{cvg['n_tags']}**")
    A(f"- Without exemplars: {cvg['without_exemplars']}")
    A("")
    A("Observed tag counts in the graded set:")
    A("")
    for t, c in tax_diag["observed_tag_counts"].items():
        A(f"- `{t}`: {c}")
    A("")
    if tax_diag["induction_notes"]:
        A("**Induction notes for the reviewer**")
        A("")
        for n in tax_diag["induction_notes"]:
            A(f"- {n}")
        A("")

    A("## Critic prompts")
    A("")
    A(f"- Prompts written: **{prompt_diag['n_prompts']}** in `{prompt_diag['dir']}`")
    A("")
    A("| field | slots | failure examples | positive controls | tags covered |")
    A("|---|---|---|---|---|")
    for f, v in prompt_diag["few_shots"].items():
        A(
            f"| `{f}` | {v['n_slots']} | {v['n_failure_examples']} | "
            f"{v['n_positive_controls']} | {', '.join(v['tags_covered']) or '—'} |"
        )
    A("")

    A("## Weight validation (advisory, non-gating)")
    A("")
    A(f"> {weights['reason']}")
    A("")
    A(f"Resampling unit: {weights.get('resampling_unit')}. ")
    A(f"Production Kendall tau: {weights.get('production_kendall_tau')}")
    A("")
    if weights.get("candidates"):
        A("| candidate | signals | AUROC | 95% CI |")
        A("|---|---|---|---|")
        for name, v in sorted(
            weights["candidates"].items(),
            key=lambda kv: -(kv[1]["auroc"] or 0),
        ):
            ci = v["auroc_ci95"]
            A(
                f"| `{name}` | {v['n_signals_weighted']} | {v['auroc']} | "
                f"[{ci[0]}, {ci[1]}] |"
            )
        A("")
        A(f"> {weights['tiebreaker']}")
        A("")

    A("## Cost Summary")
    A("")
    if not costs.get("available"):
        A(f"Not available: {costs.get('reason')}.")
    else:
        A(f"- Documents in the most recent run: **{costs['n_docs_in_run']}**")
        A(f"- Total: **${costs['total_cost_usd']}**")
        A(f"- Per-doc average: **${costs['per_doc_avg_usd']}**")
        A(
            f"- Linear projection to 2000 docs: "
            f"**${costs['projected_2000_docs_usd']}**"
        )
        A("")
        A("| model tier | calls | input tokens | output tokens | cost |")
        A("|---|---|---|---|---|")
        for model, v in costs["by_model"].items():
            A(
                f"| `{model}` | {v['calls']} | {v['input_tokens']:,} | "
                f"{v['output_tokens']:,} | ${v['cost_usd']} |"
            )
        A("")
        A(f"> {costs['caveat']}")
    A("")

    A("## Environment")
    A("")
    A(f"- Geocoder stack: **{geocoder_stack}**")
    A(f"- M2 prompt version: `{settings.M2_PROMPT_VERSION_REQUIRED}`")
    A(f"- alpha = {settings.ALPHA}, alpha_effective(degenerate) = "
      f"{settings.ALPHA_EFFECTIVE_DEGENERATE}")
    A(f"- Signal weights: {settings.SIGNAL_WEIGHTS}")
    A("")

    path = settings.artifact_path("calibration_report.md", stage, draft=draft)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# --- Build ------------------------------------------------------------------


def run_build(
    stage: str,
    *,
    prior_stage: Optional[str] = None,
    run_induction: bool = True,
    run_atomic: bool = False,
    skip_m2_precheck: bool = False,
    induction_call=None,
) -> dict:
    """
    Full M-Cal build for `stage`. Writes to `artifacts/v(N)-draft/`.

    `run_atomic` is opt-in because atomic verification is an Opus pass over
    every summary subfield of every graded doc -- the most expensive step in the
    build. Skipping it costs the atom-level `s_citation` signal, which is
    currently the only signal capable of separating missing-citation failures;
    the report says so when it is skipped.
    """
    stage = settings.normalize_stage(stage)
    log.info(f"=== M-Cal build: stage {stage} (prior={prior_stage}) ===")

    prechecks: list[Precheck] = []

    # --- precheck 1: M2 prompt version (halting) ---
    m2p = precheck_m2_prompt_version()
    prechecks.append(m2p)
    if not m2p.ok:
        if skip_m2_precheck:
            log.warning(
                "M2 prompt-version precheck FAILED but --skip-m2-precheck was "
                "passed. Thresholds fitted now describe pre-amendment prose and "
                "MUST NOT be promoted for production use."
            )
        else:
            raise PrecheckFailure(
                "M2 prompt-version precheck failed.\n\n"
                + "\n".join(m2p.checklist)
                + "\n\n(Use --skip-m2-precheck to build anyway for a dry run; "
                "the resulting thresholds are not valid for Segment B.)"
            )

    # --- precheck 2: geocoder assets (non-halting) ---
    geo = precheck_geocoder()
    prechecks.append(geo)
    geocoder_stack = "full" if geo.ok else "reduced"
    force_gate: list[str] = []
    if not geo.ok:
        force_gate.append("location")
        log.warning(
            "Geocoder assets incomplete -> reduced mode. The location bucket "
            "will be forced to gate_all_to_human. Checklist:"
        )
        for line in geo.checklist:
            log.warning(f"  {line}")

    # --- grades ---
    grade_set = grades.load_grades()
    gp = precheck_grades(grade_set)
    prechecks.append(gp)
    if not gp.ok:
        raise PrecheckFailure("\n".join(gp.checklist))
    for w in grade_set.warnings:
        log.warning(f"grades: {w}")

    # --- taxonomy ---
    log.info("Building taxonomy...")
    tax, tax_diag = taxonomy.build(
        stage,
        grade_set,
        prior_stage=prior_stage,
        run_induction=run_induction,
        call=induction_call,
    )
    tax_path = taxonomy.save(tax, draft=True, ratified=False)
    log.info(f"  taxonomy -> {tax_path.name} ({tax_diag['n_tags']} tags)")

    # --- atomic verification (optional, expensive) ---
    atomic_rates: Optional[dict] = None
    atomic_diag: dict = {"ran": False}
    if run_atomic:
        log.info("Running atomic verification (Opus)...")
        atomic_rates, atomic_diag = _run_atomic(grade_set, stage)
    else:
        log.info("Skipping atomic verification (--atomic to enable)")

    # --- scoring + calibration ---
    log.info("Scoring graded corpus...")
    items, score_diag = score_graded_corpus(
        grade_set, stage=stage, atomic_citation_rates=atomic_rates
    )
    for m in score_diag["missing_inputs"]:
        log.warning(f"  {m}")
    log.info(f"  {score_diag['n_items']} items over {score_diag['n_docs']} docs")

    log.info("Calibrating conformal thresholds...")
    thresholds = confidence.calibrate_all(items, force_gate=force_gate)
    th_path = confidence.save_thresholds(thresholds, stage, draft=True)
    cfg_path = confidence.save_confidence_config(
        stage, draft=True, geocoder_stack=geocoder_stack
    )
    for b in settings.BUCKET_ORDER:
        t = thresholds[b]
        td = "inf" if t.gate_all_to_human else f"{t.tau_deployed:.4f}"
        log.info(f"  {b:22s} N={t.n_wrong_docs:2d} tau={td}")

    # --- diagnostic artifacts ---
    _write_json(
        settings.artifact_path("weight_validation.json", stage, draft=True),
        confidence.validate_weights(items),
    )
    _write_json(
        settings.artifact_path("gate_simulation.json", stage, draft=True),
        confidence.simulate_gate(items, thresholds),
    )

    # --- critic prompts ---
    log.info("Building Critic prompts...")
    prompt_diag = critic_prompt.build_all(stage, tax, grade_set, draft=True)
    log.info(f"  {prompt_diag['n_prompts']} prompts")

    # --- acronym commons + atomic schema ---
    # Both MUST be written with draft=True. Anything written straight into
    # artifacts/v(N)/ pre-creates the promoted directory, which then makes
    # ratify() refuse to promote on the grounds that the stage already exists --
    # and would leave Segment B able to load a half-populated stage.
    try:
        from segment_b.postproc import acronyms as acr

        commons_path = acr.write_commons_seed(stage, draft=True)
        log.info(f"  acronym commons -> {Path(commons_path).name}")
    except Exception as e:
        log.warning(f"  acronym commons seed failed: {e}")

    try:
        from . import atomic_verify

        schema_path = atomic_verify.save_atomic_schema(stage, draft=True)
        log.info(f"  atomic schema -> {Path(schema_path).name}")
    except Exception as e:
        log.warning(f"  atomic schema write failed: {e}")

    # --- next batch ---
    try:
        batch_path, batch_report = _write_next_batch(grade_set)
        log.info(
            f"  next_batch -> {Path(batch_path).name} "
            f"({batch_report.get('n_selected')} docs)"
        )
    except Exception as e:
        log.warning(f"  next_batch selection failed: {e}")
        batch_report = {"error": str(e)}

    # --- report ---
    report_path = write_report(
        stage,
        draft=True,
        grade_set=grade_set,
        thresholds=thresholds,
        items=items,
        tax_diag=tax_diag,
        prompt_diag=prompt_diag,
        score_diag=score_diag,
        prechecks=prechecks,
        geocoder_stack=geocoder_stack,
    )

    draft_dir = settings.stage_dir(stage, draft=True)
    log.info("")
    log.info(f"Draft written to {draft_dir}")
    log.info("Next steps:")
    log.info(f"  1. Review {taxonomy.artifact_path_for(stage, draft=True).name}")
    log.info(f"  2. Review {report_path.name}")
    log.info(f"  3. Promote:  python -m mcal.build --stage {stage} --ratify")

    return {
        "stage": stage,
        "draft_dir": str(draft_dir),
        "prechecks": [
            {"name": p.name, "ok": p.ok, "detail": p.detail} for p in prechecks
        ],
        "geocoder_stack": geocoder_stack,
        "taxonomy": tax_diag,
        "atomic": atomic_diag,
        "scoring": score_diag,
        "thresholds": confidence.summarize(thresholds, items),
        "prompts": {
            "n_prompts": prompt_diag["n_prompts"],
            "dir": prompt_diag["dir"],
        },
        "next_batch": batch_report,
        "report": str(report_path),
        "artifacts": {
            "taxonomy": str(tax_path),
            "thresholds": str(th_path),
            "confidence_config": str(cfg_path),
        },
    }


def _run_atomic(grade_set: grades.GradeSet, stage: str) -> tuple[Optional[dict], dict]:
    """Atomic verification over the graded corpus. Returns (citation_rates, diag)."""
    from pages import load_doc

    from . import atomic_verify

    rates: dict[tuple[str, str], float] = {}
    per_doc: dict[str, dict] = {}
    verifications = []
    for doc_id in grade_set.doc_ids:
        m2_path = settings.M2_DIR / f"{doc_id}.json"
        if not m2_path.exists():
            continue
        m2 = json.loads(m2_path.read_text(encoding="utf-8"))
        try:
            doc = load_doc(doc_id)
            dv = atomic_verify.verify_document(doc, m2)
        except Exception as e:
            log.warning(f"  atomic verify failed for {doc_id}: {e}")
            per_doc[doc_id] = {"error": str(e)}
            continue
        verifications.append((doc_id, dv))
        for field, rate in (dv.citation_rates() or {}).items():
            if rate is not None:
                rates[(doc_id, field)] = rate
        per_doc[doc_id] = {"n_fields": len(dv.citation_rates() or {})}

    audit: dict = {}
    try:
        audit = atomic_verify.false_negative_audit(
            [dv for _, dv in verifications], grade_set, stage=stage
        )
    except Exception as e:
        log.warning(f"  false-negative audit failed: {e}")
        audit = {"error": str(e)}

    try:
        log_payload = atomic_verify.build_failure_log(
            [dv for _, dv in verifications], grade_set, stage=stage
        )
        atomic_verify.save_failure_log(log_payload, stage, draft=True)
    except Exception as e:
        log.warning(f"  failure log write failed: {e}")

    return rates or None, {
        "ran": True,
        "n_docs": len(verifications),
        "per_doc": per_doc,
        "false_negative_audit": audit,
    }


def _write_next_batch(grade_set: grades.GradeSet) -> tuple[str, dict]:
    """
    Rank ungraded materialized docs and write next_batch.csv.

    `grade_set` must be threaded into `selection_report` as well as
    `rank_candidates`: without it the report cannot compute observed tag counts,
    so every tag looks zero-exemplar and the "which underrepresented tags does
    this batch cover" column -- the whole point of active selection -- is
    meaningless.
    """
    ranked = active_select.rank_candidates(grade_set=grade_set)
    path = active_select.write_next_batch(ranked, n=settings.NEXT_BATCH_SIZE)
    report = active_select.selection_report(
        ranked, n=settings.NEXT_BATCH_SIZE, grade_set=grade_set
    )
    return str(path), report


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


# --- Ratification -----------------------------------------------------------


def ratify(stage: str, *, force: bool = False) -> dict:
    """
    Promote `artifacts/v(N)-draft/` to `artifacts/v(N)/` (MCAL_PLAN 3.7).

    The whole directory moves atomically-ish: a partially-promoted stage would
    let Segment B load a taxonomy from one build and thresholds from another.
    Refuses to overwrite an existing promoted stage without --force, since
    artifacts are supposed to be frozen once Segment B pins them.
    """
    stage = settings.normalize_stage(stage)
    draft_dir = settings.stage_dir(stage, draft=True)
    final_dir = settings.stage_dir(stage, draft=False)

    if not draft_dir.exists():
        raise PrecheckFailure(
            f"No draft to ratify at {draft_dir}. Run "
            f"`python -m mcal.build --stage {stage}` first."
        )

    draft_tax = taxonomy.load(stage, draft=True)
    if draft_tax is None:
        raise PrecheckFailure(f"Draft has no taxonomy artifact: {draft_dir}")

    problems = taxonomy.validate_transition(
        taxonomy.load(settings.prior_stage(stage)) if settings.prior_stage(stage) else None,
        draft_tax,
    )
    if problems:
        raise PrecheckFailure(
            "Draft taxonomy violates the add-only rule; refusing to promote:\n  - "
            + "\n  - ".join(problems)
        )

    if final_dir.exists():
        # Distinguish a genuine frozen stage from a directory that only exists
        # because some writer defaulted to draft=False. A stage is only really
        # promoted if it carries a ratified taxonomy; anything else is debris and
        # blocking on it would be a dead end the user cannot clear from the CLI.
        existing_tax = taxonomy.load(stage, draft=False)
        genuinely_promoted = existing_tax is not None and existing_tax.ratified

        if genuinely_promoted and not force:
            raise PrecheckFailure(
                f"{final_dir} already exists and holds a ratified taxonomy "
                f"(frozen_at={existing_tax.frozen_at}). Artifacts are frozen "
                f"once promoted (MCAL_PLAN 2) -- Segment B may already have "
                f"pinned this stage. Use --force to overwrite, or build the "
                f"next stage instead: --stage "
                f"v{settings.stage_number(stage) + 1} --prior {stage}"
            )

        if genuinely_promoted:
            backup = final_dir.with_name(
                f"{final_dir.name}.superseded-"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            )
            shutil.move(str(final_dir), str(backup))
            log.warning(f"Existing {final_dir.name} moved to {backup.name}")
        else:
            stray = sorted(p.name for p in final_dir.rglob("*") if p.is_file())
            log.warning(
                f"{final_dir.name} exists but has no ratified taxonomy, so it is "
                f"not a promoted stage. Replacing it. Stray files: {stray}"
            )
            shutil.rmtree(final_dir)

    shutil.copytree(draft_dir, final_dir)

    # Re-stamp the taxonomy as ratified+frozen in the promoted location.
    draft_tax.stage = stage
    taxonomy.save(draft_tax, draft=False, ratified=True)

    log.info(f"Promoted {draft_dir.name} -> {final_dir.name}")
    log.info(f"Segment B can now pin stage {stage}.")
    return {
        "stage": stage,
        "promoted_to": str(final_dir),
        "frozen_at": draft_tax.frozen_at,
        "n_tags": len(draft_tax.tags),
    }


def status() -> dict:
    """What exists on disk right now."""
    promoted: list[str] = []
    drafts: list[str] = []
    if settings.ARTIFACTS_DIR.exists():
        for p in sorted(settings.ARTIFACTS_DIR.iterdir()):
            if not p.is_dir():
                continue
            if p.name.endswith("-draft"):
                drafts.append(p.name)
            elif p.name.startswith("v") and p.name[1:].isdigit():
                promoted.append(p.name)

    grade_set = grades.load_grades()
    return {
        "promoted_stages": promoted,
        "draft_stages": drafts,
        "latest_promoted": settings.latest_stage(),
        "m2_prompt_version": precheck_m2_prompt_version().detail,
        "geocoder_stack": settings.geocoder_precheck()["stack"],
        "graded_docs": grade_set.n_docs,
        "graded_items": len(grade_set.items),
        "materialized_docs": len(settings.available_doc_ids()),
        "ungraded_materialized": sorted(
            set(settings.normalize_doc_id(d) for d in settings.available_doc_ids())
            - set(grade_set.doc_ids)
        ),
    }


# --- CLI --------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m mcal.build",
        description="Build M-Cal calibration artifacts.",
    )
    ap.add_argument("--stage", help="target stage, e.g. v1")
    ap.add_argument("--prior", help="prior stage for add-only carry-forward, e.g. v1")
    ap.add_argument(
        "--grades",
        help="(informational) grade source; paths come from mcal/settings.py",
    )
    ap.add_argument("--out", help="(informational) artifact dir; see mcal/settings.py")
    ap.add_argument(
        "--ratify",
        action="store_true",
        help="promote artifacts/v(N)-draft/ to artifacts/v(N)/",
    )
    ap.add_argument("--force", action="store_true", help="overwrite a promoted stage")
    ap.add_argument("--status", action="store_true", help="show artifact status")
    ap.add_argument(
        "--atomic",
        action="store_true",
        help="run the Opus atomic-verification pass (expensive, adds s_citation)",
    )
    ap.add_argument(
        "--no-induction",
        action="store_true",
        help="skip the Sonnet taxonomy-induction call (offline build)",
    )
    ap.add_argument(
        "--skip-m2-precheck",
        action="store_true",
        help="build despite stale M2 prose (dry run only; result is not valid)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        if args.status:
            print(json.dumps(status(), indent=2))
            return 0

        if not args.stage:
            ap.error("--stage is required unless --status is given")

        if args.ratify:
            print(json.dumps(ratify(args.stage, force=args.force), indent=2))
            return 0

        result = run_build(
            args.stage,
            prior_stage=args.prior,
            run_induction=not args.no_induction,
            run_atomic=args.atomic,
            skip_m2_precheck=args.skip_m2_precheck,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    except PrecheckFailure as e:
        log.error(f"\n{e}")
        return 2
    except taxonomy.TaxonomyVersionError as e:
        log.error(f"\n{e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
