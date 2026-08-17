"""
Conformal HUMAN_REVIEW gate + `run_manifest.json` emitter (MCAL_PLAN 3.12,
build item #12).

Last stage of Segment B. Takes the per-field Critic results from
`segment_b/critic.py`, turns each into a composite confidence score via
`mcal/confidence.py`, compares it against its bucket's conformal threshold from
`thresholds.v(N).json`, and routes the field either to output or to HUMAN_REVIEW.

--------------------------------------------------------------------------
Why the manifest is the real product of this module
--------------------------------------------------------------------------

The gate itself is ten lines of arithmetic: `accept iff composite > tau_deployed`
(MCAL_PLAN 3.3), plus `gate_all_to_human` for buckets whose calibration set is
too small for any finite threshold to carry the guarantee. Everything else here
exists because of MCAL_PLAN 7.5's multi-round protocol.

At seed v1, N_wrong_docs < 3 in most buckets, so most buckets are
`degenerate_severe` and nearly every field routes to HUMAN_REVIEW. That is the
intended behaviour, not a malfunction: Segment B seed v1 exists to produce
extractions on a targeted next batch of ~10 documents so the user can grade them
and refit the thresholds. The grades from those gated fields ARE the calibration
data for v2.

Which makes `run_manifest.json` load-bearing. MCAL_PLAN 3.12: "The manifest must
carry enough information for a human to grade the field *without* opening any
other file." So a gated field emits its raw extraction, its evidence quote, its
cited pages, its rubric answers, its composite, the threshold it was measured
against, and WHY it was gated. MCAL_PLAN 7 Q8 says the same thing from the other
direction: HUMAN_REVIEW never means "skipped".

`gate_reason` is the diagnostic that decides where next round's engineering
effort goes, and it is reported with a deliberate priority order (see
`GATE_REASON_PRIORITY`) so that "the Critic is the binding constraint" and "the
gate is too conservative" never collapse into the same value.

--------------------------------------------------------------------------
What this module does NOT do
--------------------------------------------------------------------------

It does not decide thresholds -- `mcal/confidence.py` fits them at build time and
freezes them within a stage. It does not judge -- `segment_b/critic.py` does. It
does not re-extract: MCAL_PLAN 7 Q8's single automated retry at temperature +0.2
is orchestrated here but the extraction call itself is a caller-supplied
callback, because the gate has no business knowing how any given field is
extracted (M1 regex, Opus map-reduce, geocoder cascade -- all different).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from mcal import confidence, quote_check, settings
from mcal.confidence import BucketThreshold, Signals

# The artifact loader, stage resolver and Critic vocabulary live in critic.py.
# gate.py reads them from there rather than duplicating them so that the two
# modules cannot disagree about which stage is pinned, which field routes to
# Opus, or what a note string means -- and because the RE_EXTRACT retry has to
# re-run the Critic anyway (MCAL_PLAN 7 Q8).
from segment_b import critic as critic_mod
from segment_b.critic import (
    CriticResult,
    MissingArtifactError,
    load_confidence_config,
    resolve_stage,
)

# segment_a's flat modules; the sys.path bridge is installed by mcal.settings.
from pages import Doc  # noqa: E402

log = logging.getLogger(__name__)


# --- Vocabulary -------------------------------------------------------------

VERDICT_HUMAN_REVIEW = critic_mod.VERDICT_HUMAN_REVIEW
VERDICT_RE_EXTRACT = critic_mod.VERDICT_RE_EXTRACT

# MCAL_PLAN 3.12 specifies
# `composite_below_tau|bucket_degenerate_severe|policy_private_individual|critic_verdict|null`.
# Four are added:
#   * `dependent_field_cascade`  -- MCAL_PLAN 3.10 step 2's era gate produces
#     HUMAN_REVIEW routes that are neither a threshold nor a Critic judgement
#     about the field itself, and folding them into `critic_verdict` would make
#     `key_people` look like a Critic problem when the actual problem is `year`.
#   * `reduced_geocoder_stack`   -- MCAL_PLAN 3.9a forces the location bucket to
#     gate_all_to_human when PAD-US/GNIS/Mapbox are not installed. Reported as
#     `bucket_degenerate_severe` it would look like a data-scarcity problem
#     solvable by grading more documents; it is actually solvable by running two
#     downloads.
#   * `extraction_missing`       -- the field produced no value at all, which is
#     upstream of both the Critic and the gate.
#   * `critic_missing`           -- no Critic result was supplied for the field.
#     Emitted rather than skipped, per MCAL_PLAN 7 Q8 / requirement that no field
#     is ever dropped from the manifest.
GATE_REASONS = (
    "policy_private_individual",
    "dependent_field_cascade",
    "reduced_geocoder_stack",
    "bucket_degenerate_severe",
    "extraction_missing",
    "critic_missing",
    "critic_verdict",
    "composite_below_tau",
)

# Priority for the single reported `gate_reason`, most specific first.
#
# `critic_verdict` outranks `composite_below_tau` on purpose. When the Critic
# says HUMAN_REVIEW, s_critic = 0.0 and the composite is mechanically at most
# 0.5*s_quote, so `composite_below_tau` is almost always ALSO true -- reporting
# it would make the gate look like the binding constraint in every case where
# the judge was. Conversely a field whose Critic said PASS but whose composite
# still failed is the genuine "gate too conservative" signal, and that is exactly
# when `composite_below_tau` is reported. Every applicable reason is still kept
# in `gate_reasons` for the full picture.
GATE_REASON_PRIORITY = GATE_REASONS

# Critic override note -> gate reason. Read from critic.py's constants so a
# rename cannot silently break the mapping.
_OVERRIDE_TO_REASON = {
    critic_mod.NOTE_PRIVATE_INDIVIDUAL: "policy_private_individual",
    critic_mod.NOTE_AMBIGUOUS_CAPACITY: "policy_private_individual",
    critic_mod.NOTE_DEPENDENT_CASCADE: "dependent_field_cascade",
    critic_mod.NOTE_EXTRACTION_MISSING: "extraction_missing",
}

# MCAL_PLAN 7 Q8: "RE_EXTRACT -> one automated re-extraction attempt with
# temperature +0.2, re-run through Critic."
RE_EXTRACT_TEMPERATURE_DELTA = 0.2
# segment_a/llm.py's default temperature, which is what the M2 extractors run at.
BASE_EXTRACTION_TEMPERATURE = 0.2
MAX_RE_EXTRACT_ATTEMPTS = 1

# Manifest layout. `run_manifest.json`'s top level is the bare field map from
# MCAL_PLAN 3.12; doc-level material goes under underscore-prefixed reserved
# keys, which cannot collide with a field name (every key in
# settings.ALL_FIELDS is lowercase letters, digits, `.` and `_` and none starts
# with `_`).
MANIFEST_FILENAME = "run_manifest.json"
META_KEY = "_meta"
ROLLUP_KEY = "_rollup"
MONITOR_KEY = "_null_tag_monitor"
RESERVED_KEYS = (META_KEY, ROLLUP_KEY, MONITOR_KEY)

SEGMENT_B_OUTPUT = settings.SEGMENT_B_DIR / "output"

# How many per-batch rows the rolling null-tag monitor keeps. Bounded so the file
# does not grow without limit over a 2,000-document run; the per-bucket totals it
# is actually read for are cumulative and unaffected.
MONITOR_MAX_BATCHES = 50
MONITOR_MAX_DOCS = 200


# --- Threshold loading ------------------------------------------------------


def thresholds_from_payload(payload: dict) -> dict[str, BucketThreshold]:
    """
    Rehydrate `thresholds.v(N).json` into `BucketThreshold` objects.

    Going through the real dataclass rather than reading the dict directly means
    the gate uses `BucketThreshold.accepts()` -- the same acceptance rule
    `mcal/confidence.py` used in `gate_simulation.v(N).json`. If the rule ever
    changes, the simulation and the production gate change together, which is the
    whole point of having one.

    All seven buckets are required (MCAL_PLAN 3.3, frozen per 7.5). A partial
    thresholds file would otherwise mean some fields silently had no gate.
    """
    buckets = payload.get("buckets")
    if not isinstance(buckets, dict):
        raise MissingArtifactError(
            "thresholds artifact has no `buckets` object; expected the shape "
            "written by mcal.confidence.save_thresholds."
        )
    missing = [b for b in settings.BUCKET_ORDER if b not in buckets]
    if missing:
        raise MissingArtifactError(
            f"thresholds artifact is missing bucket(s) {missing}. All "
            f"{len(settings.BUCKET_ORDER)} buckets are required and frozen "
            f"(MCAL_PLAN 3.3, 7.5); rebuild with `python -m mcal.build`."
        )

    out: dict[str, BucketThreshold] = {}
    for bucket in settings.BUCKET_ORDER:
        raw = buckets[bucket] or {}
        tau = raw.get("tau_deployed")
        gate_all = bool(raw.get("gate_all_to_human"))
        notes = list(raw.get("notes") or [])
        if tau is None:
            # A hand-edited artifact with a null threshold is treated as "no
            # finite threshold exists", i.e. gate everything. Defaulting the
            # other way would accept every field in the bucket on the strength of
            # a missing value.
            tau = float("inf")
            gate_all = True
            notes.append(
                "tau_deployed was null in the artifact; treated as +inf and "
                "forced to gate_all_to_human."
            )
        out[bucket] = BucketThreshold(
            bucket=raw.get("bucket") or bucket,
            alpha=float(raw.get("alpha", settings.ALPHA)),
            alpha_effective=float(
                raw.get("alpha_effective", raw.get("alpha", settings.ALPHA))
            ),
            n_wrong_docs=int(raw.get("N_wrong_docs", raw.get("n_wrong_docs", 0)) or 0),
            n_items=int(raw.get("n_items", 0) or 0),
            n_wrong_items=int(raw.get("n_wrong_items", 0) or 0),
            tau_raw=raw.get("tau_raw"),
            curation_slack=float(raw.get("curation_slack", 0.0) or 0.0),
            tau_deployed=float(tau),
            saturated=bool(raw.get("saturated")),
            degenerate=bool(raw.get("degenerate")),
            degenerate_severe=bool(raw.get("degenerate_severe")),
            gate_all_to_human=gate_all,
            guarantee_conditioning=str(raw.get("guarantee_conditioning") or ""),
            loo_deltas=list(raw.get("loo_deltas") or []),
            wrong_docs=list(raw.get("wrong_docs") or []),
            r_docs=dict(raw.get("r_docs") or {}),
            notes=notes,
        )
    return out


def load_bucket_thresholds(stage: str) -> dict[str, BucketThreshold]:
    """Load and rehydrate the promoted thresholds for `stage`."""
    try:
        payload = confidence.load_thresholds(stage)
    except FileNotFoundError as e:
        raise MissingArtifactError(str(e)) from e
    return thresholds_from_payload(payload)


# --- Signals ----------------------------------------------------------------


def extraction_quote_verdict(
    field: str, entry: Any, doc: Optional[Doc]
) -> Optional[str]:
    """
    `s_quote`'s input: the aggregate verdict over the EXTRACTION's own quotes.

    Not the Critic's `evidence_quote`. MCAL_PLAN 3.3 defines `s_quote` as the
    quote_check verdict on the extracted value, and the Critic's quote is already
    accounted for -- an unverifiable one forces the verdict to HUMAN_REVIEW,
    which drives `s_critic` to 0.0. Reusing it here would double-count one
    measurement as two independent signals and make the composite look more
    informative than it is.

    Returns None where `s_quote` is undefined, which `compute_signals` reads as
    "use the M1 default of 1.0":
      * M1 fields have no verbatim quote in their values by design (MCAL_PLAN
        3.3), so their composite is `0.5*s_critic + 0.5`;
      * a document-less call (manifest regeneration from persisted results) can
        verify nothing, and asserting "no" would penalize every field for the
        caller's convenience.
    """
    if field in settings.M1_FIELDS:
        return None
    if doc is None:
        return None
    evidence = critic_mod.evidence_dicts(field, entry)
    if not evidence:
        # No quotes at all. `aggregate_verdict([])` is "no" and that is right:
        # MCAL_PLAN 1(5)-(7) is a family of missing-citation failures, so absence
        # of evidence must score 0, not be excused as "not applicable".
        return "no"
    checks = quote_check.check_evidence_list(evidence, doc)
    return quote_check.aggregate_verdict(checks)


def citation_rate(field: str, entry: Any) -> Optional[float]:
    """
    `s_citation`: fraction of the field's evidence items carrying a page cite.

    Weight 0 at this stage (MCAL_PLAN 3.3) but computed and stored, because
    weight validation at n>=60 cannot retro-fit a signal nobody recorded.
    """
    evidence = critic_mod.evidence_dicts(field, entry)
    if not evidence:
        return None
    cited = sum(1 for ev in evidence if quote_check.coerce_pages(ev.get("source_pages")))
    return cited / len(evidence)


def source_agreement(field: str, entry: Any) -> Optional[tuple[int, int]]:
    """
    `s_source`: M1 provenance agreement, as `(n_agree, n_total)`.

    A deliberately crude read of what M1 actually records. `m1.py` writes
    `sources: ["NUL", "regex (first 3 pages)"]` plus a free-text `note` that says
    "disagrees" when they conflict; there is no structured agreement field. Weight
    is 0, so the approximation costs nothing today and gives weight validation
    something to look at later. Upgrading it is a Segment A change (have M1 emit
    per-source values), noted rather than faked here.
    """
    if field not in settings.M1_FIELDS or not isinstance(entry, dict):
        return None
    sources = entry.get("sources")
    if not isinstance(sources, list) or not sources:
        return None
    note = str(entry.get("note") or "").lower()
    disagrees = "disagree" in note or "conflict" in note or "flag" in note
    n_total = len(sources)
    return (1 if disagrees else n_total, n_total)


def acronym_rate(entry: Any) -> Optional[float]:
    """
    `s_acronym`: defined-first-use rate, when the acronym post-pass recorded one.

    `postproc/acronyms.py` is the natural producer; until a driver threads its
    per-field statistics onto the M2 entry this is None (weight 0, so the
    composite is unaffected) rather than a fabricated 1.0.
    """
    if isinstance(entry, dict):
        for key in ("acronym_rate", "defined_first_use_rate"):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                return max(0.0, min(1.0, float(value)))
    return None


def signals_for_field(
    field: str,
    entry: Any,
    critic_result: Optional[CriticResult],
    doc: Optional[Doc],
) -> tuple[Signals, Optional[str]]:
    """Assemble the six confidence signals for one field. Returns (signals, quote_verdict)."""
    verdict = critic_result.verdict if critic_result else VERDICT_HUMAN_REVIEW
    quote_verdict = extraction_quote_verdict(field, entry, doc)
    sig = confidence.compute_signals(
        field,
        quote_verdict=quote_verdict,
        critic_verdict=verdict,
        citation_rate=citation_rate(field, entry),
        # s_shard needs per-atom `source_chunk_id` from atomic_verify.py
        # (MCAL_PLAN 3.3), which does not run in this path. Left None.
        shard_rate=None,
        acronym_rate=acronym_rate(entry),
        source_agreement=source_agreement(field, entry),
    )
    return sig, quote_verdict


# --- Per-field gate decision ------------------------------------------------


@dataclass
class FieldGate:
    """
    One field's gate decision, serialized straight into `run_manifest.json`.

    Field names match MCAL_PLAN 3.12's schema exactly; `to_manifest_entry()`
    emits those keys first, then the audit extras. The extras are additive on
    purpose -- a reviewer reads the schema keys, and a diagnostician reads the
    rest -- and a schema-conformance test asserts the required keys are all
    present for every field.
    """

    field: str
    bucket: str
    artifact_stage: str
    extracted_value: Any = None
    evidence_quote: Optional[str] = None
    source_pages: list[int] = dc_field(default_factory=list)
    verdict: str = VERDICT_HUMAN_REVIEW
    rubric_answers: dict[str, str] = dc_field(default_factory=dict)
    composite: Optional[float] = None
    applied_tau: Optional[float] = None
    gated_to_human: bool = True
    gate_reason: Optional[str] = None
    failure_tag: Optional[str] = None
    judge_model: str = "sonnet"
    # --- audit extras (not part of the 3.12 schema) ---
    critic_verdict: Optional[str] = None
    verdict_before_override: Optional[str] = None
    gate_reasons: list[str] = dc_field(default_factory=list)
    signals: dict = dc_field(default_factory=dict)
    quote_verdict: Optional[str] = None
    quote_check: Optional[dict] = None
    bucket_flags: dict = dc_field(default_factory=dict)
    critic_overrides: list[str] = dc_field(default_factory=list)
    off_vocabulary_failure_tag: Optional[str] = None
    note: Optional[str] = None
    empty_but_valid: bool = False
    extraction_missing: bool = False
    re_extract: Optional[dict] = None
    evidence_meta: dict = dc_field(default_factory=dict)

    def to_manifest_entry(self) -> dict:
        return {
            # --- MCAL_PLAN 3.12 schema, in the order the plan writes it ---
            "extracted_value": self.extracted_value,
            "evidence_quote": self.evidence_quote,
            "source_pages": list(self.source_pages),
            "verdict": self.verdict,
            "rubric_answers": dict(self.rubric_answers),
            "composite": self.composite,
            "applied_tau": self.applied_tau,
            "gated_to_human": self.gated_to_human,
            "gate_reason": self.gate_reason,
            "failure_tag": self.failure_tag,
            "bucket": self.bucket,
            "artifact_stage": self.artifact_stage,
            "judge_model": self.judge_model,
            # --- audit extras ---
            "critic_verdict": self.critic_verdict,
            "verdict_before_override": self.verdict_before_override,
            "gate_reasons": list(self.gate_reasons),
            "signals": dict(self.signals),
            "quote_verdict": self.quote_verdict,
            "quote_check": self.quote_check,
            "bucket_flags": dict(self.bucket_flags),
            "critic_overrides": list(self.critic_overrides),
            "off_vocabulary_failure_tag": self.off_vocabulary_failure_tag,
            "note": self.note,
            "empty_but_valid": self.empty_but_valid,
            "extraction_missing": self.extraction_missing,
            "re_extract": self.re_extract,
            "evidence_meta": dict(self.evidence_meta),
        }


# The keys MCAL_PLAN 3.12 requires of every manifest entry. Exported so tests --
# and any future reviewer UI -- can assert conformance against one list.
MANIFEST_REQUIRED_KEYS = (
    "extracted_value",
    "evidence_quote",
    "source_pages",
    "verdict",
    "rubric_answers",
    "composite",
    "applied_tau",
    "gated_to_human",
    "gate_reason",
    "failure_tag",
    "bucket",
    "artifact_stage",
    "judge_model",
)


def _finite(x: Optional[float]) -> Optional[float]:
    """+/-inf and NaN -> None, so the manifest stays strict-JSON valid."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return round(f, 6)


def gate_field(
    field: str,
    entry: Any,
    critic_result: Optional[CriticResult],
    threshold: BucketThreshold,
    *,
    stage: str,
    doc: Optional[Doc] = None,
    geocoder_stack: str = "full",
) -> FieldGate:
    """
    Decide one field (MCAL_PLAN 3.12).

    The raw extraction is ALWAYS carried through, gated or not (MCAL_PLAN 7 Q8).
    That is the single most important behaviour in this function: at seed v1 the
    reviewer grades from gated fields, and a gate that dropped their values would
    make the multi-round protocol unable to produce v2's calibration data.

    `summary_of_interest` distinguishes `[]` (a legitimately routine document)
    from `null` (the field failed to generate) exactly as MCAL_PLAN 3.12
    requires: the former keeps the empty list as its `extracted_value`, the
    latter emits null AND an `extraction_missing` gate_reason. An empty list is
    never coerced into a failure anywhere in this path.
    """
    bucket = settings.bucket_for_field(field)
    value = critic_mod.extracted_value(field, entry)
    empty_but_valid = critic_mod.is_legitimately_empty(field, entry)
    missing = critic_mod.is_missing(field, entry) and not empty_but_valid
    if missing:
        # Emit null rather than whatever partial envelope survived, so
        # "generation failure" is unambiguous in the manifest.
        value = None

    sig, quote_verdict = signals_for_field(field, entry, critic_result, doc)
    score = confidence.composite(sig)

    reasons: list[str] = []

    if critic_result is None:
        reasons.append("critic_missing")
    else:
        for override in critic_result.overrides:
            mapped = _OVERRIDE_TO_REASON.get(override.split(":", 1)[0])
            if mapped and mapped not in reasons:
                reasons.append(mapped)

    if missing and "extraction_missing" not in reasons:
        reasons.append("extraction_missing")

    if threshold.gate_all_to_human:
        # Two different causes with two different remedies (see GATE_REASONS).
        if bucket == "location" and geocoder_stack == "reduced":
            reasons.append("reduced_geocoder_stack")
        elif threshold.degenerate_severe:
            reasons.append("bucket_degenerate_severe")
        else:
            reasons.append("composite_below_tau")

    critic_verdict = critic_result.verdict if critic_result else VERDICT_HUMAN_REVIEW
    if critic_verdict in (VERDICT_HUMAN_REVIEW, VERDICT_RE_EXTRACT):
        if "critic_verdict" not in reasons:
            reasons.append("critic_verdict")

    if not threshold.accepts(score) and "composite_below_tau" not in reasons:
        reasons.append("composite_below_tau")

    gated = bool(reasons)
    primary = next((r for r in GATE_REASON_PRIORITY if r in reasons), None)

    return FieldGate(
        field=field,
        bucket=bucket,
        artifact_stage=stage,
        extracted_value=value,
        evidence_quote=critic_result.evidence_quote if critic_result else None,
        source_pages=list(critic_result.source_pages) if critic_result else [],
        verdict=VERDICT_HUMAN_REVIEW if gated else critic_verdict,
        rubric_answers=dict(critic_result.rubric_answers) if critic_result else {},
        composite=_finite(score),
        applied_tau=_finite(threshold.tau_deployed),
        gated_to_human=gated,
        gate_reason=primary,
        failure_tag=critic_result.failure_tag if critic_result else None,
        judge_model=(
            critic_result.judge_model
            if critic_result
            else critic_mod.judge_model_for(field)[1]
        ),
        critic_verdict=critic_verdict,
        verdict_before_override=(
            critic_result.verdict_before_override if critic_result else None
        ),
        gate_reasons=reasons,
        signals=sig.to_dict(),
        quote_verdict=quote_verdict,
        quote_check=critic_result.quote_check if critic_result else None,
        bucket_flags={
            "gate_all_to_human": threshold.gate_all_to_human,
            "degenerate": threshold.degenerate,
            "degenerate_severe": threshold.degenerate_severe,
            "saturated": threshold.saturated,
            "tau_deployed_infinite": _finite(threshold.tau_deployed) is None,
            "N_wrong_docs": threshold.n_wrong_docs,
        },
        critic_overrides=list(critic_result.overrides) if critic_result else [],
        off_vocabulary_failure_tag=(
            critic_result.off_vocabulary_failure_tag if critic_result else None
        ),
        note=critic_result.note if critic_result else "no critic result supplied",
        empty_but_valid=empty_but_valid,
        extraction_missing=missing,
        evidence_meta=dict(critic_result.evidence_meta) if critic_result else {},
    )


# --- RE_EXTRACT retry (MCAL_PLAN 7 Q8) --------------------------------------


@dataclass
class ReExtraction:
    """
    What a re-extraction callback returns.

    `entry` is a replacement M1/M2-shaped entry for the field. `critic_result` is
    optional: if the callback only re-extracts, `run_gate` re-runs the Critic
    itself (which is the "re-run through Critic" half of MCAL_PLAN 7 Q8).
    """

    entry: Any = None
    critic_result: Optional[CriticResult] = None
    note: str = ""
    ok: bool = True


def _coerce_reextraction(raw: Any) -> Optional[ReExtraction]:
    if raw is None:
        return None
    if isinstance(raw, ReExtraction):
        return raw
    if isinstance(raw, dict):
        return ReExtraction(
            entry=raw.get("entry", raw.get("extracted_entry")),
            critic_result=raw.get("critic_result"),
            note=str(raw.get("note") or ""),
            ok=bool(raw.get("ok", True)),
        )
    # A bare replacement entry is accepted for callback convenience.
    return ReExtraction(entry=raw)


def re_extract_fields(
    fields_to_retry: Sequence[str],
    *,
    reextract: Callable,
    doc: Optional[Doc],
    m1: Optional[dict],
    m2: Optional[dict],
    critic_results: dict[str, CriticResult],
    stage: str,
    config: Optional[dict],
    call: Optional[Callable] = None,
    base_temperature: float = BASE_EXTRACTION_TEMPERATURE,
) -> tuple[dict[str, dict], dict[str, Any]]:
    """
    One automated re-extraction attempt per RE_EXTRACT field (MCAL_PLAN 7 Q8).

    Mutates `critic_results` and returns `(audit, replacement_entries)`. The
    replacement entries matter: the manifest must show the value the surviving
    verdict was passed on, or a reviewer would grade the SUPERSEDED extraction
    against the RETRY's verdict -- silently the worst possible outcome for a file
    whose whole purpose is being gradable in isolation (MCAL_PLAN 3.12).

    Three decisions worth stating:

      * **Exactly one attempt.** The plan says "one"; a retry loop on a
        systematically unextractable field would burn tokens on 2,000 documents
        to reach the same HUMAN_REVIEW.
      * **The retry's verdict replaces the original unconditionally**, even when
        it is worse. Keeping the better of two samples would bias `s_critic`
        upward relative to the calibration set, where each item was scored once,
        and the conformal guarantee assumes the new item is drawn like the
        calibration items were.
      * **A callback exception is caught.** A retry is an optimization; losing
        the document because the optimization failed is not a trade worth making.
    """
    audit: dict[str, dict] = {}
    replacements: dict[str, Any] = {}
    temperature = base_temperature + RE_EXTRACT_TEMPERATURE_DELTA

    for field in fields_to_retry:
        before = critic_results.get(field)
        record = {
            "attempted": True,
            "attempt": 1,
            "max_attempts": MAX_RE_EXTRACT_ATTEMPTS,
            "temperature": round(temperature, 4),
            "base_temperature": base_temperature,
            "verdict_before": before.verdict if before else None,
            "verdict_after": None,
            "replaced": False,
            "error": None,
            "note": "",
        }
        try:
            raw = reextract(
                field,
                attempt=1,
                temperature=temperature,
                entry=critic_mod.extracted_entry(field, m1, m2),
                critic_result=before,
            )
        except Exception as e:  # noqa: BLE001 - a failed retry must not lose a doc
            log.warning("re-extraction callback failed on %s: %s", field, e)
            record["error"] = f"{type(e).__name__}: {e}"[:200]
            audit[field] = record
            continue

        outcome = _coerce_reextraction(raw)
        if outcome is None or not outcome.ok:
            record["note"] = (outcome.note if outcome else "") or "callback declined"
            audit[field] = record
            continue

        record["note"] = outcome.note
        new_result = outcome.critic_result
        if new_result is None and outcome.entry is not None and doc is not None:
            try:
                new_result = critic_mod.critique_field(
                    field, doc, m1, m2,
                    stage=stage, config=config, call=call,
                    extracted_override=outcome.entry,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("re-critique failed on %s: %s", field, e)
                record["error"] = f"{type(e).__name__}: {e}"[:200]
                audit[field] = record
                continue

        if new_result is not None:
            new_result.add_note("re_extracted_at_temperature_+0.2")
            critic_results[field] = new_result
            record["verdict_after"] = new_result.verdict
            record["replaced"] = True
            if outcome.entry is not None:
                replacements[field] = outcome.entry
                record["entry_replaced"] = True
        elif outcome.entry is not None:
            # A new value with no new verdict (no `doc`, so no re-critique) is
            # worse than useless: the manifest would show the retry's value under
            # the original verdict, and a reviewer grading the file in isolation
            # could not tell. Keep the original extraction and say why.
            record["note"] = (
                (record["note"] + "; " if record["note"] else "")
                + "entry not adopted: no critic result and no doc to re-judge it"
            )
        audit[field] = record
    return audit, replacements


# --- summary_of_interest diagnostics (MCAL_PLAN 6) ---------------------------

SALIENCE_CRITERIA = (
    "contested",
    "unusual_impact",
    "large_magnitude",
    "novel_alternative",
    "community_pushback",
    "precedent",
    "cross_jurisdictional",
)

TAG_MANUFACTURED_SALIENCE = "T17_manufactured_salience"
TAG_DUPLICATES_SUMMARY = "T18_salience_duplicates_summary"


def _tokens(text: str) -> set[str]:
    """
    Content tokens of a string, using `quote_check`'s tokenizer.

    Reused rather than re-rolled so the overlap metric is computed on the same
    stopword-and-boilerplate-filtered vocabulary the quote checker uses. That
    matters: "environmental", "impact", "project" and "alternative" appear on
    nearly every page of nearly every EIS, and counting them as shared content
    would report high overlap between any two texts in this corpus.
    """
    return set(quote_check.content_tokens(quote_check.normalize(text or "")))


def soi_summary_overlap(soi_value: Any, summary_value: Any) -> dict:
    """
    Token-level Jaccard overlap between `summary_of_interest` and the standard
    summary (MCAL_PLAN 6, "Overlap with standard summary").

    High overlap means the field is duplicating rather than complementing, which
    is `T18_salience_duplicates_summary` at the entry level and a rubric problem
    at the corpus level. Per-entry maxima are reported alongside the aggregate
    because one duplicated entry among six is a different problem from six
    half-duplicated ones, and the aggregate cannot tell them apart.
    """
    summary_tokens: set[str] = set()
    if isinstance(summary_value, dict):
        for sub in summary_value.values():
            if isinstance(sub, dict):
                summary_tokens |= _tokens(str(sub.get("text") or ""))
            elif isinstance(sub, str):
                summary_tokens |= _tokens(sub)
    elif isinstance(summary_value, str):
        summary_tokens = _tokens(summary_value)

    entries = soi_value if isinstance(soi_value, list) else []
    per_entry: list[Optional[float]] = []
    all_tokens: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            per_entry.append(None)
            continue
        text = f"{item.get('claim') or ''} {item.get('why_notable') or ''}"
        toks = _tokens(text)
        all_tokens |= toks
        per_entry.append(_jaccard(toks, summary_tokens))

    return {
        "jaccard": _jaccard(all_tokens, summary_tokens),
        "n_soi_tokens": len(all_tokens),
        "n_summary_tokens": len(summary_tokens),
        "per_entry_jaccard": per_entry,
        "max_entry_jaccard": max([p for p in per_entry if p is not None], default=None),
        "tokenizer": "mcal.quote_check.content_tokens (stopword+boilerplate filtered)",
    }


def _jaccard(a: set[str], b: set[str]) -> Optional[float]:
    if not a and not b:
        return None
    union = a | b
    if not union:
        return None
    return round(len(a & b) / len(union), 4)


def soi_diagnostics(entry: Any, m2: Optional[dict], gate: Optional[FieldGate]) -> dict:
    """
    Per-document `summary_of_interest` diagnostics (MCAL_PLAN 6, 3.15).

    Per-document by necessity -- the headline metric is a CORPUS non-empty rate
    with a ~60% ceiling, which no single document can compute. What is emitted
    here is the per-document term of that rate plus everything else the plan asks
    for; `aggregate_soi_diagnostics` folds a batch of them into the rate itself.

    `soi_useful` is present and null on purpose: MCAL_PLAN 7 Q5 adds it as a
    reviewer-filled grading-sheet column and 6 calls it "the field's real
    acceptance test". Emitting the key makes the manifest self-describing about
    what the reviewer still owes.
    """
    value = critic_mod.extracted_value(settings.SUMMARY_OF_INTEREST, entry)
    entries = value if isinstance(value, list) else []
    missing = value is None

    counts = {c: 0 for c in SALIENCE_CRITERIA}
    off_vocabulary: list[str] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        crit = str(item.get("salience_criterion") or "").strip()
        if crit in counts:
            counts[crit] += 1
        else:
            off_vocabulary.append(crit or "(missing)")

    tag = gate.failure_tag if gate else None
    return {
        "present": not missing,
        "generation_failure": missing,
        "non_empty": bool(entries),
        "n_entries": len(entries),
        "salience_criterion_counts": counts,
        "off_vocabulary_criteria": off_vocabulary,
        "n_t17_manufactured_salience": 1 if tag == TAG_MANUFACTURED_SALIENCE else 0,
        "n_t18_duplicates_summary": 1 if tag == TAG_DUPLICATES_SUMMARY else 0,
        "overlap_with_standard_summary": soi_summary_overlap(
            value, (m2 or {}).get("summary")
        ),
        "nonempty_rate_ceiling": settings.SOI_NONEMPTY_RATE_CEILING,
        "soi_useful": None,
    }


def aggregate_soi_diagnostics(per_doc: Sequence[dict]) -> dict:
    """
    Batch-level roll-up of the per-document SOI diagnostics (MCAL_PLAN 6).

    MCAL_PLAN 3.15: a non-empty rate above ~60% is evidence the field is
    manufacturing salience rather than detecting it; a rate near 0% suggests the
    rubric is too strict. Both are reported as flags, neither is a halt.
    """
    docs = [d for d in per_doc if isinstance(d, dict) and d.get("present")]
    n = len(docs)
    non_empty = sum(1 for d in docs if d.get("non_empty"))
    counts = {c: 0 for c in SALIENCE_CRITERIA}
    for d in docs:
        for crit, k in (d.get("salience_criterion_counts") or {}).items():
            if crit in counts:
                counts[crit] += int(k or 0)
    jaccards = [
        (d.get("overlap_with_standard_summary") or {}).get("jaccard")
        for d in docs
    ]
    jaccards = [j for j in jaccards if isinstance(j, (int, float))]
    rate = (non_empty / n) if n else None
    used = [c for c, k in counts.items() if k]
    return {
        "n_docs": n,
        "n_docs_missing_field": sum(
            1 for d in per_doc if isinstance(d, dict) and not d.get("present")
        ),
        "n_non_empty": non_empty,
        "non_empty_rate": round(rate, 4) if rate is not None else None,
        "nonempty_rate_ceiling": settings.SOI_NONEMPTY_RATE_CEILING,
        "exceeds_nonempty_ceiling": bool(
            rate is not None and rate > settings.SOI_NONEMPTY_RATE_CEILING
        ),
        "near_zero_nonempty_rate": bool(rate is not None and rate < 0.05),
        "salience_criterion_counts": counts,
        "n_criteria_used": len(used),
        "criterion_distribution_collapsed": bool(used and len(used) <= 2 and n >= 5),
        "n_t17_manufactured_salience": sum(
            int(d.get("n_t17_manufactured_salience") or 0) for d in docs
        ),
        "n_t18_duplicates_summary": sum(
            int(d.get("n_t18_duplicates_summary") or 0) for d in docs
        ),
        "mean_jaccard_with_standard_summary": (
            round(sum(jaccards) / len(jaccards), 4) if jaccards else None
        ),
        "gating": False,
        "note": (
            "All summary_of_interest metrics are non-gating: the field is new and "
            "has no baseline in the Evaluation CSV (MCAL_PLAN 6). The real "
            "acceptance test is the reviewer's `soi_useful` column (7 Q5)."
        ),
    }


# --- Null-tag monitor (MCAL_PLAN 6) -----------------------------------------


def null_tag_counts(fields: dict[str, FieldGate]) -> dict[str, dict]:
    """
    Per-bucket HUMAN_REVIEW / null-`failure_tag` counts for one document.

    MCAL_PLAN 6: "whenever a field routes to HUMAN_REVIEW with failure_tag =
    null, it means the taxonomy did not have a matching category."

    Two deliberate exclusions from the numerator, both cases where a null tag is
    correct rather than a gap:

      * a POLICY route (`policy_private_individual`). `templates/rubrics/_base.md`
        decision rule 1 mandates `failure_tag = null` there, because a private
        individual's stance is reviewed regardless of whether anything is wrong
        with it. Counting it would make the monitor fire on documents with many
        commenters and send the next taxonomy revision looking for a code that
        should not exist.
      * anything gated BEFORE a judgement was made: `bucket_degenerate_severe`,
        `reduced_geocoder_stack`, `extraction_missing`, `critic_missing`. At seed
        v1 the first of those covers most fields, so counting them would put the
        rate near 1.0 in every bucket and destroy the signal precisely when it is
        supposed to be read. The other three are the same argument in miniature: a
        field with no value, or no verdict, gives the taxonomy nothing to
        categorize, so its null tag is not evidence of a missing category.

    Off-vocabulary tags are counted separately. They are the other explanation
    for a null tag (a judge inventing codes), and conflating the two would send a
    prompt problem to the taxonomy for repair.
    """
    out: dict[str, dict] = {}
    for gate in fields.values():
        row = out.setdefault(
            gate.bucket,
            {
                "n_fields": 0,
                "n_human_review": 0,
                "n_null_tag": 0,
                "n_off_vocabulary": 0,
                "n_excluded_policy": 0,
                "n_excluded_pre_judgement": 0,
            },
        )
        row["n_fields"] += 1
        if gate.off_vocabulary_failure_tag:
            row["n_off_vocabulary"] += 1
        if gate.verdict != VERDICT_HUMAN_REVIEW:
            continue
        if gate.gate_reason == "policy_private_individual":
            row["n_excluded_policy"] += 1
            continue
        if gate.gate_reason in NULL_TAG_PRE_JUDGEMENT_REASONS:
            row["n_excluded_pre_judgement"] += 1
            continue
        row["n_human_review"] += 1
        if gate.failure_tag is None:
            row["n_null_tag"] += 1
    return out


# Gate reasons that fire before the Critic has judged the field's content, and
# are therefore excluded from the null-tag numerator (see `null_tag_counts`).
NULL_TAG_PRE_JUDGEMENT_REASONS = (
    "bucket_degenerate_severe",
    "reduced_geocoder_stack",
    "extraction_missing",
    "critic_missing",
)


def monitor_path() -> Path:
    return settings.NULL_TAG_MONITOR_PATH


def _blank_monitor() -> dict:
    return {
        "artifact": "null_tag_monitor.json",
        "plan_ref": "MCAL_PLAN 2, 6",
        "threshold": settings.NULL_TAG_REFRESH_THRESHOLD,
        "rolling": True,
        "halt_condition": False,
        "note": (
            "Rolling per-bucket rate of HUMAN_REVIEW routes with failure_tag = "
            "null. Exceeding the threshold in any bucket signals that the "
            "taxonomy needs a v(N+1) refresh with new T19+ codes. NOT a Segment "
            "B halt condition, but a mandatory input to the next M-Cal "
            "recalibration (MCAL_PLAN 6)."
        ),
        "n_docs": 0,
        "stages_seen": [],
        "docs": [],
        "per_bucket": {},
        "batches": [],
    }


def update_null_tag_monitor(
    fields: dict[str, FieldGate],
    *,
    stage: str,
    doc_id: str,
    path: Optional[Path] = None,
    batch_id: Optional[str] = None,
) -> dict:
    """
    Fold one document's counts into the rolling monitor and rewrite it.

    Rolling and cumulative because MCAL_PLAN 2 lists `null_tag_monitor.json` as
    the one artifact that is NOT stage-versioned and is maintained at "batch
    level": a single document's per-bucket rate is computed over 1-5 items and is
    noise, while the decision it feeds (does the taxonomy need T19+ codes?) is
    made once per ~10-document batch.

    Written atomically. A partially written monitor would be read as a corrupted
    JSON file on the next document and the accumulated history would be lost;
    losing history to a crash mid-batch would silently reset the rate.
    """
    target = Path(path) if path is not None else monitor_path()
    data = _blank_monitor()
    if target.exists():
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data.update(loaded)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("null-tag monitor at %s unreadable (%s); starting fresh", target, e)
            data = _blank_monitor()
    # Constants always come from settings, never from the stale file.
    data["threshold"] = settings.NULL_TAG_REFRESH_THRESHOLD

    counts = null_tag_counts(fields)
    per_bucket = data.setdefault("per_bucket", {})
    for bucket, row in counts.items():
        agg = per_bucket.setdefault(
            bucket,
            {
                "n_fields": 0,
                "n_human_review": 0,
                "n_null_tag": 0,
                "n_off_vocabulary": 0,
                "n_excluded_policy": 0,
                "n_excluded_pre_judgement": 0,
            },
        )
        for key, value in row.items():
            agg[key] = int(agg.get(key, 0)) + int(value)

    needing: list[str] = []
    for bucket, agg in per_bucket.items():
        n_hr = int(agg.get("n_human_review", 0))
        n_null = int(agg.get("n_null_tag", 0))
        rate = (n_null / n_hr) if n_hr else None
        agg["null_tag_rate"] = round(rate, 4) if rate is not None else None
        agg["exceeds_threshold"] = bool(
            rate is not None and rate > settings.NULL_TAG_REFRESH_THRESHOLD
        )
        if agg["exceeds_threshold"]:
            needing.append(bucket)

    data["buckets_needing_refresh"] = sorted(needing)
    data["taxonomy_refresh_recommended"] = bool(needing)
    data["n_docs"] = int(data.get("n_docs", 0)) + 1
    stages = list(data.get("stages_seen") or [])
    if stage not in stages:
        stages.append(stage)
    data["stages_seen"] = stages
    docs = list(data.get("docs") or [])
    docs.append(doc_id)
    data["docs"] = docs[-MONITOR_MAX_DOCS:]
    batches = list(data.get("batches") or [])
    batches.append(
        {
            "batch_id": batch_id or "unbatched",
            "doc_id": doc_id,
            "stage": stage,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "per_bucket": counts,
        }
    )
    data["batches"] = batches[-MONITOR_MAX_BATCHES:]
    data["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    _atomic_write_json(target, data)
    if needing:
        log.warning(
            "null-tag rate exceeds %.0f%% in bucket(s) %s: the taxonomy needs a "
            "v(N+1) refresh with new T19+ codes (MCAL_PLAN 6). Not a halt.",
            settings.NULL_TAG_REFRESH_THRESHOLD * 100, needing,
        )
    return data


def _atomic_write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            # allow_nan=False: an inf tau or a NaN composite must fail loudly here
            # rather than produce a file that json.loads in another language
            # rejects. `_finite()` is what keeps that from happening.
            json.dump(payload, fh, indent=2, allow_nan=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


# --- Doc-level rollup -------------------------------------------------------


def build_rollup(
    doc_id: str,
    stage: str,
    fields: dict[str, FieldGate],
    critic_results: dict[str, CriticResult],
    *,
    m1: Optional[dict],
    m2: Optional[dict],
    thresholds: dict[str, BucketThreshold],
    re_extract_audit: Optional[dict] = None,
    geocoder_stack: str = "full",
    entry_overrides: Optional[dict] = None,
) -> dict:
    """Doc-level roll-up written under the manifest's `_rollup` key."""
    gated = [g for g in fields.values() if g.gated_to_human]
    reasons: dict[str, int] = {}
    for g in fields.values():
        key = g.gate_reason or "none"
        reasons[key] = reasons.get(key, 0) + 1

    verdicts: dict[str, int] = {}
    for g in fields.values():
        verdicts[g.verdict] = verdicts.get(g.verdict, 0) + 1

    per_bucket: dict[str, dict] = {}
    for bucket in settings.BUCKET_ORDER:
        members = [g for g in fields.values() if g.bucket == bucket]
        th = thresholds.get(bucket)
        per_bucket[bucket] = {
            "n_fields": len(members),
            "n_gated": sum(1 for g in members if g.gated_to_human),
            "n_passed": sum(1 for g in members if not g.gated_to_human),
            "gate_reasons": _count([g.gate_reason for g in members if g.gate_reason]),
            "tau_deployed": _finite(th.tau_deployed) if th else None,
            "gate_all_to_human": bool(th.gate_all_to_human) if th else None,
            "degenerate": bool(th.degenerate) if th else None,
            "degenerate_severe": bool(th.degenerate_severe) if th else None,
            "N_wrong_docs": th.n_wrong_docs if th else None,
        }

    soi_entry = (entry_overrides or {}).get(
        settings.SUMMARY_OF_INTEREST,
        critic_mod.extracted_entry(settings.SUMMARY_OF_INTEREST, m1, m2),
    )
    return {
        "doc_id": doc_id,
        "artifact_stage": stage,
        "geocoder_stack": geocoder_stack,
        "n_fields": len(fields),
        "n_gated": len(gated),
        "n_passed": len(fields) - len(gated),
        "gate_rate": round(len(gated) / len(fields), 4) if fields else None,
        "gate_reasons": reasons,
        "verdicts": verdicts,
        "per_bucket": per_bucket,
        "null_tag": null_tag_counts(fields),
        "summary_of_interest": soi_diagnostics(
            soi_entry, m2, fields.get(settings.SUMMARY_OF_INTEREST)
        ),
        "critic": critic_mod.critic_diagnostics(critic_results),
        "re_extract": re_extract_audit or {},
    }


def _count(values: Sequence[Optional[str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        if v:
            out[v] = out.get(v, 0) + 1
    return out


# --- Manifest ---------------------------------------------------------------


@dataclass
class GateOutput:
    """Everything one document's gate pass produced."""

    doc_id: str
    stage: str
    fields: dict[str, FieldGate]
    rollup: dict
    null_tag_monitor: dict
    manifest: dict
    manifest_path: Optional[Path] = None
    re_extract: dict = dc_field(default_factory=dict)

    @property
    def n_gated(self) -> int:
        return sum(1 for g in self.fields.values() if g.gated_to_human)

    @property
    def n_passed(self) -> int:
        return len(self.fields) - self.n_gated

    def gated_fields(self) -> list[str]:
        return sorted(f for f, g in self.fields.items() if g.gated_to_human)


def manifest_dir(doc_id: str, out_dir: Optional[Path] = None) -> Path:
    return (Path(out_dir) if out_dir is not None else SEGMENT_B_OUTPUT) / doc_id


def build_manifest(
    doc_id: str,
    stage: str,
    fields: dict[str, FieldGate],
    rollup: dict,
    monitor: Optional[dict] = None,
) -> dict:
    """
    Assemble `run_manifest.json` (MCAL_PLAN 3.12).

    Top level is the plan's bare `{field: entry}` map; doc-level material sits
    under the reserved `_meta` / `_rollup` / `_null_tag_monitor` keys. Two
    alternatives were rejected: nesting fields under a `"fields"` key would break
    the literal schema a reviewer UI is written against, and emitting the rollup
    to a separate file would break "gradable without opening any other file".
    """
    manifest: dict = {f: g.to_manifest_entry() for f, g in sorted(fields.items())}
    manifest[META_KEY] = {
        "doc_id": doc_id,
        "artifact_stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan_ref": "MCAL_PLAN 3.12",
        "schema_keys": list(MANIFEST_REQUIRED_KEYS),
        "n_fields": len(fields),
        "reserved_keys": list(RESERVED_KEYS),
        "reviewer_note": (
            "Every field is present, gated or not, with its raw extraction "
            "(MCAL_PLAN 7 Q8). Grade `extracted_value` against `evidence_quote` "
            "and `source_pages`; `gate_reason` says why the field was routed to "
            "a human. `extracted_value: []` on summary_of_interest is a "
            "legitimate 'this document is routine'; `null` plus a gate_reason is "
            "a generation failure."
        ),
    }
    manifest[ROLLUP_KEY] = rollup
    if monitor is not None:
        manifest[MONITOR_KEY] = {
            "threshold": monitor.get("threshold"),
            "taxonomy_refresh_recommended": monitor.get("taxonomy_refresh_recommended"),
            "buckets_needing_refresh": monitor.get("buckets_needing_refresh"),
            "n_docs_rolling": monitor.get("n_docs"),
            "per_bucket": monitor.get("per_bucket"),
            "path": str(monitor.get("path") or monitor_path()),
            "halt_condition": False,
        }
    return manifest


def write_manifest(
    manifest: dict, doc_id: str, *, out_dir: Optional[Path] = None
) -> Path:
    """Write `<out_dir>/<doc_id>/run_manifest.json` atomically."""
    path = manifest_dir(doc_id, out_dir) / MANIFEST_FILENAME
    return _atomic_write_json(path, manifest)


# --- Top level --------------------------------------------------------------


def run_gate(
    doc_id: str,
    m1: Optional[dict],
    m2: Optional[dict],
    critic_results: dict[str, CriticResult],
    *,
    stage: Optional[str] = None,
    doc: Optional[Doc] = None,
    thresholds: Optional[dict[str, BucketThreshold]] = None,
    config: Optional[dict] = None,
    fields: Optional[Sequence[str]] = None,
    reextract: Optional[Callable] = None,
    call: Optional[Callable] = None,
    out_dir: Optional[Path] = None,
    write: bool = True,
    update_monitor: bool = True,
    monitor_file: Optional[Path] = None,
    batch_id: Optional[str] = None,
) -> GateOutput:
    """
    Gate every field of one document and emit its manifest (MCAL_PLAN 3.12).

    Iterates `settings.ALL_FIELDS` (not the keys of `critic_results`), so a field
    the Critic never reached is emitted with `gate_reason = "critic_missing"`
    rather than vanishing. MCAL_PLAN 7 Q8 and the seed-v1 acceptance criteria in
    6 both depend on the manifest being complete.

    `reextract` enables MCAL_PLAN 7 Q8's single retry: it is called once per
    RE_EXTRACT field with `(field, attempt=1, temperature=base+0.2, entry=...,
    critic_result=...)` and may return a `ReExtraction`, a dict of the same
    shape, a bare replacement entry, or None to decline. `doc` is required for
    the retry's Critic re-run and for `s_quote`; without it the composite falls
    back to the M1 default and `quote_verdict` is reported as null.
    """
    stage = resolve_stage(stage)
    cfg = config if config is not None else load_confidence_config(stage)
    ths = thresholds if thresholds is not None else load_bucket_thresholds(stage)
    geocoder_stack = str(cfg.get("geocoder_stack") or "full")
    field_list = tuple(fields) if fields else settings.ALL_FIELDS

    results: dict[str, CriticResult] = dict(critic_results or {})

    # MCAL_PLAN 7 Q8: one automated re-extraction attempt, re-run through the
    # Critic. Runs BEFORE gating so the gate sees the retry's verdict.
    re_audit: dict[str, dict] = {}
    re_entries: dict[str, Any] = {}
    if reextract is not None:
        retry = [
            f for f in field_list
            if (results.get(f) is not None and results[f].verdict == VERDICT_RE_EXTRACT)
        ]
        if retry:
            re_audit, re_entries = re_extract_fields(
                retry,
                reextract=reextract,
                doc=doc,
                m1=m1,
                m2=m2,
                critic_results=results,
                stage=stage,
                config=cfg,
                call=call,
            )

    gates: dict[str, FieldGate] = {}
    for field in field_list:
        bucket = settings.bucket_for_field(field)
        threshold = ths.get(bucket)
        if threshold is None:  # pragma: no cover - thresholds_from_payload guards this
            raise MissingArtifactError(
                f"No threshold for bucket {bucket!r} (field {field!r})."
            )
        # A field re-extracted under MCAL_PLAN 7 Q8 is gated on its REPLACEMENT
        # value, so the manifest shows what the surviving verdict was passed on.
        entry = re_entries.get(field, critic_mod.extracted_entry(field, m1, m2))
        gates[field] = gate_field(
            field,
            entry,
            results.get(field),
            threshold,
            stage=stage,
            doc=doc,
            geocoder_stack=geocoder_stack,
        )
        if field in re_audit:
            gates[field].re_extract = re_audit[field]

    rollup = build_rollup(
        doc_id, stage, gates, results,
        m1=m1, m2=m2, thresholds=ths,
        re_extract_audit=re_audit, geocoder_stack=geocoder_stack,
        entry_overrides=re_entries,
    )

    monitor: dict = {}
    if update_monitor:
        monitor = update_null_tag_monitor(
            gates, stage=stage, doc_id=doc_id,
            path=monitor_file, batch_id=batch_id,
        )
        monitor["path"] = str(monitor_file or monitor_path())

    manifest = build_manifest(
        doc_id, stage, gates, rollup, monitor if update_monitor else None
    )
    path: Optional[Path] = None
    if write:
        path = write_manifest(manifest, doc_id, out_dir=out_dir)
        log.info(
            "%s: %d/%d fields gated to HUMAN_REVIEW -> %s",
            doc_id, sum(1 for g in gates.values() if g.gated_to_human),
            len(gates), path,
        )

    return GateOutput(
        doc_id=doc_id,
        stage=stage,
        fields=gates,
        rollup=rollup,
        null_tag_monitor=monitor,
        manifest=manifest,
        manifest_path=path,
        re_extract=re_audit,
    )
