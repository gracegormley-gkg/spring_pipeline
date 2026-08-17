"""
Confidence signals, composite scoring, and split-conformal thresholds
(MCAL_PLAN 3.3, build item #10).

Emits `thresholds.v(N).json`, `confidence_config.v(N).json`,
`weight_validation.v(N).json` and `gate_simulation.v(N).json`.
`segment_b/gate.py` reads the first two at run time.

--------------------------------------------------------------------------
The conformal construction, and why the plan's magic numbers are the right ones
--------------------------------------------------------------------------

Per bucket B:

  * Calibration set = only the docs with >=1 WRONG item in B. Size
    `N_wrong_docs(B)`. Restricting to wrong-item docs is what makes the
    guarantee a statement about wrong items rather than about items in general.
  * Per-doc nonconformity `R_doc = max{composite_i : i in doc, i in B, y_i = 0}`
    -- the score of the most confidently-wrong item in that document. Taking the
    max is what turns a per-item guarantee into a per-doc one, and is why the
    guarantee in `thresholds.json` is written in per-doc form.
  * Threshold = the k-th smallest `R_doc`, where `k = ceil((n+1)(1-alpha))`.
    Accept a new item iff `composite > tau`. By exchangeability,
    `P(R_new > tau) <= alpha`, i.e. the chance that a wrong item in an
    exchangeable document slips past the gate is at most alpha.

`k` can exceed `n`, in which case no finite threshold achieves the guarantee and
the only valid action is to gate everything. Working out where that happens:

    alpha = 0.15:  n = 6 -> k = ceil(7 * 0.85) = 6  feasible (exactly)
                   n = 5 -> k = ceil(6 * 0.85) = 6  INFEASIBLE
    alpha = 0.25:  n = 3 -> k = ceil(4 * 0.75) = 3  feasible (exactly)
                   n = 2 -> k = ceil(3 * 0.75) = 3  INFEASIBLE

So MCAL_PLAN 3.3's `N_wrong_docs < 6` degeneracy gate and `N_wrong_docs < 3`
severe gate are not heuristics -- they are precisely the feasibility boundaries
of the conformal quantile at alpha=0.15 and alpha=0.25. This module computes
feasibility directly and asserts it agrees with the plan's constants, so if
either alpha is ever renegotiated the gates move automatically instead of
silently becoming wrong.

A consequence worth stating plainly: at n = 6 and n = 3, `k = n`, so tau is the
MAXIMUM observed R_doc. Early-stage thresholds are therefore "beat every
wrong item we have ever seen", which is very conservative. That is correct
behaviour for a 8-doc calibration set, not a bug.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from . import grades as grades_mod
from . import settings

log = logging.getLogger(__name__)


# --- Signals ----------------------------------------------------------------


@dataclass
class Signals:
    """
    The six confidence signals (MCAL_PLAN 3.3). All in [0, 1].

    Only `s_quote` and `s_critic` carry weight at this stage; the rest are
    computed and stored at weight 0 so that weight validation has data to work
    with once n reaches ~60. Storing them now costs nothing and is the only way
    the later comparison is possible at all -- you cannot retro-fit a signal you
    never recorded.
    """

    s_quote: float = 0.0
    s_critic: float = 0.0
    s_source: Optional[float] = None
    s_citation: Optional[float] = None
    s_shard: Optional[float] = None
    s_acronym: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "s_quote": _r(self.s_quote),
            "s_critic": _r(self.s_critic),
            "s_source": _r(self.s_source),
            "s_citation": _r(self.s_citation),
            "s_shard": _r(self.s_shard),
            "s_acronym": _r(self.s_acronym),
        }

    def as_vector(self) -> dict[str, float]:
        """Non-None signals only, for weight-validation candidates."""
        return {
            k: v
            for k, v in {
                "s_quote": self.s_quote,
                "s_critic": self.s_critic,
                "s_source": self.s_source,
                "s_citation": self.s_citation,
                "s_shard": self.s_shard,
                "s_acronym": self.s_acronym,
            }.items()
            if v is not None
        }


def _r(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 6)


def s_quote_from_verdict(verdict: str) -> float:
    """quote_check verdict -> signal. {yes: 1.0, mixed: 0.5, no: 0.0}."""
    return settings.QUOTE_VERDICT_SCORES.get((verdict or "").lower(), 0.0)


def s_critic_from_verdict(verdict: str) -> float:
    """
    Critic verdict -> signal.

    An unrecognized verdict scores 0.0, matching segment_a/critic.py's coercion
    of unknown verdicts to HUMAN_REVIEW. Failing open here would let a malformed
    Critic response inflate confidence.
    """
    return settings.CRITIC_VERDICT_SCORES.get((verdict or "").upper(), 0.0)


def s_source_from_agreement(n_agree: int, n_total: int) -> Optional[float]:
    """
    M1 provenance agreement across {NUL/inventory, regex, Sonnet}
    -> {all: 1.0, 2/3: 0.5, disagree: 0.0}.
    """
    if not n_total:
        return None
    if n_agree >= n_total:
        return 1.0
    if n_total >= 3 and n_agree == 2:
        return 0.5
    if n_total == 2 and n_agree == 1:
        return 0.5
    return 0.0


def compute_signals(
    field: str,
    *,
    quote_verdict: Optional[str] = None,
    critic_verdict: str = "HUMAN_REVIEW",
    citation_rate: Optional[float] = None,
    shard_rate: Optional[float] = None,
    acronym_rate: Optional[float] = None,
    source_agreement: Optional[tuple[int, int]] = None,
) -> Signals:
    """
    Assemble the signal vector for one field.

    M1 fields (`year`, `eis_type`, `lead_agency`, `title`) have no verbatim
    quote in their extracted values by design, so `s_quote` defaults to 1.0 and
    the M1 composite collapses to `0.5*s_critic + 0.5` -- a 0.5 floor plus half
    the Critic verdict (MCAL_PLAN 3.3). That floor is deliberate but it does mean
    an M1 field can never score below 0.5, so an M1 bucket threshold above 0.5
    gates the entire bucket. `summarize()` flags that condition.
    """
    if field in settings.M1_FIELDS and quote_verdict is None:
        sq = settings.S_QUOTE_DEFAULT_M1
    else:
        sq = s_quote_from_verdict(quote_verdict or "no")

    return Signals(
        s_quote=sq,
        s_critic=s_critic_from_verdict(critic_verdict),
        s_source=(
            s_source_from_agreement(*source_agreement) if source_agreement else None
        ),
        s_citation=citation_rate,
        s_shard=shard_rate,
        s_acronym=acronym_rate,
    )


def composite(signals: Signals, weights: Optional[dict[str, float]] = None) -> float:
    """
    Weighted composite, clamped to [0, 1].

    Weights are renormalized over the signals that are actually present, so a
    field missing an optional signal is not penalized for its absence. With the
    frozen 0.5/0.5 weighting this is a no-op, since both weighted signals are
    always present.
    """
    w = weights or settings.SIGNAL_WEIGHTS
    vec = signals.as_vector()
    num = 0.0
    den = 0.0
    for name, weight in w.items():
        if weight == 0:
            continue
        if name in vec:
            num += weight * vec[name]
            den += weight
    if den == 0:
        return 0.0
    return max(0.0, min(1.0, num / den))


# --- Item records -----------------------------------------------------------


@dataclass
class ScoredItem:
    """One field of one doc, with its signals, composite, and human label."""

    doc_id: str
    field: str
    bucket: str
    signals: Signals
    composite: float
    y: Optional[int] = None  # 1 correct, 0 wrong, None ungraded
    failure_tags: list[str] = dc_field(default_factory=list)

    @property
    def is_wrong(self) -> bool:
        return self.y == 0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "field": self.field,
            "bucket": self.bucket,
            "signals": self.signals.to_dict(),
            "composite": _r(self.composite),
            "y": self.y,
            "failure_tags": self.failure_tags,
        }


# --- Conformal core ---------------------------------------------------------


def conformal_index(n: int, alpha: float) -> int:
    """k = ceil((n+1)(1-alpha)), the order statistic to use."""
    return math.ceil((n + 1) * (1.0 - alpha))


def is_feasible(n: int, alpha: float) -> bool:
    """Whether a finite threshold can achieve the guarantee at this n."""
    return n >= 1 and conformal_index(n, alpha) <= n


def conformal_quantile(scores: Sequence[float], alpha: float) -> Optional[float]:
    """
    The k-th smallest score, or None when infeasible.

    None means "no finite threshold achieves the guarantee" -- the caller must
    gate the bucket entirely rather than substituting a fallback value.
    """
    n = len(scores)
    if n == 0:
        return None
    k = conformal_index(n, alpha)
    if k > n:
        return None
    return sorted(scores)[k - 1]


def _tau_or_conservative(scores: Sequence[float], alpha: float) -> Optional[float]:
    """
    Conformal quantile, degrading to max(scores) when infeasible.

    Used ONLY inside leave-one-out slack estimation. Dropping a fold when its
    refit is infeasible would bias the slack downward exactly in the
    small-n regime where the slack matters most, so we substitute the most
    conservative achievable threshold (the max) and keep the fold.
    """
    if not scores:
        return None
    q = conformal_quantile(scores, alpha)
    return q if q is not None else max(scores)


@dataclass
class BucketThreshold:
    """Calibration result for one bucket. Serialized into thresholds.v(N).json."""

    bucket: str
    alpha: float
    alpha_effective: float
    n_wrong_docs: int
    n_items: int
    n_wrong_items: int
    tau_raw: Optional[float]
    curation_slack: float
    tau_deployed: float
    saturated: bool
    degenerate: bool
    degenerate_severe: bool
    gate_all_to_human: bool
    guarantee_conditioning: str
    loo_deltas: list[float] = dc_field(default_factory=list)
    wrong_docs: list[str] = dc_field(default_factory=list)
    r_docs: dict[str, float] = dc_field(default_factory=dict)
    notes: list[str] = dc_field(default_factory=list)

    def accepts(self, score: float) -> bool:
        """MCAL_PLAN 3.3: accept iff composite > tau_deployed."""
        if self.gate_all_to_human:
            return False
        return score > self.tau_deployed

    def to_dict(self) -> dict:
        return {
            "bucket": self.bucket,
            "alpha": self.alpha,
            "alpha_effective": self.alpha_effective,
            "N_wrong_docs": self.n_wrong_docs,
            "n_items": self.n_items,
            "n_wrong_items": self.n_wrong_items,
            "tau_raw": _r(self.tau_raw),
            "curation_slack": _r(self.curation_slack),
            "tau_deployed": _r(self.tau_deployed),
            "saturated": self.saturated,
            "guarantee_conditioning": self.guarantee_conditioning,
            "degenerate": self.degenerate,
            "degenerate_severe": self.degenerate_severe,
            "gate_all_to_human": self.gate_all_to_human,
            "loo_deltas": [_r(d) for d in self.loo_deltas],
            "wrong_docs": self.wrong_docs,
            "r_docs": {k: _r(v) for k, v in self.r_docs.items()},
            "notes": self.notes,
        }


GUARANTEE_TEMPLATE = (
    "P(exists i in doc : s_i > tau_{bucket} | y_i = 0, doc has >=1 wrong item "
    "in bucket {bucket}, doc exchangeable with Segment A) <= {alpha}. "
    "Distribution shift to full-corpus Segment B untested at this stage."
)

GATE_ALL_GUARANTEE = (
    "No finite threshold achieves the target error rate at this calibration "
    "size; bucket is gated entirely to HUMAN_REVIEW. No statistical guarantee "
    "is claimed or needed."
)


def calibrate_bucket(
    bucket: str,
    items: Sequence[ScoredItem],
    *,
    alpha: float = settings.ALPHA,
    alpha_degenerate: float = settings.ALPHA_EFFECTIVE_DEGENERATE,
) -> BucketThreshold:
    """
    Split-conformal calibration for one bucket (MCAL_PLAN 3.3).

    Degeneracy is decided by conformal FEASIBILITY, then cross-checked against
    the plan's `N_wrong_docs < 6` / `< 3` constants. They agree by construction
    (see the module docstring); a disagreement means alpha was changed without
    revisiting the gates, and is reported in `notes` rather than silently
    resolved either way.
    """
    notes: list[str] = []
    graded = [i for i in items if i.y is not None]
    wrong = [i for i in graded if i.is_wrong]

    # Per-doc nonconformity: the most confidently-wrong item in each doc.
    r_docs: dict[str, float] = {}
    for it in wrong:
        prev = r_docs.get(it.doc_id)
        if prev is None or it.composite > prev:
            r_docs[it.doc_id] = it.composite

    n = len(r_docs)
    wrong_docs = sorted(r_docs)
    scores = [r_docs[d] for d in wrong_docs]

    # --- degeneracy ---
    feasible_at_alpha = is_feasible(n, alpha)
    alpha_eff = alpha
    degenerate = False
    degenerate_severe = False

    if not feasible_at_alpha:
        degenerate = True
        alpha_eff = alpha_degenerate
        if not is_feasible(n, alpha_eff):
            degenerate_severe = True

    # Cross-check against the plan's hard-coded gates.
    plan_degenerate = n < settings.DEGENERATE_MIN_WRONG_DOCS
    plan_severe = n < settings.DEGENERATE_SEVERE_MIN_WRONG_DOCS
    if plan_degenerate != degenerate or plan_severe != degenerate_severe:
        notes.append(
            f"Feasibility-derived degeneracy (degenerate={degenerate}, "
            f"severe={degenerate_severe}) disagrees with MCAL_PLAN's "
            f"N_wrong_docs gates (degenerate={plan_degenerate}, "
            f"severe={plan_severe}) at n={n}, alpha={alpha}. The plan's "
            f"constants assume alpha=0.15/0.25; if alpha changed, update "
            f"DEGENERATE_MIN_WRONG_DOCS / DEGENERATE_SEVERE_MIN_WRONG_DOCS."
        )

    if degenerate and not degenerate_severe:
        notes.append(
            f"n={n} wrong docs is infeasible at alpha={alpha} "
            f"(needs the {conformal_index(n, alpha)}-th of {n} order "
            f"statistics); relaxed to alpha_effective={alpha_eff}."
        )

    # --- severe: gate everything ---
    if degenerate_severe or n == 0:
        if n == 0:
            notes.append(
                "No graded wrong items in this bucket, so there is nothing to "
                "calibrate against. Gating to HUMAN_REVIEW is the only sound "
                "action -- an empty calibration set is not evidence of quality."
            )
        return BucketThreshold(
            bucket=bucket,
            alpha=alpha,
            alpha_effective=alpha_eff,
            n_wrong_docs=n,
            n_items=len(graded),
            n_wrong_items=len(wrong),
            tau_raw=None,
            curation_slack=0.0,
            tau_deployed=float("inf"),
            saturated=False,
            degenerate=True,
            degenerate_severe=True,
            gate_all_to_human=True,
            guarantee_conditioning=GATE_ALL_GUARANTEE,
            wrong_docs=wrong_docs,
            r_docs=r_docs,
            notes=notes,
        )

    # --- tau_raw ---
    tau_raw = conformal_quantile(scores, alpha_eff)
    assert tau_raw is not None  # feasibility already established
    k = conformal_index(n, alpha_eff)
    if k == n:
        notes.append(
            f"k == n == {n}: tau is the MAXIMUM observed wrong-item score, so "
            f"acceptance requires beating every wrong item in the calibration "
            f"set. Expected and correct at this sample size."
        )

    # --- curation slack: leave-one-doc-out over the wrong-item docs only ---
    # max(delta) rather than the 95th percentile, per MCAL_PLAN 3.3: at n of
    # 2-8 the percentile is dominated by the discreteness of the empirical
    # quantile and is not meaningfully estimable.
    loo_deltas: list[float] = []
    if n >= 2:
        for i in range(n):
            held_out = scores[:i] + scores[i + 1 :]
            tau_i = _tau_or_conservative(held_out, alpha_eff)
            if tau_i is not None:
                loo_deltas.append(abs(tau_raw - tau_i))
    curation_slack = max(loo_deltas) if loo_deltas else 0.0
    if n < 2:
        notes.append(
            "n < 2: leave-one-out slack is undefined, so curation_slack = 0. "
            "The threshold is unbuffered against a single mis-grade."
        )

    # --- tau_deployed ---
    tau_unclamped = tau_raw + curation_slack
    tau_deployed = min(1.0, tau_unclamped)
    saturated = tau_unclamped >= 1.0
    if saturated:
        notes.append(
            f"tau_raw + curation_slack = {tau_unclamped:.4f} >= 1.0, clamped to "
            f"1.0. Since acceptance requires composite > tau, a clamped "
            f"threshold rejects everything -- functionally equivalent to "
            f"gate_all_to_human for this bucket."
        )

    return BucketThreshold(
        bucket=bucket,
        alpha=alpha,
        alpha_effective=alpha_eff,
        n_wrong_docs=n,
        n_items=len(graded),
        n_wrong_items=len(wrong),
        tau_raw=tau_raw,
        curation_slack=curation_slack,
        tau_deployed=tau_deployed,
        saturated=saturated,
        degenerate=degenerate,
        degenerate_severe=False,
        gate_all_to_human=False,
        guarantee_conditioning=GUARANTEE_TEMPLATE.format(
            bucket=bucket, alpha=alpha_eff
        ),
        loo_deltas=loo_deltas,
        wrong_docs=wrong_docs,
        r_docs=r_docs,
        notes=notes,
    )


def calibrate_all(
    items: Sequence[ScoredItem],
    *,
    alpha: float = settings.ALPHA,
    force_gate: Iterable[str] = (),
) -> dict[str, BucketThreshold]:
    """
    Calibrate every bucket.

    `force_gate` names buckets to gate regardless of their statistics -- used for
    the location bucket when the geocoder stack is in reduced mode
    (MCAL_PLAN 3.7 / 3.9a), where the extractor itself is known-degraded and no
    threshold on its output would be meaningful.
    """
    forced = set(force_gate)
    out: dict[str, BucketThreshold] = {}
    for bucket in settings.BUCKET_ORDER:
        bucket_items = [i for i in items if i.bucket == bucket]
        th = calibrate_bucket(bucket, bucket_items, alpha=alpha)
        if bucket in forced and not th.gate_all_to_human:
            th.gate_all_to_human = True
            th.tau_deployed = float("inf")
            th.guarantee_conditioning = GATE_ALL_GUARANTEE
            th.notes.append(
                "Forced to gate_all_to_human by the build (reduced geocoder "
                "stack). The threshold statistics above are retained for "
                "reference but are not in force."
            )
        out[bucket] = th
    return out


# --- Gate simulation --------------------------------------------------------


def simulate_gate(
    items: Sequence[ScoredItem], thresholds: dict[str, BucketThreshold]
) -> dict:
    """
    Per-bucket {gate rate, caught-error rate, false-defer rate} against the
    current graded set (MCAL_PLAN 2, `gate_simulation.v(N).json`).

    This is an IN-SAMPLE simulation -- the same items fitted the thresholds -- so
    caught-error rate is optimistic by construction and is reported as a
    sanity check, not as performance. Stated in the output so the number is not
    mistaken for a held-out estimate.
    """
    per_bucket: dict[str, dict] = {}
    for bucket, th in thresholds.items():
        bucket_items = [i for i in items if i.bucket == bucket and i.y is not None]
        n = len(bucket_items)
        if n == 0:
            per_bucket[bucket] = {
                "n_graded_items": 0,
                "gate_rate": 1.0 if th.gate_all_to_human else None,
                "caught_error_rate": None,
                "false_defer_rate": None,
                "note": "no graded items in this bucket",
            }
            continue

        gated = [i for i in bucket_items if not th.accepts(i.composite)]
        wrong = [i for i in bucket_items if i.is_wrong]
        correct = [i for i in bucket_items if i.y == 1]
        caught = [i for i in wrong if not th.accepts(i.composite)]
        false_defer = [i for i in correct if not th.accepts(i.composite)]

        per_bucket[bucket] = {
            "n_graded_items": n,
            "n_wrong": len(wrong),
            "n_correct": len(correct),
            "gate_rate": round(len(gated) / n, 4),
            "caught_error_rate": (
                round(len(caught) / len(wrong), 4) if wrong else None
            ),
            "false_defer_rate": (
                round(len(false_defer) / len(correct), 4) if correct else None
            ),
            "tau_deployed": _r(th.tau_deployed),
            "gate_all_to_human": th.gate_all_to_human,
        }

    total = [i for i in items if i.y is not None]
    n_gated = sum(
        1
        for i in total
        if not thresholds[i.bucket].accepts(i.composite)
    )
    return {
        "in_sample": True,
        "caveat": (
            "In-sample: these items fitted the thresholds, so caught_error_rate "
            "is optimistic and false_defer_rate is pessimistic. Not a held-out "
            "estimate. Use as a sanity check on gate volume only."
        ),
        "overall_gate_rate": round(n_gated / len(total), 4) if total else None,
        "n_graded_items": len(total),
        "per_bucket": per_bucket,
    }


def empirical_coverage(
    items: Sequence[ScoredItem], thresholds: dict[str, BucketThreshold]
) -> dict:
    """
    Fraction of wrong-item docs whose R_doc is correctly caught, per bucket.

    Should be >= 1-alpha by construction (MCAL_PLAN 6 asks for it to be reported
    anyway). A value below 1-alpha indicates an implementation error, not a
    statistical fluctuation, because the threshold was fitted on these very
    scores.
    """
    out: dict[str, dict] = {}
    for bucket, th in thresholds.items():
        if th.gate_all_to_human:
            out[bucket] = {"coverage": 1.0, "target": None, "gated": True}
            continue
        r = list(th.r_docs.values())
        if not r:
            out[bucket] = {"coverage": None, "target": None, "gated": False}
            continue
        caught = sum(1 for v in r if not th.accepts(v))
        cov = caught / len(r)
        target = 1.0 - th.alpha_effective
        out[bucket] = {
            "coverage": round(cov, 4),
            "target": round(target, 4),
            "meets_target": cov >= target,
            "n_wrong_docs": len(r),
            "gated": False,
        }
    return out


# --- Weight validation (diagnostic only) ------------------------------------


def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """
    AUROC via the Mann-Whitney U statistic, with ties at 0.5.

    Returns None when one class is absent. Hand-rolled rather than pulled from
    sklearn: this is the only place we would need sklearn, and the tie handling
    matters here because composite scores are heavily tied at the 0.5 M1 floor
    and at the {0.0, 0.5, 0.7, 1.0} Critic-verdict grid.
    """
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Kendall tau-b (tie-corrected). None if undefined."""
    n = len(xs)
    if n < 2:
        return None
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            p = dx * dy
            if p > 0:
                conc += 1
            elif p < 0:
                disc += 1
            else:
                if dx == 0:
                    tx += 1
                if dy == 0:
                    ty += 1
    d = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / d if d else None


CANDIDATE_WEIGHTINGS: dict[str, dict[str, float]] = {
    # The frozen production weighting.
    "quote50_critic50": {"s_quote": 0.5, "s_critic": 0.5},
    "quote_only": {"s_quote": 1.0},
    "critic_only": {"s_critic": 1.0},
    "quote70_critic30": {"s_quote": 0.7, "s_critic": 0.3},
    "quote30_critic70": {"s_quote": 0.3, "s_critic": 0.7},
    "quote40_critic40_citation20": {
        "s_quote": 0.4,
        "s_critic": 0.4,
        "s_citation": 0.2,
    },
    "all_equal": {
        "s_quote": 1.0,
        "s_critic": 1.0,
        "s_source": 1.0,
        "s_citation": 1.0,
        "s_shard": 1.0,
        "s_acronym": 1.0,
    },
}


def validate_weights(
    items: Sequence[ScoredItem],
    *,
    n_bootstrap: int = 1000,
    seed: int = 20260525,
) -> dict:
    """
    Doc-stratified paired bootstrap comparison of candidate weightings
    (MCAL_PLAN 3.3, `weight_validation.v(N).json`).

    **Not a decision gate.** At n well under
    `settings.WEIGHT_VALIDATION_MIN_N` the confidence intervals are far too wide
    for any candidate to be meaningfully dominated, and weights are frozen at
    0.5/0.5 through at least stage v3 regardless of what this reports
    (MCAL_PLAN 7.5). Reported for the record so the eventual unfreezing decision
    has a history to look at.

    Resampling is by DOCUMENT, not by item: items within a document share an
    extraction run and a set of cited pages, so item-level resampling would
    treat correlated observations as independent and understate the intervals.
    """
    graded = [i for i in items if i.y is not None]
    doc_ids = sorted({i.doc_id for i in graded})
    n_docs = len(doc_ids)

    tiebreak_note = (
        "Tiebreaker (from the module docstring): prefer the candidate with the "
        "fewest signals at non-zero weight; break remaining ties "
        "lexicographically by candidate name."
    )

    if n_docs < 2 or not graded:
        return {
            "gating": False,
            "n_docs": n_docs,
            "n_items": len(graded),
            "advisory_only": True,
            "reason": "too few documents to bootstrap",
            "tiebreaker": tiebreak_note,
            "candidates": {},
        }

    by_doc: dict[str, list[ScoredItem]] = {}
    for it in graded:
        by_doc.setdefault(it.doc_id, []).append(it)

    def score_all(weights: dict[str, float], subset: Sequence[ScoredItem]):
        return [composite(i.signals, weights) for i in subset]

    rng = random.Random(seed)
    results: dict[str, dict] = {}

    # Point estimates on the full set.
    point: dict[str, Optional[float]] = {}
    for name, w in CANDIDATE_WEIGHTINGS.items():
        point[name] = auroc(score_all(w, graded), [i.y for i in graded])

    # Doc-stratified bootstrap.
    draws: dict[str, list[float]] = {k: [] for k in CANDIDATE_WEIGHTINGS}
    for _ in range(n_bootstrap):
        sampled_docs = [rng.choice(doc_ids) for _ in range(n_docs)]
        subset: list[ScoredItem] = []
        for d in sampled_docs:
            subset.extend(by_doc[d])
        labels = [i.y for i in subset]
        if len(set(labels)) < 2:
            continue
        for name, w in CANDIDATE_WEIGHTINGS.items():
            a = auroc(score_all(w, subset), labels)
            if a is not None:
                draws[name].append(a)

    for name in CANDIDATE_WEIGHTINGS:
        d = sorted(draws[name])
        if d:
            lo = d[int(0.025 * len(d))]
            hi = d[min(len(d) - 1, int(0.975 * len(d)))]
        else:
            lo = hi = None
        results[name] = {
            "n_signals_weighted": sum(
                1 for v in CANDIDATE_WEIGHTINGS[name].values() if v
            ),
            "auroc": _r(point[name]),
            "auroc_ci95": [_r(lo), _r(hi)],
            "n_bootstrap_valid": len(d),
        }

    # Kendall tau of the production composite vs correctness, for reference.
    prod = score_all(settings.SIGNAL_WEIGHTS, graded)
    tau = kendall_tau(prod, [float(i.y) for i in graded])

    return {
        "gating": False,
        "advisory_only": True,
        "reason": (
            f"n_docs={n_docs} is below WEIGHT_VALIDATION_MIN_N="
            f"{settings.WEIGHT_VALIDATION_MIN_N}; intervals are too wide to "
            f"dominate any candidate. Weights stay frozen at 0.5/0.5 through "
            f"at least v3 per MCAL_PLAN 7.5."
        ),
        "n_docs": n_docs,
        "n_items": len(graded),
        "n_bootstrap": n_bootstrap,
        "resampling_unit": "document",
        "production_weighting": "quote50_critic50",
        "production_kendall_tau": _r(tau),
        "tiebreaker": tiebreak_note,
        "candidates": results,
    }


# --- Config artifact --------------------------------------------------------


def build_confidence_config(*, geocoder_stack: str = "full") -> dict:
    """The `confidence_config.v(N).json` payload consumed by gate.py/critic.py."""
    return {
        "signals": [
            {"name": name, "weight": weight}
            for name, weight in settings.SIGNAL_WEIGHTS.items()
        ],
        "weights_frozen_until_stage": "v3",
        "per_field_overrides": {},
        "dependent_fields": settings.DEPENDENT_FIELDS,
        "geocoder_stack": geocoder_stack,
        "judge_model_by_field": settings.default_judge_model_map(),
        "s_quote_default_m1": settings.S_QUOTE_DEFAULT_M1,
        "critic_verdict_scores": settings.CRITIC_VERDICT_SCORES,
        "quote_verdict_scores": settings.QUOTE_VERDICT_SCORES,
        "quote_check": {
            "page_tolerance": settings.QUOTE_PAGE_TOLERANCE,
            "ratio_yes": settings.QUOTE_RATIO_YES,
            "ratio_mixed": settings.QUOTE_RATIO_MIXED,
            "coverage_yes": settings.QUOTE_COVERAGE_YES,
            "coverage_mixed": settings.QUOTE_COVERAGE_MIXED,
        },
        "buckets": settings.BUCKETS,
    }


# --- Persistence ------------------------------------------------------------


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def save_thresholds(
    thresholds: dict[str, BucketThreshold], stage: str, *, draft: bool = True
) -> Path:
    payload = {
        "version": settings.normalize_stage(stage),
        "alpha": settings.ALPHA,
        "alpha_effective_degenerate": settings.ALPHA_EFFECTIVE_DEGENERATE,
        "accept_rule": "accept iff composite > tau_deployed",
        "buckets": {b: t.to_dict() for b, t in thresholds.items()},
    }
    return _write(
        settings.artifact_path("thresholds.json", stage, draft=draft), payload
    )


def save_confidence_config(
    stage: str, *, draft: bool = True, geocoder_stack: str = "full"
) -> Path:
    payload = build_confidence_config(geocoder_stack=geocoder_stack)
    payload["version"] = settings.normalize_stage(stage)
    return _write(
        settings.artifact_path("confidence_config.json", stage, draft=draft), payload
    )


def load_thresholds(stage: str) -> dict:
    path = settings.artifact_path("thresholds.json", stage, draft=False)
    if not path.exists():
        raise FileNotFoundError(
            f"No thresholds for stage {stage}: {path}. Run "
            f"`python -m mcal.build --stage {stage}` and ratify the draft."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(
    thresholds: dict[str, BucketThreshold], items: Sequence[ScoredItem]
) -> dict:
    """Roll-up for `calibration_report.v(N).md` and seed-v1 acceptance checks."""
    n_degenerate = sum(1 for t in thresholds.values() if t.degenerate)
    n_severe = sum(1 for t in thresholds.values() if t.degenerate_severe)
    original_non_degenerate = [
        b
        for b in settings.ORIGINAL_BUCKETS
        if not thresholds[b].degenerate
    ]

    warnings: list[str] = []
    # The M1 composite floor: 0.5*s_critic + 0.5 can never fall below 0.5, so a
    # threshold at or above 0.5 gates the whole bucket regardless of statistics.
    m1 = thresholds.get("M1")
    if m1 and not m1.gate_all_to_human and m1.tau_deployed >= 0.5:
        warnings.append(
            f"M1 tau_deployed={m1.tau_deployed:.3f} >= 0.5, but the M1 composite "
            f"has a hard 0.5 floor (s_quote defaults to 1.0 for fields with no "
            f"verbatim quote). Every M1 field with a PASS verdict scores exactly "
            f"1.0 and everything else scores <= 0.85, so this threshold is "
            f"effectively 'PASS only'. Not wrong, but worth knowing."
        )

    smallest_non_empty = min(
        (t.n_wrong_docs for t in thresholds.values() if t.n_wrong_docs > 0),
        default=0,
    )
    return {
        "n_buckets": len(thresholds),
        "n_degenerate": n_degenerate,
        "n_degenerate_severe": n_severe,
        "n_gate_all_to_human": sum(
            1 for t in thresholds.values() if t.gate_all_to_human
        ),
        "original_buckets_non_degenerate": original_non_degenerate,
        "n_original_non_degenerate": len(original_non_degenerate),
        # MCAL_PLAN 6 v2+ acceptance criterion 3.
        "meets_v2_criterion_4_of_6": len(original_non_degenerate) >= 4,
        "smallest_non_empty_n_wrong_docs": smallest_non_empty,
        # MCAL_PLAN 6 / 7 Q1 full-scale unlock.
        "full_scale_unlocked": smallest_non_empty
        >= settings.FULL_SCALE_MIN_WRONG_DOCS,
        "full_scale_threshold": settings.FULL_SCALE_MIN_WRONG_DOCS,
        "warnings": warnings,
        "per_bucket": {
            b: {
                "N_wrong_docs": t.n_wrong_docs,
                "tau_deployed": None if t.gate_all_to_human else _r(t.tau_deployed),
                "degenerate": t.degenerate,
                "degenerate_severe": t.degenerate_severe,
                "gate_all_to_human": t.gate_all_to_human,
                "saturated": t.saturated,
                "curation_slack": _r(t.curation_slack),
            }
            for b, t in thresholds.items()
        },
    }
