"""
Stratified sample selection for the calibration run.

Targets from the v2 plan:
  - 5 short (<200 pp), 10 medium (200-800), 5 long (800+)
  - Mix of EIS types: Draft, Final, Supplemental, ROD
  - Mix of bureaus (no more than 4 from any one)

EIS type and bureau are derived cheaply from NUL title + contributor — no LLM
in the selection stage (calls go to M1 once docs are picked).
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from typing import Optional

from config import (
    EIS_TYPE_PATTERNS,
    LONG_MIN_PAGES,
    MAX_PER_BUREAU,
    N_LONG,
    N_MEDIUM,
    N_SHORT,
    RANDOM_SEED,
    SELECTION_PATH,
    SHORT_MAX_PAGES,
)
from chunk import estimated_pages
from nul import (
    extract_nul_metadata,
    fetch_collection_works,
    find_ocr,
    get_contributors,
    load_docs,
)

log = logging.getLogger(__name__)


def cheap_eis_type(title: str, ocr_head: str = "") -> str:
    """
    Best-effort EIS type for selection-time stratification.

    Try the NUL title first (cleanest signal when present); if that's Unknown,
    fall back to a regex on the first ~2 pages of OCR text. This raised the
    type-known rate from ~50% (title only) to ~87% on this collection.
    """
    if title:
        for label, pattern in EIS_TYPE_PATTERNS.items():
            if re.search(pattern, title, re.IGNORECASE):
                return label
    if ocr_head:
        for label, pattern in EIS_TYPE_PATTERNS.items():
            if re.search(pattern, ocr_head, re.IGNORECASE):
                return label
    return "Unknown"


def primary_bureau(contributors: list[str]) -> str:
    """First contributor label, normalized. 'Unknown' if empty."""
    if not contributors:
        return "Unknown"
    label = contributors[0]
    # Strip role suffix that NUL sometimes adds, e.g. "FOO (issuing body)"
    label = re.sub(r"\s*\([^)]+\)\s*$", "", label).strip()
    return label or "Unknown"


def build_candidate_pool() -> list[dict]:
    """Build candidate records joining NUL works to OCR text + cheap features."""
    works = fetch_collection_works()
    docs = load_docs()

    candidates: list[dict] = []
    for w in works:
        match = find_ocr(w, docs)
        if match is None:
            continue
        doc_id, text = match
        pages = estimated_pages(text)
        if pages < 1 or len(text.strip()) < 200:
            continue
        nul = extract_nul_metadata(w)
        title = nul.get("title") or ""
        if isinstance(title, list):
            title = title[0] if title else ""
        contributors = get_contributors(w)
        candidates.append({
            "work_id": w.get("id"),
            "doc_id": doc_id,
            "title": title,
            "estimated_pages": pages,
            "ocr_chars": len(text),
            "eis_type_guess": cheap_eis_type(title, text[:5000]),
            "bureau_guess": primary_bureau(contributors),
            "contributors": contributors,
        })
    log.info(f"Candidate pool: {len(candidates)} (NUL works with matched OCR)")
    return candidates


def _bucket(c: dict) -> str:
    p = c["estimated_pages"]
    if p < SHORT_MAX_PAGES:
        return "short"
    if p < LONG_MIN_PAGES:
        return "medium"
    return "long"


_TOTAL_TARGET = N_SHORT + N_MEDIUM + N_LONG  # 20


def _adjust_targets(by_bucket: dict[str, list[dict]]) -> dict[str, int]:
    """
    If a bucket has fewer docs than its target (e.g. only 3 long docs exist but
    we want 5), take all of them and redistribute the deficit to the next-most-
    flexible bucket (medium → short → long, in that order).
    """
    desired = {"short": N_SHORT, "medium": N_MEDIUM, "long": N_LONG}
    available = {b: len(by_bucket[b]) for b in desired}
    actual: dict[str, int] = {}
    deficit = 0
    for b in ("short", "medium", "long"):
        take = min(desired[b], available[b])
        actual[b] = take
        deficit += desired[b] - take

    if deficit > 0:
        # Spill order: try medium first (most balanced), then short, then long.
        spill_order = ["medium", "short", "long"]
        for b in spill_order:
            headroom = available[b] - actual[b]
            if headroom <= 0:
                continue
            give = min(headroom, deficit)
            actual[b] += give
            deficit -= give
            if deficit == 0:
                break
    return actual


def select(seed: int = RANDOM_SEED, write: bool = True) -> list[dict]:
    """
    Pick up to 20 docs satisfying the v2 plan's targets.

    Strategy:
      1. Bucket candidates by estimated page count (short/medium/long).
      2. Adjust bucket targets if any bucket is too small (e.g. only 3 long
         docs in the collection — take all 3, push 2 extras to medium).
      3. Shuffle each bucket (seeded), then greedily pick docs under:
           - per-type global cap: at most TYPE_CAP per EIS type across the
             whole selection (relaxed pass 2)
           - per-bureau cap: MAX_PER_BUREAU per bureau (relaxed pass 3)
    """
    candidates = build_candidate_pool()
    rng = random.Random(seed)

    by_bucket = {"short": [], "medium": [], "long": []}
    for c in candidates:
        by_bucket[_bucket(c)].append(c)
    for v in by_bucket.values():
        rng.shuffle(v)

    available = {b: len(v) for b, v in by_bucket.items()}
    targets = _adjust_targets(by_bucket)
    total_target = sum(targets.values())

    # Global type cap: ~ total_target / 3, rounded up. Keeps any single
    # type (likely "Draft" given the data) from taking more than ~7 slots
    # out of 20.
    type_cap = max(2, (total_target + 2) // 3)

    bureau_counter: Counter = Counter()
    type_counter: Counter = Counter()
    selected: list[dict] = []

    def take(cand: dict) -> None:
        selected.append(cand)
        bureau_counter[cand["bureau_guess"]] += 1
        type_counter[cand["eis_type_guess"]] += 1

    def count_in_bucket(b: str) -> int:
        return sum(1 for s in selected if _bucket(s) == b)

    for bucket in ("short", "medium", "long"):
        pool = by_bucket[bucket]
        target = targets[bucket]

        # Pass 1: enforce type cap AND bureau cap
        for cand in pool:
            if count_in_bucket(bucket) >= target:
                break
            if bureau_counter[cand["bureau_guess"]] >= MAX_PER_BUREAU:
                continue
            if type_counter[cand["eis_type_guess"]] >= type_cap:
                continue
            take(cand)

        # Pass 2: relax type cap (still enforce bureau cap)
        if count_in_bucket(bucket) < target:
            for cand in pool:
                if cand in selected:
                    continue
                if count_in_bucket(bucket) >= target:
                    break
                if bureau_counter[cand["bureau_guess"]] >= MAX_PER_BUREAU:
                    continue
                take(cand)

        # Pass 3: relax bureau cap
        if count_in_bucket(bucket) < target:
            for cand in pool:
                if cand in selected:
                    continue
                if count_in_bucket(bucket) >= target:
                    break
                take(cand)

    sel = {
        "seed": seed,
        "n_selected": len(selected),
        "plan_targets": {"short": N_SHORT, "medium": N_MEDIUM, "long": N_LONG},
        "adjusted_targets": targets,
        "available_per_bucket": available,
        "bucket_counts": {b: count_in_bucket(b) for b in by_bucket},
        "type_counts": dict(type_counter),
        "bureau_counts": dict(bureau_counter),
        "type_cap": type_cap,
        "max_per_bureau": MAX_PER_BUREAU,
        "selected": selected,
        "notes": _selection_notes(available, targets),
    }
    if write:
        SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SELECTION_PATH, "w") as f:
            json.dump(sel, f, indent=2, ensure_ascii=False)
        log.info(f"Wrote selection of {len(selected)} docs to {SELECTION_PATH}")
        for n in sel["notes"]:
            log.info(f"  note: {n}")
    return selected


def _selection_notes(available: dict[str, int], targets: dict[str, int]) -> list[str]:
    notes: list[str] = []
    for b, t in (("short", N_SHORT), ("medium", N_MEDIUM), ("long", N_LONG)):
        if available[b] < t:
            notes.append(
                f"{b}: only {available[b]} docs available (plan asked for {t}); "
                f"taking all {available[b]} and redistributing the deficit."
            )
    if targets["short"] + targets["medium"] + targets["long"] != _TOTAL_TARGET:
        notes.append(
            f"could not reach total target of {_TOTAL_TARGET}; "
            f"selecting {sum(targets.values())} docs."
        )
    return notes


def load_selection() -> Optional[list[dict]]:
    if not SELECTION_PATH.exists():
        return None
    with open(SELECTION_PATH) as f:
        return json.load(f)["selected"]
