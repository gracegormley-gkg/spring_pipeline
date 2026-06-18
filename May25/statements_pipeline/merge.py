"""
Merge per-chunk extractions into one row per ENTITY.

Differs from `people_pipeline/merge.py` in that we do not key on stance —
stance hasn't been determined yet at this point. find_statement decides
stance off the full statement (or paraphrase block) downstream.

Same entity reaching the doc twice with two genuinely different stances
(rare but real — e.g. an agency that conditionally supports one alternative
and opposes another) collapses to one row here. find_statement gets the
window around all the evidence pages and chooses one stance for the whole
row, with a confidence label that downgrades if it had to reconcile.

This file SHADOWS people_pipeline/merge.py via the same sys.path-append
trick used for extract.

Inputs are post-verify rows with `quote_verified` set. Output is one merged
row per entity, sequence-numbered by first appearance.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict


def _normalize_entity_name(name: str) -> str:
    """Crude entity-name canonicalization for dedup.

    - lowercase, strip accents/punctuation, collapse whitespace
    - drop common honorifics
    - drop trailing parentheticals like '(D-NM)'

    Conservative; over-merging is worse than under-merging because each row
    becomes its own LLM call downstream.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(
        r"\b(mr|ms|mrs|dr|sen|senator|rep|representative|gov|governor|hon|honorable|"
        r"the|a|an)\b",
        " ",
        s,
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pick_best_quote(rows: list[dict]) -> dict:
    """Pick the most useful exemplar quote within an entity group.

    Preference: verbatim-verified > longer.
    """
    verified = [r for r in rows if r.get("quote_verified")]
    pool = verified or rows
    return max(pool, key=lambda r: len(r.get("quote") or ""))


def _dedup_pages(spans: list) -> list:
    out: list = []
    seen = set()
    for s in spans:
        if s is None:
            continue
        s = str(s).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def merge_rows(extracted_rows: list[dict]) -> list[dict]:
    """One row per normalized entity, sequence-numbered by first appearance.

    No stance fields here — stance is added downstream by find_statement.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in extracted_rows:
        key = _normalize_entity_name(r.get("entity", ""))
        if not key:
            continue
        groups[key].append(r)

    merged: list[dict] = []
    for norm_entity, group in groups.items():
        group.sort(key=lambda r: (r.get("chunk_index", 0), r.get("entity", "")))
        first = group[0]
        best = _pick_best_quote(group)

        evidence_pages: list = []
        for r in group:
            evidence_pages.extend(r.get("evidence_pages") or [])
        evidence_pages = _dedup_pages(evidence_pages)

        # Display name: prefer the longest variant in the group (often the most
        # complete / most formal version of the name).
        display_name = max((r.get("entity", "") for r in group), key=len)

        # Role: most informative non-empty role wins.
        roles = [r.get("role", "") for r in group if r.get("role")]
        role = max(roles, key=len) if roles else ""

        # Kind: majority vote, ties broken by group order.
        kind_counts: dict[str, int] = {}
        for r in group:
            k = r.get("kind", "other")
            kind_counts[k] = kind_counts.get(k, 0) + 1
        kind = max(kind_counts.items(), key=lambda kv: kv[1])[0]

        mentions = [
            {
                "chunk_index": r.get("chunk_index"),
                "evidence_pages": r.get("evidence_pages") or [],
                "attribution_mode": r.get("attribution_mode", "direct_quote"),
                "quote": r.get("quote", ""),
                "quote_verified": bool(r.get("quote_verified")),
                "entity_as_written": r.get("entity", ""),
                "role_as_written": r.get("role", ""),
            }
            for r in group
        ]
        modes_seen = sorted({m["attribution_mode"] for m in mentions})

        merged.append({
            "_first_chunk": first.get("chunk_index", 0),
            "entity": display_name,
            "kind": kind,
            "role": role,
            "attribution_mode": best.get("attribution_mode", "direct_quote"),
            "attribution_modes_seen": modes_seen,
            "summary_quote": best.get("quote", ""),
            "summary_quote_verified": bool(best.get("quote_verified")),
            "evidence_pages": evidence_pages,
            "n_mentions": len(group),
            "mentions": mentions,
        })

    # Stable sequence by first appearance, ties broken alphabetically.
    merged.sort(key=lambda r: (r["_first_chunk"], r["entity"].lower()))
    for i, r in enumerate(merged, start=1):
        r["sequence"] = i
        r.pop("_first_chunk", None)

    # Re-emit with a stable key order.
    ordered: list[dict] = []
    for r in merged:
        ordered.append({
            "sequence": r["sequence"],
            "entity": r["entity"],
            "kind": r["kind"],
            "role": r["role"],
            "attribution_mode": r["attribution_mode"],
            "attribution_modes_seen": r["attribution_modes_seen"],
            "summary_quote": r["summary_quote"],
            "summary_quote_verified": r["summary_quote_verified"],
            "evidence_pages": r["evidence_pages"],
            "n_mentions": r["n_mentions"],
            "mentions": r["mentions"],
        })
    return ordered
