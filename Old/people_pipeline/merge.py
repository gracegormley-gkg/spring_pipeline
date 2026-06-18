"""
Merge per-chunk extractions into one row per (entity, stance) pair.

Per the user's design choice: if the same person/org takes the same stance
across multiple parts of the document, collapse those mentions into one row.
If their stance changes (e.g. an agency that conditionally supports in one
section and opposes in another), they get TWO rows — one per stance.

`sequence` is assigned by order of FIRST appearance (lowest chunk_index in
the merged group). The merged row keeps:
  - the strongest verbatim-verified quote as `summary_quote`
  - all evidence pages, deduped
  - a `mentions` list with every contributing extraction (for grading)
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict


def _normalize_entity_name(name: str) -> str:
    """Crude entity-name canonicalization for dedup.

    Rules:
      - lowercase, strip accents/punctuation, collapse whitespace
      - drop common honorifics ('mr', 'ms', 'mrs', 'dr', 'sen', 'rep', 'gov', 'hon')
      - drop trailing parentheticals like '(D-NM)' or '(Sierra Club)'
    This is intentionally conservative — over-merging is worse than under-merging
    because each row is hand-graded.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\([^)]*\)", " ", s)                # drop parenthetical
    s = re.sub(r"[^\w\s]", " ", s)                  # drop punctuation
    s = re.sub(
        r"\b(mr|ms|mrs|dr|sen|senator|rep|representative|gov|governor|hon|honorable|"
        r"the|a|an)\b",
        " ",
        s,
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pick_best_quote(rows: list[dict]) -> dict:
    """Among rows that share (entity, stance), pick the most useful exemplar.

    Preference order:
      1. quote_verified == True (verbatim found in doc)
      2. longer quote (more context for the grader)
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
    """
    Inputs: rows that came out of extract.py + verify.py (one row per chunk-mention).
    Output: one row per (entity, stance), sequence-numbered by first appearance.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in extracted_rows:
        key = (_normalize_entity_name(r.get("entity", "")), r.get("stance", ""))
        if not key[0] or not key[1]:
            continue
        groups[key].append(r)

    merged: list[dict] = []
    for (norm_entity, stance), group in groups.items():
        # Order group by chunk_index so first appearance is well-defined.
        group.sort(key=lambda r: (r.get("chunk_index", 0), r.get("entity", "")))
        first = group[0]
        best = _pick_best_quote(group)
        evidence_pages: list = []
        for r in group:
            evidence_pages.extend(r.get("evidence_pages") or [])
        evidence_pages = _dedup_pages(evidence_pages)

        # Display name: prefer the longest variant in the group (often more complete).
        display_name = max((r.get("entity", "") for r in group), key=len)

        # Role: same idea — most informative non-empty role wins.
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
                "stance_basis": r.get("stance_basis", ""),
                "entity_as_written": r.get("entity", ""),
                "role_as_written": r.get("role", ""),
            }
            for r in group
        ]

        # Summary attribution mode = mode that produced the chosen summary_quote;
        # also expose all distinct modes seen for this entity-stance pair so the
        # grader can tell whether the row mixes direct quotes with sectional hits.
        modes_seen = sorted({m["attribution_mode"] for m in mentions})

        merged.append({
            # `sequence` is filled in below after we sort by first appearance.
            "_first_chunk": first.get("chunk_index", 0),
            "entity": display_name,
            "kind": kind,
            "role": role,
            "stance": stance,
            "attribution_mode": best.get("attribution_mode", "direct_quote"),
            "attribution_modes_seen": modes_seen,
            "summary_quote": best.get("quote", ""),
            "summary_quote_verified": bool(best.get("quote_verified")),
            "evidence_pages": evidence_pages,
            "n_mentions": len(group),
            "mentions": mentions,
        })

    # Assign sequence by order of first appearance, ties broken alphabetically.
    merged.sort(key=lambda r: (r["_first_chunk"], r["entity"].lower(), r["stance"]))
    for i, r in enumerate(merged, start=1):
        r["sequence"] = i
        r.pop("_first_chunk", None)

    # Reorder keys for readability in the output JSON.
    ordered: list[dict] = []
    for r in merged:
        ordered.append({
            "sequence": r["sequence"],
            "entity": r["entity"],
            "kind": r["kind"],
            "role": r["role"],
            "stance": r["stance"],
            "attribution_mode": r["attribution_mode"],
            "attribution_modes_seen": r["attribution_modes_seen"],
            "summary_quote": r["summary_quote"],
            "summary_quote_verified": r["summary_quote_verified"],
            "evidence_pages": r["evidence_pages"],
            "n_mentions": r["n_mentions"],
            "mentions": r["mentions"],
        })
    return ordered
