"""
Per-doc output writer for the statements pipeline.

Lays out files as:

    output/people/<doc_id>/
    ├── index.json                  # doc metadata, counts, list of person files
    ├── 001_sierra_club.json
    ├── 002_john_smith.json
    └── ...

One person file per (entity, stance) row. Filename = NNN_slug.json where NNN is
the zero-padded sequence number and slug is derived from the entity name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import settings


def _slug(name: str) -> str:
    """Lowercase, alphanumeric+underscore slug for filenames. Max 40 chars."""
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if not s:
        s = "anon"
    return s[:40]


def _person_filename(seq: int, entity: str) -> str:
    return f"{int(seq):03d}_{_slug(entity)}.json"


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _needs_human_review(row: dict) -> tuple[bool, list[str]]:
    """Cheap rules-based review flag (no LLM critic in this pipeline).

    Reasons:
      - kind == 'individual': private-individual stances always go to a human
        (matches the v2 policy used by people_pipeline's critic).
      - summary_quote_verified == False: the exemplar quote isn't verbatim in
        the doc, so we can't trust the source line without a human look.
    """
    reasons: list[str] = []
    if row.get("kind") == "individual":
        reasons.append("private_individual")
    if not row.get("summary_quote_verified"):
        reasons.append("quote_not_verbatim")
    return (len(reasons) > 0), reasons


def _person_record(row: dict, doc_id: str, work_id: Optional[str]) -> dict:
    """Shape one row into the per-person JSON record."""
    needs_review, reasons = _needs_human_review(row)
    return {
        "sequence": row.get("sequence"),
        "doc_id": doc_id,
        "work_id": work_id,
        "entity": row.get("entity"),
        "kind": row.get("kind"),
        "role": row.get("role"),
        "stance": row.get("stance"),
        "summary": row.get("summary", ""),
        "statement": row.get("statement"),
        "needs_human_review": needs_review,
        "human_review_reasons": reasons,
        "evidence_pages": row.get("evidence_pages"),
        "summary_quote": row.get("summary_quote"),
        "summary_quote_verified": row.get("summary_quote_verified"),
        "attribution_mode": row.get("attribution_mode"),
        "attribution_modes_seen": row.get("attribution_modes_seen"),
        "n_mentions": row.get("n_mentions"),
        "mentions": row.get("mentions"),
    }


def _statement_counts(rows: list[dict]) -> dict:
    out = {f: 0 for f in settings.STATEMENT_FORMS}
    out["present"] = 0
    out["absent"] = 0
    for r in rows:
        s = r.get("statement") or {}
        form = s.get("form") or "none"
        out[form] = out.get(form, 0) + 1
        if s.get("present"):
            out["present"] += 1
        else:
            out["absent"] += 1
    return out


def _stance_counts(rows: list[dict]) -> dict:
    counts: dict = {}
    for r in rows:
        s = r.get("stance", "?")
        counts[s] = counts.get(s, 0) + 1
    return counts


def _review_counts(rows: list[dict]) -> dict:
    counts = {"needs_review": 0, "auto_ok": 0}
    reason_counts: dict[str, int] = {}
    for r in rows:
        needs, reasons = _needs_human_review(r)
        if needs:
            counts["needs_review"] += 1
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        else:
            counts["auto_ok"] += 1
    counts["reasons"] = reason_counts
    return counts


def write_doc(
    doc_id: str,
    work_id: Optional[str],
    title: str,
    n_pages: int,
    n_chunks: int,
    n_raw_rows: int,
    rows: list[dict],
    elapsed_sec: float,
    usage_summary: dict,
) -> dict:
    """Write per-person files and index.json for one doc.

    Returns a small summary dict (counts + paths) for the pipeline-level run summary.
    """
    doc_dir = settings.PEOPLE_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    # Stable order: by sequence.
    rows_sorted = sorted(rows, key=lambda r: r.get("sequence", 10**9))

    written: list[dict] = []
    for r in rows_sorted:
        seq = r.get("sequence")
        if seq is None:
            continue
        filename = _person_filename(seq, r.get("entity", ""))
        _write_json(doc_dir / filename, _person_record(r, doc_id, work_id))
        needs, reasons = _needs_human_review(r)
        written.append({
            "sequence": seq,
            "file": filename,
            "entity": r.get("entity"),
            "kind": r.get("kind"),
            "role": r.get("role"),
            "stance": r.get("stance"),
            "statement_present": (r.get("statement") or {}).get("present", False),
            "statement_form": (r.get("statement") or {}).get("form"),
            "needs_human_review": needs,
            "human_review_reasons": reasons,
        })

    stance_counts = _stance_counts(rows_sorted)
    review_counts = _review_counts(rows_sorted)
    statement_counts = _statement_counts(rows_sorted)

    index = {
        "doc_id": doc_id,
        "work_id": work_id,
        "title": title,
        "n_pages": n_pages,
        "n_chunks": n_chunks,
        "n_raw_rows": n_raw_rows,
        "n_people": len(written),
        "stance_counts": stance_counts,
        "review_counts": review_counts,
        "statement_counts": statement_counts,
        "elapsed_sec": elapsed_sec,
        "usage": usage_summary,
        "schema": {
            "stance_vocabulary": list(settings.STANCES),
            "kind_vocabulary": list(settings.KINDS),
            "statement_forms": list(settings.STATEMENT_FORMS),
            "page_numbers": "EXACT — from per-page JSON source.",
            "statement_text": (
                "Sliced from the doc text between verbatim opening/closing anchors. "
                "Null when either anchor doesn't verify."
            ),
            "needs_human_review": (
                "True when kind=='individual' OR summary_quote_verified is False. "
                "No LLM critic in this pipeline; reasons are listed per person."
            ),
            "merge_rule": "One file per (entity, stance) — stance changes produce separate files.",
            "cost_note": "USD costs are ESTIMATES from settings.PRICES_USD_PER_M; verify against AWS Bedrock invoice.",
        },
        "people": written,
    }
    _write_json(doc_dir / "index.json", index)

    return {
        "doc_id": doc_id,
        "work_id": work_id,
        "title": title,
        "n_people": len(written),
        "stance_counts": stance_counts,
        "review_counts": review_counts,
        "statement_counts": statement_counts,
        "elapsed_sec": elapsed_sec,
        "usage": usage_summary,
        "out_dir": str(doc_dir),
    }
