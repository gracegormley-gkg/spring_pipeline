"""
Per-doc output writer for the statements pipeline.

Output layout (refurbished):

    output/people/<doc_id>/
    ├── index.json                    # sequential, link-only: ordered list
    │                                 # of complaints + their response IDs
    ├── complaints/                   # parents — one file per complaint
    │   ├── <doc_id>_C001_sierra_club.json
    │   ├── <doc_id>_C002_blm.json
    │   └── ...
    └── responses/                    # children — one file per agency reply
        ├── <doc_id>_R001_blm.json
        └── ...

Each merged ENTITY can produce 0+ complaints (find_statement returns a list).
A single Sierra Club entity may have a paraphrase complaint on page 34 AND a
full-letter complaint on page 142 — those become two complaint files linked
by the same `complainer_id`.

ID scheme (doc-prefixed, globally unique):
  - complainer_id  = <doc_id>_K<NNN>   one per merged entity
  - complaint_id   = <doc_id>_C<NNN>   one per complaint (may be 0 or many per complainer)
  - child_id       = <doc_id>_R<NNN>   one per response

The doc dir is wiped at the start of write_doc to prevent ghost files from
prior runs. raw_extract caches are NEVER touched here.
"""

from __future__ import annotations

import json
import re
import shutil
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


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _needs_human_review(row: dict) -> tuple[bool, list[str]]:
    """Cheap rules-based review flag (no LLM critic in this pipeline).

    Reasons:
      - private_individual: kind == 'individual'.
      - quote_not_verbatim: the merged exemplar quote isn't verbatim in doc.
      - low_stance_confidence: find_statement reported low confidence.
      - no_complaints: find_statement returned an empty complaints list
        (possibly because the find_statement call failed or the model couldn't
        find any position-bearing mention in the window).
    """
    reasons: list[str] = []
    if row.get("kind") == "individual":
        reasons.append("private_individual")
    if not row.get("summary_quote_verified"):
        reasons.append("quote_not_verbatim")
    if row.get("stance_confidence") == "low":
        reasons.append("low_stance_confidence")
    if not (row.get("complaints") or []):
        reasons.append("no_complaints")
    return (len(reasons) > 0), reasons


def _evidence_first_page(complaint: dict) -> int:
    """Sort key: minimum integer page in evidence_pages, or large default."""
    nums: list[int] = []
    for span in complaint.get("evidence_pages") or []:
        if not isinstance(span, str):
            continue
        try:
            if "-" in span:
                a, _ = span.split("-", 1)
                nums.append(int(a.strip()))
            else:
                nums.append(int(span.strip()))
        except ValueError:
            continue
    return min(nums) if nums else 10**9


def _clean_doc_dir(doc_dir: Path) -> None:
    """Remove the doc's people/ subdir contents to start clean. Keeps the dir
    itself. Does NOT touch raw_extract/."""
    if doc_dir.exists():
        for entry in doc_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()


def _form_counts(forms: list[str], allowed: tuple) -> dict:
    out = {f: 0 for f in allowed}
    out["present"] = 0
    for f in forms:
        if f not in out:
            f = "none"
        out[f] += 1
        if f != "none":
            out["present"] += 1
    out["absent"] = sum(1 for f in forms if f == "none")
    return out


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
    """Write per-doc complaints/, responses/, and index.json.

    Returns a per-doc summary for run_summary.json.
    """
    doc_dir = settings.PEOPLE_DIR / doc_id
    _clean_doc_dir(doc_dir)
    doc_dir.mkdir(parents=True, exist_ok=True)
    complaints_dir = doc_dir / "complaints"
    responses_dir = doc_dir / "responses"

    # Stable order: by merge sequence (first appearance of the entity).
    rows_sorted = sorted(rows, key=lambda r: r.get("sequence", 10**9))

    # ---- First pass: assign complainer_id, flatten complaints, sort ----
    flat: list[dict] = []
    for entity_idx, row in enumerate(rows_sorted, start=1):
        complainer_id = f"{doc_id}_K{entity_idx:03d}"
        needs, reasons = _needs_human_review(row)
        complaints = row.get("complaints") or []
        if not complaints:
            # Entity with no complaints — record as a complainer-only entry so
            # we don't lose the entity. Materializes as a complaint file with
            # form="none" and no statement / response.
            flat.append({
                "row": row,
                "complainer_id": complainer_id,
                "needs_review": needs,
                "review_reasons": reasons,
                "complaint": {
                    "form": "none",
                    "evidence_pages": row.get("evidence_pages") or [],
                    "complaint_summary": "",
                    "statement": {
                        "text": None,
                        "opening_anchor": "",
                        "closing_anchor": "",
                        "opening_anchor_verified": False,
                        "closing_anchor_verified": False,
                    },
                    "response": {
                        "present": False, "form": "none",
                        "agency": "", "agency_kind": "other", "summary": "",
                        "text": None,
                        "opening_anchor": "", "closing_anchor": "",
                        "opening_anchor_verified": False,
                        "closing_anchor_verified": False,
                    },
                },
                "first_page": 10**9,
                "synthetic": True,
            })
        else:
            for c in complaints:
                flat.append({
                    "row": row,
                    "complainer_id": complainer_id,
                    "needs_review": needs,
                    "review_reasons": reasons,
                    "complaint": c,
                    "first_page": _evidence_first_page(c),
                    "synthetic": False,
                })

    # Sort complaints by document order: first evidence page, then by entity
    # appearance (so two same-page complaints fall back to merge order).
    flat.sort(key=lambda f: (f["first_page"], f["row"].get("sequence", 10**9)))

    # ---- Second pass: assign complaint_id + child_id, write files ----
    complaint_records: list[dict] = []
    response_records: list[dict] = []
    sequence: list[dict] = []

    next_complaint_n = 1
    next_response_n = 1

    for order, item in enumerate(flat, start=1):
        row = item["row"]
        c = item["complaint"]
        complaint_id = f"{doc_id}_C{next_complaint_n:03d}"
        next_complaint_n += 1
        complaint_filename = f"{complaint_id}_{_slug(row.get('entity', ''))}.json"

        # Build child responses (currently 0 or 1 per complaint).
        response_ids: list[str] = []
        resp = c.get("response") or {}
        if resp.get("present"):
            child_id = f"{doc_id}_R{next_response_n:03d}"
            next_response_n += 1
            agency_slug = _slug(resp.get("agency") or resp.get("form") or "response")
            response_filename = f"{child_id}_{agency_slug}.json"
            response_record = {
                "child_id": child_id,
                "parent_id": complaint_id,
                "complainer_id": item["complainer_id"],
                "doc_id": doc_id,
                "work_id": work_id,
                "agency": resp.get("agency", ""),
                "agency_kind": resp.get("agency_kind", "other"),
                "form": resp.get("form", "none"),
                "summary": resp.get("summary", ""),
                "text": resp.get("text"),
                "opening_anchor": resp.get("opening_anchor", ""),
                "closing_anchor": resp.get("closing_anchor", ""),
                "opening_anchor_verified": resp.get("opening_anchor_verified", False),
                "closing_anchor_verified": resp.get("closing_anchor_verified", False),
                "complaint_evidence_pages": c.get("evidence_pages") or [],
            }
            _write_json(responses_dir / response_filename, response_record)
            response_ids.append(child_id)
            response_records.append({
                "child_id": child_id,
                "parent_id": complaint_id,
                "file": f"responses/{response_filename}",
                "agency": resp.get("agency", ""),
                "agency_kind": resp.get("agency_kind", "other"),
                "form": resp.get("form", "none"),
                "text_present": resp.get("text") is not None,
            })

        # Complaint record (parent).
        complaint_record = {
            "complaint_id": complaint_id,
            "complainer_id": item["complainer_id"],
            "doc_id": doc_id,
            "work_id": work_id,
            "order": order,
            # Complainer info denormalized.
            "entity": row.get("entity"),
            "kind": row.get("kind"),
            "role": row.get("role"),
            "stance": row.get("stance"),
            "stance_confidence": row.get("stance_confidence"),
            "stance_basis": row.get("stance_basis", ""),
            "summary": row.get("summary", ""),
            # Complaint-specific.
            "form": c.get("form", "none"),
            "complaint_summary": c.get("complaint_summary", ""),
            "evidence_pages": c.get("evidence_pages") or [],
            "statement": c.get("statement") or {},
            "response_ids": response_ids,
            "needs_human_review": item["needs_review"],
            "human_review_reasons": item["review_reasons"],
            # Provenance.
            "summary_quote": row.get("summary_quote"),
            "summary_quote_verified": row.get("summary_quote_verified"),
            "attribution_mode": row.get("attribution_mode"),
            "attribution_modes_seen": row.get("attribution_modes_seen"),
            "n_mentions": row.get("n_mentions"),
            "mentions": row.get("mentions"),
            "window_pages": row.get("window_pages"),
            "find_statement_error": row.get("_find_statement_error"),
            "merge_sequence": row.get("sequence"),
            "synthetic_no_complaint": item.get("synthetic", False),
        }
        _write_json(complaints_dir / complaint_filename, complaint_record)

        complaint_records.append({
            "complaint_id": complaint_id,
            "complainer_id": item["complainer_id"],
            "file": f"complaints/{complaint_filename}",
            "order": order,
            "entity": row.get("entity"),
            "kind": row.get("kind"),
            "role": row.get("role"),
            "stance": row.get("stance"),
            "stance_confidence": row.get("stance_confidence"),
            "form": c.get("form", "none"),
            "evidence_pages": c.get("evidence_pages") or [],
            "response_ids": response_ids,
            "needs_human_review": item["needs_review"],
            "human_review_reasons": item["review_reasons"],
        })

        # Sequential index entry — id-tag-only.
        sequence.append({
            "order": order,
            "complaint_id": complaint_id,
            "complainer_id": item["complainer_id"],
            "complaint_file": f"complaints/{complaint_filename}",
            "response_ids": response_ids,
        })

    # ---- Counts ----------------------------------------------------------
    n_complainers = len(rows_sorted)
    n_complaints = len(complaint_records)
    n_responses = len(response_records)

    stance_counts: dict = {}
    conf_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for row in rows_sorted:
        s = row.get("stance", "?")
        stance_counts[s] = stance_counts.get(s, 0) + 1
        c = row.get("stance_confidence")
        if c in conf_counts:
            conf_counts[c] += 1
        else:
            conf_counts["unknown"] += 1

    review_counts = {"needs_review": 0, "auto_ok": 0, "reasons": {}}
    for row in rows_sorted:
        needs, reasons = _needs_human_review(row)
        if needs:
            review_counts["needs_review"] += 1
            for r in reasons:
                review_counts["reasons"][r] = review_counts["reasons"].get(r, 0) + 1
        else:
            review_counts["auto_ok"] += 1

    statement_form_counts = _form_counts(
        [c["form"] for c in complaint_records], settings.STATEMENT_FORMS
    )
    response_form_counts = _form_counts(
        [r["form"] for r in response_records], settings.RESPONSE_FORMS
    )

    # ---- index.json ------------------------------------------------------
    index = {
        "doc_id": doc_id,
        "work_id": work_id,
        "title": title,
        "n_pages": n_pages,
        "n_chunks": n_chunks,
        "n_raw_rows": n_raw_rows,
        "n_complainers": n_complainers,
        "n_complaints": n_complaints,
        "n_responses": n_responses,
        "stance_counts": stance_counts,
        "stance_confidence_counts": conf_counts,
        "review_counts": review_counts,
        "statement_form_counts": statement_form_counts,
        "response_form_counts": response_form_counts,
        "elapsed_sec": elapsed_sec,
        "usage": usage_summary,
        "schema": {
            "ids": {
                "complainer_id": "<doc_id>_K<NNN> — one per merged entity",
                "complaint_id":  "<doc_id>_C<NNN> — one per complaint instance (entities can have multiple)",
                "child_id":      "<doc_id>_R<NNN> — one per agency response",
            },
            "stance_vocabulary": list(settings.STANCES),
            "stance_confidence_vocabulary": ["high", "medium", "low"],
            "kind_vocabulary": list(settings.KINDS),
            "statement_forms": list(settings.STATEMENT_FORMS),
            "response_forms": list(settings.RESPONSE_FORMS),
            "stance_source": (
                "Decided by find_statement at the entity level — same stance "
                "applies to all complaints from the same entity. Capped to "
                "'medium' when no contiguous statement is verified for any "
                "complaint; forced 'low' when there are no complaints."
            ),
            "page_numbers": "EXACT — from per-page JSON source.",
            "statement_text": (
                "Sliced from the doc text between verbatim opening/closing "
                "anchors. Null when either anchor doesn't verify."
            ),
            "response_text": (
                "Sliced from the doc text between verbatim response anchors. "
                "Null when either anchor doesn't verify or no response is in "
                "the window."
            ),
            "needs_human_review": (
                "True when kind=='individual' OR summary_quote_verified is "
                "False OR stance_confidence=='low' OR no complaints were "
                "produced."
            ),
            "merge_rule": (
                "One row per entity. find_statement may then return multiple "
                "complaints per entity — each becomes its own complaint file."
            ),
            "cost_note": "USD costs are ESTIMATES; verify against AWS Bedrock invoice.",
        },
        "sequence": sequence,
        "complaints": complaint_records,
        "responses": response_records,
    }
    _write_json(doc_dir / "index.json", index)

    return {
        "doc_id": doc_id,
        "work_id": work_id,
        "title": title,
        "n_complainers": n_complainers,
        "n_complaints": n_complaints,
        "n_responses": n_responses,
        "stance_counts": stance_counts,
        "stance_confidence_counts": conf_counts,
        "review_counts": review_counts,
        "statement_form_counts": statement_form_counts,
        "response_form_counts": response_form_counts,
        "elapsed_sec": elapsed_sec,
        "usage": usage_summary,
        "out_dir": str(doc_dir),
    }
