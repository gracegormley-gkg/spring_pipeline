"""
M2.5: Grading Interface.

Produces one CSV per doc with columns:
  field | extracted_value | quote | source_pages | quote_verified |
  critic_verdict | model_confidence | your_grade | your_notes

Quote + page come straight from the per-field `evidence` blocks attached by
M1/M2. `quote_verified` is a quick filter for rows that need a human:
unverified quotes are forced to HUMAN_REVIEW upstream.

`your_grade` is blank for Grace to fill in {correct, minor_issue, wrong, cant_tell}.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

GRADE_OPTIONS = "correct|minor_issue|wrong|cant_tell"


def _short(value: object, n: int = 400) -> str:
    """Compact a value into a short, readable string for the grading sheet."""
    if value is None:
        return ""
    if isinstance(value, str):
        s = value.strip()
    elif isinstance(value, (int, float, bool)):
        s = str(value)
    elif isinstance(value, dict):
        s = json.dumps(value, ensure_ascii=False)
    elif isinstance(value, list):
        s = json.dumps(value, ensure_ascii=False)
    else:
        s = str(value)
    s = " ".join(s.split())
    if len(s) > n:
        s = s[: n - 1] + "…"
    return s


def _evidence_summary(evidence_list, max_quotes: int = 10) -> tuple[str, str, str]:
    """
    Collapse an evidence list into (pages, quote_display, verified_str) for the CSV.

    - pages: comma-joined unique source_pages across the evidence list (kept
      separate so the grader can filter/sort by page)
    - quote_display: up to `max_quotes` verbatim quotes joined by ' | '. Each
      quote is prefixed with its own page tag (e.g. `[p.142] "..."`,
      `[p.142-143] "..."`) so the grader can see which page every quote came
      from at a glance — important when a row carries multiple quotes that
      span different pages. Unverified quotes are also tagged with
      `[UNVERIFIED]`.
    - verified_str: "yes" if ANY evidence quote is verbatim-verified; "no" if
      none are; "" if the evidence list is empty.
    """
    if not evidence_list:
        return ("", "", "")
    pages: list[str] = []
    quote_chunks: list[str] = []
    any_verified = False
    any_unverified = False
    for ev in evidence_list:
        if not isinstance(ev, dict):
            continue
        ev_pages = [str(p) for p in (ev.get("source_pages") or [])]
        pages.extend(ev_pages)
        q = (ev.get("quote") or "").strip()
        verified = bool(ev.get("quote_verified"))
        if verified:
            any_verified = True
        else:
            any_unverified = True
        if q:
            page_tag = f"[p.{', '.join(ev_pages)}] " if ev_pages else ""
            verif_tag = "" if verified else "[UNVERIFIED] "
            quote_chunks.append(page_tag + verif_tag + q)
    pages_str = ", ".join(dict.fromkeys(pages))
    quote_str = " | ".join(_short(q, 300) for q in quote_chunks[:max_quotes])
    if any_verified and not any_unverified:
        verified_str = "yes"
    elif any_verified and any_unverified:
        verified_str = "mixed"
    else:
        verified_str = "no"
    return (pages_str, quote_str, verified_str)


def _pages_only(spans) -> str:
    """Format M1 sources (list/dict/str) — no evidence shape here."""
    if not spans:
        return ""
    if isinstance(spans, dict):
        parts = []
        for k, v in spans.items():
            if v:
                parts.append(f"{k}: " + ", ".join(v))
        return "; ".join(parts)
    if isinstance(spans, list):
        return ", ".join(str(s) for s in spans)
    return str(spans)


def _row_no_evidence(field: str, value: object, source_pages, model_conf: str, critic: dict) -> dict:
    """Build a grading row for M1 fields (no evidence block)."""
    c = critic.get(field, {})
    return {
        "field": field,
        "extracted_value": _short(value),
        "quote": "",
        "source_pages": _pages_only(source_pages),
        "quote_verified": "",
        "critic_verdict": c.get("verdict", ""),
        "model_confidence": model_conf or c.get("model_confidence", ""),
        "your_grade": "",
        "your_notes": _short(c.get("notes", ""), 300),
    }


def _row_with_evidence(field: str, value: object, evidence_list, model_conf: str,
                      critic_field_key: str, critic: dict, *, include_notes: bool = True,
                      value_n: int = 400) -> dict:
    """Build a grading row for an M2 field with an evidence block."""
    pages_str, quote_str, verified_str = _evidence_summary(evidence_list)
    c = critic.get(critic_field_key, {})
    return {
        "field": field,
        "extracted_value": _short(value, value_n),
        "quote": quote_str,
        "source_pages": pages_str,
        "quote_verified": verified_str,
        "critic_verdict": c.get("verdict", ""),
        "model_confidence": model_conf or c.get("model_confidence", ""),
        "your_grade": "",
        "your_notes": _short(c.get("notes", ""), 200) if include_notes else "",
    }


def build_rows(doc_id: str, work_id: str, m1: dict, m2: dict, critic: dict) -> list[dict]:
    """Build the per-doc grading rows."""
    rows: list[dict] = []

    # --- M1 fields (no evidence blocks; M1 is NUL-first + cover-page regex) ---
    for f in ("title", "year", "eis_type", "lead_agency"):
        m = m1.get(f, {})
        rows.append(_row_no_evidence(
            f, m.get("value"), m.get("sources", []), m.get("confidence", ""), critic,
        ))

    # --- M2 summary — one row per subfield ---
    summary = m2.get("summary", {})
    summary_critic = critic.get("summary", {})
    for i, sub in enumerate(
        ("overview", "project_description", "affected_community", "alternatives_overview",
         "environmental_impact", "public_response")
    ):
        sf = summary.get(sub, {}) or {}
        text_val = sf.get("text") if isinstance(sf, dict) else sf
        ev = sf.get("evidence") if isinstance(sf, dict) else []
        pages_str, quote_str, verified_str = _evidence_summary(ev)
        rows.append({
            "field": f"summary.{sub}",
            # Summary subfields are the longest-form output and are the rows
            # most likely to be misread as "the CSV doesn't match M2." Pass
            # the full text through; other rows keep their tighter caps.
            "extracted_value": _short(text_val, n=10**9),
            "quote": quote_str,
            "source_pages": pages_str,
            "quote_verified": verified_str,
            "critic_verdict": summary_critic.get("verdict", ""),
            "model_confidence": summary_critic.get("model_confidence", ""),
            "your_grade": "",
            "your_notes": _short(summary_critic.get("notes", ""), 200) if i == 0 else "",
        })

    # --- M2 alternatives — one row per alternative ---
    alt = m2.get("alternatives", {})
    alt_critic = critic.get("alternatives", {})
    alternatives_list = alt.get("value") or []
    if alternatives_list:
        for j, a in enumerate(alternatives_list):
            if not isinstance(a, dict):
                continue
            display_value = {"name": a.get("name", ""), "description": a.get("description", "")}
            pages_str, quote_str, verified_str = _evidence_summary(a.get("evidence"))
            rows.append({
                "field": f"alternatives[{j}]",
                "extracted_value": _short(display_value, 400),
                "quote": quote_str,
                "source_pages": pages_str,
                "quote_verified": verified_str,
                "critic_verdict": alt_critic.get("verdict", ""),
                "model_confidence": alt_critic.get("model_confidence", ""),
                "your_grade": "",
                "your_notes": _short(alt_critic.get("notes", ""), 200) if j == 0 else "",
            })
    else:
        rows.append(_row_with_evidence(
            "alternatives", [], [], alt.get("confidence", ""), "alternatives", critic,
        ))

    # --- Themes ---
    th = m2.get("themes", {})
    rows.append(_row_with_evidence(
        "themes", th.get("value"), th.get("evidence"),
        th.get("confidence", ""), "themes", critic,
    ))

    # --- Location — one row per place ---
    loc = m2.get("location", {})
    loc_value = loc.get("value") or {}
    loc_critic = critic.get("location", {})
    places = loc_value.get("places") or []
    if places:
        for k, p in enumerate(places):
            if not isinstance(p, dict):
                continue
            display = {k2: v for k2, v in p.items() if k2 != "evidence"}
            pages_str, quote_str, verified_str = _evidence_summary(p.get("evidence"))
            rows.append({
                "field": f"location.places[{k}]",
                "extracted_value": _short(display, 300),
                "quote": quote_str,
                "source_pages": pages_str,
                "quote_verified": verified_str,
                "critic_verdict": loc_critic.get("verdict", ""),
                "model_confidence": loc_critic.get("model_confidence", ""),
                "your_grade": "",
                "your_notes": _short(loc_critic.get("notes", ""), 200) if k == 0 else "",
            })
        # Roll-up row capturing is_multi_site + geocoded for the grader
        rollup = {
            "is_multi_site": loc_value.get("is_multi_site"),
            "geocoded": loc_value.get("geocoded"),
        }
        rows.append({
            "field": "location.summary",
            "extracted_value": _short(rollup, 400),
            "quote": "",
            "source_pages": "",
            "quote_verified": "",
            "critic_verdict": loc_critic.get("verdict", ""),
            "model_confidence": loc_critic.get("model_confidence", ""),
            "your_grade": "",
            "your_notes": "",
        })
    else:
        rows.append(_row_with_evidence(
            "location", loc.get("value"), [], loc.get("confidence", ""), "location", critic,
        ))

    # --- Key people — one row per entry ---
    kp = m2.get("key_people", {})
    kp_value = kp.get("value", {})
    kp_critic = critic.get("key_people", {})
    for bucket_key, bucket_label in (
        ("agency_preparers", "key_people.agency_preparers"),
        ("cooperating_agencies", "key_people.cooperating_agencies"),
        ("public_commenters", "key_people.public_commenters"),
    ):
        entries = kp_value.get(bucket_key, []) or []
        if entries:
            for m, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                display = {k2: v for k2, v in entry.items() if k2 != "evidence"}
                pages_str, quote_str, verified_str = _evidence_summary(entry.get("evidence"))
                rows.append({
                    "field": f"{bucket_label}[{m}]",
                    "extracted_value": _short(display, 300),
                    "quote": quote_str,
                    "source_pages": pages_str,
                    "quote_verified": verified_str,
                    "critic_verdict": kp_critic.get("verdict", ""),
                    "model_confidence": kp_critic.get("model_confidence", ""),
                    "your_grade": "",
                    "your_notes": _short(kp_critic.get("notes", ""), 200) if (bucket_key == "agency_preparers" and m == 0) else "",
                })
        else:
            rows.append({
                "field": bucket_label,
                "extracted_value": "",
                "quote": "",
                "source_pages": "",
                "quote_verified": "",
                "critic_verdict": kp_critic.get("verdict", ""),
                "model_confidence": kp_critic.get("model_confidence", ""),
                "your_grade": "",
                "your_notes": "",
            })

    return rows


def write_grading_sheet(out_dir: Path, doc_id: str, work_id: str, title: str,
                        m1: dict, m2: dict, critic: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{doc_id}.csv"
    path = out_dir / fname

    rows = build_rows(doc_id, work_id, m1, m2, critic)

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# doc_id: {doc_id}\n")
        f.write(f"# work_id: {work_id}\n")
        f.write(f"# title: {title}\n")
        f.write(f"# grade options: {GRADE_OPTIONS}\n")
        f.write("# page numbers are EXACT (from per-page JSON source).\n")
        f.write("# quote_verified: yes = quote present verbatim on cited page; no = forced HUMAN_REVIEW; mixed = some verified, some not.\n")
        f.write("\n")
        writer = csv.DictWriter(f, fieldnames=[
            "field", "extracted_value", "quote", "source_pages", "quote_verified",
            "critic_verdict", "model_confidence",
            "your_grade", "your_notes",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return path
