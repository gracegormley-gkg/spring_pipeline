"""
M2.5: Grading Interface.

Produces one CSV per doc with columns:
  field | extracted_value | source_pages | critic_verdict | model_confidence | your_grade | your_notes

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


def _pages(spans) -> str:
    if not spans:
        return ""
    if isinstance(spans, dict):
        # key_people has {"preparers": [...], "commenters": [...]}
        parts = []
        for k, v in spans.items():
            if v:
                parts.append(f"{k}: " + ", ".join(v))
        return "; ".join(parts)
    if isinstance(spans, list):
        return ", ".join(str(s) for s in spans)
    return str(spans)


def build_rows(doc_id: str, work_id: str, m1: dict, m2: dict, critic: dict) -> list[dict]:
    """Build the per-doc grading rows."""
    rows: list[dict] = []

    def row(field: str, value: object, source_pages, model_conf: str):
        c = critic.get(field, {})
        rows.append({
            "field": field,
            "extracted_value": _short(value),
            "source_pages": _pages(source_pages),
            "critic_verdict": c.get("verdict", ""),
            "model_confidence": model_conf or c.get("model_confidence", ""),
            "your_grade": "",  # to be filled in by Grace
            "your_notes": _short(c.get("notes", ""), 300),
        })

    # M1 fields
    for f in ("title", "year", "eis_type", "lead_agency"):
        m = m1.get(f, {})
        row(f, m.get("value"), m.get("sources", []), m.get("confidence", ""))

    # M2 summary — break into 5 sub-rows for finer grading
    summary = m2.get("summary", {})
    summary_pages_all = []
    for sub in ("project_description", "affected_community", "alternatives_overview",
                "environmental_impact", "public_response"):
        sf = summary.get(sub, {})
        text_val = sf.get("text") if isinstance(sf, dict) else sf
        sp = sf.get("source_pages") if isinstance(sf, dict) else []
        summary_pages_all.extend(sp or [])
        rows.append({
            "field": f"summary.{sub}",
            "extracted_value": _short(text_val, 600),
            "source_pages": _pages(sp),
            "critic_verdict": critic.get("summary", {}).get("verdict", ""),
            "model_confidence": critic.get("summary", {}).get("model_confidence", ""),
            "your_grade": "",
            "your_notes": _short(critic.get("summary", {}).get("notes", ""), 200) if sub == "project_description" else "",
        })

    # M2 alternatives
    alt = m2.get("alternatives", {})
    row("alternatives", alt.get("value"), alt.get("source_pages"), alt.get("confidence", ""))

    # Themes
    th = m2.get("themes", {})
    row("themes", th.get("value"), th.get("source_pages"), th.get("confidence", ""))

    # Location
    loc = m2.get("location", {})
    row("location", loc.get("value"), loc.get("source_pages"), loc.get("confidence", ""))

    # Key people — split into 3 sub-rows
    kp = m2.get("key_people", {})
    kp_value = kp.get("value", {})
    kp_pages = kp.get("source_pages", {})
    for sub, key_path, pages_key in (
        ("key_people.agency_preparers", "agency_preparers", "preparers"),
        ("key_people.cooperating_agencies", "cooperating_agencies", "preparers"),
        ("key_people.public_commenters", "public_commenters", "commenters"),
    ):
        rows.append({
            "field": sub,
            "extracted_value": _short(kp_value.get(key_path)),
            "source_pages": _pages(kp_pages.get(pages_key, [])),
            "critic_verdict": critic.get("key_people", {}).get("verdict", ""),
            "model_confidence": critic.get("key_people", {}).get("model_confidence", ""),
            "your_grade": "",
            "your_notes": _short(critic.get("key_people", {}).get("notes", ""), 200) if sub == "key_people.agency_preparers" else "",
        })

    return rows


def write_grading_sheet(out_dir: Path, doc_id: str, work_id: str, title: str,
                        m1: dict, m2: dict, critic: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{doc_id}.csv"
    path = out_dir / fname

    rows = build_rows(doc_id, work_id, m1, m2, critic)

    with open(path, "w", newline="", encoding="utf-8") as f:
        # Header banner so Grace knows what she's grading
        f.write(f"# doc_id: {doc_id}\n")
        f.write(f"# work_id: {work_id}\n")
        f.write(f"# title: {title}\n")
        f.write(f"# grade options: {GRADE_OPTIONS}\n")
        f.write("# page numbers are ESTIMATED from char offsets at 2500 chars/page\n")
        f.write("\n")
        writer = csv.DictWriter(f, fieldnames=[
            "field", "extracted_value", "source_pages",
            "critic_verdict", "model_confidence",
            "your_grade", "your_notes",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return path
