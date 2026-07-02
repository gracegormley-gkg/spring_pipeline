"""
Generate one CSV grading sheet per doc from statements_pipeline index.json output.

One row per commenter (grouped by complainer_id). Columns:
  entity, kind, role, stance, stance_confidence,
  comment_pages_from, comment_pages_to, comment_pages_all,
  n_complaints, responded, response_ids, response_pages, response_agency,
  needs_human_review, human_review_reasons, your_grade, your_notes

Response page is resolved by searching the source doc's per-page JSON for the
response file's opening_anchor. Falls back to the parent complaint's evidence
page if no anchor match is found.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # -> spring_pipeline/
PEOPLE_DIR = ROOT / "May25" / "statements_pipeline" / "output" / "people"
DOCS_DIR = ROOT / "Documents" / "output"
OUT_DIR = PEOPLE_DIR.parent / "grading_sheets"


def _clean(text: str) -> str:
    """Collapse whitespace for anchor matching."""
    return re.sub(r"\s+", " ", text or "").strip()


def _flatten_pages(pages) -> list[int]:
    """Turn a list of page strings like '20' or '1-19' into a list of ints."""
    out: list[int] = []
    for p in pages or []:
        s = str(p).strip()
        if not s:
            continue
        if "-" in s:
            a, b = s.split("-", 1)
            try:
                a_i, b_i = int(a), int(b)
                out.extend(range(min(a_i, b_i), max(a_i, b_i) + 1))
            except ValueError:
                continue
        else:
            try:
                out.append(int(s))
            except ValueError:
                continue
    return out


def load_page_index(doc_id: str) -> list[tuple[int, str]]:
    """Return [(page_number, cleaned_text), ...] for a doc's per-page JSONs."""
    doc_dir = DOCS_DIR / doc_id
    if not doc_dir.is_dir():
        return []
    pages: list[tuple[int, str]] = []
    for pf in sorted(doc_dir.glob("page_*.json")):
        try:
            data = json.loads(pf.read_text())
        except Exception:
            continue
        pn = data.get("page_number")
        txt = _clean(data.get("text", ""))
        if pn is not None and txt:
            pages.append((int(pn), txt))
    return pages


def find_anchor_page(anchor: str, pages: list[tuple[int, str]]) -> int | None:
    """Locate the page containing the first ~80 chars of the anchor."""
    a = _clean(anchor)
    if not a:
        return None
    # Try shrinking substrings for robustness
    for n in (120, 80, 50, 30):
        needle = a[:n]
        if len(needle) < 15:
            continue
        for pn, txt in pages:
            if needle in txt:
                return pn
    return None


def build_row(complainer_id: str, complaints: list[dict], responses_by_id: dict,
              page_index: list[tuple[int, str]]) -> dict:
    """Aggregate one commenter's complaints into a single grading row."""
    # Prefer the entity/role from the first complaint with a real entity name.
    entity = ""
    kind = ""
    role = ""
    for c in complaints:
        if c.get("entity"):
            entity = c["entity"]
            kind = c.get("kind", "")
            role = c.get("role", "")
            break

    stances = []
    for c in complaints:
        s = c.get("stance", "")
        if s and s not in stances:
            stances.append(s)
    stance = ";".join(stances)

    confs = []
    for c in complaints:
        cc = c.get("stance_confidence", "")
        if cc and cc not in confs:
            confs.append(cc)
    stance_confidence = ";".join(confs)

    all_page_ints: list[int] = []
    for c in complaints:
        all_page_ints.extend(_flatten_pages(c.get("evidence_pages")))
    all_page_ints = sorted(set(all_page_ints))

    comment_pages_from = min(all_page_ints) if all_page_ints else ""
    comment_pages_to = max(all_page_ints) if all_page_ints else ""
    comment_pages_all = ",".join(str(p) for p in all_page_ints)

    resp_ids: list[str] = []
    for c in complaints:
        for r in c.get("response_ids", []) or []:
            if r and r not in resp_ids:
                resp_ids.append(r)

    resp_pages: list[str] = []
    resp_agencies: list[str] = []
    for rid in resp_ids:
        r = responses_by_id.get(rid, {})
        ag = r.get("agency") or r.get("agency_kind") or ""
        if ag and ag not in resp_agencies:
            resp_agencies.append(ag)
        rp = r.get("_page")
        resp_pages.append(str(rp) if rp is not None else "?")

    needs_review = any(c.get("needs_human_review") for c in complaints)
    reasons: list[str] = []
    for c in complaints:
        for r in c.get("human_review_reasons", []) or []:
            if r and r not in reasons:
                reasons.append(r)

    return {
        "complainer_id": complainer_id,
        "entity": entity,
        "kind": kind,
        "role": role,
        "stance": stance,
        "stance_confidence": stance_confidence,
        "comment_pages_from": comment_pages_from,
        "comment_pages_to": comment_pages_to,
        "comment_pages_all": comment_pages_all,
        "n_complaints": len(complaints),
        "responded": "Y" if resp_ids else "N",
        "response_ids": ";".join(resp_ids),
        "response_pages": ";".join(resp_pages),
        "response_agency": ";".join(resp_agencies),
        "needs_human_review": "yes" if needs_review else "no",
        "human_review_reasons": ";".join(reasons),
        "your_grade": "",
        "your_notes": "",
    }


def process_doc(doc_dir: Path) -> Path | None:
    idx_file = doc_dir / "index.json"
    if not idx_file.is_file():
        return None
    idx = json.loads(idx_file.read_text())
    doc_id = idx.get("doc_id") or doc_dir.name
    work_id = idx.get("work_id", "")
    title = idx.get("title", "")

    complaints = idx.get("complaints", []) or []
    responses = idx.get("responses", []) or []
    responses_by_id = {r["child_id"]: r for r in responses}

    # Resolve response pages by scanning source doc pages for the opening anchor.
    page_index = load_page_index(doc_id)
    for rid, r in responses_by_id.items():
        rf = doc_dir / r.get("file", "")
        page = None
        if rf.is_file() and page_index:
            try:
                data = json.loads(rf.read_text())
                anchor = data.get("opening_anchor", "")
                page = find_anchor_page(anchor, page_index)
                if page is None:
                    # Fall back to complaint's evidence page
                    ce = data.get("complaint_evidence_pages") or []
                    if ce:
                        try:
                            page = int(str(ce[0]).split("-")[0])
                        except ValueError:
                            page = None
            except Exception:
                page = None
        r["_page"] = page

    grouped: dict[str, list[dict]] = defaultdict(list)
    for c in complaints:
        grouped[c["complainer_id"]].append(c)

    # Sort commenters by first-appearance order.
    first_order: dict[str, int] = {}
    for c in complaints:
        cid = c["complainer_id"]
        if cid not in first_order:
            first_order[cid] = c.get("order", 10**9)
    commenter_ids = sorted(grouped.keys(), key=lambda k: first_order[k])

    rows = [build_row(cid, grouped[cid], responses_by_id, page_index) for cid in commenter_ids]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{doc_id}_people.csv"

    fieldnames = list(rows[0].keys()) if rows else [
        "complainer_id", "entity", "kind", "role", "stance", "stance_confidence",
        "comment_pages_from", "comment_pages_to", "comment_pages_all",
        "n_complaints", "responded", "response_ids", "response_pages",
        "response_agency", "needs_human_review", "human_review_reasons",
        "your_grade", "your_notes",
    ]

    with out_path.open("w", newline="") as f:
        f.write(f"# doc_id: {doc_id}\n")
        f.write(f"# work_id: {work_id}\n")
        f.write(f"# title: {title}\n")
        f.write(f"# n_commenters: {len(rows)}\n")
        f.write(f"# n_complaints: {len(complaints)}\n")
        f.write(f"# n_responses: {len(responses)}\n")
        f.write("# grade options: correct|minor_issue|wrong|cant_tell\n")
        f.write("# stance vocab: in_favor|opposed|conditional|neutral\n")
        f.write("# kind vocab: individual|official|organization|agency|tribe|government|other\n")
        f.write("# response_pages: resolved by searching source per-page JSON for the response opening_anchor; '?' means anchor did not match any page.\n")
        f.write("\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def main() -> None:
    doc_dirs = sorted(p for p in PEOPLE_DIR.iterdir() if p.is_dir())
    for dd in doc_dirs:
        out = process_doc(dd)
        if out:
            print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
