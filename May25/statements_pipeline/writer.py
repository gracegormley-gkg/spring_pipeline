"""
Per-doc output writer for the statements pipeline.

Lays out files as:

    output/people/<doc_id>/
    ├── index.json                     # doc metadata, counts, listings
    ├── 001_sierra_club.json           # main sequence — entities with a
    ├── 002_john_smith.json            # contiguous statement, sectional
    ├── ...                            # listing, or no-statement classification
    └── paraphrases/
        ├── amax_exploration_inc.json  # narrator_paraphrase entries — kept
        ├── ...                        # for the data, but pulled out of the
                                       # main numbered sequence

Why partition? narrator_paraphrase rows are common in older EIS docs and
swamp the numbered sequence with low-evidence entries. Pulling them into a
separate folder lets reviewers focus on entities with real statements while
preserving the paraphrase data for completeness.

Stance lives at the top level of each person record but is judged downstream
of merge by find_statement (off the full statement / paraphrase / sectional
text). The companion `stance_confidence` field flags rows whose stance was
inferred from sparse evidence. Each record also carries a `response` block —
the agency / preparer reply nearby in the doc, if there is one.
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


def _paraphrase_filename(entity: str, used: set) -> str:
    """Slug-only filename for paraphrase rows. Disambiguates collisions with _2, _3, ..."""
    base = _slug(entity)
    name = f"{base}.json"
    i = 2
    while name in used:
        name = f"{base}_{i}.json"
        i += 1
    used.add(name)
    return name


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _is_paraphrase(row: dict) -> bool:
    return ((row.get("statement") or {}).get("form")) == "narrator_paraphrase"


def _needs_human_review(row: dict) -> tuple[bool, list[str]]:
    """Cheap rules-based review flag (no LLM critic in this pipeline).

    Reasons:
      - private_individual: kind == 'individual' (matches the v2 policy used by
        people_pipeline's critic).
      - quote_not_verbatim: the exemplar quote isn't verbatim in the doc, so
        we can't trust the source line without a human look.
      - low_stance_confidence: find_statement reported low confidence — usually
        means stance was inferred from sparse / ambiguous evidence (no
        contiguous statement, conflicting paraphrases, or an unrecognized
        stance from the model).
    """
    reasons: list[str] = []
    if row.get("kind") == "individual":
        reasons.append("private_individual")
    if not row.get("summary_quote_verified"):
        reasons.append("quote_not_verbatim")
    if row.get("stance_confidence") == "low":
        reasons.append("low_stance_confidence")
    return (len(reasons) > 0), reasons


def _person_record(
    row: dict,
    doc_id: str,
    work_id: Optional[str],
    needs_review: bool,
    reasons: list[str],
    *,
    sequence: Optional[int],
    merge_sequence: Optional[int],
) -> dict:
    """Shape one row into the per-person JSON record.

    `sequence` is the output sequence (None for paraphrases). `merge_sequence`
    is the original first-appearance order from merge — preserved on every
    record for traceability.
    """
    return {
        "sequence": sequence,
        "merge_sequence": merge_sequence,
        "doc_id": doc_id,
        "work_id": work_id,
        "entity": row.get("entity"),
        "kind": row.get("kind"),
        "role": row.get("role"),
        "stance": row.get("stance"),
        "stance_confidence": row.get("stance_confidence"),
        "stance_basis": row.get("stance_basis", ""),
        "summary": row.get("summary", ""),
        "statement": row.get("statement"),
        "response": row.get("response"),
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
        if form not in out:
            form = "none"
        out[form] += 1
        if s.get("present"):
            out["present"] += 1
        else:
            out["absent"] += 1
    return out


def _response_counts(rows: list[dict]) -> dict:
    """Distribution of response_form labels (with present/absent rollup)."""
    out = {f: 0 for f in settings.RESPONSE_FORMS}
    out["present"] = 0
    out["absent"] = 0
    for r in rows:
        resp = r.get("response") or {}
        form = resp.get("form") or "none"
        if form not in out:
            form = "none"
        out[form] += 1
        if resp.get("present"):
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


def _stance_confidence_counts(rows: list[dict]) -> dict:
    """Distribution of stance_confidence labels across the doc's rows."""
    counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for r in rows:
        c = r.get("stance_confidence")
        if c in counts:
            counts[c] += 1
        else:
            counts["unknown"] += 1
    return counts


def _review_counts(reviews: list[tuple[bool, list[str]]]) -> dict:
    """Aggregate from precomputed (needs_review, reasons) tuples."""
    counts = {"needs_review": 0, "auto_ok": 0}
    reason_counts: dict[str, int] = {}
    for needs, reasons in reviews:
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

    Partitions rows into a main numbered sequence (statements + sectional +
    none) and a paraphrases bucket. Counts are reported per-bucket.

    Returns a small summary dict for the pipeline-level run summary.
    """
    doc_dir = settings.PEOPLE_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    paraphrases_dir = doc_dir / "paraphrases"

    # Stable order: by merge sequence (first appearance).
    rows_sorted = sorted(rows, key=lambda r: r.get("sequence", 10**9))

    main_rows: list[dict] = []
    paraphrase_rows: list[dict] = []
    for r in rows_sorted:
        (paraphrase_rows if _is_paraphrase(r) else main_rows).append(r)

    # ---- Main sequence ------------------------------------------------------
    main_reviews = [_needs_human_review(r) for r in main_rows]
    written_main: list[dict] = []
    for new_seq, (r, (needs, reasons)) in enumerate(zip(main_rows, main_reviews), start=1):
        merge_seq = r.get("sequence")
        filename = _person_filename(new_seq, r.get("entity", ""))
        record = _person_record(
            r, doc_id, work_id, needs, reasons,
            sequence=new_seq, merge_sequence=merge_seq,
        )
        _write_json(doc_dir / filename, record)
        written_main.append({
            "sequence": new_seq,
            "merge_sequence": merge_seq,
            "file": filename,
            "entity": r.get("entity"),
            "kind": r.get("kind"),
            "role": r.get("role"),
            "stance": r.get("stance"),
            "stance_confidence": r.get("stance_confidence"),
            "statement_present": (r.get("statement") or {}).get("present", False),
            "statement_form": (r.get("statement") or {}).get("form"),
            "response_present": (r.get("response") or {}).get("present", False),
            "response_form": (r.get("response") or {}).get("form"),
            "needs_human_review": needs,
            "human_review_reasons": reasons,
        })

    # ---- Paraphrases --------------------------------------------------------
    para_reviews = [_needs_human_review(r) for r in paraphrase_rows]
    written_para: list[dict] = []
    if paraphrase_rows:
        paraphrases_dir.mkdir(parents=True, exist_ok=True)
        used_filenames: set = set()
        for r, (needs, reasons) in zip(paraphrase_rows, para_reviews):
            merge_seq = r.get("sequence")
            filename = _paraphrase_filename(r.get("entity", ""), used_filenames)
            record = _person_record(
                r, doc_id, work_id, needs, reasons,
                sequence=None, merge_sequence=merge_seq,
            )
            _write_json(paraphrases_dir / filename, record)
            written_para.append({
                "merge_sequence": merge_seq,
                "file": f"paraphrases/{filename}",
                "entity": r.get("entity"),
                "kind": r.get("kind"),
                "role": r.get("role"),
                "stance": r.get("stance"),
                "stance_confidence": r.get("stance_confidence"),
                "response_present": (r.get("response") or {}).get("present", False),
                "response_form": (r.get("response") or {}).get("form"),
                "needs_human_review": needs,
                "human_review_reasons": reasons,
            })

    # ---- Counts -------------------------------------------------------------
    main_stance = _stance_counts(main_rows)
    main_conf = _stance_confidence_counts(main_rows)
    main_review = _review_counts(main_reviews)
    main_stmt = _statement_counts(main_rows)
    main_resp = _response_counts(main_rows)

    para_stance = _stance_counts(paraphrase_rows)
    para_conf = _stance_confidence_counts(paraphrase_rows)
    para_review = _review_counts(para_reviews)
    para_resp = _response_counts(paraphrase_rows)

    index = {
        "doc_id": doc_id,
        "work_id": work_id,
        "title": title,
        "n_pages": n_pages,
        "n_chunks": n_chunks,
        "n_raw_rows": n_raw_rows,
        "n_people": len(written_main),
        "n_paraphrases": len(written_para),
        "stance_counts": main_stance,
        "stance_confidence_counts": main_conf,
        "review_counts": main_review,
        "statement_counts": main_stmt,
        "response_counts": main_resp,
        "paraphrase_counts": {
            "stance": para_stance,
            "stance_confidence": para_conf,
            "review": para_review,
            "response": para_resp,
        },
        "elapsed_sec": elapsed_sec,
        "usage": usage_summary,
        "schema": {
            "stance_vocabulary": list(settings.STANCES),
            "stance_confidence_vocabulary": ["high", "medium", "low"],
            "kind_vocabulary": list(settings.KINDS),
            "statement_forms": list(settings.STATEMENT_FORMS),
            "response_forms": list(settings.RESPONSE_FORMS),
            "stance_source": (
                "Decided by find_statement off the doc-text window — NOT by the "
                "per-chunk extractor. Capped to 'medium' for narrator_paraphrase / "
                "sectional / unverified-anchor rows; forced to 'low' for form='none'."
            ),
            "page_numbers": "EXACT — from per-page JSON source.",
            "statement_text": (
                "Sliced from the doc text between verbatim opening/closing anchors. "
                "Null when either anchor doesn't verify."
            ),
            "response_text": (
                "Sliced from the doc text between verbatim response anchors. Null "
                "when either response anchor doesn't verify or no response was "
                "found in the window. `response.summary` is always populated when "
                "response.present is true."
            ),
            "needs_human_review": (
                "True when kind=='individual' OR summary_quote_verified is False "
                "OR stance_confidence=='low'. No LLM critic in this pipeline; "
                "reasons are listed per person."
            ),
            "merge_rule": (
                "One file per entity. Stance is decided downstream of merge, so "
                "the same entity at most produces one row even if mentions span "
                "multiple chunks with different attribution modes."
            ),
            "paraphrase_split": (
                "Rows where statement.form == 'narrator_paraphrase' are written "
                "to paraphrases/<slug>.json (no NNN_ prefix, no entry in the main "
                "sequence). The data is preserved; the paraphrase counts and "
                "listing in `paraphrases` mirror the main shape."
            ),
            "cost_note": "USD costs are ESTIMATES from settings.PRICES_USD_PER_M; verify against AWS Bedrock invoice.",
        },
        "people": written_main,
        "paraphrases": written_para,
    }
    _write_json(doc_dir / "index.json", index)

    return {
        "doc_id": doc_id,
        "work_id": work_id,
        "title": title,
        "n_people": len(written_main),
        "n_paraphrases": len(written_para),
        "stance_counts": main_stance,
        "stance_confidence_counts": main_conf,
        "review_counts": main_review,
        "statement_counts": main_stmt,
        "response_counts": main_resp,
        "paraphrase_counts": {
            "stance": para_stance,
            "stance_confidence": para_conf,
            "review": para_review,
            "response": para_resp,
        },
        "elapsed_sec": elapsed_sec,
        "usage": usage_summary,
        "out_dir": str(doc_dir),
    }
