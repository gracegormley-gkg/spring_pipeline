"""
Human-grade loading and normalization.

MCAL_PLAN 3.1 specifies the grade source as
`segment_a/output/grading_sheets/*.csv` filtered on `your_grade != "ok"`. That
is not usable as written, for three reasons found by auditing the repo:

  1. All 9 grading sheets are unfilled -- 0 of 333 rows have `your_grade`.
  2. `"ok"` is not in those sheets' vocabulary anyway; grading.py:21 declares
     `correct|minor_issue|wrong|cant_tell`.
  3. The real grades live in `May25/Evaluation - Sheet1.csv`, which is
     transposed (docs as columns, fields as rows), free-text, and covers 8
     docs -- not the 9 that MCAL_PLAN 0 assumes. `p0491_35556036091957` has a
     grading sheet but no Evaluation-sheet column, so it is ungraded.

So this module supports both shapes and merges them:

  * `load_evaluation_sheet()` -- the seed-v1 source. Grades are COARSE: one
    verdict per field per doc. `location` is a single item, not one per place;
    `alternatives[0]` covers the whole list; `key people` covers all three
    role buckets.
  * `load_grading_sheets()` -- the source from v2 onward, once the per-doc
    sheets are actually filled in. Grades are ITEM-level: one row per
    `location.places[0]`, `key_people.cooperating_agencies[2]`, etc.
  * `load_grades()` -- both, with item-level grades preferred per (doc, field)
    when present.

Two normalization decisions worth stating explicitly, because they change the
calibration numbers:

  A. "ok, missing citation - pg 35" counts as INCORRECT (y_i = 0), tagged
     T01_missing_citation. Justification: MCAL_PLAN 3.5's decision table sends
     Q1=no (no page cite) to RE_EXTRACT, and MCAL_PLAN 6 makes missing-citation
     rate a gating target. Treating it as "ok" would erase the single most
     common observed failure (4/8 on summary.public_response).

  B. The doc-level note "includes undefined acronyms" (present on 8/8 docs)
     does NOT set y_i = 0 on any field. It is recorded as a doc-level tag and
     as a per-field `acronym_issue` flag feeding the s_acronym signal and the
     MCAL_PLAN 6 acronym gating target. Justification: MCAL_PLAN 3.5 routes an
     acronym failure to PASS_WITH_NOTE + T04, explicitly not to RE_EXTRACT or
     HUMAN_REVIEW. Folding it into y_i would mark all 8 docs wrong on every
     summary field and collapse the distinction between an unglossed acronym
     and a fabricated magnitude.

Tag inference here is a SEED HEURISTIC, not the authority. `raw_grade` is
always preserved so that taxonomy.py's TnT-LLM induction clusters the human's
own words rather than my regexes.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Iterable, Optional

from . import settings

log = logging.getLogger(__name__)


# --- Field-key normalization -------------------------------------------------
# The Evaluation sheet's row labels differ from the canonical keys in
# settings.ALL_FIELDS: capitalization (`EIS_type`), spaces (`key people`),
# and positional suffixes (`alternatives[0]`).

_EVAL_ROW_TO_FIELD = {
    "title": "title",
    "year": "year",
    "eis_type": "eis_type",
    "eis type": "eis_type",
    "lead_agency": "lead_agency",
    "lead agency": "lead_agency",
    "summary.overview": "summary.overview",
    "summary.project_description": "summary.project_description",
    "summary.affected_community": "summary.affected_community",
    "summary.alternatives_overview": "summary.alternatives_overview",
    "summary.environmental_impact": "summary.environmental_impact",
    "summary.public_response": "summary.public_response",
    "alternatives": "alternatives",
    "alternatives[0]": "alternatives",
    "themes": "themes",
    "location": "location",
    "key people": "key_people",
    "key_people": "key_people",
    "summary_of_interest": settings.SUMMARY_OF_INTEREST,
}

# Rows in the Evaluation sheet that are metadata, not field grades.
_EVAL_META_ROWS = {"id", "slug", "notes"}


def canonical_field(raw: str) -> Optional[str]:
    """
    Normalize a grade-sheet field label to a canonical key.

    Handles both Evaluation-sheet labels (`EIS_type`, `key people`) and
    grading-sheet dotted/bracketed paths (`location.places[0]`,
    `key_people.cooperating_agencies[2]`, `summary.public_response`).

    Returns None for metadata rows and unrecognized labels.
    """
    s = (raw or "").strip()
    if not s:
        return None
    low = s.lower()
    if low in _EVAL_META_ROWS:
        return None
    if low in _EVAL_ROW_TO_FIELD:
        return _EVAL_ROW_TO_FIELD[low]

    # Grading-sheet item paths: strip [n] indices, then collapse item-level
    # sub-buckets up to the canonical field. Coarse granularity means
    # location.places[0] and location.summary both map to `location`.
    stripped = re.sub(r"\[\d+\]", "", low).strip()
    if stripped in _EVAL_ROW_TO_FIELD:
        return _EVAL_ROW_TO_FIELD[stripped]
    head = stripped.split(".", 1)[0]
    if head in ("location", "key_people", "alternatives", "themes"):
        return _EVAL_ROW_TO_FIELD.get(head, head)
    if stripped.startswith("summary."):
        return _EVAL_ROW_TO_FIELD.get(stripped)
    return None


# --- Grade text -> (correct, tags) ------------------------------------------

_UNGRADED = {"", "-", "n/a", "na", "tbd", "?"}

# Phrases that mean "correct" when they are the whole cell.
_CLEAN_OK = {"ok", "correct", "good", "fine", "yes"}

# Grading-sheet vocabulary (grading.py:21).
_SHEET_GRADE_CORRECT = {"correct"}
_SHEET_GRADE_WRONG = {"wrong", "minor_issue"}
_SHEET_GRADE_UNGRADED = {"cant_tell", "can't_tell", "cant tell"}


@dataclass
class GradeItem:
    """One human grade for one (doc, field)."""

    doc_id: str
    field: str
    bucket: str
    correct: bool
    raw_grade: str
    failure_tags: list[str] = dc_field(default_factory=list)
    acronym_issue: bool = False
    notes: str = ""
    granularity: str = "coarse"      # "coarse" | "item"
    source: str = "evaluation_sheet"  # "evaluation_sheet" | "grading_sheet"
    item_key: Optional[str] = None    # original label, e.g. "location.places[0]"

    @property
    def y(self) -> int:
        """Binary correctness for conformal prediction. 1 = correct."""
        return 1 if self.correct else 0


def classify_grade(field: str, raw: str) -> tuple[Optional[bool], list[str]]:
    """
    Parse a free-text grade cell into (correct, failure_tags).

    Returns correct=None for an ungraded cell -- callers MUST drop those rather
    than defaulting to correct. Silently treating a blank as correct is how a
    calibration set gets quietly optimistic; the Lincoln Hwy `key people` cell
    is blank and must not count as a pass.
    """
    s = (raw or "").strip()
    low = s.lower().strip().rstrip(".")

    if low in _UNGRADED:
        return None, []
    if low in _SHEET_GRADE_UNGRADED:
        return None, []

    tags: list[str] = []

    # --- Explicit grading-sheet vocabulary -------------------------------
    if low in _SHEET_GRADE_CORRECT:
        return True, []
    if low in _SHEET_GRADE_WRONG:
        return False, []

    # --- A bare "ok" with no qualifier -----------------------------------
    if low in _CLEAN_OK:
        return True, []

    # --- Qualified grades: "ok, missing citation - pg 35" ----------------
    # Any qualifier after "ok" is a defect. See decision (A) in the module
    # docstring.
    has_missing_cite = bool(re.search(r"missing\s+cit", low))
    if has_missing_cite:
        tags.append("T01_missing_citation")

    if "hallucinat" in low or "fabricat" in low:
        tags.append("T03_outside_text_fabrication")

    is_wrong_prefixed = low.startswith("wrong") or low.startswith("incorrect")

    # --- Field-specific patterns -----------------------------------------
    if field == "year":
        if is_wrong_prefixed:
            tags.append("T11_year_ocr_error")
    elif field == "eis_type":
        if is_wrong_prefixed or "rod" in low:
            tags.append("T12_eis_type_confused_with_rod")
    elif field == "location":
        if re.search(r"\bno\s+geocode", low):
            tags.append("T06_geocode_missing")
        if "specificity" in low or "coarse" in low:
            tags.append("T07_geocode_wrong_specificity")
        if re.search(r"\bnational\b", low) or re.search(r"\bno\s+location\b", low):
            tags.append("T08_scope_misclassified_national")
        # "has 3 locations (all listed in the doc) one geocoded"
        if re.search(r"\bone\s+geocoded\b", low) or re.search(r"\bpartial", low):
            tags.append("T09_multi_site_partial_geocode")
        if re.search(r"\b\d+\s+locations?\b", low) and "one geocoded" in low:
            if "T09_multi_site_partial_geocode" not in tags:
                tags.append("T09_multi_site_partial_geocode")
    elif field == "key_people":
        if "cooperator" in low or "cooperating" in low:
            tags.append("T05_commenter_mislabeled_as_cooperator")
        # An "empty"/"nearly empty" key_people result has no dedicated code in
        # the T01-T18 seed taxonomy. Deliberately left untagged so that
        # taxonomy.py induction can propose a T19+ code from the human's own
        # wording, rather than being force-fit to an ill-matching seed code.
    elif field == "alternatives":
        if low in ("empty", "[]") or "empty" in low:
            tags.append("T10_alternatives_chapter_missed")
    elif field.startswith("summary.") or field == settings.SUMMARY_OF_INTEREST:
        # A "wrong:" grade citing a figure is a numeric hallucination
        # (MCAL_PLAN 1(3), 1(4)): "$659 million" vs "$369 million",
        # "Magnitude 7.5" vs "Magnitude 7.0".
        if is_wrong_prefixed and re.search(r"[\d$]", low):
            tags.append("T02_numeric_hallucination")

    # --- Verdict ----------------------------------------------------------
    if is_wrong_prefixed:
        return False, _dedupe(tags)
    if tags:
        return False, _dedupe(tags)
    if low.startswith("ok"):
        # "ok" plus an unrecognized qualifier. Conservative: treat as a defect
        # and leave it untagged so induction can name it.
        if low not in _CLEAN_OK:
            return False, []
        return True, []

    # Anything else is a free-text description of a problem
    # ("empty", "no geocode", "nearly empty", "all commenters = cooperators").
    return False, _dedupe(tags)


def _dedupe(xs: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for x in xs:
        seen.setdefault(x, None)
    return list(seen)


def _has_acronym_issue(doc_note: str) -> bool:
    return "acronym" in (doc_note or "").lower()


# --- Evaluation sheet loader ------------------------------------------------


@dataclass
class GradeSet:
    """All human grades available at a given calibration stage."""

    items: list[GradeItem] = dc_field(default_factory=list)
    doc_notes: dict[str, str] = dc_field(default_factory=dict)
    slugs: dict[str, str] = dc_field(default_factory=dict)
    overall_notes: str = ""
    # doc_ids that appear in a grade source but have no usable grades.
    ungraded_doc_ids: list[str] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)

    # --- basic views ---
    @property
    def doc_ids(self) -> list[str]:
        return sorted({i.doc_id for i in self.items})

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    def for_bucket(self, bucket: str) -> list[GradeItem]:
        return [i for i in self.items if i.bucket == bucket]

    def for_doc(self, doc_id: str) -> list[GradeItem]:
        did = settings.normalize_doc_id(doc_id)
        return [i for i in self.items if i.doc_id == did]

    def for_field(self, field: str) -> list[GradeItem]:
        return [i for i in self.items if i.field == field]

    def get(self, doc_id: str, field: str) -> Optional[GradeItem]:
        did = settings.normalize_doc_id(doc_id)
        for i in self.items:
            if i.doc_id == did and i.field == field:
                return i
        return None

    # --- conformal-prediction views (MCAL_PLAN 3.3) ---
    def wrong_docs(self, bucket: str) -> list[str]:
        """
        doc_ids with >= 1 wrong item in `bucket`.

        This is the per-bucket calibration set: MCAL_PLAN 3.3 restricts both
        tau_raw and the leave-one-doc-out curation slack to these docs.
        """
        return sorted({i.doc_id for i in self.for_bucket(bucket) if not i.correct})

    def n_wrong_docs(self, bucket: str) -> int:
        return len(self.wrong_docs(bucket))

    def wrong_items(self, bucket: str, doc_id: str) -> list[GradeItem]:
        did = settings.normalize_doc_id(doc_id)
        return [
            i for i in self.for_bucket(bucket) if i.doc_id == did and not i.correct
        ]

    # --- taxonomy-induction view (MCAL_PLAN 3.1) ---
    def induction_rows(self) -> list[dict]:
        """
        Rows for TnT-LLM failure-mode induction: every non-correct grade plus
        every grade carrying a free-text note, with the human's original
        wording preserved.
        """
        rows = []
        for i in self.items:
            if i.correct and not i.notes:
                continue
            rows.append(
                {
                    "doc_id": i.doc_id,
                    "field": i.field,
                    "bucket": i.bucket,
                    "raw_grade": i.raw_grade,
                    "notes": i.notes,
                    "seed_tags": list(i.failure_tags),
                }
            )
        return rows

    def tag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for i in self.items:
            for t in i.failure_tags:
                counts[t] = counts.get(t, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def summary(self) -> dict:
        """Roll-up for calibration_report.v(N).md."""
        per_bucket = {}
        for b in settings.BUCKET_ORDER:
            items = self.for_bucket(b)
            per_bucket[b] = {
                "n_items": len(items),
                "n_wrong_items": sum(1 for i in items if not i.correct),
                "n_wrong_docs": self.n_wrong_docs(b),
                "wrong_docs": self.wrong_docs(b),
            }
        return {
            "n_docs": self.n_docs,
            "n_items": len(self.items),
            "n_wrong_items": sum(1 for i in self.items if not i.correct),
            "granularity": sorted({i.granularity for i in self.items}),
            "sources": sorted({i.source for i in self.items}),
            "per_bucket": per_bucket,
            "tag_counts": self.tag_counts(),
            "ungraded_doc_ids": self.ungraded_doc_ids,
            "warnings": self.warnings,
        }


def load_evaluation_sheet(path: Optional[Path] = None) -> GradeSet:
    """
    Parse the transposed, free-text Evaluation sheet.

    Layout: row 1 is `ID` + doc_ids, row 2 is `slug`, rows 3..N are field
    grades, and there is a doc-level `notes` row plus a trailing free-text
    "Overall Notes:" row.
    """
    p = path or settings.EVALUATION_CSV
    if not p.exists():
        raise FileNotFoundError(f"Evaluation sheet not found: {p}")

    with open(p, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh)]

    if not rows:
        raise ValueError(f"Evaluation sheet is empty: {p}")

    gs = GradeSet()

    # --- header: doc_ids ---
    header = rows[0]
    if (header[0] or "").strip().lower() != "id":
        gs.warnings.append(
            f"Evaluation sheet row 1 starts with {header[0]!r}, expected 'ID'; "
            "parsing positionally anyway."
        )
    doc_cols: dict[int, str] = {}
    for idx, cell in enumerate(header[1:], start=1):
        did = settings.normalize_doc_id(cell)
        if did:
            doc_cols[idx] = did

    if not doc_cols:
        raise ValueError(f"No doc_id columns found in {p}")

    # --- body ---
    doc_notes: dict[str, str] = {}
    field_rows: list[tuple[str, list[str]]] = []
    for r in rows[1:]:
        if not r:
            continue
        label = (r[0] or "").strip()
        low = label.lower()
        if low.startswith("overall notes"):
            gs.overall_notes = label
            continue
        if not label or not any((c or "").strip() for c in r[1:]):
            continue
        if low == "slug":
            for idx, did in doc_cols.items():
                if idx < len(r):
                    gs.slugs[did] = (r[idx] or "").strip()
            continue
        if low == "notes":
            for idx, did in doc_cols.items():
                if idx < len(r):
                    doc_notes[did] = (r[idx] or "").strip()
            continue
        field_rows.append((label, r))

    gs.doc_notes = doc_notes

    # --- grades ---
    seen_labels: set[str] = set()
    for label, r in field_rows:
        fld = canonical_field(label)
        if fld is None:
            gs.warnings.append(
                f"Evaluation sheet row {label!r} did not map to a canonical "
                "field key; skipped."
            )
            continue
        if fld in seen_labels:
            gs.warnings.append(
                f"Field {fld!r} appears in more than one Evaluation sheet row "
                f"(latest label {label!r}); later rows overwrite earlier ones."
            )
        seen_labels.add(fld)

        bucket = settings.bucket_for_field(fld)
        for idx, did in doc_cols.items():
            raw = (r[idx] or "").strip() if idx < len(r) else ""
            correct, tags = classify_grade(fld, raw)
            if correct is None:
                continue
            gs.items.append(
                GradeItem(
                    doc_id=did,
                    field=fld,
                    bucket=bucket,
                    correct=correct,
                    raw_grade=raw,
                    failure_tags=tags,
                    acronym_issue=_has_acronym_issue(doc_notes.get(did, "")),
                    notes=doc_notes.get(did, ""),
                    granularity="coarse",
                    source="evaluation_sheet",
                    item_key=label,
                )
            )

    graded = {i.doc_id for i in gs.items}
    for did in doc_cols.values():
        if did not in graded:
            gs.ungraded_doc_ids.append(did)

    return gs


# --- Grading-sheet loader (v2 onward) ---------------------------------------


def _read_grading_sheet(path: Path) -> tuple[dict[str, str], list[dict]]:
    """
    Read one per-doc grading sheet.

    The file has 6 `#`-prefixed comment lines carrying doc_id / work_id /
    title, then a blank line, then the CSV header. csv.DictReader cannot see
    past that preamble, so strip it first.
    """
    meta: dict[str, str] = {}
    body: list[str] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                k, _, v = line[1:].partition(":")
                meta[k.strip().lower()] = v.strip()
                continue
            if not line.strip() and not body:
                continue
            body.append(line)
    if not body:
        return meta, []
    return meta, list(csv.DictReader(body))


def load_grading_sheets(directory: Optional[Path] = None) -> GradeSet:
    """
    Load item-level grades from per-doc grading sheets.

    Only rows with a filled `your_grade` produce a GradeItem. Because those
    sheets are currently 100% unfilled this returns an empty GradeSet today;
    it becomes the primary source from v2 once the next batch is graded.
    """
    d = directory or settings.GRADING_SHEETS_DIR
    gs = GradeSet()
    if not d.exists():
        gs.warnings.append(f"Grading-sheet directory does not exist: {d}")
        return gs

    for path in sorted(d.glob("*.csv")):
        meta, rows = _read_grading_sheet(path)
        doc_id = settings.normalize_doc_id(meta.get("doc_id") or path.stem)
        if not rows:
            gs.warnings.append(f"No data rows in grading sheet {path.name}")
            continue

        n_graded = 0
        for row in rows:
            raw = (row.get("your_grade") or "").strip()
            if not raw:
                continue
            label = (row.get("field") or "").strip()
            fld = canonical_field(label)
            if fld is None:
                gs.warnings.append(
                    f"{path.name}: field {label!r} did not map to a canonical key"
                )
                continue
            correct, tags = classify_grade(fld, raw)
            if correct is None:
                continue
            explicit = (row.get("your_failure_tag") or "").strip()
            if explicit:
                tags = _dedupe([explicit] + tags)
            n_graded += 1
            gs.items.append(
                GradeItem(
                    doc_id=doc_id,
                    field=fld,
                    bucket=settings.bucket_for_field(fld),
                    correct=correct,
                    raw_grade=raw,
                    failure_tags=tags,
                    notes=(row.get("your_notes") or "").strip(),
                    granularity="item",
                    source="grading_sheet",
                    item_key=label,
                )
            )
        if n_graded == 0:
            gs.ungraded_doc_ids.append(doc_id)

    return gs


# --- Merge ------------------------------------------------------------------


def load_grades(
    *,
    evaluation_csv: Optional[Path] = None,
    grading_sheets_dir: Optional[Path] = None,
    prefer: str = "item",
) -> GradeSet:
    """
    Load all available human grades.

    Item-level grading-sheet rows take precedence over coarse Evaluation-sheet
    cells for the same (doc_id, field); when a doc is graded at item level, its
    coarse grades for those fields are dropped rather than double-counted.

    An item-level (doc, field) may contribute SEVERAL GradeItems -- one per
    array element, e.g. location.places[0] and location.places[1] both fold
    into field `location`. That is intended: MCAL_PLAN 3.3's per-doc
    nonconformity is `max{s_i : y_i = 0}` over items in the bucket, so finer
    granularity strictly sharpens the calibration set.
    """
    coarse = load_evaluation_sheet(evaluation_csv)
    fine = load_grading_sheets(grading_sheets_dir)

    merged = GradeSet(
        doc_notes=dict(coarse.doc_notes),
        slugs=dict(coarse.slugs),
        overall_notes=coarse.overall_notes,
        warnings=list(coarse.warnings) + list(fine.warnings),
    )

    if prefer == "item":
        shadowed = {(i.doc_id, i.field) for i in fine.items}
        merged.items = [i for i in coarse.items if (i.doc_id, i.field) not in shadowed]
        merged.items += fine.items
    else:
        shadowed = {(i.doc_id, i.field) for i in coarse.items}
        merged.items = list(coarse.items)
        merged.items += [i for i in fine.items if (i.doc_id, i.field) not in shadowed]

    graded_docs = {i.doc_id for i in merged.items}
    merged.ungraded_doc_ids = sorted(
        set(coarse.ungraded_doc_ids) | set(fine.ungraded_doc_ids)
    ) 
    merged.ungraded_doc_ids = [d for d in merged.ungraded_doc_ids if d not in graded_docs]

    # Surface the n=8-vs-9 discrepancy loudly: MCAL_PLAN 0 and 6 both assume
    # n=9 at seed v1, but one doc has a grading sheet and no Evaluation column.
    on_disk = {settings.normalize_doc_id(d) for d in settings.available_doc_ids()}
    missing = sorted(on_disk - graded_docs)
    if missing:
        merged.warnings.append(
            f"{len(graded_docs)} of {len(on_disk)} docs with OCR on disk are "
            f"graded. Ungraded: {missing}. MCAL_PLAN 0 assumes n=9 at seed v1; "
            f"the actual seed n is {len(graded_docs)}."
        )

    merged.warnings.extend(_stale_label_warnings(graded_docs))

    return merged


def _stale_label_warnings(graded_docs: set[str]) -> list[str]:
    """
    Warn when the extraction artifacts are NEWER than the grades that label them.

    This is a real hazard the plan does not address. MCAL_PLAN 5 item #4 makes
    re-running M2 a hard prerequisite so that tau is fitted to the prose Segment B
    will actually ship. But re-running M2 REWRITES the very text a human graded:
    a label like "ok, missing citation - pg 190" describes one specific
    extraction, and after a rerun that extraction no longer exists. Pairing new
    artifacts with old labels silently corrupts the (composite, y) pairs that
    tau is fitted on.

    Measured after the build-item-#4 rerun: all 40 graded summary subfields were
    rewritten, and the s_quote gap between missing-citation and clean subfields
    moved from +0.028 (old artifacts, consistent) to -0.005 (new artifacts, stale
    labels) -- i.e. the signal inverted into noise.

    A file-mtime comparison is crude but catches the case that matters, and it
    catches it every round rather than only when someone remembers.
    """
    warnings: list[str] = []
    if not settings.EVALUATION_CSV.exists():
        return warnings

    grade_mtime = settings.EVALUATION_CSV.stat().st_mtime
    newer: list[str] = []
    for doc_id in sorted(graded_docs):
        m2 = settings.M2_DIR / f"{doc_id}.json"
        if m2.exists() and m2.stat().st_mtime > grade_mtime:
            newer.append(doc_id)

    if newer:
        warnings.append(
            f"STALE LABELS: {len(newer)}/{len(graded_docs)} graded docs have M2 "
            f"output NEWER than the grade source "
            f"({settings.EVALUATION_CSV.name}). Those grades describe prose that "
            f"has since been regenerated, so the (composite, y) pairs used to fit "
            f"tau are only approximately valid. This is expected immediately "
            f"after the MCAL_PLAN 5 item #4 M2 rerun and is not fixable by code: "
            f"either re-grade these docs against the current output, or accept "
            f"that this stage's thresholds are indicative and treat the next "
            f"stage -- graded under the shipped pipeline -- as the first "
            f"internally consistent one. Affected: {newer}"
        )
    return warnings
