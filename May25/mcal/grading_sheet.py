"""
Blinded, tag-aware grading sheets (MCAL_PLAN 7 Q5, 6; build item #14).

Replaces `segment_a/grading.py`'s single-sheet output for M-Cal-era grading
rounds. `segment_a/grading.py` is untouched and still drives
`segment_a/run.py process` / `run.py grade`; this module is additive and imports
`_short` from it so the two agree on value rendering.

Four defects in the existing sheet, each with the plan reference that requires
fixing it:

1. **No `your_failure_tag` column.** MCAL_PLAN 7 Q5 requires the reviewer to fill
   `your_grade` AND `your_failure_tag`. `mcal/grades.py:load_grading_sheets`
   already reads `your_failure_tag` (grades.py:557) and prefers it over its own
   regex tag inference -- so the column the loader treats as authoritative has
   never existed in the file. Added, with the per-field tag vocabulary printed in
   the preamble so the reviewer does not have to hold 20 codes in their head or
   open `taxonomy.v(N).json`.

2. **No `soi_useful` column.** MCAL_PLAN 7 Q5 and 6 make this the real acceptance
   test for `summary_of_interest`: "did `summary_of_interest` tell me something
   the standard summary didn't?" No automated metric substitutes for it, and the
   field is new so there is no baseline to fall back on.

3. **Blinding is broken.** MCAL_PLAN 7 Q5 wants a first pass WITHOUT
   `critic_verdict`, then a second pass revealing it for meta-analysis. The
   current sheet not only shows `critic_verdict` as a visible column, it
   PRE-POPULATES `your_notes` with the Critic's own notes (grading.py:118, :137,
   :175, :197, :231, :279). A reviewer whose notes field already argues a
   position is not an independent labeler, and the whole point of the blind pass
   is that `s_critic` and the human grade must be independent -- they are the two
   things whose agreement MCAL_PLAN 3.3 uses to fit tau. Anchoring the human to
   the Critic inflates that agreement and biases every threshold downstream.

   Fixed by emitting TWO sheets into two directories: `blind/{doc_id}.csv` with no
   `critic_verdict`, no `critic_notes`, and an EMPTY `your_notes`; and
   `reveal/{doc_id}.csv` with everything. Critic notes in the reveal sheet live in
   their own `critic_notes` column, never in `your_notes`.

   Two directories rather than two filenames in one directory because
   `mcal/grades.py:load_grading_sheets` does a non-recursive `glob("*.csv")` and
   would otherwise load both sheets for the same document and count every grade
   twice, silently doubling `n_wrong_items` in each bucket.

4. **`_short()` truncates the values that matter.** MCAL_PLAN 6 acceptance item 4
   requires the reviewer to be able to grade a field WITHOUT opening another file.
   `grading.py` caps `extracted_value` at 400 chars for `alternatives`/`themes`,
   300 for `location.places[i]` and `key_people.*[i]`, and quotes at 300 chars x
   10 quotes. A location entry truncated mid-JSON cannot be graded, and a
   truncated quote cannot be checked against the page. This module truncates
   nothing in `extracted_value` or `quote`; only `_short`'s whitespace collapsing
   is retained, which is needed to keep a cell on one CSV row.

Row keys reuse `grading.py`'s dotted/bracketed convention (`location.places[0]`,
`key_people.cooperating_agencies[2]`, `alternatives[0]`, `summary.public_response`)
so `mcal/grades.py:canonical_field` parses them unchanged. Round-tripping is
asserted in `tests/test_grading_sheet.py`, not assumed.

Two input paths, because the reviewer's source changes with the round:
  * `rows_from_extractions(m1, m2, critic)` -- the segment_a JSONs, same inputs
    `grading.write_grading_sheet` takes.
  * `rows_from_manifest(manifest)` -- `run_manifest.json` from
    `segment_b/gate.py`. This is the important one at seed v1: nearly every
    bucket is `degenerate_severe`, so nearly every field routes to HUMAN_REVIEW
    (MCAL_PLAN 0, 7 Q1), and the manifest is what carries the raw extractions the
    reviewer grades (MCAL_PLAN 7 Q8).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from . import quote_check as qc
from . import settings
from . import taxonomy as taxonomy_mod

# segment_a bridge is installed by the settings import. `_short` is imported
# rather than reimplemented so both sheet generations render a dict or list
# identically -- otherwise a v1 sheet and a v2 sheet of the same document would
# differ in ways that look like extraction changes.
from grading import GRADE_OPTIONS, _short  # noqa: E402
from pages import Doc  # noqa: E402

log = logging.getLogger(__name__)


# --- Columns ----------------------------------------------------------------

# Shared columns, in reviewer reading order: what was extracted, what supports
# it, then the reviewer's own three cells.
BASE_COLUMNS = (
    "field",
    "extracted_value",
    "quote",
    "source_pages",
    "quote_verified",
    "model_confidence",
    "your_grade",
    "your_failure_tag",
    "your_notes",
    "soi_useful",
)

# Columns only the reveal sheet carries. `critic_notes` is a NEW column, not a
# reuse of `your_notes` -- see defect 3 in the module docstring.
CRITIC_COLUMNS = ("critic_verdict", "critic_failure_tag", "critic_notes")

BLIND_COLUMNS = BASE_COLUMNS
REVEAL_COLUMNS = (
    BASE_COLUMNS[:5] + CRITIC_COLUMNS + BASE_COLUMNS[5:]
)

PASS_BLIND = "blind"
PASS_REVEAL = "reveal"
PASSES = (PASS_BLIND, PASS_REVEAL)

# Reviewer cells. Never pre-populated in either sheet.
REVIEWER_COLUMNS = ("your_grade", "your_failure_tag", "your_notes", "soi_useful")

# `soi_useful` is a DOC-level answer living in a per-row CSV. It is emitted as a
# column on every row (blank), and the preamble directs the reviewer to fill it
# once, on the `summary_of_interest` row. A dedicated pseudo-field row was the
# alternative and was rejected: `grades.canonical_field` would return None for it
# and `load_grading_sheets` would emit a warning for every sheet.
SOI_OPTIONS = "yes|no"


def columns_for(pass_name: str) -> tuple[str, ...]:
    if pass_name == PASS_BLIND:
        return BLIND_COLUMNS
    if pass_name == PASS_REVEAL:
        return REVEAL_COLUMNS
    raise ValueError(f"Unknown pass {pass_name!r}; expected one of {PASSES}")


# --- Row model --------------------------------------------------------------


@dataclass
class SheetRow:
    """
    One gradable row. `field` is the dotted/bracketed path, not a display label.

    Critic material is carried on the row but only rendered into the reveal
    sheet, so a single row list produces both passes and the two cannot drift.
    """

    field: str
    extracted_value: str = ""
    quote: str = ""
    source_pages: str = ""
    quote_verified: str = ""
    model_confidence: str = ""
    critic_verdict: str = ""
    critic_failure_tag: str = ""
    critic_notes: str = ""

    @property
    def canonical_field(self) -> Optional[str]:
        """The canonical key `mcal/grades.py` will fold this row into."""
        from . import grades as grades_mod

        return grades_mod.canonical_field(self.field)

    def to_dict(self, pass_name: str) -> dict:
        cols = columns_for(pass_name)
        base = {
            "field": self.field,
            "extracted_value": self.extracted_value,
            "quote": self.quote,
            "source_pages": self.source_pages,
            "quote_verified": self.quote_verified,
            "model_confidence": self.model_confidence,
            "critic_verdict": self.critic_verdict,
            "critic_failure_tag": self.critic_failure_tag,
            "critic_notes": self.critic_notes,
        }
        row = {c: base.get(c, "") for c in cols}
        for c in REVIEWER_COLUMNS:
            if c in row:
                row[c] = ""
        return row


# --- Value rendering --------------------------------------------------------
# `_short` with an effectively unbounded cap: keep its whitespace collapsing and
# dict/list JSON rendering, drop its truncation (defect 4).

_NO_TRUNCATION = 10 ** 9


def render_value(value: Any) -> str:
    """Full, single-line rendering of an extracted value. Never truncated."""
    return _short(value, _NO_TRUNCATION)


def _drop_evidence(value: Any) -> Any:
    """
    Strip `evidence` blocks from a display value.

    The quotes are rendered into their own `quote` column, so leaving them inside
    `extracted_value` too would double every row's width for no added information
    -- and it is the width, not the truncation, that makes the current sheet hard
    to read in a spreadsheet.
    """
    if isinstance(value, dict):
        return {k: _drop_evidence(v) for k, v in value.items() if k != "evidence"}
    if isinstance(value, list):
        return [_drop_evidence(v) for v in value]
    return value


def render_evidence(
    evidence: Sequence[dict], *, doc: Optional[Doc] = None
) -> tuple[str, str, str]:
    """
    `(quote, source_pages, quote_verified)` for one row's evidence list.

    Each quote is prefixed with its own page tag, as `grading.py` does, so a row
    carrying several quotes still shows which page each came from. Nothing is
    truncated and no quote is dropped (`grading.py` keeps only the first 10).

    When `doc` is supplied, `quote_verified` is RECOMPUTED with
    `mcal/quote_check.py` instead of trusting the `quote_verified` boolean
    segment_a stored. That boolean comes from `pages.find_quote`, an exact
    whitespace-collapsed substring search, which MCAL_PLAN 3.2 replaced precisely
    because it reports OCR-damaged-but-present quotes as absent. Grading against
    the weaker verifier would have the reviewer adjudicating false alarms.
    Verdicts are the three-valued `yes|mixed|no` the column already uses.
    """
    if not evidence:
        return "", "", ""

    pages: list[str] = []
    chunks: list[str] = []
    verdicts: list[str] = []

    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        raw_pages = ev.get("source_pages") or []
        ev_pages = [str(p) for p in raw_pages]
        pages.extend(ev_pages)
        quote = (ev.get("quote") or "").strip()

        if doc is not None:
            check = qc.check_quote(quote, raw_pages, doc)
            verdict = check.verified
        else:
            verdict = "yes" if ev.get("quote_verified") else "no"
        verdicts.append(verdict)

        if quote:
            tag = f"[p.{', '.join(ev_pages)}] " if ev_pages else "[no page cited] "
            flag = "" if verdict == "yes" else f"[{verdict.upper()}] "
            chunks.append(tag + flag + " ".join(quote.split()))

    pages_str = ", ".join(dict.fromkeys(pages))
    quote_str = " | ".join(chunks)
    if not verdicts:
        rolled = ""
    elif set(verdicts) == {"yes"}:
        rolled = "yes"
    elif set(verdicts) == {"no"}:
        rolled = "no"
    else:
        rolled = "mixed"
    return quote_str, pages_str, rolled


# --- Field expansion --------------------------------------------------------
# One canonical field can produce many rows. The paths below are exactly
# `grading.py`'s, so `grades.canonical_field` folds them back correctly:
#   alternatives[0] -> alternatives
#   location.places[0], location.summary -> location
#   key_people.cooperating_agencies[2] -> key_people
#   summary_of_interest[0] -> summary_of_interest

KEY_PEOPLE_BUCKETS = (
    "agency_preparers",
    "cooperating_agencies",
    "consulted_entities",
    "public_commenters",
)


def expand_field(field: str, value: Any) -> list[tuple[str, Any]]:
    """
    `[(row_key, display_value)]` for one canonical field's unwrapped value.

    An empty collection still yields one row with an empty value rather than no
    rows. That is load-bearing: `alternatives == []` is the MCAL_PLAN 1(8)
    Buffalo failure and `summary_of_interest == []` is a legitimate MCAL_PLAN 3.15
    result, and a reviewer cannot grade a row that is not in the file. Emitting
    nothing would make "the extractor returned empty" indistinguishable from "this
    document was not graded on that field".
    """
    if field == "alternatives":
        items = value if isinstance(value, list) else []
        if not items:
            return [("alternatives", [])]
        return [
            (f"alternatives[{i}]", _alt_display(a))
            for i, a in enumerate(items)
        ]

    if field == "location":
        if not isinstance(value, dict):
            return [("location", value)]
        places = value.get("places")
        places = places if isinstance(places, list) else []
        if not places:
            return [("location", _drop_evidence(value))]
        rows: list[tuple[str, Any]] = [
            (f"location.places[{i}]", _drop_evidence(p))
            for i, p in enumerate(places)
        ]
        rows.append(
            (
                "location.summary",
                {
                    k: v
                    for k, v in value.items()
                    if k not in ("places", "evidence")
                },
            )
        )
        return rows

    if field == "key_people":
        if not isinstance(value, dict):
            return [("key_people", value)]
        rows = []
        buckets = list(KEY_PEOPLE_BUCKETS)
        # Any role bucket the pipeline added after this list was written still
        # gets a row rather than being silently dropped. Scalars are NOT buckets:
        # key_people_pipeline.py attaches `comment_response_present: bool`, and
        # rendering that as an empty entry row invites a grade on a row with
        # nothing in it.
        scalars: dict[str, Any] = {}
        for extra, extra_val in value.items():
            if extra in buckets:
                continue
            if isinstance(extra_val, list):
                buckets.append(extra)
            elif extra != "evidence":
                scalars[extra] = extra_val
        for bucket in buckets:
            entries = value.get(bucket)
            entries = entries if isinstance(entries, list) else []
            if not entries:
                rows.append((f"key_people.{bucket}", ""))
                continue
            for i, e in enumerate(entries):
                rows.append((f"key_people.{bucket}[{i}]", _drop_evidence(e)))
        if scalars:
            rows.append(("key_people.summary", scalars))
        return rows

    if field == settings.SUMMARY_OF_INTEREST:
        items = value if isinstance(value, list) else []
        if not items:
            return [(settings.SUMMARY_OF_INTEREST, [])]
        return [
            (f"{settings.SUMMARY_OF_INTEREST}[{i}]", _drop_evidence(e))
            for i, e in enumerate(items)
        ]

    return [(field, _drop_evidence(value))]


def _alt_display(a: Any) -> Any:
    if not isinstance(a, dict):
        return a
    return {k: v for k, v in a.items() if k != "evidence"}


def item_evidence(value: Any) -> list[dict]:
    """
    Evidence dicts attached directly to one item (not to its children).

    Shallow on purpose: a `location.places[0]` row shows that place's own quotes,
    and pulling in nested evidence would attribute a neighbouring place's support
    to it.
    """
    if isinstance(value, dict):
        ev = value.get("evidence")
        if isinstance(ev, list):
            return [e for e in ev if isinstance(e, dict)]
        if isinstance(ev, dict):
            return [ev]
    return []


# --- Critic lookup ----------------------------------------------------------


def _critic_entry(critic: Optional[dict], field: str) -> dict:
    """
    One field's Critic result, tolerant of both critic shapes in this repo.

    `segment_a/critic.py` keys coarsely (`summary`, `alternatives`, `location`,
    `key_people`) and reports `{verdict, notes, model_confidence}`.
    `segment_b/critic.py:as_dict` keys by canonical field and reports
    `{verdict, failure_tag, note, ...}`. Both are accepted so a sheet can be
    rebuilt from whichever run produced the JSON on disk.
    """
    if not isinstance(critic, dict):
        return {}
    if field in critic and isinstance(critic[field], dict):
        return critic[field]
    if field.startswith("summary.") and isinstance(critic.get("summary"), dict):
        return critic["summary"]
    head = field.split(".", 1)[0].split("[", 1)[0]
    entry = critic.get(head)
    return entry if isinstance(entry, dict) else {}


def _critic_cells(entry: dict) -> tuple[str, str, str]:
    verdict = str(entry.get("verdict") or "")
    tag = str(entry.get("failure_tag") or "")
    notes = entry.get("note")
    if notes is None:
        notes = entry.get("notes")
    return verdict, tag, render_value(notes)


def _model_confidence(entry: Any, critic_entry: dict) -> str:
    if isinstance(entry, dict) and entry.get("confidence"):
        return str(entry["confidence"])
    return str(critic_entry.get("model_confidence") or "")


# --- Rows from segment_a extractions ---------------------------------------


def rows_from_extractions(
    m1: Optional[dict],
    m2: Optional[dict],
    critic: Optional[dict] = None,
    *,
    doc: Optional[Doc] = None,
    fields: Sequence[str] = settings.ALL_FIELDS,
) -> list[SheetRow]:
    """
    Build rows from the segment_a M1/M2/Critic JSONs.

    Covers all 15 canonical fields in `settings.ALL_FIELDS`, including
    `summary_of_interest`, which `grading.py:build_rows` predates and omits. Field
    unwrapping goes through `segment_b/critic.py`'s `extracted_entry` /
    `extracted_value` / `evidence_dicts` so that the sheet shows exactly the value
    the Critic judged and the gate will emit -- three separate unwrappers for the
    same 15 shapes is how they drift.

    That import inverts the usual dependency direction (segment_b depends on mcal,
    not the reverse), so it is deliberately LAZY and confined to this function: it
    never runs at import time, so there is no cycle, and a caller that only needs
    the manifest path never loads segment_b at all. The alternative -- a fourth
    copy of the unwrapper -- costs more than the inversion does. Contrast
    `rows_from_manifest`, which does NOT import `segment_b/gate.py` for its
    reserved-key list: that rule is a single character (`_`) and gate.py is a
    heavy import to pull in for it.
    """
    from segment_b import critic as critic_mod

    rows: list[SheetRow] = []
    for field in fields:
        entry = critic_mod.extracted_entry(field, m1, m2)
        value = critic_mod.extracted_value(field, entry)
        c = _critic_entry(critic, field)
        verdict, tag, notes = _critic_cells(c)
        conf = _model_confidence(entry, c)

        # Field-level evidence, used for rows that have none of their own
        # (the summary subfields, themes, M1 fields).
        field_ev = critic_mod.evidence_dicts(field, entry)

        expanded = expand_field(field, value)
        for i, (row_key, display) in enumerate(expanded):
            ev = item_evidence(_raw_item(field, value, i)) or (
                field_ev if len(expanded) == 1 else []
            )
            quote, pages, verified = render_evidence(ev, doc=doc)
            if field in settings.M1_FIELDS and not pages:
                # M1 values are not verbatim quotes; show provenance instead so
                # the cell is not simply blank (grading.py:_pages_only).
                pages = _m1_sources(entry)
            rows.append(
                SheetRow(
                    field=row_key,
                    extracted_value=render_value(display),
                    quote=quote,
                    source_pages=pages,
                    quote_verified=verified,
                    model_confidence=conf,
                    critic_verdict=verdict,
                    critic_failure_tag=tag,
                    critic_notes=notes,
                )
            )
    return rows


def _raw_item(field: str, value: Any, index: int) -> Any:
    """The un-stripped source item behind `expand_field`'s row `index`."""
    if field == "alternatives" and isinstance(value, list):
        return value[index] if index < len(value) else None
    if field == settings.SUMMARY_OF_INTEREST and isinstance(value, list):
        return value[index] if index < len(value) else None
    if field == "location" and isinstance(value, dict):
        places = value.get("places")
        if isinstance(places, list) and index < len(places):
            return places[index]
        return None
    if field == "key_people" and isinstance(value, dict):
        flat: list[Any] = []
        buckets = list(KEY_PEOPLE_BUCKETS)
        for extra, extra_val in value.items():
            if extra not in buckets and isinstance(extra_val, list):
                buckets.append(extra)
        for bucket in buckets:
            entries = value.get(bucket)
            entries = entries if isinstance(entries, list) else []
            if not entries:
                flat.append(None)
                continue
            flat.extend(entries)
        return flat[index] if index < len(flat) else None
    return value


def _m1_sources(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    src = entry.get("sources")
    if isinstance(src, list):
        return ", ".join(str(s) for s in src)
    if isinstance(src, dict):
        return "; ".join(f"{k}: {', '.join(v)}" for k, v in src.items() if v)
    return str(src or "")


# --- Rows from run_manifest.json -------------------------------------------

# `segment_b/gate.py` puts doc-level material under underscore-prefixed reserved
# keys (`_meta`, `_rollup`, `_null_tag_monitor`) and asserts that no canonical
# field name starts with `_`. Testing the prefix rather than importing gate.py's
# RESERVED_KEYS keeps the dependency direction one-way: segment_b imports mcal,
# not the reverse.
MANIFEST_RESERVED_PREFIX = "_"


def manifest_fields(manifest: dict) -> list[str]:
    """Canonical field keys in a `run_manifest.json`, in plan order where known."""
    keys = [
        k
        for k in (manifest or {})
        if not k.startswith(MANIFEST_RESERVED_PREFIX)
        and isinstance(manifest[k], dict)
    ]
    order = {f: i for i, f in enumerate(settings.ALL_FIELDS)}
    return sorted(keys, key=lambda k: (order.get(k, len(order)), k))


def rows_from_manifest(
    manifest: dict, *, doc: Optional[Doc] = None
) -> list[SheetRow]:
    """
    Build rows from `segment_b/gate.py`'s `run_manifest.json` (MCAL_PLAN 3.12).

    This is the seed-v1 and v2 path. Almost every field is gated to HUMAN_REVIEW
    at those stages, so the manifest -- not a Critic PASS -- is what the reviewer
    grades from, and MCAL_PLAN 7 Q8 guarantees it carries the raw extraction for
    exactly this purpose.

    Two honest limitations, both surfaced in the sheet rather than hidden:

    * The manifest stores ONE representative `evidence_quote` and `source_pages`
      per field, not per item. Where the extracted value still carries nested
      `evidence` blocks -- `location.places[i]`, `key_people.*[i]`,
      `alternatives[i]`, `summary_of_interest[i]`, all of which survive
      `critic.extracted_value` -- per-item evidence is recovered from it. Where it
      does not, the item rows share the field-level quote, and `source_pages`
      carries `(field-level)` so the reviewer knows the cite was not
      item-specific.
    * `model_confidence` does not exist in the 3.12 schema. The nearest
      equivalent is the gate's own `composite`, which is what the confidence
      machinery actually acts on, so that is what the column carries, prefixed
      `composite=` so it is never mistaken for the extractor's self-reported
      confidence.
    """
    rows: list[SheetRow] = []
    for field in manifest_fields(manifest):
        entry = manifest[field] or {}
        value = entry.get("extracted_value")
        verdict = str(entry.get("verdict") or "")
        tag = str(entry.get("failure_tag") or "")
        notes = _manifest_notes(entry)
        conf = _manifest_confidence(entry)

        field_ev: list[dict] = []
        if entry.get("evidence_quote"):
            field_ev = [
                {
                    "quote": entry.get("evidence_quote"),
                    "source_pages": entry.get("source_pages") or [],
                    "quote_verified": entry.get("quote_verdict") in (None, "yes"),
                }
            ]

        expanded = expand_field(field, value)
        for i, (row_key, display) in enumerate(expanded):
            own = item_evidence(_raw_item(field, value, i))
            if own:
                quote, pages, verified = render_evidence(own, doc=doc)
            else:
                quote, pages, verified = render_evidence(field_ev, doc=doc)
                if pages and len(expanded) > 1:
                    pages = f"{pages} (field-level)"
            rows.append(
                SheetRow(
                    field=row_key,
                    extracted_value=render_value(display),
                    quote=quote,
                    source_pages=pages,
                    quote_verified=verified,
                    model_confidence=conf,
                    critic_verdict=verdict,
                    critic_failure_tag=tag,
                    critic_notes=notes,
                )
            )
    return rows


def _manifest_notes(entry: dict) -> str:
    """Critic note plus the gate's reason, which the reveal pass wants together."""
    parts = []
    if entry.get("note"):
        parts.append(str(entry["note"]))
    if entry.get("gate_reason"):
        parts.append(f"gate_reason={entry['gate_reason']}")
    if entry.get("gated_to_human"):
        parts.append("gated_to_human=true")
    return render_value("; ".join(parts)) if parts else ""


def _manifest_confidence(entry: dict) -> str:
    comp = entry.get("composite")
    if comp is None:
        return ""
    try:
        return f"composite={float(comp):.3f}"
    except (TypeError, ValueError):
        return ""


# --- Preamble ---------------------------------------------------------------
# `#`-prefixed lines then a blank line then the header, matching
# `grading.py:write_grading_sheet` -- `grades.py:_read_grading_sheet` parses that
# shape by splitting each comment on the first ":".


def tag_vocabulary_lines(
    rows: Sequence[SheetRow],
    tax: Optional["taxonomy_mod.Taxonomy"] = None,
) -> list[str]:
    """
    One `# tags[<field>]: ...` line per canonical field present in the sheet.

    Required by MCAL_PLAN 7 Q5: the reviewer fills `your_failure_tag`, so the
    allowed vocabulary has to be in front of them. Per FIELD rather than one
    global list, because the vocabulary is field-scoped in `taxonomy.py`
    (`Tag.applies_to`) and `critic_prompt.py` restricts the Critic the same way --
    offering the reviewer `T06_geocode_missing` on a summary row invites an
    off-field tag, which pollutes the null-tag monitor that MCAL_PLAN 6 uses to
    decide when the taxonomy needs new codes.

    Falls back to the in-code seed taxonomy when no artifact has been built:
    `mcal/artifacts/` does not exist until `mcal/build.py` runs and a human
    ratifies the draft (MCAL_PLAN 3.7), and a sheet that cannot be produced before
    the first build would make the first build ungradable.
    """
    t = tax if tax is not None else _fallback_taxonomy()
    seen: list[str] = []
    for r in rows:
        cf = r.canonical_field
        if cf and cf not in seen:
            seen.append(cf)
    order = {f: i for i, f in enumerate(settings.ALL_FIELDS)}
    seen.sort(key=lambda f: (order.get(f, len(order)), f))

    lines: list[str] = []
    for field in seen:
        names = [tg.name for tg in t.for_field(field)]
        lines.append(f"# tags[{field}]: {', '.join(names) if names else '(none)'}")
    return lines


def _fallback_taxonomy() -> "taxonomy_mod.Taxonomy":
    """Promoted taxonomy if one exists, otherwise the in-code seed."""
    try:
        current = taxonomy_mod.load_current()
    except Exception as e:  # noqa: BLE001 - a sheet must still be writable
        log.debug("taxonomy load failed, using seed: %s", e)
        current = None
    return current or taxonomy_mod.seed_taxonomy("v1")


def build_preamble(
    doc_id: str,
    *,
    pass_name: str,
    rows: Sequence[SheetRow],
    work_id: Optional[str] = None,
    title: str = "",
    source: str = "",
    artifact_stage: Optional[str] = None,
    tax: Optional["taxonomy_mod.Taxonomy"] = None,
) -> list[str]:
    """The `#` comment block above the header."""
    lines = [
        f"# doc_id: {doc_id}",
        f"# work_id: {work_id or ''}",
        f"# title: {title}",
        f"# pass: {pass_name}",
        f"# source: {source}",
        f"# artifact_stage: {artifact_stage or ''}",
        f"# grade options: {GRADE_OPTIONS}",
    ]
    if pass_name == PASS_BLIND:
        lines += [
            "# BLIND PASS (MCAL_PLAN 7 Q5). critic_verdict is deliberately "
            "ABSENT and your_notes is deliberately EMPTY. Grade from "
            "extracted_value, quote and source_pages only. Do not open the "
            "reveal sheet until this one is finished: the human grade and the "
            "Critic verdict must be independent, because their agreement is what "
            "fits the conformal thresholds.",
        ]
    else:
        lines += [
            "# REVEAL PASS (MCAL_PLAN 7 Q5). critic_verdict, "
            "critic_failure_tag and critic_notes are shown for META-ANALYSIS "
            "ONLY. Fill this sheet only after the blind pass is complete. The "
            "Critic's notes are in critic_notes; your_notes is yours.",
        ]
    lines += [
        "# page numbers are EXACT (from per-page JSON source).",
        "# quote_verified: yes = quote located on the cited page +/-2 "
        "(OCR-normalized, mcal/quote_check.py); mixed = partially supported; "
        "no = not located.",
        f"# your_failure_tag: one tag from the tags[<field>] line below matching "
        f"this row's field, or blank if your_grade is 'correct'. Leave blank "
        f"rather than guessing -- an untagged defect is a signal that the "
        f"taxonomy needs a new code.",
        f"# soi_useful ({SOI_OPTIONS}): fill ONCE, on the "
        f"{settings.SUMMARY_OF_INTEREST} row. Did {settings.SUMMARY_OF_INTEREST} "
        f"tell you something the standard summary.* fields did not? An empty "
        f"{settings.SUMMARY_OF_INTEREST} is a correct result for a routine "
        f"document, so 'no' on an empty list is not a criticism of the field.",
        "# extracted_value and quote are NOT truncated (MCAL_PLAN 6 acceptance "
        "item 4: gradable without opening another file).",
    ]
    lines += tag_vocabulary_lines(rows, tax)
    return lines


# --- Writing ----------------------------------------------------------------


def pass_dir(out_dir: Path, pass_name: str) -> Path:
    if pass_name not in PASSES:
        raise ValueError(f"Unknown pass {pass_name!r}; expected one of {PASSES}")
    return Path(out_dir) / pass_name


def sheet_path(out_dir: Path, doc_id: str, pass_name: str) -> Path:
    return pass_dir(out_dir, pass_name) / f"{doc_id}.csv"


def write_sheet(
    out_dir: Path,
    doc_id: str,
    rows: Sequence[SheetRow],
    *,
    pass_name: str,
    work_id: Optional[str] = None,
    title: str = "",
    source: str = "",
    artifact_stage: Optional[str] = None,
    tax: Optional["taxonomy_mod.Taxonomy"] = None,
) -> Path:
    """Write one pass's sheet. Returns the path."""
    path = sheet_path(out_dir, doc_id, pass_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns_for(pass_name)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        for line in build_preamble(
            doc_id,
            pass_name=pass_name,
            rows=rows,
            work_id=work_id,
            title=title,
            source=source,
            artifact_stage=artifact_stage,
            tax=tax,
        ):
            fh.write(line + "\n")
        fh.write("\n")
        writer = csv.DictWriter(fh, fieldnames=list(cols))
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_dict(pass_name))
    return path


def write_sheets(
    out_dir: Path,
    doc_id: str,
    rows: Sequence[SheetRow],
    **kw,
) -> dict[str, Path]:
    """
    Write both the blind and the reveal sheet. `{pass_name: path}`.

    Separate directories -- `<out_dir>/blind/` and `<out_dir>/reveal/` -- so that
    `mcal/grades.py:load_grading_sheets`, which does a non-recursive
    `glob("*.csv")`, sees one sheet per document. Pointing it at the blind
    directory is what MCAL_PLAN 7 Q5 intends: the blind pass is the calibration
    label, the reveal pass is meta-analysis.
    """
    return {p: write_sheet(out_dir, doc_id, rows, pass_name=p, **kw) for p in PASSES}


def build_and_write(
    out_dir: Path,
    doc_id: str,
    *,
    m1: Optional[dict] = None,
    m2: Optional[dict] = None,
    critic: Optional[dict] = None,
    manifest: Optional[dict] = None,
    doc: Optional[Doc] = None,
    work_id: Optional[str] = None,
    title: str = "",
    artifact_stage: Optional[str] = None,
    tax: Optional["taxonomy_mod.Taxonomy"] = None,
) -> dict[str, Path]:
    """
    Build rows from whichever source is supplied and write both sheets.

    A manifest wins when both are given: it is the later artifact and it records
    what Segment B actually emitted under a pinned artifact stage, including the
    gate decision. Raises rather than writing an empty sheet when neither source
    is usable -- a zero-row grading sheet looks like a graded document with no
    defects, which is the most expensive possible failure mode for a calibration
    set.
    """
    if manifest:
        rows = rows_from_manifest(manifest, doc=doc)
        source = "run_manifest.json"
        if artifact_stage is None:
            meta = manifest.get("_meta") or {}
            artifact_stage = meta.get("artifact_stage")
    elif m1 or m2:
        rows = rows_from_extractions(m1, m2, critic, doc=doc)
        source = "segment_a m1/m2/critic"
    else:
        raise ValueError(
            f"Cannot build a grading sheet for {doc_id!r}: supply `manifest=` "
            f"(segment_b/gate.py run_manifest.json) or `m1=`/`m2=` "
            f"(segment_a/output/{{m1,m2}}/{doc_id}.json). Refusing to write an "
            f"empty sheet, which would read as 'graded, no defects found'."
        )
    if not rows:
        raise ValueError(
            f"Grading sheet for {doc_id!r} would have zero rows. Source "
            f"({source}) parsed but contained none of "
            f"{list(settings.ALL_FIELDS)}."
        )
    return write_sheets(
        out_dir,
        doc_id,
        rows,
        work_id=work_id,
        title=title,
        source=source,
        artifact_stage=artifact_stage,
        tax=tax,
    )


# --- Reading back -----------------------------------------------------------


def read_sheet(path: Path) -> tuple[dict[str, str], list[dict]]:
    """
    Read one sheet into `(preamble_meta, rows)`.

    Deliberately the same parse `grades.py:_read_grading_sheet` performs, so a
    file this module writes and a file `segment_a/grading.py` writes are read
    identically. Kept here as well so `merge_blind_into_reveal` and the
    `soi_useful` reader do not have to reach into a private function of another
    module.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"No grading sheet at {path}")
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
    rows = list(csv.DictReader(body)) if body else []
    return meta, rows


def read_soi_useful(path: Path) -> Optional[str]:
    """
    The doc-level `soi_useful` answer, or None if unfilled.

    First non-empty value wins and a disagreement is logged rather than silently
    resolved: the column is doc-level by convention, not by construction, so two
    different answers mean the reviewer used it per-row and the sheet needs a
    human look before the answer is trusted.
    """
    _meta, rows = read_sheet(path)
    answers = [
        (r.get("field") or "", (r.get("soi_useful") or "").strip().lower())
        for r in rows
    ]
    filled = [(f, a) for f, a in answers if a]
    if not filled:
        return None
    distinct = {a for _f, a in filled}
    if len(distinct) > 1:
        log.warning(
            "%s: soi_useful filled with conflicting values %s on rows %s; "
            "taking the first. It is a doc-level answer (MCAL_PLAN 7 Q5).",
            Path(path).name,
            sorted(distinct),
            [f for f, _a in filled],
        )
    return filled[0][1]


def merge_blind_into_reveal(blind_path: Path, reveal_path: Path) -> Path:
    """
    Copy the blind pass's reviewer answers into the reveal sheet, in place.

    MCAL_PLAN 7 Q5's second pass exists "for meta-analysis only", which means the
    reviewer should not retype pass-1 answers into it -- retyping is where pass-1
    answers get quietly revised after seeing the Critic, which is the exact
    contamination the blinding was for. Rows are matched on `field`; a blind row
    with no counterpart in the reveal sheet is reported and skipped rather than
    appended, because a field mismatch means the two sheets were generated from
    different extractions and merging them would be wrong.
    """
    _bmeta, brows = read_sheet(blind_path)
    rmeta, rrows = read_sheet(reveal_path)
    answers = {
        (r.get("field") or ""): {c: (r.get(c) or "") for c in REVIEWER_COLUMNS}
        for r in brows
    }
    reveal_fields = {r.get("field") or "" for r in rrows}
    orphans = sorted(f for f in answers if f and f not in reveal_fields)
    if orphans:
        log.warning(
            "%s: %d blind row(s) have no reveal counterpart and were not merged: "
            "%s. The two sheets were probably built from different extractions.",
            Path(blind_path).name,
            len(orphans),
            orphans[:5],
        )

    for r in rrows:
        got = answers.get(r.get("field") or "")
        if got:
            for c, v in got.items():
                if v:
                    r[c] = v

    cols = list(REVEAL_COLUMNS)
    path = Path(reveal_path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for k in (
            "doc_id", "work_id", "title", "pass", "source", "artifact_stage",
        ):
            fh.write(f"# {k}: {rmeta.get(k, '')}\n")
        fh.write(
            "# merged_from_blind: reviewer answers copied from the blind pass "
            "(MCAL_PLAN 7 Q5). Revising them here defeats the blinding.\n"
        )
        fh.write("\n")
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rrows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


def audit_blinding(path: Path) -> list[str]:
    """
    Problems that would compromise the blind pass, as reason codes.

    Empty means the sheet is safely blind. Exists so a build (or a test) can
    assert blinding rather than trusting that the writer got it right -- the
    defect this module fixes was in a file whose author also intended it to be
    blind.
    """
    meta, rows = read_sheet(path)
    problems: list[str] = []
    if meta.get("pass") != PASS_BLIND:
        problems.append(f"pass_is_{meta.get('pass') or 'unset'}_not_blind")
    header = set(rows[0].keys()) if rows else set()
    for col in CRITIC_COLUMNS:
        if col in header:
            problems.append(f"critic_column_present:{col}")
    for r in rows:
        for col in REVIEWER_COLUMNS:
            if (r.get(col) or "").strip():
                problems.append(f"prepopulated_reviewer_cell:{col}")
                break
        else:
            continue
        break
    return problems


# --- Round-trip guard -------------------------------------------------------


def unparsable_fields(rows: Iterable[SheetRow]) -> list[str]:
    """
    Row keys `mcal/grades.py:canonical_field` cannot resolve.

    A row whose key does not resolve is silently dropped by
    `load_grading_sheets` (with a warning nobody reads), so its grade never
    reaches the calibration set. Non-empty is a bug in this module, not in the
    reviewer's file, and `build_and_write` callers should treat it as fatal.
    """
    from . import grades as grades_mod

    return sorted(
        {
            r.field
            for r in rows
            if grades_mod.canonical_field(r.field) is None
        }
    )
