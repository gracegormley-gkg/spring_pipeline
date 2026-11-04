"""
Failure taxonomy: seed codes, TnT-LLM induction, add-only stage versioning
(MCAL_PLAN 3.1 / 2, build item #9).

Emits `artifacts/v(N)/taxonomy.v(N).json`, consumed by `critic_prompt.py` (to
build rubrics and select few-shot exemplars) and by the reviewer UI.

Three things this module guarantees:

1. **Add-only versioning.** T01-T18 are never renamed, renumbered or dropped.
   v(N+1) may add T19+ codes, or mark an old code `deprecated` with
   `superseded_by`. This matters because grades collected under v1 carry v1 tag
   strings; renumbering would silently corrupt every prior round's labels.

2. **Induction is advisory, ratification is human.** The Sonnet induction pass
   proposes clusters over the human's own free-text notes and writes a DRAFT.
   Nothing is promoted until a human ratifies it. MCAL_PLAN 6 makes
   `taxonomy.v1.json is human-ratified` an explicit seed-v1 acceptance item.

3. **Induction never invents a seed code.** The seed set is authored from
   MCAL_PLAN 2 and is present whether or not the LLM runs, so a build works
   offline and a flaky LLM call cannot shrink the taxonomy.

On exemplars: MCAL_PLAN 3.1 wants `{doc_id, field, note}` triples per tag.
These come from `mcal/grades.py`, whose seed-v1 source is the coarse Evaluation
sheet -- so exemplars are field-level, not item-level. T17/T18 have no exemplars
at seed v1 by construction (`summary_of_interest` is a new field with zero
graded examples); the Critic still checks for them via rubric Q7.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import grades as grades_mod
from . import settings

log = logging.getLogger(__name__)


# --- Seed taxonomy (MCAL_PLAN 2) --------------------------------------------
# T01-T18. Descriptions are written for two audiences: the Critic, which sees
# them in its rubric, and the human reviewer picking a `your_failure_tag`. They
# therefore state the OBSERVABLE test, not the underlying cause.

SEED_TAXONOMY: tuple[dict, ...] = (
    {
        "id": "T01",
        "name": "T01_missing_citation",
        "description": (
            "A claim in the extracted value has no page cite, or cites a page "
            "that does not support it. Most common observed failure: 4/8 docs on "
            "summary.public_response."
        ),
        "applies_to": [
            "summary.*",
            "summary_of_interest",
            "alternatives",
            "location",
            "key_people",
        ],
    },
    {
        "id": "T02",
        "name": "T02_numeric_hallucination",
        "description": (
            "A figure is wrong, or is attached to the wrong entity/alternative. "
            "NOTE: both graded instances turned out to be scope-qualifier loss "
            "rather than fabrication -- the figure was present and correctly "
            "paired, but a limiting qualifier was dropped. See T19."
        ),
        # Widened from the plan's two subfields: affected_community carries
        # population and demographic figures, alternatives_overview carries
        # counts, and public_response carries comment tallies. Any of them can
        # misreport a figure, and each of those rubrics references this code.
        "applies_to": ["summary.*", "summary_of_interest"],
    },
    {
        "id": "T03",
        "name": "T03_outside_text_fabrication",
        "description": (
            "The value asserts something absent from the cited pages -- "
            "prior-injection, e.g. completing a plausible NEPA sentence. "
            "Detectable: fails substring/coverage verification."
        ),
        # Includes the structured fields: an invented alternative label, an
        # off-taxonomy theme, or a place the document never names are the same
        # failure as an invented prose clause.
        "applies_to": [
            "summary.*",
            "summary_of_interest",
            "alternatives",
            "themes",
            "location",
            "key_people",
        ],
    },
    {
        "id": "T04",
        "name": "T04_undefined_acronym",
        "description": (
            "An acronym is used without expansion on first use within the field, "
            "and no expansion exists in the document glossary or the NEPA "
            "commons. Observed in 8/8 docs."
        ),
        # Every field that emits human-readable prose can carry an unglossed
        # acronym, not just the summaries. Without `location` and `key_people`
        # here, an acronym defect on an agency name ("USACE", "SHPO") is
        # untaggable: the Critic is told to emit T04 by the shared rubric Q3 but
        # the tag is absent from the field's vocabulary, so it becomes null and
        # inflates the null-tag monitor -- which is the signal that decides when
        # the taxonomy needs new codes. `title` is included for the same reason.
        "applies_to": [
            "summary.*",
            "summary_of_interest",
            "themes",
            "alternatives",
            "location",
            "key_people",
            "title",
            "lead_agency",
        ],
    },
    {
        "id": "T05",
        "name": "T05_commenter_mislabeled_as_cooperator",
        "description": (
            "An entity is labeled a cooperating agency without the document "
            "formally designating it as one under 40 CFR 1501.8. Usually a "
            "commenter or draft-EIS recipient swept in from the Consultation "
            "chapter. Observed in 5/8 docs."
        ),
        # lead_agency included: promoting a cooperating agency to lead is the
        # same category error, and lead_agency's rubric references this code.
        "applies_to": ["key_people", "lead_agency"],
    },
    {
        "id": "T06",
        "name": "T06_geocode_missing",
        "description": "A named place was extracted but no coordinates were resolved.",
        "applies_to": ["location"],
    },
    {
        "id": "T07",
        "name": "T07_geocode_wrong_specificity",
        "description": (
            "Coordinates resolved at a coarser admin level than the document's "
            "actual scope -- e.g. the containing city for a specific corridor."
        ),
        "applies_to": ["location"],
    },
    {
        "id": "T08",
        "name": "T08_scope_misclassified_national",
        "description": (
            "A national or international rulemaking with no site was treated as "
            "an absent location instead of scope=national."
        ),
        "applies_to": ["location"],
    },
    {
        "id": "T09",
        "name": "T09_multi_site_partial_geocode",
        "description": (
            "The document names several primary sites but only a subset resolved "
            "to coordinates; the rest were dropped."
        ),
        "applies_to": ["location"],
    },
    {
        "id": "T10",
        "name": "T10_alternatives_chapter_missed",
        "description": (
            "Structural identification of the Alternatives chapter failed and the "
            "field returned empty rather than a not-found status."
        ),
        "applies_to": ["alternatives"],
    },
    {
        "id": "T11",
        "name": "T11_year_ocr_error",
        "description": (
            "Publication year is wrong. Observed only on pre-1980 scans, where "
            "the real date sits on a signature/approval page or transmittal "
            "letter rather than the cover."
        ),
        "applies_to": ["year"],
    },
    {
        "id": "T12",
        "name": "T12_eis_type_confused_with_rod",
        "description": (
            "Document type misread, typically because 'Record of Decision' "
            "appeared in body or citation text rather than as a heading."
        ),
        "applies_to": ["eis_type"],
    },
    {
        "id": "T13",
        "name": "T13_pre_1978_nepa_format",
        "description": (
            "Document predates the modern 40 CFR 1501.8 schema, so the "
            "cooperating-agency category does not apply as defined."
        ),
        "applies_to": ["key_people"],
    },
    {
        "id": "T14",
        "name": "T14_regional_scope_underspecified",
        "description": (
            "Scope is regional but the document names fewer than two primary "
            "sites, so no defensible region could be resolved."
        ),
        "applies_to": ["location"],
    },
    {
        "id": "T15",
        "name": "T15_jargon_without_gloss",
        "description": (
            "A NEPA-specific term or regulatory citation is used without an "
            "in-line gloss on first mention, contrary to the plain-language "
            "clause."
        ),
        "applies_to": ["summary.*", "summary_of_interest"],
    },
    {
        "id": "T16",
        "name": "T16_abstract_when_concrete_available",
        "description": (
            "The value uses abstract nominalizations where the document supports "
            "a concrete description with named entities and quantities. Logged "
            "only at v1 (Critic Q6b), not yet gating."
        ),
        "applies_to": ["summary.*", "summary_of_interest"],
    },
    {
        "id": "T17",
        "name": "T17_manufactured_salience",
        "description": (
            "A summary_of_interest entry is tagged with a salience criterion the "
            "cited page does not support, or why_notable is grounded in general "
            "NEPA knowledge rather than the page. Includes finding something "
            "'notable' in a routine document."
        ),
        "applies_to": ["summary_of_interest"],
    },
    {
        "id": "T18",
        "name": "T18_salience_duplicates_summary",
        "description": (
            "A summary_of_interest entry restates standard-summary content "
            "without independently meeting a salience criterion."
        ),
        "applies_to": ["summary_of_interest"],
    },
)

SEED_IDS = tuple(t["id"] for t in SEED_TAXONOMY)
FIRST_INDUCED_ID = 19  # T19+ per MCAL_PLAN 2

# Codes proposed from empirical analysis of the graded corpus rather than from
# the plan. Kept separate from SEED_TAXONOMY so the plan's T01-T18 stays
# verbatim, and offered to the human as induction candidates.
#
# T19 exists because both cases the Evaluation sheet labels numeric errors are
# actually scope-qualifier loss: the LA Transit "$659 million (Alt. V)" figure
# is correctly paired on pp.31/214/215 but the document scopes it to "the
# Rail/Bus Alternatives I-V", and "Magnitude 7.5" is verbatim on p.146 attached
# to Newport-Inglewood while the human's 7.0 comes from p.145's "maximum
# credible event" framing. Neither is a fabrication, so neither can be caught by
# substring verification -- they need their own code so the distinction survives
# into the next round's grades.
PROPOSED_TAXONOMY: tuple[dict, ...] = (
    {
        "id": "T19",
        "name": "T19_scope_qualifier_dropped",
        "description": (
            "A figure or superlative is reported without the qualifier the "
            "document attaches to it -- a range restricted to a subset of "
            "alternatives, a 'maximum credible' framing, a geographic limit. "
            "The figure itself verifies, so substring checks cannot detect this; "
            "it is prevented at generation time by the plain-language clause and "
            "must be caught by the reviewer or by a comparative-claim check."
        ),
        "applies_to": ["summary.*", "summary_of_interest"],
        "origin": "empirical_v1",
    },
    {
        "id": "T20",
        "name": "T20_role_bucket_underpopulated",
        "description": (
            "A key_people role bucket is empty or nearly empty when the document "
            "does contain qualifying entities. Distinct from T05, which is "
            "over-inclusion; this is under-inclusion. Observed on Fuel Economy "
            "('nearly empty')."
        ),
        "applies_to": ["key_people"],
        "origin": "empirical_v1",
    },
)


# --- Data model -------------------------------------------------------------


@dataclass
class Tag:
    id: str
    name: str
    description: str
    applies_to: list[str] = dc_field(default_factory=list)
    exemplars: list[dict] = dc_field(default_factory=list)
    origin: str = "seed"
    deprecated: bool = False
    superseded_by: Optional[str] = None

    @property
    def number(self) -> int:
        return int(self.id.lstrip("Tt"))

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "applies_to": self.applies_to,
            "exemplars": self.exemplars,
            "origin": self.origin,
        }
        if self.deprecated:
            d["deprecated"] = True
            d["superseded_by"] = self.superseded_by
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Tag":
        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            applies_to=list(d.get("applies_to") or []),
            exemplars=list(d.get("exemplars") or []),
            origin=d.get("origin", "seed"),
            deprecated=bool(d.get("deprecated")),
            superseded_by=d.get("superseded_by"),
        )


@dataclass
class Taxonomy:
    stage: str
    tags: list[Tag] = dc_field(default_factory=list)
    frozen_at: Optional[str] = None
    ratified: bool = False
    induction_notes: list[str] = dc_field(default_factory=list)

    # --- lookup ---
    def by_name(self, name: str) -> Optional[Tag]:
        for t in self.tags:
            if t.name == name:
                return t
        return None

    def by_id(self, tag_id: str) -> Optional[Tag]:
        for t in self.tags:
            if t.id == tag_id:
                return t
        return None

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tags]

    @property
    def active(self) -> list[Tag]:
        return [t for t in self.tags if not t.deprecated]

    def for_field(self, field: str) -> list[Tag]:
        """
        Tags applicable to a field. `applies_to` entries may be exact keys or the
        wildcard "summary.*".
        """
        out = []
        for t in self.active:
            for pat in t.applies_to:
                if pat == field or (pat.endswith(".*") and field.startswith(pat[:-1])):
                    out.append(t)
                    break
        return out

    def next_id(self) -> str:
        used = {t.number for t in self.tags}
        n = max(FIRST_INDUCED_ID, (max(used) + 1) if used else FIRST_INDUCED_ID)
        while n in used:
            n += 1
        return f"T{n:02d}"

    def to_dict(self) -> dict:
        return {
            "version": self.stage,
            "frozen_at": self.frozen_at,
            "ratified": self.ratified,
            "tags": [t.to_dict() for t in self.tags],
            "induction_notes": self.induction_notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Taxonomy":
        return cls(
            stage=d.get("version") or "v1",
            tags=[Tag.from_dict(t) for t in d.get("tags") or []],
            frozen_at=d.get("frozen_at"),
            ratified=bool(d.get("ratified")),
            induction_notes=list(d.get("induction_notes") or []),
        )


# --- Construction -----------------------------------------------------------


def seed_taxonomy(stage: str = "v1", *, include_proposed: bool = True) -> Taxonomy:
    """
    Build the taxonomy from the hard-coded seed, with no LLM involvement.

    `include_proposed` adds the empirically-derived T19/T20 candidates. They are
    marked `origin="empirical_v1"` so a reviewer can tell them apart from the
    plan's own codes and reject them without touching T01-T18.
    """
    tags = [Tag.from_dict(t) for t in SEED_TAXONOMY]
    if include_proposed:
        tags += [Tag.from_dict(t) for t in PROPOSED_TAXONOMY]
    return Taxonomy(stage=settings.normalize_stage(stage), tags=tags)


class TaxonomyVersionError(ValueError):
    """Raised when a stage transition would violate the add-only rule."""


def carry_forward(prior: Taxonomy, stage: str) -> Taxonomy:
    """
    Start a new stage from the prior one, add-only (MCAL_PLAN 2, 3.7).

    Every prior tag is copied verbatim, including deprecated ones. Any seed code
    missing from the prior taxonomy is restored -- a stage must never be able to
    lose T01-T18, even if a prior artifact was hand-edited.
    """
    tags = [Tag.from_dict(t.to_dict()) for t in prior.tags]
    have = {t.id for t in tags}
    for s in SEED_TAXONOMY:
        if s["id"] not in have:
            log.warning(
                f"Seed code {s['id']} was absent from {prior.stage}; restoring. "
                "T01-T18 cannot be dropped (MCAL_PLAN 2)."
            )
            tags.append(Tag.from_dict(s))
    return Taxonomy(stage=settings.normalize_stage(stage), tags=tags)


def validate_transition(prior: Optional[Taxonomy], new: Taxonomy) -> list[str]:
    """
    Enforce the add-only rule. Returns a list of violations; empty means valid.

    Checked separately from `carry_forward` so that a hand-edited draft is also
    validated before promotion.
    """
    problems: list[str] = []

    seen_ids: dict[str, str] = {}
    seen_names: dict[str, str] = {}
    for t in new.tags:
        if t.id in seen_ids:
            problems.append(f"duplicate id {t.id}")
        seen_ids[t.id] = t.name
        if t.name in seen_names:
            problems.append(f"duplicate name {t.name}")
        seen_names[t.name] = t.id

    # Seed codes must be present and unrenamed.
    for s in SEED_TAXONOMY:
        got = new.by_id(s["id"])
        if got is None:
            problems.append(f"seed code {s['id']} is missing")
        elif got.name != s["name"]:
            problems.append(
                f"seed code {s['id']} renamed {s['name']!r} -> {got.name!r}; "
                "T01-T18 are frozen"
            )

    if prior is not None:
        for pt in prior.tags:
            nt = new.by_id(pt.id)
            if nt is None:
                problems.append(f"{pt.id} present in {prior.stage} but dropped")
            elif nt.name != pt.name:
                problems.append(
                    f"{pt.id} renamed {pt.name!r} -> {nt.name!r} between "
                    f"{prior.stage} and {new.stage}"
                )
        if settings.stage_number(new.stage) <= settings.stage_number(prior.stage):
            problems.append(
                f"stage {new.stage} does not advance past prior {prior.stage}"
            )

    for t in new.tags:
        if t.deprecated and not t.superseded_by:
            problems.append(f"{t.id} deprecated without superseded_by")
        if t.superseded_by and new.by_id(t.superseded_by) is None:
            problems.append(f"{t.id}.superseded_by={t.superseded_by} does not exist")

    return problems


# --- Exemplar attachment ----------------------------------------------------


def attach_exemplars(
    tax: Taxonomy, grade_set: "grades_mod.GradeSet", *, max_per_tag: int = 5
) -> Taxonomy:
    """
    Populate each tag's `exemplars` from the graded corpus.

    Exemplars are `{doc_id, field, note}` per MCAL_PLAN 3.1, plus the raw grade
    text, which is what `critic_prompt.py` actually shows the Critic as a
    few-shot. Diversified across documents first: three exemplars from three
    documents teach the Critic more than three from one.
    """
    by_tag: dict[str, list[dict]] = {}
    for item in grade_set.items:
        for tag_name in item.failure_tags:
            by_tag.setdefault(tag_name, []).append(
                {
                    "doc_id": item.doc_id,
                    "field": item.field,
                    "note": item.raw_grade,
                    "bucket": item.bucket,
                }
            )

    # T04 needs special handling. grades.py records the doc-level "includes
    # undefined acronyms" note as an `acronym_issue` flag rather than a failure
    # tag, because MCAL_PLAN 3.5 routes an acronym failure to PASS_WITH_NOTE and
    # folding it into y_i would mark every summary field on all 8 docs wrong.
    # But the Critic's rubric Q3 still needs few-shot exemplars, so harvest them
    # from the flag. Restricted to fields the tag actually applies to, so we do
    # not offer a `location` row as an example of an unglossed acronym.
    t04 = tax.by_name("T04_undefined_acronym")
    if t04 is not None and "T04_undefined_acronym" not in by_tag:
        applicable = {f for f in settings.ALL_FIELDS if t04 in tax.for_field(f)}
        for item in grade_set.items:
            if item.acronym_issue and item.field in applicable:
                by_tag.setdefault("T04_undefined_acronym", []).append(
                    {
                        "doc_id": item.doc_id,
                        "field": item.field,
                        "note": item.notes or "includes undefined acronyms",
                        "bucket": item.bucket,
                        "source": "doc_level_acronym_flag",
                    }
                )

    for tag in tax.tags:
        pool = by_tag.get(tag.name, [])
        chosen: list[dict] = []
        seen_docs: set[str] = set()
        # First pass: one per document.
        for ex in pool:
            if ex["doc_id"] not in seen_docs:
                chosen.append(ex)
                seen_docs.add(ex["doc_id"])
            if len(chosen) >= max_per_tag:
                break
        # Second pass: fill remaining slots with anything left.
        if len(chosen) < max_per_tag:
            for ex in pool:
                if ex not in chosen:
                    chosen.append(ex)
                if len(chosen) >= max_per_tag:
                    break
        tag.exemplars = chosen

    return tax


def coverage_report(tax: Taxonomy) -> dict:
    """
    Which tags have exemplars and which do not.

    A tag with zero exemplars cannot contribute a few-shot example, which is why
    MCAL_PLAN 3.5 backfills few-shot slots with positive controls. T17/T18 are
    expected to be empty at seed v1.
    """
    with_ex = [t.name for t in tax.active if t.exemplars]
    without = [t.name for t in tax.active if not t.exemplars]
    return {
        "n_tags": len(tax.active),
        "n_with_exemplars": len(with_ex),
        "with_exemplars": with_ex,
        "without_exemplars": without,
        "expected_empty_at_seed": [
            "T17_manufactured_salience",
            "T18_salience_duplicates_summary",
        ],
    }


# --- TnT-LLM induction ------------------------------------------------------

_INDUCTION_SYSTEM = """\
You are clustering human grader notes from an NLP pipeline that extracts \
structured metadata from US Environmental Impact Statements.

You will receive a list of notes. Each records something a human found WRONG \
with one extracted field of one document, in the human's own words.

You are also given an EXISTING taxonomy of failure modes. Your job:

1. For each note, decide whether it fits an EXISTING code. Most will.
2. Identify any recurring failure mode that the existing codes do NOT capture. \
Propose a NEW code only when at least two notes share it and no existing code \
fits. Do not propose a new code for a single unusual note.
3. Never propose renaming, renumbering, merging or deleting an existing code. \
The existing codes are frozen; you may only add.

Return ONLY JSON:
{
  "assignments": [
    {"note_index": 0, "tag_name": "T01_missing_citation", "confidence": "high|medium|low"}
  ],
  "proposed_new_codes": [
    {"name": "T99_snake_case_name", "description": "observable test, one or two sentences",
     "applies_to": ["field.key"], "supporting_note_indices": [3, 7],
     "why_existing_codes_insufficient": "one sentence"}
  ],
  "notes_for_human": ["anything ambiguous worth a human decision"]
}

Name proposed codes in the same style as the existing ones: T<number>_<snake_case>. \
Use T99 as a placeholder number; the caller assigns the real number.
"""


def induce(
    grade_set: "grades_mod.GradeSet",
    prior: Taxonomy,
    *,
    call=None,
    max_notes: int = 200,
) -> dict:
    """
    Cluster the graded corpus's free-text notes (MCAL_PLAN 3.1).

    Advisory only -- returns the raw proposal for a human to ratify. `call` is
    injectable so tests and offline builds do not need Bedrock; when omitted,
    `llm.sonnet` is used.

    Returns `{"assignments": [...], "proposed_new_codes": [...],
    "notes_for_human": [...], "skipped": bool}`.
    """
    rows = grade_set.induction_rows()[:max_notes]
    if not rows:
        return {
            "assignments": [],
            "proposed_new_codes": [],
            "notes_for_human": ["No non-correct grades to induce over."],
            "skipped": True,
        }

    if call is None:
        from llm import sonnet as call  # segment_a bridge

    payload = {
        "existing_taxonomy": [
            {"name": t.name, "description": t.description, "applies_to": t.applies_to}
            for t in prior.active
        ],
        "notes": [
            {
                "index": i,
                "field": r["field"],
                "bucket": r["bucket"],
                "grade_text": r["raw_grade"],
                "doc_note": r["notes"],
            }
            for i, r in enumerate(rows)
        ],
    }

    try:
        out = call(
            system=_INDUCTION_SYSTEM,
            user=json.dumps(payload, ensure_ascii=False, indent=1),
            max_tokens=4000,
        )
    except Exception as e:
        # Induction is an enrichment, not a dependency. A failed call must not
        # take down the build -- the seed taxonomy is already complete.
        log.error(f"Taxonomy induction call failed: {e}")
        return {
            "assignments": [],
            "proposed_new_codes": [],
            "notes_for_human": [f"Induction call failed: {e}"],
            "skipped": True,
        }

    out = out if isinstance(out, dict) else {}
    return {
        "assignments": list(out.get("assignments") or []),
        "proposed_new_codes": list(out.get("proposed_new_codes") or []),
        "notes_for_human": list(out.get("notes_for_human") or []),
        "skipped": False,
    }


def apply_induction(tax: Taxonomy, induction: dict) -> Taxonomy:
    """
    Fold induced proposals into a taxonomy, assigning real T19+ numbers.

    Proposals that duplicate an existing name are dropped. Everything added is
    marked `origin="induced"` so a reviewer can see what came from the LLM.
    """
    for prop in induction.get("proposed_new_codes") or []:
        raw_name = (prop.get("name") or "").strip()
        if not raw_name:
            continue
        # Strip the placeholder number and re-number from our own counter.
        _, _, suffix = raw_name.partition("_")
        if not suffix:
            log.warning(f"Skipping malformed induced code name {raw_name!r}")
            continue
        if any(t.name.partition("_")[2] == suffix for t in tax.tags):
            log.info(f"Induced code {suffix!r} duplicates an existing tag; skipped")
            continue
        new_id = tax.next_id()
        tax.tags.append(
            Tag(
                id=new_id,
                name=f"{new_id}_{suffix}",
                description=(prop.get("description") or "").strip(),
                applies_to=list(prop.get("applies_to") or []),
                origin="induced",
            )
        )
        log.info(f"Induced new code {new_id}_{suffix}")

    for n in induction.get("notes_for_human") or []:
        if n:
            tax.induction_notes.append(str(n))
    return tax


# --- Persistence ------------------------------------------------------------


def artifact_path_for(stage: str, *, draft: bool) -> Path:
    return settings.artifact_path("taxonomy.json", stage, draft=draft)


def save(tax: Taxonomy, *, draft: bool = True, ratified: bool = False) -> Path:
    """
    Write the taxonomy artifact.

    Drafts go to `artifacts/v(N)-draft/`; promotion to `artifacts/v(N)/` happens
    via `build.py` once the human ratifies (MCAL_PLAN 3.7). `frozen_at` is only
    stamped on a ratified, non-draft write -- a draft is by definition not frozen.
    """
    tax.ratified = ratified
    if ratified and not draft:
        tax.frozen_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = artifact_path_for(tax.stage, draft=draft)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tax.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load(stage: str, *, draft: bool = False) -> Optional[Taxonomy]:
    path = artifact_path_for(stage, draft=draft)
    if not path.exists():
        return None
    return Taxonomy.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_current(stage: Optional[str] = None) -> Optional[Taxonomy]:
    """Load the promoted taxonomy for `stage`, or the latest promoted stage."""
    s = stage or settings.latest_stage()
    return load(s) if s else None


# --- Build entry point ------------------------------------------------------


def build(
    stage: str,
    grade_set: "grades_mod.GradeSet",
    *,
    prior_stage: Optional[str] = None,
    run_induction: bool = True,
    call=None,
    include_proposed: bool = True,
) -> tuple[Taxonomy, dict]:
    """
    Build the taxonomy for `stage`. Returns `(taxonomy, diagnostics)`.

    Seed build: start from SEED_TAXONOMY. Recalibration: carry the prior stage
    forward add-only, then extend. Either way exemplars are re-attached from the
    full accumulated grade set, so few-shots improve every round even when the
    code list does not change (MCAL_PLAN 7.5 "What does change round-to-round").
    """
    stage = settings.normalize_stage(stage)
    prior: Optional[Taxonomy] = None

    if prior_stage:
        prior = load(prior_stage)
        if prior is None:
            raise FileNotFoundError(
                f"--prior {prior_stage} requested but "
                f"{artifact_path_for(prior_stage, draft=False)} does not exist. "
                f"Promoted stages on disk: {settings.latest_stage() or 'none'}"
            )
        tax = carry_forward(prior, stage)
    else:
        tax = seed_taxonomy(stage, include_proposed=include_proposed)

    induction: dict = {"skipped": True}
    if run_induction:
        induction = induce(grade_set, tax, call=call)
        if not induction.get("skipped"):
            tax = apply_induction(tax, induction)

    tax = attach_exemplars(tax, grade_set)

    problems = validate_transition(prior, tax)
    if problems:
        raise TaxonomyVersionError(
            "Add-only taxonomy rule violated:\n  - " + "\n  - ".join(problems)
        )

    diagnostics = {
        "stage": stage,
        "prior_stage": prior_stage,
        "n_tags": len(tax.tags),
        "n_seed": len(SEED_TAXONOMY),
        "n_proposed": sum(1 for t in tax.tags if t.origin == "empirical_v1"),
        "n_induced": sum(1 for t in tax.tags if t.origin == "induced"),
        "induction_skipped": bool(induction.get("skipped")),
        "induction_notes": tax.induction_notes,
        "coverage": coverage_report(tax),
        "observed_tag_counts": grade_set.tag_counts(),
    }
    return tax, diagnostics
