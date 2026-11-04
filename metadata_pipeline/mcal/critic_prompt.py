"""
Per-field Critic prompt builder (MCAL_PLAN 3.5, build item #11).

Emits `artifacts/v(N)/critic_prompts/{field}.v(N).md`, loaded by
`segment_b/critic.py` at run time.

Each prompt is assembled from:
  1. shared role header + anti-hallucination clause + private-individual
     definition      -> templates/critic_header.md
  2. shared rubric Q1-Q6 + base decision table
                     -> templates/rubrics/_base.md
  3. per-field overlay: field description, Q7+, decision overrides
                     -> templates/rubrics/{field}.md
  4. failure-coverage few-shot exemplars, chosen by greedy set-cover over the
     tags observed for that field, backfilled with positive controls
  5. the applicable failure-tag vocabulary for that field
  6. strict JSON output schema with evidence_quote first

Why assemble rather than hand-write 15 whole prompts: Q1-Q6 and the
anti-hallucination clause must be byte-identical across fields, or per-bucket
conformal thresholds are no longer comparable -- a Critic that is stricter on
one field purely because its prompt drifted would shift that bucket's score
distribution and silently invalidate its tau.

Prompts are built from the taxonomy and the graded set, both of which are frozen
within a stage, so a rebuild at the same stage is deterministic.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import grades as grades_mod
from . import settings
from . import taxonomy as taxonomy_mod

log = logging.getLogger(__name__)


# --- Template loading -------------------------------------------------------

HEADER_PATH = settings.TEMPLATES_DIR / "critic_header.md"
BASE_RUBRIC_PATH = settings.RUBRICS_DIR / "_base.md"

# Sections critic_header.md must define. Asserted on load so a renamed heading
# fails loudly at build time rather than silently dropping the
# anti-hallucination clause from every prompt.
REQUIRED_HEADER_SECTIONS = (
    "ROLE",
    "ANTI_HALLUCINATION",
    "PRIVATE_INDIVIDUAL",
    "VERDICTS",
    "OUTPUT",
)
REQUIRED_RUBRIC_SECTIONS = ("QUESTIONS", "DECISION")

_BODY_SEPARATOR = "\n---\n"
_SECTION_RE = re.compile(r"^## +(\w+)\s*$", re.MULTILINE)


class TemplateError(RuntimeError):
    pass


def _body(path: Path) -> str:
    """Strip the human-facing provenance header above the first `---` rule."""
    if not path.exists():
        raise TemplateError(f"Missing Critic template: {path}")
    text = path.read_text(encoding="utf-8")
    _, sep, body = text.partition(_BODY_SEPARATOR)
    return (body if sep else text).strip()


def parse_sections(text: str) -> dict[str, str]:
    """Split a template body into `## NAME` sections."""
    out: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[m.group(1)] = text[m.end() : end].strip()
    return out


def load_header() -> dict[str, str]:
    secs = parse_sections(_body(HEADER_PATH))
    missing = [s for s in REQUIRED_HEADER_SECTIONS if s not in secs]
    if missing:
        raise TemplateError(
            f"{HEADER_PATH.name} is missing section(s) {missing}. "
            f"Found: {sorted(secs)}. These headings are load-bearing -- "
            f"critic_prompt.py extracts them by name."
        )
    return secs


def load_base_rubric() -> dict[str, str]:
    secs = parse_sections(_body(BASE_RUBRIC_PATH))
    missing = [s for s in REQUIRED_RUBRIC_SECTIONS if s not in secs]
    if missing:
        raise TemplateError(f"{BASE_RUBRIC_PATH.name} missing section(s) {missing}")
    return secs


def rubric_path_for(field: str) -> Path:
    """Field keys contain dots; filenames use underscores."""
    return settings.RUBRICS_DIR / f"{field.replace('.', '_')}.md"


def load_field_rubric(field: str) -> dict[str, str]:
    path = rubric_path_for(field)
    if not path.exists():
        raise TemplateError(
            f"No rubric overlay for field {field!r} at {path}. Every field in "
            f"settings.ALL_FIELDS needs one."
        )
    return parse_sections(_body(path))


# Tag references inside a rubric overlay, e.g. `T01_missing_citation`.
_TAG_REF_RE = re.compile(r"\bT\d{2}_[a-z0-9_]+")


def check_tag_references(field: str, tax: "taxonomy_mod.Taxonomy") -> list[str]:
    """
    Every tag a field's rubric tells the Critic to use must be in that field's
    vocabulary.

    Without this check the two drift silently: a decision table saying
    "failure_tag = T02_numeric_hallucination" while the FAILURE TAGS section
    omits T02 leaves the Critic to either invent the string or fall back to
    null. Both corrupt the null-tag monitor (MCAL_PLAN 6), which is the signal
    that triggers the next taxonomy revision -- so a bookkeeping slip here would
    masquerade as evidence that the taxonomy needs new codes.

    Returns a list of dangling tag names; empty means consistent.
    """
    path = rubric_path_for(field)
    if not path.exists():
        return []
    referenced = set(_TAG_REF_RE.findall(path.read_text(encoding="utf-8")))
    available = {t.name for t in tax.for_field(field)}
    return sorted(referenced - available)


# --- Few-shot selection -----------------------------------------------------


@dataclass
class FewShot:
    kind: str            # "failure" | "positive_control"
    doc_id: str
    field: str
    tag: Optional[str]
    note: str
    slug: str = ""

    def render(self, index: int) -> str:
        if self.kind == "failure":
            return (
                f"### Example {index} — defect ({self.tag})\n"
                f"Document: {self.slug or self.doc_id}\n"
                f"Field: `{self.field}`\n"
                f"A human grader recorded: \"{self.note}\"\n"
                f"Correct handling: this is `{self.tag}`. Do not pass it.\n"
            )
        return (
            f"### Example {index} — correct (positive control)\n"
            f"Document: {self.slug or self.doc_id}\n"
            f"Field: `{self.field}`\n"
            f"A human grader recorded: \"{self.note or 'ok'}\"\n"
            f"Correct handling: `PASS`. Do not invent a defect to justify a "
            f"lower verdict.\n"
        )


MIN_FEWSHOT_SLOTS = 3


def select_few_shots(
    field: str,
    tax: "taxonomy_mod.Taxonomy",
    grade_set: "grades_mod.GradeSet",
    *,
    k: Optional[int] = None,
) -> list[FewShot]:
    """
    Greedy set-cover over the tags observed for `field` (MCAL_PLAN 3.5 item 5).

    `K = min(3, #distinct tags with >=1 exemplar)`, then remaining slots are
    filled with positive controls -- correctly-graded examples of the same field.
    Never fewer than 3 total slots.

    Positive controls are not decoration. A Critic shown only failures learns
    that its job is to find one, which inflates RE_EXTRACT on clean fields and
    pushes composite scores down uniformly -- shifting the whole bucket
    distribution rather than separating good from bad.
    """
    applicable = {t.name for t in tax.for_field(field)}

    # Tags actually observed on THIS field, most frequent first. Field-specific
    # rather than corpus-wide: showing the location rubric a missing-citation
    # example from a summary field teaches the wrong discrimination.
    observed: dict[str, list[grades_mod.GradeItem]] = {}
    for item in grade_set.for_field(field):
        if item.correct:
            continue
        for tag in item.failure_tags:
            if tag in applicable:
                observed.setdefault(tag, []).append(item)

    ordered_tags = sorted(observed, key=lambda t: (-len(observed[t]), t))
    if k is None:
        k = min(MIN_FEWSHOT_SLOTS, len(ordered_tags))

    chosen: list[FewShot] = []
    used_docs: set[str] = set()
    for tag in ordered_tags[:k]:
        # Prefer an exemplar from a document not already used, for diversity.
        pool = observed[tag]
        pick = next((i for i in pool if i.doc_id not in used_docs), pool[0])
        used_docs.add(pick.doc_id)
        chosen.append(
            FewShot(
                kind="failure",
                doc_id=pick.doc_id,
                field=pick.field,
                tag=tag,
                note=pick.raw_grade,
                slug=grade_set.slugs.get(pick.doc_id, ""),
            )
        )

    # Backfill with positive controls.
    if len(chosen) < MIN_FEWSHOT_SLOTS:
        for item in grade_set.for_field(field):
            if len(chosen) >= MIN_FEWSHOT_SLOTS:
                break
            if not item.correct or item.doc_id in used_docs:
                continue
            used_docs.add(item.doc_id)
            chosen.append(
                FewShot(
                    kind="positive_control",
                    doc_id=item.doc_id,
                    field=item.field,
                    tag=None,
                    note=item.raw_grade,
                    slug=grade_set.slugs.get(item.doc_id, ""),
                )
            )

    # Still short (e.g. summary_of_interest has no grades at all at seed v1):
    # emit a synthetic positive control so the slot count is honoured and the
    # Critic still sees that PASS is an available answer.
    while len(chosen) < MIN_FEWSHOT_SLOTS:
        chosen.append(
            FewShot(
                kind="positive_control",
                doc_id="(none graded yet)",
                field=field,
                tag=None,
                note=(
                    "This field has no graded examples at this calibration "
                    "stage. Judge it on the rubric alone, and remember that a "
                    "clean field is a PASS."
                ),
            )
        )
    return chosen


# --- Assembly ---------------------------------------------------------------

OUTPUT_SCHEMA_EXAMPLE = {
    "evidence_quote": "string|null",
    "rubric_answers": {"Q1": "yes|no|n/a"},
    "verdict": "PASS|PASS_WITH_NOTE|RE_EXTRACT|HUMAN_REVIEW",
    "failure_tag": "T01_missing_citation|null",
    "note": "string|null",
}


def build_prompt(
    field: str,
    tax: "taxonomy_mod.Taxonomy",
    grade_set: "grades_mod.GradeSet",
    *,
    stage: str,
) -> str:
    """Assemble the full Critic prompt for one field."""
    header = load_header()
    base = load_base_rubric()
    overlay = load_field_rubric(field)

    bucket = settings.bucket_for_field(field)
    judge = "opus" if field in settings.OPUS_JUDGED_FIELDS else "sonnet"
    tags = tax.for_field(field)
    few_shots = select_few_shots(field, tax, grade_set)

    parts: list[str] = []

    parts.append(
        f"<!-- generated by mcal/critic_prompt.py for stage {stage}; "
        f"do not hand-edit, edit templates/ instead -->"
    )
    parts.append(f"# Critic prompt — `{field}`\n")
    parts.append(
        f"- Calibration stage: `{stage}`\n"
        f"- Conformal bucket: `{bucket}`\n"
        f"- Judge model: `{judge}`\n"
    )

    parts.append("## ROLE\n\n" + header["ROLE"])
    parts.append("## EVIDENCE RULES\n\n" + header["ANTI_HALLUCINATION"])
    parts.append("## PRIVATE INDIVIDUAL\n\n" + header["PRIVATE_INDIVIDUAL"])
    parts.append("## VERDICTS\n\n" + header["VERDICTS"])

    parts.append("## FIELD UNDER REVIEW\n\n" + overlay.get("FIELD_DESCRIPTION", field))

    # Rubric: shared Q1-Q6, then the field's Q7+.
    rubric = ["## RUBRIC\n", "Answer every question in order.\n"]
    rubric.append("### Shared checks\n\n" + base["QUESTIONS"])
    if overlay.get("QUESTIONS"):
        rubric.append(f"### Checks specific to `{field}`\n\n" + overlay["QUESTIONS"])
    parts.append("\n".join(rubric))

    # Decision: field overrides first, then the shared fallthrough.
    decision = ["## DECISION\n"]
    if overlay.get("DECISION"):
        decision.append(
            f"Field-specific rules for `{field}` — apply these first:\n\n"
            + overlay["DECISION"]
        )
    decision.append("Shared fallthrough table:\n\n" + base["DECISION"])
    parts.append("\n".join(decision))

    # Tag vocabulary. Restricting it per field measurably reduces off-field tags
    # (a location verdict tagged T01 tells the null-tag monitor nothing useful).
    if tags:
        lines = [
            "## FAILURE TAGS AVAILABLE FOR THIS FIELD\n",
            "Use exactly one of these strings, or `null`. Do not invent a tag.",
            "If a defect is real but no tag fits, set `failure_tag: null` and "
            "describe it in `note` — the null-tag rate is monitored and drives "
            "the next taxonomy revision.\n",
        ]
        for t in tags:
            lines.append(f"- `{t.name}` — {t.description}")
        parts.append("\n".join(lines))

    # Few-shots.
    fs = ["## EXAMPLES\n", "Drawn from human grades on this corpus.\n"]
    for i, shot in enumerate(few_shots, 1):
        fs.append(shot.render(i))
    parts.append("\n".join(fs))

    parts.append(
        "## OUTPUT\n\n"
        + header["OUTPUT"]
        + "\n\nSchema recap:\n\n```json\n"
        + json.dumps(OUTPUT_SCHEMA_EXAMPLE, indent=2)
        + "\n```"
    )

    return "\n\n".join(parts).rstrip() + "\n"


# --- Persistence ------------------------------------------------------------


def prompt_dir(stage: str, *, draft: bool) -> Path:
    return settings.stage_dir(stage, draft=draft) / "critic_prompts"


def prompt_path(field: str, stage: str, *, draft: bool) -> Path:
    s = settings.normalize_stage(stage)
    return prompt_dir(stage, draft=draft) / f"{field}.{s}.md"


def build_all(
    stage: str,
    tax: "taxonomy_mod.Taxonomy",
    grade_set: "grades_mod.GradeSet",
    *,
    draft: bool = True,
    fields: Optional[tuple[str, ...]] = None,
) -> dict:
    """
    Build and write every per-field Critic prompt. Returns diagnostics.
    """
    stage = settings.normalize_stage(stage)
    fields = fields or settings.ALL_FIELDS
    out_dir = prompt_dir(stage, draft=draft)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    fewshot_summary: dict[str, dict] = {}
    dangling: dict[str, list[str]] = {}
    for field in fields:
        bad = check_tag_references(field, tax)
        if bad:
            dangling[field] = bad
        text = build_prompt(field, tax, grade_set, stage=stage)
        path = prompt_path(field, stage, draft=draft)
        path.write_text(text, encoding="utf-8")
        written[field] = str(path.relative_to(settings.MAY25_ROOT))
        shots = select_few_shots(field, tax, grade_set)
        fewshot_summary[field] = {
            "n_slots": len(shots),
            "n_failure_examples": sum(1 for s in shots if s.kind == "failure"),
            "n_positive_controls": sum(1 for s in shots if s.kind == "positive_control"),
            "tags_covered": [s.tag for s in shots if s.tag],
            "n_tags_applicable": len(tax.for_field(field)),
        }

    if dangling:
        raise TemplateError(
            "Rubric overlays reference failure tags that are not in the "
            "corresponding field's vocabulary:\n"
            + "\n".join(f"  {f}: {tags}" for f, tags in dangling.items())
            + "\nEither widen the tag's `applies_to` in taxonomy.py or stop "
            "referencing it in templates/rubrics/."
        )

    return {
        "stage": stage,
        "draft": draft,
        "n_prompts": len(written),
        "dir": str(out_dir.relative_to(settings.MAY25_ROOT)),
        "prompts": written,
        "few_shots": fewshot_summary,
        "judge_model_by_field": settings.default_judge_model_map(),
    }


def load_prompt(field: str, stage: str) -> str:
    """Read a promoted prompt. Used by segment_b/critic.py at load time."""
    path = prompt_path(field, stage, draft=False)
    if not path.exists():
        raise FileNotFoundError(
            f"No Critic prompt for {field!r} at stage {stage}: {path}. "
            f"Run `python -m mcal.build --stage {stage}` and ratify the draft."
        )
    return path.read_text(encoding="utf-8")
