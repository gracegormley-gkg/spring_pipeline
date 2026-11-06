"""
Evidence-first, per-field Critic (MCAL_PLAN 3.11, build items #2 and #14).

Replaces `segment_a/critic.py`, which fails in four distinct ways that the
Evaluation CSV makes visible:

1. **Coarse granularity.** segment_a emits NINE verdicts: one for all six
   `summary.*` subfields together, one for the whole `alternatives` list, one for
   all three `key_people` buckets. A document whose `public_response` is missing
   citations while `project_description` is clean gets a single blended verdict,
   so `s_critic` -- half of the composite (MCAL_PLAN 3.3) -- is the same number
   for a good subfield and a bad one. Per-bucket conformal thresholds cannot
   separate what the Critic never separated. This module emits ONE verdict PER
   FIELD in `settings.ALL_FIELDS` (15 of them). The extra token cost was
   explicitly approved (MCAL_PLAN 7 Q2).
2. **No evidence commitment.** segment_a asks for `verdict` first and never
   requires the Critic to quote anything. MCAL_PLAN 3.5 inverts that: the
   response must carry `evidence_quote` BEFORE `verdict`.
3. **No deterministic check on the judge.** The Critic is an LLM and can
   hallucinate its own supporting quote as readily as the extractor can
   hallucinate a claim. MCAL_PLAN 4 Q2 names the fix as one of the two
   "load-bearing" layers: re-verify the Critic's own `evidence_quote` against the
   cited pages with `mcal/quote_check.py`, and override the verdict to
   HUMAN_REVIEW when it does not verify. Prompt words alone are known
   insufficient here -- the Lincoln Hwy wildlife clause survived a prompt that
   already forbade fabrication.
4. **Judge too weak for the hardest fields.** All five `summary.*` subfields plus
   `summary_of_interest` route to Opus (MCAL_PLAN 3.11, 7 Q2); everything else
   stays on Sonnet.

Layered so that each layer fails safe:

    prompt (per-field, built by mcal/critic_prompt.py)
      -> LLM response
        -> schema validation      unknown verdict -> HUMAN_REVIEW
                                  off-vocabulary failure_tag -> null + counted
          -> quote-verify override  unverifiable evidence -> HUMAN_REVIEW
            -> policy overrides      private/ambiguous capacity -> HUMAN_REVIEW
              -> dependent cascade   year untrustworthy -> key_people HUMAN_REVIEW

Every override records `verdict_before_override` and appends to `overrides`, so
a HUMAN_REVIEW route is always attributable to a specific layer. That
attribution is what `gate.py` turns into `gate_reason`, and MCAL_PLAN 3.12 wants
it precisely so a reviewer can tell "the gate is too conservative" from "the
Critic is the binding constraint".

Nothing here raises on a bad LLM response, a missing extraction, or a failed
call. A document that trips every failure mode still produces 15 results, all
routed to HUMAN_REVIEW with a note. At seed v1 most fields are gated anyway
(MCAL_PLAN 0), so the cost of an exception is losing a whole document's worth of
gradable output -- the opposite of what the multi-round protocol needs.

Cost/usage: judging goes through `llm.call_json_with_usage`, which records into
the process-wide usage collector. A caller that wraps a document in
`llm.start_usage_session()` / `llm.end_usage_session()` gets the Critic slice in
its per-doc roll-up for free (MCAL_PLAN 2, calibration_report Cost Summary).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Optional, Sequence

from mcal import quote_check, settings
from mcal import taxonomy as taxonomy_mod
from mcal.critic_prompt import load_prompt
from mcal.quote_check import QuoteCheck

# segment_b sibling: the capacity classifier is the operational implementation of
# MCAL_PLAN 3.5's "private individual" definition, including the dual-capacity
# case. Reusing it (rather than re-deriving a private/non-private test here) is
# what keeps the extractor's own routing and the Critic's policy override from
# disagreeing about the same commenter.
from segment_b.postproc.key_people_pipeline import (  # noqa: E402
    classify_capacity,
    cited_passage,
)

# segment_a's flat modules; the sys.path bridge is installed by mcal.settings.
from llm import call_json_with_usage  # noqa: E402
from pages import Doc  # noqa: E402

log = logging.getLogger(__name__)


# --- Vocabulary -------------------------------------------------------------

# Same tuple segment_a/critic.py used, kept verbatim so a verdict string means
# the same thing on both sides of the migration and old critic artifacts remain
# readable by mcal/grades.py.
VERDICTS = ("PASS", "PASS_WITH_NOTE", "RE_EXTRACT", "HUMAN_REVIEW")

VERDICT_HUMAN_REVIEW = "HUMAN_REVIEW"
VERDICT_RE_EXTRACT = "RE_EXTRACT"

# Verdicts that make a field's value untrustworthy for anything that depends on
# it (MCAL_PLAN 3.10 step 2 era gate; settings.DEPENDENT_FIELDS).
UNTRUSTWORTHY_VERDICTS = (VERDICT_RE_EXTRACT, VERDICT_HUMAN_REVIEW)

# Note strings. These are matched by gate.py to derive `gate_reason`, so they are
# named constants rather than inline literals -- a typo would silently degrade
# the gate's diagnostics rather than fail.
NOTE_EVIDENCE_UNVERIFIABLE = "critic_evidence_unverifiable"
NOTE_UNKNOWN_VERDICT = "critic_returned_unknown_verdict"
NOTE_OFF_VOCABULARY_TAG = "critic_failure_tag_off_vocabulary"
NOTE_PRIVATE_INDIVIDUAL = "policy_private_individual"
NOTE_AMBIGUOUS_CAPACITY = "policy_ambiguous_capacity"
NOTE_DEPENDENT_CASCADE = "dependent_field_cascade"
NOTE_LLM_FAILED = "critic_call_failed"
NOTE_EXTRACTION_MISSING = "extraction_missing"
NOTE_SCHEMA_ORDER = "evidence_quote_not_first_in_response"

# Rubric keys every field must carry in `rubric_answers`. Q6b is included even
# though it is logged-only at v1 (MCAL_PLAN 3.5, 3.12) -- run_manifest.json is
# where the offline concreteness audit reads it from, so a missing key would
# quietly empty that audit rather than show up as an error.
BASE_RUBRIC_KEYS = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6a", "Q6b")
# MCAL_PLAN 3.5 Q7(a)-(c); Q7d (closed criterion vocabulary) comes from
# templates/rubrics/summary_of_interest.md and is carried for the same reason.
SOI_RUBRIC_KEYS = ("Q7a", "Q7b", "Q7c", "Q7d")

RUBRIC_ANSWER_SYNONYMS = {
    "y": "yes",
    "yes": "yes",
    "true": "yes",
    "n": "no",
    "no": "no",
    "false": "no",
    "n/a": "n/a",
    "na": "n/a",
    "not applicable": "n/a",
    "": "n/a",
}
RUBRIC_ANSWER_MISSING = "n/a"

# Fields where an EMPTY extraction is a substantive, correct answer rather than a
# failure (MCAL_PLAN 3.15 rule 2, 3.12). Deliberately narrow:
#   * `summary_of_interest == []` means "this document is routine", which is the
#     expected output for most EISs and the load-bearing anti-hallucination
#     provision for the field.
#   * `alternatives == []` is NOT here: MCAL_PLAN 1(8) requires a
#     `{status: "alternatives_chapter_not_found"}` object instead of silence.
#   * `location` with no sites is NOT here either: a national rulemaking carries
#     `textual_location: "national"` (MCAL_PLAN 3.9 step 2), so a genuinely empty
#     location value is a failure.
EMPTY_IS_VALID_FIELDS = (settings.SUMMARY_OF_INTEREST,)

# --- Prompt sizing ----------------------------------------------------------

# segment_a truncated the cited-page blob at a bare `[:80_000]` with no signal
# that it had happened. The cap is kept (an EIS page is ~2-4k chars of OCR, and
# min..max+/-2 over scattered citations can reach hundreds of pages) but it is now
# explicit, reported in `evidence_meta`, and logged when it bites.
EVIDENCE_MAX_CHARS = 80_000
# Per-page hard cap, applied only when a single page alone exceeds the budget.
EVIDENCE_MAX_CHARS_PER_PAGE = 20_000
EVIDENCE_PAGE_TOLERANCE = 2

# MCAL_PLAN 7 Q3 specifies the window as `[min(cited)-2 .. max(cited)+2]`. Taken
# literally that is unusable for a field citing pp. 31, 214 and 215 (LA Transit's
# project_description does exactly this): the window becomes 187 pages, of which
# ~180 are irrelevant, and the char cap then evicts the pages that mattered. So:
# honour the literal contiguous window when the citation set is tight, and fall
# back to per-citation windows merged when it is not. Recorded as
# `window_mode: "contiguous" | "clustered"` so the deviation is visible in the
# manifest rather than implicit.
EVIDENCE_MAX_CONTIGUOUS_SPAN_PAGES = 30

MAX_EXTRACTED_VALUE_CHARS = 24_000
MAX_TOKENS = 2000
# Judging is held at temperature 0. MCAL_PLAN 7 Q8's "+0.2" applies to the
# RE_EXTRACT retry of the EXTRACTOR (see gate.py), not to the judge: a
# non-deterministic judge would make `s_critic`, and therefore every conformal
# threshold fitted on it, irreproducible.
TEMPERATURE = 0.0


class MissingArtifactError(RuntimeError):
    """Raised when a required M-Cal artifact has not been built/promoted."""


# --- Stage + artifact loading (cached per process) ---------------------------

_PROMPT_CACHE: dict[tuple[str, str], str] = {}
_CONFIG_CACHE: dict[str, dict] = {}
_TAXONOMY_CACHE: dict[str, "taxonomy_mod.Taxonomy"] = {}
_TAG_VOCAB_CACHE: dict[tuple[str, str], dict[str, str]] = {}


def clear_artifact_cache() -> None:
    """
    Drop every cached artifact.

    Artifacts are frozen within a stage (MCAL_PLAN 2), so caching them for the
    life of the process is safe and saves 15 file reads per document. Tests that
    rewrite artifacts under a monkeypatched `settings.ARTIFACTS_DIR` must call
    this, which is why it is public.
    """
    _PROMPT_CACHE.clear()
    _CONFIG_CACHE.clear()
    _TAXONOMY_CACHE.clear()
    _TAG_VOCAB_CACHE.clear()


def resolve_stage(stage: Optional[str] = None) -> str:
    """
    Pin an M-Cal stage, or fail with an actionable message.

    `mcal/artifacts/` does not exist until `mcal/build.py` has run and the draft
    has been ratified (MCAL_PLAN 3.7). Segment B must not silently invent
    defaults in that state: an un-calibrated gate would emit PASS verdicts with
    no statistical backing at all, which is strictly worse than refusing to run.
    """
    if stage:
        return settings.normalize_stage(stage)
    latest = settings.latest_stage()
    if latest:
        return latest
    drafts = []
    if settings.ARTIFACTS_DIR.exists():
        drafts = sorted(
            p.name for p in settings.ARTIFACTS_DIR.iterdir()
            if p.is_dir() and p.name.endswith("-draft")
        )
    hint = (
        f"Draft stage(s) present but not promoted: {drafts}. Ratify "
        f"taxonomy.<stage>-draft.json to promote the directory."
        if drafts
        else "No artifacts directory content at all."
    )
    raise MissingArtifactError(
        f"No promoted M-Cal stage found under {settings.ARTIFACTS_DIR}. {hint} "
        f"Build one with:\n"
        f"    python -m mcal.build --stage v1 "
        f"--grades segment_a/output/grading_sheets/ --out mcal/artifacts/\n"
        f"then pass the stage explicitly, e.g. run_critic(..., stage='v1')."
    )


def prompt_for(field: str, stage: str) -> str:
    """Per-field Critic prompt (MCAL_PLAN 3.5), cached per process."""
    key = (field, stage)
    if key not in _PROMPT_CACHE:
        _PROMPT_CACHE[key] = load_prompt(field, stage)
    return _PROMPT_CACHE[key]


def load_confidence_config(stage: str) -> dict:
    """
    `confidence_config.v(N).json` (MCAL_PLAN 2), cached per process.

    Lives here rather than in `mcal/confidence.py` (which only ships
    `load_thresholds`) so that `gate.py` and `critic.py` read the artifact
    through one code path and cannot disagree about, say, which field routes to
    Opus. `gate.py` imports it from this module for that reason.
    """
    stage = settings.normalize_stage(stage)
    if stage in _CONFIG_CACHE:
        return _CONFIG_CACHE[stage]
    path = settings.artifact_path("confidence_config.json", stage, draft=False)
    if not path.exists():
        raise MissingArtifactError(
            f"No confidence config for stage {stage}: {path}. Run "
            f"`python -m mcal.build --stage {stage}` and ratify the draft "
            f"(MCAL_PLAN 3.7)."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise MissingArtifactError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise MissingArtifactError(
            f"{path} must contain a JSON object, got {type(payload).__name__}."
        )
    _CONFIG_CACHE[stage] = payload
    return payload


def load_taxonomy(stage: str) -> "taxonomy_mod.Taxonomy":
    """Promoted taxonomy for `stage`, cached. Raises when absent."""
    stage = settings.normalize_stage(stage)
    if stage in _TAXONOMY_CACHE:
        return _TAXONOMY_CACHE[stage]
    tax = taxonomy_mod.load_current(stage)
    if tax is None:
        raise MissingArtifactError(
            f"No promoted taxonomy for stage {stage}: "
            f"{taxonomy_mod.artifact_path_for(stage, draft=False)}. The Critic "
            f"cannot validate `failure_tag` without it, and accepting arbitrary "
            f"tag strings would corrupt the null-tag monitor (MCAL_PLAN 6). Run "
            f"`python -m mcal.build --stage {stage}` and ratify the draft."
        )
    _TAXONOMY_CACHE[stage] = tax
    return tax


def tag_vocabulary(field: str, stage: str) -> dict[str, str]:
    """
    Allowed `failure_tag` values for one field, as `{lookup_key: canonical_name}`.

    Both the canonical name (`T01_missing_citation`) and the bare id (`T01`) are
    accepted as lookup keys, case-insensitively. That is normalization, not
    permissiveness: the id unambiguously identifies one tag, and MCAL_PLAN 3.12's
    own schema example writes `"failure_tag": "T01|null"`, so a model that copies
    the schema literally must not be punished for it. Anything else really is
    off-vocabulary and becomes null.
    """
    key = (field, stage)
    if key in _TAG_VOCAB_CACHE:
        return _TAG_VOCAB_CACHE[key]
    tax = load_taxonomy(stage)
    vocab: dict[str, str] = {}
    for tag in tax.for_field(field):
        vocab[tag.name.lower()] = tag.name
        vocab[tag.id.lower()] = tag.name
    _TAG_VOCAB_CACHE[key] = vocab
    return vocab


def judge_model_for(field: str, config: Optional[dict] = None) -> tuple[str, str]:
    """
    (model_id, label) for a field's judge (MCAL_PLAN 3.11, 7 Q2).

    Sonnet by default; the five `summary.*` subfields plus `summary_of_interest`
    go to Opus. `confidence_config.judge_model_by_field` overrides, and accepts
    either a short label ("opus") or a full model id, because the config is
    hand-editable and a user pinning a specific Bedrock model id is a reasonable
    thing to want.
    """
    override = ((config or {}).get("judge_model_by_field") or {}).get(field)
    raw = str(override).strip() if override else ""
    if raw:
        low = raw.lower()
        if low == "opus":
            return settings.MODEL_OPUS, "opus"
        if low == "sonnet":
            return settings.MODEL_SONNET, "sonnet"
        return raw, _model_label(raw)
    model = settings.judge_model_for_field(field)
    return model, _model_label(model)


def _model_label(model: str) -> str:
    low = (model or "").lower()
    if "opus" in low:
        return "opus"
    if "sonnet" in low:
        return "sonnet"
    if "haiku" in low:
        return "haiku"
    return model or "unknown"


# --- Extraction access ------------------------------------------------------


def extracted_entry(field: str, m1: Optional[dict], m2: Optional[dict]) -> Any:
    """
    The raw M1/M2 entry for a field, wrapper and all.

    Generalizes segment_a's `_extracted_for_field` from 9 coarse keys to the 15
    canonical keys in `settings.ALL_FIELDS`. `summary.X` maps onto
    `m2["summary"]["X"]`, which is the shape `m2.py:SUMMARY_SCHEMA_KEYS` writes.
    A sentinel is NOT returned for a missing field -- None is returned and the
    caller routes to HUMAN_REVIEW with `extraction_missing`, because "the
    extractor produced nothing" and "the extractor produced an empty answer" are
    different facts and MCAL_PLAN 3.12 requires them to stay distinguishable.
    """
    m1 = m1 or {}
    m2 = m2 or {}
    if field in settings.M1_FIELDS:
        return m1.get(field)
    if field.startswith("summary."):
        return (m2.get("summary") or {}).get(field.split(".", 1)[1])
    return m2.get(field)


def extracted_value(field: str, entry: Any) -> Any:
    """
    The value a human would grade, unwrapped from its M1/M2 envelope.

    This is what lands in `run_manifest.json.extracted_value` (MCAL_PLAN 3.12:
    "the actual extraction -- string or structured object"), so the envelope
    metadata (`confidence`, `sources`, `evidence`) is deliberately dropped: a
    reviewer grading a gated field needs the answer, not the plumbing.
    """
    if entry is None:
        return None
    if field.startswith("summary."):
        if isinstance(entry, dict):
            return entry.get("text")
        return entry
    if isinstance(entry, dict) and "value" in entry:
        return entry.get("value")
    return entry


def is_missing(field: str, entry: Any) -> bool:
    """
    Did the extractor fail to produce this field at all?

    Emptiness is NOT missingness. `summary_of_interest == []` is a legitimate
    result (MCAL_PLAN 3.15 rule 2) and must never be coerced into a failure, so
    only `None` (and, for the envelope shapes, a `None` value) counts as missing.
    """
    if entry is None:
        return True
    value = extracted_value(field, entry)
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def is_legitimately_empty(field: str, entry: Any) -> bool:
    """`summary_of_interest == []`: a substantive 'this document is routine'."""
    if field not in EMPTY_IS_VALID_FIELDS or entry is None:
        return False
    value = extracted_value(field, entry)
    return isinstance(value, (list, tuple)) and len(value) == 0


def evidence_dicts(field: str, entry: Any) -> list[dict]:
    """
    Every `{quote, source_pages}` evidence dict attached to a field's extraction.

    One walker instead of segment_a's five hand-written per-field branches. The
    shapes differ (`summary.X.evidence`, `alternatives.value[i].evidence`,
    `location.value.places[i].evidence`, `key_people.value.<bucket>[i].evidence`,
    `summary_of_interest[i].evidence`) but they are all "an `evidence` list
    somewhere inside", so a bounded recursive walk is both shorter and immune to
    the next shape change -- `consulted_entities`, added by
    key_people_pipeline.py after segment_a/critic.py was written, is picked up
    with no extra code.
    """
    out: list[dict] = []
    seen: set[int] = set()

    def walk(node: Any, depth: int) -> None:
        if depth > 6 or len(out) > 400:
            return
        if isinstance(node, dict):
            if id(node) in seen:
                return
            seen.add(id(node))
            ev = node.get("evidence")
            if isinstance(ev, list):
                for item in ev:
                    if isinstance(item, dict) and "quote" in item:
                        out.append(item)
            elif isinstance(ev, dict) and "quote" in ev:
                out.append(ev)
            for key, child in node.items():
                if key == "evidence":
                    continue
                walk(child, depth + 1)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child, depth + 1)

    walk(entry, 0)
    return out


def source_pages_for_field(field: str, m1: Optional[dict], m2: Optional[dict]) -> list[int]:
    """
    Cited pages for a field, as ints.

    M1 fields default to the front matter (`1-3`), matching segment_a, but an M1
    entry that carries its own evidence -- `year_adjudicator.py` cites the
    signature page, which MCAL_PLAN 1(1) says is often page 70, not page 2 --
    overrides that default. Without this the year Critic would be handed the
    cover pages and asked to verify a quote from a transmittal letter.

    NOT capped. segment_a capped the span list at 6-10 entries per field; that
    cap silently removed pages from the set the quote-verify override searches,
    which turns a verifiable quote into an unverifiable one for a bookkeeping
    reason. The prompt-size problem the cap existed to solve is handled where it
    belongs, in `build_evidence_section`, which drops CONTEXT pages before cited
    ones and reports what it dropped.
    """
    entry = extracted_entry(field, m1, m2)
    pages: list[int] = []
    for ev in evidence_dicts(field, entry):
        pages.extend(quote_check.coerce_pages(ev.get("source_pages")))
    if isinstance(entry, dict):
        pages.extend(quote_check.coerce_pages(entry.get("source_pages")))
    # summary_of_interest entries carry a scalar `page` alongside their evidence.
    value = extracted_value(field, entry)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("page") is not None:
                pages.extend(quote_check.coerce_pages(item.get("page")))
    if not pages and field in settings.M1_FIELDS:
        pages = [1, 2, 3]
    return sorted({p for p in pages if p >= 1})


# --- EVIDENCE section (MCAL_PLAN 7 Q3) --------------------------------------


@dataclass
class EvidenceBlock:
    """Rendered EVIDENCE section plus the bookkeeping that explains it."""

    text: str
    cited_pages: list[int]
    pages_included: list[int]
    pages_omitted: list[int]
    window_mode: str            # "contiguous" | "clustered" | "none"
    truncated: bool
    n_chars: int

    def to_dict(self) -> dict:
        return {
            "cited_pages": self.cited_pages,
            "pages_included": self.pages_included,
            "pages_omitted": self.pages_omitted,
            "window_mode": self.window_mode,
            "truncated": self.truncated,
            "n_chars": self.n_chars,
        }


def evidence_pages(
    cited: Sequence[int],
    *,
    tolerance: int = EVIDENCE_PAGE_TOLERANCE,
    max_span: int = EVIDENCE_MAX_CONTIGUOUS_SPAN_PAGES,
) -> tuple[list[int], str]:
    """
    The page window the Critic sees, and which rule produced it.

    MCAL_PLAN 7 Q3 asks for `[min(cited)-2 .. max(cited)+2]`. That is right for a
    tight citation set and pathological for a scattered one, so the literal rule
    applies while `max(cited) - min(cited) <= max_span` and per-citation windows
    are merged beyond it. Both modes are supersets of `cited +/- tolerance`, which
    is the range `quote_check.check_quote` searches, so the Critic is never shown
    less than the deterministic checker will look at.
    """
    pages = sorted({int(p) for p in cited if int(p) >= 1})
    if not pages:
        return [], "none"
    span = pages[-1] - pages[0]
    if span <= max_span:
        lo = max(1, pages[0] - tolerance)
        hi = pages[-1] + tolerance
        return list(range(lo, hi + 1)), "contiguous"
    return quote_check.expand_with_tolerance(pages, tolerance), "clustered"


def build_evidence_section(
    doc: Doc,
    cited: Sequence[int],
    *,
    tolerance: int = EVIDENCE_PAGE_TOLERANCE,
    max_chars: int = EVIDENCE_MAX_CHARS,
) -> EvidenceBlock:
    """
    Full OCR text of the evidence window, interleaved with `[[PAGE n]]` markers.

    When the char budget binds, pages are dropped by distance from the nearest
    CITED page, furthest first -- so a cited page is only ever dropped after
    every context page has been. segment_a's blunt `[:80_000]` truncated
    mid-sentence at whatever page happened to be at the boundary, which meant a
    field citing a late page could have its evidence removed entirely while
    keeping 80k chars of irrelevant context, with nothing in the output to say
    so.
    """
    window, mode = evidence_pages(cited, tolerance=tolerance)
    if not window:
        return EvidenceBlock(
            text="", cited_pages=list(cited), pages_included=[], pages_omitted=[],
            window_mode=mode, truncated=False, n_chars=0,
        )

    by_num = {p.page_num: p.text for p in doc.pages}
    cited_set = {int(p) for p in cited}
    available = [p for p in window if p in by_num]
    # (distance from nearest cited page, page number) -> keep-priority order.
    ranked = sorted(
        available,
        key=lambda p: (min((abs(p - c) for c in cited_set), default=0), p),
    )

    budget = max_chars
    kept: dict[int, str] = {}
    omitted: list[int] = []
    truncated = False
    for pnum in ranked:
        raw = by_num.get(pnum) or ""
        marker = f"[[PAGE {pnum}]]\n"
        body = raw[:EVIDENCE_MAX_CHARS_PER_PAGE]
        if len(body) < len(raw):
            truncated = True
        cost = len(marker) + len(body) + 2
        if cost > budget:
            if not kept:
                # Even one page does not fit: keep a truncated slice of the most
                # relevant page rather than sending an empty EVIDENCE section.
                room = max(0, budget - len(marker) - 2)
                kept[pnum] = body[:room]
                truncated = True
                budget = 0
                continue
            omitted.append(pnum)
            truncated = True
            continue
        kept[pnum] = body
        budget -= cost

    parts = [f"[[PAGE {p}]]\n{kept[p]}" for p in sorted(kept)]
    text = "\n\n".join(parts)
    if truncated:
        dropped_cited = sorted(p for p in omitted if p in cited_set)
        log.warning(
            "EVIDENCE cap of %d chars bit: %d of %d window pages included, "
            "%d omitted (%d of them CITED: %s), mode=%s",
            max_chars, len(kept), len(available), len(omitted),
            len(dropped_cited), dropped_cited, mode,
        )
    return EvidenceBlock(
        text=text,
        cited_pages=sorted(cited_set),
        pages_included=sorted(kept),
        pages_omitted=sorted(omitted),
        window_mode=mode,
        truncated=truncated,
        n_chars=len(text),
    )


# --- Capacity findings (MCAL_PLAN 3.5 private-individual policy) -------------


@dataclass
class CapacityFinding:
    """One commenter whose stance capacity bears on the policy override."""

    name: str
    stance: Optional[str]
    capacity: str               # private | non_private | ambiguous
    basis: str
    source: str                 # "pipeline" | "classified" | "legacy_kind"

    @property
    def requires_human_review(self) -> bool:
        return self.capacity in ("private", "ambiguous")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "stance": self.stance,
            "capacity": self.capacity,
            "basis": self.basis,
            "source": self.source,
            "requires_human_review": self.requires_human_review,
        }


def capacity_findings(entry: Any, doc: Optional[Doc]) -> list[CapacityFinding]:
    """
    Capacity of every stance-bearing commenter in a `key_people` extraction.

    Three input shapes are handled, newest first, because Segment B will run over
    documents extracted by both pipelines during the migration:

      1. `key_people_pipeline.py` output, which already carries a `capacity`
         block from `classify_capacity` and a per-entity `human_review` flag.
         Trusted as-is -- re-deriving it here could disagree with the extractor's
         own routing, and the extractor had the section text in hand.
      2. segment_a `m2.py` output, which carries `{name, kind, stance, evidence}`
         and no capacity at all. `classify_capacity` is run here against the
         cited passage, so the policy applies to legacy extractions too.
      3. Neither -- `kind == "private"` alone, segment_a/critic.py's original
         test, kept as a last resort.

    Only stance-bearing entries matter: MCAL_PLAN 3.5 Q5 is about "a stance about
    a private individual", and a private individual merely listed as a draft
    recipient is not a policy trigger.
    """
    value = extracted_value("key_people", entry)
    if not isinstance(value, dict):
        return []
    findings: list[CapacityFinding] = []
    for commenter in value.get("public_commenters") or []:
        if not isinstance(commenter, dict):
            continue
        stance = commenter.get("stance")
        if not stance:
            continue
        name = str(commenter.get("name") or "").strip()

        cap = commenter.get("capacity")
        if isinstance(cap, dict) and cap.get("capacity"):
            findings.append(
                CapacityFinding(
                    name=name,
                    stance=str(stance),
                    capacity=str(cap.get("capacity")),
                    basis=str(cap.get("basis") or ""),
                    source="pipeline",
                )
            )
            continue
        if commenter.get("human_review") is True:
            findings.append(
                CapacityFinding(
                    name=name, stance=str(stance), capacity="ambiguous",
                    basis="extractor_flagged_human_review", source="pipeline",
                )
            )
            continue

        if doc is not None:
            ev_list = commenter.get("evidence") or []
            ev = ev_list[0] if isinstance(ev_list, list) and ev_list else {}
            passage = cited_passage(doc, ev if isinstance(ev, dict) else {})
            derived = classify_capacity(
                name, passage, kind=str(commenter.get("kind") or "") or None
            )
            findings.append(
                CapacityFinding(
                    name=name, stance=str(stance), capacity=derived.capacity,
                    basis=derived.basis, source="classified",
                )
            )
            continue

        kind = str(commenter.get("kind") or "").strip().lower()
        findings.append(
            CapacityFinding(
                name=name,
                stance=str(stance),
                capacity="private" if kind == "private" else "non_private",
                basis=f"extractor_kind={kind or 'unset'}",
                source="legacy_kind",
            )
        )
    return findings


# --- Result type ------------------------------------------------------------


@dataclass
class CriticResult:
    """
    One field's Critic outcome (MCAL_PLAN 3.11).

    `verdict_before_override` is the schema-validated verdict the model actually
    returned, and `overrides` names every deterministic layer that changed it
    afterwards. Keeping both is what makes a HUMAN_REVIEW auditable: MCAL_PLAN 6
    tracks the Critic's `evidence_quote` verifiable rate as a diagnostic, and
    that number is unrecoverable if the override overwrites its own input.
    """

    field: str
    bucket: str
    evidence_quote: Optional[str] = None
    rubric_answers: dict[str, str] = dc_field(default_factory=dict)
    verdict: str = VERDICT_HUMAN_REVIEW
    verdict_before_override: Optional[str] = None
    failure_tag: Optional[str] = None
    note: Optional[str] = None
    judge_model: str = "sonnet"
    judge_model_id: str = ""
    source_pages: list[int] = dc_field(default_factory=list)
    quote_check: Optional[dict] = None
    # --- audit trail ---
    overrides: list[str] = dc_field(default_factory=list)
    notes: list[str] = dc_field(default_factory=list)
    schema_violations: list[str] = dc_field(default_factory=list)
    off_vocabulary_failure_tag: Optional[str] = None
    evidence_meta: dict = dc_field(default_factory=dict)
    capacity_findings: list[dict] = dc_field(default_factory=list)
    usage: dict = dc_field(default_factory=dict)
    empty_but_valid: bool = False
    extraction_missing: bool = False

    def add_note(self, note: str) -> None:
        if note and note not in self.notes:
            self.notes.append(note)
        self.note = "; ".join(self.notes) or None

    def override_to(self, verdict: str, reason: str) -> None:
        """Apply a deterministic override, preserving the model's own verdict."""
        if self.verdict_before_override is None:
            self.verdict_before_override = self.verdict
        if reason not in self.overrides:
            self.overrides.append(reason)
        self.verdict = verdict
        self.add_note(reason)

    def force(self, verdict: str, reason: str) -> None:
        """
        Set a verdict where the model never produced one.

        Distinct from `override_to` because `verdict_before_override` means "the
        verdict the judge returned before deterministic layers touched it". A
        failed call, a missing extraction, or a response coerced during parsing
        has no such verdict, and recording the substituted value as though the
        model had returned it would inflate MCAL_PLAN 6's Critic diagnostics with
        verdicts no judge ever emitted.
        """
        if reason not in self.overrides:
            self.overrides.append(reason)
        self.verdict = verdict
        self.add_note(reason)

    def to_dict(self) -> dict:
        return {
            # MCAL_PLAN 3.5 output schema order: evidence before verdict.
            "evidence_quote": self.evidence_quote,
            "rubric_answers": dict(self.rubric_answers),
            "verdict": self.verdict,
            "verdict_before_override": self.verdict_before_override,
            "failure_tag": self.failure_tag,
            "note": self.note,
            "judge_model": self.judge_model,
            "judge_model_id": self.judge_model_id,
            "source_pages": list(self.source_pages),
            "quote_check": self.quote_check,
            "field": self.field,
            "bucket": self.bucket,
            "overrides": list(self.overrides),
            "notes": list(self.notes),
            "schema_violations": list(self.schema_violations),
            "off_vocabulary_failure_tag": self.off_vocabulary_failure_tag,
            "evidence_meta": dict(self.evidence_meta),
            "capacity_findings": list(self.capacity_findings),
            "usage": dict(self.usage),
            "empty_but_valid": self.empty_but_valid,
            "extraction_missing": self.extraction_missing,
        }

    @classmethod
    def from_dict(cls, payload: dict, *, field: Optional[str] = None) -> "CriticResult":
        """
        Rehydrate from `to_dict()` output.

        Needed so a driver can reuse a checkpointed `critic.json` instead of
        re-judging a document. The per-field Critic split makes a full pass
        materially more expensive than Segment A's, so a rerun that only wants to
        redo the gate must not have to pay for the judgements again.

        Unknown keys are ignored rather than raising: `critic.json` files written
        by an older build should still be loadable, and a schema addition should
        not invalidate an expensive artifact.
        """
        p = dict(payload or {})
        fld = field or p.get("field") or ""
        bucket = p.get("bucket") or (
            settings.bucket_for_field(fld) if fld in settings.FIELD_TO_BUCKET else ""
        )
        return cls(
            field=fld,
            bucket=bucket,
            evidence_quote=p.get("evidence_quote"),
            rubric_answers=dict(p.get("rubric_answers") or {}),
            verdict=p.get("verdict") or VERDICT_HUMAN_REVIEW,
            verdict_before_override=p.get("verdict_before_override"),
            failure_tag=p.get("failure_tag"),
            note=p.get("note"),
            judge_model=p.get("judge_model") or "sonnet",
            judge_model_id=p.get("judge_model_id") or "",
            source_pages=list(p.get("source_pages") or []),
            quote_check=p.get("quote_check"),
            overrides=list(p.get("overrides") or []),
            notes=list(p.get("notes") or []),
            schema_violations=list(p.get("schema_violations") or []),
            off_vocabulary_failure_tag=p.get("off_vocabulary_failure_tag"),
            evidence_meta=dict(p.get("evidence_meta") or {}),
            capacity_findings=list(p.get("capacity_findings") or []),
            usage=dict(p.get("usage") or {}),
            empty_but_valid=bool(p.get("empty_but_valid")),
            extraction_missing=bool(p.get("extraction_missing")),
        )


def results_from_payload(payload: dict) -> dict[str, "CriticResult"]:
    """
    Rehydrate a whole `critic.json` into `{field: CriticResult}`.

    Fields absent from the payload are NOT synthesized here. `gate.run_gate`
    iterates `settings.ALL_FIELDS` and emits `gate_reason="critic_missing"` for
    anything it does not find, which is the honest representation of a partial
    checkpoint -- inventing a HUMAN_REVIEW result would make an incomplete
    Critic run indistinguishable from a completed one that deferred.
    """
    return {
        field: CriticResult.from_dict(entry, field=field)
        for field, entry in (payload or {}).items()
        if isinstance(entry, dict)
    }


# --- Response validation ----------------------------------------------------


def _normalize_rubric_key(key: str) -> str:
    """`q6B` / `Q6B` / ` q6b ` -> `Q6b`, so the manifest's keys are stable."""
    s = str(key or "").strip()
    if not s:
        return ""
    if s[0] in "qQ":
        return "Q" + s[1:].lower()
    return s


def normalize_rubric_answers(
    raw: Any, field: str, violations: list[str]
) -> dict[str, str]:
    """
    Coerce `rubric_answers` into a stable `{Qn: yes|no|n/a}` map.

    Required keys are filled with `n/a` rather than dropped: MCAL_PLAN 3.12 reads
    `rubric_answers.Q6b` out of the manifest for an offline audit, and a missing
    key would make an unanswered question indistinguishable from a question the
    field never asked. Unrecognized answer strings are preserved verbatim and
    recorded as schema violations -- silently rewriting them to `no` would invent
    a defect, which is the failure mode this whole module exists to prevent.
    """
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            nk = _normalize_rubric_key(key)
            if not nk:
                continue
            sv = str(value).strip().lower() if value is not None else ""
            out[nk] = RUBRIC_ANSWER_SYNONYMS.get(sv, sv or RUBRIC_ANSWER_MISSING)
            if sv and sv not in RUBRIC_ANSWER_SYNONYMS:
                violations.append(f"rubric_answer_unrecognized:{nk}={sv[:24]}")
    elif raw is not None:
        violations.append(f"rubric_answers_not_an_object:{type(raw).__name__}")

    required = list(BASE_RUBRIC_KEYS)
    if field == settings.SUMMARY_OF_INTEREST:
        required += list(SOI_RUBRIC_KEYS)
    for key in required:
        out.setdefault(key, RUBRIC_ANSWER_MISSING)
    return out


def _coerce_verdict(raw: Any, violations: list[str]) -> str:
    """
    Unknown verdict -> HUMAN_REVIEW, exactly as segment_a/critic.py did.

    Failing safe is not optional here: `confidence.s_critic_from_verdict` scores
    an unrecognized verdict 0.0, so accepting one would put a field into the
    composite with a score that contradicts its recorded verdict.
    """
    v = str(raw or "").strip().upper()
    if v in VERDICTS:
        return v
    violations.append(f"{NOTE_UNKNOWN_VERDICT}:{str(raw)[:32]}")
    return VERDICT_HUMAN_REVIEW


def _coerce_failure_tag(
    raw: Any, field: str, stage: str, violations: list[str]
) -> tuple[Optional[str], Optional[str]]:
    """
    (canonical_tag, off_vocabulary_original).

    An off-vocabulary tag becomes null AND is reported, because the null-tag
    monitor (MCAL_PLAN 6) treats "HUMAN_REVIEW with failure_tag = null" as
    evidence the taxonomy needs new T19+ codes. If a hallucinated tag were kept,
    that signal would be diluted; if it were dropped silently, a systematically
    off-vocabulary judge would look like a taxonomy gap. Both matter, so both are
    recorded.
    """
    if raw is None:
        return None, None
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none", "n/a", "-"):
        return None, None
    canonical = tag_vocabulary(field, stage).get(s.lower())
    if canonical:
        return canonical, None
    violations.append(f"{NOTE_OFF_VOCABULARY_TAG}:{s[:40]}")
    return None, s


def _coerce_evidence_quote(raw: Any, violations: list[str]) -> Optional[str]:
    if raw is None:
        return None
    if not isinstance(raw, str):
        violations.append(f"evidence_quote_not_a_string:{type(raw).__name__}")
        raw = str(raw)
    q = raw.strip()
    return q or None


def _check_key_order(raw: Any, violations: list[str]) -> None:
    """
    Enforce MCAL_PLAN 3.11's "schema field order enforced".

    `json.loads` preserves object key order, so the order the model emitted is
    observable. A violation is recorded but not fatal: the ordering rule is a
    behavioural nudge (committing to evidence before a verdict reduces post-hoc
    rationalization), while the enforcement that actually matters is the
    deterministic quote check below. Failing the response outright would trade a
    real verdict for a schema complaint.
    """
    if not isinstance(raw, dict):
        return
    keys = list(raw.keys())
    if "verdict" in keys and "evidence_quote" in keys:
        if keys.index("verdict") < keys.index("evidence_quote"):
            violations.append(NOTE_SCHEMA_ORDER)


# --- The call ---------------------------------------------------------------


def build_user_message(
    field: str,
    value: Any,
    source_pages: Sequence[int],
    evidence: EvidenceBlock,
    *,
    capacity: Sequence[CapacityFinding] = (),
    empty_but_valid: bool = False,
) -> str:
    """
    The user half of the Critic call: value, citations, then EVIDENCE.

    EVIDENCE is a dedicated, last section (MCAL_PLAN 7 Q3) -- last because it is
    by far the largest block and putting the instructions after it would bury
    them. The system half is the per-field prompt from `mcal/critic_prompt.py`,
    which already carries the role header, anti-hallucination clause, rubric,
    decision table, tag vocabulary and output schema.
    """
    rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(rendered) > MAX_EXTRACTED_VALUE_CHARS:
        rendered = rendered[:MAX_EXTRACTED_VALUE_CHARS] + "\n... [value truncated]"

    parts = [f"## FIELD UNDER REVIEW\n\n`{field}`"]
    parts.append(f"## EXTRACTED VALUE\n\n```json\n{rendered}\n```")
    parts.append(
        "## CITED PAGES\n\n"
        + (", ".join(str(p) for p in source_pages) if source_pages else "(none cited)")
    )
    if empty_but_valid:
        parts.append(
            "## NOTE ON EMPTINESS\n\n"
            "This value is an EMPTY LIST. For `summary_of_interest` an empty list "
            "is a CORRECT and expected result for a routine document "
            "(MCAL_PLAN 3.15 rule 2, rubric Q7c). Do not treat emptiness as a "
            "defect, and do not invent a reason to fail it. If emptiness is "
            "plausible for this document, answer Q7c `yes` and return `PASS` "
            "with `evidence_quote: null`."
        )
    if capacity:
        # The deterministic capacity read is put in front of the Critic so its Q5
        # answer is grounded in the same cue evidence the extractor used, rather
        # than in the model's own guess about who a person is.
        lines = [
            f"- {c.name or '(unnamed)'} — stance `{c.stance}` — capacity "
            f"`{c.capacity}` ({c.basis})"
            for c in capacity
        ]
        parts.append(
            "## DETERMINISTIC CAPACITY FINDINGS (advisory)\n\n"
            "Computed from the cited passage by the extraction pipeline, using "
            "the PRIVATE_INDIVIDUAL definition above. `private` or `ambiguous` "
            "forces HUMAN_REVIEW as policy regardless of your verdict.\n\n"
            + "\n".join(lines)
        )

    if evidence.text:
        header = f"## EVIDENCE\n\nOCR text of pages {_page_summary(evidence.pages_included)}"
        if evidence.truncated:
            header += (
                f" (TRUNCATED to {evidence.n_chars} chars; pages "
                f"{_page_summary(evidence.pages_omitted)} omitted — if the "
                f"support you need is not here, that is `RE_EXTRACT`, not a "
                f"judgement that the document lacks it)"
            )
        header += ", interleaved with `[[PAGE n]]` markers.\n\n"
        parts.append(header + evidence.text)
    else:
        parts.append(
            "## EVIDENCE\n\n(no cited pages resolved in this document — you have "
            "no evidence to verify against; answer Q1 `no`)"
        )
    return "\n\n".join(parts)


def _page_summary(pages: Sequence[int]) -> str:
    """Compact page list: [3,4,5,9] -> '3-5, 9'."""
    ps = sorted(set(int(p) for p in pages))
    if not ps:
        return "(none)"
    runs: list[tuple[int, int]] = []
    start = prev = ps[0]
    for p in ps[1:]:
        if p == prev + 1:
            prev = p
            continue
        runs.append((start, prev))
        start = prev = p
    runs.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _default_call(
    model: str, system: str, user: str, *, max_tokens: int, temperature: float
) -> tuple[dict, dict]:
    return call_json_with_usage(
        model, system, user, max_tokens=max_tokens, temperature=temperature
    )


def _invoke(call: Callable, model, system, user, *, max_tokens, temperature):
    """
    Normalize an injected callable's return to `(payload, usage)`.

    `llm.call_json_with_usage` returns a tuple; `llm.call_json` and test fakes
    return just the payload. Accepting both keeps test doubles trivial without
    losing the usage figures that feed the cost summary.
    """
    out = call(model, system, user, max_tokens=max_tokens, temperature=temperature)
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        return out[0], out[1]
    return out, {}


# --- Per-field judging ------------------------------------------------------


def critique_field(
    field: str,
    doc: Doc,
    m1: Optional[dict],
    m2: Optional[dict],
    *,
    stage: str,
    config: Optional[dict] = None,
    call: Optional[Callable] = None,
    max_tokens: int = MAX_TOKENS,
    evidence_max_chars: int = EVIDENCE_MAX_CHARS,
    extracted_override: Any = None,
    temperature: float = TEMPERATURE,
) -> CriticResult:
    """
    Judge ONE field. Never raises.

    `extracted_override` lets `gate.py` re-run the Critic over a re-extracted
    value (MCAL_PLAN 7 Q8) without rebuilding the whole M2 payload.
    """
    bucket = settings.bucket_for_field(field)
    model_id, model_label = judge_model_for(field, config)
    result = CriticResult(
        field=field, bucket=bucket, judge_model=model_label, judge_model_id=model_id
    )

    entry = extracted_entry(field, m1, m2) if extracted_override is None else extracted_override
    value = extracted_value(field, entry)
    result.empty_but_valid = is_legitimately_empty(field, entry)
    result.extraction_missing = is_missing(field, entry) and not result.empty_but_valid

    pages = source_pages_for_field(field, m1, m2)
    if extracted_override is not None:
        override_pages: list[int] = []
        for ev in evidence_dicts(field, entry):
            override_pages.extend(quote_check.coerce_pages(ev.get("source_pages")))
        if override_pages:
            pages = sorted(set(override_pages))
    result.source_pages = pages

    capacity = capacity_findings(entry, doc) if field == "key_people" else []
    result.capacity_findings = [c.to_dict() for c in capacity]

    if result.extraction_missing:
        # No value at all: there is nothing for a judge to verify, and paying for
        # a call to be told so is waste. MCAL_PLAN 7 Q8 still requires the field
        # to be emitted, which gate.py does from this result.
        result.rubric_answers = normalize_rubric_answers(None, field, result.schema_violations)
        result.force(VERDICT_HUMAN_REVIEW, NOTE_EXTRACTION_MISSING)
        return result

    evidence = build_evidence_section(
        doc, pages, tolerance=EVIDENCE_PAGE_TOLERANCE, max_chars=evidence_max_chars
    )
    result.evidence_meta = evidence.to_dict()

    try:
        system = prompt_for(field, stage)
    except FileNotFoundError as e:
        # A missing prompt is a build problem, not a document problem, and it
        # affects every document identically. Surfacing it is more useful than
        # 2,000 HUMAN_REVIEW routes with an obscure note.
        raise MissingArtifactError(str(e)) from e

    user = build_user_message(
        field, value, pages, evidence,
        capacity=capacity, empty_but_valid=result.empty_but_valid,
    )

    fn = call or _default_call
    try:
        raw, usage = _invoke(
            fn, model_id, system, user, max_tokens=max_tokens, temperature=temperature
        )
    except Exception as e:  # noqa: BLE001 - one bad call must not lose a document
        log.warning("Critic call failed on %s (%s): %s", field, model_label, e)
        result.rubric_answers = normalize_rubric_answers(None, field, result.schema_violations)
        result.force(VERDICT_HUMAN_REVIEW, f"{NOTE_LLM_FAILED}:{type(e).__name__}")
        result.add_note(str(e)[:200])
        return result

    result.usage = usage or {}
    if not isinstance(raw, dict):
        result.schema_violations.append(f"response_not_an_object:{type(raw).__name__}")
        raw = {}

    _check_key_order(raw, result.schema_violations)
    result.evidence_quote = _coerce_evidence_quote(
        raw.get("evidence_quote"), result.schema_violations
    )
    result.rubric_answers = normalize_rubric_answers(
        raw.get("rubric_answers"), field, result.schema_violations
    )
    result.verdict = _coerce_verdict(raw.get("verdict"), result.schema_violations)
    tag, off_vocab = _coerce_failure_tag(
        raw.get("failure_tag"), field, stage, result.schema_violations
    )
    result.failure_tag = tag
    result.off_vocabulary_failure_tag = off_vocab
    if off_vocab:
        result.add_note(f"{NOTE_OFF_VOCABULARY_TAG}:{off_vocab[:40]}")
    if NOTE_SCHEMA_ORDER in result.schema_violations:
        result.add_note(NOTE_SCHEMA_ORDER)
    note = raw.get("note")
    if isinstance(note, str) and note.strip():
        result.add_note(note.strip()[:400])
    if any(v.startswith(NOTE_UNKNOWN_VERDICT) for v in result.schema_violations):
        result.force(VERDICT_HUMAN_REVIEW, NOTE_UNKNOWN_VERDICT)

    apply_quote_verify_override(result, doc, tolerance=EVIDENCE_PAGE_TOLERANCE)
    apply_policy_overrides(result, capacity)
    return result


def apply_quote_verify_override(
    result: CriticResult, doc: Doc, *, tolerance: int = EVIDENCE_PAGE_TOLERANCE
) -> CriticResult:
    """
    The deterministic anti-hallucination layer (MCAL_PLAN 3.11, 4 Q2 layer 3).

    Re-verify the Critic's OWN `evidence_quote` against the cited pages with
    `mcal/quote_check.py`. Anything short of `verified == "yes"` -- including
    `mixed` -- overrides the verdict to HUMAN_REVIEW with note
    `critic_evidence_unverifiable`. `mixed` is included deliberately: a
    half-matching quote is exactly what a paraphrase-from-memory looks like, and
    a judge that cannot copy 20 characters accurately has not demonstrated it
    read the page.

    Two documented exemptions, both about not manufacturing a failure:

      * a legitimately empty `summary_of_interest` has no claim to support, so
        there is no quote to verify and the check is skipped (MCAL_PLAN 3.15
        rule 2 / rubric Q7c: emptiness must never be penalized);
      * a verdict of RE_EXTRACT with `evidence_quote = null` is the response
        MCAL_PLAN 3.5 explicitly PRESCRIBES when no supporting substring exists.
        Overriding it to HUMAN_REVIEW would suppress the automated re-extraction
        retry (7 Q8) and send a mechanically fixable field to a human instead.

    Everything else with a null or unverifiable quote is overridden, including a
    PASS with no quote -- a PASS is a claim that support exists, so failing to
    produce it is the failure the layer is for.
    """
    if result.empty_but_valid:
        result.quote_check = None
        result.add_note("quote_verify_skipped:empty_summary_of_interest_is_valid")
        return result

    if result.evidence_quote is None:
        if result.verdict == VERDICT_RE_EXTRACT:
            result.quote_check = None
            result.add_note("evidence_quote_null_consistent_with_RE_EXTRACT")
            return result
        result.quote_check = {"verified": "no", "reason": "no_evidence_quote_returned"}
        result.override_to(VERDICT_HUMAN_REVIEW, NOTE_EVIDENCE_UNVERIFIABLE)
        return result

    check: QuoteCheck = quote_check.check_quote(
        result.evidence_quote,
        result.source_pages,
        doc,
        tolerance=tolerance,
        # Never widen to the whole document. The question is whether the quote is
        # on the pages the extraction CITED; finding it elsewhere would confirm a
        # mis-citation while reporting it as support.
        search_whole_doc_if_no_pages=False,
    )
    result.quote_check = check.to_dict()
    if check.verified != "yes":
        result.override_to(VERDICT_HUMAN_REVIEW, NOTE_EVIDENCE_UNVERIFIABLE)
    return result


def apply_policy_overrides(
    result: CriticResult, capacity: Sequence[CapacityFinding] = ()
) -> CriticResult:
    """
    Private-individual and dual-capacity policy (MCAL_PLAN 3.5, 3.11, 5).

    Permanent policy, not calibration: MCAL_PLAN 5 lists "Removing
    private-individual -> HUMAN_REVIEW" under explicitly deferred/skipped with
    the annotation "(policy call, permanent)". So this override is unconditional
    and has no threshold.

    Generalized from segment_a's `_apply_private_commenter_override`, which only
    looked at `key_people.value.public_commenters[*].kind == "private"`, in two
    directions:

      * structurally, via `capacity_findings`, which understands the
        `key_people_pipeline.py` capacity block and falls back to running
        `classify_capacity` on legacy extractions;
      * by rubric, via Q5. `templates/rubrics/_base.md` asks Q5 of EVERY field
        and makes `Q5 = yes -> HUMAN_REVIEW, failure_tag = null` its first
        decision rule, because `summary.public_response` can attribute a stance
        to a private individual just as `key_people` can. segment_a checked only
        the structured field and would have passed such a summary.

    An ambiguous (dual-capacity) passage routes to HUMAN_REVIEW too, per
    MCAL_PLAN 3.5: "If the passage is ambiguous about which capacity is being
    expressed, route to HUMAN_REVIEW regardless of Critic verdict."
    """
    private = [c for c in capacity if c.capacity == "private"]
    ambiguous = [c for c in capacity if c.capacity == "ambiguous"]

    if private:
        names = ", ".join(c.name or "(unnamed)" for c in private[:4])
        result.override_to(VERDICT_HUMAN_REVIEW, NOTE_PRIVATE_INDIVIDUAL)
        result.add_note(f"private_individual_stance:{names}")
    elif ambiguous:
        names = ", ".join(c.name or "(unnamed)" for c in ambiguous[:4])
        result.override_to(VERDICT_HUMAN_REVIEW, NOTE_AMBIGUOUS_CAPACITY)
        result.add_note(f"ambiguous_capacity_stance:{names}")

    if result.rubric_answers.get("Q5") == "yes" and not (private or ambiguous):
        result.override_to(VERDICT_HUMAN_REVIEW, NOTE_PRIVATE_INDIVIDUAL)
        result.add_note("rubric_Q5_private_individual_stance")

    # MCAL_PLAN base decision rule 1: the policy route carries no failure tag --
    # a human review triggered by policy is not evidence of a defect, and letting
    # it carry a tag would pollute the tag distribution the next taxonomy
    # revision is fitted on.
    if NOTE_PRIVATE_INDIVIDUAL in result.overrides or NOTE_AMBIGUOUS_CAPACITY in result.overrides:
        result.failure_tag = None
    return result


# --- Dependent-field cascade ------------------------------------------------


def apply_dependent_cascade(
    results: dict[str, CriticResult], config: Optional[dict] = None
) -> dict[str, CriticResult]:
    """
    Cascade untrustworthy fields onto their dependents (MCAL_PLAN 3.10 step 2).

    `settings.DEPENDENT_FIELDS` declares `{year: [key_people]}`: if `year` is
    RE_EXTRACT or HUMAN_REVIEW, `key_people` cannot be era-gated -- the pre-1978
    branch in `key_people_pipeline.apply_era_gate` turns on a year we have just
    said we do not believe -- so `key_people` becomes HUMAN_REVIEW.

    MUST run after every field is judged, which is why it is a separate pass over
    the finished result set rather than part of `critique_field`. Fields are
    judged independently and in arbitrary order, so a cascade applied during
    judging would depend on iteration order.

    The dependency map is read from `confidence_config.dependent_fields` when
    present, falling back to `settings.DEPENDENT_FIELDS`, so a recalibration can
    add a dependency without a code change. One pass only, deliberately: the
    declared graph is a single edge, and a transitive closure would let a future
    config typo cascade an entire document to HUMAN_REVIEW silently.
    """
    mapping = (config or {}).get("dependent_fields") or settings.DEPENDENT_FIELDS
    for source, dependents in (mapping or {}).items():
        src = results.get(source)
        if src is None or src.verdict not in UNTRUSTWORTHY_VERDICTS:
            continue
        for dep in dependents or []:
            target = results.get(dep)
            if target is None or target.verdict == VERDICT_HUMAN_REVIEW:
                if target is not None and NOTE_DEPENDENT_CASCADE not in target.overrides:
                    # Already HUMAN_REVIEW for another reason: record the cascade
                    # anyway so gate.py can report it, without losing the
                    # original, more specific reason (which is listed first).
                    target.overrides.append(NOTE_DEPENDENT_CASCADE)
                    target.add_note(f"{NOTE_DEPENDENT_CASCADE}:{source}={src.verdict}")
                continue
            target.override_to(VERDICT_HUMAN_REVIEW, NOTE_DEPENDENT_CASCADE)
            target.add_note(f"{NOTE_DEPENDENT_CASCADE}:{source}={src.verdict}")
    return results


# --- Top level --------------------------------------------------------------


def run_critic(
    doc: Doc,
    m1: Optional[dict],
    m2: Optional[dict],
    *,
    stage: Optional[str] = None,
    fields: Optional[Sequence[str]] = None,
    config: Optional[dict] = None,
    call: Optional[Callable] = None,
    max_tokens: int = MAX_TOKENS,
    evidence_max_chars: int = EVIDENCE_MAX_CHARS,
    apply_cascade: bool = True,
) -> dict[str, CriticResult]:
    """
    Judge every field of one document (MCAL_PLAN 3.11).

    Returns `{field: CriticResult}` with one entry per field in
    `settings.ALL_FIELDS` -- 15 entries, against segment_a's 9. Per-field
    granularity is the point of the rewrite: `s_critic` is half the composite,
    and a verdict shared across six summary subfields makes the summary buckets'
    conformal thresholds unfittable.

    `call` injects the LLM for tests. `config` skips the artifact read when the
    caller already holds `confidence_config`. Wrap the whole document in
    `llm.start_usage_session()` / `end_usage_session()` to capture the Critic's
    token cost.
    """
    stage = resolve_stage(stage)
    cfg = config if config is not None else load_confidence_config(stage)
    field_list = tuple(fields) if fields else settings.ALL_FIELDS

    results: dict[str, CriticResult] = {}
    for field in field_list:
        results[field] = critique_field(
            field, doc, m1, m2,
            stage=stage, config=cfg, call=call,
            max_tokens=max_tokens, evidence_max_chars=evidence_max_chars,
        )

    if apply_cascade:
        apply_dependent_cascade(results, cfg)
    return results


def critic_diagnostics(results: dict[str, CriticResult]) -> dict:
    """
    Per-document Critic diagnostics (MCAL_PLAN 6).

    `evidence_quote` verifiable rate is a named diagnostic in MCAL_PLAN 6;
    `n_off_vocabulary_tags` feeds the null-tag monitor in `gate.py`; the override
    counts are what tell you whether HUMAN_REVIEW volume is coming from the
    judge, from policy, or from the quote checker.
    """
    verifiable = [
        r for r in results.values()
        if r.quote_check is not None and not r.empty_but_valid
    ]
    n_yes = sum(1 for r in verifiable if (r.quote_check or {}).get("verified") == "yes")
    counts: dict[str, int] = {}
    for r in results.values():
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    overrides: dict[str, int] = {}
    for r in results.values():
        for o in r.overrides:
            key = o.split(":", 1)[0]
            overrides[key] = overrides.get(key, 0) + 1
    return {
        "n_fields": len(results),
        "verdicts": counts,
        "overrides": overrides,
        "n_quote_checked": len(verifiable),
        "n_quote_verified": n_yes,
        "evidence_quote_verifiable_rate": (
            round(n_yes / len(verifiable), 4) if verifiable else None
        ),
        "n_off_vocabulary_tags": sum(
            1 for r in results.values() if r.off_vocabulary_failure_tag
        ),
        "n_null_tag_human_review": sum(
            1 for r in results.values()
            if r.verdict == VERDICT_HUMAN_REVIEW and r.failure_tag is None
        ),
        "n_schema_violations": sum(len(r.schema_violations) for r in results.values()),
        "n_evidence_truncated": sum(
            1 for r in results.values() if r.evidence_meta.get("truncated")
        ),
    }


def as_dict(results: dict[str, CriticResult]) -> dict[str, dict]:
    """JSON-serializable form, for persisting alongside the manifest."""
    return {f: r.to_dict() for f, r in results.items()}
