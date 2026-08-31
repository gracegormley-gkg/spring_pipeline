"""
M2: Semantic Extraction.

Every extracted field carries an `evidence` list — each item is a verbatim
quote that has been located on a real page of the doc via
`evidence.verify_and_locate`. Quotes that cannot be found verbatim keep
`quote_verified=false` so the critic forces HUMAN_REVIEW.

Fields:
  - Summary: Opus, map-reduce. Each subfield's evidence is a list of verified
    verbatim quotes selected from the per-chunk findings.
  - Alternatives: Sonnet on the structurally-identified Alternatives chapter.
    Each alternative carries a verbatim quote naming/describing it.
  - Themes: Sonnet from the summary. Evidence is the union of summary evidence.
  - Location: Sonnet on first 30 pages + Project/Study Area. Each place carries
    a verbatim quote.
  - Key People: agency_preparers, cooperating_agencies, public_commenters —
    each entry carries a verbatim quote.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# `config` is imported as a MODULE, not via `from config import ...`, purely for
# SUMMARY_MAP_MODEL: a `from` import binds the value at import time, so flipping
# the switch (or a test monkeypatching it) would have no effect.
import config
from config import (
    CHUNK_PAGES,
    FIRST_30_PAGES,
    THEMES,
)
from chunk import (
    Chunk,
    chunks_for_doc,
    first_pages,
    text_for_ceq_chapter,
)
from evidence import (
    Evidence,
    evidence_for_quotes,
    mark_carried,
    union_pages,
    verify_and_locate,
)
from llm import opus, sonnet
from pages import Doc
from prompts import PROMPT_VERSION, plain_language_clause, summary_of_interest_prompt

log = logging.getLogger(__name__)


# --- Summary (Opus, map-reduce) ---------------------------------------------

SUMMARY_SCHEMA_KEYS = [
    "overview",
    "project_description",
    "affected_community",
    "alternatives_overview",
    "environmental_impact",
    "public_response",
]

# Keys whose evidence the `overview` field unions over.
SUMMARY_SUBFIELDS = [
    "project_description",
    "affected_community",
    "alternatives_overview",
    "environmental_impact",
    "public_response",
]

# --- Output budgets ---------------------------------------------------------
# Raised when the MCAL_PLAN 3.14 plain-language clause shipped (build item #4).
#
# The clause lengthens every subfield (in-line glosses, named entities,
# quantities instead of nominalizations) AND roughly doubles the quotes per
# subfield, because the concreteness rule produces more discrete claims and each
# claim needs its own citation. Measured on Operation Breakthrough before/after:
# public_response 6 -> 13 quotes, environmental_impact 7 -> 12, overview 11 -> 16.
#
# At the old SUMMARY_REDUCE budget of 6000, that document -- the SMALLEST of the
# eight graded, with only 2 chunks -- consumed ~4,629 output tokens, leaving 23%
# headroom. Documents with the full 12 chunks carry several times as many
# candidate quotes, and 4 of 8 truncated mid-string on the first rerun,
# surfacing as `JSONDecodeError: Unterminated string`.
#
# These are caps, not targets: raising them costs nothing unless the model
# actually needs the room. Undersizing them costs a whole document.
SUMMARY_MAP_MAX_TOKENS = 8_000
SUMMARY_REDUCE_MAX_TOKENS = 24_000
SUMMARY_OF_INTEREST_MAX_TOKENS = 8_000


def _summary_map_model():
    """
    The callable that runs the map step: `opus` or `sonnet`, per
    `config.SUMMARY_MAP_MODEL`.

    Resolved per call rather than bound at import so a test -- or a stage boundary
    -- can change it without reloading the module. See that config constant for
    why the default is Opus despite this being the pipeline's largest cost lever.
    """
    choice = (config.SUMMARY_MAP_MODEL or "opus").strip().lower()
    if choice == "sonnet":
        return sonnet
    if choice == "opus":
        return opus
    raise ValueError(
        f"config.SUMMARY_MAP_MODEL must be 'opus' or 'sonnet', got "
        f"{config.SUMMARY_MAP_MODEL!r}"
    )


def _summary_map_one(chunk: Chunk) -> dict:
    """Per-chunk summary map step. Returns text + verbatim supporting quotes per field.

    Each field returns a `quotes` LIST so the reducer has enough candidates to
    cite every claim it carries forward — earlier versions returned a single
    quote per field per chunk, which made the final per-doc summary
    under-cited (you'd see 3 claims in `text` backed by only 1-2 quotes).

    The model is `config.SUMMARY_MAP_MODEL` (default Opus). Note that Sonnet
    honours `temperature` while Opus ignores it, so flipping the switch also makes
    the MCAL_PLAN 7 Q8 retry's +0.2 resample actually take effect on this step.
    """
    out = _summary_map_model()(
        system=(
            "You are summarizing a slice of an Environmental Impact Statement.\n"
            "Output JSON with EXACTLY these five keys; for each key return BOTH a "
            "short text and a LIST of verbatim supporting quotes drawn from THIS CHUNK:\n"
            "{\n"
            '  "project_description":   {"text": "1-3 sentences", "quotes": ["...", "..."]},\n'
            '  "affected_community":    {"text": "...",            "quotes": ["..."]},\n'
            '  "alternatives_overview": {"text": "...",            "quotes": ["..."]},\n'
            '  "environmental_impact":  {"text": "...",            "quotes": ["..."]},\n'
            '  "public_response":       {"text": "...",            "quotes": ["..."]}\n'
            "}\n"
            "Rules:\n"
            "- If the chunk says nothing about a field, return text=\"\" and quotes=[].\n"
            "- Each entry in `quotes` MUST be copied character-for-character from the "
            "chunk. Pick short sentences or phrases (10-200 chars). A downstream verifier "
            "substring-matches each one against the document; quotes that aren't present "
            "verbatim are flagged for human review.\n"
            "- Provide ONE quote per substantive claim in `text`. If your text has 4 "
            "claims, return 4 quotes. Do not write a sentence in `text` that no quote "
            "supports.\n"
            "- Do not invent. Do not paraphrase inside the `quotes` field.\n\n"
            # MCAL_PLAN 3.14 (build item #4): plain-language + concreteness.
            "WRITING CONSTRAINTS:\n" + plain_language_clause()
        ),
        user=(
            f"Chunk #{chunk.index} (pages {chunk.start_page}-{chunk.end_page}"
            f"{', section: ' + chunk.label if chunk.label else ''}):\n\n{chunk.text}"
        ),
        max_tokens=SUMMARY_MAP_MAX_TOKENS,
    )
    return {
        "chunk_index": chunk.index,
        "start_page": chunk.start_page,
        "end_page": chunk.end_page,
        "ceq_chapter": chunk.ceq_chapter,
        "findings": out,
    }


def _summary_reduce(partials: list[dict], doc: Doc) -> dict:
    """
    Reduce chunk summaries into the final 6-field doc summary.

    The reducer is asked to:
      - synthesize 2-4 sentence text per subfield
      - return ONE verbatim quote PER SUBSTANTIVE CLAIM in that text (drawn
        from the per-chunk `quotes` lists). Earlier versions capped this at
        1-2 quotes per field, which left later sentences uncited.
      - write a 3-5 sentence overview that draws on the others, similarly
        cited claim-by-claim

    After the model responds, every quote is verified against `doc` and the
    verified page numbers are written into `evidence`.
    """
    payload = json.dumps([
        {
            "chunk_index": p["chunk_index"],
            "pages": f"{p['start_page']}-{p['end_page']}",
            "ceq_chapter": p["ceq_chapter"],
            "findings": p["findings"],
        }
        for p in partials
    ], ensure_ascii=False)

    out = opus(
        system=(
            "You consolidate per-chunk findings into one document-level summary.\n"
            "Output JSON with EXACTLY these keys:\n"
            "{\n"
            '  "overview":              {"text": "3-5 sentences", "quotes": ["..."]},\n'
            '  "project_description":   {"text": "2-4 sentences", "quotes": ["..."]},\n'
            '  "affected_community":    {"text": "...",           "quotes": ["..."]},\n'
            '  "alternatives_overview": {"text": "...",           "quotes": ["..."]},\n'
            '  "environmental_impact":  {"text": "...",           "quotes": ["..."]},\n'
            '  "public_response":       {"text": "...",           "quotes": ["..."], "based_on_main_doc_only": true}\n'
            "}\n"
            "Rules:\n"
            "- overview text encapsulates the other five — what the project is, where, "
            "the alternatives evaluated, the main environmental impacts, and the public "
            "response. Write so a non-expert can understand the document at a glance. "
            "Derive it from the other five fields; do not introduce new facts.\n"
            "- For EVERY key (including overview), provide ONE verbatim quote per "
            "substantive claim in your text. If your text has four claims, return four "
            "quotes. Do not write a sentence in `text` that no quote supports. Pick "
            "quotes from the per-chunk `quotes` arrays VERBATIM (do not edit, paraphrase, "
            "or join). Each quote you return must appear character-for-character "
            "somewhere in the input.\n"
            "- public_response is always limited to the main document; set the flag true.\n"
            "- If no chunks support a field, return text=\"\" and quotes=[].\n\n"
            # MCAL_PLAN 3.14 (build item #4): plain-language + concreteness.
            "WRITING CONSTRAINTS:\n" + plain_language_clause()
        ),
        user=f"Per-chunk findings:\n{payload}",
        max_tokens=SUMMARY_REDUCE_MAX_TOKENS,
    )

    # Normalize shape and verify quotes.
    final: dict = {}
    for key in SUMMARY_SCHEMA_KEYS:
        entry = out.get(key) or {}
        text = (entry.get("text") if isinstance(entry, dict) else entry) or ""
        quotes = entry.get("quotes") if isinstance(entry, dict) else []
        if not isinstance(quotes, list):
            quotes = []
        evidence_list = evidence_for_quotes([q for q in quotes if isinstance(q, str)], doc)
        final[key] = {"text": text, "evidence": evidence_list}
        if key == "public_response":
            final[key]["based_on_main_doc_only"] = bool(
                (entry.get("based_on_main_doc_only") if isinstance(entry, dict) else True)
                if entry else True
            )

    # Overview evidence: when none of the overview's OWN quotes verify, append up
    # to three verified quotes borrowed from the five subfields, so the overview
    # row in the grading sheet still points at real pages.
    #
    # Two things this deliberately does NOT do any more:
    #
    #   * It no longer REPLACES the overview's own evidence. The old code assigned
    #     over the list, so the unverified entries -- the ones carrying
    #     `quote_verified: False` and "Quote not found verbatim in doc" -- were
    #     discarded exactly when they were the most informative thing on the row.
    #     The failure was thereby converted into a clean-looking success.
    #   * It no longer presents borrowed quotes as first-party. Each carried entry
    #     is tagged `evidence_source="carried_from_subfields"` with the field it
    #     came from, and `mcal/confidence` excludes carried entries from `s_quote`
    #     (see `evidence.own_evidence`). Otherwise `s_quote` measured whether the
    #     SUBFIELDS were cited, and reported it as the overview's score -- which is
    #     why all 8 graded docs show `summary.overview` "ok" with
    #     `quote_verified: yes` for claims nothing ever checked.
    #
    # `settings.ATOMIC_VERIFY_FIELDS` already excludes overview for this exact
    # reason ("a roll-up whose evidence is carried forward"); the scoring path was
    # the one place that had not been told.
    overview_ev = final["overview"]["evidence"]
    if not any(ev.get("quote_verified") for ev in overview_ev):
        carried: list[Evidence] = []
        for sub in SUMMARY_SUBFIELDS:
            for ev in final[sub]["evidence"]:
                if not ev.get("quote_verified"):
                    continue
                key_id = (ev.get("quote", ""), tuple(ev.get("source_pages", []) or []))
                if any(
                    key_id
                    == (e2.get("quote", ""), tuple(e2.get("source_pages", []) or []))
                    for e2 in carried
                ):
                    continue
                carried.append(mark_carried(ev, f"summary.{sub}"))
                if len(carried) >= 3:
                    break
            if len(carried) >= 3:
                break
        if carried:
            final["overview"]["evidence"] = list(overview_ev) + carried
    return final


def extract_summary(chunks: list[Chunk], doc: Doc, max_chunks: int = 12, parallel: int = 4) -> dict:
    """Run Opus over chunks in parallel, then reduce. Caps chunk count to control cost.

    NOTE on `max_chunks`: this is a real recall ceiling, not just a cost knob. A
    1500-page doc yields ~31 chunks at CHUNK_PAGES=50, so only the first 12
    CEQ-tagged-then-document-order chunks are ever summarized. Content past that
    point cannot appear in the summary, which downstream shows up as an
    apparently-missing claim rather than as a truncation. Reported in
    `chunking_meta.chunks_truncated`, which `segment_b/gate.py` now reads and
    routes the summary-derived buckets to HUMAN_REVIEW on.
    """
    if not chunks:
        return {k: {"text": "", "evidence": []} for k in SUMMARY_SCHEMA_KEYS}
    return _summary_pipeline(chunks, doc, max_chunks, parallel)[0]


def partials_path(out_dir: Path, doc_id: str) -> Path:
    """Where the map stage's per-chunk findings are checkpointed."""
    return Path(out_dir) / f"{doc_id}.summary_partials.json"


def save_partials(out_dir: Path, doc_id: str, partials: list[dict]) -> Optional[Path]:
    """
    Checkpoint the map stage before the reduce runs. Best effort.

    The map calls are the most expensive artifact the pipeline produces -- up to
    12 Opus calls over 50 pages each -- and until now they existed only as a local
    variable and as text inside the reduce prompt. A `JSONDecodeError` or a throttle
    in `_summary_reduce` discarded all of them, which is not hypothetical: the
    comment on `SUMMARY_REDUCE_MAX_TOKENS` records that truncation hit 4 of 8
    documents on one run.

    Two things this buys beyond crash recovery:
      * the reduce can be re-run on its own, which is what makes the MCAL_PLAN 7 Q8
        retry affordable for `summary.*` -- 41 of the 59 RE_EXTRACT verdicts in the
        v1 batch are summary fields, and without this a retry means redoing the
        whole map stage;
      * the reduce output becomes auditable against what the map stage actually
        reported.

    A write failure is logged and swallowed: losing the checkpoint is bad, losing
    the document because the checkpoint could not be written is worse.
    """
    if not partials:
        return None
    try:
        path = partials_path(out_dir, doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(partials, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return path
    except Exception as e:  # noqa: BLE001 - a checkpoint must not lose a document
        log.warning(f"could not checkpoint summary partials for {doc_id}: {e}")
        return None


def load_partials(out_dir: Path, doc_id: str) -> Optional[list[dict]]:
    """The checkpointed map findings, or None if there is no usable checkpoint."""
    path = partials_path(out_dir, doc_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning(f"unreadable summary partials for {doc_id}: {e}")
        return None
    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        log.warning(f"summary partials for {doc_id} are not a list of objects")
        return None
    return data


def reduce_from_partials(partials: list[dict], doc: Doc) -> dict:
    """Re-run only the reduce step over checkpointed map findings."""
    return _summary_reduce(partials, doc)


def extract_summary_and_salience(
    chunks: list[Chunk],
    doc: Doc,
    max_chunks: int = 12,
    parallel: int = 4,
    partials_dir: Optional[Path] = None,
) -> tuple[dict, list[dict], dict]:
    """
    Standard summary + `summary_of_interest`, sharing one map pass.

    Returns `(summary, summary_of_interest, meta)`. The salience call reuses the
    already-computed per-chunk findings rather than re-reading the document, so
    its marginal cost is one Opus reduce call (MCAL_PLAN 3.15 "Cost").

    `partials_dir` checkpoints the map findings to disk before the reduce runs; see
    `save_partials`. Omitting it keeps the old in-memory-only behaviour.
    """
    if not chunks:
        return (
            {k: {"text": "", "evidence": []} for k in SUMMARY_SCHEMA_KEYS},
            [],
            {
                "n_chunks_summarized": 0,
                "n_chunks_available": 0,
                "chunks_truncated": False,
                "n_chunks_failed": 0,
                "failed_chunk_indices": [],
                "summary_source_incomplete": False,
            },
        )
    summary, partials, failed = _summary_pipeline(
        chunks, doc, max_chunks, parallel,
        partials_dir=partials_dir, doc_id=doc.doc_id,
    )
    soi = _summary_of_interest(partials, summary, doc)
    # `n_chunks_summarized` alone conflated two different gaps: "capped at
    # max_chunks by design" and "a map call died". Both mean the reduce ran on
    # less than the document, but they have different remedies (raise the cap /
    # retry the doc), so they are reported separately.
    #
    # `summary_source_incomplete` is the union, and is what the gate keys on: from
    # a reviewer's point of view a claim absent because page 700 was never read is
    # indistinguishable from a claim the document does not make, and that is the
    # thing that must not pass silently.
    truncated = len(chunks) > max_chunks
    meta = {
        "n_chunks_summarized": len(partials),
        "n_chunks_available": len(chunks),
        "chunks_truncated": truncated,
        "n_chunks_failed": len(failed),
        "failed_chunk_indices": sorted(failed),
        "summary_source_incomplete": bool(truncated or failed),
    }
    return summary, soi, meta


def _summary_pipeline(
    chunks: list[Chunk],
    doc: Doc,
    max_chunks: int,
    parallel: int,
    *,
    partials_dir: Optional[Path] = None,
    doc_id: Optional[str] = None,
) -> tuple[dict, list[dict], list[int]]:
    """
    Map over chunks in parallel, then reduce.

    Returns `(summary, partials, failed_chunk_indices)`. The third element used to
    be a local that only reached `log.error`, which is why
    `p1074_35556036854040` shows `n_chunks_summarized: 11` against a cap of 12 in
    the v1 batch with nothing in its artifacts recording that a call died.
    """
    # Prefer chunks tagged with a CEQ chapter; fill the rest in document order.
    tagged = [c for c in chunks if c.ceq_chapter]
    untagged = [c for c in chunks if not c.ceq_chapter]
    selected = (tagged + untagged)[:max_chunks]
    selected.sort(key=lambda c: c.index)

    partials: list[dict] = []
    failed: list[int] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_summary_map_one, c): c for c in selected}
        for fut in as_completed(futures):
            try:
                partials.append(fut.result())
            except Exception as e:
                idx = futures[fut].index
                failed.append(idx)
                log.warning(f"Summary map failed for chunk {idx}: {e}")
    partials.sort(key=lambda p: p["chunk_index"])
    if failed:
        # Logged AND returned. Logging alone left the artifacts clean: a document
        # summarized from 11 of 17 chunks looked identical to one summarized from
        # 12 of 17 by design.
        log.error(
            f"{len(failed)}/{len(selected)} summary map calls failed "
            f"(chunks {sorted(failed)}); the reduce step is running on partial "
            f"input and the summary may be missing supported claims."
        )
    # Checkpoint BEFORE the reduce: that is the whole point. The reduce is what
    # truncates and throttles, and everything expensive has already happened.
    if partials_dir is not None:
        save_partials(partials_dir, doc_id or doc.doc_id, partials)

    return _summary_reduce(partials, doc), partials, failed


# --- summary_of_interest (Opus, second reduce) -------------------------------
# MCAL_PLAN 3.15, build item #5. A salience-weighted summary emitted ALONGSIDE
# the standard one, never replacing it.

SALIENCE_CRITERIA = (
    "contested",
    "unusual_impact",
    "large_magnitude",
    "novel_alternative",
    "community_pushback",
    "precedent",
    "cross_jurisdictional",
)

SOI_MAX_CLAIMS = 6


def _summary_of_interest(partials: list[dict], summary: dict, doc: Doc) -> list[dict]:
    """
    Second reduce: what is notable about THIS document vs a typical EIS.

    Operates on the per-chunk findings already computed for the standard
    summary, plus the standard summary itself so the model can honour rule 3
    ("do not restate the standard summary").

    An empty list is a CORRECT result for a routine document (MCAL_PLAN 3.15
    rule 2) and is returned as `[]`, never as None. `gate.py` distinguishes a
    legitimate empty result from a generation failure via `extracted_value: []`
    vs `null`, so the two must not be conflated here.

    `page` is derived from `verify_and_locate`, not taken from the model. The
    model does not reliably see page numbers in the reduce payload (it gets
    per-chunk page RANGES), so a model-supplied page would fail verification for
    the wrong reason. This matches how every other M2 field resolves pages.
    """
    if not partials:
        return []

    payload = json.dumps(
        [
            {
                "chunk_index": p["chunk_index"],
                "pages": f"{p['start_page']}-{p['end_page']}",
                "ceq_chapter": p["ceq_chapter"],
                "findings": p["findings"],
            }
            for p in partials
        ],
        ensure_ascii=False,
    )
    standard = json.dumps(
        {k: (v.get("text") or "") for k, v in (summary or {}).items()},
        ensure_ascii=False,
    )

    try:
        out = opus(
            system=summary_of_interest_prompt(),
            user=(
                f"Per-chunk findings:\n{payload}\n\n"
                f"The STANDARD summary already produced for this document "
                f"(do not restate it):\n{standard}"
            ),
            max_tokens=SUMMARY_OF_INTEREST_MAX_TOKENS,
        )
    except Exception as e:
        # Distinguishable from a legitimate empty list by the caller: an
        # exception here means we could not determine salience at all.
        log.error(f"summary_of_interest reduce failed: {e}")
        raise

    raw = out.get("summary_of_interest") if isinstance(out, dict) else None
    if raw is None and isinstance(out, list):
        raw = out
    if not isinstance(raw, list):
        log.warning(
            f"summary_of_interest: expected a list, got {type(raw).__name__}; "
            "treating as empty"
        )
        return []

    entries: list[dict] = []
    for item in raw[:SOI_MAX_CLAIMS]:
        if not isinstance(item, dict):
            continue
        claim = (item.get("claim") or "").strip()
        quote = (item.get("evidence_quote") or "").strip()
        if not claim:
            continue

        criterion = (item.get("salience_criterion") or "").strip()
        if criterion not in SALIENCE_CRITERIA:
            # Keep the entry but mark the criterion invalid rather than dropping
            # it: an off-taxonomy criterion is itself signal that the rubric is
            # not discriminating, and MCAL_PLAN 6 tracks criterion distribution.
            log.warning(f"summary_of_interest: unknown criterion {criterion!r}")
            criterion_note = f"invalid_criterion:{criterion or 'missing'}"
            criterion = "unclassified"
        else:
            criterion_note = None

        ev = verify_and_locate(quote, doc)
        pages = ev.get("source_pages") or []
        entry = {
            "claim": claim,
            "salience_criterion": criterion,
            "page": int(pages[0]) if pages else None,
            "evidence_quote": quote,
            "why_notable": (item.get("why_notable") or "").strip(),
            "evidence": [ev],
        }
        if criterion_note:
            entry["note"] = criterion_note
        entries.append(entry)

    if len(raw) > SOI_MAX_CLAIMS:
        log.info(
            f"summary_of_interest: model returned {len(raw)} claims, capped at "
            f"{SOI_MAX_CLAIMS} per MCAL_PLAN 3.15 rule 4"
        )
    return entries


# --- Alternatives ------------------------------------------------------------

def extract_alternatives(doc: Doc, chapters: list[dict]) -> dict:
    chapter_text = text_for_ceq_chapter(doc, chapters, "Alternatives")
    if chapter_text is None:
        return {
            "value": [],
            "confidence": "low",
            "evidence": [],
            "note": "Alternatives chapter not detected structurally — skipped per v2 plan (no word-regex fallback).",
        }
    chunk_text, start_page, end_page = chapter_text
    # Cap to ~80 pages to stay within context (rough char cap on the chapter slice)
    cap = 80 * 2500
    if len(chunk_text) > cap:
        chunk_text = chunk_text[:cap]
        end_page = min(end_page, start_page + 79)

    out = sonnet(
        system=(
            "You list the alternatives evaluated in the Alternatives chapter of an EIS. "
            "Respond ONLY with JSON:\n"
            "{\n"
            '  "alternatives": [\n'
            '    {\n'
            '      "name": "<short name>",\n'
            '      "description": "<1-2 sentences, your own words>",\n'
            '      "quote": "<verbatim phrase from the excerpt that names/describes this alternative>"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Include the No Action alternative if present. The `quote` MUST be copied "
            "character-for-character from the excerpt; the downstream verifier will "
            "substring-match it against the source document."
        ),
        user=(
            f"Alternatives chapter (pages {start_page}-{end_page}):\n\n{chunk_text}"
        ),
        max_tokens=4000,
    )
    raw = out.get("alternatives") or []
    enriched: list[dict] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        quote = (a.get("quote") or "").strip()
        ev = verify_and_locate(quote, doc) if quote else {
            "quote": "", "source_pages": [], "quote_verified": False,
            "note": "No quote returned by extractor.",
        }
        enriched.append({
            "name": (a.get("name") or "").strip(),
            "description": (a.get("description") or "").strip(),
            "evidence": [ev],
        })
    return {
        "value": enriched,
        "confidence": "high" if enriched and any(e["evidence"][0].get("quote_verified") for e in enriched) else "low",
        "chapter_pages": f"{start_page}-{end_page}",
    }


# --- Themes ------------------------------------------------------------------

def extract_themes(summary: dict) -> dict:
    """Sonnet, given chunk summaries (here we use the reduced doc summary), assigns themes."""
    payload = json.dumps({
        k: (v.get("text") if isinstance(v, dict) else v)
        for k, v in summary.items()
    }, ensure_ascii=False)
    out = sonnet(
        system=(
            "You classify Environmental Impact Statements into a fixed taxonomy.\n"
            "Choose 1-3 themes and 2-5 subthemes from the taxonomy below.\n"
            "Respond ONLY with JSON:\n"
            "{\n"
            '  "themes": ["<theme>"],\n'
            '  "subthemes": ["<subtheme>"],\n'
            '  "justification": "<1-2 sentences citing which schema field supports the choice>",\n'
            '  "self_confidence": "high|medium|low"\n'
            "}\n\n"
            f"TAXONOMY:\n{json.dumps(THEMES, indent=2)}"
        ),
        user=f"Document summary fields:\n{payload}",
        max_tokens=600,
    )
    # Themes are derived from the summary and are never asked for quotes of their
    # own (see the JSON schema above: themes/subthemes/justification/
    # self_confidence). Carrying the summary's verified evidence forward gives the
    # grading sheet concrete pages to point at, but 100% of it is second-hand, so
    # every entry is tagged. `mcal/confidence` excludes carried evidence from
    # `s_quote`, which for this field means `s_quote` is not applicable rather than
    # a laundered 1.0 -- the extractor cannot be credited for citations it was
    # never asked to make, nor faulted for their absence.
    carried: list[Evidence] = []
    for key in SUMMARY_SUBFIELDS:
        sf = summary.get(key) or {}
        for ev in sf.get("evidence", []) or []:
            if not ev.get("quote_verified"):
                continue
            pages = tuple(ev.get("source_pages", []) or [])
            key_id = (ev.get("quote", ""), pages)
            if any(
                key_id == (e2.get("quote", ""), tuple(e2.get("source_pages", []) or []))
                for e2 in carried
            ):
                continue
            carried.append(mark_carried(ev, f"summary.{key}"))
            if len(carried) >= 3:
                break
        if len(carried) >= 3:
            break
    return {
        "value": {
            "themes": out.get("themes", []),
            "subthemes": out.get("subthemes", []),
        },
        "confidence": out.get("self_confidence", "medium"),
        "justification": out.get("justification", ""),
        "evidence": carried,
    }


# --- Location ----------------------------------------------------------------

def extract_location(doc: Doc, chapters: list[dict]) -> dict:
    pieces: list[tuple[str, int, int]] = []
    first_n = min(FIRST_30_PAGES, doc.n_pages)
    pieces.append((first_pages(doc, first_n), 1, first_n))
    for label in ("Project Area", "Study Area"):
        # Best-effort — these labels aren't standard CEQ but may have been
        # detected as a chapter heading by the regex-based detector.
        for ch in chapters:
            if label.lower() in (ch.get("label") or "").lower():
                seg = doc.full_text[ch["start_char"]:ch["end_char"]]
                pieces.append((seg[:60_000], ch["start_page"], ch["end_page"]))

    joined = "\n\n---\n\n".join(p[0] for p in pieces)

    out = sonnet(
        system=(
            "You extract the geographic location of an EIS project.\n"
            "Respond ONLY with JSON:\n"
            "{\n"
            '  "places": [\n'
            '    {\n'
            '      "name": "<place>",\n'
            '      "kind": "point|corridor|region",\n'
            '      "state": "<US state or null>",\n'
            '      "quote": "<verbatim phrase from the excerpts that establishes this place>"\n'
            "    }\n"
            "  ],\n"
            '  "is_multi_site": true|false,\n'
            '  "self_confidence": "high|medium|low",\n'
            '  "note": "<short>"\n'
            "}\n"
            "For corridors give endpoints in the name (\"Akron, OH to Cleveland, OH\"). "
            "The `quote` MUST be copied character-for-character from the excerpts."
        ),
        user=f"First 30 pages and any Project/Study Area excerpts:\n{joined}",
        max_tokens=800,
    )
    raw_places = out.get("places") or []
    enriched_places: list[dict] = []
    for p in raw_places:
        if not isinstance(p, dict):
            continue
        quote = (p.get("quote") or "").strip()
        ev = verify_and_locate(quote, doc) if quote else {
            "quote": "", "source_pages": [], "quote_verified": False,
            "note": "No quote returned by extractor.",
        }
        enriched_places.append({
            "name": (p.get("name") or "").strip(),
            "kind": (p.get("kind") or "").strip(),
            "state": p.get("state"),
            "evidence": [ev],
        })
    geocoded = _geocode_places(enriched_places)
    return {
        "value": {
            "places": enriched_places,
            "is_multi_site": out.get("is_multi_site", False),
            "geocoded": geocoded,
        },
        "confidence": out.get("self_confidence", "medium"),
        "note": out.get("note", ""),
    }


def _geocode_places(places: list[dict]) -> list[dict]:
    """Best-effort geocode via Nominatim. Skips silently if geopy missing."""
    try:
        from geopy.geocoders import Nominatim  # type: ignore
    except ImportError:
        return [{"name": p.get("name"), "lat": None, "lon": None, "note": "geopy not installed"} for p in places]
    geo = Nominatim(user_agent="eis_pipeline_segmentA")
    out: list[dict] = []
    for p in places:
        name = p.get("name")
        if not name:
            continue
        try:
            time.sleep(1.1)  # Nominatim rate limit
            r = geo.geocode(name, timeout=10)
            if r:
                out.append({"name": name, "lat": round(r.latitude, 6), "lon": round(r.longitude, 6), "address": r.address})
            else:
                out.append({"name": name, "lat": None, "lon": None})
        except Exception as e:
            out.append({"name": name, "lat": None, "lon": None, "error": str(e)})
    return out


# --- Key People / Groups -----------------------------------------------------

def extract_key_people(doc: Doc, chapters: list[dict]) -> dict:
    """
    Three categories per the v2 plan:
      - agency_preparers       : from Preparers/Consultation chapter
      - cooperating_agencies   : from Consultation chapter
      - public_commenters      : only when main doc has comment-response content

    Every entry carries a verbatim quote + verified page (via evidence helpers).
    """
    text = doc.full_text
    consultation = text_for_ceq_chapter(doc, chapters, "Consultation")
    preparers_text = consultation[0] if consultation else first_pages(doc, FIRST_30_PAGES)

    preparers_out = sonnet(
        system=(
            "You list (a) agency staff who prepared this EIS and (b) cooperating "
            "agencies / tribal nations consulted. Respond ONLY with JSON:\n"
            "{\n"
            '  "agency_preparers": [\n'
            '    {\n'
            '      "name": "<full name>",\n'
            '      "role": "<role/title>",\n'
            '      "quote": "<verbatim phrase from the excerpt naming this person>"\n'
            '    }\n'
            "  ],\n"
            '  "cooperating_agencies": [\n'
            '    {\n'
            '      "name": "<agency or nation>",\n'
            '      "quote": "<verbatim phrase from the excerpt naming this agency>"\n'
            '    }\n'
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Do NOT include private individuals.\n"
            "- Do NOT attribute stances.\n"
            "- Quotes MUST be copied character-for-character from the excerpt; the "
            "downstream verifier will substring-match them against the document."
        ),
        user=f"Consultation/Preparers excerpt:\n{preparers_text[:60_000]}",
        max_tokens=2000,
    )

    preparers = _enrich_named_list(preparers_out.get("agency_preparers", []), doc)
    cooperating = _enrich_named_list(preparers_out.get("cooperating_agencies", []), doc)

    # Public commenters: detect comment-response content first
    has_comment_response = bool(re.search(
        r"\b(comment(?:s)?\s+and\s+response|response\s+to\s+comments)\b",
        text, re.IGNORECASE,
    ))

    commenters_block: list[dict] = []
    if has_comment_response:
        m = re.search(r"\b(comment(?:s)?\s+and\s+response|response\s+to\s+comments)\b", text, re.IGNORECASE)
        start = m.start() if m else 0
        excerpt = text[start : start + 60_000]
        excerpt_start_page = doc.page_at_offset(start)
        excerpt_end_page = doc.page_at_offset(start + len(excerpt))
        out = sonnet(
            system=(
                "You list public commenters with attributed stances from a comments-and-response "
                "section. Respond ONLY with JSON:\n"
                "{\n"
                '  "commenters": [\n'
                '    {\n'
                '      "name": "<last name only for private individuals, full name for officials/organizations>",\n'
                '      "kind": "private|organization|official|tribal",\n'
                '      "stance": "support|oppose|conditional|neutral",\n'
                '      "quote": "<verbatim quote attributed to this commenter>"\n'
                "    }\n"
                "  ]\n"
                "}\n"
                "Rules: only include commenters whose stance is CLEARLY attributed. Use last name only "
                "for private individuals (or 'private commenter'). Quotes MUST be verbatim from the excerpt."
            ),
            user=(
                f"Comment-response excerpt (pages {excerpt_start_page}-{excerpt_end_page}):\n\n{excerpt}"
            ),
            max_tokens=2000,
        )
        raw_commenters = out.get("commenters") or []
        for c in raw_commenters:
            if not isinstance(c, dict):
                continue
            quote = (c.get("quote") or "").strip()
            ev = verify_and_locate(quote, doc) if quote else {
                "quote": "", "source_pages": [], "quote_verified": False,
                "note": "No quote returned by extractor.",
            }
            commenters_block.append({
                "name": (c.get("name") or "").strip(),
                "kind": (c.get("kind") or "").strip(),
                "stance": (c.get("stance") or "").strip(),
                "evidence": [ev],
            })

    return {
        "value": {
            "agency_preparers": preparers,
            "cooperating_agencies": cooperating,
            "public_commenters": commenters_block,
            "comment_response_present": has_comment_response,
        },
        "confidence": "high" if preparers else "medium",
    }


def _enrich_named_list(items: list[dict], doc: Doc) -> list[dict]:
    """For each {name, role?, quote} item, verify the quote and attach evidence."""
    out: list[dict] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        quote = (item.get("quote") or "").strip()
        ev = verify_and_locate(quote, doc) if quote else {
            "quote": "", "source_pages": [], "quote_verified": False,
            "note": "No quote returned by extractor.",
        }
        cleaned = {k: v for k, v in item.items() if k != "quote"}
        cleaned["evidence"] = [ev]
        out.append(cleaned)
    return out


# --- Top-level ---------------------------------------------------------------

def run_m2(
    doc: Doc,
    chunked: Optional[dict] = None,
    partials_dir: Optional[Path] = None,
) -> dict:
    """
    Run all M2 extractors. `chunked` is the output of chunks_for_doc; computed if
    omitted.

    `partials_dir` checkpoints the summary map stage to disk before the reduce runs
    (see `save_partials`), which is what lets a failed reduce be resumed and what
    makes a `summary.*` re-extraction cost one Opus call instead of thirteen.
    """
    chunked = chunked or chunks_for_doc(doc)
    chunks: list[Chunk] = chunked["chunks"]
    chapters: list[dict] = chunked["chapters"]

    log.info(f"M2: {len(chunks)} chunks, {len(chapters)} CEQ-mapped chapters detected")

    # summary_of_interest reuses this call's per-chunk findings (MCAL_PLAN 3.15).
    summary, summary_of_interest, summary_meta = extract_summary_and_salience(
        chunks, doc, partials_dir=partials_dir
    )
    alternatives = extract_alternatives(doc, chapters)
    themes = extract_themes(summary)
    location = extract_location(doc, chapters)
    key_people = extract_key_people(doc, chapters)

    return {
        "summary": summary,
        # Stamped into every M2 artifact so `_prompt_version.txt` can be
        # verified PER FILE rather than by "was it regenerated in this
        # invocation". The multi-round protocol (MCAL_PLAN 7.5) reruns subsets,
        # so an invocation-scoped check refuses to stamp a set that is in fact
        # consistent -- which is exactly what happened on the first rerun.
        "_prompt_version": PROMPT_VERSION,
        # Always emitted, including when empty. An empty list means "this
        # document is routine", which is a substantive result and must stay
        # distinguishable from a generation failure (MCAL_PLAN 3.12).
        "summary_of_interest": summary_of_interest,
        "alternatives": alternatives,
        "themes": themes,
        "location": location,
        "key_people": key_people,
        "chunking_meta": {
            "n_chunks": len(chunks),
            "n_chapters_detected": len(chapters),
            "chapters": [
                {"label": c["label"], "ceq_chapter": c["ceq_chapter"],
                 "start_page": c["start_page"],
                 "end_page": c["end_page"]}
                for c in chapters
            ],
            "chunk_size_pages": CHUNK_PAGES,
            **summary_meta,
        },
    }
