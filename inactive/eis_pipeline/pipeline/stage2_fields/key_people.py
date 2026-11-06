"""
Stage 2.5 — Key people/groups.

Pipeline:
  1. Optional Haiku gap-fill on stakeholder-dense chunks (adds to record.ner)
  2. Frequency + rule filter (lighter on entities with a `dict_*` source)
  3. Score by frequency × chunk-spread, cap at MAX_KEY_PEOPLE
  4. Haiku triage — only scrubs entities sourced from spaCy. Dict and
     gap-fill sources bypass (they're already curated by construction).
  5. Per-entity Opus call: stance + role + opinion summary + evidence
  6. Per-entity Opus call: emotive/representative quote (with substring check)
"""

from __future__ import annotations

import logging
import re
import textwrap
from collections import Counter
from typing import TYPE_CHECKING

from ..config import (
    MAX_KEY_PEOPLE,
    MIN_ENTITY_FREQUENCY,
    MODELS,
    NER_GAPFILL_MAX_CALLS,
    NER_GAPFILL_MAX_EXISTING_ENTITIES,
    NER_GAPFILL_MIN_WORDS,
    QUOTE_MAX_WORDS,
    QUOTE_MIN_WORDS,
)
from ..schema import EISRecord, EvidencePointer, KeyPersonOrGroup, QuoteItem
from .retrieval import combine_chunk_context, get_chunks_mentioning

if TYPE_CHECKING:
    from ..llm_client import LLMClient

logger = logging.getLogger(__name__)

_BOILERPLATE_PHRASES = frozenset({
    "sincerely", "respectfully", "dear sir", "dear madam", "to whom it may concern",
    "enclosure", "attachment", "cc:", "bcc:",
})

# Sources that bypass triage / rule filter — already vetted as real stakeholders.
_TRUSTED_SOURCES = {"dict_agency", "dict_tribe", "dict_ngo", "haiku_gapfill"}


def run(record: EISRecord, client: "LLMClient") -> None:
    """Populate record.key_people_and_groups. Mutates record in place."""
    # ---- Step 1: Haiku gap-fill on stakeholder-dense chunks ----
    _gap_fill_ner(record, client)

    all_entities = record.ner.people + record.ner.organizations
    if not all_entities:
        logger.info("No NER entities for %s", record.doc_id)
        return

    sources = record.ner.sources

    # ---- Step 2: frequency filter (dict/gap-fill bypass min-frequency) ----
    full_text_lower = " ".join(c.text for c in record.chunks if c.used).lower()
    freq: Counter = Counter()
    for name in all_entities:
        count = full_text_lower.count(name.lower())
        is_trusted = sources.get(name) in _TRUSTED_SOURCES
        if count >= MIN_ENTITY_FREQUENCY or is_trusted:
            freq[name] = max(count, 1)

    # ---- Step 3: rule filter (skipped for trusted sources) ----
    filtered = {
        name: count for name, count in freq.items()
        if sources.get(name) in _TRUSTED_SOURCES or _passes_rule_filter(name)
    }

    chunk_spread: dict[str, int] = {}
    for name in filtered:
        name_lower = name.lower()
        spread = sum(1 for c in record.chunks if c.used and name_lower in c.text.lower())
        chunk_spread[name] = spread

    # Score = frequency × spread; cap at MAX_KEY_PEOPLE
    scored = sorted(
        filtered.keys(),
        key=lambda n: freq[n] * chunk_spread.get(n, 1),
        reverse=True,
    )
    top_entities = scored[:MAX_KEY_PEOPLE]

    if not top_entities:
        return

    # ---- Step 4: triage (spaCy-sourced only) ----
    to_triage = [n for n in top_entities if sources.get(n) not in _TRUSTED_SOURCES]
    trusted = [n for n in top_entities if sources.get(n) in _TRUSTED_SOURCES]
    if to_triage:
        triaged = _triage_entities(to_triage, record, client)
    else:
        triaged = []
    # Preserve original score-order of the full top_entities list, keeping only
    # those that were either trusted or survived triage.
    keep = set(triaged) | set(trusted)
    top_entities = [n for n in top_entities if n in keep]

    # ---- Step 5: per-entity stance + role + opinion + quote (one Opus call) ----
    results: list[KeyPersonOrGroup] = []
    for name in top_entities:
        chunks_for_entity = get_chunks_mentioning(record.chunks, name)
        if not chunks_for_entity:
            continue

        entity_type = _guess_type(name, record.ner)
        first_chunk_id = _first_appearance_chunk(name, record.chunks)

        pack = _extract_entity_pack(
            name, entity_type, chunks_for_entity, record, client,
        )

        results.append(KeyPersonOrGroup(
            name=name,
            type=entity_type,
            role=pack.get("role"),
            opinion_summary=pack.get("opinion_summary"),
            first_appearance_chunk=first_chunk_id,
            appearance_order=None,  # filled in after we sort by document order
            stance=pack.get("stance", "insufficient_information"),
            stance_evidence=pack.get("evidence", []),
            quote=pack.get("quote"),
        ))

    # Sort by document appearance order (first_appearance_chunk position),
    # then number sequentially. Entities with no first_appearance_chunk go last.
    chunk_index = {c.chunk_id: i for i, c in enumerate(record.chunks)}
    results.sort(
        key=lambda r: chunk_index.get(r.first_appearance_chunk or "", len(record.chunks)),
    )
    for i, entity in enumerate(results, start=1):
        entity.appearance_order = i

    record.key_people_and_groups = results
    logger.info("Key people/groups: %d extracted for %s", len(results), record.doc_id)


# ---------------------------------------------------------------------------
# Haiku gap-fill
# ---------------------------------------------------------------------------

_GAPFILL_SYSTEM = textwrap.dedent("""\
    You are extracting stakeholder names from a chunk of a U.S. Environmental
    Impact Statement.

    Return only entities that play a substantive role in this excerpt — agencies,
    companies, advocacy groups, tribes, named officials, community organizations.

    Exclude:
      - Citation authors ("Smith et al., 1974")
      - Generic titles with no name ("the Administrator")
      - Names that only appear as a form-letter signature
      - Place names, road names, brand names that aren't organizations

    Respond with ONLY valid JSON:
    {
      "people":        ["First Last", ...],   // must be full names
      "organizations": ["Group Name", ...]
    }
""")


def _gap_fill_ner(record: EISRecord, client: "LLMClient") -> None:
    """
    Run Haiku on stakeholder-dense chunks to catch entities spaCy + dictionaries missed.

    Triggers:
      - Unconditional: chunks tagged `comments_and_responses`
      - Conditional:   chunks where existing entity count < NER_GAPFILL_MAX_EXISTING_ENTITIES
                       AND chunk has > NER_GAPFILL_MIN_WORDS words
    Total calls capped at NER_GAPFILL_MAX_CALLS.
    """
    existing_names = set(n.lower() for n in record.ner.people + record.ner.organizations)
    candidate_chunks = []

    for chunk in record.chunks:
        if not chunk.used:
            continue
        is_comments = "comments_and_responses" in chunk.topic_tags
        word_count = len(chunk.text.split())

        existing_in_chunk = sum(
            1 for n in existing_names if n in chunk.text.lower()
        )
        is_sparse = (
            word_count >= NER_GAPFILL_MIN_WORDS
            and existing_in_chunk < NER_GAPFILL_MAX_EXISTING_ENTITIES
        )

        if is_comments or is_sparse:
            # Comments chunks get priority (sorted first)
            priority = 0 if is_comments else 1
            candidate_chunks.append((priority, chunk))

    candidate_chunks.sort(key=lambda x: (x[0], x[1].chunk_id))
    selected = [c for _, c in candidate_chunks[:NER_GAPFILL_MAX_CALLS]]

    if not selected:
        return

    logger.info(
        "NER gap-fill: scanning %d chunks for %s (max=%d)",
        len(selected), record.doc_id, NER_GAPFILL_MAX_CALLS,
    )

    added_people: list[str] = []
    added_orgs: list[str] = []
    for chunk in selected:
        try:
            result = client.call_json(
                model=MODELS["light"],
                system=_GAPFILL_SYSTEM,
                messages=[{"role": "user", "content": chunk.text[:32_000]}],
                max_tokens=512,
                temperature=0.1,
                label=f"ner_gapfill/{record.doc_id}/{chunk.chunk_id}",
            )
        except Exception as exc:
            logger.warning("Gap-fill failed on chunk %s: %s", chunk.chunk_id, exc)
            continue

        for p in result.get("people") or []:
            if not isinstance(p, str) or not p.strip():
                continue
            key = p.lower().strip()
            if key not in existing_names:
                existing_names.add(key)
                added_people.append(p.strip())
                record.ner.sources[p.strip()] = "haiku_gapfill"

        for o in result.get("organizations") or []:
            if not isinstance(o, str) or not o.strip():
                continue
            key = o.lower().strip()
            if key not in existing_names:
                existing_names.add(key)
                added_orgs.append(o.strip())
                record.ner.sources[o.strip()] = "haiku_gapfill"

    record.ner.people.extend(added_people)
    record.ner.organizations.extend(added_orgs)
    record.ner.deduped_count = len(record.ner.people) + len(record.ner.organizations)
    if added_people or added_orgs:
        logger.info(
            "NER gap-fill added: %d people + %d orgs to %s",
            len(added_people), len(added_orgs), record.doc_id,
        )


# ---------------------------------------------------------------------------
# Rule-based filter (for spaCy-sourced entities only)
# ---------------------------------------------------------------------------

def _passes_rule_filter(name: str) -> bool:
    name_stripped = name.strip()
    name_lower = name_stripped.lower()

    if len(name_stripped) < 3:
        return False

    # Single-token name: keep only if it looks like an acronym (org)
    if len(name_stripped.split()) == 1 and name_stripped[0].isupper():
        if not re.match(r"^[A-Z]{2,6}$", name_stripped):
            return False

    if any(bp in name_lower for bp in _BOILERPLATE_PHRASES):
        return False

    if re.match(r"^[\d\s\-/,\.]+$", name_stripped):
        return False

    return True


# ---------------------------------------------------------------------------
# Haiku triage (scrubs spaCy-sourced entities only)
# ---------------------------------------------------------------------------

_TRIAGE_SYSTEM = textwrap.dedent("""\
    You are reviewing a list of named entities extracted from a U.S. Environmental Impact Statement.
    Some are genuine stakeholders (agencies, companies, advocacy groups, named officials).
    Others are noise (citation authors, form-letter signatories, generic job titles, etc.).

    Return ONLY a JSON array of the names from the input list that are GENUINE stakeholders.
    Exclude: people who appear only as letter signatories, citation authors, or in generic titles.
    Include: agencies, companies, advocacy organizations, named officials with substantive roles,
             community groups, named individuals taking a documented stance.

    Respond with ONLY a valid JSON array: ["Name 1", "Name 2", ...]
""")


def _triage_entities(entities: list[str], record: EISRecord, client: "LLMClient") -> list[str]:
    if not entities:
        return entities

    user_msg = (
        f"Document title: {record.title or record.doc_id}\n\n"
        f"Entity list:\n" + "\n".join(f"- {e}" for e in entities)
    )

    try:
        result = client.call_json(
            model=MODELS["light"],
            system=_TRIAGE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=1024,
            temperature=0.1,
            label=f"key_people_triage/{record.doc_id}",
        )
        if isinstance(result, list):
            kept = set(result)
            return [e for e in entities if e in kept]
    except Exception as exc:
        logger.warning("Entity triage failed for %s: %s", record.doc_id, exc)

    return entities


# ---------------------------------------------------------------------------
# Stance + role + opinion summary (one Opus call per entity)
# ---------------------------------------------------------------------------

_ENTITY_PERSON_SYSTEM = textwrap.dedent("""\
    You are analyzing how a named person is presented in a U.S. Environmental
    Impact Statement.

    Based ONLY on the language attributed to or about this person in the
    provided document text, return ALL of the following in a single response:

      - "role": their title/position/affiliation (e.g. "EPA Acting Director",
                "spokesperson for Cook County Forest Preserve District"). Use
                null if not stated.
      - "opinion_summary": 1–2 sentences in plain English describing what this
                person says or thinks about the project. Use null if no
                substantive opinion appears in the text.
      - "stance": one of "supportive", "opposed", "mixed", "neutral",
                  or "insufficient_information"
      - "evidence": chunk IDs and pages supporting the role/opinion/stance.
      - "quote": a VERBATIM quote of {quote_min}–{quote_max} words attributed
                 to or directly about this person, that is emotionally charged
                 or clearly encapsulates their stance. Skip hollow boilerplate
                 ("we sincerely hope you will consider").
                 The text MUST be an exact substring of the provided document
                 excerpts — copy character-for-character, no paraphrasing.
                 If no qualifying verbatim quote exists, set "quote" to null.
      - "quote_chunk_id" and "quote_page": where the quote was found (or null).

    Respond with ONLY valid JSON:
    {{
      "role": "...",
      "opinion_summary": "...",
      "stance": "...",
      "evidence": [{{"chunk_id": "c03", "pages": [88]}}],
      "quote": "exact verbatim text" or null,
      "quote_chunk_id": "c03" or null,
      "quote_page": 88 or null
    }}
""")

_ENTITY_GROUP_SYSTEM = textwrap.dedent("""\
    You are analyzing how a named group, agency, or organization is presented
    in a U.S. Environmental Impact Statement.

    Based ONLY on the language attributed to or about this group in the
    provided document text, return ALL of the following in a single response:

      - "role": their role in the project (e.g. "lead agency", "cooperating
                agency", "objecting party", "consulted agency", "contractor",
                "advocacy group submitting comments"). Use null if not clear.
      - "opinion_summary": 1–2 sentences in plain English describing the
                group's position on the project. Use null if there is no
                substantive position.
      - "stance": one of "supportive", "opposed", "mixed", "neutral",
                  or "insufficient_information"
      - "evidence": chunk IDs and pages supporting the role/opinion/stance.
      - "quote": a VERBATIM quote of {quote_min}–{quote_max} words attributed
                 to or directly about this group, that is emotionally charged
                 or clearly encapsulates their stance. Skip hollow boilerplate.
                 The text MUST be an exact substring of the provided document
                 excerpts. If none qualifies, set "quote" to null.
      - "quote_chunk_id" and "quote_page": where the quote was found (or null).

    Respond with ONLY valid JSON:
    {{
      "role": "...",
      "opinion_summary": "...",
      "stance": "...",
      "evidence": [{{"chunk_id": "c03", "pages": [88]}}],
      "quote": "exact verbatim text" or null,
      "quote_chunk_id": "c03" or null,
      "quote_page": 88 or null
    }}
""")


def _extract_entity_pack(
    name: str,
    entity_type: str,
    chunks: list,
    record: EISRecord,
    client: "LLMClient",
) -> dict:
    """
    One Opus call returns stance + role + opinion summary + quote candidate +
    evidence. The quote is substring-checked in-process; if it fails, we null
    it (no retry call). This is the consolidated replacement for the previous
    two-call pattern (_extract_stance_and_role + _extract_quote).
    """
    context = combine_chunk_context(chunks, max_chars=20_000)
    user_msg = f"Entity: {name}\n\nDocument excerpts:\n{context}"
    system_tmpl = _ENTITY_PERSON_SYSTEM if entity_type == "person" else _ENTITY_GROUP_SYSTEM
    system = system_tmpl.format(quote_min=QUOTE_MIN_WORDS, quote_max=QUOTE_MAX_WORDS)

    valid_stances = {"supportive", "opposed", "mixed", "neutral", "insufficient_information"}

    try:
        result = client.call_json(
            model=MODELS["heavy"],
            system=system,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=768,
            temperature=0.1,
            label=f"entity_pack/{record.doc_id}/{name[:30]}",
        )
    except Exception as exc:
        logger.warning("Entity extraction failed for %s/%s: %s", record.doc_id, name, exc)
        return {
            "stance": "insufficient_information",
            "role": None,
            "opinion_summary": None,
            "evidence": [],
            "quote": None,
        }

    stance = result.get("stance", "insufficient_information")
    if stance not in valid_stances:
        stance = "insufficient_information"

    role = result.get("role")
    role = role.strip() if isinstance(role, str) and role.strip() else None

    opinion = result.get("opinion_summary")
    opinion = opinion.strip() if isinstance(opinion, str) and opinion.strip() else None

    raw_evidence = result.get("evidence") or []
    evidence = [
        EvidencePointer(chunk_id=e["chunk_id"], pages=e.get("pages", []))
        for e in raw_evidence
        if isinstance(e, dict) and "chunk_id" in e
    ]

    # Quote: substring-check in-process. No retry — failure means null.
    quote = _verify_quote(result, chunks, name, opinion)

    return {
        "stance": stance,
        "role": role,
        "opinion_summary": opinion,
        "evidence": evidence,
        "quote": quote,
    }


def _verify_quote(
    result: dict,
    chunks: list,
    name: str,
    opinion_summary: str | None,
) -> QuoteItem | None:
    """
    Validate the quote returned in the consolidated Opus response.
    Returns None unless:
      - quote text is non-empty
      - opinion_summary was populated (no substantive opinion → no real quote)
      - word count is in [QUOTE_MIN_WORDS, QUOTE_MAX_WORDS]
      - text appears as a verbatim substring of the cited chunks
    """
    if not opinion_summary:
        return None

    quote_text = result.get("quote")
    if not quote_text or not isinstance(quote_text, str):
        return None
    quote_text = quote_text.strip()
    if not quote_text:
        return None

    word_count = len(quote_text.split())
    if word_count < QUOTE_MIN_WORDS or word_count > QUOTE_MAX_WORDS:
        logger.debug("Quote word count out of range for %s: %d words", name, word_count)
        return None

    full_context = " ".join(c.text for c in chunks)
    if quote_text not in full_context:
        logger.debug("Quote substring check failed for %s — nulling", name)
        return None

    chunk_id = result.get("quote_chunk_id") or ""
    page = result.get("quote_page") or 0
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 0

    return QuoteItem(
        text=quote_text,
        chunk_id=chunk_id,
        page=page,
        substring_verified=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guess_type(name: str, ner_result) -> str:
    if name in ner_result.organizations:
        return "organization"
    if name in ner_result.people:
        return "person"
    return "unknown"


def _first_appearance_chunk(name: str, chunks: list) -> str | None:
    name_lower = name.lower()
    for chunk in chunks:
        if chunk.used and name_lower in chunk.text.lower():
            return chunk.chunk_id
    return None
