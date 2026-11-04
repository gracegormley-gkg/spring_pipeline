"""
Stage 3f — Stakeholders.

Per synthesis_plan §Stakeholders. The most subtle module in the pipeline; this
is the v1 implementation, scoped to organizations only (persons excluded per
PII v1 policy).

Pipeline:
  1. Locate public_comments / response_to_comments sections. If neither
     present, set stakeholder_status="no_comment_section_found" and return
     stakeholders=[].
  2. Block detection (PRIMARY = structural parser):
     - First pass: regex for COMMENT-N markers ("COMMENT 42:", "COMMENT NO. 42:",
       "Comment #42").
     - Fallback: split the comments section into ~3000-char "letter-shaped"
       blocks at blank-line boundaries. Less precise but never goes silent.
  3. Per block, comment/response split: find "Response:" / "Agency Response:"
     markers; everything before -> comment_text, after -> agency_response_text.
     Quote selection ONLY reads comment_text.
  4. Per block, sentence-split comment_text into spans. Each span gets a
     stable span_id like "block_42_s3" + char_offset_raw + page.
  5. Author extraction (Haiku): identify org name; SKIP block if person.
  6. Span-ID quote selection (Sonnet): cue-rank spans by stance markers,
     send top-3 candidates, Sonnet returns a span_id, code copies the raw
     text from the alignment-mapped span (model never types the quote).
  7. Two-pass stance:
     - Haiku classifies stance + stance_target + confidence
     - If confidence < 0.7 OR stance == "mixed": Sonnet re-classifies
  8. Build Stakeholder + StanceRecord with the linked Quote.

Cost shape per block: 1 Haiku (author) + 1 Sonnet (quote select) + 1 Haiku
(stance) + maybe 1 Sonnet (stance retry). ~$0.005 per block typical. For 20
blocks that's $0.10 — well under per-doc budget.

What's deliberately deferred (synthesis_plan §Open risks 4):
  - Letterhead-only blocks without COMMENT markers (simple chunking handles
    these but author extraction is less reliable).
  - Hearing transcripts with speaker labels — handled by chunking heuristic.
  - Form-letter group headers — collapsed into single block.
  - Multi-target stance attribution on long mixed-stance commenters
    (NLI-with-targets is v2 per synthesis_plan).
  - Stance.reference_id resolution to alternative_id (v2).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import config
from .ingest import IngestArtifact, char_to_page
from .schema import (
    CommentAuthor,
    CommentBlock,
    EISRecord,
    Quote,
    SpanRecord,
    Stakeholder,
    StanceRecord,
    StanceTarget,
)

if TYPE_CHECKING:
    from .llm_client import LLMClient
    from .sections import SectionsArtifact

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Numbered comment headers: "COMMENT 42:", "COMMENT NO. 42:", "Comment #42"
_COMMENT_HEADER_RE = re.compile(
    r"^[ \t]*(?:COMMENT|Comment)\s*(?:NO\.?\s*|#)?\s*(\d+)[\s:.-]",
    re.MULTILINE,
)

# Agency-response markers within a block
_RESPONSE_HEADER_RE = re.compile(
    r"^[ \t]*(?:RESPONSE|Response|Agency\s+Response)\s*[:.-]?\s*$",
    re.MULTILINE,
)

# Sentence splitter — keeps reasonable boundaries for OCR'd text. Splits on
# . ! ? followed by whitespace + capital, plus on hard newlines if the
# previous line ends with sentence-terminal punctuation.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
)

# Stance markers used to cue-rank spans before sending to Sonnet.
_STANCE_CUES = [
    "oppose", "opposed", "object to", "object", "concern", "concerns",
    "support", "supports", "endorse", "endorses", "favor", "in favor of",
    "we cannot", "we urge", "we recommend", "we request", "we ask",
    "strongly oppose", "strongly support", "encourage",
    "disagree", "agree", "should not", "should", "must not", "must",
    "in opposition", "in support",
]

# Cap how many candidate spans we send to Sonnet for quote selection.
_MAX_QUOTE_CANDIDATES = 3

# Char count cap on each span (keeps quote selection prompts tight).
_MAX_SPAN_CHARS = 400

# Skip very short spans during candidate selection (under this many chars).
_MIN_SPAN_CHARS = 30

# Two-pass stance threshold: if Haiku confidence is below this OR stance
# is "mixed", retry with Sonnet.
_STANCE_RETRY_THRESHOLD = 0.7

# Fallback block sizing when no COMMENT-N markers exist
_FALLBACK_BLOCK_TARGET_CHARS = 3000

# Cap total blocks we process per doc (cost guard)
_MAX_BLOCKS = 30


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run(
    record: EISRecord,
    ingest: IngestArtifact,
    sections_artifact: "SectionsArtifact",
    llm: "LLMClient | None" = None,
) -> list[str]:
    """Mutate `record.comment_blocks` and `record.stakeholders`.
    Sets `record.stakeholder_status`. Returns warnings."""
    warnings: list[str] = []

    by_name = sections_artifact.by_name() if hasattr(sections_artifact, "by_name") else {}
    pc = by_name.get("public_comments")
    rc = by_name.get("response_to_comments")
    if (pc is None or pc.char_span is None or pc.status != "ok") and \
       (rc is None or rc.char_span is None or rc.status != "ok"):
        record.comment_blocks = []
        record.stakeholders = []
        record.stakeholder_status = "no_comment_section_found"
        warnings.append("stakeholders: no public_comments/response_to_comments section detected")
        return warnings

    # Use whichever section exists; if both, span them together.
    spans: list[tuple[int, int]] = []
    if pc and pc.char_span:
        spans.append(pc.char_span)
    if rc and rc.char_span:
        spans.append(rc.char_span)
    section_start = min(s for s, _ in spans)
    section_end = max(e for _, e in spans)
    section_text = ingest.raw_text[section_start:section_end]

    # 2. Block detection
    blocks_raw = _detect_blocks(section_text, section_start)
    if not blocks_raw:
        record.comment_blocks = []
        record.stakeholders = []
        record.stakeholder_status = "needs_review"
        warnings.append("stakeholders: comments section present but no blocks detected")
        return warnings

    if llm is None:
        # Build CommentBlocks but no stakeholders (need LLM for author/quote/stance)
        record.comment_blocks = [
            _build_comment_block_no_split(b, ingest)
            for b in blocks_raw[:_MAX_BLOCKS]
        ]
        record.stakeholders = []
        record.stakeholder_status = "skipped_no_llm"
        warnings.append("stakeholders: blocks detected but no llm provided for author/quote/stance")
        return warnings

    comment_blocks: list[CommentBlock] = []
    stakeholders: list[Stakeholder] = []
    appearance_order = 1

    for raw_block in blocks_raw[:_MAX_BLOCKS]:
        block_id = raw_block["block_id"]
        block_start = raw_block["start"]
        block_end = raw_block["end"]
        block_text = ingest.raw_text[block_start:block_end]

        # 3. Comment/response split
        comment_text_span, response_span, split_status = _split_comment_response(
            block_text, block_start
        )

        # 4. Sentence-split comment_text into spans
        comment_text_local: tuple[int, int] | None = None
        if comment_text_span is not None:
            comment_text_local = (
                comment_text_span[0] - block_start,
                comment_text_span[1] - block_start,
            )
        spans = _build_spans(
            block_id, block_text, comment_text_local, response_span, block_start, ingest
        )

        # 5. Author extraction (and org/person filter)
        author_info = _extract_author(block_text[:1500], llm)
        if author_info is None or author_info.get("is_individual"):
            # Still emit the CommentBlock (downstream may want it for
            # provenance / coverage), but don't emit a Stakeholder.
            comment_blocks.append(CommentBlock(
                block_id=block_id,
                char_span_raw=(block_start, block_end),
                pages=_block_pages(block_start, block_end, ingest),
                comment_text_span=comment_text_span,
                agency_response_span=response_span,
                split_status=split_status,
                spans=spans,
            ))
            continue

        author_name = author_info.get("name")
        author_type = author_info.get("type") or "organization"
        if not author_name:
            comment_blocks.append(CommentBlock(
                block_id=block_id,
                char_span_raw=(block_start, block_end),
                pages=_block_pages(block_start, block_end, ingest),
                comment_text_span=comment_text_span,
                agency_response_span=response_span,
                split_status=split_status,
                spans=spans,
            ))
            continue

        # 6. Span-ID quote selection. Only send is_comment_text spans.
        comment_spans = [sp for sp in spans if sp.is_comment_text]
        selected_span = _select_quote_span(comment_spans, ingest, llm)

        # 7. Stance classification (two-pass)
        if selected_span is not None:
            sentence_text = ingest.raw_text[
                selected_span.char_span_raw[0]:selected_span.char_span_raw[1]
            ]
            context_text = _surrounding_context(selected_span, comment_spans, ingest)
            stance_data = _classify_stance(sentence_text, context_text, llm)
        else:
            stance_data = None

        # 8. Build Quote + StanceRecord
        stance_records: list[StanceRecord] = []
        if stance_data is not None:
            stance = stance_data.get("stance") or "neutral"
            stance_target = stance_data.get("stance_target") or "unknown"
            stance_confidence = float(stance_data.get("confidence") or 0.5)
            quote: Quote | None = None
            if selected_span is not None:
                qs, qe = selected_span.char_span_raw
                raw_text = ingest.raw_text[qs:qe]
                # Light dehyphenation for display only; raw stays verbatim.
                display_text = re.sub(r"-\n\s*", "", raw_text)
                display_text = re.sub(r"\s+", " ", display_text).strip()
                page = char_to_page(ingest.pages, qs) or 1
                section_for_quote = "public_comments"  # spans are restricted to comment_text
                source_text_hash = "sha256:" + hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
                quote = Quote(
                    text_raw=raw_text,
                    text_display=display_text,
                    char_offset_raw=(qs, qe),
                    page=page,
                    section=section_for_quote,
                    source_text_hash=source_text_hash,
                    normalization_rules_applied=["dehyphenation_for_display"],
                    quote_status="ok",
                    span_id=selected_span.span_id,
                )
            stance_records.append(StanceRecord(
                stance=stance,  # type: ignore[arg-type]
                stance_target=StanceTarget(type=stance_target),  # type: ignore[arg-type]
                stance_confidence=stance_confidence,
                quote=quote,
                sequence_order=1,
            ))
        # No stance found — still emit the stakeholder with zero stance_records,
        # so we don't lose the author identification.

        comment_blocks.append(CommentBlock(
            block_id=block_id,
            char_span_raw=(block_start, block_end),
            pages=_block_pages(block_start, block_end, ingest),
            comment_text_span=comment_text_span,
            agency_response_span=response_span,
            split_status=split_status,
            spans=spans,
        ))
        stakeholders.append(Stakeholder(
            comment_block_id=block_id,
            comment_author=CommentAuthor(name=author_name, type=author_type),  # type: ignore[arg-type]
            authorship_role="primary_author",
            stance_records=stance_records,
            appearance_order=appearance_order,
        ))
        appearance_order += 1

    record.comment_blocks = comment_blocks
    record.stakeholders = stakeholders
    if stakeholders:
        record.stakeholder_status = "ok"
    elif comment_blocks:
        record.stakeholder_status = "needs_review"
        warnings.append(
            "stakeholders: blocks present but all were filtered (persons-only or "
            "author-extraction failed)"
        )
    else:
        record.stakeholder_status = "no_comment_section_found"

    return warnings


# ---------------------------------------------------------------------------
# Block detection
# ---------------------------------------------------------------------------

def _detect_blocks(section_text: str, section_start: int) -> list[dict]:
    """Return list of {block_id, start, end} dicts in raw-text coordinates.
    First tries COMMENT-N regex; falls back to chunking on blank lines."""
    matches = list(_COMMENT_HEADER_RE.finditer(section_text))
    if matches:
        blocks: list[dict] = []
        for i, m in enumerate(matches):
            start = section_start + m.start()
            end = section_start + (
                matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
            )
            comment_num = m.group(1)
            blocks.append({
                "block_id": f"block_{comment_num}",
                "start": start,
                "end": end,
            })
        logger.info("Stage 3f: detected %d COMMENT-N blocks", len(blocks))
        return blocks

    # Fallback: split on blank lines, target ~3000-char blocks.
    paragraphs = _split_blank_lines(section_text)
    if not paragraphs:
        return []
    blocks = []
    cur_start = section_start + paragraphs[0][0]
    cur_end = section_start + paragraphs[0][1]
    cur_chars = paragraphs[0][1] - paragraphs[0][0]
    block_idx = 1
    for s, e in paragraphs[1:]:
        s_abs = section_start + s
        e_abs = section_start + e
        if cur_chars + (e_abs - s_abs) >= _FALLBACK_BLOCK_TARGET_CHARS:
            blocks.append({
                "block_id": f"block_{block_idx}",
                "start": cur_start,
                "end": cur_end,
            })
            block_idx += 1
            cur_start = s_abs
            cur_end = e_abs
            cur_chars = e_abs - s_abs
        else:
            cur_end = e_abs
            cur_chars += e_abs - s_abs
    blocks.append({"block_id": f"block_{block_idx}", "start": cur_start, "end": cur_end})
    logger.info("Stage 3f: detected %d fallback chunked blocks (no COMMENT-N markers)",
                len(blocks))
    return blocks


def _split_blank_lines(text: str) -> list[tuple[int, int]]:
    """Return list of (start, end) for paragraphs separated by 1+ blank lines.
    Skips empty paragraphs and those under 50 chars."""
    out: list[tuple[int, int]] = []
    pos = 0
    n = len(text)
    while pos < n:
        # Skip leading whitespace/newlines
        while pos < n and text[pos] in " \t\r\n":
            pos += 1
        if pos >= n:
            break
        start = pos
        # Walk until we hit a blank line (\n followed by whitespace then \n) or EOF
        while pos < n:
            nl = text.find("\n", pos)
            if nl == -1:
                pos = n
                break
            j = nl + 1
            while j < n and text[j] in " \t":
                j += 1
            if j < n and text[j] == "\n":
                pos = nl + 1
                break
            pos = nl + 1
        end = pos
        if end - start >= 50:
            out.append((start, end))
    return out


# ---------------------------------------------------------------------------
# Comment/response split
# ---------------------------------------------------------------------------

def _split_comment_response(
    block_text: str,
    block_start: int,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None, str]:
    """Return (comment_text_span, agency_response_span, split_status) in
    absolute char_offset_raw coordinates."""
    m = _RESPONSE_HEADER_RE.search(block_text)
    if m is None:
        return (
            (block_start, block_start + len(block_text)),
            None,
            "no_response",
        )
    comment_end = block_start + m.start()
    response_start = block_start + m.end()
    if comment_end <= block_start:
        # Block starts with "Response:" — degenerate; flag as split-failed
        return (None, None, "comment_response_split_failed")
    return (
        (block_start, comment_end),
        (response_start, block_start + len(block_text)),
        "ok",
    )


# ---------------------------------------------------------------------------
# Sentence splitting + spans
# ---------------------------------------------------------------------------

def _build_spans(
    block_id: str,
    block_text: str,
    comment_text_local: tuple[int, int] | None,
    response_span_abs: tuple[int, int] | None,
    block_start: int,
    ingest: IngestArtifact,
) -> list[SpanRecord]:
    """Sentence-split block text. Each span gets a stable span_id and is
    flagged is_comment_text=True for spans that fall inside comment_text,
    False for spans inside the agency response."""
    spans: list[SpanRecord] = []
    span_idx = 1

    # If comment_text is the whole block (no response), use full block
    if comment_text_local is None:
        # Block degenerate (split failed). Still create one span covering
        # the whole block, flagged is_comment_text=False (defensive: we
        # don't know which side is which).
        sp = SpanRecord(
            span_id=f"{block_id}_s1",
            char_span_raw=(block_start, block_start + len(block_text)),
            page=char_to_page(ingest.pages, block_start) or 1,
            is_comment_text=False,
        )
        spans.append(sp)
        return spans

    cs, ce = comment_text_local
    comment_text = block_text[cs:ce]
    spans.extend(_sentences_to_spans(
        block_id, comment_text, block_start + cs, ingest, span_idx,
        is_comment_text=True,
    ))
    span_idx += len(spans)

    if response_span_abs is not None:
        rs, re_ = response_span_abs
        response_text = ingest.raw_text[rs:re_]
        spans.extend(_sentences_to_spans(
            block_id, response_text, rs, ingest, span_idx,
            is_comment_text=False,
        ))

    return spans


def _sentences_to_spans(
    block_id: str,
    text: str,
    text_offset: int,
    ingest: IngestArtifact,
    start_idx: int,
    *,
    is_comment_text: bool,
) -> list[SpanRecord]:
    """Sentence-split `text` and emit SpanRecord per sentence."""
    out: list[SpanRecord] = []
    pos = 0
    idx = start_idx
    n = len(text)
    # Use finditer to keep offsets accurate.
    boundaries = [m.start() for m in _SENTENCE_SPLIT_RE.finditer(text)]
    boundaries.append(n)
    last = 0
    for b in boundaries:
        if b - last < 3:
            last = b
            continue
        # Trim leading whitespace inside the span
        s = last
        while s < b and text[s] in " \t\r\n":
            s += 1
        if s >= b:
            last = b
            continue
        sentence_start = text_offset + s
        sentence_end = text_offset + b
        if sentence_end <= sentence_start:
            last = b
            continue
        out.append(SpanRecord(
            span_id=f"{block_id}_s{idx}",
            char_span_raw=(sentence_start, sentence_end),
            page=char_to_page(ingest.pages, sentence_start) or 1,
            is_comment_text=is_comment_text,
        ))
        idx += 1
        last = b
    return out


def _build_comment_block_no_split(raw: dict, ingest: IngestArtifact) -> CommentBlock:
    """Build a CommentBlock without LLM-driven split (used when llm is None)."""
    block_start = raw["start"]
    block_end = raw["end"]
    block_text = ingest.raw_text[block_start:block_end]
    comment_text_span, response_span, split_status = _split_comment_response(
        block_text, block_start
    )
    comment_text_local = None
    if comment_text_span is not None:
        comment_text_local = (
            comment_text_span[0] - block_start, comment_text_span[1] - block_start,
        )
    spans = _build_spans(
        raw["block_id"], block_text, comment_text_local, response_span,
        block_start, ingest,
    )
    return CommentBlock(
        block_id=raw["block_id"],
        char_span_raw=(block_start, block_end),
        pages=_block_pages(block_start, block_end, ingest),
        comment_text_span=comment_text_span,
        agency_response_span=response_span,
        split_status=split_status,
        spans=spans,
    )


def _block_pages(start: int, end: int, ingest: IngestArtifact) -> list[int]:
    s_page = char_to_page(ingest.pages, start) or 1
    e_page = char_to_page(ingest.pages, max(end - 1, start)) or s_page
    return list(range(s_page, e_page + 1))


# ---------------------------------------------------------------------------
# Quote span selection (cue-rank + Sonnet)
# ---------------------------------------------------------------------------

def _cue_rank_spans(spans: list[SpanRecord], ingest: IngestArtifact) -> list[SpanRecord]:
    """Score each span by stance-cue word count, return descending."""
    scored: list[tuple[int, SpanRecord]] = []
    for sp in spans:
        s, e = sp.char_span_raw
        if e - s < _MIN_SPAN_CHARS or e - s > _MAX_SPAN_CHARS:
            continue
        text = ingest.raw_text[s:e].lower()
        score = sum(1 for cue in _STANCE_CUES if cue in text)
        scored.append((score, sp))
    scored.sort(key=lambda t: -t[0])
    # Always return at least the top-N even if score=0
    return [sp for _, sp in scored[:_MAX_QUOTE_CANDIDATES]]


def _select_quote_span(
    spans: list[SpanRecord],
    ingest: IngestArtifact,
    llm: "LLMClient",
) -> SpanRecord | None:
    candidates = _cue_rank_spans(spans, ingest)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Build candidate prompt: span_id + sentence text
    lines = []
    for sp in candidates:
        s, e = sp.char_span_raw
        text = ingest.raw_text[s:e].strip().replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        lines.append(f"- span_id: {sp.span_id}\n  sentence: {text}")
    candidates_str = "\n".join(lines)

    prompt_template = (PROMPTS_DIR / "3f_quote_select.txt").read_text(encoding="utf-8")
    prompt = prompt_template.replace("{candidates}", candidates_str)

    try:
        result = llm.call_json(
            model=llm.models["sonnet"],
            system="You are a careful reader. Return only the requested JSON.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.1,
            label="3f_quote_select",
        )
    except Exception as exc:
        logger.warning("3f quote selection LLM call failed: %s", exc)
        return candidates[0]  # fall back to highest cue-ranked

    if not isinstance(result, dict):
        return candidates[0]
    selected_id = result.get("selected_span_id")
    if not isinstance(selected_id, str):
        return None
    by_id = {sp.span_id: sp for sp in candidates}
    return by_id.get(selected_id)


def _surrounding_context(
    selected: SpanRecord,
    all_spans: list[SpanRecord],
    ingest: IngestArtifact,
) -> str:
    """Get up to 2 sentences of context surrounding the selected span (same block)."""
    idx = next((i for i, sp in enumerate(all_spans) if sp.span_id == selected.span_id), -1)
    if idx == -1:
        return ""
    start = max(0, idx - 1)
    end = min(len(all_spans), idx + 2)
    parts: list[str] = []
    for sp in all_spans[start:end]:
        if sp.span_id == selected.span_id:
            continue
        s, e = sp.char_span_raw
        parts.append(re.sub(r"\s+", " ", ingest.raw_text[s:e].strip()))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Author extraction
# ---------------------------------------------------------------------------

def _extract_author(block_excerpt: str, llm: "LLMClient") -> dict | None:
    prompt_template = (PROMPTS_DIR / "3f_author.txt").read_text(encoding="utf-8")
    prompt = prompt_template.replace("{block_text}", block_excerpt)
    try:
        result = llm.call_json(
            model=llm.models["haiku"],
            system="You are a careful classifier. Return only the requested JSON.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.1,
            label="3f_author",
        )
    except Exception as exc:
        logger.warning("3f author extraction failed: %s", exc)
        return None
    if not isinstance(result, dict):
        return None
    return result


# ---------------------------------------------------------------------------
# Stance classification (two-pass)
# ---------------------------------------------------------------------------

def _classify_stance(
    sentence: str, context: str, llm: "LLMClient",
) -> dict | None:
    prompt_template = (PROMPTS_DIR / "3f_stance.txt").read_text(encoding="utf-8")
    prompt = (
        prompt_template
        .replace("{sentence}", sentence)
        .replace("{context}", context or "(no surrounding context)")
    )

    # Pass 1: Haiku
    haiku_result = _stance_call(prompt, llm, model_key="haiku", label="3f_stance_haiku")
    if haiku_result is None:
        # Try Sonnet directly if Haiku failed
        return _stance_call(prompt, llm, model_key="sonnet", label="3f_stance_sonnet_fallback")

    confidence = float(haiku_result.get("confidence") or 0.0)
    stance = haiku_result.get("stance") or ""

    # Pass 2: Sonnet retry if low confidence or mixed
    if confidence < _STANCE_RETRY_THRESHOLD or stance == "mixed":
        logger.info("Stage 3f: stance retry on Sonnet (haiku confidence=%.2f, stance=%r)",
                    confidence, stance)
        sonnet_result = _stance_call(
            prompt, llm, model_key="sonnet", label="3f_stance_sonnet_retry"
        )
        if sonnet_result is not None:
            return sonnet_result

    return haiku_result


def _stance_call(
    prompt: str, llm: "LLMClient", *, model_key: str, label: str,
) -> dict | None:
    try:
        result = llm.call_json(
            model=llm.models[model_key],
            system="You are a careful classifier. Return only the requested JSON.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
            label=label,
        )
    except Exception as exc:
        logger.warning("Stance call (%s) failed: %s", label, exc)
        return None
    if not isinstance(result, dict):
        return None
    stance = result.get("stance")
    target = result.get("stance_target")
    if stance not in {"supportive", "opposed", "mixed", "neutral"}:
        return None
    if target not in {
        "proposed_action", "no_action", "specific_alternative",
        "mitigation", "process", "unknown",
    }:
        target = "unknown"
        result["stance_target"] = "unknown"
    return result
