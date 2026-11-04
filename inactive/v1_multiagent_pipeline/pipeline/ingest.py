"""
Stage 1 — Ingest.

Loads a doc from docs_with_digits.json, normalizes text (whitespace, soft-hyphen
dehyphenation, Unicode quote normalization), builds an alignment map from
normalized→raw character offsets, and splits raw text into ~2500-char fake
pages on line boundaries.

Output schema (per build brief §6 Stage 1):
{
  "publication_id": str,
  "physical_record_ids": [str],
  "raw_text": str,
  "text_normalized": str,
  "alignment_map": [{"norm_start","norm_end","raw_start","raw_end"}],
  "pages": [{"page_num","char_start_raw","char_end_raw"}],
  "nul_metadata": {...}  # raw NUL response (if found)
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .nul_client import NULClient, NULRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes (intermediate, not persisted as Pydantic)
# ---------------------------------------------------------------------------

@dataclass
class AlignmentSpan:
    """Maps a normalized-text span to its raw-text origin."""
    norm_start: int
    norm_end: int
    raw_start: int
    raw_end: int


@dataclass
class FakePage:
    page_num: int
    char_start_raw: int
    char_end_raw: int


@dataclass
class IngestArtifact:
    publication_id: str
    physical_record_ids: list[str]
    raw_text: str
    text_normalized: str
    alignment_map: list[AlignmentSpan]
    pages: list[FakePage]
    nul_metadata: dict[str, Any] = field(default_factory=dict)
    raw_text_hash: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "publication_id": self.publication_id,
            "physical_record_ids": self.physical_record_ids,
            "raw_text_hash": self.raw_text_hash,
            "raw_text_length": len(self.raw_text),
            "text_normalized_length": len(self.text_normalized),
            "alignment_map": [asdict(a) for a in self.alignment_map],
            "pages": [asdict(p) for p in self.pages],
            "nul_metadata": self.nul_metadata,
        }


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

# Module-level cache: avoid re-reading docs_with_digits.json across calls in
# the same process. (Each entrypoint call clears nothing — assume the file is
# stable for the session.)
_DOCS_CACHE: dict[str, dict[str, str]] = {}


def load_docs_json(json_path: str | Path) -> dict[str, str]:
    """Lazily load and cache docs_with_digits.json."""
    key = str(Path(json_path).resolve())
    if key not in _DOCS_CACHE:
        with open(key, encoding="utf-8") as fh:
            _DOCS_CACHE[key] = json.load(fh)
        logger.info("Loaded docs_with_digits.json: %d keys from %s",
                    len(_DOCS_CACHE[key]), key)
    return _DOCS_CACHE[key]


def run(
    json_path: str | Path,
    doc_key: str,
    nul_client: NULClient,
) -> IngestArtifact:
    """Run Stage 1 for a single doc_key. Returns the in-memory artifact."""
    docs = load_docs_json(json_path)
    if doc_key not in docs:
        raise KeyError(
            f"doc_key={doc_key!r} not found in {json_path}. "
            f"First 5 keys: {list(docs.keys())[:5]}"
        )

    raw_text: str = docs[doc_key]
    raw_hash = _sha256(raw_text)
    logger.info("Stage 1: %s — %d chars, hash=%s", doc_key, len(raw_text), raw_hash[:12])

    text_normalized, alignment_map = _normalize_with_alignment(raw_text)
    pages = _split_into_fake_pages(raw_text, config.CHARS_PER_FAKE_PAGE)
    logger.info(
        "Stage 1: normalized %d chars; %d alignment spans; %d fake pages",
        len(text_normalized), len(alignment_map), len(pages),
    )

    nul_record = nul_client.get(doc_key)
    nul_metadata = _nul_to_dict(nul_record)

    return IngestArtifact(
        publication_id=doc_key,
        physical_record_ids=[doc_key],
        raw_text=raw_text,
        text_normalized=text_normalized,
        alignment_map=alignment_map,
        pages=pages,
        nul_metadata=nul_metadata,
        raw_text_hash=raw_hash,
    )


def write_artifact(artifact: IngestArtifact, output_dir: str | Path) -> Path:
    """Persist the JSON view of the artifact (without raw_text — too big for casual reads)."""
    path = Path(output_dir) / f"{artifact.publication_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_json(), indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Unicode quote/dash normalization map.
_NORMALIZE_MAP = {
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2026": "...",
    "\u00a0": " ",  # non-breaking space
}


def _normalize_with_alignment(raw: str) -> tuple[str, list[AlignmentSpan]]:
    """
    Build a normalized text and an alignment map back to the raw text.

    Rules:
      1. Soft-hyphen dehyphenation: '-\n' (optionally with surrounding spaces)
         collapses to '' (the broken word is rejoined).
      2. Unicode normalization (NFC) + curly→straight quote folding.
      3. Repeated whitespace not collapsed; we want char-level alignment.

    The alignment_map records every contiguous run of (norm_start, norm_end,
    raw_start, raw_end) where the mapping is identity. Substitutions and
    deletions force a new span boundary.
    """
    raw_nfc = unicodedata.normalize("NFC", raw)
    out_chars: list[str] = []
    spans: list[AlignmentSpan] = []
    cur_norm_start = 0
    cur_raw_start = 0
    cur_len = 0  # length of the current identity run (1:1 mapping)

    def flush_run(norm_pos: int, raw_pos: int) -> None:
        """Close the current identity span if it has content."""
        nonlocal cur_norm_start, cur_raw_start, cur_len
        if cur_len > 0:
            spans.append(AlignmentSpan(
                norm_start=cur_norm_start,
                norm_end=cur_norm_start + cur_len,
                raw_start=cur_raw_start,
                raw_end=cur_raw_start + cur_len,
            ))
        cur_norm_start = norm_pos
        cur_raw_start = raw_pos
        cur_len = 0

    i = 0
    n = len(raw_nfc)
    while i < n:
        ch = raw_nfc[i]

        # Rule 1: soft-hyphen dehyphenation across line break
        # Pattern: '-' optionally followed by '\r'? '\n' optionally with spaces
        if ch == "-" and i + 1 < n and raw_nfc[i + 1] in "\r\n":
            # Consume '-' + newline run + any leading whitespace on next line
            flush_run(len(out_chars), i)
            j = i + 1
            while j < n and raw_nfc[j] in "\r\n":
                j += 1
            while j < n and raw_nfc[j] in " \t":
                j += 1
            # Emit nothing; advance raw cursor to j
            i = j
            cur_norm_start = len(out_chars)
            cur_raw_start = i
            continue

        # Rule 2: substitution
        sub = _NORMALIZE_MAP.get(ch)
        if sub is not None:
            flush_run(len(out_chars), i)
            out_chars.append(sub)
            # If sub is multi-char, we record a single-span substitution.
            spans.append(AlignmentSpan(
                norm_start=len(out_chars) - len(sub),
                norm_end=len(out_chars),
                raw_start=i,
                raw_end=i + 1,
            ))
            i += 1
            cur_norm_start = len(out_chars)
            cur_raw_start = i
            continue

        # Default: identity copy. Extend current run.
        if cur_len == 0:
            cur_norm_start = len(out_chars)
            cur_raw_start = i
        out_chars.append(ch)
        cur_len += 1
        i += 1

    # Flush trailing run
    if cur_len > 0:
        spans.append(AlignmentSpan(
            norm_start=cur_norm_start,
            norm_end=cur_norm_start + cur_len,
            raw_start=cur_raw_start,
            raw_end=cur_raw_start + cur_len,
        ))

    text_normalized = "".join(out_chars)
    return text_normalized, spans


# ---------------------------------------------------------------------------
# Fake page splitting
# ---------------------------------------------------------------------------

def _split_into_fake_pages(raw_text: str, chars_per_page: int) -> list[FakePage]:
    """
    Split raw text on line boundaries into ~chars_per_page chunks.
    Records raw-text char ranges; never breaks mid-line.
    """
    pages: list[FakePage] = []
    page_num = 1
    cur_start = 0
    cur_len = 0

    pos = 0
    n = len(raw_text)
    while pos < n:
        # Find next newline (inclusive) or end-of-text.
        nl = raw_text.find("\n", pos)
        line_end = nl + 1 if nl != -1 else n
        line_len = line_end - pos
        cur_len += line_len
        pos = line_end

        if cur_len >= chars_per_page or pos >= n:
            pages.append(FakePage(
                page_num=page_num,
                char_start_raw=cur_start,
                char_end_raw=pos,
            ))
            page_num += 1
            cur_start = pos
            cur_len = 0

    # Drop trailing empty page if any (shouldn't happen with the loop above).
    return [p for p in pages if p.char_end_raw > p.char_start_raw]


# ---------------------------------------------------------------------------
# Page lookup helper (for downstream stages)
# ---------------------------------------------------------------------------

def char_to_page(pages: list[FakePage], char_offset_raw: int) -> int | None:
    """Map a raw-text character offset to a 1-indexed page number."""
    for p in pages:
        if p.char_start_raw <= char_offset_raw < p.char_end_raw:
            return p.page_num
    return None


def page_range_for_span(pages: list[FakePage], span: tuple[int, int]) -> tuple[int, int] | None:
    s, e = span
    s_page = char_to_page(pages, s)
    e_page = char_to_page(pages, max(e - 1, s))
    if s_page is None or e_page is None:
        return None
    return (s_page, e_page)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def _sha256(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def _nul_to_dict(rec: NULRecord) -> dict[str, Any]:
    return {
        "found": rec.found,
        "cache_hit": rec.cache_hit,
        "title": rec.title,
        "creator": rec.creator,
        "contributor": rec.contributor,
        "date_created": rec.date_created,
        "raw": rec.raw,
    }
