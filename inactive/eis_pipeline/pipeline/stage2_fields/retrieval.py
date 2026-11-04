"""
Retrieval helpers: filter and rank chunks for a given field's context window.
"""

from __future__ import annotations

from ..schema import ChunkRecord


def get_chunks_by_tags(
    chunks: list[ChunkRecord],
    tags: list[str],
    max_chunks: int = 8,
) -> list[ChunkRecord]:
    """
    Return usable chunks matching any of the given topic tags, ranked by
    number of matching tags (descending), then by document order.
    """
    usable = [c for c in chunks if c.used]
    scored: list[tuple[int, int, ChunkRecord]] = []
    for i, chunk in enumerate(usable):
        match_count = sum(1 for t in tags if t in chunk.topic_tags)
        if match_count > 0:
            scored.append((-match_count, i, chunk))
    scored.sort()
    return [c for _, _, c in scored[:max_chunks]]


def get_chunks_by_keyword(
    chunks: list[ChunkRecord],
    keywords: list[str],
    max_chunks: int = 8,
    case_sensitive: bool = False,
) -> list[ChunkRecord]:
    """
    Return usable chunks containing any of the given keywords.
    Ranked by keyword hit count (descending), then document order.
    """
    usable = [c for c in chunks if c.used]
    scored: list[tuple[int, int, ChunkRecord]] = []

    for i, chunk in enumerate(usable):
        text = chunk.text if case_sensitive else chunk.text.lower()
        hits = sum(
            text.count(kw if case_sensitive else kw.lower())
            for kw in keywords
        )
        if hits > 0:
            scored.append((-hits, i, chunk))
    scored.sort()
    return [c for _, _, c in scored[:max_chunks]]


def get_chunks_mentioning(
    chunks: list[ChunkRecord],
    entity_name: str,
    max_chunks: int | None = None,
) -> list[ChunkRecord]:
    """Return all usable chunks that mention a given entity name (case-insensitive)."""
    name_lower = entity_name.lower()
    results = [
        c for c in chunks
        if c.used and name_lower in c.text.lower()
    ]
    if max_chunks is not None:
        results = results[:max_chunks]
    return results


def combine_chunk_context(chunks: list[ChunkRecord], max_chars: int = 60_000) -> str:
    """
    Concatenate chunk texts into a context string, with section headers.
    Truncates to max_chars.
    """
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        header = f"[Chunk {chunk.chunk_id} | Pages {chunk.pages[0] if chunk.pages else '?'}–{chunk.pages[-1] if chunk.pages else '?'} | {chunk.title}]"
        block = f"{header}\n{chunk.text}"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(block[:remaining] + "\n[TRUNCATED]")
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)
