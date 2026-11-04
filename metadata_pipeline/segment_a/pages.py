"""
Local page-JSON loader.

Source of truth for doc text. Assumes one JSON file per page per doc, laid out as:

    Documents/output/<doc_id>/page_NNNN.json

Each file: {"page_number": <int>, "text": "...", ...}

Pages are joined into `full_text` in page_num order, with a marker between pages so
quote→page resolution is unambiguous. A `Doc` exposes:

  - .full_text        joined text for chunking / prompts
  - .pages            list of {page_num, text}
  - .page_at_offset() char offset in full_text → page_num
  - .text_for_pages() page span → text (used by the Critic to pull cited pages)
  - .find_quote()     verbatim quote (whitespace-normalized) → page_num where it appears

Swap to a Mongo loader later by reimplementing `load_doc` — same return shape.
"""

from __future__ import annotations

import bisect
import json
import logging
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Separator used between pages in full_text. Distinctive enough to find page
# boundaries but cheap for the LLM to ignore.
PAGE_SEP = "\n\n"


@dataclass
class Page:
    page_num: int
    text: str


@dataclass
class Doc:
    doc_id: str
    pages: list[Page]
    full_text: str
    # offset-into-full_text where each page's text begins; aligned with self.pages
    _page_starts: list[int] = field(default_factory=list)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    def page_at_offset(self, offset: int) -> int:
        """Char offset in full_text → page_num. Clamps to first/last page on out-of-range."""
        if not self.pages:
            return 1
        if offset <= 0:
            return self.pages[0].page_num
        idx = bisect_right(self._page_starts, offset) - 1
        idx = max(0, min(idx, len(self.pages) - 1))
        return self.pages[idx].page_num

    def text_for_pages(self, start_page: int, end_page: int) -> str:
        """Concatenated text of pages in [start_page, end_page] (inclusive)."""
        out: list[str] = []
        for p in self.pages:
            if start_page <= p.page_num <= end_page:
                out.append(p.text)
        return PAGE_SEP.join(out)

    def find_quote(self, quote: str) -> Optional[tuple[int, int]]:
        """
        Locate a verbatim quote in the doc.

        Whitespace-normalized: the quote and each page are collapsed to single spaces
        before comparison. Returns (page_num, offset_in_full_text) on a hit, else None.
        Searches page-by-page so we never falsely match a quote that spans a page seam.
        """
        if not quote.strip():
            return None
        norm_quote = _normalize(quote)
        if not norm_quote:
            return None
        # Search per-page so we can return the exact page_num. (A quote that spans
        # two pages would be missed — that's intentional; we want a single-page citation.)
        for p in self.pages:
            norm_page = _normalize(p.text)
            if norm_quote in norm_page:
                # Best-effort offset into full_text — used by the critic to pull text.
                offset = self._page_starts[self._index_of_page(p.page_num)]
                return p.page_num, offset
        return None


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _index_of_page(self_doc, page_num: int) -> int:  # pragma: no cover — used inline
    raise NotImplementedError


# Attach _index_of_page as a method (avoid forward-ref pain in dataclass body).
def _attach_helpers() -> None:
    def _idx(self: Doc, page_num: int) -> int:
        for i, p in enumerate(self.pages):
            if p.page_num == page_num:
                return i
        return 0
    Doc._index_of_page = _idx  # type: ignore[attr-defined]


_attach_helpers()


# --- Loader -----------------------------------------------------------------

def _pages_dir(pages_data_dir: Path, doc_id: str) -> Path:
    return pages_data_dir / doc_id


def load_doc(doc_id: str, pages_data_dir: Optional[Path] = None) -> Doc:
    """Load a doc from `<pages_data_dir>/<doc_id>/<page_num>.json` files."""
    from config import PAGES_DATA_DIR
    base = pages_data_dir or PAGES_DATA_DIR
    doc_dir = _pages_dir(base, doc_id)
    if not doc_dir.exists():
        raise FileNotFoundError(f"No pages_data dir for doc_id {doc_id}: {doc_dir}")

    pages: list[Page] = []
    for f in sorted(doc_dir.glob("*.json"), key=_page_sort_key):
        with open(f) as fh:
            obj = json.load(fh)
        page_num = int(
            obj.get("page_number")
            or obj.get("page_num")
            or obj.get("page")
            or _page_num_from_filename(f)
        )
        text = obj.get("text") or obj.get("body") or obj.get("content") or ""
        pages.append(Page(page_num=page_num, text=text))

    pages.sort(key=lambda p: p.page_num)
    if not pages:
        raise FileNotFoundError(f"No page JSON files found under {doc_dir}")

    # Build full_text and remember each page's start offset for char→page lookups.
    parts: list[str] = []
    page_starts: list[int] = []
    cursor = 0
    for i, p in enumerate(pages):
        page_starts.append(cursor)
        parts.append(p.text)
        cursor += len(p.text)
        if i < len(pages) - 1:
            cursor += len(PAGE_SEP)
    full_text = PAGE_SEP.join(parts)

    return Doc(doc_id=doc_id, pages=pages, full_text=full_text, _page_starts=page_starts)


def list_doc_ids(pages_data_dir: Optional[Path] = None) -> list[str]:
    """List doc_ids that have a pages_data subdirectory."""
    from config import PAGES_DATA_DIR
    base = pages_data_dir or PAGES_DATA_DIR
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def _page_num_from_filename(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def _page_sort_key(path: Path):
    """Sort by leading integer in filename so '2.json' < '10.json'."""
    return (_page_num_from_filename(path), path.name)
