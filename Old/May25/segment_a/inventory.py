"""
Local CSV inventory loader (replacement for NUL API metadata lookups).

Source: a MARC-export-shaped CSV at config.INVENTORY_CSV_PATH. We treat it as
the authoritative work catalogue for docs that aren't in the NUL collection
(e.g., P0491 docs scanned from a different microfilm run).

Columns we use:
  - 955$b   accession_number (e.g. "35556036091957"). Match key.
  - 245$ab  title.
  - 260$cg  publication date (we extract a 4-digit year from this).
  - 710     creator/contributor MARC field. We pull every $b subfield as the
            list of bureaus / sub-agencies (e.g. "Bureau of Outdoor Recreation").

`lookup_work(doc_id)` returns a dict shaped like a NUL work so the existing
m1.py / selection.py helpers (`get_year`, `get_contributors`, `extract_title`)
keep working unchanged. The doc_id can be either bare digits ("35556036091957")
or prefixed ("p0491_35556036091957", "P0491_..." — case-insensitive); we match
on the trailing 14-digit accession.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Column names (verified against the actual CSV header).
COL_ACCESSION = "955$b"
COL_TITLE = "245$ab"
COL_DATE = "260$cg"
COL_710 = "710"

# 14-digit accession at the end of a doc_id like "p0491_35556036091957".
_ACCESSION_DIGITS_RE = re.compile(r"(\d{14})")
# Strip MARC indicator prefix ("10", "1\\", etc.) at the start of a 710 cell.
# Every subfield begins with `$<letter-or-digit>`; we capture text up to the
# next subfield marker or end of cell.
_SUBFIELD_RE = re.compile(r"\$([a-z0-9])([^$]*)", re.IGNORECASE)
# Grab first 4-digit year >= 1900 from a 260$cg cell (handles "1974.", "[1974]", etc.).
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


# --- Loader ------------------------------------------------------------------

_INVENTORY_CACHE: Optional[dict[str, dict]] = None


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    # The export uses Latin-1-ish bytes (e.g. nbsp 0xA0). UTF-8 raises
    # UnicodeDecodeError; latin-1 reads cleanly with no character loss.
    with open(path, newline="", encoding="latin-1") as f:
        rdr = csv.reader(f)
        headers = next(rdr)
        rows = [r for r in rdr]
    return headers, rows


def _strip_marc_quoting(cell: str) -> str:
    """
    Strip the outer double-quotes some MARC exports leave inside CSV values.

    Cells often look like: "10$aUnited States.$bNational Park Service."
    csv.reader returns that with the outer quotes still attached because they
    were doubled inside a quoted CSV field. We trim them so subfield extraction
    doesn't trail a stray `"`.
    """
    s = (cell or "").strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s


def _subfields(cell: str) -> list[tuple[str, str]]:
    """Return [(code, value), ...] of MARC subfields in a cell, in order."""
    if not cell:
        return []
    s = _strip_marc_quoting(cell)
    return [(m.group(1).lower(), m.group(2).strip()) for m in _SUBFIELD_RE.finditer(s)]


def _bureaus_from_710(cell: str) -> list[str]:
    """Pull every $b subfield from a 710 cell, lightly cleaned."""
    out: list[str] = []
    for code, val in _subfields(cell):
        if code == "b":
            v = val.strip().rstrip(".").strip().rstrip('"').strip().rstrip(".").strip()
            if v:
                out.append(v)
    return out


def _year_from_260(cell: str) -> Optional[int]:
    if not cell:
        return None
    m = _YEAR_RE.search(_strip_marc_quoting(cell))
    return int(m.group(1)) if m else None


def _clean_title(cell: str) -> str:
    return _strip_marc_quoting(cell).strip()


def _accession_key(value: str) -> Optional[str]:
    """Normalise a doc_id or accession into the 14-digit match key, or None."""
    if not value:
        return None
    m = _ACCESSION_DIGITS_RE.search(value)
    return m.group(1) if m else None


# --- Public API --------------------------------------------------------------

def load_inventory(path: Optional[Path] = None, force: bool = False) -> dict[str, dict]:
    """
    Load the CSV once and index it by accession digits → row dict.

    Returns: {"35556036091957": {"title": ..., "year": ..., "bureaus": [...], "row_index": N}, ...}
    """
    global _INVENTORY_CACHE
    if _INVENTORY_CACHE is not None and not force:
        return _INVENTORY_CACHE

    from config import INVENTORY_CSV_PATH
    p = Path(path) if path else INVENTORY_CSV_PATH
    if not p.exists():
        raise FileNotFoundError(f"Inventory CSV not found: {p}")

    headers, rows = _read_csv(p)
    try:
        i_acc = headers.index(COL_ACCESSION)
        i_title = headers.index(COL_TITLE)
        i_date = headers.index(COL_DATE)
        i_710 = headers.index(COL_710)
    except ValueError as e:
        raise RuntimeError(
            f"Inventory CSV missing required column ({e}). "
            f"Got headers[:8]={headers[:8]} ..."
        )

    index: dict[str, dict] = {}
    dupes = 0
    for n, row in enumerate(rows):
        if len(row) <= max(i_acc, i_title, i_date, i_710):
            continue
        acc_raw = (row[i_acc] or "").strip()
        key = _accession_key(acc_raw)
        if not key:
            continue
        rec = {
            "accession": acc_raw,
            "title": _clean_title(row[i_title]),
            "year": _year_from_260(row[i_date]),
            "bureaus": _bureaus_from_710(row[i_710]),
            "row_index": n,
        }
        if key in index:
            dupes += 1
            # Keep the first; later duplicates are typically reprints/copies.
            continue
        index[key] = rec

    log.info(
        f"Loaded inventory from {p.name}: {len(index)} unique accessions "
        f"(of {len(rows)} rows; {dupes} duplicate keys skipped)"
    )
    _INVENTORY_CACHE = index
    return index


def lookup_work(doc_id: str) -> Optional[dict]:
    """
    Return a NUL-work-shaped dict for `doc_id`, or None if not in the inventory.

    Shape mirrors what nul.fetch_collection_works() returns for the fields the
    pipeline actually consumes:
        id                - synthetic, "csv:<accession>"
        accession_number  - matches doc_id digits
        title             - from 245$ab
        create_date       - "<year>" string (so nul.get_year picks it up)
        contributor       - list of {"label_with_role": "<bureau> (lead)"}
                            (so nul.get_contributors / m1.extract_lead_agency
                             see them as joint contributors)
        nul_metadata      - same fields, for any code that prefers that nesting
        source            - "inventory_csv"
    """
    inv = load_inventory()
    key = _accession_key(doc_id)
    if not key:
        return None
    rec = inv.get(key)
    if rec is None:
        return None

    contributors = [{"label_with_role": f"{b} (lead)", "label": b} for b in rec["bureaus"]]
    work = {
        "id": f"csv:{rec['accession']}",
        "accession_number": rec["accession"],
        "title": rec["title"],
        "create_date": str(rec["year"]) if rec["year"] else None,
        "contributor": contributors,
        "creator": [],  # leave empty; m1 unions contributor + creator
        "nul_metadata": {
            "title": rec["title"],
            "year": rec["year"],
            "bureaus": rec["bureaus"],
        },
        "source": "inventory_csv",
        "_csv_row_index": rec["row_index"],
    }
    return work
