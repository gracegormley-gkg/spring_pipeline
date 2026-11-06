"""
I/O layer: reads document folder, parses METS, loads TXT pages and CONFIDENCES.
Also provides load_from_digits_json() adapter for docs_with_digits.json input.
Produces a Document dataclass consumed by all downstream stages.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PAGE_SEPARATOR = "\f"


@dataclass
class PageRecord:
    page_num: int          # 1-indexed
    text: str
    char_start: int        # offset in Document.full_text
    char_end: int
    median_confidence: float | None = None


@dataclass
class METSData:
    title: str | None = None
    agency: str | None = None
    date: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class Document:
    doc_id: str
    project_id: str
    full_text: str              # pages joined with \f
    pages: list[PageRecord]
    mets: METSData
    doc_dir: Path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_document(doc_dir: str | Path) -> Document:
    """
    Load a single EIS document folder.

    Expected layout:
        P0491_<DOC_ID>/
        ├── TXT/<DOC_ID>_<PAGE>.txt
        ├── TXT/<DOC_ID>_<PAGE>.json   (optional per-page tokens)
        ├── CONFIDENCES/<DOC_ID>_<PAGE>.json
        ├── mets.xml
        └── mets.yaml

    Parses defensively — logs what schemas are actually present.
    """
    doc_dir = Path(doc_dir).resolve()
    if not doc_dir.is_dir():
        raise FileNotFoundError(f"Document directory not found: {doc_dir}")

    folder_name = doc_dir.name  # e.g. "P0491_35556036063543"
    parts = folder_name.split("_", 1)
    if len(parts) == 2:
        project_id, doc_id = parts[0], parts[1]
    else:
        project_id, doc_id = "UNKNOWN", folder_name
        logger.warning("Could not parse project_id/doc_id from folder name: %s", folder_name)

    mets = _load_mets(doc_dir)
    pages = _load_pages(doc_dir, doc_id)

    # Build full_text with form-feed separators and record char offsets
    page_records: list[PageRecord] = []
    chunks: list[str] = []
    offset = 0
    for i, (text, conf) in enumerate(pages, start=1):
        start = offset
        chunks.append(text)
        end = start + len(text)
        page_records.append(PageRecord(
            page_num=i,
            text=text,
            char_start=start,
            char_end=end,
            median_confidence=conf,
        ))
        offset = end + 1  # +1 for the \f separator

    full_text = PAGE_SEPARATOR.join(p for p, _ in pages)

    return Document(
        doc_id=doc_id,
        project_id=project_id,
        full_text=full_text,
        pages=page_records,
        mets=mets,
        doc_dir=doc_dir,
    )


# ---------------------------------------------------------------------------
# METS parsing
# ---------------------------------------------------------------------------

def _load_mets(doc_dir: Path) -> METSData:
    """Try mets.xml first, fall back to mets.yaml."""
    xml_path = doc_dir / "mets.xml"
    yaml_path = doc_dir / "mets.yaml"

    if xml_path.exists():
        try:
            return _parse_mets_xml(xml_path)
        except Exception as exc:
            logger.warning("Failed to parse mets.xml: %s — trying mets.yaml", exc)

    if yaml_path.exists():
        try:
            return _parse_mets_yaml(yaml_path)
        except Exception as exc:
            logger.warning("Failed to parse mets.yaml: %s", exc)

    logger.warning("No METS file found in %s", doc_dir)
    return METSData()


def _parse_mets_xml(path: Path) -> METSData:
    from lxml import etree  # type: ignore

    tree = etree.parse(str(path))
    root = tree.getroot()

    # Common METS/MODS XPaths — try several in priority order
    ns = {
        "mets": "http://www.loc.gov/METS/",
        "mods": "http://www.loc.gov/mods/v3",
        "dc":   "http://purl.org/dc/elements/1.1/",
    }

    title = (
        _xpath_text(root, ".//mods:titleInfo/mods:title", ns)
        or _xpath_text(root, ".//dc:title", ns)
        or _xpath_text(root, ".//title", {})
    )

    agency = (
        _xpath_text(root, ".//mods:name[mods:role/mods:roleTerm='creator']/mods:namePart", ns)
        or _xpath_text(root, ".//mods:name/mods:namePart", ns)
        or _xpath_text(root, ".//dc:creator", ns)
        or _xpath_text(root, ".//dc:publisher", ns)
    )

    date = (
        _xpath_text(root, ".//mods:originInfo/mods:dateIssued", ns)
        or _xpath_text(root, ".//mods:originInfo/mods:dateCreated", ns)
        or _xpath_text(root, ".//dc:date", ns)
    )

    # Log what we found so we can learn the actual schema
    logger.info(
        "METS XML parsed — title=%r agency=%r date=%r",
        title, agency, date,
    )

    raw: dict[str, Any] = {}
    try:
        # Dump a sample of the XML as text for diagnostics
        raw["sample"] = etree.tostring(root, pretty_print=True).decode()[:2000]
    except Exception:
        pass

    return METSData(title=title, agency=agency, date=date, raw=raw)


def _parse_mets_yaml(path: Path) -> METSData:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    logger.info("METS YAML keys: %s", list(data.keys()))

    # Try common key names defensively
    title = (
        data.get("title")
        or data.get("Title")
        or _deep_get(data, "titleInfo", "title")
    )
    agency = (
        data.get("agency")
        or data.get("creator")
        or data.get("bureau")
        or data.get("contributor")
    )
    date = (
        data.get("date")
        or data.get("dateIssued")
        or data.get("date_created")
        or data.get("year")
    )

    if date is not None:
        date = str(date)

    return METSData(title=title, agency=agency, date=date, raw=data)


# ---------------------------------------------------------------------------
# TXT + CONFIDENCES loading
# ---------------------------------------------------------------------------

def _load_pages(doc_dir: Path, doc_id: str) -> list[tuple[str, float | None]]:
    """
    Returns list of (page_text, median_confidence) tuples, sorted by page number.
    """
    txt_dir = doc_dir / "TXT"
    conf_dir = doc_dir / "CONFIDENCES"

    if not txt_dir.exists():
        raise FileNotFoundError(f"TXT directory not found: {txt_dir}")

    # Find all .txt files and sort numerically by page suffix
    txt_files = sorted(
        txt_dir.glob(f"*.txt"),
        key=lambda p: _extract_page_num(p.stem),
    )

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {txt_dir}")

    logger.info("Found %d TXT pages in %s", len(txt_files), txt_dir)

    pages: list[tuple[str, float | None]] = []
    for txt_path in txt_files:
        text = txt_path.read_text(encoding="utf-8", errors="replace")

        # Find matching confidence file
        stem = txt_path.stem  # e.g. "35556036063543_00000001"
        conf_path = conf_dir / f"{stem}.json" if conf_dir.exists() else None
        conf = _load_confidence(conf_path)

        pages.append((text, conf))

    return pages


def _extract_page_num(stem: str) -> int:
    """Extract numeric page suffix from a filename stem like '35556036063543_00000042'."""
    match = re.search(r"_(\d+)$", stem)
    if match:
        return int(match.group(1))
    # fallback: try to parse the whole stem as a number
    try:
        return int(stem)
    except ValueError:
        return 0


def _load_confidence(conf_path: Path | None) -> float | None:
    """
    Parse a CONFIDENCES JSON file. Handles multiple known shapes:
      1. {"words": [{"word": "...", "confidence": 0.95}, ...]}
      2. {"median_confidence": 0.95}
      3. {"confidence": 0.95}
      4. [{"text": "...", "confidence": 0.95}, ...]
    Logs the schema on first encounter. Returns None if unparseable.
    """
    if conf_path is None or not conf_path.exists():
        return None

    try:
        with open(conf_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        logger.debug("Could not load confidence file %s: %s", conf_path, exc)
        return None

    # Shape 1: {"words": [...]}
    if isinstance(data, dict) and "words" in data:
        words = data["words"]
        confs = [w.get("confidence") for w in words if isinstance(w, dict) and w.get("confidence") is not None]
        if confs:
            return statistics.median(confs)

    # Shape 2/3: single scalar
    if isinstance(data, dict):
        for key in ("median_confidence", "confidence", "score"):
            if key in data and isinstance(data[key], (int, float)):
                return float(data[key])

    # Shape 4: list of word objects
    if isinstance(data, list):
        confs = [item.get("confidence") for item in data if isinstance(item, dict) and item.get("confidence") is not None]
        if confs:
            return statistics.median(confs)

    logger.debug("Unknown confidence schema in %s: %s", conf_path, str(data)[:200])
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xpath_text(root: Any, xpath: str, ns: dict) -> str | None:
    try:
        elements = root.xpath(xpath, namespaces=ns)
        if elements:
            result = elements[0]
            return result.text.strip() if hasattr(result, "text") and result.text else str(result).strip() or None
    except Exception:
        pass
    return None


def _deep_get(d: dict, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)  # type: ignore[assignment]
    return d


# ---------------------------------------------------------------------------
# Page index helpers (used by downstream stages)
# ---------------------------------------------------------------------------

def char_offset_to_page(doc: Document, offset: int) -> int | None:
    """Return the 1-indexed page number for a character offset in doc.full_text."""
    for page in doc.pages:
        if page.char_start <= offset <= page.char_end:
            return page.page_num
    return None


def page_range_text(doc: Document, start_page: int, end_page: int) -> str:
    """Extract concatenated text for a page range (inclusive, 1-indexed)."""
    parts = [
        p.text for p in doc.pages
        if start_page <= p.page_num <= end_page
    ]
    return PAGE_SEPARATOR.join(parts)


# ---------------------------------------------------------------------------
# Adapter: load from docs_with_digits.json
# ---------------------------------------------------------------------------

_NUL_API_BASE = "https://api.dc.library.northwestern.edu/api/v2"
_NUL_COLLECTION_ID = "f2fc1bd8-c37f-4486-b28a-509f0e0362e1"
_CHARS_PER_FAKE_PAGE = 2500


def load_from_digits_json(
    json_path: str | Path,
    doc_key: str,
    fetch_nul_metadata: bool = True,
) -> Document:
    """
    Load a document from docs_with_digits.json.

    Args:
        json_path: path to docs_with_digits.json
        doc_key: key in the JSON, e.g. "P0491_35556036063543"
        fetch_nul_metadata: if True, attempt to fetch title/agency/date from NUL API

    The text has no page separators, so we split into fake ~2500-char pages
    on line boundaries. Confidence scores are unavailable and set to None.
    """
    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as fh:
        all_docs: dict[str, str] = json.load(fh)

    if doc_key not in all_docs:
        raise KeyError(
            f"Key {doc_key!r} not found in {json_path}. "
            f"Available keys (first 5): {list(all_docs.keys())[:5]}"
        )

    full_text: str = all_docs[doc_key]
    logger.info("Loaded %s: %d chars from %s", doc_key, len(full_text), json_path.name)

    # Parse project_id and doc_id from key
    parts = doc_key.split("_", 1)
    project_id = parts[0] if len(parts) == 2 else "UNKNOWN"
    doc_id = parts[1] if len(parts) == 2 else doc_key

    # Split into fake pages on line boundaries at ~2500 chars
    page_texts = _split_into_fake_pages(full_text, _CHARS_PER_FAKE_PAGE)
    logger.info("Split into %d fake pages (~%d chars each)", len(page_texts), _CHARS_PER_FAKE_PAGE)

    # Build PageRecord list
    page_records: list[PageRecord] = []
    offset = 0
    for i, text in enumerate(page_texts, start=1):
        start = offset
        end = start + len(text)
        page_records.append(PageRecord(
            page_num=i,
            text=text,
            char_start=start,
            char_end=end,
            median_confidence=None,  # not available from this source
        ))
        offset = end + 1  # +1 for the \f separator

    full_text_joined = PAGE_SEPARATOR.join(page_texts)

    # Optionally fetch NUL metadata
    mets = METSData()
    if fetch_nul_metadata:
        mets = _fetch_nul_metadata(doc_key, doc_id)

    return Document(
        doc_id=doc_id,
        project_id=project_id,
        full_text=full_text_joined,
        pages=page_records,
        mets=mets,
        doc_dir=json_path.parent,
    )


def _split_into_fake_pages(text: str, chars_per_page: int) -> list[str]:
    """
    Split a flat string into chunks of ~chars_per_page, breaking on newlines.
    Never splits mid-line.
    """
    lines = text.splitlines(keepends=True)
    pages: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        current.append(line)
        current_len += len(line)
        if current_len >= chars_per_page:
            pages.append("".join(current).rstrip())
            current = []
            current_len = 0

    if current:
        pages.append("".join(current).rstrip())

    return [p for p in pages if p.strip()]  # drop blank pages


def _fetch_nul_metadata(doc_key: str, doc_id: str) -> METSData:
    """
    Fetch title, agency (contributor), and date from the NUL Digital Collections API.
    Returns empty METSData on any failure — does not raise.
    """
    try:
        # Search by accession_number
        params = urllib.parse.urlencode({
            "query": f'accession_number:"{doc_key}"',
            "size": 1,
        })
        url = f"{_NUL_API_BASE}/search?{params}"
        logger.debug("NUL API request: %s", url)

        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        hits = data.get("data", [])

        # Fallback: try bare barcode if no hit with full key
        if not hits:
            params2 = urllib.parse.urlencode({
                "query": f'accession_number:"{doc_id}"',
                "size": 1,
            })
            url2 = f"{_NUL_API_BASE}/search?{params2}"
            req2 = urllib.request.Request(url2, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                data2 = json.loads(resp2.read())
            hits = data2.get("data", [])

        if not hits:
            logger.info("NUL API: no match for %s", doc_key)
            return METSData()

        work = hits[0]
        title = _extract_nul_title(work)
        agency = _extract_nul_agency(work)
        date = _extract_nul_date(work)

        logger.info("NUL API match: title=%r agency=%r date=%r", title, agency, date)
        time.sleep(0.3)  # be polite
        return METSData(title=title, agency=agency, date=date, raw=work)

    except Exception as exc:
        logger.warning("NUL API fetch failed for %s: %s — continuing without metadata", doc_key, exc)
        return METSData()


def _extract_nul_title(work: dict) -> str | None:
    title = work.get("title")
    if isinstance(title, list):
        return title[0] if title else None
    return title or None


def _extract_nul_agency(work: dict) -> str | None:
    contributors = work.get("contributor") or []
    if isinstance(contributors, list) and contributors:
        first = contributors[0]
        if isinstance(first, dict):
            return first.get("label") or first.get("id") or None
        return str(first)
    return None


def _extract_nul_date(work: dict) -> str | None:
    dates = work.get("date_created")
    if isinstance(dates, list) and dates:
        return str(dates[0])
    if isinstance(dates, str):
        return dates
    return None
