"""Tests for pipeline/io_layer.py using synthetic fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.io_layer import load_document, char_offset_to_page, page_range_text


FIXTURES = Path(__file__).parent / "fixtures"


def _make_doc_dir(tmp_path: Path, pages: list[str], confidences: list[dict | None] = None) -> Path:
    """Build a minimal synthetic document folder."""
    doc_id = "99999999999999"
    folder = tmp_path / f"P0491_{doc_id}"
    txt_dir = folder / "TXT"
    conf_dir = folder / "CONFIDENCES"
    txt_dir.mkdir(parents=True)
    conf_dir.mkdir(parents=True)

    for i, text in enumerate(pages, start=1):
        page_str = f"{i:08d}"
        (txt_dir / f"{doc_id}_{page_str}.txt").write_text(text, encoding="utf-8")
        conf_data = None
        if confidences and i - 1 < len(confidences):
            conf_data = confidences[i - 1]
        if conf_data is not None:
            (conf_dir / f"{doc_id}_{page_str}.json").write_text(
                json.dumps(conf_data), encoding="utf-8"
            )

    # Minimal mets.yaml
    (folder / "mets.yaml").write_text(
        "title: Test EIS Document\nagency: Bureau of Land Management\ndate: '1985'\n",
        encoding="utf-8",
    )

    return folder


# ---------------------------------------------------------------------------
# Basic load
# ---------------------------------------------------------------------------

def test_load_document_basic(tmp_path):
    folder = _make_doc_dir(tmp_path, pages=["Page one text.", "Page two text."])
    doc = load_document(folder)

    assert doc.doc_id == "99999999999999"
    assert doc.project_id == "P0491"
    assert len(doc.pages) == 2
    assert doc.pages[0].page_num == 1
    assert doc.pages[1].page_num == 2
    assert "Page one text." in doc.full_text
    assert "Page two text." in doc.full_text


def test_page_separator_present(tmp_path):
    folder = _make_doc_dir(tmp_path, pages=["AAA", "BBB", "CCC"])
    doc = load_document(folder)
    assert doc.full_text.count("\f") == 2


def test_page_offsets_correct(tmp_path):
    pages = ["Hello world", "Goodbye world"]
    folder = _make_doc_dir(tmp_path, pages=pages)
    doc = load_document(folder)

    # char_start of page 1 should be 0
    assert doc.pages[0].char_start == 0
    assert doc.pages[0].char_end == len("Hello world")


def test_char_offset_to_page(tmp_path):
    folder = _make_doc_dir(tmp_path, pages=["AAAA", "BBBB", "CCCC"])
    doc = load_document(folder)

    # First char → page 1
    assert char_offset_to_page(doc, 0) == 1
    # First char of page 2 = len("AAAA") + 1 (for \f)
    p2_start = doc.pages[1].char_start
    assert char_offset_to_page(doc, p2_start) == 2


def test_page_sort_order(tmp_path):
    """Pages must sort numerically, not lexicographically."""
    doc_id = "99999999999999"
    folder = tmp_path / f"P0491_{doc_id}"
    txt_dir = folder / "TXT"
    txt_dir.mkdir(parents=True)
    (folder / "mets.yaml").write_text("title: Sort Test\n", encoding="utf-8")

    # Write pages 1–12 to exercise numeric vs lexicographic sort
    for i in [1, 2, 9, 10, 11, 12]:
        (txt_dir / f"{doc_id}_{i:08d}.txt").write_text(f"Page {i}", encoding="utf-8")

    doc = load_document(folder)
    page_nums = [p.page_num for p in doc.pages]
    assert page_nums == sorted(page_nums)


# ---------------------------------------------------------------------------
# METS parsing
# ---------------------------------------------------------------------------

def test_mets_yaml_title(tmp_path):
    folder = _make_doc_dir(tmp_path, pages=["text"])
    doc = load_document(folder)
    assert doc.mets.title == "Test EIS Document"
    assert doc.mets.agency == "Bureau of Land Management"
    assert doc.mets.date == "1985"


def test_missing_mets(tmp_path):
    """Should still load — just with empty METS."""
    doc_id = "88888888888888"
    folder = tmp_path / f"P0491_{doc_id}"
    txt_dir = folder / "TXT"
    txt_dir.mkdir(parents=True)
    (txt_dir / f"{doc_id}_00000001.txt").write_text("Some text", encoding="utf-8")

    doc = load_document(folder)
    assert doc.mets.title is None
    assert doc.mets.agency is None


# ---------------------------------------------------------------------------
# Confidence parsing
# ---------------------------------------------------------------------------

def test_confidence_word_list_shape(tmp_path):
    conf = {"words": [{"word": "hello", "confidence": 0.9}, {"word": "world", "confidence": 0.8}]}
    folder = _make_doc_dir(tmp_path, pages=["hello world"], confidences=[conf])
    doc = load_document(folder)
    assert doc.pages[0].median_confidence == pytest.approx(0.85)


def test_confidence_scalar_shape(tmp_path):
    conf = {"median_confidence": 0.92}
    folder = _make_doc_dir(tmp_path, pages=["text"], confidences=[conf])
    doc = load_document(folder)
    assert doc.pages[0].median_confidence == pytest.approx(0.92)


def test_confidence_missing(tmp_path):
    folder = _make_doc_dir(tmp_path, pages=["text"], confidences=[None])
    doc = load_document(folder)
    assert doc.pages[0].median_confidence is None


# ---------------------------------------------------------------------------
# page_range_text
# ---------------------------------------------------------------------------

def test_page_range_text(tmp_path):
    folder = _make_doc_dir(tmp_path, pages=["Alpha", "Beta", "Gamma", "Delta"])
    doc = load_document(folder)
    result = page_range_text(doc, 2, 3)
    assert "Beta" in result
    assert "Gamma" in result
    assert "Alpha" not in result
    assert "Delta" not in result
