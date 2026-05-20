"""Tests for pipeline/stage0_triage.py using synthetic fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.io_layer import load_document, METSData
from pipeline import stage0_triage
from pipeline.schema import EISRecord, OCRInfo


FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> "load_document":
    return load_document(FIXTURES / name)


def _make_record(doc) -> EISRecord:
    return EISRecord(
        doc_id=doc.doc_id,
        project_id=doc.project_id,
        ocr=OCRInfo(median_confidence=None, page_count=len(doc.pages), unclear_document_flag=False),
    )


# ---------------------------------------------------------------------------
# EIS type detection
# ---------------------------------------------------------------------------

def test_eis_type_draft(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    folder = _make_doc_dir(tmp_path, pages=["DRAFT ENVIRONMENTAL IMPACT STATEMENT. Proposed project."])
    doc = load_document(folder)
    record = _make_record(doc)
    stage0_triage._eis_type(doc, record)
    assert record.eis_type == "Draft"


def test_eis_type_final(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    folder = _make_doc_dir(tmp_path, pages=["FINAL ENVIRONMENTAL IMPACT STATEMENT. Approved project."])
    doc = load_document(folder)
    record = _make_record(doc)
    stage0_triage._eis_type(doc, record)
    assert record.eis_type == "Final"


def test_eis_type_unlabelled_conflict(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    folder = _make_doc_dir(tmp_path, pages=["DRAFT FINAL ENVIRONMENTAL IMPACT STATEMENT"])
    doc = load_document(folder)
    record = _make_record(doc)
    stage0_triage._eis_type(doc, record)
    assert record.eis_type == "Unlabelled"


def test_eis_type_supplemental(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    folder = _make_doc_dir(tmp_path, pages=["SUPPLEMENTAL ENVIRONMENTAL IMPACT STATEMENT"])
    doc = load_document(folder)
    record = _make_record(doc)
    stage0_triage._eis_type(doc, record)
    assert record.eis_type == "Supplemental"


def test_eis_type_abbreviation(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    folder = _make_doc_dir(tmp_path, pages=["This is a DEIS for the proposed highway."])
    doc = load_document(folder)
    record = _make_record(doc)
    stage0_triage._eis_type(doc, record)
    assert record.eis_type == "Draft"


# ---------------------------------------------------------------------------
# Word count and length category
# ---------------------------------------------------------------------------

def test_length_short(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    short_text = " ".join(["word"] * 500)
    folder = _make_doc_dir(tmp_path, pages=[short_text])
    doc = load_document(folder)
    record = _make_record(doc)
    stage0_triage._word_count_and_length(doc, record)
    assert record.length_category == "short"
    assert record.word_count == 500


def test_length_long(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    long_text = " ".join(["word"] * 70_000)
    folder = _make_doc_dir(tmp_path, pages=[long_text])
    doc = load_document(folder)
    record = _make_record(doc)
    stage0_triage._word_count_and_length(doc, record)
    assert record.length_category == "long"


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

def test_date_year_extraction(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    # Repeat 1983 multiple times — it should win
    text = "This document prepared in 1983. 1983 study. Filed 1983."
    folder = _make_doc_dir(tmp_path, pages=[text])
    doc = load_document(folder)
    # Override METS date so we test extraction
    doc.mets.date = None
    doc.mets.agency = None
    record = _make_record(doc)
    warnings: list[str] = []
    stage0_triage._date_and_year(doc, record, warnings)
    assert record.year == 1983


def test_date_mets_wins(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    text = "Report from 1975."
    folder = _make_doc_dir(tmp_path, pages=[text])
    doc = load_document(folder)
    doc.mets.date = "1985"  # METS says 1985, text says 1975
    record = _make_record(doc)
    warnings: list[str] = []
    stage0_triage._date_and_year(doc, record, warnings)
    assert record.year == 1985
    # Mismatch > 1 year should generate a warning
    assert any("year_mismatch" in w for w in warnings)


def test_date_too_early_rejected(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    text = "Survey conducted in 1945."  # Pre-NEPA — should be ignored
    folder = _make_doc_dir(tmp_path, pages=[text])
    doc = load_document(folder)
    doc.mets.date = None
    record = _make_record(doc)
    warnings: list[str] = []
    stage0_triage._date_and_year(doc, record, warnings)
    assert record.year is None
    assert "year_not_found" in warnings


# ---------------------------------------------------------------------------
# TOC detection
# ---------------------------------------------------------------------------

def test_toc_detected(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    text = "Some preamble\n\nTable of Contents\n\nChapter 1 .............. 5\n"
    folder = _make_doc_dir(tmp_path, pages=[text, "Chapter content"])
    doc = load_document(folder)
    record = _make_record(doc)
    stage0_triage._headings_and_toc(doc, record)
    assert record.has_toc is True


def test_toc_not_in_late_page(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    # TOC only appears past 10% mark — should not set has_toc
    filler = " ".join(["word"] * 500)
    toc_page = "\n\nTable of Contents\n\nChapter 1 .............. 5\n"
    pages = [filler] * 10 + [toc_page]
    folder = _make_doc_dir(tmp_path, pages=pages)
    doc = load_document(folder)
    record = _make_record(doc)
    stage0_triage._headings_and_toc(doc, record)
    assert record.has_toc is False


# ---------------------------------------------------------------------------
# OCR confidence
# ---------------------------------------------------------------------------

def test_ocr_flag_set_for_low_confidence(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    conf = {"median_confidence": 0.65}
    folder = _make_doc_dir(tmp_path, pages=["text"], confidences=[conf])
    doc = load_document(folder)
    record = _make_record(doc)
    warnings: list[str] = []
    stage0_triage._ocr_confidence(doc, record, warnings)
    assert record.ocr.unclear_document_flag is True
    assert any("ocr_unclear" in w for w in warnings)


def test_ocr_flag_clear_for_good_confidence(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    conf = {"median_confidence": 0.95}
    folder = _make_doc_dir(tmp_path, pages=["text"], confidences=[conf])
    doc = load_document(folder)
    record = _make_record(doc)
    warnings: list[str] = []
    stage0_triage._ocr_confidence(doc, record, warnings)
    assert record.ocr.unclear_document_flag is False


# ---------------------------------------------------------------------------
# Heading detection — regressions against TEST_RUN_2 (Harlem Ave) failures
# ---------------------------------------------------------------------------

def test_is_real_heading_accepts_real_chapter():
    assert stage0_triage._is_real_heading("CHAPTER 1 INTRODUCTION AND PURPOSE")
    assert stage0_triage._is_real_heading("SECTION 4 ENVIRONMENTAL IMPACTS")
    assert stage0_triage._is_real_heading("APPENDIX A PUBLIC HEARING TRANSCRIPT")
    assert stage0_triage._is_real_heading("2.1 EXISTING CONDITIONS")


def test_is_real_heading_rejects_bare_legal_citation():
    # Harlem Ave was returning "Section 4" ~10 times from legal citations
    assert not stage0_triage._is_real_heading("Section 4")
    assert not stage0_triage._is_real_heading("Section 4(f)")
    assert not stage0_triage._is_real_heading("Section 102")


def test_is_real_heading_rejects_mailing_addresses():
    # "143 SOUTH THIRD STREET" was being treated as a heading
    assert not stage0_triage._is_real_heading("143 SOUTH THIRD STREET")
    assert not stage0_triage._is_real_heading("1234 PENNSYLVANIA AVENUE NW")
    assert not stage0_triage._is_real_heading("PHILADELPHIA PA 19103")


def test_is_real_heading_rejects_ocr_garbage():
    # Mostly non-alphabetic OCR fragments
    assert not stage0_triage._is_real_heading("5 / -- 6 ::")
    assert not stage0_triage._is_real_heading("1. 2. 3. 4.")


def test_headings_pattern_does_not_cross_newlines(tmp_path):
    # "5 F\nUNITED STATES DEPARTMENT..." was matching as a single heading
    # because the old regex used \s+ which includes newlines.
    from tests.test_io_layer import _make_doc_dir
    text = "Body text. 5 F\nUNITED STATES DEPARTMENT OF AGRICULTURE\nbody continues."
    folder = _make_doc_dir(tmp_path, pages=[text])
    doc = load_document(folder)
    record = _make_record(doc)
    stage0_triage._headings_and_toc(doc, record)
    # No real heading present, has_headings requires >=5
    assert not record.has_headings


# ---------------------------------------------------------------------------
# Full-name filter (spaCy PERSON sanitization)
# ---------------------------------------------------------------------------

def test_is_full_name_accepts_normal_names():
    assert stage0_triage._is_full_name("John Smith")
    assert stage0_triage._is_full_name("Mary J. Brown")
    assert stage0_triage._is_full_name("Henry N. Barkhausen")
    assert stage0_triage._is_full_name("Mary-Jane Watson-Parker")
    assert stage0_triage._is_full_name("Vincent van Gogh")
    assert stage0_triage._is_full_name("Leonardo da Vinci")
    assert stage0_triage._is_full_name("John Q. Public")


def test_is_full_name_rejects_single_token():
    assert not stage0_triage._is_full_name("Smith")
    assert not stage0_triage._is_full_name("John")


def test_is_full_name_rejects_all_caps():
    # "UNITED STATES" was bypassing the original regex because both tokens
    # were capitalized — the fixed filter requires lowercase letters in each
    # name component.
    assert not stage0_triage._is_full_name("UNITED STATES")
    assert not stage0_triage._is_full_name("USDA NRCS")


def test_is_full_name_rejects_lowercase():
    assert not stage0_triage._is_full_name("george marienthal")


# ---------------------------------------------------------------------------
# Lead agency
# ---------------------------------------------------------------------------

def test_lead_agency_from_mets(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    folder = _make_doc_dir(tmp_path, pages=["text"])
    doc = load_document(folder)
    # METS fixture sets agency to "Bureau of Land Management"
    record = _make_record(doc)
    warnings: list[str] = []
    stage0_triage._lead_agency(doc, record, warnings)
    assert record.lead_agency.name == "Bureau of Land Management"
    assert record.lead_agency.source == "mets"


def test_lead_agency_abbreviation_in_text(tmp_path):
    from tests.test_io_layer import _make_doc_dir
    folder = _make_doc_dir(tmp_path, pages=["Prepared by the EPA for review."])
    doc = load_document(folder)
    doc.mets.agency = None  # No METS agency
    record = _make_record(doc)
    warnings: list[str] = []
    stage0_triage._lead_agency(doc, record, warnings)
    assert record.lead_agency.name == "Environmental Protection Agency"
