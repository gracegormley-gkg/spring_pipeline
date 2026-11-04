"""Smoke tests for the locked schema.

Run from the merged_pipeline/ directory:
    pytest tests/test_schema.py -v
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline.schema import (
    CommentAuthor,
    CommentBlock,
    CriticVerdict,
    EISRecord,
    Quote,
    SectionRecord,
    SpanRecord,
    Stakeholder,
    StanceRecord,
    StanceTarget,
    ValidationField,
)


def test_empty_record_validates():
    r = EISRecord(publication_id="test_001")
    assert r.publication_id == "test_001"
    assert r.validation.verdicts == []
    assert r.cross_publication_links.status == "deferred_v1"
    assert r.stakeholder_extraction_scope == "organizations_only"
    assert r.excluded_stakeholder_types == ["person"]


def _make_block_with_span(block_id: str, span_id: str, *, is_comment_text: bool):
    return CommentBlock(
        block_id=block_id,
        char_span_raw=(4000, 5000),
        pages=[213],
        spans=[
            SpanRecord(
                span_id=span_id,
                char_span_raw=(4821, 4912),
                page=213,
                is_comment_text=is_comment_text,
            ),
        ],
    )


def _make_stakeholder(block_id: str, span_id: str | None, *, quote_section="public_comments"):
    quote = None
    if span_id is not None:
        quote = Quote(
            text_raw="We object to the proposed alignment.",
            text_display="We object to the proposed alignment.",
            char_offset_raw=(4821, 4857),
            page=213,
            section=quote_section,
            source_text_hash="sha256:abc",
            span_id=span_id,
        )
    return Stakeholder(
        comment_block_id=block_id,
        comment_author=CommentAuthor(name="Sierra Club", type="organization"),
        stance_records=[
            StanceRecord(
                stance="opposed",
                stance_target=StanceTarget(type="proposed_action"),
                stance_confidence=0.86,
                quote=quote,
            ),
        ],
    )


def test_populated_record_with_span_id_quote():
    block = _make_block_with_span("block_42", "block_42_s1", is_comment_text=True)
    sh = _make_stakeholder("block_42", "block_42_s1")
    r = EISRecord(
        publication_id="test_002",
        sections=[SectionRecord(name="public_comments", char_span=(0, 10000), pages=(1, 30))],
        comment_blocks=[block],
        stakeholders=[sh],
        validation=ValidationField(verdicts=[
            CriticVerdict(field_path="title", tier="haiku", verdict="pass"),
            CriticVerdict(field_path="summary.value", tier="sonnet", verdict="partial",
                          reasoning="one claim not grounded"),
        ]),
    )
    assert len(r.validation.verdicts) == 2
    assert r.stakeholders[0].stance_records[0].quote.span_id == "block_42_s1"


def test_stakeholder_block_id_must_match_an_existing_block():
    sh = _make_stakeholder("block_does_not_exist", None)
    with pytest.raises(ValidationError):
        EISRecord(
            publication_id="test_003",
            sections=[SectionRecord(name="public_comments", char_span=(0, 10000), pages=(1, 30))],
            comment_blocks=[_make_block_with_span("block_real", "block_real_s1", is_comment_text=True)],
            stakeholders=[sh],
        )


def test_stance_quote_cannot_come_from_agency_response_span():
    """The killer validator: a quote whose span_id resolves to is_comment_text=False
    must be rejected — that's an agency response, not a stakeholder utterance.
    Synthesis_plan.md §Stakeholders — comment/response split."""
    block = _make_block_with_span("block_99", "block_99_s1", is_comment_text=False)
    sh = _make_stakeholder("block_99", "block_99_s1", quote_section="response_to_comments")
    with pytest.raises(ValidationError, match="agency-response span"):
        EISRecord(
            publication_id="test_004",
            sections=[SectionRecord(name="response_to_comments", char_span=(0, 1000), pages=(1, 2))],
            comment_blocks=[block],
            stakeholders=[sh],
        )


def test_stance_quote_span_id_must_belong_to_block():
    block = _make_block_with_span("block_42", "block_42_s1", is_comment_text=True)
    sh = _make_stakeholder("block_42", "wrong_span_id")
    with pytest.raises(ValidationError, match="not in block"):
        EISRecord(
            publication_id="test_005",
            sections=[SectionRecord(name="public_comments", char_span=(0, 10000), pages=(1, 30))],
            comment_blocks=[block],
            stakeholders=[sh],
        )


def test_subtheme_parent_must_match_a_chosen_primary():
    from pipeline.schema import ThemesField, ThemeEntry, SubthemeEntry
    with pytest.raises(ValidationError, match="not in chosen primaries"):
        ThemesField(
            primary=[ThemeEntry(value="transportation", confidence=0.9)],
            subthemes=[SubthemeEntry(value="nuclear_power", confidence=0.8, parent="energy_infrastructure")],
        )


def test_cross_publication_links_must_be_deferred_in_v1():
    from pipeline.schema import DeferredCrossLinks
    cpl = DeferredCrossLinks()
    assert cpl.status == "deferred_v1"
    # Schema enforces this via the EISRecord validator; field uses a fixed Literal
