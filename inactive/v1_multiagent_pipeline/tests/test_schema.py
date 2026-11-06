"""
Schema tests — verify Pydantic v2 model construction, validators, and the
deferred-v1 invariants per build brief §4.

Run: pytest tests/test_schema.py
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline.schema import (
    Alternative,
    CommentAuthor,
    CommentBlock,
    Component,
    DeferredCrossLinks,
    DeferredField,
    EISRecord,
    FieldWithStatus,
    Provenance,
    Quote,
    SectionRecord,
    Stakeholder,
    StanceRecord,
    StanceTarget,
    SubthemeEntry,
    ThemeEntry,
    ThemesField,
)


# ---------------------------------------------------------------------------
# Minimal valid record — sanity check the defaults
# ---------------------------------------------------------------------------

def test_minimal_record_validates() -> None:
    rec = EISRecord(publication_id="p1074_test")
    # Defaults should produce a valid model
    assert rec.publication_id == "p1074_test"
    assert rec.cross_publication_links.status == "deferred_v1"
    assert rec.historical_context.status == "deferred_v1"
    assert rec.project_status.status == "deferred_v1"
    assert rec.stakeholder_extraction_scope == "organizations_only"
    assert rec.excluded_stakeholder_types == ["person"]


# ---------------------------------------------------------------------------
# cross_publication_links must be deferred_v1
# ---------------------------------------------------------------------------

def test_cross_publication_links_must_be_deferred() -> None:
    """Trying to set status to anything other than 'deferred_v1' must fail."""
    with pytest.raises(ValidationError):
        EISRecord(
            publication_id="x",
            cross_publication_links=DeferredCrossLinks.model_validate(
                {"value": [], "status": "ok"}
            ),
        )


# ---------------------------------------------------------------------------
# Quote offset validation
# ---------------------------------------------------------------------------

def test_quote_offsets_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        Quote(
            text_raw="x",
            text_display="x",
            char_offset_raw=(100, 50),  # end < start
            page=1,
            section="public_comments",
            source_text_hash="sha256:dead",
        )


def test_quote_offsets_non_negative() -> None:
    with pytest.raises(ValidationError):
        Quote(
            text_raw="x",
            text_display="x",
            char_offset_raw=(-1, 5),
            page=1,
            section="public_comments",
            source_text_hash="sha256:dead",
        )


# ---------------------------------------------------------------------------
# Stance quote.section must be public_comments / response_to_comments
# ---------------------------------------------------------------------------

def test_stance_quote_section_constraint() -> None:
    """Quote.section is a Literal — invalid sections fail at quote construction."""
    with pytest.raises(ValidationError):
        Quote(
            text_raw="x",
            text_display="x",
            char_offset_raw=(0, 5),
            page=1,
            section="summary",  # type: ignore[arg-type]
            source_text_hash="sha256:dead",
        )


# ---------------------------------------------------------------------------
# Stakeholder.comment_block_id must reference a comment_blocks[] entry
# ---------------------------------------------------------------------------

def test_stakeholder_block_id_must_match_comment_blocks() -> None:
    rec_kwargs = {
        "publication_id": "p1074_test",
        "comment_blocks": [
            CommentBlock(block_id="block_1", char_span_raw=(0, 100), pages=[5])
        ],
        "stakeholders": [
            Stakeholder(
                comment_block_id="block_999",  # not in comment_blocks
                comment_author=CommentAuthor(name="Sierra Club", type="organization"),
                authorship_role="primary_author",
            )
        ],
    }
    with pytest.raises(ValidationError):
        EISRecord(**rec_kwargs)


def test_stakeholder_block_id_match_passes() -> None:
    rec = EISRecord(
        publication_id="p1074_test",
        comment_blocks=[CommentBlock(block_id="block_1", char_span_raw=(0, 100), pages=[5])],
        stakeholders=[
            Stakeholder(
                comment_block_id="block_1",
                comment_author=CommentAuthor(name="Sierra Club", type="organization"),
                authorship_role="primary_author",
            )
        ],
    )
    assert rec.stakeholders[0].comment_author.name == "Sierra Club"


# ---------------------------------------------------------------------------
# Subtheme parent must be in chosen primaries
# ---------------------------------------------------------------------------

def test_subtheme_parent_must_be_chosen_primary() -> None:
    with pytest.raises(ValidationError):
        ThemesField(
            primary=[ThemeEntry(value="transportation", confidence=0.9)],
            subthemes=[
                SubthemeEntry(value="nuclear_power", confidence=0.8, parent="energy_infrastructure")
            ],
        )


def test_subtheme_parent_match_passes() -> None:
    tf = ThemesField(
        primary=[ThemeEntry(value="transportation", confidence=0.9)],
        subthemes=[SubthemeEntry(value="highway", confidence=0.8, parent="transportation")],
    )
    assert tf.primary[0].value == "transportation"


# ---------------------------------------------------------------------------
# Provenance.section must reference a sections[].name (when sections present)
# ---------------------------------------------------------------------------

def test_provenance_section_must_match_sections() -> None:
    """If a provenance points to 'rod' but rod isn't in sections, validation fails."""
    rec_kwargs = {
        "publication_id": "p1074_test",
        "sections": [SectionRecord(name="cover", char_span=(0, 100), pages=(1, 1))],
        "title": FieldWithStatus[str](
            value="x",
            provenance=Provenance(source="regex", section="rod"),  # rod not in sections
        ),
    }
    with pytest.raises(ValidationError):
        EISRecord(**rec_kwargs)


def test_provenance_section_match_passes() -> None:
    rec = EISRecord(
        publication_id="p1074_test",
        sections=[SectionRecord(name="cover", char_span=(0, 100), pages=(1, 1))],
        title=FieldWithStatus[str](
            value="x",
            provenance=Provenance(source="regex", section="cover"),
        ),
    )
    assert rec.title.value == "x"


# ---------------------------------------------------------------------------
# project_area_polygon requires a project_area NamedPlace with polygon
# ---------------------------------------------------------------------------

def test_polygon_requires_named_place() -> None:
    from pipeline.schema import LocationField, NamedPlace

    # project_area_polygon set, but no named_place with role=project_area+polygon
    with pytest.raises(ValidationError):
        LocationField(
            named_places=[NamedPlace(name="X NF", role="context_reference")],
            project_area_polygon={"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}},
        )


def test_polygon_with_matching_named_place_passes() -> None:
    from pipeline.schema import LocationField, NamedPlace

    poly = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}}
    loc = LocationField(
        named_places=[NamedPlace(name="X NF", role="project_area", polygon=poly)],
        project_area_polygon=poly,
    )
    assert loc.project_area_polygon is not None


# ---------------------------------------------------------------------------
# Section char_span ordering
# ---------------------------------------------------------------------------

def test_section_invalid_char_span_fails() -> None:
    with pytest.raises(ValidationError):
        SectionRecord(name="cover", char_span=(100, 50), pages=(1, 2))


def test_section_status_ok_requires_char_span() -> None:
    with pytest.raises(ValidationError):
        SectionRecord(name="cover", char_span=None, status="ok")
