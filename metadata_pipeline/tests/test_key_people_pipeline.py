"""
Tests for segment_b/postproc/key_people_pipeline.py (MCAL_PLAN 3.10 / 4 Q3, item #7).

No network and no LLM calls: every Sonnet call goes through the module-level
`sonnet` name and is monkeypatched.

The load-bearing regression is TestHeadingWhitelist: a bare "Consultation and
Coordination" heading must NOT license a cooperating-agency extraction. That
single behaviour is the 5/8 failure in MCAL_PLAN 1(10), and every real graded doc
in this corpus exercises the no-whitelisted-heading path, so the fallback +
HUMAN_REVIEW route is tested against real documents too.
"""

from __future__ import annotations

import json

import pytest

from mcal import settings
from segment_b.postproc import key_people_pipeline as kp

from conftest import BUFFALO, LINCOLN_HWY, OPERATION_BREAKTHROUGH

# MCAL_PLAN 1(9d)/1(1): the CAFE rulemaking whose `year` Critic verdict is
# HUMAN_REVIEW (NUL says 1977, the regex says 1979) -- the live dependent-field
# cascade case. Not in conftest's constant set, so it is named here.
FUEL_ECONOMY = "p1074_35556036861797"


# --- fake Sonnet ------------------------------------------------------------


def fake_sonnet(**by_marker):
    """
    Dispatch a fake Sonnet response on a marker in the system prompt.

    Markers: preparers, cooperating, fallback, consulted, commenters.
    An absent marker returns {}, i.e. "the model found nothing".
    """
    markers = (
        ("fallback", "formally designated"),
        ("cooperating", "COOPERATING AGENCIES from a section"),
        ("preparers", "who PREPARED"),
        ("consulted", "role-tag the entities"),
        ("commenters", "PUBLIC COMMENTERS"),
    )

    def _fake(system, user, **kw):
        for key, needle in markers:
            if needle in system:
                value = by_marker.get(key, {})
                if callable(value):
                    return value(user)
                return value
        return {}

    return _fake


def no_llm(monkeypatch):
    """Every Sonnet call is a hard failure."""

    def _boom(system, user, **kw):
        raise AssertionError("no LLM call expected here")

    monkeypatch.setattr(kp, "sonnet", _boom)


# --- heading whitelist (the 5/8 bug) ----------------------------------------


class TestHeadingWhitelist:
    """MCAL_PLAN 3.10 step 3 / 4 Q3."""

    @pytest.mark.parametrize(
        "heading",
        [
            "Consultation and Coordination",
            "CONSULTATION AND COORDINATION",
            "5.0 Consultation and Coordination",
            "Consultation and Coordination with Others",
            "List of Persons Consulted",
            "LIST OF AGENCIES AND PERSONS CONSULTED",
            "Agencies Consulted",
            "Coordination",
            "Distribution List",
            "List of Preparers",
            "Comments and Coordination",
            # Bundled heading: names cooperating AND consulted agencies, so it is
            # not authority for a 1501.8 designation. Rejecting it costs one
            # fallback call; accepting it re-creates the 5/8 failure.
            "Cooperating and Consulted Agencies",
        ],
    )
    def test_does_not_match_cooperating_whitelist(self, heading):
        assert kp.match_heading(heading, kp.COOPERATING_HEADINGS) is None

    @pytest.mark.parametrize(
        "heading,phrase",
        [
            ("COOPERATING AGENCIES", "cooperating agencies"),
            ("Cooperating Agencies", "cooperating agencies"),
            ("5.2 Cooperating Agencies", "cooperating agencies"),
            ("IV. Cooperating Agencies", "cooperating agencies"),
            ("A. Cooperating Agencies", "cooperating agencies"),
            ("CHAPTER 5 COOPERATING AGENCIES", "cooperating agencies"),
            ("Cooperating Agency", "cooperating agencies"),
            ("Coordination with Cooperating Agencies", "cooperating agencies"),
            ("Joint Lead Agencies", "joint lead agencies"),
            ("ASSISTING AGENCIES", "assisting agencies"),
        ],
    )
    def test_matches_cooperating_whitelist(self, heading, phrase):
        hit = kp.match_heading(heading, kp.COOPERATING_HEADINGS)
        assert hit is not None
        assert hit.matched_phrase == phrase

    def test_ocr_damage_still_matches(self):
        """OCR-normalized, per the plan: 0/O and 1/l confusions must not break it."""
        assert kp.match_heading("C00PERATING AGENCIES", kp.COOPERATING_HEADINGS)
        assert kp.match_heading("J01NT LEAD AGENC1ES", kp.COOPERATING_HEADINGS)

    @pytest.mark.parametrize(
        "heading,expected",
        [
            ("Comments Received", True),
            ("RESPONSE TO COMMENTS", True),
            ("Responses to Comments", True),
            ("7.0 Public Hearing Transcripts", True),
            ("Consultation and Coordination", False),
            ("Comments and Coordination", False),
            ("List of Preparers", False),
        ],
    )
    def test_comment_whitelist(self, heading, expected):
        assert bool(kp.match_heading(heading, kp.COMMENT_HEADINGS)) is expected

    def test_roman_numeral_stripper_does_not_eat_the_leading_c(self):
        """
        Regression: a marker pattern of `[ivxlcdm]+` matches the "C" of
        "Cooperating" and silently drops the ratio below threshold. Markers must
        require trailing punctuation.
        """
        assert kp.strip_heading_decoration("Cooperating Agency") == "Cooperating Agency"
        assert kp.strip_heading_decoration("IV. Cooperating Agencies") == "Cooperating Agencies"
        assert kp.strip_heading_decoration("C. Cooperating Agencies") == "Cooperating Agencies"

    def test_toc_decoration_is_stripped(self):
        assert (
            kp.strip_heading_decoration("Cooperating Agencies ......... 5-3")
            == "Cooperating Agencies"
        )
        assert kp.strip_heading_decoration("5.2 Cooperating Agencies 5-3") == "Cooperating Agencies"

    def test_prose_length_guard(self):
        long_line = (
            "Cooperating agencies were invited to participate in the preparation "
            "of this statement and their comments are reproduced in appendix B "
            "together with the responses of the lead agency."
        )
        assert kp.match_heading(long_line, kp.COOPERATING_HEADINGS) is None

    @pytest.mark.parametrize(
        "line,expected",
        [
            ("COOPERATING AGENCIES", True),
            ("Cooperating Agencies", True),
            ("5.2 Cooperating Agencies", True),
            # Real prose from the Fuel Economy scan that fuzzy-matches a whitelist
            # phrase but is body text.
            ("Comments to the proposed fuel economy have been received and", False),
            ("cooperating agencies were consulted about the project", False),
            ("Cooperating Agencies,", False),
            ("", False),
        ],
    )
    def test_looks_like_heading_line(self, line, expected):
        assert kp.looks_like_heading_line(line) is expected


class TestFindSections:
    def test_finds_body_section_with_pages(self, doc_factory):
        doc = doc_factory(
            "TABLE OF CONTENTS\nCooperating Agencies ......... 5-1\n" + "x" * 3000,
            "COOPERATING AGENCIES\nThe Bureau of Reclamation is a cooperating agency.\n",
            "5.3 OTHER MATTERS\nUnrelated text.\n",
        )
        sections = kp.find_sections(doc, kp.COOPERATING_HEADINGS)
        assert len(sections) == 1
        assert sections[0].heading == "COOPERATING AGENCIES"
        assert sections[0].start_page == 2
        assert "Bureau of Reclamation" in sections[0].text

    def test_toc_row_is_skipped(self, doc_factory):
        """A dot-leader row proves a section exists; it is not that section."""
        doc = doc_factory(
            "x" * 3100 + "\nCooperating Agencies ................ 5-1\nMore contents\n"
        )
        assert kp.find_sections(doc, kp.COOPERATING_HEADINGS) == []

    def test_front_matter_is_skipped(self, doc_factory):
        doc = doc_factory("COOPERATING AGENCIES\nsomething\n")
        assert kp.find_sections(doc, kp.COOPERATING_HEADINGS) == []
        assert kp.find_sections(doc, kp.COOPERATING_HEADINGS, skip_toc_chars=0)

    def test_section_ends_at_the_next_numbered_heading(self, doc_factory):
        body = "The Corps of Engineers is a cooperating agency.\n" * 10
        doc = doc_factory(
            "y" * 3100,
            "COOPERATING AGENCIES\n" + body + "6.0 LIST OF PREPARERS\nJane Doe\n",
        )
        section = kp.find_sections(doc, kp.COOPERATING_HEADINGS)[0]
        assert "Corps of Engineers" in section.text
        assert "LIST OF PREPARERS" not in section.text

    def test_repeated_running_header_is_not_a_second_section(self, doc_factory):
        doc = doc_factory(
            "z" * 3100,
            "COOPERATING AGENCIES\n" + ("filler line\n" * 5) + "COOPERATING AGENCIES\nmore\n",
        )
        assert len(kp.find_sections(doc, kp.COOPERATING_HEADINGS)) == 1

    def test_consultation_chapter_never_yields_a_cooperating_section(self, doc_factory):
        """The catch-all chapter is not authority. This is the 5/8 bug, end to end."""
        doc = doc_factory(
            "w" * 3100,
            "CONSULTATION AND COORDINATION\n"
            "The following agencies were consulted: Environmental Protection Agency, "
            "Fish and Wildlife Service.\n"
            "Copies of the draft were sent to the Buffalo Public Library.\n",
        )
        assert kp.find_sections(doc, kp.COOPERATING_HEADINGS) == []
        assert kp.find_sections(doc, kp.CONSULTATION_HEADINGS)

    def test_limit_is_respected(self, doc_factory):
        chunk = "COOPERATING AGENCIES\n" + ("body\n" * 3) + "9.9 END\n"
        doc = doc_factory("q" * 3100, chunk * 5)
        assert len(kp.find_sections(doc, kp.COOPERATING_HEADINGS, limit=2)) == 2

    def test_empty_document_degrades(self, doc_factory):
        assert kp.find_sections(doc_factory(""), kp.COOPERATING_HEADINGS) == []


# --- era gate ---------------------------------------------------------------


class TestEraGate:
    """MCAL_PLAN 3.10 step 2 + settings.DEPENDENT_FIELDS."""

    def test_config_declares_the_dependency(self):
        assert settings.DEPENDENT_FIELDS["year"] == ["key_people"]
        assert kp.KEY_PEOPLE_DEPENDS_ON_YEAR is True

    @pytest.mark.parametrize("verdict", ["RE_EXTRACT", "HUMAN_REVIEW"])
    def test_untrustworthy_year_gates_the_whole_field(self, verdict):
        gate = kp.apply_era_gate(1985, verdict)
        assert gate.field_human_review is True
        assert gate.cooperating_human_review is True
        assert gate.tags == []          # not a pre-1978 problem
        assert "dependent-field cascade" in gate.reason
        # The era is unknown, so the designation-check fallback is suppressed --
        # but a whitelisted heading is still era-independent evidence.
        assert gate.allow_cooperating_fallback is False
        assert gate.skip_cooperating_extraction is False

    @pytest.mark.parametrize("verdict", [None, "", "UNKNOWN", "pass_maybe"])
    def test_missing_or_unknown_verdict_is_untrustworthy(self, verdict):
        gate = kp.apply_era_gate(1985, verdict)
        assert gate.field_human_review is True

    @pytest.mark.parametrize("verdict", ["PASS", "PASS_WITH_NOTE", "pass"])
    def test_pre_1978_routes_only_cooperating_with_T13(self, verdict):
        gate = kp.apply_era_gate(1974, verdict)
        assert gate.field_human_review is False
        assert gate.cooperating_human_review is True
        assert gate.skip_cooperating_extraction is True
        assert gate.allow_cooperating_fallback is False
        assert gate.tags == [kp.T_PRE_1978]
        assert "1501.8" in gate.reason

    def test_1978_itself_is_post_schema(self):
        gate = kp.apply_era_gate(1978, "PASS")
        assert gate.cooperating_human_review is False
        assert gate.tags == []

    def test_post_1978_is_clean(self):
        gate = kp.apply_era_gate(1996, "PASS_WITH_NOTE")
        assert (gate.field_human_review, gate.cooperating_human_review) == (False, False)

    def test_trustworthy_verdict_but_no_year_cannot_era_gate(self):
        gate = kp.apply_era_gate(None, "PASS")
        assert gate.field_human_review is False
        assert gate.cooperating_human_review is True
        assert gate.tags == []          # we do not know it is pre-1978
        assert "cannot run" in gate.reason

    def test_unparseable_year(self):
        gate = kp.apply_era_gate("nineteen seventy four", "PASS")
        assert gate.year is None
        assert gate.cooperating_human_review is True

    def test_to_dict_carries_the_config(self):
        got = kp.apply_era_gate(1980, "PASS").to_dict()
        assert got["dependent_fields_config"] == {"year": ["key_people"]}

    def test_year_signal_from_segment_a_shapes(self):
        m1 = {"year": {"value": 1977, "confidence": "low"}}
        critic = {"year": {"verdict": "HUMAN_REVIEW"}}
        assert kp.year_signal_from_artifacts(m1, critic) == (1977, "HUMAN_REVIEW")

    def test_year_signal_tolerates_absent_artifacts(self):
        assert kp.year_signal_from_artifacts(None, None) == (None, None)
        assert kp.year_signal_from_artifacts({"year": {}}, {"year": {}}) == (None, None)
        assert kp.year_signal_from_artifacts({"year": "1980"}, {}) == (1980, None)


# --- capacity classification ------------------------------------------------


class TestClassifyCapacity:
    """MCAL_PLAN 3.5 operational definition of "private individual"."""

    def test_no_institutional_identification_is_private(self):
        got = kp.classify_capacity(
            "Robert Johnson",
            "Mr. Robert Johnson stated that the noise from the proposed freeway "
            "would harm his neighborhood.",
        )
        assert got.capacity == "private"
        assert got.requires_human_review is True

    @pytest.mark.parametrize(
        "passage",
        [
            "Dr. Jane Smith questioned the air-quality analysis.",
            "Jane Smith, Director of the State Water Commission, supported it.",
            "Mayor Jane Smith opposed the alignment.",
            "Council Member Jane Smith asked about relocation.",
            "Jane Smith, speaking on behalf of the Audubon Society, objected.",
            "Jane Smith, Chairman of the Tribal Council, requested consultation.",
            "Secretary Jane Smith transmitted the comments.",
            "Jane Smith, an engineer for the Department of Transportation, testified.",
        ],
    )
    def test_titles_and_affiliations_are_non_private(self, passage):
        got = kp.classify_capacity("Jane Smith", passage)
        assert got.capacity == "non_private"
        assert got.requires_human_review is False
        assert got.non_private_cues

    def test_dual_capacity_is_ambiguous_and_human_review(self):
        """
        MCAL_PLAN 3.5: capacity binds to the passage at the point of stance
        attribution; an ambiguous passage is HUMAN_REVIEW regardless of Critic.
        """
        got = kp.classify_capacity(
            "Ann Lee",
            "Mayor Ann Lee, who is also a resident of the affected neighborhood, "
            "opposed the alignment.",
        )
        assert got.capacity == "ambiguous"
        assert got.basis == "dual_capacity_cues_in_cited_passage"
        assert got.requires_human_review is True

    def test_same_person_two_passages_two_verdicts(self):
        """Dual capacity across chapters: each stance is judged on its own passage."""
        official = kp.classify_capacity(
            "Ann Lee", "Mayor Ann Lee submitted the city's official comments."
        )
        personal = kp.classify_capacity(
            "Ann Lee", "Ann Lee said the noise keeps her children awake at night."
        )
        assert official.capacity == "non_private"
        assert personal.capacity == "private"

    def test_organization_is_never_private(self):
        got = kp.classify_capacity("Sierra Club", "The Sierra Club submitted comments.")
        assert got.capacity == "non_private"
        assert got.basis == "entity_name_is_an_organization"

    def test_agency_name_is_never_private(self):
        got = kp.classify_capacity(
            "New York State Department of Transportation", "The department objected."
        )
        assert got.capacity == "non_private"

    def test_name_absent_from_passage_is_ambiguous(self):
        got = kp.classify_capacity("Bob Nobody", "This passage names no one at all.")
        assert got.capacity == "ambiguous"
        assert got.basis == "name_not_found_in_cited_passage"
        assert got.requires_human_review is True

    def test_surname_only_mention_is_located(self):
        got = kp.classify_capacity(
            "Robert A. Johnson", "Mr. Johnson objected to the relocation plan."
        )
        assert got.name_located is True

    def test_empty_name(self):
        assert kp.classify_capacity("", "anything").capacity == "ambiguous"

    def test_cue_window_is_local_to_the_name(self):
        """
        A far-away agency mention must not make a private commenter non-private.
        That is the mislabeling failure in a different costume.
        """
        passage = (
            "The Environmental Protection Agency reviewed the draft statement. "
            + ("filler sentence about air quality. " * 60)
            + "Robert Johnson said the project would flood his pasture."
        )
        got = kp.classify_capacity("Robert Johnson", passage)
        assert got.capacity == "private"

    def test_extractor_kind_never_overrides_the_passage(self):
        got = kp.classify_capacity(
            "Robert Johnson",
            "Robert Johnson said the project would flood his pasture.",
            kind="official",
        )
        assert got.capacity == "private"

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Robert Johnson", True),
            ("Dr. Jane A. Smith", True),
            ("Sierra Club", False),
            ("Department of the Interior", False),
            ("Seneca Nation of Indians", False),
            ("", False),
        ],
    )
    def test_person_vs_organization(self, name, expected):
        assert kp.looks_like_person_name(name) is expected

    def test_to_dict_is_auditable(self):
        got = kp.classify_capacity(
            "Jane Smith", "Dr. Jane Smith questioned the analysis."
        ).to_dict()
        assert got["capacity"] == "non_private"
        assert got["requires_human_review"] is False
        assert any("dr." in c for c in got["non_private_cues"])


# --- cooperating agencies ---------------------------------------------------

COOP_DOC = (
    "a" * 3100,
    "COOPERATING AGENCIES\n"
    "The Bureau of Reclamation is a cooperating agency for this statement.\n"
    "9.9 NEXT\n",
    "CONSULTATION AND COORDINATION\n"
    "The Environmental Protection Agency was consulted. Copies of the draft were "
    "sent to the Buffalo Public Library.\n",
)

CONSULTATION_ONLY_DOC = (
    "b" * 3100,
    "CONSULTATION AND COORDINATION\n"
    "The Environmental Protection Agency was consulted. The Seneca Nation of "
    "Indians was contacted. Copies of the draft were sent to the Buffalo Public "
    "Library and to Mayor Ann Lee.\n",
)


class TestCooperatingAgencies:
    def test_extracted_only_from_a_whitelisted_heading(self, doc_factory, monkeypatch):
        seen: list[str] = []

        def coop(user):
            seen.append(user)
            return {
                "cooperating_agencies": [
                    {
                        "name": "Bureau of Reclamation",
                        "designation_phrase": "is a cooperating agency",
                        "quote": "The Bureau of Reclamation is a cooperating agency",
                    }
                ]
            }

        monkeypatch.setattr(kp, "sonnet", fake_sonnet(cooperating=coop))
        doc = doc_factory(*COOP_DOC)
        entries, meta = kp.extract_cooperating_agencies(doc, [])

        assert meta["source"] == "heading_whitelist"
        assert meta["human_review"] is False
        assert [e["name"] for e in entries] == ["Bureau of Reclamation"]
        assert entries[0]["authority"] == "heading_whitelist"
        assert entries[0]["evidence"][0]["quote_verified"] is True
        # The prompt saw the whitelisted section only, never the catch-all chapter.
        assert "Bureau of Reclamation" in seen[0]
        assert "Buffalo Public Library" not in seen[0]

    def test_no_whitelisted_heading_uses_the_fallback_and_human_review(
        self, doc_factory, monkeypatch
    ):
        monkeypatch.setattr(
            kp,
            "sonnet",
            fake_sonnet(
                fallback={
                    "answer": "uncertain",
                    "cooperating_agencies": [],
                    "reasoning": "The text lists agencies consulted, not designated.",
                }
            ),
        )
        entries, meta = kp.extract_cooperating_agencies(
            doc_factory(*CONSULTATION_ONLY_DOC), []
        )
        assert entries == []
        assert meta["source"] == "sonnet_fallback"
        assert meta["human_review"] is True
        assert meta["fallback_answer"] == "uncertain"

    def test_fallback_no_answer_is_empty(self, doc_factory, monkeypatch):
        monkeypatch.setattr(kp, "sonnet", fake_sonnet(fallback={"answer": "no"}))
        entries, meta = kp.extract_cooperating_agencies(
            doc_factory(*CONSULTATION_ONLY_DOC), []
        )
        assert entries == []
        assert meta["human_review"] is True

    def test_fallback_yes_keeps_entities_but_still_human_review(
        self, doc_factory, monkeypatch
    ):
        monkeypatch.setattr(
            kp,
            "sonnet",
            fake_sonnet(
                fallback={
                    "answer": "yes",
                    "cooperating_agencies": [
                        {
                            "name": "Environmental Protection Agency",
                            "quote": "The Environmental Protection Agency was consulted",
                        }
                    ],
                    "reasoning": "designated in the letter of transmittal",
                }
            ),
        )
        entries, meta = kp.extract_cooperating_agencies(
            doc_factory(*CONSULTATION_ONLY_DOC), []
        )
        assert [e["name"] for e in entries] == ["Environmental Protection Agency"]
        assert entries[0]["authority"] == "sonnet_fallback"
        assert meta["human_review"] is True

    def test_fallback_llm_failure_degrades(self, doc_factory, monkeypatch):
        def explode(system, user, **kw):
            raise RuntimeError("model down")

        monkeypatch.setattr(kp, "sonnet", explode)
        entries, meta = kp.extract_cooperating_agencies(
            doc_factory(*CONSULTATION_ONLY_DOC), []
        )
        assert entries == []
        assert meta["extractor_ok"] is False
        assert meta["fallback_answer"] == "unavailable"

    def test_junk_llm_output_degrades(self, doc_factory, monkeypatch):
        monkeypatch.setattr(kp, "sonnet", lambda system, user, **kw: ["nope"])
        entries, meta = kp.extract_cooperating_agencies(doc_factory(*COOP_DOC), [])
        assert entries == []
        assert meta["extractor_ok"] is False

    def test_era_gate_skip_makes_no_llm_call(self, doc_factory, monkeypatch):
        no_llm(monkeypatch)
        gate = kp.apply_era_gate(1974, "PASS")
        entries, meta = kp.extract_cooperating_agencies(
            doc_factory(*COOP_DOC), [], era_gate=gate
        )
        assert entries == []
        assert meta["source"] == "skipped_by_era_gate"
        assert meta["human_review"] is True

    def test_duplicate_names_are_deduped(self, doc_factory, monkeypatch):
        monkeypatch.setattr(
            kp,
            "sonnet",
            fake_sonnet(
                cooperating={
                    "cooperating_agencies": [
                        {"name": "Bureau of Reclamation", "quote": ""},
                        {"name": "BUREAU OF RECLAMATION", "quote": ""},
                    ]
                }
            ),
        )
        entries, _ = kp.extract_cooperating_agencies(doc_factory(*COOP_DOC), [])
        assert len(entries) == 1


class TestConsultedEntities:
    def test_role_tagging_and_enum_coercion(self, doc_factory, monkeypatch):
        monkeypatch.setattr(
            kp,
            "sonnet",
            fake_sonnet(
                consulted={
                    "consulted_entities": [
                        {
                            "name": "Environmental Protection Agency",
                            "role": "consulted_agency",
                            "quote": "The Environmental Protection Agency was consulted",
                        },
                        {
                            "name": "Seneca Nation of Indians",
                            "role": "tribe",
                            "quote": "The Seneca Nation of Indians was contacted",
                        },
                        {
                            "name": "Buffalo Public Library",
                            "role": "recipient_of_draft",
                            "quote": "Copies of the draft were sent to the Buffalo Public Library",
                        },
                        # A model that ignores instructions and tries to
                        # reintroduce the label must not be able to.
                        {"name": "Rogue Agency", "role": "cooperator", "quote": ""},
                    ]
                }
            ),
        )
        doc = doc_factory(*CONSULTATION_ONLY_DOC)
        entries, meta = kp.extract_consulted_entities(doc, [])
        roles = {e["name"]: e["role"] for e in entries}
        assert roles["Environmental Protection Agency"] == "consulted_agency"
        assert roles["Seneca Nation of Indians"] == "tribe"
        assert roles["Buffalo Public Library"] == "recipient_of_draft"
        assert roles["Rogue Agency"] == "other"
        assert all(e["role"] in kp.CONSULTED_ROLES for e in entries)
        assert all("cooperat" not in e["role"] for e in entries)

    def test_cooperating_agencies_are_excluded(self, doc_factory, monkeypatch):
        monkeypatch.setattr(
            kp,
            "sonnet",
            fake_sonnet(
                consulted={
                    "consulted_entities": [
                        {"name": "Bureau of Reclamation", "role": "consulted_agency"},
                        {"name": "Environmental Protection Agency", "role": "consulted_agency"},
                    ]
                }
            ),
        )
        entries, meta = kp.extract_consulted_entities(
            doc_factory(*COOP_DOC), [], exclude_names=["BUREAU OF RECLAMATION"]
        )
        assert [e["name"] for e in entries] == ["Environmental Protection Agency"]

    def test_no_consultation_chapter_makes_no_llm_call(self, doc_factory, monkeypatch):
        no_llm(monkeypatch)
        entries, meta = kp.extract_consulted_entities(doc_factory("c" * 3100), [])
        assert entries == []
        assert meta["source"] == "not_found"

    def test_ceq_chapter_is_preferred_over_heading_scan(self, doc_factory, monkeypatch):
        monkeypatch.setattr(kp, "sonnet", fake_sonnet(consulted={"consulted_entities": []}))
        doc = doc_factory(*CONSULTATION_ONLY_DOC)
        chapters = [
            {
                "label": "CONSULTATION AND COORDINATION",
                "ceq_chapter": "Consultation",
                "start_char": 3100,
                "end_char": len(doc.full_text),
                "start_page": 2,
                "end_page": 2,
            }
        ]
        _, meta = kp.extract_consulted_entities(doc, chapters)
        assert meta["source"] == "ceq_consultation_chapter"


# --- public commenters ------------------------------------------------------

COMMENT_DOC = (
    "d" * 3100,
    "COMMENTS RECEIVED\n"
    "Mr. Robert Johnson stated that the noise from the proposed freeway would "
    "harm his neighborhood and opposed the project.\n"
    "Dr. Jane Smith, Director of the State Water Commission, supported the "
    "proposed action.\n"
    "The Sierra Club opposed the selected alternative.\n",
)


class TestPublicCommenters:
    PAYLOAD = {
        "public_commenters": [
            {
                "name": "Robert Johnson",
                "kind": "private",
                "stance": "oppose",
                "affiliation": None,
                "quote": "Mr. Robert Johnson stated that the noise from the proposed freeway",
            },
            {
                "name": "Jane Smith",
                "kind": "official",
                "stance": "support",
                "affiliation": "State Water Commission",
                "quote": "Dr. Jane Smith, Director of the State Water Commission, supported the",
            },
            {
                "name": "Sierra Club",
                "kind": "organization",
                "stance": "oppose",
                "affiliation": None,
                "quote": "The Sierra Club opposed the selected alternative.",
            },
        ]
    }

    def test_no_comment_heading_means_empty_and_no_llm_call(self, doc_factory, monkeypatch):
        no_llm(monkeypatch)
        entries, meta = kp.extract_public_commenters(
            doc_factory("e" * 3100 + "\nCONSULTATION AND COORDINATION\nEPA was consulted.\n"), []
        )
        assert entries == []
        assert meta["source"] == "no_comment_chapter"
        assert "empty list" in meta["note"]

    def test_stances_and_capacities(self, doc_factory, monkeypatch):
        monkeypatch.setattr(kp, "sonnet", fake_sonnet(commenters=self.PAYLOAD))
        entries, meta = kp.extract_public_commenters(doc_factory(*COMMENT_DOC), [])
        by_name = {e["name"]: e for e in entries}

        assert by_name["Robert Johnson"]["stance"] == "oppose"
        assert by_name["Robert Johnson"]["capacity"]["capacity"] == "private"
        assert by_name["Robert Johnson"]["human_review"] is True

        assert by_name["Jane Smith"]["capacity"]["capacity"] == "non_private"
        assert by_name["Jane Smith"]["human_review"] is False

        assert by_name["Sierra Club"]["capacity"]["capacity"] == "non_private"

        # One private stance is enough to gate the bucket (policy, MCAL_PLAN 3.11).
        assert meta["human_review"] is True
        assert meta["n_private_or_ambiguous"] == 1

    def test_off_enum_stance_is_nulled_not_invented(self, doc_factory, monkeypatch):
        monkeypatch.setattr(
            kp,
            "sonnet",
            fake_sonnet(
                commenters={
                    "public_commenters": [
                        {"name": "Sierra Club", "kind": "organization", "stance": "mostly against"}
                    ]
                }
            ),
        )
        entries, _ = kp.extract_public_commenters(doc_factory(*COMMENT_DOC), [])
        assert entries[0]["stance"] is None

    def test_llm_failure_degrades(self, doc_factory, monkeypatch):
        def explode(system, user, **kw):
            raise RuntimeError("model down")

        monkeypatch.setattr(kp, "sonnet", explode)
        entries, meta = kp.extract_public_commenters(doc_factory(*COMMENT_DOC), [])
        assert entries == []
        assert meta["extractor_ok"] is False

    def test_evidence_is_verified_and_pages_resolved(self, doc_factory, monkeypatch):
        monkeypatch.setattr(kp, "sonnet", fake_sonnet(commenters=self.PAYLOAD))
        entries, _ = kp.extract_public_commenters(doc_factory(*COMMENT_DOC), [])
        ev = entries[0]["evidence"][0]
        assert ev["quote_verified"] is True
        assert ev["source_pages"] == ["2"]


# --- Critic role-check hook -------------------------------------------------


class TestRoleCheck:
    def _result(self, doc_factory, monkeypatch):
        monkeypatch.setattr(
            kp,
            "sonnet",
            fake_sonnet(
                cooperating={
                    "cooperating_agencies": [
                        {
                            "name": "Bureau of Reclamation",
                            "quote": "The Bureau of Reclamation is a cooperating agency",
                        }
                    ]
                },
                consulted={
                    "consulted_entities": [
                        {"name": "Environmental Protection Agency", "role": "consulted_agency"}
                    ]
                },
                preparers={"agency_preparers": [{"name": "Jane Doe", "role": "Planner"}]},
            ),
        )
        doc = doc_factory(*COOP_DOC)
        return kp.run_key_people_pipeline(doc, [], year=1990, year_critic_verdict="PASS"), doc

    def test_items_carry_everything_the_critic_needs(self, doc_factory, monkeypatch):
        result, _ = self._result(doc_factory, monkeypatch)
        items = {i["item_id"]: i for i in result["role_check_items"]}
        coop = items["cooperating_agencies[0]"]
        assert coop["question"] == kp.ROLE_CHECK_QUESTION
        assert coop["expected_answer"] == "a"
        assert set(coop["options"]) == {"a", "b", "c"}
        assert coop["entity"] == "Bureau of Reclamation"
        assert "cooperating agency" in coop["cited_passage"]
        assert coop["quote_verified"] is True
        assert items["consulted_entities[0]"]["expected_answer"] == "c"
        assert items["agency_preparers[0]"]["expected_answer"] == "c"

    def test_agreement_passes(self, doc_factory, monkeypatch):
        result, _ = self._result(doc_factory, monkeypatch)
        answers = {i["item_id"]: i["expected_answer"] for i in result["role_check_items"]}
        out = kp.apply_role_check_answers(result, answers)
        assert out["verdict"] == "PASS"
        assert out["tags"] == []
        assert out["n_unanswered"] == 0

    def test_mismatch_is_re_extract_with_T05(self, doc_factory, monkeypatch):
        result, _ = self._result(doc_factory, monkeypatch)
        answers = {i["item_id"]: i["expected_answer"] for i in result["role_check_items"]}
        answers["cooperating_agencies[0]"] = "b"    # the Critic reads it as a commenter
        out = kp.apply_role_check_answers(result, answers)
        assert out["verdict"] == "RE_EXTRACT"
        assert out["tags"] == [kp.T_COMMENTER_AS_COOPERATOR]
        assert out["mismatches"][0]["entity"] == "Bureau of Reclamation"
        assert kp.T_COMMENTER_AS_COOPERATOR in result["tags"]

    def test_unanswered_items_are_reported_not_passed(self, doc_factory, monkeypatch):
        result, _ = self._result(doc_factory, monkeypatch)
        out = kp.apply_role_check_answers(result, {})
        assert out["n_checked"] == 0
        assert out["n_unanswered"] == len(result["role_check_items"])
        assert out["verdict"] == "PASS"        # nothing checked, nothing refuted

    def test_garbage_answers_are_treated_as_unanswered(self, doc_factory, monkeypatch):
        result, _ = self._result(doc_factory, monkeypatch)
        out = kp.apply_role_check_answers(
            result, {i["item_id"]: "maybe?" for i in result["role_check_items"]}
        )
        assert out["n_checked"] == 0


# --- end to end -------------------------------------------------------------


class TestRunPipeline:
    ALL_SONNET = dict(
        preparers={
            "agency_preparers": [
                {
                    "name": "Jane Doe",
                    "role": "Environmental Planner",
                    "organization": "FHWA",
                    "quote": "",
                }
            ]
        },
        cooperating={
            "cooperating_agencies": [
                {
                    "name": "Bureau of Reclamation",
                    "designation_phrase": "is a cooperating agency",
                    "quote": "The Bureau of Reclamation is a cooperating agency",
                }
            ]
        },
        consulted={
            "consulted_entities": [
                {"name": "Environmental Protection Agency", "role": "consulted_agency"},
                {"name": "Buffalo Public Library", "role": "recipient_of_draft"},
            ]
        },
        commenters={"public_commenters": []},
    )

    def test_four_buckets_and_no_cooperator_label_leakage(self, doc_factory, monkeypatch):
        monkeypatch.setattr(kp, "sonnet", fake_sonnet(**self.ALL_SONNET))
        res = kp.run_key_people_pipeline(
            doc_factory(*COOP_DOC), [], year=1990, year_critic_verdict="PASS"
        )
        assert res["counts"] == {
            "agency_preparers": 1,
            "cooperating_agencies": 1,
            "consulted_entities": 2,
            "public_commenters": 0,
        }
        assert res["human_review"] is False
        assert res["tags"] == []
        # Everyone from the catch-all chapter is in consulted_entities, and none
        # of them is labelled a cooperator. This is MCAL_PLAN 1(10)'s fix.
        assert {e["name"] for e in res["consulted_entities"]} == {
            "Environmental Protection Agency",
            "Buffalo Public Library",
        }

    def test_no_cooperating_heading_gates_only_that_bucket(self, doc_factory, monkeypatch):
        payload = dict(self.ALL_SONNET)
        payload["fallback"] = {"answer": "uncertain", "cooperating_agencies": []}
        monkeypatch.setattr(kp, "sonnet", fake_sonnet(**payload))
        res = kp.run_key_people_pipeline(
            doc_factory(*CONSULTATION_ONLY_DOC), [], year=1990, year_critic_verdict="PASS"
        )
        assert res["cooperating_agencies"] == []
        assert res["human_review_fields"]["cooperating_agencies"] is True
        assert res["human_review_fields"]["consulted_entities"] is False
        assert res["human_review"] is True
        assert res["consulted_entities"]
        assert any("whitelist" in n for n in res["notes"])

    def test_era_gate_does_not_suppress_the_raw_extraction(self, doc_factory, monkeypatch):
        """MCAL_PLAN 3.12 / 7 Q8: emit the extraction alongside the gate decision."""
        monkeypatch.setattr(kp, "sonnet", fake_sonnet(**self.ALL_SONNET))
        res = kp.run_key_people_pipeline(
            doc_factory(*COOP_DOC), [], year=1977, year_critic_verdict="HUMAN_REVIEW"
        )
        assert res["human_review"] is True
        assert res["era_gate"]["field_human_review"] is True
        # Preparers and consulted entities are still extracted and visible.
        assert res["counts"]["agency_preparers"] == 1
        assert res["counts"]["consulted_entities"] == 2
        # The whitelisted heading is still honoured (era-independent evidence),
        # so the reviewer of this gated field gets a candidate answer.
        assert res["counts"]["cooperating_agencies"] == 1
        assert res["sources"]["cooperating_agencies"]["source"] == "heading_whitelist"
        assert res["human_review_fields"]["cooperating_agencies"] is True

    def test_pre_1978_tags_T13(self, doc_factory, monkeypatch):
        monkeypatch.setattr(kp, "sonnet", fake_sonnet(**self.ALL_SONNET))
        res = kp.run_key_people_pipeline(
            doc_factory(*COOP_DOC), [], year=1971, year_critic_verdict="PASS"
        )
        assert kp.T_PRE_1978 in res["tags"]
        assert res["human_review_fields"]["cooperating_agencies"] is True
        assert res["counts"]["agency_preparers"] == 1

    def test_artifact_shapes_feed_the_era_gate(self, doc_factory, monkeypatch):
        monkeypatch.setattr(kp, "sonnet", fake_sonnet(**self.ALL_SONNET))
        res = kp.run_key_people_pipeline(
            doc_factory(*COOP_DOC),
            [],
            m1={"year": {"value": 1974}},
            critic={"year": {"verdict": "PASS"}},
        )
        assert res["era_gate"]["year"] == 1974
        assert kp.T_PRE_1978 in res["tags"]

    def test_private_commenter_gates_the_field(self, doc_factory, monkeypatch):
        payload = dict(self.ALL_SONNET)
        payload["commenters"] = TestPublicCommenters.PAYLOAD
        monkeypatch.setattr(kp, "sonnet", fake_sonnet(**payload))
        doc = doc_factory(*(COOP_DOC + COMMENT_DOC[1:]))
        res = kp.run_key_people_pipeline(doc, [], year=1990, year_critic_verdict="PASS")
        assert res["human_review_fields"]["public_commenters"] is True
        assert res["human_review"] is True
        assert any("private individual" in n for n in res["notes"])

    def test_total_llm_failure_still_returns_a_usable_result(self, doc_factory, monkeypatch):
        def explode(system, user, **kw):
            raise RuntimeError("no credentials")

        monkeypatch.setattr(kp, "sonnet", explode)
        res = kp.run_key_people_pipeline(
            doc_factory(*COOP_DOC), [], year=1990, year_critic_verdict="PASS"
        )
        assert res["counts"] == {
            "agency_preparers": 0,
            "cooperating_agencies": 0,
            "consulted_entities": 0,
            "public_commenters": 0,
        }
        assert res["role_check_items"] == []

    def test_whitelisted_heading_survives_an_untrustworthy_year(
        self, doc_factory, monkeypatch
    ):
        """
        Era-unknown must not throw away era-INDEPENDENT evidence: a heading that
        says "COOPERATING AGENCIES" over a list means what it says whatever the
        year is. The bucket is still gated.
        """
        monkeypatch.setattr(kp, "sonnet", fake_sonnet(**self.ALL_SONNET))
        res = kp.run_key_people_pipeline(
            doc_factory(*COOP_DOC), [], year=1985, year_critic_verdict="RE_EXTRACT"
        )
        assert res["counts"]["cooperating_agencies"] == 1
        assert res["human_review"] is True
        assert res["era_gate"]["allow_cooperating_fallback"] is False

    def test_untrustworthy_year_without_a_heading_suppresses_the_fallback(
        self, doc_factory, monkeypatch
    ):
        no_llm_for_fallback = fake_sonnet(**self.ALL_SONNET)   # has no `fallback` key
        monkeypatch.setattr(kp, "sonnet", no_llm_for_fallback)
        res = kp.run_key_people_pipeline(
            doc_factory(*CONSULTATION_ONLY_DOC), [], year=1985, year_critic_verdict="RE_EXTRACT"
        )
        assert res["cooperating_agencies"] == []
        assert (
            res["sources"]["cooperating_agencies"]["source"]
            == "fallback_suppressed_by_era_gate"
        )
        assert any("suppressed" in n for n in res["notes"])

    def test_m2_adapter_shape(self, doc_factory, monkeypatch):
        monkeypatch.setattr(kp, "sonnet", fake_sonnet(**self.ALL_SONNET))
        res = kp.run_key_people_pipeline(
            doc_factory(*COOP_DOC), [], year=1990, year_critic_verdict="PASS"
        )
        field = kp.as_m2_key_people_field(res)
        # The three Segment A keys keep their names, plus the new bucket.
        assert set(field["value"]) == {
            "agency_preparers",
            "cooperating_agencies",
            "consulted_entities",
            "public_commenters",
            "comment_response_present",
        }
        assert field["value"]["comment_response_present"] is False
        assert field["confidence"] == "high"

    def test_comment_response_present_means_a_heading_was_found(
        self, doc_factory, monkeypatch
    ):
        """
        Segment A set this flag from a document-wide regex for the phrase
        "response to comments", which is why commenters leaked in from
        non-comment chapters. It now means a comment HEADING exists.
        """
        payload = dict(self.ALL_SONNET)
        payload["commenters"] = TestPublicCommenters.PAYLOAD
        monkeypatch.setattr(kp, "sonnet", fake_sonnet(**payload))
        doc = doc_factory(*(COOP_DOC + COMMENT_DOC[1:]))
        field = kp.as_m2_key_people_field(
            kp.run_key_people_pipeline(doc, [], year=1990, year_critic_verdict="PASS")
        )
        assert field["value"]["comment_response_present"] is True
        assert len(field["value"]["public_commenters"]) == 3
        assert field["confidence"] == "low"      # a private stance gates the field

    def test_chapters_are_recovered_when_not_supplied(self, doc_factory, monkeypatch):
        """`chapters=None` must not silently disable the CEQ-chapter routes."""
        monkeypatch.setattr(kp, "sonnet", fake_sonnet(**self.ALL_SONNET))
        res = kp.run_key_people_pipeline(
            doc_factory(*COOP_DOC), year=1990, year_critic_verdict="PASS"
        )
        assert res["counts"]["cooperating_agencies"] == 1


# --- regressions on real graded documents -----------------------------------


def _critic_verdict(doc_id: str, field: str):
    path = settings.CRITIC_DIR / f"{doc_id}.json"
    if not path.exists():
        pytest.skip(f"no Critic output for {doc_id}")
    entry = json.loads(path.read_text()).get(field) or {}
    return entry.get("verdict")


class TestGradedCorpusRegressions:
    """
    MCAL_PLAN 1(10): 5 of 8 docs had "all commenters = cooperators".

    On the real corpus, NONE of the graded docs has a heading matching the
    cooperating whitelist -- they are 1970s documents whose entity lists live
    under "Consultation and Coordination" or nowhere. So the correct behaviour on
    every one of them is: empty cooperating_agencies + HUMAN_REVIEW, never a list
    scraped from the catch-all chapter.
    """

    @pytest.mark.parametrize(
        "doc_id", [LINCOLN_HWY, BUFFALO, OPERATION_BREAKTHROUGH, FUEL_ECONOMY]
    )
    def test_no_graded_doc_has_a_whitelisted_cooperating_heading(self, doc_loader, doc_id):
        doc = doc_loader(doc_id)
        assert kp.find_sections(doc, kp.COOPERATING_HEADINGS) == []

    @pytest.mark.parametrize("doc_id", [LINCOLN_HWY, BUFFALO, OPERATION_BREAKTHROUGH])
    def test_uncertain_fallback_yields_empty_plus_human_review(
        self, doc_loader, doc_id, monkeypatch
    ):
        monkeypatch.setattr(
            kp,
            "sonnet",
            fake_sonnet(
                fallback={
                    "answer": "uncertain",
                    "cooperating_agencies": [],
                    "reasoning": "1970s format; no formal designation stated.",
                }
            ),
        )
        entries, meta = kp.extract_cooperating_agencies(doc_loader(doc_id), [])
        assert entries == []
        assert meta["source"] == "sonnet_fallback"
        assert meta["human_review"] is True

    def test_lincoln_hwy_has_real_comment_sections(self, doc_loader):
        """
        The commenter bucket is licensed by a heading, and Lincoln Hwy genuinely
        has one -- so the whitelist is not merely rejecting everything.
        """
        sections = kp.find_sections(doc_loader(LINCOLN_HWY), kp.COMMENT_HEADINGS)
        assert sections
        assert all("comment" in s.matched_phrase for s in sections)
        assert all(s.text.strip() for s in sections)

    @pytest.mark.parametrize("doc_id", [BUFFALO, OPERATION_BREAKTHROUGH, FUEL_ECONOMY])
    def test_docs_without_a_comment_heading_yield_no_commenters(
        self, doc_loader, doc_id, monkeypatch
    ):
        no_llm(monkeypatch)
        entries, meta = kp.extract_public_commenters(doc_loader(doc_id), [])
        assert entries == []
        assert meta["source"] == "no_comment_chapter"

    def test_fuel_economy_year_verdict_gates_key_people(self, doc_loader, monkeypatch):
        """
        The live dependent-field cascade case: Segment A's Critic returned
        HUMAN_REVIEW on `year` (NUL 1977 vs regex 1979), so key_people must be
        HUMAN_REVIEW unconditionally, and -- since this document has no
        whitelisted cooperating heading -- the designation-check fallback must be
        suppressed rather than asked a question it cannot answer.
        """
        verdict = _critic_verdict(FUEL_ECONOMY, "year")
        assert verdict in kp.CRITIC_VERDICTS_UNTRUSTWORTHY, (
            "fixture drift: this regression assumes Fuel Economy's year verdict "
            f"is untrustworthy, got {verdict}"
        )
        monkeypatch.setattr(
            kp, "sonnet", fake_sonnet(preparers={"agency_preparers": []})
        )
        res = kp.run_key_people_pipeline(
            doc_loader(FUEL_ECONOMY), [], year=1977, year_critic_verdict=verdict
        )
        assert res["human_review"] is True
        assert res["era_gate"]["field_human_review"] is True
        assert res["cooperating_agencies"] == []
        assert (
            res["sources"]["cooperating_agencies"]["source"]
            == "fallback_suppressed_by_era_gate"
        )

    def test_old_extractor_scraped_the_catch_all_chapter(self, m2_loader):
        """
        Documents what the replaced code did, so the fix has a baseline. Segment
        A's key_people output put every entity it found into one of three buckets
        with no consulted_entities bucket at all -- there was nowhere else to put
        a consulted agency or a draft recipient.
        """
        old = m2_loader(LINCOLN_HWY).get("key_people", {}).get("value", {})
        assert "consulted_entities" not in old
        assert set(old) <= {
            "agency_preparers",
            "cooperating_agencies",
            "public_commenters",
            "comment_response_present",
        }
