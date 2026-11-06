"""
CEQ chapter detection on 1970s documents (PIPELINE_REVIEW 2.9).

`chunk.detect_chapters` is upstream of `alternatives`, `location` and
`key_people`, and `extract_alternatives` has no fallback by design, so a missed
Alternatives chapter produces `alternatives: []` -- which is not in
`EMPTY_IS_VALID_FIELDS` and therefore registers as a failure rather than a
result.

The review attributed the weakness to the heading heuristic (uppercase dominance,
marker prefix, 3000-char TOC guard). Measuring it says otherwise: the heuristic
finds the headings, and `_map_to_ceq` then discards them, because the alias table
contained only modern CEQ vocabulary and no document in this corpus uses it. These
are NEPA 102(2)(C) documents from the 1970s, organized as a numbered five-point
response:

    1. DESCRIPTION OF PROPOSED ACTION
    2. DESCRIPTION OF THE EXISTING ENVIRONMENT
    PROBABLE ENVIRONMENTAL IMPACTS
    9. STAFF DISCUSSION OF SIGNIFICANT ENVIRONMENTAL MATTERS
    EVALUATION AND DISPOSITION OF COMMENTS RECEIVED FROM DRAFT ENVIRONMENTAL
        STATEMENT

That is why `Purpose and Need` was detected zero times in 21 documents: they do
not have a chapter by that name, they have "DESCRIPTION OF PROPOSED ACTION".

`TestBeforeAndAfter` measures both tables over the real corpus so the improvement
is a number rather than a claim, and `test_no_new_runaway_spans` guards the
obvious way this could have gone wrong -- more detections shortening or inflating
the spans that feed the extractors.
"""

from __future__ import annotations

import collections

import pytest

import chunk as ck
import config
from mcal import settings


# The table as it was: modern CEQ vocabulary only.
MODERN_ONLY = {
    "Purpose and Need": [
        "purpose and need", "purpose of and need for", "background and need",
    ],
    "Alternatives": [
        "alternatives", "alternatives considered", "proposed action and alternatives",
    ],
    "Affected Environment": [
        "affected environment", "existing environment", "environmental setting",
    ],
    "Environmental Consequences": [
        "environmental consequences", "environmental impacts", "impacts",
    ],
    "Mitigation": ["mitigation", "mitigation measures"],
    "Consultation": [
        "consultation", "consultation and coordination", "list of preparers",
        "preparers",
    ],
}


def _precedence(table):
    return sorted(
        ((alias, canonical) for canonical, aliases in table.items()
         for alias in aliases),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )


def _measure(table, docs):
    original = ck._ALIAS_PRECEDENCE
    ck._ALIAS_PRECEDENCE = _precedence(table)
    try:
        counts = collections.Counter()
        zero_chapter_docs = 0
        runaway = 0
        for doc in docs:
            chapters = ck.detect_chapters(doc)
            if not chapters:
                zero_chapter_docs += 1
            for h in chapters:
                counts[h["ceq_chapter"]] += 1
                pages = h["end_page"] - h["start_page"] + 1
                if doc.n_pages and pages / doc.n_pages > 0.8:
                    runaway += 1
        return {
            "counts": counts,
            "zero_chapter_docs": zero_chapter_docs,
            "runaway_spans": runaway,
        }
    finally:
        ck._ALIAS_PRECEDENCE = original


@pytest.fixture(scope="module")
def corpus():
    from pages import load_doc

    docs = []
    for d in sorted(settings.PAGES_DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        try:
            docs.append(load_doc(d.name, settings.PAGES_DATA_DIR))
        except Exception:  # noqa: BLE001 - a partial corpus is still measurable
            continue
    if len(docs) < 20:
        pytest.skip("corpus not materialized on this machine")
    return docs


# --- Alias precedence --------------------------------------------------------


class TestAliasPrecedence:
    def test_longest_alias_wins(self):
        """
        Precedence used to be dict order, which is only safe while no alias is a
        prefix of another. It no longer is.
        """
        assert ck._map_to_ceq("PROPOSED ACTION AND ALTERNATIVES") == "Alternatives"
        assert ck._map_to_ceq("DESCRIPTION OF PROPOSED ACTION") == "Purpose and Need"
        assert ck._map_to_ceq("PROPOSED ACTION") == "Purpose and Need"

    def test_dict_order_would_have_got_that_wrong(self):
        """The regression this guards, executed under the old rule."""
        original = ck._ALIAS_PRECEDENCE
        ck._ALIAS_PRECEDENCE = [
            (alias, canonical)
            for canonical, aliases in config.CHAPTER_ALIASES.items()
            for alias in aliases
        ]
        try:
            assert ck._map_to_ceq("PROPOSED ACTION AND ALTERNATIVES") == (
                "Purpose and Need"
            )
        finally:
            ck._ALIAS_PRECEDENCE = original

    def test_precedence_is_actually_sorted_longest_first(self):
        lengths = [len(alias) for alias, _ in ck._ALIAS_PRECEDENCE]
        assert lengths == sorted(lengths, reverse=True)

    @pytest.mark.parametrize(
        "heading,expected",
        [
            ("1. DESCRIPTION OF PROPOSED ACTION", "Purpose and Need"),
            ("2. DESCRIPTION OF THE EXISTING ENVIRONMENT", "Affected Environment"),
            ("II. DESCRIPTION OF THE ENVIRONMENT", "Affected Environment"),
            ("PROBABLE ENVIRONMENTAL IMPACTS", "Environmental Consequences"),
            (
                "9. STAFF DISCUSSION OF SIGNIFICANT ENVIRONMENTAL MATTERS",
                "Environmental Consequences",
            ),
            ("MITIGATING MEASURES DURING MINING OPERATIONS", "Mitigation"),
            (
                "EVALUATION AND DISPOSITION OF COMMENTS RECEIVED FROM DRAFT "
                "ENVIRONMENTAL STATEMENT",
                "Consultation",
            ),
            ("COMMENTS RECEIVED", "Consultation"),
            ("5. ALTERNATIVE FREEWAY CORRIDOR LOCATIONS.", "Alternatives"),
            ("C. RECOMMENDED ALTERNATIVE", "Alternatives"),
            ("CHAPTER IX - CONSULTATION & COORDINATION WITH OTHERS", "Consultation"),
        ],
    )
    def test_real_headings_from_the_corpus_now_map(self, heading, expected):
        """
        Every one of these was collected from a document in this corpus by dumping
        the lines that passed the heading heuristic and failed `_map_to_ceq`.
        """
        assert ck._map_to_ceq(heading) == expected

    @pytest.mark.parametrize(
        "heading",
        [
            "UNITED STATES ENVIRONMENTAL PROTECTION AGENCY",
            "ENVIRONMENTAL PROTECTION AGENCY WASHINGTON, D.C. 20460",
            "DATE COMMENT RECEIVED",
            "NATURE OF COMMENTS",
            "AGENCY REQUESTS CONFERENCE PRIOR TO MAKING COMMENTS",
            "ECONOMIC IMPACTS",
            "SOCIAL IMPACTS",
        ],
    )
    def test_lookalikes_still_do_not_map(self, heading):
        """
        The reason `_map_to_ceq` keeps START-anchored matching rather than
        containment: these all contain an alias and none of them is a chapter.
        """
        assert ck._map_to_ceq(heading) is None


# --- The corpus measurement --------------------------------------------------


class TestBeforeAndAfter:
    def test_detection_improves_on_every_chapter(self, corpus):
        before = _measure(MODERN_ONLY, corpus)
        after = _measure(config.CHAPTER_ALIASES, corpus)
        for chapter in (
            "Purpose and Need", "Alternatives", "Affected Environment",
            "Environmental Consequences", "Mitigation", "Consultation",
        ):
            assert after["counts"][chapter] >= before["counts"][chapter], chapter

    def test_the_numbers(self, corpus):
        """
        Recorded rather than merely compared, so a future change that quietly
        undoes this is visible. 21 documents.

            chapter                      before  after
            Purpose and Need                  0      9
            Alternatives                     17     18
            Affected Environment              1      7
            Environmental Consequences       10     16
            Mitigation                        2      4
            Consultation                      2     10
        """
        before = _measure(MODERN_ONLY, corpus)
        after = _measure(config.CHAPTER_ALIASES, corpus)
        assert before["counts"]["Purpose and Need"] == 0
        assert after["counts"]["Purpose and Need"] >= 8
        assert before["counts"]["Consultation"] == 2
        assert after["counts"]["Consultation"] >= 9
        assert before["counts"]["Affected Environment"] == 1
        assert after["counts"]["Affected Environment"] >= 6

    def test_no_document_is_left_without_a_chapter(self, corpus):
        before = _measure(MODERN_ONLY, corpus)
        after = _measure(config.CHAPTER_ALIASES, corpus)
        assert before["zero_chapter_docs"] == 1
        assert after["zero_chapter_docs"] == 0

    def test_no_new_runaway_spans(self, corpus):
        """
        The way this could have gone wrong. A span runs to the next DETECTED
        heading, so adding detections changes every span, and a spurious one could
        either truncate a real chapter or swallow the document.
        """
        before = _measure(MODERN_ONLY, corpus)
        after = _measure(config.CHAPTER_ALIASES, corpus)
        assert after["runaway_spans"] <= before["runaway_spans"]

    def test_alternatives_is_detected_for_nearly_every_document(self, corpus):
        """
        `extract_alternatives` has no fallback, and `[]` is not in
        EMPTY_IS_VALID_FIELDS, so a miss here registers as a failure.
        """
        after = _measure(config.CHAPTER_ALIASES, corpus)
        assert after["counts"]["Alternatives"] >= 18


class TestSpanBoundaryIsReported:
    def test_the_flag_is_present_and_at_most_one_span_is_unbounded(self, corpus):
        """
        A span that ends at EOF may be returning most of the document rather than a
        chapter, so `text_for_ceq_chapter` callers can now tell the two apart.

        At most one per document, and never more: dedup keeps the LONGEST-span
        occurrence of each chapter, so the EOF-terminated heading is often not the
        one retained -- which is why this is "at most one", not "exactly one".
        """
        n_with_eof = 0
        for doc in corpus:
            chapters = ck.detect_chapters(doc)
            if not chapters:
                continue
            assert all(isinstance(h["ends_at_eof"], bool) for h in chapters)
            eof = [h for h in chapters if h["ends_at_eof"]]
            assert len(eof) <= 1, doc.doc_id
            if eof:
                n_with_eof += 1
                # It has to be the last span in the document.
                assert eof[0] is max(chapters, key=lambda h: h["start_char"])
        assert n_with_eof, "no unbounded span anywhere; the flag is never exercised"

    def test_a_lone_detection_is_flagged_unbounded(self, corpus):
        """
        The case worth knowing about: one detected chapter means its span is the
        whole document from that point on.
        """
        singles = [
            (doc, ck.detect_chapters(doc))
            for doc in corpus
            if len(ck.detect_chapters(doc)) == 1
        ]
        for doc, chapters in singles:
            assert chapters[0]["ends_at_eof"] is True, doc.doc_id
