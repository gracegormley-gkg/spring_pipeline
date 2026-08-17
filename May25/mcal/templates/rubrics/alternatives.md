# Rubric overlay — `alternatives`

Empty on 1/8 docs (Buffalo Light Rail) because structural identification
of the Alternatives chapter failed on an atypical title, and the extractor
returned `[]` silently rather than a not-found status (MCAL_PLAN §1(8)).

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

List of `{name, description, evidence}` for the action alternatives the
document enumerates.

## QUESTIONS

- **Q7.** Is the list **empty**? If so, does the value carry an explicit `alternatives_chapter_not_found` status rather than a bare empty list?
- **Q8.** Is each entry an **action alternative** the document evaluates, rather than a chapter heading, a mitigation measure, or a design option within one alternative?
- **Q9.** Is the no-action / null alternative present? Nearly every EIS has one; its absence usually means the chapter was only partly read.
- **Q10.** Are the alternative labels the document's own ("Alternative V", "the Null") rather than invented?

## DECISION

1. Q7 = empty **and** no explicit status → `RE_EXTRACT`, `failure_tag = T10_alternatives_chapter_missed`. A silent empty list is never acceptable.
2. Q7 = empty **with** an explicit not-found status → `HUMAN_REVIEW`, `failure_tag = T10_alternatives_chapter_missed`.
3. Q8 = **no** → `RE_EXTRACT`, `failure_tag = T03_outside_text_fabrication`.
4. Q10 = **no** → `RE_EXTRACT`, `failure_tag = T03_outside_text_fabrication`.
5. Q9 = **no** → `PASS_WITH_NOTE`, `note` observing the missing no-action alternative.
6. Otherwise fall through to the base decision table.

