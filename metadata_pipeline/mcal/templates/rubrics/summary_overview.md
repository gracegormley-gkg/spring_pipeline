# Rubric overlay — `summary.overview`

Graded `ok` on 8/8 docs. Its evidence is carried forward from the five
subfields when its own quotes are unverified (`m2.py`), so a citation defect
here often reflects a subfield defect.

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

3–5 sentence orientation to the whole document: what is proposed, where,
the alternatives, the main impacts, the public response.

## QUESTIONS

- **Q7.** Does the overview introduce any fact not present in the five subfields? It is a synthesis, not a new extraction.
- **Q8.** Would a reader with no NEPA background be able to say what is being proposed, where, and what is being decided, from this text alone?

## DECISION

1. Q7 = **yes** → `RE_EXTRACT`, `failure_tag = T03_outside_text_fabrication`.
2. Q8 = **no** → `PASS_WITH_NOTE`, `failure_tag = T16_abstract_when_concrete_available`.
3. Otherwise fall through to the base decision table.

