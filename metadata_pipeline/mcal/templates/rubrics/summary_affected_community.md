# Rubric overlay — `summary.affected_community`

Two of 8 docs were missing citations (MCAL_PLAN §1(6)).

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

2–4 sentences on who is affected: communities, populations, tribes,
institutions.

## QUESTIONS

- **Q7.** Are affected communities named concretely ("three census tracts in south Milwaukee; the Ho-Chunk Nation; two elementary schools within 500 feet") rather than as "nearby stakeholders" or "the surrounding community"?
- **Q8.** Is every demographic or population figure cited to a page that actually carries it?
- **Q9.** If a tribe or tribal nation is named, does the cited passage support the relationship asserted (consulted, affected, objecting)?

## DECISION

1. Q8 = **no** → `RE_EXTRACT`, `failure_tag = T02_numeric_hallucination`.
2. Q9 = **no** → `RE_EXTRACT`, `failure_tag = T03_outside_text_fabrication`.
3. Q7 = **no** → `PASS_WITH_NOTE`, `failure_tag = T16_abstract_when_concrete_available`.
4. Otherwise fall through to the base decision table.

