# Rubric overlay — `summary.alternatives_overview`

One of 8 docs was missing a citation (MCAL_PLAN §1(6)–(7)). This is the
summary *subfield*; the standalone `alternatives` list has its own rubric.

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

2–4 sentences on the range of alternatives evaluated, including the
no-action alternative and the preferred alternative if identified.

## QUESTIONS

- **Q7.** Is each alternative the value names actually present in the document's alternatives analysis, under a recognizable label?
- **Q8.** If the value identifies a "preferred" alternative, does the document itself designate it as preferred at the cited page?
- **Q9.** If the value claims alternatives were "considered and dismissed", does the cited page give the dismissal rationale?

## DECISION

1. Q7 = **no** → `RE_EXTRACT`, `failure_tag = T03_outside_text_fabrication`.
2. Q8 = **no** → `RE_EXTRACT`, `failure_tag = T03_outside_text_fabrication`. Designating a preferred alternative the document does not is a substantive error.
3. Q9 = **no** → `PASS_WITH_NOTE`.
4. Otherwise fall through to the base decision table.

