# Rubric overlay — `eis_type`

Wrong on 1/8 docs: Lincoln Hwy, a Final EIS, was flagged ROD because a
first-page regex matched \"Record of Decision\" in body or citation text
(MCAL_PLAN §1(2)).

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

One of `Draft`, `Final`, `Supplemental`, `ROD`, `Unknown`.

## QUESTIONS

- **Q7.** Does the evidence show the type as a **section heading or cover-page designation**, rather than as body text or a citation?
- **Q8.** Is any occurrence of "Record of Decision" a heading describing THIS document, or is it a citation to a future or separate document? (A Final EIS routinely says a ROD "will be issued".)
- **Q9.** If the document is both supplemental and draft/final, was the more specific `Supplemental` chosen?
- **Q10.** Does the filename slug and the document-provided title agree with the chosen type?

## DECISION

1. Q8 = "citation to a separate document" while the reported type is `ROD` → `RE_EXTRACT`, `failure_tag = T12_eis_type_confused_with_rod`.
2. Q7 = **no** → `RE_EXTRACT`, `failure_tag = T12_eis_type_confused_with_rod`.
3. Q9 = **no** → `RE_EXTRACT`.
4. Q10 = **no** → `PASS_WITH_NOTE`, `note` naming the disagreement.
5. Otherwise fall through to the base decision table.

