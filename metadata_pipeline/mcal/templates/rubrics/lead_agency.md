# Rubric overlay — `lead_agency`

Graded `ok` on 8/8 docs. Included for completeness; the only recurring
risk is joint-lead documents, where the extractor emits more than two agencies.

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

List of lead agencies, from catalogue contributors or the first pages.

## QUESTIONS

- **Q7.** Is each named entity a federal, state, tribal or local **agency**, rather than a contractor, consultant, preparer, or cooperating agency?
- **Q8.** If more than one agency is listed, does the document actually designate them **joint leads**, rather than one lead plus cooperators?

## DECISION

1. Q7 = **no** → `RE_EXTRACT`, `note` naming the non-agency entity.
2. Q8 = **no** → `PASS_WITH_NOTE`, `failure_tag = T05_commenter_mislabeled_as_cooperator` if a cooperator was promoted to lead.
3. Otherwise fall through to the base decision table.

