# Rubric overlay — `title`

Graded `ok` on 8/8 docs. No dedicated fix in MCAL_PLAN §1; it passes
through the generic machinery. Note that 8/8 is weak evidence — the Wilson
interval still admits a true error rate near 30%.

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

The document's title, taken from catalogue metadata or the cover page.

## QUESTIONS

- **Q7.** Is the title a plausible EIS title rather than a fragment, a running header, or a page label?
- **Q8.** If catalogue metadata and the cover page disagree, does the chosen title match the cover page of THIS document?

## DECISION

M1 fields carry no verbatim quote by design, so Q2 and Q4 are usually `n/a`
and the deterministic quote-verify override is skipped. Judge support from the
first pages supplied in EVIDENCE.

1. Q7 = **no** → `RE_EXTRACT`, `failure_tag = null`.
2. Q8 = **no** → `PASS_WITH_NOTE`, `note` naming both candidates.
3. Otherwise fall through to the base decision table.

