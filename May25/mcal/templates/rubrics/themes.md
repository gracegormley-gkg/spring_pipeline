# Rubric overlay — `themes`

Graded `ok` on 8/8 docs. The real risk is closed-vocabulary drift: `m2.py`
trusts the returned strings verbatim with no taxonomy validation, so an
off-taxonomy theme passes through silently.

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

1–3 themes and 2–5 subthemes drawn from the frozen 10-theme taxonomy in
`segment_a/config.py`.

## QUESTIONS

- **Q7.** Is every theme and subtheme drawn **verbatim** from the frozen taxonomy supplied in EVIDENCE? Near-misses and paraphrases do not count.
- **Q8.** Does each subtheme belong to one of the themes selected?
- **Q9.** Is each theme supported by the document's actual subject matter, not merely plausible for an EIS?

## DECISION

1. Q7 = **no** → `RE_EXTRACT`, `note` listing the off-taxonomy strings. The vocabulary is closed.
2. Q8 = **no** → `RE_EXTRACT`.
3. Q9 = **no** → `RE_EXTRACT`, `failure_tag = T03_outside_text_fabrication`.
4. Otherwise fall through to the base decision table.

`themes` carries no prose, so Q3/Q6a/Q6b are `n/a`.

