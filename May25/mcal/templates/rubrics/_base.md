# Base rubric (shared)

Questions Q1–Q6 from MCAL_PLAN §3.5, applied to every field. Per-field files in
this directory add questions numbered Q7+ and may override the decision table.

The `{{FIELD}}` and `{{FIELD_DESCRIPTION}}` placeholders are substituted by
`mcal/critic_prompt.py`.

Q6(b) is **logged only at v1**. Sonnet-judged concreteness is subjective and
uncalibrated at n=8, so its answer is recorded in `run_manifest.json` under
`rubric_answers.Q6b` for offline audit and does not affect the verdict. It is
promoted to `PASS_WITH_NOTE + T16` only after a spot-check against the graded
corpus (MCAL_PLAN §3.5).

---

## QUESTIONS

- **Q1.** Does the extracted value cite at least one page?
- **Q2.** Does every claim in the value correspond to a substring in the cited pages, allowing OCR-normalization tolerance? Check coordinating conjunctions separately: a clause introducing a new subject is its own claim.
- **Q3.** Is every acronym defined on first use within this value, or expanded by the document's own glossary?
- **Q4.** Is every figure, date and quantity in the value identical to the source — including any scope qualifier the document attaches to it?
- **Q5.** Does the value assert a stance, opinion or position attributed to a **private individual** (see PRIVATE_INDIVIDUAL above)?
- **Q6a.** Are NEPA-specific terms and regulatory citations glossed in-line on first mention, using support from the cited pages?
- **Q6b.** Does the value describe its subject in concrete terms — named entities, specified quantities, plain nouns — rather than abstract nominalizations, where the document supports concreteness?

## DECISION

Apply in order; the first matching rule wins.

1. Q5 = **yes** → `HUMAN_REVIEW`, `failure_tag = null`. (Policy, not calibrated. A private individual's stance is always reviewed by a human.)
2. Q1 = **no** → `RE_EXTRACT`, `failure_tag = T01_missing_citation`.
3. Q2 = **no** → `RE_EXTRACT`, `failure_tag = T03_outside_text_fabrication`.
4. Q4 = **no** → `RE_EXTRACT`. Use `T02_numeric_hallucination` when a figure is wrong or attached to the wrong entity; use `T19_scope_qualifier_dropped` when the figure is correct but a limiting qualifier was omitted.
5. Q3 = **no** → `PASS_WITH_NOTE`, `failure_tag = T04_undefined_acronym`.
6. Q6a = **no** → `PASS_WITH_NOTE`, `failure_tag = T15_jargon_without_gloss`.
7. Otherwise → `PASS`.

Q6b never changes the verdict at this stage. Record it and move on.
