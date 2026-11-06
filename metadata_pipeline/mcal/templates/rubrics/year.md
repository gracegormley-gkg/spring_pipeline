# Rubric overlay — `year`

Wrong on 3/8 docs, **all pre-1980** (MCAL_PLAN §1(1)). Cause: OCR noise on
1970s scans plus a regex limited to the first three pages. Old EISs often carry
the operative date on a signature/approval page or transmittal letter, not the
cover. `segment_b/year_adjudicator.py` now always runs and supplies
`source_type`.

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

Publication year, with the `source_type` the adjudicator assigned:
`signature`, `transmittal`, `cover`, `body`, or `adjudicated`.

## QUESTIONS

- **Q7.** Is the year within 1969–2026, and consistent with the document's own internal dating?
- **Q8.** Does the `source_type` match where the evidence quote actually comes from? A date lifted from body prose must not be labelled `signature`.
- **Q9.** Priority check: if the document contains a signature, approval or transmittal date that differs from the reported year, was the higher-priority date chosen? Priority is **signature > transmittal > cover > body**.
- **Q10.** Could the reported digits be an OCR misread of a different year (`l972`/`1972`, `197O`/`1970`, `I97I`/`1971`)? If so, does the surrounding text disambiguate?

## DECISION

1. Q7 = **no** → `RE_EXTRACT`, `failure_tag = T11_year_ocr_error`.
2. Q9 = **no** → `RE_EXTRACT`, `failure_tag = T11_year_ocr_error`; `note` must state the higher-priority date you found and where.
3. Q10 = **no** (i.e. an OCR misread is plausible and unresolved) → `HUMAN_REVIEW`, `failure_tag = T11_year_ocr_error`.
4. Q8 = **no** → `PASS_WITH_NOTE`.
5. Otherwise fall through to the base decision table.

`year` is a dependent field: `key_people` cascades to HUMAN_REVIEW whenever
this verdict is `RE_EXTRACT` or `HUMAN_REVIEW`, because the pre-1978 era gate
cannot be applied without a trustworthy year. Do not soften a verdict to avoid
that cascade.

