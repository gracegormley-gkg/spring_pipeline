# Rubric overlay — `summary.public_response`

**The most common failure in the corpus**: 4/8 docs missing citations
(MCAL_PLAN §1(5)). Comment-response tables are structurally different from body
chapters and the extractor had no per-subfield citation enforcement.

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

Public and agency comment on the document, and the agency's response.
Every claim needs a page cite. An empty result is valid; free-text without
cites is not.

## QUESTIONS

- **Q7.** Are stances attributed to named commenters, or explicitly to "a private commenter"?
- **Q8.** If no comment-response chapter was identified, does the value say so explicitly (`no_comment_response_chapter_identified`) rather than silently summarizing the main document as though it were comment?
- **Q9.** Does the value distinguish what commenters said from what the agency replied? Conflating the two misattributes positions.
- **Q10.** If the value claims comment CHANGED the analysis, scope or preferred alternative, does the cited page state that?

## DECISION

1. Q7 = **no** → `RE_EXTRACT`, `failure_tag = T01_missing_citation`.
2. Q8 = **no** → `RE_EXTRACT`, `failure_tag = T03_outside_text_fabrication`.
3. Q9 = **no** → `RE_EXTRACT`, `failure_tag = T03_outside_text_fabrication`.
4. Q10 = **no** → `RE_EXTRACT`, `failure_tag = T03_outside_text_fabrication`.
5. Otherwise fall through to the base decision table.

An **empty** `public_response` on a document with no comment chapter is a `PASS`.
Do not penalize a correct emptiness.

