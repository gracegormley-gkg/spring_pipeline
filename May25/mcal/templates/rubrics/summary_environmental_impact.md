# Rubric overlay — `summary.environmental_impact`

The worst-performing summary subfield: one numeric error and one
outside-text fabrication in 8 docs, plus a missing citation. This is its own CP
bucket (`summary_numeric`) because its error profile is numeric.

The fabrication — \"or important wildlife habitats are affected\" appended to a
real impact sentence — is the canonical example of prior-injection and is
catchable. The magnitude case (7.5 vs 7.0) is **not**: p.146 says \"a Magnitude
7.5 earthquake occurring on the Newport-Inglewood Fault\" verbatim, while the
human's 7.0 comes from p.145's \"maximum credible event\" framing. Treat that as
`T19`, and expect that only a reader comparing pages can catch it.

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

2–4 sentences on the principal environmental consequences, with
magnitudes where the document quantifies them.

## QUESTIONS

- **Q7.** Coordination check: for every `or` / `and` / `as well as` / `along with` that introduces a NEW subject, is that clause independently present in the cited pages? Answer `no` if any such clause is unsupported, even when the rest of the sentence verifies.
- **Q8.** Is every magnitude, concentration, acreage, duration and emission figure character-for-character correct, including decimals? `7.5` ≠ `7.0`.
- **Q9.** Is each figure attached to the same hazard, fault, pollutant, resource or alternative the document attaches it to?
- **Q10.** For any superlative or bounding claim ("maximum credible", "most severe", "worst case", "largest"): does the cited page use that same framing, over the same scope?

## DECISION

1. Q7 = **no** → `RE_EXTRACT`, `failure_tag = T03_outside_text_fabrication`; `note` must quote the unsupported clause.
2. Q8 = **no** or Q9 = **no** → `RE_EXTRACT`, `failure_tag = T02_numeric_hallucination`.
3. Q10 = **no** → `RE_EXTRACT`, `failure_tag = T19_scope_qualifier_dropped`.
4. Otherwise fall through to the base decision table.

