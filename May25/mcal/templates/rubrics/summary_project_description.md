# Rubric overlay — `summary.project_description`

One numeric error and two missing-citation defects across 8 docs. The
numeric case (LA Transit, `$659 million (Alt. V)`) is **not** a fabrication —
the figure is on pp.31/214/215 and correctly paired with Alt. V. The document
scopes it to \"the Rail/Bus Alternatives I–V\"; the summary dropped that
restriction and presented it as the overall minimum. That is `T19`, not `T02`.

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

2–4 sentences: what is physically proposed, at what scale, and what
decision is before the agency.

## QUESTIONS

- **Q7.** Is the thing being built or decided named concretely ("a 47-mile 500-kV transmission line"), rather than by regulatory category ("a linear energy infrastructure element")?
- **Q8.** For every cost, length, acreage or capacity figure: is the figure attached to the same alternative or entity the document attaches it to?
- **Q9.** For every range or superlative ("costs range from X", "the largest", "the minimum"): does the document state that range over the SAME set the value implies? Check for a restricting phrase such as "for Alternatives I–V" or "of the rail options".

## DECISION

1. Q8 = **no** → `RE_EXTRACT`, `failure_tag = T02_numeric_hallucination`.
2. Q9 = **no** → `RE_EXTRACT`, `failure_tag = T19_scope_qualifier_dropped`; `note` must quote the qualifier the value dropped.
3. Q7 = **no** → `PASS_WITH_NOTE`, `failure_tag = T16_abstract_when_concrete_available`.
4. Otherwise fall through to the base decision table.

