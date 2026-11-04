# M2 summary clause — plain language + concreteness

Appended verbatim to each `summary.*` map and reduce prompt in
`segment_a/m2.py` (MCAL_PLAN §3.14, build item #4).

This file is the single source of truth for the clause. It is consumed by:

- `segment_a/prompts.py` → the M2 map and reduce prompts (generation side)
- `mcal/critic_prompt.py` → Critic rubric Q6 (verification side)

Keeping one copy matters: Q6 asks whether the extractor obeyed this clause, so
if the two drifted apart the Critic would be grading against a rubric the
generator never saw.

**Do not edit without bumping `PROMPT_VERSION` in `segment_a/prompts.py`.**
`mcal/build.py` refuses to calibrate unless the M2 output on disk was produced
under the current version (MCAL_PLAN §3.7 step-0 precheck), because τ must be
fitted to the same prose Segment B will ship.

---

Write for a reader with no background in NEPA or federal environmental review. Two constraints:

**1. Plain language.** When you use a domain-specific term (e.g. 'cumulative effects', 'tiered review', 'Section 106 consultation', 'de minimis finding', 'Preferred Alternative', 'scoping', 'programmatic EIS'), briefly gloss it in-line on first mention within this subfield, using text drawn from or directly supported by the cited pages. Same for regulatory citations ('40 CFR §1502', 'NEPA §101'). Prefer plain nouns and active voice over nominalizations and passive constructions. Acronym expansion is handled by a separate post-processor; you do NOT need to expand acronyms yourself, but you MUST NOT use undefined domain jargon.

**2. Concreteness.** Describe the project and its impacts in terms a non-specialist reader can visualize and act on:

- Name the thing being built or decided, not its regulatory category. ('A 47-mile 500-kV transmission line from Substation X to Substation Y' — not 'a linear energy infrastructure element'.)
- Specify quantities, locations, and durations when the document provides them. ('The project would affect approximately 1,200 acres of sagebrush habitat over 30 years' — not 'the project has land-use implications'.)
- Name affected communities concretely. ('Residents of three census tracts in south Milwaukee; the Ho-Chunk Nation; two elementary schools within 500 feet of the alignment' — not 'nearby stakeholders'.)
- State the decision under review plainly. ('The Bureau of Land Management is deciding whether to approve, approve with modifications, or deny the proposed right-of-way' — not 'the agency is engaged in a decisional process'.)

**Both constraints are bounded by the cited pages.** Every plain-language rephrasing or concrete detail must still be supported by the cited pages — do not invent glosses or specifics from world knowledge. If the document does not support a plain-language rephrasing or a concrete specification, quote the document's own language verbatim and leave it unglossed rather than fabricating.

**Preserve scope qualifiers exactly.** When the document limits a figure or a
superlative to a subset — 'for the Rail/Bus Alternatives I–V, costs range
from…', 'the maximum credible event', 'within the starter line area' — carry
that qualifier into your text. Dropping it converts a true scoped claim into a
false general one, and a downstream verifier cannot detect the error because
the figure itself is genuinely present in the source.
