# M2 prompt — `summary_of_interest`

A second, salience-weighted summary emitted **alongside** the five standard
`summary.*` subfields, never replacing them (MCAL_PLAN §3.15, build item #5).

Runs as a separate Opus reduce call after the standard summary reduce, over the
already-computed per-chunk findings rather than raw text — so its input is small
relative to the standard map-reduce.

Consumed by `segment_a/prompts.py`. Schema is enforced in
`mcal/artifacts/v(N)/atomic_schema.v(N).json` and checked by Critic rubric Q7.

**Audience default (tunable).** Written for *a researcher studying
environmental review and its policy and community consequences*. Per MCAL_PLAN
§3.15 and §8 this is the single knob most worth revisiting after reading a batch
of outputs — the opening line of the Purpose section is where to change it.

---

Produce a SECOND, separate summary called `summary_of_interest`. This is NOT a replacement for the standard summary — both are emitted and both are kept.

**Purpose.** Surface what a researcher studying environmental review and its policy and community consequences would find *notable* about this specific document, relative to a typical Environmental Impact Statement. Routine content belongs in the standard summary, not here.

**Salience criteria.** Include a claim only if it matches one of the following, and tag it with the criterion it matches:

- `contested` — the document records substantive disagreement: between agencies, between the agency and commenters, or among its own technical findings.
- `unusual_impact` — an impact category, affected population, or resource that is atypical for this project type.
- `large_magnitude` — the largest quantified impacts in the document (acreage, cost, displacement, emissions, duration, population affected).
- `novel_alternative` — an alternative beyond the standard no-action / preferred / minor-variant pattern.
- `community_pushback` — public comment that visibly changed the analysis, the scope, or the preferred alternative.
- `precedent` — the document explicitly frames itself as precedent-setting, first-of-kind, or programmatic for future actions.
- `cross_jurisdictional` — friction or coordination burden across agencies, states, or tribal nations.

**Rules.**

1. Every claim requires a page cite and a verbatim `evidence_quote` from that page — identical evidentiary standard to the standard summary. The `why_notable` sentence must also be grounded in the cited page, not in world knowledge about NEPA practice generally.
2. **If the document is routine and nothing meets the criteria above, return an empty list.** An empty `summary_of_interest` is a CORRECT and expected output for an unremarkable document. **Do NOT manufacture interest.** Most EISs are routine; a pipeline that finds something 'notable' in every document is producing noise.
3. Do not restate the standard summary. If a claim already appears in `summary.*` and does not independently meet a salience criterion above, leave it out.
4. Cap at 6 claims. If more than 6 qualify, keep the most contested and the largest-magnitude.
5. Apply the same plain-language and concreteness constraints as the standard summary — write for a reader with no NEPA background, gloss domain terms on first mention, name concrete entities and quantities.

**Output schema.** A JSON object with a single key `summary_of_interest` whose value is a list (possibly empty) of:

```json
{
  "claim": "string",
  "salience_criterion": "contested|unusual_impact|large_magnitude|novel_alternative|community_pushback|precedent|cross_jurisdictional",
  "evidence_quote": "string (verbatim from the cited page)",
  "why_notable": "one sentence, grounded in the cited page"
}
```

Return `{"summary_of_interest": []}` for a routine document.
