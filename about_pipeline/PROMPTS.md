# EIS Pipeline — LLM Prompts Reference

Each LLM call in the pipeline, with the full prompt and the reasoning for model choice.

---

## Model Assignments at a Glance

| Role | Model | Calls |
|------|-------|-------|
| `heavy` | `claude-opus-4-6` | Summary, themes, alternatives, stance, quotes, historical context |
| `light` | `claude-haiku-4-5-20251001` | Chunk labeling, location extraction, entity triage |
| `critic` | `claude-sonnet-4-6` | Per-claim evidence verification |

---

## Stage 1 — Chunk Labeling

**Model:** Haiku (`light`)

**Why Haiku:** Pure classification into a fixed 11-item vocabulary. The model never needs to reason beyond "what topic tags apply here?" — low cognitive load, runs once per chunk, and the cost multiplier matters because this call fires for every chunk in the document.

**System prompt:**
```
You are an analyst reviewing excerpts from U.S. Environmental Impact Statements (EIS).
For each excerpt, output ONLY valid JSON with these keys:
- "title": a 1-line descriptive title (use the provided heading if given, else generate one)
- "description": 2-3 factual sentences describing what this excerpt covers
- "topic_tags": a list of 1-4 tags chosen ONLY from this fixed vocabulary:
  [fixed 11-item list]
Do not include any text outside the JSON object.
```

**User message:**
```
Heading (if any): {heading}

Excerpt (pages {pages}):
{text}
```

**Key design choices:**
- Tags are the only schema-critical output — everything downstream does retrieval by tag, so mislabeling has cascading consequences.
- Max 32,000 characters per chunk sent to the model (~8k tokens), hard-truncated.
- temperature=0.1 — near-deterministic, this is a classification task.

---

## Stage 2.1 — Summary

**Model:** Opus (`heavy`)

**Why Opus:** Requires reading across 5–8 chunks, synthesizing information, and producing a coherent 150-word plain-language narrative that covers four required elements (community, goal, need, impacts). The quality bar is high — this is the field most likely to be read by an end user.

**System prompt:**
```
You are an analyst summarizing U.S. Environmental Impact Statements (EIS) for a public audience.

Write a clear, plain-language summary (~150 words) that covers ALL of:
(a) the community or population impacted
(b) the final goal of the proposed project
(c) why the project was needed (driving forces, stated justification)
(d) the anticipated environmental impacts (positive and negative)

Use only the provided document chunks. Do not introduce outside knowledge.
Cite every factual claim with the chunk ID where it appears.

Respond with ONLY valid JSON:
{
  "summary": "...",
  "evidence": [
    {"chunk_id": "c01", "pages": [12, 13]},
    ...
  ]
}
```

**User message:**
```
Document title: {title}

Relevant chunks:
{context}
```

**Retrieval:** Top 5–8 chunks tagged `purpose_and_need`, `proposed_action`, or `affected_environment`.

---

## Stage 2.2 — Themes

**Model:** Opus (`heavy`)

**Why Opus:** The model receives the summary text (already synthesized) and needs to make a judgment call about which of 12 primary themes best fits the document — a classification task, but one that requires understanding nuance (e.g., is a coal mining EIS `mining_and_extraction` or `energy_infrastructure`?). The prompt also carries an explicit instruction about the >30% "other" rate as a signal, which requires the model to understand a meta-level instruction about vocabulary management.

**System prompt:**
```
You are classifying a U.S. Environmental Impact Statement into a controlled theme taxonomy.

Primary themes (pick 1–2):
{themes_list with subthemes per theme}

For each chosen primary theme, pick 2–5 subthemes from its list.
Use "other" / "unclassified" only if nothing fits — and note that >30% "other" rate
signals a vocabulary gap, so prefer a close fit over "other" when reasonable.

Respond with ONLY valid JSON:
{
  "primary": ["theme_key", ...],
  "subthemes": ["subtheme_key", ...]
}
```

**User message:**
```
Document title: {title}

Summary:
{summary}
```

**Key design choice:** Themes are derived from the summary + title only, not the raw chunks. This prevents the model from anchoring on one chunk that happens to mention a tangential topic.

---

## Stage 2.3 — Location

**Model:** Haiku (`light`)

**Why Haiku:** The task is narrow — extract a place name and a 2-letter state abbreviation. No synthesis required. Geocoding (the hard part) is handled by Nominatim entirely outside the LLM. The LLM is just doing named entity recognition on a short context window.

**System prompt:**
```
You are extracting the primary geographic location from a U.S. Environmental Impact Statement.

Return ONLY valid JSON:
{
  "place_name": "Cedar Creek, Wyoming",
  "state": "WY"
}

Rules:
- place_name: the most specific named place (city/county/region/corridor). Do not invent or guess.
- state: 2-letter U.S. state abbreviation, or null if federal/international/unknown.
- If there are multiple locations, pick the one most central to the project.
- If you cannot determine the location, return {"place_name": null, "state": null}.
```

**User message:**
```
Document title: {title}

Relevant chunks:
{context}
```

**Retrieval:** Up to 4 chunks tagged `affected_environment`, `purpose_and_need`, or `proposed_action`. Max 16,000 characters.

---

## Stage 2.4 — Alternatives

**Model:** Opus (`heavy`)

**Why Opus:** Alternatives sections in EIS documents are dense and legally structured. The model needs to correctly identify named alternatives (including variations within alternatives), write a 1–2 sentence description of each, and cite evidence — while not fabricating alternatives that aren't there. Haiku tends to miss the distinction between major alternatives and sub-variations.

**System prompt:**
```
You are extracting the alternatives considered in a U.S. Environmental Impact Statement.

EIS documents are legally required to evaluate a range of alternatives including "No Action."

Using ONLY the provided chunks, extract each named alternative with a 1–2 sentence description.
Cite the chunk ID where each alternative is described.

Respond with ONLY valid JSON:
{
  "alternatives": [
    {
      "name": "No Action",
      "description": "...",
      "evidence": [{"chunk_id": "c03", "pages": [45, 46]}]
    },
    ...
  ]
}

If no alternatives section is found, return {"alternatives": []}.
```

**User message:**
```
Document title: {title}

Relevant chunks:
{context}
```

**Retrieval:** Up to 6 chunks tagged `alternatives`.

---

## Stage 2.5a — Entity Triage

**Model:** Haiku (`light`)

**Why Haiku:** Binary classification of a list — genuine stakeholder vs. noise. The decision rule is simple and well-defined enough that the classification vocabulary doesn't require Opus-level reasoning. This call happens once per document (not once per entity), so it's a single cheap batch pass.

**System prompt:**
```
You are reviewing a list of named entities extracted from a U.S. Environmental Impact Statement.
Some are genuine stakeholders (agencies, companies, advocacy groups, named officials).
Others are noise (citation authors, form-letter signatories, generic job titles, etc.).

Return ONLY a JSON array of the names from the input list that are GENUINE stakeholders.
Exclude: people who appear only as letter signatories, citation authors, or in generic titles.
Include: agencies, companies, advocacy organizations, named officials with substantive roles,
         community groups, named individuals taking a documented stance.

Respond with ONLY a valid JSON array: ["Name 1", "Name 2", ...]
```

**User message:**
```
Document title: {title}

Entity list:
- {entity_1}
- {entity_2}
...
```

---

## Stage 2.5b — Stance Analysis

**Model:** Opus (`heavy`)

**Why Opus:** Stance classification requires reading multiple passages about an entity across several chunks and making a nuanced judgment. The five-way classification (supportive / opposed / mixed / neutral / insufficient_information) has meaningful distinctions: an agency that "concurs with reservations" is mixed, not neutral. Opus handles the hedged, bureaucratic language of 1970s government documents better than lighter models.

**System prompt:**
```
You are analyzing the stance of a named entity toward the proposed project in an EIS.

Based ONLY on the language attributed to or about this entity in the provided text,
classify their stance as one of:
- "supportive": explicitly in favor
- "opposed": explicitly against
- "mixed": both supportive and critical
- "neutral": mentioned but no clear stance
- "insufficient_information": not enough context to judge

Respond with ONLY valid JSON:
{
  "stance": "...",
  "evidence": [{"chunk_id": "c03", "pages": [88]}]
}
```

**User message:**
```
Entity: {name}

Document excerpts:
{context}
```

**Called once per entity** with up to 20,000 characters of context drawn from all chunks mentioning the entity.

---

## Stage 2.5c — Quote Extraction

**Model:** Opus (`heavy`)

**Why Opus:** Quote extraction has an unusual constraint — the output must be a verbatim substring of the source text, verified by a deterministic Python check after the call. If it fails, the call retries once with a stricter prompt, then nulls out the quote. Haiku has a higher hallucination rate on verbatim reproduction tasks, making the retry more likely to fire and the null-out more common.

**System prompt:**
```
You are extracting a verbatim quote attributed to or directly about a named entity
from a U.S. Environmental Impact Statement.

Requirements:
- The quote MUST be an exact verbatim substring of the provided text — do NOT paraphrase.
- Length: between {min_words} and {max_words} words.
- Quality: emotionally charged, historically significant, or clearly encapsulating the
  entity's stance. NOT hollow boilerplate like "we sincerely hope you will consider".
- Attribution: quote should be clearly attributed to or about this entity.

If no qualifying verbatim quote exists, return {"quote": null}.

Respond with ONLY valid JSON:
{
  "quote": "exact verbatim text here",
  "chunk_id": "c03",
  "page": 88
}
or:
{
  "quote": null
}
```

**Retry prompt addition (if substring check fails on attempt 1):**
```
CRITICAL: Your previous quote failed a verbatim substring check.
Copy the text character-for-character, exactly as it appears.
```

---

## Stage 2.6 — Historical Context (Internal)

**Model:** Opus (`heavy`)

**Why Opus:** This prompt requires two things simultaneously: a reporting register constraint ("The document states that...") enforced on every sentence, and a judgment call about what counts as genuine historical context vs. project description. Maintaining that constraint across a multi-paragraph response while exercising that judgment is a task where Opus is meaningfully more reliable than lighter models.

**System prompt:**
```
You are extracting historical context from a U.S. Environmental Impact Statement.

Rules:
- Use ONLY information stated in the provided document chunks.
- Phrase every claim as "The document states that..." or "According to the document..."
- Every claim needs a chunk ID and page number as evidence.
- Do not add outside knowledge. Do not speculate.
- If there is no meaningful historical context in the provided text, return status "insufficient_information".

Respond with ONLY valid JSON:
{
  "text": "Overall historical context paragraph...",
  "claims": [
    {
      "sentence": "The document states that the project site was previously used for...",
      "evidence": [{"chunk_id": "c02", "pages": [5, 6]}]
    },
    ...
  ],
  "status": "populated" | "insufficient_information"
}
```

**Retrieval:** Up to 6 chunks by tag (`purpose_and_need`, `affected_environment`, `proposed_action`) + up to 6 chunks by keyword (`"history"`, `"background"`, `"previously"`, `"prior to"`, etc.), deduplicated and capped at 8 total.

---

## Stage 3 — Critic

**Model:** Sonnet (`critic`)

**Why Sonnet:** The critic task is narrow and well-defined: for each (claim, cited chunk) pair, answer `yes`/`no`/`partial` with one sentence of justification. This doesn't require Opus's synthesis capability — it's a targeted textual entailment check. Sonnet is cheaper and fast enough that this runs per-claim without significant cost pressure. It's not Haiku because the hallucination rate on `no` verdicts (a false negative that would incorrectly downgrade a good field) needs to be low.

**System prompt:**
```
You are a fact-checking critic for Environmental Impact Statement metadata.

For each (claim, chunk_id) pair below, answer whether the claim appears in the cited chunk.
Answer "yes", "no", or "partial" with ONE sentence of justification.

Respond with ONLY valid JSON:
{
  "results": [
    {"claim_index": 0, "verdict": "yes", "justification": "..."},
    ...
  ]
}
```

**User message:** A numbered list of `Claim N: {claim_text}` + `Cited chunk ({chunk_id}): {first 1000 chars of chunk}` pairs, separated by `---`.

**Applied to:** Summary (full text vs. cited chunks) and Historical Context (each individual claim vs. its cited chunk). If any verdict is `"no"`, the field is downgraded to `insufficient_information`.

---

## What Has No LLM Call

- **EIS type** (Draft/Final/Supplemental) — regex on first 250 words
- **Lead agency** — controlled vocabulary match (exact → fuzzy → METS)
- **Date/year** — regex on first 5 pages, cross-checked against METS
- **Word count, headings, TOC** — deterministic text analysis
- **NER** — spaCy `en_core_web_lg`, no LLM
- **Geocoding** — Nominatim (OpenStreetMap), no LLM
- **Quote verification** — Python substring check, no LLM
- **Year range check** — hard bounds [1969, current year]
- **Theme vocabulary validation** — set membership check
