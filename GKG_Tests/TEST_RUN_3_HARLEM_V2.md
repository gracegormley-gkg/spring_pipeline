# Test Run 3 — Harlem Ave V2 (post 5/14-meeting fixes)

**Date:** 2026-05-15
**Document:** FAP Route 42 (Illinois Route 43) Harlem Avenue, Cook County, Illinois
**Accession key:** `p1074_35556036099737`
**Output:** `GKG_Tests/harlem_v2.json`
**Compared against:** `harlem_ave_result.json` (Test Run 2, 2026-05-13)

This run validates the post-meeting changes: tightened heading detector,
retrieval fallback, layered NER (en_core_web_trf + tribes/NGOs dicts + Haiku
gap-fill), expanded key-people schema (role + opinion summary), appearance-order
sort, layman summary, token ledger.

---

## Cost & Performance

| | This run (v2) | Previous (v1) | Δ |
|---|---|---|---|
| Total tokens | 330,315 | 201,935 | **+64%** |
| Cost | **$4.42** | $2.13 | **+$2.29** |
| Chunks | 13 | 39 | −67% |
| Key people | 16 | 5 | +220% |
| Run time | ~3 min | ~1.5 min | |

**Budget overrun:** The $4.00 cap was hit during Stage 3 (critic), which means
the summary's critic verdict was skipped. All other fields completed. This is
the budget guard working correctly, but I underestimated. The cost increase is
driven almost entirely by more entities going through stance + quote extraction
(16 × 2 Opus calls ≈ 32 calls vs. 5 × 2 = 10 calls before).

---

## Heading detector ✅ Major improvement / ⚠️ User intuition is correct

**Sections: 39 → 13.** The address/letterhead noise is gone — no more
"143 SOUTH THIRD STREET" or "5 F\nUNITED STATES DEPARTMENT OF AGRICULTURE" in
the sections list. All five new regression tests prevent that class of failure.

**But all 13 sections are still named "Section 4(f) ..."** because the only
"Section"-prefixed strings in this document are legal citations to NEPA Section
102(2)(c) and DOT Act Section 4(f). The 1972 typewritten document has no formal
structural headings — no "PURPOSE AND NEED" or "AFFECTED ENVIRONMENT" caps
banners. So the regex is doing exactly what we asked (matching valid heading-like
strings, rejecting addresses) but it's matching legal citations because that's
all the document offers.

**The user's intuition is exactly right:** an AI-gathered TOC is the real fix.
Two possible designs:
- **Cheap:** Haiku reads the first 3 pages of the doc and is asked "list the
  major sections of this document with their start pages." Then we map to chunks.
- **Better:** Haiku reads every chunk header (first ~500 words of each fixed-page
  chunk) and labels them; we use the labels to decide where real section
  boundaries are, then re-merge chunks at those boundaries.

Either way, this is the next high-value fix. The deterministic regex is now
*precise* (low false positives) but *recall is gated by document formatting* —
docs without typeset headings won't expose structural sections this way.

**Chunk size distribution is the visible symptom of this:**

| Chunk | Pages | Words | Tags |
|---|---|---|---|
| c01 | 1–1 | 381 | proposed_action, affected_environment, alternatives, ... |
| c02 | 1–1 | 381 | proposed_action, consultation, appendix |
| **c03** | **1–23** | **8,906** | proposed_action, affected_environment, alternatives, mitigation |
| c04 | 24–24 | 414 | comments_and_responses, mitigation |
| c05 | 24–28 | 2,020 | comments_and_responses, mitigation, consultation |
| c06 | 29–32 | 1,569 | consultation, comments_and_responses, mitigation |
| c07 | 33–33 | 384 | mitigation, consultation, comments_and_responses |
| c08 | 33–35 | 1,123 | comments_and_responses, consultation, mitigation |
| c09 | 36–36 | 365 | comments_and_responses, consultation |
| **c10** | **36–52** | **7,038** | comments_and_responses, consultation, mitigation |
| c11 | 53–53 | 410 | proposed_action, consultation, mitigation |
| **c12** | **53–64** | **5,090** | proposed_action, alternatives, mitigation, consultation |
| c13 | 65–67 | 1,162 | proposed_action, mitigation, consultation |

Three chunks (c03, c10, c12) hold 21k of the 27k words. The seven smallest
chunks are sub-page fragments. This is what "AI-gathered TOC" would fix.

---

## Summary ✅ Detailed strong, layman works as intended

**Detailed (180 words):** Reads cleanly, covers all four required elements
accurately. Cites c03 throughout (correct — that's the main body). Mentions the
3.0-mile widening, 16-foot median, Tinley Creek 3-acre acquisition with 4.8-acre
replacement, Navajo Creek flooding concern. This is a strong factual summary.

**Layman (148 words):** Successful translation. Example diff:
> Detailed: "approximately 3 acres of Cook County Forest Preserve District land
> (Tinley Creek Division) for backslopes, diminishing the tract's natural
> resource preservation value"
>
> Layman: "About 3 acres of forest preserve land would be needed, but the
> county planned to replace it with 4.8 acres of new preserved land"

It dropped the bureaucratic precision ("Tinley Creek Division", "backslopes",
"diminishing the tract's natural resource preservation value") and kept the
substance. This is exactly the desired transformation. Verdict: layman summary
is shipping-quality.

**Note:** Critic skipped due to budget overrun, so no LLM verification of the
detailed summary this run. The summary still looks correct, but the
"critic_error: summary: Budget exceeded" warning means we don't have an
independent verdict.

---

## Alternatives ❌ Regression — returned empty, was 9 last time

This is a real regression and the most concerning result. Last run extracted 9
correctly described alternatives (No Action, Alignment 1, Alignment 2, Variations
A–E, A-1). This run returned an empty list.

**Root cause:** The previous run had 39 chunks; one was specifically labeled
`alternatives` and contained the focused alternatives discussion. This run has
13 chunks with a much larger chunk c03 (8,906 words, pages 1–23) that contains
the alternatives content but is also tagged with 4 other topics. The Stage 2
alternatives module retrieved chunks c01, c03, c12 (all tagged `alternatives`),
fed Opus c01+c03+c12 totaling ~14k words, and Opus returned `{"alternatives": []}`.

Possible reasons:
1. **Naming mismatch.** The doc calls them "Alignment 1", "Variation A", etc.,
   not "Alternative 1". The Opus prompt asks for "each named alternative" — Opus
   may not have recognized "Variation A" as an alternative.
2. **Diluted context.** A focused 1.5k-word alternatives chunk is easier to
   extract from than a 9k-word chunk where alternatives are one of many topics.
3. **Truncation.** `combine_chunk_context` defaults to 60k chars — c01+c03+c12
   is ~80k chars. The end of c12 may have been truncated, which is where the
   recommended alternative is summarized.

**The fix:** Strengthen the alternatives prompt to recognize "Alternative",
"Alignment", "Variation", "Option", and similar terminology. Also consider
splitting the giant c03 into sub-chunks. **Higher priority:** verify alternatives
isn't silently failing on other docs in the collection.

---

## Themes ✅ Strong, same as last run

`transportation` / `highways_and_roads`. No regression.

---

## Location ✅ Correct

"Cook County, Illinois" → (41.82, -87.76). Close to but slightly different from
last run's "Harlem Avenue, Cook County, Illinois" → (41.83, -87.80). Both are
correct; "Cook County" is more abstract but still maps to roughly the right
place.

---

## Key People & Groups ✅ Major improvement — biggest win of this iteration

**16 entities with rich metadata** (was 5 with mostly empty fields). The new
`role` and `opinion_summary` fields are populated for the majority. Sorted by
document appearance order as requested.

Highlights:
- **Cook County Forest Preserve District** — correctly captured with
  `stance="mixed"`, `role="consulted agency"`, and an `opinion_summary` that
  specifically names the "little or no value" objection. This is the key
  stakeholder that v1 missed entirely.
- **Federal Highway Administration** — caught by Haiku gap-fill (`source:
  haiku_gapfill`). Wasn't in spaCy's output.
- **Illinois Department of Transportation** — correctly identified as lead
  agency with substantive opinion summary.
- **City of Palos Heights** — captured as `consulted agency`, mayor's comments
  acknowledged.
- **Richard H. Golterman** (Chief Highway Engineer) — role identified
  correctly. Stance correctly marked `insufficient_information` because the
  document doesn't expose his personal view.

Weaker spots:
- **"United States"** appears as an organization with role "lead agency
  (funding and oversight via Federal Highway Administration)" — this is a way
  too generic entity that the triage should have rejected.
- **"STATE OF ILLINOIS"** appears as its own entity, separate from "Illinois
  Department of Transportation". Both have nearly-identical opinion summaries.
  This is a dedupe failure caused by all-caps OCR formatting in some sections.
- **"Northeastern Illinois Planning"** — truncated org name (should be
  "...Planning Commission"). spaCy gave us the truncated form and triage
  accepted it.
- **No quotes anywhere** (0/16 quotes verified). Either the Opus quote
  extraction is consistently failing the verbatim substring check, or this doc
  genuinely has no quotable passages. Worth investigating — last run also got 0
  verified quotes but produced 1 entity entry.

**Appearance order sort works** — Cook County (c01) is order 1; Arthur L. Janura
(c10) is order 16. Increasing chunk_id throughout.

---

## NER raw output ⚠️ Still messy — dedupe is the bottleneck

Final raw NER: **53 people, 262 organizations.** The trf model is finding a lot,
but dedupe is failing on text variants:

- `Cook County Forest Preserve District` and `the Cook County Forest Preserve
  District` and `Cook County Forest\nPreserve District` and `Cook County Forest
  Pre-` are all kept as separate entities. The dedup compares lowercased strings
  exactly — it doesn't strip leading "the" or normalize line-break splits or
  hyphen-truncations.
- `"AVE"`, `"BLACK"`, `"Concur"`, `"Liason Offi-\ncer"`, `"No\ncer"`,
  `"Right of Way"` — pure OCR garbage that survives because spaCy ORG is
  permissive.
- `"Sigmung C. Ziejewski"` and `"Sigmund C. Ziejewski"` (typo) and
  `"S.C. Ziejewski"` (initials) — three forms of the same person, all kept.

The triage step does filter these out of the *final output* (most don't make it
into the 16 final entities), but they bloat the NER list and slow the rule
filter. Worth a dedup-normalization pass that:
1. Strips leading "the ", "The ", "The\n"
2. Replaces internal newlines and hyphens-at-line-break with spaces
3. Drops fragments under 3 alphabetic characters
4. Optionally: fuzzy-merges variants whose normalized forms have >0.85 similarity

---

## Dictionary lookups — partial success

The new tribes/NGOs/agencies dicts found:
- **Agencies (dict_agency):** Department of Transportation, Housing and Urban
  Development, Department of Agriculture, Department of Interior, U.S. Army
  Corps of Engineers, Federal Railroad Administration, National Park Service,
  Environmental Protection Agency. 8 deterministic catches.
- **NGOs (dict_ngo):** Conservation Fund. (Only one — this doc has few real
  environmental NGOs because it's a 1972 road-widening EIS with mostly state
  agencies in consultation.)
- **Tribes (dict_tribe):** Zero — no tribal stakeholders in this Chicago-area
  road project. Expected.

**The dict pass works as designed** — it found the agencies deterministically
without needing spaCy to catch them. The provenance tracking (`sources` field
in NERResult) makes it clear which entities came from which layer.

---

## Haiku gap-fill — worked, found real stakeholders

Source attribution shows 5 entities tagged `haiku_gapfill`:
- Federal Highway Administration
- Department of the Interior (duplicate of agency dict — gap-fill shouldn't have
  re-added this; need to dedupe against existing entities more carefully)
- Cook County Forest Preserve District
- Northeastern Illinois Planning Commission
- Cook County Forest Preserve Commission

The first three are real stakeholders that strengthened the final output. The
last two are partial duplicates of spaCy-caught entities (variant spellings).

**One issue:** the dedup check in `_gap_fill_ner` uses `n.lower() in
existing_names` — but `existing_names` is a set of names, not a substring check.
Need to also check for substring inclusion (e.g., "Cook County Forest Preserve
Commission" vs the spaCy variant "the Cook County Forest Preserve Commission").

---

## EIS type ⚠️ Still "Unlabelled" — fix was scoped out

Same root cause as last run: title says "FINAL", consultation appendix
references "the DRAFT". The title-check tiebreaker fix was deferred since it
wasn't in the meeting scope.

## Date ⚠️ Still 1971 — fix was scoped out

Same root cause: extractor pulls the earliest date, which is a 1971 draft
submission stamp.

## Lead Agency ⚠️ Still vague "Department of Transportation"

NUL API has no contributor for this accession. The agency dict catches both DOT
and FHWA in the body text but the dict picks the first match. Worth adding an
FHWA-specific second-pass since this is a highway doc.

---

## Token Ledger ✅ Working

`output/token_ledger.json` was created with this run logged. Per-model breakdown
captured. Lifetime totals computed correctly. Ready to accumulate across runs.

---

## Issues to Fix Before Next Run

| Priority | Issue | Fix |
|---|---|---|
| **High** | Alternatives regression — returned empty | Expand alternatives prompt to recognize "Alignment", "Variation", "Option"; check `combine_chunk_context` truncation on large multi-tag chunks |
| **High** | Chunk size highly uneven (3 chunks hold 80% of words) | **AI-gathered TOC**: Haiku reads doc to identify real structural sections; use those for chunking instead of regex-only |
| **High** | Cost doubled ($2.13 → $4.42) | Consolidate stance + quote into one Opus call; skip quote when `opinion_summary` is null; or cap final entities at ~10 instead of all 16 |
| Medium | NER dedupe doesn't normalize variants | Strip "the ", collapse newlines/hyphen-breaks, drop short fragments, optionally fuzzy-merge |
| Medium | Gap-fill re-adds entities already in dict layer | Use substring-aware dedup in `_gap_fill_ner` |
| Medium | "United States" and "STATE OF ILLINOIS" appear in final key-people output | Strengthen triage prompt to reject generic country/state names |
| Low | Quotes consistently failing substring check (0/16 verified) | Investigate whether OCR text mismatches are breaking the check, or whether Opus is paraphrasing |
| Low | EIS type, date, lead agency — all unchanged from last run | Scoped out of this iteration; queue for next |

---

## Overall Assessment

**This iteration delivered on the meeting goals:**
- Heading detector is much more precise (no more address noise)
- NER overhaul produced 3× more real stakeholders with substantive role +
  opinion fields
- Cook County Forest Preserve District (the key missing stakeholder from v1) is
  now correctly captured
- Layman summary works as intended
- Token ledger is in place
- Sort by appearance order is in place

**But two new concerns surfaced:**
1. **Alternatives regression.** This is the most urgent — we *lost* a working
   field. Worth re-running with the prompt fix before declaring v2 stable.
2. **Cost doubled.** Mostly a function of "cut it down less" plus the extra
   stance+role call. At $4.42/doc, the 181-doc collection would cost ~$800.
   Worth the consolidation pass before running the full collection.

**The user's intuition about AI-gathered TOC is correct.** That's the next
high-value piece of work and the right framing for the chunking problem.
Deterministic regex caps the recall at "what the document formats as a
heading," which for typewritten 1970s docs is often nothing structural.
