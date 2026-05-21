# Test Run 2 — Harlem Avenue Highway Improvement EIS

**Date:** 2026-05-13
**Document:** FAP Route 42 (Illinois Route 43) Harlem Avenue (119th Street to 143rd Street), Cook County, Illinois
**Accession key:** `p1074_35556036099737`
**Input source:** `docs_with_digits.json`
**Output:** `GKG_Test 1/harlem_ave_result.json`

---

## Document Profile

| | |
|---|---|
| Title | Environmental Impact Statement, FAP Route 42 (Illinois Route 43) Harlem Avenue (119th Street to 143rd Street) Cook County, Illinois |
| Lead agency | Department of Transportation (DOT) — via regex |
| Year | 1971 (see date issue below) |
| Word count | 26,908 (medium) |
| Pages (fake) | 67 |
| EIS type | Unlabelled (see below) |
| Chunks | 39 |

---

## Cost & Performance

| | |
|---|---|
| Total tokens | 201,935 |
| Total cost | $2.13 |
| Warnings | 1 (`ocr_confidence_unavailable` — expected, using JSON source) |
| Apollo comparison | $0.39 → $2.13 (5.5× more — expected: 39 chunks vs. 1) |

Cost scales roughly with chunk count. At $2.13 for a medium doc, the 181-doc collection would cost ~$385 at this rate — within range but worth tracking as we move to longer docs.

---

## Field-by-Field Results

### Summary ✅ Strong
The summary is detailed and accurate: it correctly names the project (Harlem Ave widening, 2-lane to 4-lane divided highway with 16-foot median, 119th–143rd St), the community affected (City of Palos Heights), the reason (traffic congestion on a major north-south arterial), and the key environmental impact (3-acre acquisition from Cook County Forest Preserve District's Tinley Creek Division, offset by 4.8 acres of replacement land). It also captures the flooding concern near 127th Street/Navajo Creek raised by citizens at the public hearing. Evidence is cited to three chunks (c04, c32, c35). This is a meaningful improvement over Apollo — the multi-chunk retrieval path worked.

### Location ✅ Correct
Extracted "Harlem Avenue, Cook County, Illinois" and geocoded it to lat 41.83, lon -87.80. This is accurate — the Harlem Ave corridor on the southwest side of Chicago metro.

### Themes ✅ Strong — major improvement over Apollo
Correctly classified as `transportation` / `highways_and_roads`. This is the main thing this test was designed to verify, and it worked cleanly. No "other" fallback.

### Alternatives ✅ Excellent
Returned 9 alternatives with accurate descriptions:
- No Action (Alternative A)
- Alternative B, Alignment 1 (new right-of-way)
- Alternative B, Alignment 2 (existing right-of-way — recommended)
- Variations A, B, C, D, E, and A-1 for the Palos Heights shopping area segment

All are real alternatives from the document, correctly named and described. The recommended alignment and variation are both correctly identified and flagged. This is exactly what the Apollo run missed, and it worked well here because the chunk labeler correctly tagged c04 with `alternatives`.

### Key People & Groups ⚠️ Running but weak output
spaCy ran this time (740 deduped entities from 1,169 raw), which is progress. But only 5 entities made the final output, and quality is poor:
- **"Cook County"** — `insufficient_information`. Correct that it's present, but this is a place name, not a stakeholder.
- **"Richard H. Golterman"** — `neutral`. The only person with any evidence. Likely a minor agency official. Not the most important stakeholder.
- **"Cook County Forest"** — `insufficient_information`. A truncated version of "Cook County Forest Preserve District," which is actually the key objecting party in this document. Should have been classified as `opposed` or `mixed`.
- **"Department of Transportation"** — `insufficient_information`. The lead agency; not a meaningful stakeholder entry.
- **"Illinois Route 43"** — `insufficient_information`. This is a road name, not a person or organization. Triage noise.

**What's missing:** The Cook County Forest Preserve District (objected to the 3-acre taking and the "little or no value" framing), the City of Palos Heights (involved in the shopping area alignment decision), and citizens who raised flooding concerns at the public hearing. These are the genuine stakeholders. The NER list contains their names — e.g. Henry N. Barkhausen, Peter B. Bensinger, Thomas G. Cots — but they appear to be individual state agency directors who provided routine consultation letters, not stakeholders with strong stances.

**Root cause:** The consultation appendix of this EIS consists largely of form-letter responses from 20+ Illinois state agency directors, which floods the NER list with official names who have no meaningful stance on the project. The triage step passed some of these through, then Opus correctly assigned them `insufficient_information`. The actual objecting party (Forest Preserve District as an institution) got deduplicated into fragments. This is a known limitation of the current approach on docs where most named entities are consultation correspondents rather than stakeholders.

### Historical Context — Internal ⚠️ Correctly empty
Returned `insufficient_information`. This is the right answer: a 1972 Final EIS for a road widening has no historical backstory to surface. The model correctly recognized this rather than fabricating context. Same result as Apollo, same correct behavior.

### EIS Type ⚠️ Unlabelled again — same root cause as Apollo
The document cover clearly states "FINAL COMBINED ENVIRONMENTAL/SECTION 4(f) STATEMENT" in large type. But the consultation appendix repeatedly references "the DRAFT Environmental/Section 4(f) Statement" (the prior version sent to agencies for comment), which triggers the Draft pattern. The triage logic sees both signals and returns Unlabelled.

**Root cause confirmed:** The "Draft" references in Final EIS documents always refer to the prior draft version, not the document itself. The fix proposed in the Apollo report (prefer "Final" when it appears in the title) would resolve this.

### Lead Agency ⚠️ Vague — NUL API had no contributor for this doc
Got "Department of Transportation" via regex. Technically correct but doesn't distinguish between FHWA (federal) and IDOT (state) — both are prominently involved. The NUL API returned no contributor field for this accession, so the fallback regex fired. The actual lead agency by NEPA standards is the Federal Highway Administration (FHWA), with the Illinois Department of Transportation preparing the document. A closer read of the cover would resolve this.

### Date ⚠️ Off by one year
Returned 1971, but the document was signed April 25, 1972 and accepted by FHWA June 5, 1972. The 1971 date comes from a visible timestamp in the consultation appendix: "SEP 1 3 1971" (the date the draft was sent to the Council on Environmental Quality). The correct publication year is 1972. **Root cause:** The date extractor is pulling the first date-like string from the early pages, which happens to be the draft's CEQ submission date, not the final document date. This is a fake-page pagination issue — the consultation appendix content is appearing in what the pipeline treats as "early pages."

---

## Section Detection — New Issue

The 39 detected sections are largely noise. Examples from the output:

- `"Section 4"` (appears ~10 times — these are legal citations, not section headings)
- `"143 SOUTH THIRD STREET\nPHILADELPHIA"` (a mailing address from a consultation letter)
- `"5 F\nUNITED STATES JEPARTMENT OF AGRICULTURE\nSOIL CONSERVATION SERVICE\nP"` (letterhead)
- `"35 AIT\nSEP"` (OCR fragment)
- `"75 TH\nAVL AVE\nTOTE"` (map label)

**Root cause:** This document is ~60% consultation appendix — dozens of agency response letters, each with a letterhead address. The heading detector's regex is matching all-caps lines and numeric section references, so it fires on "Section 4(f)" citations, street addresses, and letterhead. This produced 39 chunks instead of the 5–8 that would result from a clean section-based split (Summary, Description, Impacts, Alternatives, Consultation, Appendices). The 39-chunk split likely contributed to the key people quality problem: stakeholder-relevant content got spread across many tiny consultation-letter chunks that never got coherent tags.

---

## Issues to Fix Before Next Run

| Priority | Issue | Fix |
|----------|-------|-----|
| High | Section detector firing on legal citations ("Section 4", "Section 102") and mailing addresses | Add a filter to `stage0_triage._detect_headings()`: reject matches shorter than ~4 words, reject matches that are purely numeric or contain street/address patterns |
| High | EIS type: "Final" in title should break the tie | Add title-check tiebreaker in `stage0_triage._eis_type()`: if title contains "Final" and no other ambiguity, prefer Final |
| Medium | Date extractor picking up draft submission date instead of publication date | Prioritize dates near signature blocks (look for "Date" label nearby) over bare dates in the first pages |
| Medium | Key people: consultation-appendix letter signatories dominate NER list | In triage prompt, add instruction to deprioritize names that appear only in agency consultation lists/letterheads; favor names with substantive quotes or objections |
| Low | Lead agency: NUL API missing contributor → vague DOT match | Add a second-pass regex for FHWA specifically ("Federal Highway Administration") in the agency matcher |

---

## Overall Assessment

This run confirmed the pipeline handles medium-length multi-chunk documents correctly. The three fields that failed or were empty in Apollo — summary, themes, and alternatives — all performed well here. Alternatives extraction in particular was excellent: 9 correctly described alternatives from a document with a detailed Section IV. The multi-chunk retrieval architecture worked as designed.

The new issues exposed here are distinct from Apollo's: the heading detector misbehaves on documents where the majority of the text is a consultation appendix full of letterhead and legal citations. This is a common EIS structure (Final EIS = main statement + all agency comment letters + responses), so fixing the heading detector will matter for a large fraction of the collection.

**Next recommended test:** A longer doc (60k–100k words) with a clean section structure — a land management or dam EIS from the late 1970s–80s where the main body dominates over the consultation appendix. Candidates: `p1074_35556036806552` (28k), `p1074_35556036093169` (29k) or any of the 50k-range p1074 docs. Also worth trying the companion Route 77 doc (`p1074_35556036100261`, 26k) to see if the section detection issue reproduces on a structurally similar document.
