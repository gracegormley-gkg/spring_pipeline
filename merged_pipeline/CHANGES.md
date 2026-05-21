# Pipeline Changes Log

## 2026-05-21 — Section span-extension fix

### Bug

`pipeline/sections.py:_regex_detect`, `pipeline/sections_ai_toc.py`, and
`pipeline/sections_embedding.py` all set each detected section's `char_span`
to **only the heading text itself** (e.g. the 7 characters spelling
`"SUMMARY"`), not heading-to-next-heading.

Concretely, for the lead-paint EIS (`p1074_35556036813020`):

```json
"summary": {
  "value": null,
  "status": "needs_review",
  "provenance": {
    "source": "regex",
    "char_offset_raw": [561, 568],   // ← 7-char span: just the word "SUMMARY"
    "section": "summary"
  }
}
```

Downstream consequence: `pipeline/retrieval.get_section_text` slices the raw
text using `char_span`, so Stage 3 producers received 7-character windows
containing only the heading word. The LLM correctly returned
`sufficient_information: false`, and:

- `summary.value`         stayed `null`
- `layman_summary.value`  stayed `null` (gated on `summary`)
- `themes.primary[]`      stayed empty (gated on `summary`)
- `alternatives[]`        empty when `alternatives` heading was matched
- `stakeholders[]`        empty when `public_comments` heading was matched

The bug was masked on the original Castaic-Haskell smoke-test doc
(`p1074_35556035057348`) because its short text had no regex section hits at
all — Stage 2 marked everything `not_found` and Stage 3 fell through to
keyword-search fallback (1500-char windows). The bug only surfaced on docs
where Stage 2 *succeeded* in finding a heading.

### Fix

Added `_extend_spans_to_next_heading()` to `pipeline/sections.py` and called
it after the regex / AI-TOC / embedding cascade completes (right before the
"stub `not_found`" pass).

It does, in place:

1. Collects all non-cover sections that have `status == "ok"` and a
   `char_span`.
2. Sorts them by `char_span[0]` (heading start position).
3. For each section, sets its span end to either:
   - the start of the next heading in the sorted list, or
   - end of document if it's the last detected section.
4. Re-derives `pages` from the new span via `ingest.page_range_for_span`.

Cover is intentionally left alone — it's a deterministic first-N-pages
metadata span and is allowed to overlap with the start of summary /
purpose_and_need so Stage 3a can extract title/agency from cover-page text.

The fix is detector-agnostic: regex, AI-TOC, and embedding spans all get
extended uniformly, since they all originally suffered from the same
heading-only-span behavior.

### Verification

- All 27 existing tests in `claude_tests/test_sections.py` still pass.
- `p1074_35556036813020` (lead-paint EIS, ~700KB) now produces:
  - non-null `summary.value` (~180 words)
  - non-null `layman_summary.value` (80-120 words)
  - populated `themes.primary[]` and `subthemes[]`
  - much wider `summary.provenance.char_offset_raw` (heading → next heading)

### Files changed

- `pipeline/sections.py`
  - inserted `_extend_spans_to_next_heading()` helper
  - call it once at the end of the detection cascade in `run()`
