"""
Build a PowerPoint deck summarizing the pipeline change proposals in
PIPELINE_RESEARCH.md. Output: PIPELINE_RESEARCH.pptx in the same directory.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# Palette
NAVY = RGBColor(0x1F, 0x2A, 0x44)
STEEL = RGBColor(0x3E, 0x5C, 0x76)
ACCENT = RGBColor(0xC2, 0x5B, 0x3F)      # terracotta
MOSS = RGBColor(0x55, 0x74, 0x4A)        # keep / confirmed
AMBER = RGBColor(0xB2, 0x87, 0x1F)       # measure / caution
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
PAPER = RGBColor(0xF7, 0xF4, 0xEC)
RULE = RGBColor(0xD4, 0xCE, 0xC1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height

BLANK = prs.slide_layouts[6]


def add_slide():
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    bg.line.fill.background()
    bg.fill.solid()
    bg.fill.fore_color.rgb = PAPER
    bg.shadow.inherit = False
    return s


def add_text(slide, left, top, width, height, text, *,
             size=14, bold=False, color=INK, align=PP_ALIGN.LEFT,
             font="Helvetica"):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    lines = text.split("\n") if isinstance(text, str) else text
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def add_rect(slide, left, top, width, height, color, line_color=None):
    r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    r.fill.solid()
    r.fill.fore_color.rgb = color
    if line_color is None:
        r.line.fill.background()
    else:
        r.line.color.rgb = line_color
        r.line.width = Pt(0.75)
    r.shadow.inherit = False
    return r


def add_rule(slide, left, top, width):
    return add_rect(slide, left, top, width, Emu(9525), RULE)  # ~1pt line


def header(slide, eyebrow, title):
    """Standard section header with a thin rule."""
    add_text(slide, Inches(0.55), Inches(0.38), Inches(10), Inches(0.3),
             eyebrow.upper(), size=10, bold=True, color=ACCENT)
    add_text(slide, Inches(0.55), Inches(0.6), Inches(12.3), Inches(0.7),
             title, size=28, bold=True, color=NAVY)
    add_rule(slide, Inches(0.55), Inches(1.25), Inches(12.2))


def footer(slide, text):
    add_text(slide, Inches(0.55), Inches(7.1), Inches(12.3), Inches(0.3),
             text, size=9, color=MUTED)


def bullet(slide, left, top, width, items, size=14, line_gap=0.33,
           bold_lead=False):
    """items: list of (lead_bold, body) OR plain strings."""
    y = top
    for item in items:
        if isinstance(item, tuple):
            lead, body = item
        else:
            lead, body = None, item
        # bullet dot
        add_rect(slide, left, y + Inches(0.09), Inches(0.08), Inches(0.08),
                 STEEL)
        tb = slide.shapes.add_textbox(left + Inches(0.22), y,
                                      width - Inches(0.22), Inches(0.6))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = Emu(0); tf.margin_right = Emu(0)
        tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
        p = tf.paragraphs[0]
        if lead:
            r1 = p.add_run()
            r1.text = lead + " "
            r1.font.name = "Helvetica"; r1.font.size = Pt(size)
            r1.font.bold = True; r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.name = "Helvetica"; r2.font.size = Pt(size)
        r2.font.color.rgb = INK
        # Estimate height from wrap width
        # Rough: 85 chars per inch at 14pt. Over-estimate line count.
        chars_per_line = max(1, int(width.inches * 85 / (size / 11.0)))
        text_len = len(lead or "") + 1 + len(body)
        lines = max(1, (text_len + chars_per_line - 1) // chars_per_line)
        y += Inches(line_gap * lines)
    return y


# -----------------------------------------------------------------------------
# Slide 1 — Title
# -----------------------------------------------------------------------------
s = add_slide()
# Left accent bar
add_rect(s, 0, 0, Inches(0.35), SH, NAVY)
add_text(s, Inches(0.9), Inches(1.4), Inches(11), Inches(0.4),
         "EIS EXTRACTION PIPELINE", size=12, bold=True, color=ACCENT)
add_text(s, Inches(0.9), Inches(1.9), Inches(11.5), Inches(1.8),
         "What should change", size=58, bold=True, color=NAVY)
add_text(s, Inches(0.9), Inches(3.6), Inches(11.5), Inches(1.0),
         "A change report informed by two literature reviews\n"
         "(methodology + EIS/NEPA domain) and a repo-expert brief",
         size=20, color=STEEL)
add_rule(s, Inches(0.9), Inches(5.1), Inches(5))
add_text(s, Inches(0.9), Inches(5.25), Inches(11), Inches(0.4),
         "Companion to PIPELINE.md and PIPELINE_REVIEW.md",
         size=12, color=MUTED)
add_text(s, Inches(0.9), Inches(6.7), Inches(11), Inches(0.3),
         "17 proposed changes · 0 negations of existing design",
         size=11, bold=True, color=STEEL)

# -----------------------------------------------------------------------------
# Slide 2 — What we did
# -----------------------------------------------------------------------------
s = add_slide()
header(s, "Context", "How this report was produced")

add_text(s, Inches(0.55), Inches(1.5), Inches(12), Inches(0.4),
         "Three agents, run in parallel:", size=16, bold=True, color=NAVY)

# three columns
col_w = Inches(3.9); gap = Inches(0.25); x0 = Inches(0.55); y0 = Inches(2.0)
cols = [
    ("Repo expert",
     "Read PIPELINE.md and PIPELINE_REVIEW.md end-to-end. "
     "Produced a self-contained brief so the researchers didn't start cold.",
     NAVY),
    ("Methodology researcher",
     "Searched 2022–2026 literature on conformal prediction, grounding, "
     "LLM-as-judge, long-document extraction, and retry strategy.",
     STEEL),
    ("Domain researcher",
     "NEPABench / NEPATEC schema, 1973 CEQ guidelines, 1970s government OCR, "
     "role/era disambiguation, free-stack geocoding.",
     ACCENT),
]
for i, (title, body, color) in enumerate(cols):
    x = x0 + i * (col_w + gap)
    add_rect(s, x, y0, col_w, Inches(0.05), color)
    add_text(s, x, y0 + Inches(0.2), col_w, Inches(0.5),
             title, size=18, bold=True, color=color)
    add_text(s, x, y0 + Inches(0.85), col_w, Inches(3),
             body, size=13, color=INK)

add_text(s, Inches(0.55), Inches(5.4), Inches(12), Inches(0.4),
         "Operating constraints respected throughout:",
         size=14, bold=True, color=NAVY)
bullet(s, Inches(0.55), Inches(5.85), Inches(12.3), [
    "Measure the claim before writing the fix.",
    "New signals land at weight 0, or not at all (weights frozen through v3).",
    "Promoted artifacts are immutable; taxonomy is add-only.",
    "Record declines where the decision is visible.",
], size=12, line_gap=0.32)
footer(s, "Full research outputs in /tmp/eis-methodology-lit-review.md and "
          "/tmp/eis-domain-lit-review.md")

# -----------------------------------------------------------------------------
# Slide 3 — What the literature CONFIRMS (keep as-is)
# -----------------------------------------------------------------------------
s = add_slide()
header(s, "Section 0", "What the literature confirms — keep as-is")

add_text(s, Inches(0.55), Inches(1.5), Inches(12), Inches(0.5),
         "Four existing decisions are strengthened, not negated.",
         size=16, color=STEEL)

items = [
    ("Evidence-first Critic.",
     "LLM-as-judge literature (Thakur 2025 et al.) shows full-document "
     "judges inflate PASS rates via position / length / authority bias."),
    ("Blind sheet with composite-column removed.",
     "Raw blind-grade vs Critic agreement overstates κ by 33–41 points. "
     "The §2.4 fix was exactly the right instrument."),
    ("Chapter alias retrofit to 1970s vocabulary.",
     "The 1973 CEQ guidelines (38 FR 20550) codified the numbered-section "
     "template the pipeline now detects. Correct ground truth."),
    ("Removal of the s_citation evidence-level fallback.",
     "Multi-dim conformal literature (Zhao 2024, Blot 2025) is explicit that "
     "numerically redundant signals in composed scores are strictly harmful."),
]
y = Inches(2.3)
for title, body in items:
    add_rect(s, Inches(0.55), y + Inches(0.1), Inches(0.12), Inches(0.12), MOSS)
    add_text(s, Inches(0.85), y, Inches(11.8), Inches(0.4),
             title, size=16, bold=True, color=MOSS)
    add_text(s, Inches(0.85), y + Inches(0.42), Inches(11.8), Inches(0.8),
             body, size=13, color=INK)
    y += Inches(1.05)

footer(s, "No existing decision in PIPELINE.md is reversed by this report.")

# -----------------------------------------------------------------------------
# Slide 4 — Drop-in changes overview
# -----------------------------------------------------------------------------
s = add_slide()
header(s, "Section 1", "Drop-in changes — no refit, no stage boundary")

add_text(s, Inches(0.55), Inches(1.5), Inches(12.3), Inches(1.0),
         "Schema additions, data-resource swaps, and audit upgrades. "
         "None touch frozen weights, thresholds, or promoted artifacts.",
         size=16, color=STEEL)

tiles = [
    ("1.1", "Ship location on\na free stack",
     "GNIS + Census + Mordecai 3", "Defuses the 100% force-gate"),
    ("1.2", "Add project_sponsor\nand prepared_by",
     "In NEPATEC v2.0 / CEQ std.", "Closes a canonical gap"),
    ("1.3", "Benchmark on\nEISBench + MAPLEv2",
     "7 of 15 fields overlap", "Externally comparable numbers"),
    ("1.4", "Wikidata for\nrole / era",
     "P39 + P580/P582", "Replaces hand-coded table"),
    ("1.5", "Pin aliases to\n1973 CEQ + §7/§8",
     "38 FR 20550", "Named officials for key_people"),
    ("1.6", "LOO threshold\nstability + κ",
     "LOO-StabCP, ICLR 2025", "Makes thin-bucket honesty visible"),
]
tx, ty = Inches(0.55), Inches(2.8)
tw, th = Inches(4.0), Inches(2.0)
hgap, vgap = Inches(0.15), Inches(0.2)
for i, (num, title, src, benefit) in enumerate(tiles):
    col = i % 3; row = i // 3
    x = tx + col * (tw + hgap)
    y = ty + row * (th + vgap)
    add_rect(s, x, y, tw, th, WHITE, line_color=RULE)
    add_rect(s, x, y, Inches(0.4), th, NAVY)
    add_text(s, x, y + Inches(0.15), Inches(0.4), Inches(0.4),
             num, size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s, x + Inches(0.55), y + Inches(0.15), tw - Inches(0.65), Inches(0.9),
             title, size=15, bold=True, color=NAVY)
    add_text(s, x + Inches(0.55), y + Inches(1.1), tw - Inches(0.65), Inches(0.3),
             src, size=10, color=ACCENT, bold=True)
    add_text(s, x + Inches(0.55), y + Inches(1.4), tw - Inches(0.65), Inches(0.5),
             benefit, size=12, color=INK)

footer(s, "Each tile is one slide to follow.")

# -----------------------------------------------------------------------------
# Slide 5 — 1.1 Reduced-stack location (highest leverage)
# -----------------------------------------------------------------------------
s = add_slide()
header(s, "1.1 · Highest leverage", "Ship location on a free reduced stack")

add_rect(s, Inches(0.55), Inches(1.5), Inches(12.3), Inches(0.65), ACCENT)
add_text(s, Inches(0.75), Inches(1.6), Inches(12), Inches(0.5),
         "Today: location is 100% force-gated on every document.",
         size=16, bold=True, color=WHITE)
add_text(s, Inches(0.75), Inches(1.88), Inches(12), Inches(0.3),
         "Zero automated calibration data has been produced for this bucket.",
         size=12, color=WHITE)

add_text(s, Inches(0.55), Inches(2.5), Inches(12), Inches(0.4),
         "Finding from the domain review:", size=15, bold=True, color=NAVY)
add_text(s, Inches(0.55), Inches(2.9), Inches(12.3), Inches(1.0),
         "The CEQ data standard defines location at city / county / "
         "geographic-area resolution. GNIS covers that. PAD-US and Mapbox "
         "are luxury upgrades, not schema requirements.",
         size=14, color=INK)

add_text(s, Inches(0.55), Inches(4.1), Inches(12), Inches(0.4),
         "Minimum viable free stack:", size=15, bold=True, color=NAVY)
bullet(s, Inches(0.55), Inches(4.5), Inches(12.3), [
    ("GNIS (USGS Board on Geographic Names).",
     "~2M US features. Free, no key. Carries Census codes natively."),
    ("US Census Geocoder API.",
     "Free, no key, high rate ceiling. Native join target for GNIS."),
    ("Mordecai 3 (Halterman).",
     "Open-source neural geoparser. Designed for place names in prose, "
     "which matches 1970s EIS narrative text."),
], size=13, line_gap=0.38)

add_rect(s, Inches(0.55), Inches(6.5), Inches(12.3), Inches(0.5), MOSS)
add_text(s, Inches(0.75), Inches(6.58), Inches(12), Inches(0.4),
         "Why this is safe: adding data to an empty bucket is strictly "
         "additive — no stale thresholds get re-fit against a changed distribution.",
         size=11, bold=True, color=WHITE)

# -----------------------------------------------------------------------------
# Slide 6 — 1.2 Schema additions
# -----------------------------------------------------------------------------
s = add_slide()
header(s, "1.2", "Two canonical NEPATEC fields are missing")

add_text(s, Inches(0.55), Inches(1.5), Inches(12), Inches(0.5),
         "Both are explicit on 1970s cover and signature pages — and both are "
         "currently conflated into lead_agency.",
         size=15, color=STEEL)

# two cards
cards = [
    ("project_sponsor",
     "Often a non-federal entity: state agency, port authority, utility. "
     "On 1970s cover pages, named explicitly and distinct from the federal "
     "lead agency.",
     "Example: 'Prepared at the request of the Port of Seattle'"),
    ("prepared_by",
     "The contractor or consultant that drafted the EIS. A 1970s role that "
     "lead_agency cannot represent.",
     "Example: 'Prepared for the Corps of Engineers by Dames & Moore, Inc.'"),
]
y = Inches(2.4)
for name, body, example in cards:
    add_rect(s, Inches(0.55), y, Inches(12.3), Inches(1.9), WHITE, line_color=RULE)
    add_rect(s, Inches(0.55), y, Inches(0.12), Inches(1.9), ACCENT)
    add_text(s, Inches(0.85), y + Inches(0.15), Inches(5), Inches(0.5),
             name, size=20, bold=True, color=NAVY, font="Courier New")
    add_text(s, Inches(0.85), y + Inches(0.7), Inches(11.5), Inches(0.6),
             body, size=13, color=INK)
    add_text(s, Inches(0.85), y + Inches(1.45), Inches(11.5), Inches(0.4),
             example, size=11, color=MUTED)
    y += Inches(2.1)

add_text(s, Inches(0.55), Inches(6.7), Inches(12.3), Inches(0.5),
         "Add-only, following the pipeline's own convention for taxonomy codes.",
         size=11, bold=True, color=STEEL)

# -----------------------------------------------------------------------------
# Slide 7 — 1.3 Benchmark on EISBench + MAPLEv2
# -----------------------------------------------------------------------------
s = add_slide()
header(s, "1.3", "Get externally comparable numbers")

add_text(s, Inches(0.55), Inches(1.5), Inches(12.3), Inches(0.6),
         "NEPABench's EISBench (1,365 entries, 14 fields) and the MAPLEv2 "
         "evaluator overlap with 7 of this pipeline's 15 fields.",
         size=15, color=STEEL)

# overlap diagram
left_x = Inches(0.9)
right_x = Inches(7.5)
box_w = Inches(5)
# left box
add_rect(s, left_x, Inches(2.5), box_w, Inches(3.3), WHITE, line_color=RULE)
add_text(s, left_x, Inches(2.6), box_w, Inches(0.4),
         "THIS PIPELINE · 15 fields", size=11, bold=True, color=ACCENT,
         align=PP_ALIGN.CENTER)
left_only = ["summary.* (6 subfields)", "summary_of_interest",
             "key_people", "atom-level verification",
             "M-Cal conformal loop"]
add_text(s, left_x + Inches(0.3), Inches(3.1), box_w - Inches(0.6), Inches(2.6),
         "\n".join(["• " + x for x in left_only]), size=12, color=INK)

# right box
add_rect(s, right_x, Inches(2.5), box_w, Inches(3.3), WHITE, line_color=RULE)
add_text(s, right_x, Inches(2.6), box_w, Inches(0.4),
         "EISBench / MAPLEv2 · 14 fields", size=11, bold=True, color=STEEL,
         align=PP_ALIGN.CENTER)
right_only = ["project_sponsor (gap)",
              "prepared_by (gap)",
              "project_sector (controlled)",
              "process_family",
              "document_type / document_title split"]
add_text(s, right_x + Inches(0.3), Inches(3.1), box_w - Inches(0.6), Inches(2.6),
         "\n".join(["• " + x for x in right_only]), size=12, color=INK)

# center overlap
overlap_x = Inches(5.9)
add_rect(s, overlap_x, Inches(3.0), Inches(1.5), Inches(2.3), NAVY)
add_text(s, overlap_x, Inches(3.1), Inches(1.5), Inches(0.4),
         "SHARED", size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
shared = ["title", "year", "eis_type", "lead_agency",
          "location", "themes", "alternatives"]
add_text(s, overlap_x, Inches(3.5), Inches(1.5), Inches(1.9),
         "\n".join(shared), size=10, color=WHITE, align=PP_ALIGN.CENTER)

add_text(s, Inches(0.55), Inches(6.1), Inches(12.3), Inches(0.4),
         "Run MAPLEv2 on the current graded corpus. Zero-cost — evaluation "
         "only, no change to extraction or gating.",
         size=13, color=INK)
add_text(s, Inches(0.55), Inches(6.55), Inches(12.3), Inches(0.4),
         "The 7 shared fields become externally comparable; the 8 unique "
         "fields remain a genuine extension.",
         size=12, color=MUTED)

# -----------------------------------------------------------------------------
# Slide 8 — 1.4 Wikidata, 1.5 CEQ aliases (two-up)
# -----------------------------------------------------------------------------
s = add_slide()
header(s, "1.4 · 1.5", "Replace tables with authoritative sources")

col_w = Inches(6.0)
# Left: Wikidata
x = Inches(0.55); y = Inches(1.5)
add_rect(s, x, y, Inches(0.12), Inches(5.2), STEEL)
add_text(s, x + Inches(0.3), y, Inches(5.5), Inches(0.5),
         "1.4 · Wikidata role / era",
         size=18, bold=True, color=NAVY)
add_text(s, x + Inches(0.3), y + Inches(0.55), Inches(5.5), Inches(0.4),
         "Replace hand-coded table of (agency, year) → plausible officials",
         size=12, color=STEEL)
bullet(s, x + Inches(0.3), y + Inches(1.1), Inches(5.7), [
    ("P39.", "position held"),
    ("P580 / P582.", "start / end time qualifiers"),
    ("Cacheable nightly.", "one SPARQL query against query.wikidata.org"),
    ("Fallbacks.",
     "Federal Register API; American Presidency Project (UCSB)."),
], size=12, line_gap=0.42)
add_text(s, x + Inches(0.3), y + Inches(4.5), Inches(5.7), Inches(0.6),
         "Not a signal — a data-source swap. The key_people postproc already "
         "consumes the adjudicated year.",
         size=10, color=MUTED)

# Right: CEQ 1973
x = Inches(7.0); y = Inches(1.5)
add_rect(s, x, y, Inches(0.12), Inches(5.2), STEEL)
add_text(s, x + Inches(0.3), y, Inches(5.5), Inches(0.5),
         "1.5 · Pin aliases to 1973 CEQ",
         size=18, bold=True, color=NAVY)
add_text(s, x + Inches(0.3), y + Inches(0.55), Inches(5.5), Inches(0.4),
         "38 FR 20550, 1 August 1973 — the de jure source",
         size=12, color=STEEL)
bullet(s, x + Inches(0.3), y + Inches(1.1), Inches(5.7), [
    ("Current table.",
     "Inductive, fitted on 21 observed documents."),
    ("Risk.",
     "Inductive expansion is how the 1970s / modern vocabulary "
     "mismatch happened in the first place."),
    ("Bonus: §7 slice.",
     "Objections raised by other Federal/State/local agencies — "
     "named officials with dated affiliations."),
    ("Bonus: §8 slice.",
     "Discussion of problems and objections."),
], size=12, line_gap=0.42)
add_text(s, x + Inches(0.3), y + Inches(5.0), Inches(5.7), Inches(0.5),
         "§7 and §8 feed key_people directly — no role/era inference needed.",
         size=10, color=MUTED)

# -----------------------------------------------------------------------------
# Slide 9 — 1.6 LOO threshold stability
# -----------------------------------------------------------------------------
s = add_slide()
header(s, "1.6 · Hardest edge", "Make thin-bucket honesty visible")

add_rect(s, Inches(0.55), Inches(1.5), Inches(12.3), Inches(0.9), AMBER)
add_text(s, Inches(0.75), Inches(1.6), Inches(12), Inches(0.5),
         "Finding (methodology review):",
         size=12, bold=True, color=WHITE)
add_text(s, Inches(0.75), Inches(1.9), Inches(12), Inches(0.5),
         "Per-bucket thresholds with n ≈ 5–6 samples per bucket are below "
         "the α=0.05 split-conformal validity floor of 19.",
         size=14, bold=True, color=WHITE)

add_text(s, Inches(0.55), Inches(2.7), Inches(12.3), Inches(0.8),
         "The current refit ritual does real-looking work but produces "
         "high-variance estimates. The literature-aligned mitigation that "
         "respects \"don't touch what's frozen\" is to measure and surface the "
         "instability.",
         size=13, color=INK)

add_text(s, Inches(0.55), Inches(3.9), Inches(12.3), Inches(0.4),
         "Three audit diagnostics, zero-cost, every round:",
         size=14, bold=True, color=NAVY)
bullet(s, Inches(0.55), Inches(4.3), Inches(12.3), [
    ("LOO threshold stability.",
     "max |τ_k(fit) − τ_k(LOO)| per bucket. "
     "Flag buckets where LOO span exceeds the threshold interval. "
     "No auto-adjust. (Liang et al., ICLR 2025)"),
    ("Spearman ρ matrix across the 6 recorded signals.",
     "Pre-registered redundancy veto at ρ > 0.9. "
     "Prevents re-discovering 's_citation ≡ s_quote' retrospectively."),
    ("Headline is κ, not raw agreement.",
     "Raw blind-grade/Critic agreement overstates κ by 33–41 points "
     "(Thakur 2025)."),
], size=13, line_gap=0.42)

# -----------------------------------------------------------------------------
# Slide 10 — Section 2: Weight-0 signals (overview + table)
# -----------------------------------------------------------------------------
s = add_slide()
header(s, "Section 2", "Ten weight-0 signals to record now")

add_text(s, Inches(0.55), Inches(1.5), Inches(12.3), Inches(0.9),
         "Each is independent of s_quote and s_critic by construction, "
         "plausibly discriminative on items stuck at 1.0, and recorded only "
         "until weights unfreeze at v4+.",
         size=13, color=STEEL)

rows = [
    ("Signal", "Stage", "Addresses"),
    ("s_coverage_margin", "quote-gate", "Saturation tie-break (continuous 0.70→1.0)"),
    ("s_citation_breadth", "post-extract", "Saturation tie-break (cited_pages − 1)"),
    ("s_adjacent_concordance", "quote-gate", "Within-doc decoy for <3-token spans"),
    ("s_span_rarity", "quote-gate", "Common-bigram false matches down-weighted"),
    ("s_short_span_requires_corroboration", "quote-gate", "Audit flag → HUMAN_REVIEW"),
    ("s_recall_ceiling_exposed", "M2 input", "Measure if 600-page cap hurts"),
    ("s_critic_swap", "critic", "Verdict stability under quote paraphrase"),
    ("s_critic_adjacent_contrast", "critic", "Evidence-Lock mitigation (±1 pages)"),
    ("s_resample_agreement", "M1 / year / critic", "Non-temperature consistency"),
    ("s_prompt_paraphrase_delta", "retry", "Measure before replacing retry path"),
]

x0 = Inches(0.55); y0 = Inches(2.6)
widths = [Inches(5.0), Inches(2.3), Inches(5.0)]
row_h = Inches(0.37)
for i, row in enumerate(rows):
    is_header = (i == 0)
    bg = NAVY if is_header else (WHITE if i % 2 else PAPER)
    tx = x0
    for j, cell in enumerate(row):
        add_rect(s, tx, y0 + i * row_h, widths[j], row_h,
                 bg, line_color=(RULE if not is_header else None))
        add_text(s, tx + Inches(0.1), y0 + i * row_h + Inches(0.08),
                 widths[j] - Inches(0.15), row_h,
                 cell,
                 size=10 if is_header else 10,
                 bold=is_header,
                 color=WHITE if is_header else INK,
                 font="Courier New" if (j == 0 and not is_header) else "Helvetica")
        tx += widths[j]

footer(s, "All ten log to the manifest (reveal side, not blind). "
          "No weight change. No threshold change.")

# -----------------------------------------------------------------------------
# Slide 11 — Section 3: Four highest-value contradictions
# -----------------------------------------------------------------------------
s = add_slide()
header(s, "Section 3", "Four contradictions to pipeline assumptions")

add_text(s, Inches(0.55), Inches(1.5), Inches(12.3), Inches(0.6),
         "Where the external literature flatly contradicts an implicit or "
         "explicit pipeline assumption. Each is a checkable claim.",
         size=13, color=STEEL)

items = [
    ("Evidence-first Critic is only half-conservative",
     "Conservative against false-refutal; permissive against "
     "incompleteness-driven false-passing. "
     "Source: Evidence Lock, arXiv 2508.05353, 2025.",
     "Cheap fix: s_critic_adjacent_contrast at weight 0."),
    ("Per-bucket thresholds with n ≈ 5–6 are not finite-sample valid",
     "Split-conformal α=0.05 floor is 19 per stratum. "
     "Current refits are high-variance estimates.",
     "v3: LOO stability diagnostic. v4 question: pool buckets or go marginal."),
    ("Temperature retry has no literature support on Opus",
     "Opus ignores temperature. Self-consistency literature prefers "
     "paraphrase retry and N≥3 resample.",
     "Measurement first (s_prompt_paraphrase_delta); replacement at stage boundary."),
    ("Ratio-only on <3-content-token OCR spans is a known failure regime",
     "Nguyen et al. 2025: OCR-noise QA degrades sharply below ~15 characters.",
     "Audit flag now → HUMAN_REVIEW routing rule at v4."),
]
y = Inches(2.4)
for i, (title, claim, mitigation) in enumerate(items):
    add_rect(s, Inches(0.55), y, Inches(0.4), Inches(1.1), ACCENT)
    add_text(s, Inches(0.55), y + Inches(0.3), Inches(0.4), Inches(0.4),
             str(i + 1), size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s, Inches(1.1), y, Inches(11.6), Inches(0.4),
             title, size=14, bold=True, color=NAVY)
    add_text(s, Inches(1.1), y + Inches(0.35), Inches(11.6), Inches(0.5),
             claim, size=11, color=INK)
    add_text(s, Inches(1.1), y + Inches(0.78), Inches(11.6), Inches(0.4),
             "→ " + mitigation, size=11, bold=True, color=MOSS)
    y += Inches(1.2)

# -----------------------------------------------------------------------------
# Slide 12 — Section 4: Between-stage changes
# -----------------------------------------------------------------------------
s = add_slide()
header(s, "Section 4", "Between-stage changes — only at a stage boundary")

add_text(s, Inches(0.55), Inches(1.5), Inches(12.3), Inches(0.9),
         "These change extraction behavior and must be bundled with a rerun "
         "and refit, per the pipeline's own classification rules.",
         size=13, color=STEEL)

items = [
    ("Static prompt-paraphrase retry replacing temperature +0.2.",
     "Paraphrase is version-controlled, deterministic, independent of the first "
     "attempt. Costs the same as the current retry.",
     "Trigger: s_prompt_paraphrase_delta shows measurable lift."),
    ("Hierarchical two-level reduce past the 12-chunk cap.",
     "ToM 2025 (EMNLP, arXiv:2511.00489): reduce 1–6 → A, 7–12 → B, 13–16 → C, "
     "then reduce {A, B, C}. Keeps every call under caching reach.",
     "Trigger: s_recall_ceiling_exposed data shows the cap hurts a visible field."),
    ("Hard routing: <3-content-token quotes → HUMAN_REVIEW regardless of verdict.",
     "Removes a known high-false-accept regime from the automated path.",
     "Trigger: audit flag false-accept signal measurable at next refit."),
]
y = Inches(2.6)
for title, body, trigger in items:
    add_rect(s, Inches(0.55), y, Inches(12.3), Inches(1.3), WHITE, line_color=RULE)
    add_rect(s, Inches(0.55), y, Inches(0.12), Inches(1.3), STEEL)
    add_text(s, Inches(0.85), y + Inches(0.1), Inches(11.8), Inches(0.4),
             title, size=14, bold=True, color=NAVY)
    add_text(s, Inches(0.85), y + Inches(0.5), Inches(11.8), Inches(0.5),
             body, size=11, color=INK)
    add_text(s, Inches(0.85), y + Inches(1.0), Inches(11.8), Inches(0.3),
             "⟶ " + trigger, size=11, bold=True, color=AMBER)
    y += Inches(1.45)

# -----------------------------------------------------------------------------
# Slide 13 — Experiments to run first
# -----------------------------------------------------------------------------
s = add_slide()
header(s, "Section 5", "Three experiments to run before anything in §4")

add_text(s, Inches(0.55), Inches(1.5), Inches(12.3), Inches(0.6),
         "Each answers a question that gates a between-stage change. All cheap.",
         size=14, color=STEEL)

exps = [
    ("Re-OCR a 50-doc sample.",
     "Run through olmOCR 2 (AI2) or Churro. Compare Gate A/B pass-rate uplift.",
     "If OCR quality is the actual bottleneck, the recall-ceiling and "
     "short-span problems may shrink before they are 'solved.'"),
    ("Read Raab et al. 2025 (arXiv:2507.08109).",
     "Audit / Alignment / Optimization of LM-Powered Subroutines.",
     "Substantively the same calibration-of-LM-subroutines problem as M-Cal. "
     "Worth a close read before freezing v3 signal weights."),
    ("Run MAPLEv2 on the current graded corpus.",
     "On the 7 shared fields (title, year, eis_type, lead_agency, location, "
     "themes, alternatives).",
     "Externally comparable baseline — the number before any changes land is "
     "the benchmark for every subsequent claim."),
]
y = Inches(2.3)
for i, (title, how, why) in enumerate(exps):
    add_rect(s, Inches(0.55), y, Inches(0.5), Inches(1.4), NAVY)
    add_text(s, Inches(0.55), y + Inches(0.45), Inches(0.5), Inches(0.5),
             str(i + 1), size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s, Inches(1.2), y + Inches(0.05), Inches(11.5), Inches(0.5),
             title, size=16, bold=True, color=NAVY)
    add_text(s, Inches(1.2), y + Inches(0.5), Inches(11.5), Inches(0.4),
             how, size=12, color=INK)
    add_text(s, Inches(1.2), y + Inches(0.95), Inches(11.5), Inches(0.5),
             why, size=11, color=MUTED)
    y += Inches(1.6)

# -----------------------------------------------------------------------------
# Slide 14 — Do NOT do
# -----------------------------------------------------------------------------
s = add_slide()
header(s, "Section 6", "Explicitly do NOT do")

add_text(s, Inches(0.55), Inches(1.5), Inches(12.3), Inches(0.8),
         "Each is tempting but either (a) already evaluated and declined "
         "for good reason, or (b) not supported by the literature in this "
         "setting.",
         size=13, color=STEEL)

donts = [
    ("Add atom→chunk attribution to resurrect s_shard.",
     "Infrastructure cost without demonstrated discriminative lift."),
    ("Replace the two-gate quote check.",
     "Stress-tested on 602 within-document decoys. Augment, don't substitute."),
    ("Enable full-document prompt caching for the Critic.",
     "Would undo the Evidence-Lock mitigation and inflate PASS rates."),
    ("Unfreeze weights before n ≈ 60.",
     "Small-sample conformal is where threshold variance is worst."),
    ("Expand aliases inductively beyond the 1973 CEQ guidelines.",
     "Inductive expansion is how the mismatch happened the first time."),
]
y = Inches(2.6)
for title, why in donts:
    add_rect(s, Inches(0.55), y + Inches(0.1), Inches(0.3), Inches(0.3), ACCENT)
    add_text(s, Inches(0.55), y + Inches(0.1), Inches(0.3), Inches(0.3),
             "✗", size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s, Inches(1.0), y, Inches(11.8), Inches(0.4),
             title, size=14, bold=True, color=NAVY)
    add_text(s, Inches(1.0), y + Inches(0.37), Inches(11.8), Inches(0.3),
             why, size=11, color=INK)
    y += Inches(0.78)

# -----------------------------------------------------------------------------
# Slide 15 — Priority order
# -----------------------------------------------------------------------------
s = add_slide()
header(s, "Section 7", "Priority order")

add_text(s, Inches(0.55), Inches(1.5), Inches(12.3), Inches(0.5),
         "Grouped by cost and risk, highest leverage first.",
         size=14, color=STEEL)

prios = [
    ("1", "§1.1 Reduced-stack location",
     "Defuses a 100% force-gate.", MOSS),
    ("2", "§1.2 project_sponsor + prepared_by schema add",
     "Closes a canonical gap.", MOSS),
    ("3", "§1.3 MAPLEv2 on shared fields",
     "Externally comparable numbers on the record.", MOSS),
    ("4", "§1.6 LOO stability + Spearman ρ matrix + κ-as-headline",
     "Makes current calibration honesty visible.", AMBER),
    ("5", "§2 weight-0 signals (all 10)",
     "Logged in parallel; one pass through the manifest writer.", AMBER),
    ("6", "§1.4 Wikidata role/era + §1.5 1973 CEQ aliases",
     "Domain wins that don't touch calibration.", AMBER),
    ("7", "§5 experiments",
     "Gate §4 changes.", STEEL),
    ("8", "§4 between-stage changes",
     "Only after a rerun is budgeted.", STEEL),
]
y = Inches(2.2)
for num, title, body, color in prios:
    add_rect(s, Inches(0.55), y, Inches(0.5), Inches(0.5), color)
    add_text(s, Inches(0.55), y + Inches(0.08), Inches(0.5), Inches(0.4),
             num, size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s, Inches(1.3), y + Inches(0.03), Inches(11.6), Inches(0.4),
             title, size=13, bold=True, color=NAVY)
    add_text(s, Inches(1.3), y + Inches(0.3), Inches(11.6), Inches(0.3),
             body, size=11, color=INK)
    y += Inches(0.56)

# Legend
y_leg = Inches(6.8)
leg = [("Zero-cost / drop-in", MOSS), ("Audit / weight-0", AMBER),
       ("Experiment / later", STEEL)]
x_leg = Inches(0.55)
for label, color in leg:
    add_rect(s, x_leg, y_leg + Inches(0.08), Inches(0.2), Inches(0.2), color)
    add_text(s, x_leg + Inches(0.3), y_leg + Inches(0.05), Inches(2.5),
             Inches(0.3), label, size=10, color=MUTED)
    x_leg += Inches(2.6)

# -----------------------------------------------------------------------------
# Slide 16 — Closing
# -----------------------------------------------------------------------------
s = add_slide()
add_rect(s, 0, 0, Inches(0.35), SH, NAVY)
add_text(s, Inches(0.9), Inches(1.4), Inches(11), Inches(0.4),
         "SUMMARY", size=12, bold=True, color=ACCENT)
add_text(s, Inches(0.9), Inches(1.9), Inches(11.5), Inches(1.4),
         "17 proposed changes.\nZero negations of existing design.",
         size=42, bold=True, color=NAVY)
add_rule(s, Inches(0.9), Inches(4.0), Inches(5))

add_text(s, Inches(0.9), Inches(4.2), Inches(11.5), Inches(0.4),
         "Four current decisions are strengthened by external evidence.",
         size=15, color=STEEL)
add_text(s, Inches(0.9), Inches(4.6), Inches(11.5), Inches(0.4),
         "Five existing decisions are extended, not replaced.",
         size=15, color=STEEL)
add_text(s, Inches(0.9), Inches(5.0), Inches(11.5), Inches(0.4),
         "Three previously unforeseen upgrades are proposed.",
         size=15, color=STEEL)

add_text(s, Inches(0.9), Inches(6.0), Inches(11.5), Inches(0.4),
         "Full report:", size=11, color=MUTED)
add_text(s, Inches(0.9), Inches(6.25), Inches(11.5), Inches(0.4),
         "PIPELINE_RESEARCH.md", size=14, bold=True, color=NAVY,
         font="Courier New")

out_path = "/Users/gracegormley/Desktop/EIS Job/spring_pipeline/metadata_pipeline/PIPELINE_RESEARCH.pptx"
prs.save(out_path)
print("saved:", out_path)
print("slides:", len(prs.slides.__iter__.__self__._sldIdLst))
