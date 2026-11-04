"""
Stage 3g — Theme classification.

Strategy (synthesis_plan.md §Themes):
  - Input: record.summary.value + record.title.value
  - Model: Haiku (cheap; classification, not synthesis)
  - Hard gate: out-of-vocab primaries dropped; orphaned subthemes (whose parent
    isn't in chosen primaries) dropped; field -> "needs_review" if anything
    was dropped.
  - Abstention: if record.summary.status != "ok", do NOT call the LLM. Instead
    set themes.status = "skipped_summary_unavailable" with primary=[], subthemes=[].

Per synthesis_plan: themes are downstream of summary. If summary failed or is
degraded, themes inherit that uncertainty rather than papering over it with
guesses against the title alone.

Vocab is config.THEMES (13 primary + per-primary scoped subthemes). Schema's
ThemesField validator enforces the parent-must-be-chosen-primary rule at
record-construction time, so this stage hard-filters before assignment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from . import config
from .schema import EISRecord, SubthemeEntry, ThemeEntry, ThemesField

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def run(
    record: EISRecord,
    llm: "LLMClient | None" = None,
) -> list[str]:
    """Mutate `record.themes`. Returns warnings.

    Abstains (without calling LLM) when summary isn't `ok`. Returns themes
    with status='skipped_summary_unavailable' in that case.
    """
    warnings: list[str] = []

    summary_value = record.summary.value
    summary_status = record.summary.status
    title_value = record.title.value or record.publication_id

    # Abstention: bad summary -> abstain entirely.
    if summary_status != "ok" or not summary_value:
        record.themes = ThemesField(
            primary=[],
            subthemes=[],
            status="skipped_summary_unavailable",
        )
        warnings.append(
            f"themes: skipped (summary.status={summary_status!r}, "
            f"value={'present' if summary_value else 'empty'})"
        )
        return warnings

    if llm is None:
        record.themes = ThemesField(
            primary=[], subthemes=[],
            status="needs_review",
        )
        warnings.append("themes: no llm provided")
        return warnings

    raw_primaries, raw_subthemes = _haiku_classify(summary_value, title_value, llm)
    if raw_primaries is None:
        record.themes = ThemesField(
            primary=[], subthemes=[],
            status="needs_review",
        )
        warnings.append("themes: Haiku classification failed")
        return warnings

    primaries, subthemes, drops = _vocab_gate(raw_primaries, raw_subthemes)

    status = "needs_review" if drops else "ok"
    record.themes = ThemesField(
        primary=primaries,
        subthemes=subthemes,
        status=status,
    )
    if drops:
        warnings.append(
            f"themes: vocab gate dropped {len(drops)} entries -> {drops}"
        )

    return warnings


# ---------------------------------------------------------------------------
# Vocab gate (hard filter)
# ---------------------------------------------------------------------------

def _vocab_gate(
    raw_primaries: list[dict],
    raw_subthemes: list[dict],
) -> tuple[list[ThemeEntry], list[SubthemeEntry], list[str]]:
    """Drop OOV primaries, drop subthemes without a chosen primary parent.
    Returns (primaries, subthemes, list_of_dropped_entries_for_logging)."""
    drops: list[str] = []

    # Pass 1: primaries against the 13-item vocab.
    primaries: list[ThemeEntry] = []
    seen_primary: set[str] = set()
    for p in raw_primaries:
        if not isinstance(p, dict):
            drops.append(f"primary:non-dict({p!r})")
            continue
        value = p.get("value")
        if value not in config.ALL_PRIMARY_THEMES:
            drops.append(f"primary:oov({value!r})")
            continue
        if value in seen_primary:
            continue  # de-dupe
        conf = p.get("confidence")
        if not isinstance(conf, (int, float)):
            conf = 0.5
        primaries.append(ThemeEntry(value=value, confidence=float(conf)))
        seen_primary.add(value)

    chosen_primary_set = {p.value for p in primaries}

    # Pass 2: subthemes — must be in vocab AND parent must be a chosen primary.
    subthemes: list[SubthemeEntry] = []
    seen_subtheme: set[tuple[str, str]] = set()
    for s in raw_subthemes:
        if not isinstance(s, dict):
            drops.append(f"subtheme:non-dict({s!r})")
            continue
        value = s.get("value")
        parent = s.get("parent")
        if not isinstance(value, str) or not isinstance(parent, str):
            drops.append(f"subtheme:bad-shape({s!r})")
            continue
        if value not in config.ALL_SUBTHEMES:
            drops.append(f"subtheme:oov({value!r})")
            continue
        if parent not in chosen_primary_set:
            drops.append(f"subtheme:orphan({value!r}, parent={parent!r})")
            continue
        # The subtheme must actually be listed under that parent in config.THEMES
        if value not in config.THEMES.get(parent, []):
            drops.append(f"subtheme:wrong-parent({value!r} not under {parent!r})")
            continue
        if (value, parent) in seen_subtheme:
            continue
        conf = s.get("confidence")
        if not isinstance(conf, (int, float)):
            conf = 0.5
        subthemes.append(SubthemeEntry(value=value, confidence=float(conf), parent=parent))
        seen_subtheme.add((value, parent))

    return primaries, subthemes, drops


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _haiku_classify(
    summary: str,
    title: str,
    llm: "LLMClient",
) -> tuple[list[dict] | None, list[dict] | None]:
    prompt_template = (PROMPTS_DIR / "3g_themes.txt").read_text(encoding="utf-8")
    primary_list = "\n".join(f"  - {p}" for p in config.ALL_PRIMARY_THEMES)
    subthemes_by_primary = "\n".join(
        f"  - {p}: {json.dumps(subs)}" for p, subs in config.THEMES.items()
    )
    prompt = (
        prompt_template
        .replace("{primary_list}", primary_list)
        .replace("{subthemes_by_primary}", subthemes_by_primary)
        .replace("{title}", title)
        .replace("{summary}", summary)
    )

    try:
        result = llm.call_json(
            model=llm.models["haiku"],
            system="You are a careful classifier. Return only the requested JSON.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
            label="3g_themes",
        )
    except Exception as exc:
        logger.warning("3g themes LLM call failed: %s", exc)
        return None, None

    if not isinstance(result, dict):
        return None, None
    primary = result.get("primary")
    subthemes = result.get("subthemes")
    if not isinstance(primary, list):
        primary = []
    if not isinstance(subthemes, list):
        subthemes = []
    return primary, subthemes
