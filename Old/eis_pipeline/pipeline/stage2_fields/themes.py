"""
Stage 2.2 — Theme classification.
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import TYPE_CHECKING

from ..config import ALL_SUBTHEMES, ALL_THEMES, MODELS, THEMES
from ..schema import EISRecord, ThemesField

if TYPE_CHECKING:
    from ..llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = textwrap.dedent("""\
    You are classifying a U.S. Environmental Impact Statement into a controlled theme taxonomy.

    Primary themes (pick 1–2):
    {themes_list}

    For each chosen primary theme, pick 2–5 subthemes from its list.
    Use "other" / "unclassified" only if nothing fits — and note that >30% "other" rate
    signals a vocabulary gap, so prefer a close fit over "other" when reasonable.

    Respond with ONLY valid JSON:
    {{
      "primary": ["theme_key", ...],
      "subthemes": ["subtheme_key", ...]
    }}
""")

_USER_TMPL = textwrap.dedent("""\
    Document title: {title}

    Summary:
    {summary}
""")


def run(record: EISRecord, client: "LLMClient") -> None:
    """Populate record.themes. Mutates record in place."""
    themes_list = "\n".join(
        f"  {theme}: {json.dumps(subs)}" for theme, subs in THEMES.items()
    )
    summary_text = record.summary.text if record.summary else ""

    user_msg = _USER_TMPL.format(
        title=record.title or record.doc_id,
        summary=summary_text or "(no summary available)",
    )

    try:
        result = client.call_json(
            model=MODELS["heavy"],
            system=_SYSTEM.format(themes_list=themes_list),
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=256,
            temperature=0.1,
            label=f"themes/{record.doc_id}",
        )
        primary = [t for t in (result.get("primary") or []) if t in ALL_THEMES]
        subthemes = [s for s in (result.get("subthemes") or []) if s in ALL_SUBTHEMES]

        if not primary:
            logger.warning("No valid primary themes returned for %s", record.doc_id)
            primary = ["other"]
            subthemes = ["unclassified"]

        record.themes = ThemesField(primary=primary, subthemes=subthemes)
    except Exception as exc:
        logger.error("Theme classification failed for %s: %s", record.doc_id, exc)
        record.themes = ThemesField(primary=["other"], subthemes=["unclassified"])
