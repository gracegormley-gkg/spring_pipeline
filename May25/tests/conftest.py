"""
pytest configuration for the May25 pipeline.

`mcal` is a real package under May25/, but `segment_a` is not -- it uses flat
imports. Adding May25/ to sys.path lets tests `import mcal`; mcal.settings then
installs the segment_a bridge on import.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MAY25_ROOT = Path(__file__).resolve().parent.parent
if str(MAY25_ROOT) not in sys.path:
    sys.path.insert(0, str(MAY25_ROOT))

from mcal import settings  # noqa: E402


# --- Fixtures ---------------------------------------------------------------

# Docs used across regression tests, with the failure each one exhibits.
LINCOLN_HWY = "p1074_35556039563135"   # wildlife-clause fabrication (MCAL_PLAN 1(4))
LA_TRANSIT = "p1074_35556038322269"    # magnitude + cost figures (MCAL_PLAN 1(3),(4))
BUFFALO = "p1074_35556036811230"       # empty alternatives (MCAL_PLAN 1(8))
OPERATION_BREAKTHROUGH = "p1074_35556036058550"


def _has_doc(doc_id: str) -> bool:
    return settings.resolve_doc_dir(doc_id) is not None


needs_corpus = pytest.mark.skipif(
    not settings.PAGES_DATA_DIR.exists(),
    reason="per-page OCR JSON not materialized on this machine",
)


@pytest.fixture(scope="session")
def doc_loader():
    """Lazy, cached doc loader. Loading a 347-page doc is not free."""
    from pages import load_doc

    cache: dict[str, object] = {}

    def _load(doc_id: str):
        if doc_id not in cache:
            if not _has_doc(doc_id):
                pytest.skip(f"doc {doc_id} not available locally")
            cache[doc_id] = load_doc(doc_id)
        return cache[doc_id]

    return _load


@pytest.fixture(scope="session")
def m2_loader():
    """Load a doc's Segment A M2 output."""
    import json

    cache: dict[str, dict] = {}

    def _load(doc_id: str) -> dict:
        if doc_id not in cache:
            p = settings.M2_DIR / f"{doc_id}.json"
            if not p.exists():
                pytest.skip(f"no M2 output for {doc_id}")
            cache[doc_id] = json.loads(p.read_text())
        return cache[doc_id]

    return _load


@pytest.fixture(scope="session")
def m2_pre_amendment_loader():
    """
    Load a doc's M2 output as it stood BEFORE the MCAL_PLAN build-#4/#5 prompt
    amendment (plain-language + concreteness clause, `summary_of_interest`).

    `segment_a/output/m2_pre_amendment/` is the archived pre-rerun corpus. It is
    needed for two distinct reasons, not just nostalgia:

      * The human grades in `Evaluation - Sheet1.csv` were written against this
        prose. Any measurement that pairs the grades with the CURRENT artifacts
        is pairing labels with text they do not describe, so the archive is the
        only internally-consistent input for label-conditioned statistics
        (tests/test_atomic_verify.py::TestT01Invisibility).
      * Several regressions pin a specific defective sentence. When the rerun
        rewrote that sentence the history would otherwise be lost, so the
        before-state is asserted here rather than deleted.
    """
    import json

    cache: dict[str, dict] = {}

    def _load(doc_id: str) -> dict:
        if doc_id not in cache:
            p = settings.M2_PRE_AMENDMENT_DIR / f"{doc_id}.json"
            if not p.exists():
                pytest.skip(f"no pre-amendment M2 archive for {doc_id}")
            cache[doc_id] = json.loads(p.read_text())
        return cache[doc_id]

    return _load


def _cited_pages(m2: dict, subfield: str) -> list:
    ev = m2.get("summary", {}).get(subfield, {}).get("evidence", []) or []
    return [p for e in ev for p in (e.get("source_pages") or [])]


@pytest.fixture(scope="session")
def graded_pages(m2_loader):
    """All pages cited by a doc's summary.* subfields."""

    def _pages(doc_id: str, subfield: str) -> list[str]:
        return _cited_pages(m2_loader(doc_id), subfield)

    return _pages


@pytest.fixture(scope="session")
def graded_pages_pre_amendment(m2_pre_amendment_loader):
    """`graded_pages` against the pre-amendment archive.

    The rerun changed WHICH pages each subfield cites (Lincoln Hwy
    environmental_impact: 10 distinct pages -> 15), and some measurements are
    taken over the cited-page window, so the two page sets are not
    interchangeable.
    """

    def _pages(doc_id: str, subfield: str) -> list[str]:
        return _cited_pages(m2_pre_amendment_loader(doc_id), subfield)

    return _pages


# --- Synthetic M1 / M2 payloads ---------------------------------------------
#
# `segment_b/critic.py` and `segment_b/gate.py` consume whole M1+M2 documents, so
# their tests need a payload with all 15 canonical fields. Module-level builders
# rather than fixtures, matching how the doc-id constants above are already
# imported (`from conftest import LINCOLN_HWY`), and so that both test modules
# describe the same baseline document.

# Page texts of the baseline synthetic document. Page 4 carries the quotable
# impact sentence, page 5 the comment stance.
SYNTHETIC_PAGES = (
    "COVER PAGE Final Environmental Impact Statement Lincoln Highway 1972",
    "front matter table of contents",
    "front matter list of figures",
    "The proposed reconstruction would affect approximately 1,200 acres of "
    "sagebrush habitat over 30 years in Cook County, Illinois, at a cost of "
    "$13.2 million.",
    "Mayor Alice Chen of Chicago Heights supported the alignment. Residents of "
    "three census tracts objected during the public hearing.",
    "Chapter 3 Alternatives considered included a no-build option and a "
    "north relocation of the corridor.",
)

# A quote that really is on page 4 of SYNTHETIC_PAGES.
VERIFIABLE_QUOTE = "approximately 1,200 acres of sagebrush habitat over 30 years"
# Plausible NEPA boilerplate that appears nowhere in it -- the Lincoln Hwy
# fabrication pattern from MCAL_PLAN 1(4).
FABRICATED_QUOTE = "or important wildlife habitats are affected by the undertaking"


def build_m1(**overrides) -> dict:
    """A well-formed M1 payload (the shape `segment_a/m1.py` writes)."""
    m1 = {
        "title": {
            "value": "Lincoln Hwy upgrade from Cicero Ave. to Chicago Road",
            "confidence": "high",
            "sources": ["NUL"],
        },
        "year": {
            "value": 1972,
            "confidence": "low",
            "sources": ["NUL", "regex (first 3 pages)"],
            "note": "NUL=1972 disagrees with regex=1976 - flag",
        },
        "eis_type": {
            "value": "Final",
            "confidence": "high",
            "sources": ["Sonnet (first 2 pages)"],
        },
        "lead_agency": {
            "value": ["Federal Highway Administration"],
            "confidence": "high",
            "sources": ["NUL"],
        },
    }
    m1.update(overrides)
    return m1


def _ev(quote: str = VERIFIABLE_QUOTE, page: str = "4") -> dict:
    return {"quote": quote, "source_pages": [page], "quote_verified": True}


def build_m2(**overrides) -> dict:
    """A well-formed M2 payload covering all 11 non-M1 canonical fields."""
    from mcal import settings as s

    summary = {}
    for key in ("overview",) + tuple(
        f.split(".", 1)[1] for f in s.SUMMARY_SUBFIELDS
    ):
        summary[key] = {
            "text": f"Plain-language {key.replace('_', ' ')} of the project.",
            "evidence": [_ev()],
        }
    m2 = {
        "summary": summary,
        "summary_of_interest": [],
        "alternatives": {
            "value": [
                {
                    "name": "No Action",
                    "description": "No build.",
                    "evidence": [_ev("a no-build option", "6")],
                }
            ],
            "confidence": "high",
        },
        "themes": {
            "value": {"themes": ["Transportation Infrastructure"], "subthemes": []},
            "evidence": [_ev()],
        },
        "location": {
            "value": {
                "places": [
                    {
                        "name": "Cook County, IL",
                        "evidence": [_ev("Cook County, Illinois", "4")],
                    }
                ],
                "is_multi_site": False,
                "scope": "corridor",
            }
        },
        "key_people": {
            "value": {
                "agency_preparers": [],
                "cooperating_agencies": [],
                "consulted_entities": [],
                "public_commenters": [
                    {
                        "name": "Alice Chen",
                        "kind": "official",
                        "stance": "support",
                        "capacity": {
                            "capacity": "non_private",
                            "basis": "institutional_identification_in_cited_passage",
                        },
                        "evidence": [
                            _ev("Mayor Alice Chen of Chicago Heights supported", "5")
                        ],
                    }
                ],
            }
        },
    }
    m2.update(overrides)
    return m2


@pytest.fixture
def doc_factory():
    """
    Build a synthetic `pages.Doc` from page texts.

    Lets deterministic post-processors (acronym pre-pass, year adjudicator) be
    unit-tested on hand-written page content, without a corpus and without
    reimplementing Doc's offset bookkeeping in each test.
    """
    from pages import PAGE_SEP, Doc, Page

    def _make(*page_texts: str, doc_id: str = "synthetic"):
        pages = [Page(page_num=i + 1, text=t) for i, t in enumerate(page_texts)]
        starts: list[int] = []
        cursor = 0
        for i, p in enumerate(pages):
            starts.append(cursor)
            cursor += len(p.text) + (len(PAGE_SEP) if i < len(pages) - 1 else 0)
        return Doc(
            doc_id=doc_id,
            pages=pages,
            full_text=PAGE_SEP.join(p.text for p in pages),
            _page_starts=starts,
        )

    return _make


# --- Synthetic M-Cal artifacts ----------------------------------------------
#
# `mcal/artifacts/` does not exist until `mcal/build.py` runs and a human
# ratifies the draft (MCAL_PLAN 3.7). Tests for `segment_b/critic.py` and
# `segment_b/gate.py` therefore CONSTRUCT a promoted stage in a tmp dir and
# repoint `settings.ARTIFACTS_DIR` at it, rather than depending on a build that
# has not happened. `settings.stage_dir()` reads the module global at call time,
# so monkeypatching the attribute is sufficient for every artifact path helper.


class SyntheticArtifacts:
    """A promoted M-Cal stage on disk, built without running mcal.build."""

    def __init__(self, root, stage: str = "v1"):
        self.root = root
        self.stage = stage

    # --- thresholds ---
    def write_thresholds(
        self,
        *,
        tau: float = 0.4,
        per_bucket: dict | None = None,
        gate_all: tuple[str, ...] = (),
        degenerate_severe: tuple[str, ...] = (),
    ):
        """
        Write `thresholds.v(N).json` with hand-chosen taus.

        Hand-written rather than fitted via `confidence.calibrate_all`, because a
        real seed-v1 fit makes every bucket `degenerate_severe` (MCAL_PLAN 0) and
        every field gated, which would make the `composite_below_tau` and
        acceptance paths untestable.
        """
        import json

        from mcal import confidence, settings as s

        taus = dict(per_bucket or {})
        buckets = {}
        for bucket in s.BUCKET_ORDER:
            severe = bucket in degenerate_severe
            gated = severe or bucket in gate_all
            t = float("inf") if gated else float(taus.get(bucket, tau))
            buckets[bucket] = {
                "bucket": bucket,
                "alpha": s.ALPHA,
                "alpha_effective": (
                    s.ALPHA_EFFECTIVE_DEGENERATE if gated else s.ALPHA
                ),
                "N_wrong_docs": 0 if gated else 6,
                "n_items": 0 if gated else 12,
                "n_wrong_items": 0 if gated else 6,
                "tau_raw": None if gated else t,
                "curation_slack": 0.0,
                "tau_deployed": t,
                "saturated": False,
                "guarantee_conditioning": (
                    confidence.GATE_ALL_GUARANTEE
                    if gated
                    else confidence.GUARANTEE_TEMPLATE.format(
                        bucket=bucket, alpha=s.ALPHA
                    )
                ),
                "degenerate": gated,
                "degenerate_severe": severe,
                "gate_all_to_human": gated,
                "loo_deltas": [],
                "wrong_docs": [],
                "r_docs": {},
                "notes": [],
            }
        payload = {
            "version": self.stage,
            "alpha": s.ALPHA,
            "alpha_effective_degenerate": s.ALPHA_EFFECTIVE_DEGENERATE,
            "accept_rule": "accept iff composite > tau_deployed",
            "buckets": buckets,
        }
        path = s.artifact_path("thresholds.json", self.stage, draft=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    # --- confidence config ---
    def write_config(self, **overrides):
        import json

        from mcal import confidence, settings as s

        payload = confidence.build_confidence_config()
        payload["version"] = self.stage
        payload.update(overrides)
        path = s.artifact_path("confidence_config.json", self.stage, draft=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    # --- taxonomy ---
    def write_taxonomy(self):
        from mcal import taxonomy

        tax = taxonomy.seed_taxonomy(self.stage)
        return taxonomy.save(tax, draft=False, ratified=True)

    # --- critic prompts ---
    def write_prompts(self, fields=None, body: str | None = None):
        """
        Write a promoted per-field prompt for each field.

        Deliberately NOT built via `critic_prompt.build_all`: that needs a
        `GradeSet`, and what `segment_b/critic.py` consumes is only "the text of
        `critic_prompts/{field}.v(N).md`". A stub keeps these tests independent of
        prompt-assembly changes.
        """
        from mcal import critic_prompt, settings as s

        fields = fields or s.ALL_FIELDS
        out = {}
        for field in fields:
            path = critic_prompt.prompt_path(field, self.stage, draft=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                body
                or (
                    f"# Critic prompt — `{field}`\n\n"
                    f"## ROLE\n\nYou are a strict quote-anchored verifier.\n\n"
                    f"## OUTPUT\n\nReturn ONLY JSON with `evidence_quote` first.\n"
                ),
                encoding="utf-8",
            )
            out[field] = path
        return out

    def write_all(self, **threshold_kw):
        self.write_taxonomy()
        self.write_config()
        self.write_thresholds(**threshold_kw)
        self.write_prompts()
        return self


@pytest.fixture
def mcal_artifacts(tmp_path, monkeypatch):
    """
    A promoted synthetic M-Cal stage v1, with `settings` repointed at it.

    Also repoints `NULL_TAG_MONITOR_PATH` (computed at import time, so it needs
    its own patch) and clears `segment_b.critic`'s per-process artifact cache
    before and after, since that cache is keyed by stage and would otherwise leak
    one test's artifacts into the next.
    """
    from mcal import settings as s
    from segment_b import critic as critic_mod

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(s, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(s, "NULL_TAG_MONITOR_PATH", artifacts / "null_tag_monitor.json")
    critic_mod.clear_artifact_cache()
    try:
        yield SyntheticArtifacts(artifacts).write_all()
    finally:
        critic_mod.clear_artifact_cache()

