"""
LLM client wrapping the Anthropic Messages API.

Supports two backends:
  - "anthropic" : direct Anthropic API. Env: ANTHROPIC_API_KEY.
  - "bedrock"   : Amazon Bedrock via anthropic.AnthropicBedrock. Auth picks
                  up the boto3 standard credential chain (env vars
                  AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, or AWS_PROFILE,
                  or ~/.aws/credentials, or instance role) — plus the
                  Bedrock-specific bearer-token env AWS_BEARER_TOKEN_BEDROCK
                  for the "Amazon Bedrock API key" feature.
                  Required env: AWS_REGION (or AWS_DEFAULT_REGION).

Backend selection:
  - LLMClient(backend="anthropic" | "bedrock") forces a backend.
  - LLMClient() (default backend="auto") inspects env: prefers Bedrock when
    AWS creds OR a Bedrock bearer token are present and ANTHROPIC_API_KEY is
    NOT set; otherwise prefers direct Anthropic.

Model IDs differ between backends. The client exposes `.models` (a dict
mapped {haiku, sonnet, opus} -> string) populated from config.ANTHROPIC_MODELS
or config.BEDROCK_MODELS based on the resolved backend. Callers should pass
fully-resolved model strings via the `model=` arg as before; this attribute
is convenience for higher-level code that wants to write
`llm.models["haiku"]`.

Pricing in `_COST_PER_MTK` is shared between backends — Bedrock pricing for
these models tracks Anthropic's published rates closely; we use the same
rates and accept ~5% drift on accounting. If your Bedrock invoice diverges,
override here.

Other features carry over from the original v1_multi port:
  - dry-run mode (no API calls; estimates tokens)
  - per-model usage tally (UsageAccumulator)
  - hard budget cap (raises BudgetExceededError) — synthesis_plan §Per-doc cost cap
  - retry/backoff on rate-limit + connection errors
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import anthropic
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from . import config

logger = logging.getLogger(__name__)

Backend = Literal["anthropic", "bedrock"]

# Approximate cost per million tokens (input/output) by model.
# Both direct-API and Bedrock model IDs map to the same rates (Bedrock tracks
# Anthropic pricing within a few %). Overrides go here.
_COST_PER_MTK: dict[str, dict[str, float]] = {
    # Direct API IDs
    "claude-opus-4-7":           {"input": 15.00, "output": 75.00},
    "claude-opus-4-6":           {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":         {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input":  0.80, "output":  4.00},
    # Bedrock cross-region inference profiles (us.* prefix)
    "us.anthropic.claude-opus-4-7":              {"input": 15.00, "output": 75.00},
    "us.anthropic.claude-sonnet-4-6":            {"input":  3.00, "output": 15.00},
    "us.anthropic.claude-haiku-4-5":             {"input":  0.80, "output":  4.00},
    "us.anthropic.claude-haiku-4-5-20251001":    {"input":  0.80, "output":  4.00},
    # EU / APAC inference profiles (rates match)
    "eu.anthropic.claude-opus-4-7":              {"input": 15.00, "output": 75.00},
    "eu.anthropic.claude-sonnet-4-6":            {"input":  3.00, "output": 15.00},
    "eu.anthropic.claude-haiku-4-5":             {"input":  0.80, "output":  4.00},
    "eu.anthropic.claude-haiku-4-5-20251001":    {"input":  0.80, "output":  4.00},
    "apac.anthropic.claude-opus-4-7":            {"input": 15.00, "output": 75.00},
    "apac.anthropic.claude-sonnet-4-6":          {"input":  3.00, "output": 15.00},
    "apac.anthropic.claude-haiku-4-5":           {"input":  0.80, "output":  4.00},
    "apac.anthropic.claude-haiku-4-5-20251001":  {"input":  0.80, "output":  4.00},
    # Raw Bedrock model IDs (in case some accounts can call without a profile)
    "anthropic.claude-opus-4-7":                 {"input": 15.00, "output": 75.00},
    "anthropic.claude-sonnet-4-6":               {"input":  3.00, "output": 15.00},
    "anthropic.claude-haiku-4-5":                {"input":  0.80, "output":  4.00},
    "anthropic.claude-haiku-4-5-20251001":       {"input":  0.80, "output":  4.00},
}


@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0


@dataclass
class UsageAccumulator:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: int = 0
    by_model: dict[str, ModelUsage] = field(default_factory=dict)
    _budget_usd: float | None = None

    def set_budget(self, budget_usd: float) -> None:
        self._budget_usd = budget_usd

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        rates = _COST_PER_MTK.get(model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
        self.total_cost_usd += cost
        self.calls += 1

        per_model = self.by_model.setdefault(model, ModelUsage())
        per_model.input_tokens += input_tokens
        per_model.output_tokens += output_tokens
        per_model.cost_usd += cost
        per_model.calls += 1

        if self._budget_usd is not None and self.total_cost_usd > self._budget_usd:
            raise BudgetExceededError(
                f"Budget ${self._budget_usd:.2f} exceeded — "
                f"current total ${self.total_cost_usd:.4f}"
            )


class BudgetExceededError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

def _resolve_backend(requested: str) -> Backend:
    """Return one of the two concrete backends.

    "auto" prefers Bedrock when AWS auth is present and ANTHROPIC_API_KEY is
    NOT set. Otherwise prefers direct Anthropic.
    """
    if requested in ("anthropic", "bedrock"):
        return requested  # type: ignore[return-value]
    if requested != "auto":
        raise ValueError(
            f"Unknown LLM backend {requested!r}; expected 'auto', 'anthropic', or 'bedrock'"
        )
    # Auto-detect
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_bedrock_creds = bool(
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
    )
    if has_bedrock_creds and not has_anthropic:
        return "bedrock"
    return "anthropic"


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Wraps Anthropic Messages API; supports direct API and Bedrock backends.

    See module docstring for env-var requirements per backend.
    """

    def __init__(
        self,
        api_key: str | None = None,
        dry_run: bool = False,
        budget_usd: float | None = None,
        backend: str = "auto",
        aws_region: str | None = None,
    ) -> None:
        self.dry_run = dry_run
        self.usage = UsageAccumulator()
        if budget_usd is not None:
            self.usage.set_budget(budget_usd)

        # Resolve backend first so .models reflects the right model IDs.
        self.backend: Backend = _resolve_backend(
            backend if backend != "auto" else config.LLM_BACKEND
        )
        self.models: dict[str, str] = (
            dict(config.BEDROCK_MODELS) if self.backend == "bedrock"
            else dict(config.ANTHROPIC_MODELS)
        )

        if dry_run:
            self._client = None
            return

        if self.backend == "anthropic":
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set and api_key not provided. "
                    "Export the key or pass it explicitly. To use Amazon "
                    "Bedrock instead, set backend='bedrock' (or unset "
                    "ANTHROPIC_API_KEY and set AWS_REGION + standard AWS creds)."
                )
            self._client = anthropic.Anthropic(api_key=key)
            logger.info("LLMClient backend=anthropic (direct API)")
        else:  # bedrock
            region = aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            if not region:
                raise ValueError(
                    "AWS_REGION not set. Bedrock requires a region "
                    "(e.g. AWS_REGION=us-east-1). Standard AWS credentials "
                    "(AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, AWS_PROFILE, "
                    "or AWS_BEARER_TOKEN_BEDROCK) must also be configured."
                )
            self._client = anthropic.AnthropicBedrock(aws_region=region)
            logger.info("LLMClient backend=bedrock region=%s", region)

    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def call(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.2,
        label: str = "",
    ) -> str:
        """Make a single call. Returns the text content of the first response block."""
        if self.dry_run:
            logger.info("[DRY RUN] %s | model=%s | sys=%d chars | msg=%d chars",
                        label, model, len(system),
                        sum(len(m["content"]) for m in messages))
            est_input = _estimate_tokens(system + " ".join(m["content"] for m in messages))
            self.usage.record(model, est_input, 0)
            return json.dumps({"dry_run": True, "label": label})

        logger.debug("LLM call: %s | model=%s | backend=%s", label, model, self.backend)
        response = self._client.messages.create(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        self.usage.record(model, input_tokens, output_tokens)

        text = response.content[0].text
        logger.debug("LLM response: %s | in=%d out=%d tokens", label, input_tokens, output_tokens)
        return text

    def call_json(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.2,
        label: str = "",
    ) -> Any:
        """Like call(), but parses the response as JSON. Raises ValueError on parse failure."""
        raw = self.call(model, system, messages, max_tokens, temperature, label)
        if self.dry_run:
            return {"dry_run": True}
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
            text = "\n".join(inner)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM did not return valid JSON for '{label}'.\n"
                f"Raw response (first 500 chars): {raw[:500]}"
            ) from exc


def _estimate_tokens(text: str) -> int:
    """Rough token estimate using tiktoken cl100k_base."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4
