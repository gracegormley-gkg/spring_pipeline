"""
Anthropic API wrapper with dry-run mode, retry/backoff, token counting, and budget tracking.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# Approximate cost per million tokens (input/output) by model.
# Update as Anthropic adjusts pricing.
_COST_PER_MTK: dict[str, dict[str, float]] = {
    "claude-opus-4-6":           {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":         {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input":  0.80,  "output":  4.00},
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


class LLMClient:
    """
    Wraps the Anthropic Messages API.
    - dry_run=True: logs prompts but never calls the API; returns a stub response.
    - usage accumulates across all calls; check .usage after a run.
    """

    def __init__(
        self,
        api_key: str | None = None,
        dry_run: bool = False,
        budget_usd: float | None = None,
    ) -> None:
        self.dry_run = dry_run
        self.usage = UsageAccumulator()
        if budget_usd is not None:
            self.usage.set_budget(budget_usd)

        if not dry_run:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set and api_key not provided. "
                    "Export the key or pass it explicitly."
                )
            self._client = anthropic.Anthropic(api_key=key)
        else:
            self._client = None  # type: ignore[assignment]

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
        """
        Make a single call. Returns the text content of the first response block.

        Args:
            model: model ID string
            system: system prompt
            messages: list of {"role": "user"|"assistant", "content": "..."}
            max_tokens: max output tokens
            temperature: sampling temperature
            label: human-readable label for logging (e.g. "summary/doc123")
        """
        if self.dry_run:
            logger.info("[DRY RUN] %s | model=%s | sys=%d chars | msg=%d chars",
                        label, model,
                        len(system),
                        sum(len(m["content"]) for m in messages))
            # Estimate tokens for budget tracking even in dry-run
            est_input = _estimate_tokens(system + " ".join(m["content"] for m in messages))
            self.usage.record(model, est_input, 0)
            return json.dumps({"dry_run": True, "label": label})

        logger.debug("LLM call: %s | model=%s", label, model)
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
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # remove first and last fence lines
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
    """Rough token estimate using tiktoken cl100k_base (close enough for budget tracking)."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4  # fallback: ~4 chars per token
