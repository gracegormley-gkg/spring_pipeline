"""
Anthropic client wrapper.

Three model tiers per the v2 plan:
  - Haiku  → cheap structured ops (title fallback, section labeling)
  - Sonnet → verification, critique, list extraction
  - Opus   → summary synthesis

JSON-only outputs are coaxed via system prompt + post-parse cleanup.
Retries on transient failures.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from config import MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET

log = logging.getLogger(__name__)

# Lazy import so the module loads even without the SDK installed (e.g. for
# running deterministic-only smoke tests).
_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. Run `pip install anthropic`."
            ) from e

        # Prefer Bedrock when the bearer token is present (no separate
        # ANTHROPIC_API_KEY needed). Fall back to direct Anthropic API when
        # an ANTHROPIC_API_KEY is set instead.
        bedrock_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        api_key = os.environ.get("ANTHROPIC_API_KEY")

        if bedrock_token:
            region = (
                os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or "us-east-1"
            )
            _client = anthropic.AnthropicBedrock(aws_region=region)
            log.info(f"LLM: using AnthropicBedrock (region={region})")
        elif api_key:
            _client = anthropic.Anthropic(api_key=api_key)
            log.info("LLM: using direct Anthropic API")
        else:
            raise RuntimeError(
                "No credentials: set AWS_BEARER_TOKEN_BEDROCK (Bedrock) or "
                "ANTHROPIC_API_KEY (direct API)."
            )
    return _client


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
        if s.startswith("json"):
            s = s[4:].strip()
    return s


def _extract_first_json_object(s: str) -> Optional[str]:
    """Find the first balanced {...} JSON object in s, ignoring braces inside strings."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _normalize_usage(usage_obj, model: str) -> dict:
    """Pull (input_tokens, output_tokens, cache_*) off an SDK usage object into a plain dict."""
    if usage_obj is None:
        return {
            "model": model,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
    return {
        "model": model,
        "input_tokens": int(getattr(usage_obj, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage_obj, "output_tokens", 0) or 0),
        "cache_creation_input_tokens": int(getattr(usage_obj, "cache_creation_input_tokens", 0) or 0),
        "cache_read_input_tokens": int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0),
    }


def call_with_usage(
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 1500,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> tuple[str, dict]:
    """Single Anthropic call. Returns (assistant_text, usage_dict)."""
    client = _get_client()
    last_exc: Optional[Exception] = None
    omit_temperature = "opus" in model.lower()
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            if not omit_temperature:
                kwargs["temperature"] = temperature
            resp = client.messages.create(**kwargs)
            text = "".join(
                getattr(b, "text", "") for b in resp.content
                if getattr(b, "type", "") == "text"
            )
            usage = _normalize_usage(getattr(resp, "usage", None), model)
            return text, usage
        except Exception as e:
            last_exc = e
            wait = 2 ** attempt
            log.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_exc}")


def call(
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 1500,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> str:
    """Single Anthropic call. Returns the assistant text. Usage is discarded."""
    text, _ = call_with_usage(
        model, system, user,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
    )
    return text


def call_json_with_usage(
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 1500,
    temperature: float = 0.2,
) -> tuple[dict, dict]:
    """LLM call that must return JSON. Returns (parsed_json, usage_dict)."""
    raw, usage = call_with_usage(
        model, system, user,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned), usage
    except json.JSONDecodeError:
        candidate = _extract_first_json_object(raw) or _extract_first_json_object(cleaned)
        if candidate:
            try:
                return json.loads(candidate), usage
            except json.JSONDecodeError as e:
                log.error(f"JSON parse failed (after extraction). Raw output: {raw[:500]}")
                raise RuntimeError(f"LLM did not return valid JSON: {e}") from e
        log.error(f"JSON parse failed. Raw output: {raw[:500]}")
        raise RuntimeError("LLM did not return valid JSON: no JSON object found")


def call_json(
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 1500,
    temperature: float = 0.2,
) -> dict:
    """LLM call that must return JSON. Strips code fences before parsing."""
    out, _ = call_json_with_usage(
        model, system, user,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return out


def haiku(system: str, user: str, **kw) -> dict:
    return call_json(MODEL_HAIKU, system, user, **kw)


def sonnet(system: str, user: str, **kw) -> dict:
    return call_json(MODEL_SONNET, system, user, **kw)


def opus(system: str, user: str, **kw) -> dict:
    return call_json(MODEL_OPUS, system, user, **kw)
