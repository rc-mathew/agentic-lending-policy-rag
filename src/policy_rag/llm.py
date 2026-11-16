"""Thin Gemini wrapper built for the free tier.

Three things matter on a free quota, and every LLM call in this project goes through here:

1. Disk cache: the same (model, prompt, config) is never paid for twice. Re-running an eval
   after a code change only spends quota on the calls that actually changed.
2. Client-side throttle: stays under the per-minute limit instead of hitting 429s.
3. Retry with backoff on 429 / 5xx, and a clear error once the *daily* quota is gone
   (retrying that is pointless until it resets at midnight Pacific).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass

from .config import CACHE_DIR, settings


class DailyQuotaExhausted(RuntimeError):
    """Raised when the free-tier requests-per-day limit is used up."""


@dataclass
class LLMResult:
    text: str
    model: str
    cached: bool
    input_tokens: int = 0
    output_tokens: int = 0


class _RateLimiter:
    """Sliding-window limiter: at most `rpm` calls in any 60-second window."""

    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()
            if len(self.calls) >= self.rpm:
                time.sleep(60 - (now - self.calls[0]) + 0.1)
            self.calls.append(time.monotonic())


_limiter = _RateLimiter(settings.llm_max_rpm)
_client = None


def _get_client():
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add it.")
        from google import genai

        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _cache_key(model: str, prompt: str, system: str | None, temperature: float, schema: str) -> str:
    raw = json.dumps([model, prompt, system, temperature, schema], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    response_schema: type | None = None,
    use_cache: bool = True,
    max_retries: int = 5,
) -> LLMResult:
    """Generate text (or JSON matching `response_schema`, a Pydantic model) from Gemini."""
    model = model or settings.gen_model
    schema_name = response_schema.__name__ if response_schema else ""
    key = _cache_key(model, prompt, system, temperature, schema_name)
    path = CACHE_DIR / f"{key}.json"

    if use_cache and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return LLMResult(**{**data, "cached": True})

    from google.genai import errors, types

    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system,
        response_mime_type="application/json" if response_schema else None,
        response_schema=response_schema,
    )

    delay = 5.0
    for attempt in range(max_retries + 1):
        _limiter.wait()
        try:
            resp = _get_client().models.generate_content(
                model=model, contents=prompt, config=config
            )
            break
        except errors.APIError as e:
            msg = str(e).lower()
            if e.code == 429 and "per day" in msg:
                raise DailyQuotaExhausted(
                    f"Daily free-tier quota for {model} is used up; it resets at midnight "
                    "Pacific (13:30 IST). Cached results are still available."
                ) from e
            if e.code in (429, 500, 503) and attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise

    usage = getattr(resp, "usage_metadata", None)
    result = LLMResult(
        text=resp.text or "",
        model=model,
        cached=False,
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
    )
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in result.__dict__.items() if k != "cached"}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return result
