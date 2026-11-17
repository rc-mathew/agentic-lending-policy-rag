"""Thin LLM wrapper built for free tiers: Gemini by default, Groq for models named "groq:<model>".

Three things matter on a free quota, and every LLM call in this project goes through here:

1. Disk cache: the same (model, prompt, config) is never paid for twice. Re-running an eval
   after a code change only spends quota on the calls that actually changed.
2. Client-side throttle: stays under the per-minute limit instead of hitting 429s.
3. Retry with backoff on 429 / 5xx, and a clear error once the *daily* quota is gone
   (retrying that is pointless until it resets at midnight Pacific).

Groq is here only so the eval judge can be a different model from the one it grades, and so
judge calls do not spend the Gemini daily quota. It goes through Groq's OpenAI-compatible REST
endpoint with the standard library, so it adds no dependency.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass

from .config import CACHE_DIR, settings
from .tracing import llm_outputs, traceable


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
_groq_limiter = _RateLimiter(settings.groq_max_rpm)
_client = None
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


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


def _groq_call(model: str, prompt: str, system: str | None, temperature: float,
               response_schema: type | None, max_retries: int) -> LLMResult:
    """One chat completion on Groq. JSON mode plus the schema in the system prompt; the caller
    validates the JSON with the Pydantic model, exactly as it does for Gemini."""
    if not settings.groq_api_key:
        raise RuntimeError(f"JUDGE_MODEL is {model!r} but GROQ_API_KEY is not set in .env.")
    sys_text = system or ""
    if response_schema is not None:
        sys_text += (
            "\n\nReply with ONLY a JSON object matching this JSON schema:\n"
            + json.dumps(response_schema.model_json_schema())
        )
    body = {
        "model": model.removeprefix("groq:"),
        "temperature": temperature,
        "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": prompt}],
    }
    if response_schema is not None:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        GROQ_URL, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {settings.groq_api_key}",
                 "Content-Type": "application/json", "User-Agent": "policy-rag"},
    )
    delay = 10.0
    for attempt in range(max_retries + 1):
        _groq_limiter.wait()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace").lower()
            if e.code == 429 and ("per day" in detail or "(rpd)" in detail or "(tpd)" in detail):
                raise DailyQuotaExhausted(
                    f"Daily Groq free-tier limit for {model} is used up; it resets at 00:00 UTC "
                    "(05:30 IST). Cached results are still available."
                ) from e
            if e.code in (429, 500, 503) and attempt < max_retries:
                wait = float(e.headers.get("retry-after") or delay)
                time.sleep(min(wait, 120))
                delay *= 2
                continue
            raise RuntimeError(f"Groq error {e.code}: {detail[:300]}") from e
    usage = data.get("usage") or {}
    return LLMResult(
        text=data["choices"][0]["message"]["content"] or "",
        model=model,
        cached=False,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )


@traceable(run_type="llm", name="llm", process_outputs=llm_outputs)
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
    """Generate text (or JSON matching `response_schema`, a Pydantic model) from Gemini or Groq."""
    model = model or settings.gen_model
    # The full JSON schema, not just the class name: editing a field must not hit stale cache.
    schema_name = (
        json.dumps(response_schema.model_json_schema(), sort_keys=True) if response_schema else ""
    )
    key = _cache_key(model, prompt, system, temperature, schema_name)
    path = CACHE_DIR / f"{key}.json"

    if use_cache and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return LLMResult(**{**data, "cached": True})

    if model.startswith("groq:"):
        result = _groq_call(model, prompt, system, temperature, response_schema, max_retries)
        if use_cache:
            _save(path, result)
        return result

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
        _save(path, result)
    return result


def _save(path, result: LLMResult) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.__dict__.items() if k != "cached"}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
