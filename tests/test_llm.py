"""Offline tests for the Gemini wrapper: caching and the rate limiter. No API calls."""

import json
import time

from policy_rag import llm


def test_cache_key_is_stable_and_sensitive():
    a = llm._cache_key("m", "p", None, 0.0, "")
    assert a == llm._cache_key("m", "p", None, 0.0, "")
    assert a != llm._cache_key("m", "p2", None, 0.0, "")
    assert a != llm._cache_key("m", "p", "sys", 0.0, "")
    assert a != llm._cache_key("m", "p", None, 0.5, "")


def test_cached_response_skips_api(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    key = llm._cache_key("test-model", "hello", None, 0.0, "")
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"text": "hi", "model": "test-model", "input_tokens": 1, "output_tokens": 1})
    )

    def boom():
        raise AssertionError("API should not be called on a cache hit")

    monkeypatch.setattr(llm, "_get_client", boom)
    r = llm.generate("hello", model="test-model")
    assert r.cached and r.text == "hi"


def test_rate_limiter_allows_burst_up_to_rpm():
    lim = llm._RateLimiter(rpm=5)
    t0 = time.monotonic()
    for _ in range(5):
        lim.wait()
    assert time.monotonic() - t0 < 0.5


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_groq_judge_path_sends_schema_and_caches(tmp_path, monkeypatch):
    from pydantic import BaseModel

    class Verdict(BaseModel):
        ok: bool

    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    object.__setattr__(llm.settings, "groq_api_key", "test-key")
    sent = {}

    def fake_urlopen(req, timeout):
        sent["body"] = json.loads(req.data)
        sent["auth"] = req.headers["Authorization"]
        return _FakeResponse({
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        })

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("no Gemini")))

    r = llm.generate("grade this", system="be strict", model="groq:openai/gpt-oss-120b",
                     response_schema=Verdict)
    assert Verdict.model_validate_json(r.text).ok
    assert sent["body"]["model"] == "openai/gpt-oss-120b"
    assert sent["body"]["response_format"] == {"type": "json_object"}
    assert '"ok"' in sent["body"]["messages"][0]["content"]  # schema is in the system prompt
    assert sent["auth"] == "Bearer test-key"
    assert (r.input_tokens, r.output_tokens) == (10, 3)

    # Second call is served from the cache: no network.
    monkeypatch.setattr(llm.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cached")))
    assert llm.generate("grade this", system="be strict", model="groq:openai/gpt-oss-120b",
                        response_schema=Verdict).cached
    object.__setattr__(llm.settings, "groq_api_key", "")


def test_groq_without_key_fails_clearly(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    object.__setattr__(llm.settings, "groq_api_key", "")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm.generate("x", model="groq:openai/gpt-oss-120b")
