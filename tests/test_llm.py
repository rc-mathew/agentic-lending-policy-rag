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
