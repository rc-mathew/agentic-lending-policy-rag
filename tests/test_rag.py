"""Offline tests for plain RAG: fake retriever, fake Gemini. No network."""

import json

from policy_rag import rag
from policy_rag.llm import LLMResult
from policy_rag.retrieve import Hit


def _hit(i, doc="rbi_cb_credit_facilities", page=22):
    return Hit(score=0.9 - i / 10, chunk_id=f"c{i}", doc_id=doc, title="Credit Facilities Directions",
               issuer="RBI", status="current", page_start=page, page_end=page,
               context="Chapter III > A. General", text=f"chunk text {i}")


class FakeRetriever:
    def search(self, query, k=5):
        return [_hit(i) for i in range(k)]


def _fake_generate(payload):
    def gen(prompt, **kw):
        gen.prompt, gen.kw = prompt, kw
        return LLMResult(text=json.dumps(payload), model="fake", cached=False,
                         input_tokens=100, output_tokens=20)
    return gen


def test_citations_validated_and_rendered(monkeypatch):
    gen = _fake_generate({"answerable": True, "claims": [
        {"text": "Cooling-off is at least one day.", "sources": [1, 9]},  # 9 is out of range
        {"text": "Unsupported claim.", "sources": []},
    ]})
    monkeypatch.setattr(rag, "generate", gen)
    r = rag.answer("cooling-off?", FakeRetriever(), k=3)
    assert r.answerable
    assert r.claims[0]["sources"] == [1]
    assert r.invalid_citations == 1 and r.uncited_claims == 1
    out = r.render()
    assert "[1]" in out and "Sources:\n[1] Credit Facilities Directions (RBI), p. 22" in out
    assert gen.kw["response_schema"] is rag.Answer
    assert "[3] Credit Facilities" in gen.prompt and "Question: cooling-off?" in gen.prompt


def test_abstention(monkeypatch):
    monkeypatch.setattr(rag, "generate", _fake_generate({"answerable": False, "claims": []}))
    r = rag.answer("What is the repo rate today?", FakeRetriever())
    assert not r.answerable and r.claims == []
    assert r.render().startswith("I can't answer that")


def test_repealed_sources_are_marked():
    h = _hit(0)
    h.status = "repealed"
    assert "[REPEALED]" in rag.format_sources([h])


def test_hit_label_page_range():
    h = _hit(0)
    h.page_end = 24
    assert h.label.endswith("pp. 22-24")


def test_answer_prompt_requires_a_direct_verdict_first():
    # Regression: the KGB 07:00 question got both rules cited but no "is it allowed?" verdict.
    from policy_rag.rag import SYSTEM

    assert "FIRST claim answers the question directly" in SYSTEM
    assert "Yes, No or Partly" in SYSTEM
    # Regression: "Can a bank charge X?" was answered "Yes, a bank cannot charge X".
    assert 'never "Yes, it cannot' in SYSTEM


def test_answer_schema_puts_reasoning_before_claims():
    # Regression: forced to give a verdict with no working, Flash-Lite said 07:00 was allowed
    # under an 08:00 limit. The schema makes it write the comparison first.
    from policy_rag.rag import SYSTEM, Answer

    fields = list(Answer.model_json_schema()["properties"])
    assert fields.index("reasoning") < fields.index("claims")
    assert 'Fill "reasoning" BEFORE the claims' in SYSTEM
    assert Answer.model_validate({"answerable": False, "claims": []}).reasoning == ""
