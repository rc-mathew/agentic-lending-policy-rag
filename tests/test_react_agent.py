"""Offline tests for the hand-written ReAct loop: a scripted fake model and a fake retriever."""

import json

from policy_rag import react_agent
from policy_rag.llm import LLMResult
from policy_rag.retrieve import Hit


def _hit(cid, doc="rbi_cb_credit_facilities", page=55, text="LTV up to 80% for 30-75 lakh"):
    return Hit(score=0.9, chunk_id=cid, doc_id=doc, title="Credit Facilities", issuer="RBI",
               status="current", page_start=page, page_end=page, context="Housing", text=text)


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, k=4):
        self.queries.append(query)
        return [_hit("c1"), _hit("c2", page=56)]  # same chunks every time: tests de-duplication


def scripted(*replies):
    """Fake generate(): returns the scripted replies in order and records each prompt."""
    replies = list(replies)
    def gen(prompt, **kw):
        gen.prompts.append(prompt)
        return LLMResult(text=json.dumps(replies.pop(0)), model="fake", cached=False,
                         input_tokens=1000, output_tokens=50)
    gen.prompts = []
    return gen


FINAL = {"answerable": True, "claims": [{"text": "Max loan is Rs 32 lakh.", "sources": [1]}]}


def test_search_then_tool_then_finish(monkeypatch):
    gen = scripted(
        {"thought": "need LTV rule", "action": "search", "query": "housing loan LTV ratio"},
        {"thought": "80% applies", "action": "max_loan_for_ltv",
         "args_json": '{"collateral_value": 4000000, "ltv_pct": 80}'},
        {"thought": "done", "action": "finish"},
        FINAL,
    )
    monkeypatch.setattr(react_agent, "generate", gen)
    r = react_agent.run("Flat costs 40 lakh, max loan?", FakeRetriever())
    assert [s.action for s in r.steps] == ["search", "max_loan_for_ltv", "finish"]
    assert r.steps[1].observation == '{"result": 3200000}'
    assert r.stopped_by == "finish" and r.calls == 4
    assert "max_loan_for_ltv" in gen.prompts[-1] and "= 3200000" in gen.prompts[-1]
    assert r.claims == [{"text": "Max loan is Rs 32 lakh.", "sources": [1]}]
    assert len(r.hits) == 2


def test_repeat_search_adds_no_duplicate_sources(monkeypatch):
    gen = scripted(
        {"thought": "a", "action": "search", "query": "q1"},
        {"thought": "b", "action": "search", "query": "q2"},
        {"thought": "c", "action": "finish"},
        FINAL,
    )
    monkeypatch.setattr(react_agent, "generate", gen)
    r = react_agent.run("q", FakeRetriever())
    assert len(r.hits) == 2
    assert "No new passages" in r.steps[1].observation


def test_step_cap_forces_final_answer(monkeypatch):
    search = {"thought": "more", "action": "search", "query": "x"}
    gen = scripted(*([search] * 3), FINAL)
    monkeypatch.setattr(react_agent, "generate", gen)
    r = react_agent.run("q", FakeRetriever(), max_steps=3)
    assert r.stopped_by == "max_steps" and len(r.steps) == 3 and r.calls == 4


def test_token_cap_stops_early(monkeypatch):
    search = {"thought": "more", "action": "search", "query": "x"}
    gen = scripted(search, search, FINAL)
    monkeypatch.setattr(react_agent, "generate", gen)
    r = react_agent.run("q", FakeRetriever(), max_tokens=1500)  # each call costs 1,050 tokens
    assert r.stopped_by == "max_tokens" and len(r.steps) == 2


def test_bad_tool_args_become_an_observation(monkeypatch):
    gen = scripted(
        {"thought": "calc", "action": "emi", "args_json": "{not json"},
        {"thought": "done", "action": "finish"},
        {"answerable": False, "claims": []},
    )
    monkeypatch.setattr(react_agent, "generate", gen)
    r = react_agent.run("emi?", FakeRetriever())
    assert "not valid JSON" in r.steps[0].observation
    assert not r.answerable and r.render().startswith("I can't answer")


def test_empty_query_falls_back_to_question(monkeypatch):
    gen = scripted({"thought": "s", "action": "search", "query": ""},
                   {"thought": "f", "action": "finish"}, FINAL)
    monkeypatch.setattr(react_agent, "generate", gen)
    ret = FakeRetriever()
    react_agent.run("What is the LTV?", ret)
    assert ret.queries == ["What is the LTV?"]
