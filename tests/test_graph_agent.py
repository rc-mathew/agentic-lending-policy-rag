"""Offline tests for the LangGraph agent: a fake model that answers by schema, a fake retriever."""

import json

import pytest

pytest.importorskip("langgraph")

from policy_rag import graph_agent
from policy_rag.llm import LLMResult
from policy_rag.retrieve import Hit


def _hit(cid, doc="rbi_cb_credit_facilities", page=55):
    return Hit(score=0.9, chunk_id=cid, doc_id=doc, title="Credit Facilities", issuer="RBI",
               status="current", page_start=page, page_end=page, context="Housing",
               text=f"passage {cid}")


class FakeRetriever:
    """Each query returns two fresh chunks, so retries bring new evidence."""
    def __init__(self):
        self.queries = []

    def search(self, query, k=4):
        self.queries.append(query)
        n = len(self.queries)
        return [_hit(f"q{n}a"), _hit(f"q{n}b", page=56)]


def by_schema(script):
    """Fake generate(): replies come from per-schema queues, e.g. {"Grade": [..., ...]}."""
    queues = {k: list(v) for k, v in script.items()}
    def gen(prompt, response_schema=None, **kw):
        name = response_schema.__name__
        gen.calls.append(name)
        gen.prompts[name] = prompt
        return LLMResult(text=json.dumps(queues[name].pop(0)), model="fake", cached=False,
                         input_tokens=500, output_tokens=50)
    gen.calls, gen.prompts = [], {}
    return gen


ANSWER = {"answerable": True, "claims": [{"text": "Max loan Rs 32 lakh.", "sources": [1]}]}


def test_happy_path_with_calculation(monkeypatch):
    gen = by_schema({
        "Plan": [{"route": "search", "queries": ["individual housing loan LTV ratio"],
                  "needs_calculation": True}],
        "Grade": [{"relevant": [1], "sufficient": True, "better_query": ""}],
        "CalcPlan": [{"calls": [{"tool": "max_loan_for_ltv",
                                 "args_json": '{"collateral_value": 4000000, "ltv_pct": 80}'}]}],
        "Answer": [ANSWER],
    })
    monkeypatch.setattr(graph_agent, "generate", gen)
    r = graph_agent.run("Flat costs 40 lakh, max loan?", FakeRetriever())
    assert gen.calls == ["Plan", "Grade", "CalcPlan", "Answer"]
    assert [h.chunk_id for h in r.hits] == ["q1a"]  # grader dropped the irrelevant passage
    assert "max_loan_for_ltv" in gen.prompts["Answer"] and "= 3200000" in gen.prompts["Answer"]
    assert r.stopped_by == "sufficient" and r.calls == 4
    assert [s.action for s in r.steps] == ["plan", "retrieve", "grade", "max_loan_for_ltv",
                                           "generate"]


def test_corrective_retry_uses_graders_query(monkeypatch):
    gen = by_schema({
        "Plan": [{"route": "search", "queries": ["SBI loan acknowledgement"],
                  "needs_calculation": False}],
        "Grade": [{"relevant": [], "sufficient": False,
                   "better_query": "State Bank of India acknowledgement of loan application"},
                  {"relevant": [1], "sufficient": True, "better_query": ""}],
        "Answer": [ANSWER],
    })
    monkeypatch.setattr(graph_agent, "generate", gen)
    ret = FakeRetriever()
    r = graph_agent.run("Will SBI acknowledge my application?", ret)
    assert ret.queries == ["SBI loan acknowledgement",
                           "State Bank of India acknowledgement of loan application"]
    assert [h.chunk_id for h in r.hits] == ["q2a"]  # passage [1] of the second round
    assert r.stopped_by == "sufficient"


def test_retrieval_cap(monkeypatch):
    weak = {"relevant": [], "sufficient": False, "better_query": "another query"}
    gen = by_schema({
        "Plan": [{"route": "search", "queries": ["q"], "needs_calculation": False}],
        "Grade": [weak, {**weak, "better_query": "yet another"}],
        "Answer": [{"answerable": False, "claims": []}],
    })
    monkeypatch.setattr(graph_agent, "generate", gen)
    ret = FakeRetriever()
    r = graph_agent.run("q", ret, max_retrievals=2)
    assert len(ret.queries) == 2 and r.stopped_by == "max_retrievals" and not r.answerable


def test_out_of_scope_skips_retrieval(monkeypatch):
    gen = by_schema({
        "Plan": [{"route": "out_of_scope", "queries": [], "needs_calculation": False}],
        "Answer": [{"answerable": False, "claims": []}],
    })
    monkeypatch.setattr(graph_agent, "generate", gen)
    ret = FakeRetriever()
    r = graph_agent.run("What do US CFPB rules say?", ret)
    assert ret.queries == [] and gen.calls == ["Plan", "Answer"]
    assert r.stopped_by == "out_of_scope" and not r.answerable


def test_multi_part_question_runs_every_planned_query(monkeypatch):
    gen = by_schema({
        "Plan": [{"route": "search", "queries": ["RBI recovery call hours", "KGB contact hours"],
                  "needs_calculation": False}],
        "Grade": [{"relevant": [1, 3], "sufficient": True, "better_query": ""}],
        "Answer": [{"answerable": True, "claims": [{"text": "x", "sources": [1, 2]}]}],
    })
    monkeypatch.setattr(graph_agent, "generate", gen)
    ret = FakeRetriever()
    r = graph_agent.run("Is KGB's 07:00 call time allowed?", ret)
    assert ret.queries == ["RBI recovery call hours", "KGB contact hours"]
    assert [h.chunk_id for h in r.hits] == ["q1a", "q2a"]
    assert r.claims[0]["sources"] == [1, 2]


def test_grader_sees_text_late_in_a_long_chunk(monkeypatch):
    """Regression: a 600-char cut hid RBI's 8:00 a.m. rule (char 1,109) from the grader."""
    class LongChunk(FakeRetriever):
        def search(self, query, k=4):
            h = _hit("long")
            h.text = "x " * 550 + "calling the borrower before 8:00 a.m. is not allowed"
            return [h]
    gen = by_schema({
        "Plan": [{"route": "search", "queries": ["recovery call hours"], "needs_calculation": False}],
        "Grade": [{"relevant": [1], "sufficient": True, "better_query": ""}],
        "Answer": [ANSWER],
    })
    monkeypatch.setattr(graph_agent, "generate", gen)
    graph_agent.run("When can recovery agents call?", LongChunk())
    assert "before 8:00 a.m." in gen.prompts["Grade"]
