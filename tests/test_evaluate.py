"""Offline tests for the scoring harness: fake results, fake judge."""

import json
from types import SimpleNamespace

from policy_rag import evaluate
from policy_rag.llm import LLMResult
from policy_rag.retrieve import Hit


def _hit(doc, page):
    return Hit(score=0.9, chunk_id="c", doc_id=doc, title="T", issuer="RBI", status="current",
               page_start=page, page_end=page, context="ctx", text=f"text of {doc} p{page}")


def _result(answerable=True, claims=None, hits=None):
    return SimpleNamespace(answerable=answerable, claims=claims or [], hits=hits or [],
                           input_tokens=100, output_tokens=20, seconds=1.5, cached=False)


def _judge(payload):
    def j(prompt, **kw):
        j.prompt = prompt
        return LLMResult(text=json.dumps(payload), model="judge", cached=False,
                         input_tokens=300, output_tokens=30)
    return j


GOLD = {"id": "g01", "type": "direct", "question": "Q?", "expected_answer": "5%",
        "answers": [{"doc_id": "cf", "page": 28}]}
UNANS = {"id": "g51", "type": "unanswerable", "question": "Repo rate?", "expected_answer": "-",
         "answers": []}


def test_answered_question_is_judged_per_claim():
    res = _result(claims=[{"text": "a", "sources": [1]}, {"text": "b", "sources": [2]}],
                  hits=[_hit("cf", 28), _hit("kyc", 5)])
    j = _judge({"correctness": 1, "claims": [{"claim": 1, "supported": True},
                                             {"claim": 2, "supported": False}], "reason": "ok"})
    row = evaluate.score_one(GOLD, res, judge=j)
    assert row["correctness"] == 1 and row["citation_accuracy"] == 0.5
    assert row["retrieval_hit"] and row["cites_gold"]
    assert "text of cf p28" in j.prompt and "EXPECTED: 5%" in j.prompt


def test_unanswerable_needs_no_judge():
    def boom(*a, **k):
        raise AssertionError("judge must not be called")
    good = evaluate.score_one(UNANS, _result(answerable=False), judge=boom)
    bad = evaluate.score_one(UNANS, _result(claims=[{"text": "6.5%", "sources": [1]}],
                                            hits=[_hit("x", 1)]), judge=boom)
    assert good["abstention_correct"] is True and good["correctness"] == 1
    assert bad["abstention_correct"] is False and bad["correctness"] == 0


def test_false_refusal_scores_zero_without_judge():
    def boom(*a, **k):
        raise AssertionError("judge must not be called")
    row = evaluate.score_one(GOLD, _result(answerable=False, hits=[_hit("cf", 28)]), judge=boom)
    assert row["false_refusal"] is True and row["correctness"] == 0 and row["retrieval_hit"]


def test_summary_by_type():
    rows = [
        {"type": "direct", "correctness": 1, "citation_accuracy": 1, "retrieval_hit": True,
         "cites_gold": True, "abstention_correct": None, "false_refusal": False,
         "system_tokens": 100, "judge_tokens": 300, "seconds": 2, "cached": False},
        {"type": "unanswerable", "correctness": 0, "citation_accuracy": None, "retrieval_hit": None,
         "cites_gold": None, "abstention_correct": False, "false_refusal": None,
         "system_tokens": 100, "judge_tokens": 0, "seconds": 4, "cached": False},
    ]
    s = evaluate.summarise(rows)
    assert s["overall"]["correctness"] == 0.5
    assert s["unanswerable"]["abstention_correct"] == 0
    assert s["direct"]["false_refusal"] == 0
    assert s["cost"]["latency_p50_s"] == 3


def test_load_gold_filters_drafts(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({**GOLD, "status": "draft"}) + "\n"
                 + json.dumps({**UNANS, "status": "verified"}) + "\n")
    assert [r["id"] for r in evaluate.load_gold(p)] == ["g51"]
    assert len(evaluate.load_gold(p, include_drafts=True)) == 2
