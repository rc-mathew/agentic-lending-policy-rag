"""Score a QA system on the gold set: cheap deterministic checks plus ONE judge call per answer.

Why not RAGAS for everything: RAGAS spends several LLM calls per answer across its metrics, and
on the free tier that quota is the project's tightest constraint. Here:

- No LLM needed: retrieval hit (a gold page among the retrieved chunks), whether a cited chunk
  covers a gold page, abstention on the 10 unanswerable questions, false refusals.
- One judge call per answered, answerable question returns
  * correctness against the expected answer (1 / 0.5 / 0), and
  * for every claim, whether the text of the sources it cites actually supports it
    -> citation accuracy (the share of claims their citations support).

Any system works if `run(question)` returns an object with `answerable`, `claims`
([{"text", "sources"}], 1-based indexes into `hits`), `hits`, token counts and `seconds`.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from .config import settings
from .llm import generate

JUDGE_SYSTEM = """You grade answers from a lending-policy assistant. Be strict and literal.

1. correctness: compare the ANSWER with the EXPECTED answer.
   1 = all key facts (numbers, days, conditions) match; 0.5 = partly right or missing a key fact;
   0 = wrong, contradicts it, or does not answer.
2. For EACH numbered claim, supported = true only if the text of the sources cited for that claim
   states it (paraphrase is fine; adding facts the sources do not state is not)."""


class ClaimVerdict(BaseModel):
    claim: int = Field(description="Claim number as given.")
    supported: bool


class Judgement(BaseModel):
    correctness: float = Field(description="1, 0.5 or 0")
    claims: list[ClaimVerdict]
    reason: str = Field(description="One short sentence.")


def load_gold(path: Path, include_drafts: bool = False) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    return rows if include_drafts else [r for r in rows if r.get("status") == "verified"]


def _covers_gold(hit, gold_pages: set[tuple[str, int]]) -> bool:
    return any((hit.doc_id, p) in gold_pages for p in range(hit.page_start, hit.page_end + 1))


def judge_prompt(item: dict, result) -> str:
    claims = []
    for i, c in enumerate(result.claims, 1):
        srcs = "\n".join(
            f"    [{n}] {result.hits[n - 1].label}: {result.hits[n - 1].text}" for n in c["sources"]
        ) or "    (no sources cited)"
        claims.append(f"Claim {i}: {c['text']}\n  Cited sources:\n{srcs}")
    return (
        f"QUESTION: {item['question']}\n\nEXPECTED: {item['expected_answer']}\n\n"
        f"ANSWER (as claims):\n\n" + "\n\n".join(claims)
    )


def score_one(item: dict, result, judge: Callable = generate) -> dict:
    gold_pages = {(a["doc_id"], a["page"]) for a in item["answers"]}
    expect_answer = bool(gold_pages)
    abstained = not result.answerable
    row = {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "abstained": abstained,
        "retrieval_hit": any(_covers_gold(h, gold_pages) for h in result.hits) if expect_answer else None,
        "cites_gold": (
            any(_covers_gold(result.hits[n - 1], gold_pages) for c in result.claims for n in c["sources"])
            if expect_answer and not abstained else None
        ),
        "abstention_correct": (abstained if not expect_answer else None),
        "false_refusal": (abstained if expect_answer else None),
        "correctness": None,
        "citation_accuracy": None,
        "judge_reason": "",
        "claims": result.claims,
        "system_tokens": result.input_tokens + result.output_tokens,
        "judge_tokens": 0,
        "seconds": round(result.seconds, 2),
        "cached": result.cached,
        "calls": getattr(result, "calls", 1),
        "guardrail": getattr(result, "guardrail", ""),
        "stopped_by": getattr(result, "stopped_by", ""),
        "trace": [
            {"action": s.action, "detail": s.detail} for s in getattr(result, "steps", [])
        ],
    }
    if not expect_answer:
        row["correctness"] = 1.0 if abstained else 0.0
        return row
    if abstained:
        row["correctness"] = 0.0
        return row

    res = judge(
        judge_prompt(item, result), system=JUDGE_SYSTEM, model=settings.judge_model,
        response_schema=Judgement,
    )
    j = Judgement.model_validate_json(res.text)
    verdicts = {v.claim: v.supported for v in j.claims}
    supported = [verdicts.get(i, False) for i in range(1, len(result.claims) + 1)]
    row["correctness"] = min(1.0, max(0.0, j.correctness))
    row["citation_accuracy"] = round(sum(supported) / len(supported), 3) if supported else 0.0
    row["judge_reason"] = j.reason
    row["judge_tokens"] = res.input_tokens + res.output_tokens
    return row


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def summarise(rows: list[dict]) -> dict:
    def block(rs: list[dict]) -> dict:
        return {
            "n": len(rs),
            "correctness": _mean(r["correctness"] for r in rs),
            "citation_accuracy": _mean(r["citation_accuracy"] for r in rs),
            "retrieval_hit": _mean(r["retrieval_hit"] for r in rs),
            "cites_gold": _mean(r["cites_gold"] for r in rs),
            "abstention_correct": _mean(r["abstention_correct"] for r in rs),
            "false_refusal": _mean(r["false_refusal"] for r in rs),
        }

    out = {"overall": block(rows)}
    for t in sorted({r["type"] for r in rows}):
        out[t] = block([r for r in rows if r["type"] == t])
    secs = [r["seconds"] for r in rows if not r["cached"]] or [r["seconds"] for r in rows]
    out["cost"] = {
        "avg_system_tokens": round(statistics.mean(r["system_tokens"] for r in rows)) if rows else 0,
        "avg_judge_tokens": round(statistics.mean(r["judge_tokens"] for r in rows)) if rows else 0,
        "latency_p50_s": round(statistics.median(secs), 2) if secs else None,
        "avg_system_calls": round(statistics.mean(r.get("calls", 1) for r in rows), 2) if rows else 0,
    }
    return out
