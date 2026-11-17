"""A ReAct agent written by hand in plain Python: no framework, so every moving part is visible.

Loop (one LLM call per step):
    THOUGHT  what is still missing
    ACTION   search(query) | emi | tenure_for_emi | foir | max_loan_for_ltv | finish
    OBSERVE  the tool's result is appended to the history
Then one final call writes the answer with the same cited-claims schema as plain RAG, over every
source the agent collected, so both systems are scored the same way.

What it can do that plain RAG cannot: rewrite a vague question into a better search, search
twice for a two-part (multi-hop) question, do arithmetic with a tool instead of in its head, and
stop searching when the evidence is enough.

Stopping conditions: at most MAX_STEPS actions and MAX_TOKENS (30k) tokens across all calls. Either
limit forces the final answer from whatever has been collected so far.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from .llm import generate
from .rag import SYSTEM as ANSWER_SYSTEM
from .rag import Answer, RagResult, build_prompt
from .retrieve import Hit
from .tools import TOOL_SPECS, call_tool
from .tracing import answer_outputs, traceable

MAX_STEPS = 5
MAX_TOKENS = 30_000
SEARCH_K = 4
# Agents read whole chunks (<=500 tokens, ~1,700 chars). An earlier 700-char cut hid rules that
# sit late in a chunk (RBI's 8:00 a.m. recovery-call limit is at char 1,109).
SNIPPET_CHARS = 2000

Action = Literal["search", "emi", "tenure_for_emi", "foir", "max_loan_for_ltv", "finish"]


class Step(BaseModel):
    thought: str = Field(description="One or two sentences: what is known, what is missing.")
    action: Action
    query: str = Field(default="", description="For search: a precise search query.")
    args_json: str = Field(
        default="", description='For a calculator: its arguments as JSON, e.g. {"principal": 2000000}'
    )


def _tools_text() -> str:
    lines = ["- search(query): find passages in RBI Directions and bank lending policies."]
    for t in TOOL_SPECS:
        params = ", ".join(t["parameters"]["properties"])
        lines.append(f"- {t['name']}({params}): {t['description']}")
    lines.append("- finish: you have enough evidence (or none exists) to answer.")
    return "\n".join(lines)


STEP_SYSTEM = f"""You research questions about Indian bank lending rules. Each turn, choose ONE action.

Tools:
{_tools_text()}

How to work:
- Search before answering anything about rules. Write search queries in the documents' own
  wording (e.g. "pre-payment charges floating rate loans individuals"), not the user's.
- If the question names a bank, put the bank's name in the query.
- A question with two parts or two documents needs two searches.
- Use a calculator for ANY arithmetic; never compute in your head. Take rule values (LTV caps,
  rates, days) from the passages you found.
- Do not repeat a search you already ran. Finish as soon as the evidence answers the question,
  or after two searches find nothing relevant (then the answer is that it is not covered)."""


@dataclass
class TraceStep:
    thought: str
    action: str
    detail: str
    observation: str


@dataclass
class AgentResult(RagResult):
    steps: list[TraceStep] = field(default_factory=list)
    calls: int = 0
    stopped_by: str = ""  # "finish" | "max_steps" | "max_tokens"


def _observe_hits(new: list[Hit], offset: int) -> str:
    if not new:
        return "No new passages (all already collected)."
    return "\n\n".join(
        f"[{offset + i}] {h.label}\nSection: {h.context}\n{h.text[:SNIPPET_CHARS]}"
        for i, h in enumerate(new, 1)
    )


def _history(question: str, steps: list[TraceStep]) -> str:
    parts = [f"QUESTION: {question}"]
    for i, s in enumerate(steps, 1):
        parts.append(f"STEP {i}\nThought: {s.thought}\nAction: {s.action} {s.detail}\n"
                     f"Observation:\n{s.observation}")
    parts.append("Choose the next action.")
    return "\n\n".join(parts)


@traceable(run_type="chain", name="react_agent", process_outputs=answer_outputs)
def run(question: str, retriever, model: str | None = None,
        max_steps: int = MAX_STEPS, max_tokens: int = MAX_TOKENS) -> AgentResult:
    t0 = time.perf_counter()
    hits: list[Hit] = []
    seen: set[str] = set()
    steps: list[TraceStep] = []
    calcs: list[str] = []
    tokens_in = tokens_out = calls = 0
    cached = True
    stopped_by = "max_steps"

    for _ in range(max_steps):
        if tokens_in + tokens_out >= max_tokens:
            stopped_by = "max_tokens"
            break
        res = generate(_history(question, steps), system=STEP_SYSTEM, model=model,
                       response_schema=Step)
        calls += 1
        tokens_in += res.input_tokens
        tokens_out += res.output_tokens
        cached &= res.cached
        step = Step.model_validate_json(res.text)

        if step.action == "finish":
            steps.append(TraceStep(step.thought, "finish", "", ""))
            stopped_by = "finish"
            break
        if step.action == "search":
            query = step.query.strip() or question
            new = [h for h in retriever.search(query, k=SEARCH_K) if h.chunk_id not in seen]
            seen.update(h.chunk_id for h in new)
            obs = _observe_hits(new, len(hits))
            hits.extend(new)
            steps.append(TraceStep(step.thought, "search", f'"{query}"', obs))
        else:
            try:
                args = json.loads(step.args_json or "{}")
            except json.JSONDecodeError as e:
                args, out = {}, {"error": f"args_json is not valid JSON: {e}"}
            else:
                out = call_tool(step.action, args)
            obs = json.dumps(out)
            if "result" in out:
                calcs.append(f"{step.action}({json.dumps(args)}) = {out['result']}")
            steps.append(TraceStep(step.thought, step.action, json.dumps(args), obs))

    # Final answer: same schema and checks as plain RAG, over everything collected.
    prompt = build_prompt(question, hits)
    if calcs:
        prompt += "\n\nCalculator results (exact; use these numbers, cite the rule's source):\n"
        prompt += "\n".join(f"- {c}" for c in calcs)
    final = generate(prompt, system=ANSWER_SYSTEM, model=model, response_schema=Answer)
    calls += 1
    tokens_in += final.input_tokens
    tokens_out += final.output_tokens
    cached &= final.cached
    parsed = Answer.model_validate_json(final.text)

    claims, invalid, uncited = [], 0, 0
    for c in parsed.claims:
        valid = sorted({n for n in c.sources if 1 <= n <= len(hits)})
        invalid += len(set(c.sources)) - len(valid)
        uncited += not valid
        claims.append({"text": c.text, "sources": valid})

    return AgentResult(
        question=question,
        answerable=parsed.answerable and bool(claims),
        claims=claims if parsed.answerable else [],
        hits=hits,
        invalid_citations=invalid,
        uncited_claims=uncited,
        cached=cached,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        seconds=time.perf_counter() - t0,
        model=final.model,
        raw=final.text,
        steps=steps,
        calls=calls,
        stopped_by=stopped_by,
    )
