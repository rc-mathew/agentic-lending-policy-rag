"""The LangGraph agent: corrective agentic RAG as an explicit graph.

    START -> plan -> retrieve -> grade --(enough evidence)--> calculate? -> generate -> END
              |                    \\--(weak, retries left)--> retrieve (with a rewritten query)
              \\--(clearly off-topic)--> generate (abstains)

Nodes:
- plan       (LLM) routes the question and writes 1-3 search queries in the documents' own
             wording: a multi-part question is split, a bank's abbreviation is expanded ("SBI" ->
             "State Bank of India"), and it notes whether arithmetic is needed.
- retrieve   (no LLM) runs every new query, de-duplicates chunks.
- grade      (LLM) marks which retrieved passages are relevant, keeps only those, and says
             whether they suffice; if not, proposes one better query (corrective RAG).
- calculate  (LLM picks tools, Python runs them) EMI / tenure / FOIR / LTV calculators.
- generate   (LLM) the same cited-claims answer as plain RAG, over the graded sources.

Where the hand-written ReAct loop lets the model decide everything step by step, this graph fixes
the control flow and asks the model only small, checkable questions. Caps: MAX_RETRIEVALS rounds
of retrieval and MAX_TOKENS (30k) tokens; either one sends the graph straight to generate.
"""

from __future__ import annotations

import json
import time
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from .llm import generate
from .rag import SYSTEM as ANSWER_SYSTEM
from .rag import Answer, build_prompt, calculator_hits
from .react_agent import AgentResult, TraceStep
from .retrieve import Hit
from .tools import TOOL_SPECS, call_tool
from .tracing import answer_outputs, traceable

MAX_RETRIEVALS = 2
MAX_TOKENS = 30_000
SEARCH_K = 4
# Agents read whole chunks (<=500 tokens, ~1,700 chars). An earlier 600-char cut hid rules that
# sit late in a chunk (RBI's 8:00 a.m. recovery-call limit is at char 1,109) and the grader
# rejected the right evidence.
SNIPPET_CHARS = 2000


# ---- structured outputs the model must return --------------------------------------------------

class Plan(BaseModel):
    route: Literal["search", "out_of_scope"] = Field(
        description="out_of_scope ONLY if clearly not about Indian banking or lending rules."
    )
    queries: list[str] = Field(description="1-3 search queries in regulatory wording.")
    needs_calculation: bool = Field(description="True if the answer needs arithmetic.")


class Grade(BaseModel):
    relevant: list[int] = Field(description="Numbers of the passages that help answer.")
    sufficient: bool = Field(description="True if the relevant passages answer the question.")
    better_query: str = Field(default="", description="If not sufficient: one new search query.")


class ToolCall(BaseModel):
    tool: Literal["emi", "tenure_for_emi", "foir", "max_loan_for_ltv"]
    args_json: str = Field(description='Arguments as JSON, e.g. {"principal": 2000000}')


class CalcPlan(BaseModel):
    calls: list[ToolCall] = Field(description="At most 3 calculator calls, values from the passages.")


PLAN_SYSTEM = """You plan research for questions about Indian bank lending rules (RBI Directions
and banks' own lending policies).
- route = out_of_scope for questions clearly outside Indian banking/lending (e.g. US law,
  income tax), and for questions that ask for a live figure that changes day to day (today's
  repo rate, a bank's current interest rate, which bank is cheapest now): the documents are
  rules and policies, not market data. Questions about the RULES for rates (how often MCLR is
  reviewed, when a reset may happen) are "search".
- Write 1-3 search queries in the documents' formal wording. Split questions with two parts or
  two documents into separate queries. Expand bank abbreviations (SBI -> State Bank of India).
- needs_calculation = true if an amount, EMI, ratio, days or compensation must be computed."""

GRADE_SYSTEM = """You check retrieved passages for a question about Indian lending rules.
- relevant: numbers of passages that contain facts needed for the answer.
- sufficient: true only if those passages together answer every part of the question.
- better_query: if not sufficient, ONE new search query for what is missing (formal wording)."""

CALC_SYSTEM = """Plan calculator calls for the question, using ONLY rule values stated in the
passages (LTV caps, rates, days, amounts) and numbers given in the question. At most 3 calls.
Tools:
""" + "\n".join(
    f"- {t['name']}({', '.join(t['parameters']['properties'])}): {t['description']}"
    for t in TOOL_SPECS
)


class State(TypedDict, total=False):
    question: str
    route: str
    needs_calculation: bool
    pending: list[str]     # queries not yet run
    ran: list[str]         # queries already run
    hits: list[Hit]        # graded, relevant evidence so far
    candidates: list[Hit]  # retrieved this round, awaiting grading
    seen: list[str]        # chunk ids already retrieved
    sufficient: bool
    retrievals: int
    calcs: list[str]
    steps: list[TraceStep]
    tokens_in: int
    tokens_out: int
    calls: int
    cached: bool
    answer: Answer
    stopped_by: str


def _numbered(hits: list[Hit], start: int = 1) -> str:
    return "\n\n".join(
        f"[{start + i}] {h.label}\nSection: {h.context}\n{h.text[:SNIPPET_CHARS]}"
        for i, h in enumerate(hits)
    )


def build_graph(retriever, model: str | None = None, max_retrievals: int = MAX_RETRIEVALS,
                max_tokens: int = MAX_TOKENS):
    from langgraph.graph import END, START, StateGraph

    def llm(state: State, prompt: str, system: str, schema):
        res = generate(prompt, system=system, model=model, response_schema=schema)
        usage = {
            "tokens_in": state.get("tokens_in", 0) + res.input_tokens,
            "tokens_out": state.get("tokens_out", 0) + res.output_tokens,
            "calls": state.get("calls", 0) + 1,
            "cached": state.get("cached", True) and res.cached,
        }
        return schema.model_validate_json(res.text), usage

    def over_budget(state: State) -> bool:
        return state.get("tokens_in", 0) + state.get("tokens_out", 0) >= max_tokens

    def plan(state: State) -> dict:
        p, usage = llm(state, f"QUESTION: {state['question']}", PLAN_SYSTEM, Plan)
        queries = [q.strip() for q in p.queries if q.strip()][:3] or [state["question"]]
        step = TraceStep(f"route={p.route}, calc={p.needs_calculation}", "plan",
                         " | ".join(queries), "")
        return {**usage, "route": p.route, "needs_calculation": p.needs_calculation,
                "pending": queries if p.route == "search" else [], "ran": [], "hits": [],
                "seen": [], "retrievals": 0, "calcs": [], "steps": [step]}

    def retrieve(state: State) -> dict:
        seen, new = list(state["seen"]), []
        for q in state["pending"]:
            for h in retriever.search(q, k=SEARCH_K):
                if h.chunk_id not in seen:
                    seen.append(h.chunk_id)
                    new.append(h)
        step = TraceStep("", "retrieve", " | ".join(state["pending"]),
                         f"{len(new)} new passages")
        return {"candidates": new, "seen": seen, "ran": state["ran"] + state["pending"],
                "pending": [], "retrievals": state["retrievals"] + 1,
                "steps": state["steps"] + [step]}

    def grade(state: State) -> dict:
        cands = state["candidates"]
        if not cands:
            g, usage = Grade(relevant=[], sufficient=False, better_query=""), {}
        else:
            prompt = (f"QUESTION: {state['question']}\n\nALREADY KEPT:\n"
                      f"{_numbered(state['hits']) or '(none)'}\n\nNEW PASSAGES:\n"
                      f"{_numbered(cands, start=len(state['hits']) + 1)}")
            g, usage = llm(state, prompt, GRADE_SYSTEM, Grade)
        offset = len(state["hits"])
        keep = [cands[n - offset - 1] for n in sorted(set(g.relevant))
                if offset < n <= offset + len(cands)]
        pending = []
        if not g.sufficient and g.better_query.strip() and g.better_query not in state["ran"]:
            pending = [g.better_query.strip()]
        step = TraceStep(f"kept {len(keep)}/{len(cands)}, sufficient={g.sufficient}", "grade",
                         pending[0] if pending else "", "")
        return {**usage, "hits": state["hits"] + keep, "candidates": [],
                "sufficient": g.sufficient, "pending": pending, "steps": state["steps"] + [step]}

    def after_plan(state: State) -> str:
        return "retrieve" if state["route"] == "search" else "generate"

    def after_grade(state: State) -> str:
        if state["sufficient"] or not state["pending"]:
            nxt = "calculate" if state["needs_calculation"] else "generate"
            return nxt
        if state["retrievals"] >= max_retrievals or over_budget(state):
            return "calculate" if state["needs_calculation"] else "generate"
        return "retrieve"

    def calculate(state: State) -> dict:
        if over_budget(state):
            return {"stopped_by": "max_tokens"}
        prompt = f"QUESTION: {state['question']}\n\nPASSAGES:\n{_numbered(state['hits'])}"
        cp, usage = llm(state, prompt, CALC_SYSTEM, CalcPlan)
        calcs, steps = list(state["calcs"]), list(state["steps"])
        for call in cp.calls[:3]:
            try:
                args = json.loads(call.args_json or "{}")
                out = call_tool(call.tool, args)
            except json.JSONDecodeError as e:
                args, out = {}, {"error": f"bad JSON: {e}"}
            if "result" in out:
                calcs.append(f"{call.tool}({json.dumps(args)}) = {out['result']}")
            steps.append(TraceStep("", call.tool, json.dumps(args), json.dumps(out)))
        return {**usage, "calcs": calcs, "steps": steps}

    def generate_node(state: State) -> dict:
        # Computed figures become numbered sources, so the answer can cite them.
        hits = state["hits"] + calculator_hits(state["calcs"])
        prompt = build_prompt(state["question"], hits)
        a, usage = llm(state, prompt, ANSWER_SYSTEM, Answer)
        stopped = state.get("stopped_by") or (
            "out_of_scope" if state["route"] == "out_of_scope"
            else "sufficient" if state.get("sufficient")
            else "max_retrievals")
        return {**usage, "answer": a, "stopped_by": stopped, "hits": hits,
                "steps": state["steps"] + [TraceStep("", "generate", "", "")]}

    g = StateGraph(State)
    for name, fn in [("plan", plan), ("retrieve", retrieve), ("grade", grade),
                     ("calculate", calculate), ("generate", generate_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after_plan, ["retrieve", "generate"])
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, ["retrieve", "calculate", "generate"])
    g.add_edge("calculate", "generate")
    g.add_edge("generate", END)
    return g.compile()


@traceable(run_type="chain", name="graph_agent", process_outputs=answer_outputs)
def run(question: str, retriever, model: str | None = None, graph=None, **caps) -> AgentResult:
    t0 = time.perf_counter()
    graph = graph or build_graph(retriever, model=model, **caps)
    s = graph.invoke({"question": question}, {"recursion_limit": 25})
    hits, parsed = s["hits"], s["answer"]
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
        cached=s.get("cached", False),
        input_tokens=s.get("tokens_in", 0),
        output_tokens=s.get("tokens_out", 0),
        seconds=time.perf_counter() - t0,
        model=model or "",
        steps=s["steps"],
        calls=s.get("calls", 0),
        stopped_by=s.get("stopped_by", ""),
    )
