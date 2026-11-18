"""Logic behind the Streamlit demo (app.py), kept here so it can be tested without Streamlit.

The demo is public, and every question it answers spends the owner's free Gemini quota
(500 requests a day for 3.5 Flash-Lite). So it caps questions per visitor session and per day,
and the example questions are answered from the on-disk cache, which costs nothing.
"""

from __future__ import annotations

import datetime as dt
import os
import threading
from dataclasses import dataclass, field

SYSTEMS = {
    "Plain RAG": "plain",
    "ReAct agent": "react",
    "LangGraph agent": "graph",
}

SYSTEM_NOTES = {
    "plain": "One search, one Gemini call. The baseline: fast and cheap, no retries or tools.",
    "react": "A hand-written loop: the model chooses to search again, use a calculator, or answer "
             "(at most 5 steps).",
    "graph": "A LangGraph state machine: plan, retrieve, grade the passages, re-search if they "
             "are not enough, calculate, then answer.",
}

EXAMPLES = [
    "Kerala Grameena Bank contacts borrowers from 07:00. Is that allowed under RBI rules?",
    "How soon after I fully repay my loan must the bank return my property documents?",
    "A flat costs ₹40 lakh. What is the maximum home loan under RBI's LTV limits?",
    "Will SBI acknowledge receipt of my loan application?",
    "What is RBI's repo rate today?",
]

# Gemini calls a question costs, at most, per system (plain: 1; agents: capped by their loops).
MAX_CALLS = {"plain": 1, "react": 6, "graph": 7}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class DailyBudget:
    """Shared across sessions of one server process: Gemini calls allowed per day."""

    limit: int = field(default_factory=lambda: _int_env("DEMO_DAILY_CALLS", 300))
    day: str = ""
    used: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _roll(self) -> None:
        # The day of Gemini's quota, which resets at midnight Pacific (UTC-8; DST ignored).
        today = dt.datetime.now(dt.timezone(dt.timedelta(hours=-8))).date().isoformat()
        if today != self.day:
            self.day, self.used = today, 0

    def can_spend(self, calls: int) -> bool:
        with self.lock:
            self._roll()
            return self.used + calls <= self.limit

    def spend(self, calls: int) -> None:
        with self.lock:
            self._roll()
            self.used += calls

    def left(self) -> int:
        with self.lock:
            self._roll()
            return max(0, self.limit - self.used)


SESSION_LIMIT = _int_env("DEMO_SESSION_QUESTIONS", 10)


def run(system: str, question: str, retriever):
    """Answer with the chosen system, behind the guardrails. Returns a RagResult/AgentResult."""
    from .guardrails import guarded

    if system == "plain":
        from .rag import answer

        fn = lambda q: answer(q, retriever)
    elif system == "react":
        from .react_agent import run as react_run

        fn = lambda q: react_run(q, retriever)
    elif system == "graph":
        from .graph_agent import run as graph_run

        fn = lambda q: graph_run(q, retriever)
    else:
        raise ValueError(f"unknown system {system!r}")
    return guarded(fn)(question)


def calls_spent(result) -> int:
    """Gemini calls that actually hit the API (cache hits cost nothing)."""
    if getattr(result, "model", "") == "guardrail":
        return 0
    if result.cached:
        return 0
    return getattr(result, "calls", 1) or 1


# Calculator outputs that are rupee amounts; the rest are months, percentages or ratios.
_RUPEE_TOOLS = {"emi", "max_loan_for_ltv"}
_RUPEE_KEYS = {"max_loan"}
_CALCULATOR = "calculator"  # retrieve.CALCULATOR; not imported, to keep this module light


def readable_calculation(text: str) -> str:
    """Display line for a calculator source: "tool(...) = 400000" -> "tool: ₹4,00,000".

    Page display only. The agents and the scored runs see the raw text, unchanged.
    """
    import ast

    from .tools import format_inr

    call, sep, raw = text.partition(" = ")
    if not sep:
        return text
    tool = call.split("(", 1)[0]
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return text
    if isinstance(value, dict):
        parts = [f"{k}: {format_inr(v) if k in _RUPEE_KEYS else v}" for k, v in value.items()]
        return f"{tool}: " + ", ".join(parts)
    if isinstance(value, (int, float)) and tool in _RUPEE_TOOLS:
        return f"{tool}: {format_inr(value)}"
    return f"{tool}: {value}"


def answer_view(result) -> dict:
    """Plain data for the page: verdict text, claims with citations, the sources they cite."""
    blocked = getattr(result, "guardrail", "").startswith("blocked")
    used = sorted({n for c in result.claims for n in c["sources"]})
    sources = []
    for n in used:
        if 1 <= n <= len(result.hits):
            h = result.hits[n - 1]
            sources.append({
                "n": n,
                "label": h.label,
                "issuer": h.issuer,
                "repealed": h.status == "repealed",
                "section": h.context,
                "text": h.text,
                "readable": readable_calculation(h.text) if h.doc_id == _CALCULATOR else "",
            })
    return {
        "blocked": blocked,
        "answerable": result.answerable and not blocked,
        "claims": [
            {"text": c["text"], "cites": "".join(f"[{n}]" for n in c["sources"])}
            for c in result.claims
        ],
        "sources": sources,
        "other_hits": [h.label for i, h in enumerate(result.hits, 1) if i not in used],
        "guardrail": getattr(result, "guardrail", ""),
        "steps": [
            {"action": s.action, "thought": s.thought, "detail": s.detail,
             "observation": s.observation}
            for s in getattr(result, "steps", [])
        ],
        "stats": {
            "calls": getattr(result, "calls", 0 if blocked else 1),
            "tokens_in": result.input_tokens,
            "tokens_out": result.output_tokens,
            "seconds": round(result.seconds, 1),
            "cached": result.cached,
            "stopped_by": getattr(result, "stopped_by", ""),
        },
    }
