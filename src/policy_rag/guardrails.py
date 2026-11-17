"""Guardrails around every system (plain RAG, ReAct, LangGraph): one wrapper, same rules.

Input, before anything is sent to the LLM:
- empty or over-long questions are rejected;
- personal identifiers are redacted (PAN, Aadhaar, card and account numbers, phone, email):
  a policy question never needs them, and they must not reach a third-party API or logs;
- obvious prompt-injection attempts ("ignore previous instructions", "reveal your system
  prompt") are refused.

Output, after the system answers:
- every claim must cite at least one retrieved source; uncited claims are removed;
- if nothing cited is left, the answer becomes an abstention.

A blocked question returns an abstention with `guardrail` set, so the evaluation and the demo
handle it like any other answer.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from .rag import RagResult
from .tracing import answer_outputs, traceable

MAX_QUESTION_CHARS = 1000

_PII = [  # most specific first: a 16-digit card contains a 12-digit "Aadhaar", a phone is digits
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    ("CARD", re.compile(r"\b(?:[0-9][ -]?){12,18}[0-9]\b")),
    ("AADHAAR", re.compile(r"\b[2-9][0-9]{3}[ -]?[0-9]{4}[ -]?[0-9]{4}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+91[ -]?)?[6-9][0-9]{9}(?!\d)")),
    ("ACCOUNT", re.compile(r"\b[0-9]{9,18}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
]

_INJECTION = re.compile(
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules)"
    r"|disregard (your|the) (instructions|rules|system prompt)"
    r"|(reveal|print|show) (me )?(your|the) (system )?prompt"
    r"|you are now (?!eligible)",
    re.IGNORECASE,
)


@dataclass
class InputCheck:
    ok: bool
    question: str          # redacted question to send on
    redacted: list[str]    # kinds of identifier removed
    reason: str = ""


def check_input(question: str) -> InputCheck:
    q = (question or "").strip()
    if not q:
        return InputCheck(False, q, [], "empty question")
    if len(q) > MAX_QUESTION_CHARS:
        return InputCheck(False, q[:MAX_QUESTION_CHARS], [],
                          f"question longer than {MAX_QUESTION_CHARS} characters")
    if _INJECTION.search(q):
        return InputCheck(False, q, [], "looks like an attempt to override the assistant's rules")
    redacted = []
    for kind, pat in _PII:
        q, n = pat.subn(f"[{kind}]", q)
        if n:
            redacted.append(kind)
    return InputCheck(True, q, redacted)


def enforce_citations(result) -> list[str]:
    """Drop claims that cite nothing; abstain if none remain. Returns notes on what changed."""
    notes = []
    kept = [c for c in result.claims if c["sources"]]
    if len(kept) < len(result.claims):
        notes.append(f"removed {len(result.claims) - len(kept)} uncited claim(s)")
        result.claims = kept
    if result.answerable and not kept:
        result.answerable = False
        notes.append("no cited claims left: abstained")
    return notes


def guarded(system: Callable[[str], RagResult]) -> Callable[[str], RagResult]:
    """Wrap a system's run(question) with the input and output guardrails."""
    @traceable(run_type="chain", name="guarded", process_outputs=answer_outputs,
               process_inputs=lambda i: {"question": check_input(i["question"]).question})
    def run(question: str):
        check = check_input(question)
        if not check.ok:
            r = RagResult(question=question, answerable=False, claims=[], hits=[],
                          seconds=0.0, model="guardrail")
            r.guardrail = f"blocked: {check.reason}"
            return r
        t0 = time.perf_counter()
        r = system(check.question)
        r.question = question
        notes = enforce_citations(r)
        if check.redacted:
            notes.insert(0, f"redacted {', '.join(check.redacted)}")
        r.guardrail = "; ".join(notes)
        r.seconds = max(r.seconds, time.perf_counter() - t0)
        return r
    return run
