"""Plain RAG: retrieve top-k chunks once, then one Gemini call that must cite every claim.

The model returns structured JSON (Pydantic schema): a list of claims, each with the numbers of
the sources that support it, or `answerable=false` when the sources do not contain the answer.
Citations are checked after the call: numbers outside the source list are dropped and counted,
and a claim left with no valid source is reported as uncited. This is the baseline the agents
are compared against, so it has no query rewriting, no retries and no tools.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .llm import generate
from .retrieve import Hit
from .tracing import answer_outputs, traceable

SYSTEM = """You answer questions about Indian bank lending rules using ONLY the numbered sources.

Rules:
- Every claim must cite the source numbers that support it. Never cite a source that does not
  say it.
- Use only facts stated in the sources. Copy numbers, days, amounts and percentages exactly.
- If the sources do not contain the answer, set answerable to false and give no claims. Do not
  answer from general knowledge.
- RBI Directions are the regulation; a bank's own policy applies only to that bank. Say which one
  a claim comes from when it matters.
- Fill "reasoning" BEFORE the claims. Write each rule's limit and each value being checked,
  then compare them step by step (e.g. "RBI: no calls before 08:00. Bank: from 07:00.
  07:00 is earlier than 08:00, so the bank's window starts inside the barred period").
  The reasoning is never shown to the user.
- The FIRST claim answers the question directly, following the reasoning. For a yes/no,
  "is it allowed" or comparison question, start it with Yes, No or Partly and give the reason,
  citing every source the conclusion rests on (e.g. both the RBI rule and the bank's policy).
  Supporting facts follow.
- Keep it short: at most 5 claims, each one sentence."""


class Claim(BaseModel):
    text: str = Field(description="One sentence, a single fact from the sources.")
    sources: list[int] = Field(description="Numbers of the sources that state this fact.")


class Answer(BaseModel):
    # First field on purpose: the model writes its working before the claims (it generates in
    # schema order). Flash-Lite answered "Yes, 07:00 is allowed" against an 08:00 limit when it
    # had to give the verdict without working it out first.
    reasoning: str = Field(
        default="",
        description="Working, written first: each limit and value, compared step by step. "
        "Not shown to the user.",
    )
    answerable: bool = Field(description="False if the sources do not contain the answer.")
    claims: list[Claim]


@dataclass
class RagResult:
    question: str
    answerable: bool
    claims: list[dict]  # {"text", "sources": [hit index, 1-based]}
    hits: list[Hit]
    invalid_citations: int = 0
    uncited_claims: int = 0
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    model: str = ""
    raw: str = field(default="", repr=False)

    def render(self) -> str:
        if not self.answerable:
            body = "I can't answer that from the documents I have."
        else:
            body = "\n".join(
                f"- {c['text']} " + "".join(f"[{n}]" for n in c["sources"]) for c in self.claims
            )
        used = sorted({n for c in self.claims for n in c["sources"]})
        refs = "\n".join(f"[{n}] {self.hits[n - 1].label}" for n in used)
        return body + (f"\n\nSources:\n{refs}" if refs else "")


def format_sources(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        status = " [REPEALED]" if h.status == "repealed" else ""
        blocks.append(f"[{i}] {h.label}{status}\nSection: {h.context}\n{h.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[Hit]) -> str:
    return f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"


@traceable(run_type="chain", name="plain_rag", process_outputs=answer_outputs)
def answer(question: str, retriever, k: int = 5, model: str | None = None) -> RagResult:
    t0 = time.perf_counter()
    hits = retriever.search(question, k=k)
    res = generate(build_prompt(question, hits), system=SYSTEM, model=model, response_schema=Answer)
    parsed = Answer.model_validate_json(res.text)

    claims, invalid, uncited = [], 0, 0
    for c in parsed.claims:
        valid = sorted({n for n in c.sources if 1 <= n <= len(hits)})
        invalid += len(set(c.sources)) - len(valid)
        uncited += not valid
        claims.append({"text": c.text, "sources": valid})

    return RagResult(
        question=question,
        answerable=parsed.answerable and bool(claims),
        claims=claims if parsed.answerable else [],
        hits=hits,
        invalid_citations=invalid,
        uncited_claims=uncited,
        cached=res.cached,
        input_tokens=res.input_tokens,
        output_tokens=res.output_tokens,
        seconds=time.perf_counter() - t0,
        model=res.model,
        raw=res.text,
    )
