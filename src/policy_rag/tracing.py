"""Opt-in LangSmith tracing.

Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env (free Developer plan) to see every
question as one trace: the guardrail, the agent's steps or graph nodes (LangGraph traces itself),
each retrieval with its passages, and each Gemini call with tokens and latency. With tracing off,
or langsmith not installed, these decorators change nothing.

Guardrails redact personal identifiers before any call, so traces carry no PAN/Aadhaar/card data.
"""

from __future__ import annotations

from collections.abc import Callable


def traceable(**kwargs) -> Callable[[Callable], Callable]:
    try:
        from langsmith import traceable as _traceable
    except ImportError:  # langsmith is an optional extra
        return lambda fn: fn
    return _traceable(**kwargs)


def llm_outputs(result) -> dict:
    """LangSmith shows token usage for llm runs from `usage_metadata`."""
    return {
        "output": result.text,
        "cached": result.cached,
        "usage_metadata": {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.input_tokens + result.output_tokens,
        },
    }


def retriever_outputs(hits) -> dict:
    """LangSmith renders retriever runs from a list of documents."""
    return {"documents": [
        {"page_content": h.text, "type": "Document",
         "metadata": {"source": h.label, "chunk_id": h.chunk_id, "score": round(h.score, 4)}}
        for h in hits
    ]}


def answer_outputs(result) -> dict:
    return {
        "answerable": result.answerable,
        "answer": result.render(),
        "guardrail": getattr(result, "guardrail", ""),
        "calls": getattr(result, "calls", 1),
        "tokens": result.input_tokens + result.output_tokens,
    }
