"""Ask the LangGraph agent a question and watch the path it takes through the graph.

    python scripts/graph_agent.py "Will SBI acknowledge receipt of my loan application?"
    python scripts/graph_agent.py --mermaid      # print the graph as a Mermaid diagram

Prints each node visited (plan / retrieve / grade / calculator / generate), the cited answer,
calls, tokens and why it stopped (sufficient / max_retrievals / out_of_scope / max_tokens).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_rag import graph_agent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--mermaid", action="store_true", help="print the graph as Mermaid and exit")
    args = ap.parse_args()

    if args.mermaid:
        print(graph_agent.build_graph(retriever=None).get_graph().draw_mermaid())
        return 0
    if not args.question:
        ap.error("give a question, or --mermaid")

    from policy_rag.guardrails import guarded
    from policy_rag.retrieve import Retriever

    retriever = Retriever()
    r = guarded(lambda q: graph_agent.run(q, retriever))(args.question)
    print(f"\nQ: {r.question}\n")
    for s in getattr(r, "steps", []):
        line = f"  {s.action:16} {s.thought}"
        if s.detail:
            line += f"\n  {'':16} -> {s.detail}"
        if s.observation and s.action not in ("plan", "grade", "generate"):
            line += f"\n  {'':16} = {s.observation}"
        print(line)
    print()
    print(r.render())
    print(f"\n[{getattr(r, 'calls', 0)} calls | {r.input_tokens} in / {r.output_tokens} out tokens | "
          f"{r.seconds:.1f}s | stopped: {getattr(r, 'stopped_by', 'guardrail')}{' | cached' if r.cached else ''}]")
    if getattr(r, "guardrail", ""):
        print(f"[guardrail: {r.guardrail}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
