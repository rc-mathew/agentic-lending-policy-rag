"""Ask the hand-written ReAct agent a question and watch its reasoning trace.

    python scripts/agent.py "A flat costs Rs 40 lakh. What is the maximum home loan?"
    python scripts/agent.py --steps 3 "..."     # tighter step cap

Prints each THOUGHT / ACTION / OBSERVATION, then the cited answer, calls, tokens and why it
stopped (finish / max_steps / max_tokens).
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_rag import react_agent
from policy_rag.retrieve import Retriever


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--steps", type=int, default=react_agent.MAX_STEPS)
    args = ap.parse_args()

    r = react_agent.run(args.question, Retriever(), max_steps=args.steps)
    print(f"\nQ: {r.question}\n")
    for i, s in enumerate(r.steps, 1):
        print(f"STEP {i}  thought: {s.thought}")
        print(f"        action:  {s.action} {s.detail}")
        if s.observation:
            first = s.observation.splitlines()
            shown = "\n".join(first[:4]) + ("\n..." if len(first) > 4 else "")
            print(textwrap.indent(shown, "        | "))
        print()
    print(r.render())
    print(f"\n[{r.calls} calls | {r.input_tokens} in / {r.output_tokens} out tokens | "
          f"{r.seconds:.1f}s | stopped by {r.stopped_by}{' | cached' if r.cached else ''}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
