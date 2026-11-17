"""Run a system over the gold set and score it.

    python scripts/evaluate.py --system plain                   # verified gold items only
    python scripts/evaluate.py --system plain --include-drafts  # all 60 candidates
    python scripts/evaluate.py --system plain --limit 5         # quick check
    python scripts/evaluate.py --system graph --gold data/eval/calc_holdout.jsonl
    python scripts/evaluate.py --system graph --gold data/eval/calc_holdout.jsonl --calc-mode v1

Writes data/eval/results/<system>.jsonl (one row per question) and <system>_summary.json, and
prints the scores by question type. Calls are cached, so after a quota stop a re-run resumes
where it stopped without paying again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_rag.evaluate import load_gold, score_one, summarise
from policy_rag.llm import DailyQuotaExhausted

GOLD = ROOT / "data" / "eval" / "gold_candidates.jsonl"
OUT = ROOT / "data" / "eval" / "results"


def make_system(name: str, mode: str = "rerank"):
    """Every system is scored behind the same guardrails it runs with in the demo."""
    from policy_rag.guardrails import guarded

    return guarded(_raw_system(name, mode))


def _raw_system(name: str, mode: str):
    from policy_rag.retrieve import Retriever

    retriever = Retriever(mode=mode)
    if name == "plain":
        from policy_rag.rag import answer

        return lambda q: answer(q, retriever)
    if name == "react":
        from policy_rag.react_agent import run

        return lambda q: run(q, retriever)
    if name == "graph":
        from policy_rag import graph_agent

        graph = graph_agent.build_graph(retriever)  # compile once, reuse for every question
        return lambda q: graph_agent.run(q, retriever, graph=graph)
    raise SystemExit(f"unknown system {name!r} (available: plain, react, graph)")


def fmt(v) -> str:
    return "  –  " if v is None else f"{v:>5.0%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="plain")
    ap.add_argument("--retriever", default="rerank", choices=["dense", "hybrid", "rerank"])
    ap.add_argument("--gold", type=Path, default=GOLD)
    ap.add_argument("--include-drafts", action="store_true", help="score unverified items too")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calc-mode", choices=["v1", "v2"], default="v2",
                    help="v1 = calculator setup of the scored gold runs (ablation)")
    args = ap.parse_args()
    os.environ["CALC_MODE"] = args.calc_mode  # read when the agent modules are imported

    items = load_gold(args.gold, include_drafts=args.include_drafts)
    if not items:
        print("No verified gold items yet. Tick them in the review Sheet, or use --include-drafts.")
        return 1
    if args.limit:
        items = items[: args.limit]

    run = make_system(args.system, args.retriever)
    rows = []
    try:
        for i, item in enumerate(items, 1):
            row = score_one(item, run(item["question"]))
            rows.append(row)
            print(f"[{i}/{len(items)}] {item['id']} {item['type']:12} correct={row['correctness']}"
                  f" cite_acc={row['citation_accuracy']} abstained={row['abstained']}")
    except DailyQuotaExhausted as e:
        print(f"\nStopped after {len(rows)} questions: {e}\nRe-run later; cached answers are free.")

    if not rows:
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    # Default gold set keeps the original names; other question files and v1 get suffixes,
    # so a held-out run never overwrites the gold-set results.
    name = f"{args.system}_{args.retriever}"
    if args.gold.resolve() != GOLD.resolve():
        name += f"_{args.gold.stem}"
    if args.calc_mode != "v2":
        name += f"_calc{args.calc_mode}"
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summarise(rows)
    s["meta"] = {"system": args.system, "retriever": args.retriever, "items": len(rows),
                 "of": len(items),
                 "drafts_included": args.include_drafts, "gold": args.gold.name,
                 "calc_mode": args.calc_mode}
    (OUT / f"{name}_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")

    print(f"\n{args.system}: {len(rows)} of {len(items)} questions"
          f"{' (includes unverified drafts)' if args.include_drafts else ''}\n")
    print(f"{'':14} {'n':>3} {'correct':>7} {'cite acc':>8} {'retr hit':>8} {'cites gold':>10} "
          f"{'abstain ok':>10} {'false refuse':>12}")
    for k, b in s.items():
        if k in ("cost", "meta"):
            continue
        print(f"{k:14} {b['n']:>3} {fmt(b['correctness']):>7} {fmt(b['citation_accuracy']):>8} "
              f"{fmt(b['retrieval_hit']):>8} {fmt(b['cites_gold']):>10} "
              f"{fmt(b['abstention_correct']):>10} {fmt(b['false_refusal']):>12}")
    c = s["cost"]
    print(f"\navg tokens: system {c['avg_system_tokens']}, judge {c['avg_judge_tokens']} | "
          f"latency p50 {c['latency_p50_s']}s")
    print(f"Saved {OUT / (name + '.jsonl')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
