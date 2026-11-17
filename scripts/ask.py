"""Ask the plain-RAG baseline a question, or run it over the probe set.

    python scripts/ask.py "Can a bank charge interest on penal charges?"
    python scripts/ask.py --probe          # all 15 probe questions, saves answers to data/eval/
    python scripts/ask.py --probe --limit 3

Each answer shows its claims with [n] citations, the sources used, and cost/latency. Answers are
cached on disk, so re-running only spends quota on new or changed prompts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_rag.guardrails import guarded
from policy_rag.llm import DailyQuotaExhausted
from policy_rag.rag import answer
from policy_rag.retrieve import Retriever

PROBES = ROOT / "data" / "eval" / "probe_questions.jsonl"
OUT = ROOT / "data" / "eval" / "plain_rag_probe_answers.jsonl"


def show(r) -> None:
    print(f"\nQ: {r.question}\n")
    print(r.render())
    flags = []
    if r.invalid_citations:
        flags.append(f"{r.invalid_citations} invalid citation(s) dropped")
    if r.uncited_claims:
        flags.append(f"{r.uncited_claims} uncited claim(s)")
    if getattr(r, "guardrail", ""):
        flags.append(f"guardrail: {r.guardrail}")
    print(
        f"\n[{r.model}{' cached' if r.cached else ''} | {r.input_tokens} in / {r.output_tokens} out "
        f"tokens | {r.seconds:.1f}s{' | ' + '; '.join(flags) if flags else ''}]"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--probe", action="store_true", help="run the 15 probe questions")
    ap.add_argument("--limit", type=int, default=0, help="only the first N probe questions")
    ap.add_argument("-k", type=int, default=5, help="chunks retrieved")
    ap.add_argument("--retriever", default="rerank", choices=["dense", "hybrid", "rerank"])
    args = ap.parse_args()
    if not args.question and not args.probe:
        ap.error("give a question or --probe")

    retriever = Retriever(mode=args.retriever)
    ask = guarded(lambda q: answer(q, retriever, k=args.k))
    if args.question:
        show(ask(args.question))
        return 0

    probes = [json.loads(line) for line in PROBES.open(encoding="utf-8")]
    if args.limit:
        probes = probes[: args.limit]
    rows = []
    try:
        for p in probes:
            r = ask(p["question"])
            show(r)
            gold = {(a["doc_id"], a["page"]) for a in p["answers"]}
            cited = [r.hits[n - 1] for c in r.claims for n in c["sources"]]
            rows.append({
                "id": p["id"],
                "question": p["question"],
                "answerable": r.answerable,
                "claims": r.claims,
                "cites_gold_page": any(
                    (h.doc_id, pg) in gold for h in cited for pg in range(h.page_start, h.page_end + 1)
                ),
                "retrieved": [h.chunk_id for h in r.hits],
                "invalid_citations": r.invalid_citations,
                "uncited_claims": r.uncited_claims,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "seconds": round(r.seconds, 2),
                "cached": r.cached,
            })
    except DailyQuotaExhausted as e:
        print(f"\nStopped: {e}")

    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(rows)
    if n:
        fresh = [r for r in rows if not r["cached"]]
        print(
            f"\n{n} answered | answerable {sum(r['answerable'] for r in rows)}/{n} | "
            f"cites the gold page {sum(r['cites_gold_page'] for r in rows)}/{n} | "
            f"invalid citations {sum(r['invalid_citations'] for r in rows)} | "
            f"uncited claims {sum(r['uncited_claims'] for r in rows)} | "
            f"API calls this run {len(fresh)} | "
            f"avg tokens in/out {sum(r['input_tokens'] for r in rows) // n}/"
            f"{sum(r['output_tokens'] for r in rows) // n}"
        )
        print(f"Saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
