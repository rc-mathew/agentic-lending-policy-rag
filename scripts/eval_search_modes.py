"""Compare dense, hybrid (BM25 + dense) and hybrid + cross-encoder re-rank on the probe set.

    python scripts/eval_search_modes.py

Uses the `policy_chunks` index (section-aware chunks). Same hit rule as eval_retrieval.py: a top-k
chunk from an answer document whose page range covers the answer page. No LLM calls.
Saves data/eval/search_modes_results.json.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_rag.retrieve import MODES, Retriever

PROBES = ROOT / "data" / "eval" / "probe_questions.jsonl"
OUT = ROOT / "data" / "eval" / "search_modes_results.json"
K = 5


def rank_of(hits, answers) -> int | None:
    for r, h in enumerate(hits, 1):
        if any(h.doc_id == a["doc_id"] and h.page_start <= a["page"] <= h.page_end for a in answers):
            return r
    return None


def main() -> int:
    probes = [json.loads(line) for line in PROBES.open(encoding="utf-8")]
    retriever = Retriever(mode="rerank")  # loads everything once; modes switch below
    results, misses = {}, {}
    for mode in MODES:
        retriever.mode = mode
        ranks, secs = [], []
        for p in probes:
            t0 = time.perf_counter()
            hits = retriever.search(p["question"], k=K)
            secs.append(time.perf_counter() - t0)
            ranks.append(rank_of(hits, p["answers"]))
        n = len(probes)
        results[mode] = {
            **{f"hit@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 3) for k in (1, 3, 5)},
            "mrr@5": round(sum(1 / r for r in ranks if r) / n, 3),
            "latency_ms_p50": round(statistics.median(secs) * 1000),
        }
        misses[mode] = [f"{p['id']} rank={r or '>5'}" for p, r in zip(probes, ranks, strict=True)
                        if r != 1]

    print(f"\n{len(probes)} probe questions, top-{K}, section-aware index\n")
    print(f"{'mode':8} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR@5':>6} {'p50 ms':>7}")
    for m, r in results.items():
        print(f"{m:8} {r['hit@1']:>6.0%} {r['hit@3']:>6.0%} {r['hit@5']:>6.0%} {r['mrr@5']:>6.2f} "
              f"{r['latency_ms_p50']:>7}")
    print("\nNot at rank 1:")
    for m, ms in misses.items():
        print(f"  {m}: {', '.join(ms) or 'none'}")
    OUT.write_text(json.dumps({"results": results, "misses": misses}, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
