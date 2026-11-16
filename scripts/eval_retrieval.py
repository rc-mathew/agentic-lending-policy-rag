"""Embed each chunking strategy into Qdrant (local mode) and score retrieval on the probe set.

    python scripts/eval_retrieval.py            # embeds once, then reuses the stored vectors
    python scripts/eval_retrieval.py --rebuild  # re-embed everything (after re-chunking)

A question counts as answered at k if any of the top-k chunks comes from an answer document and
its page range covers the answer page. Reports hit@1/3/5 and MRR@5 per strategy, the token check
against BGE-small's 512-token limit, and the misses so they can be inspected.

Memory: embeds in batches of 32. Close browsers first on an 8 GB laptop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_rag.chunking import STRATEGIES
from policy_rag.config import QDRANT_DIR, settings

PROCESSED = ROOT / "data" / "processed"
PROBES = ROOT / "data" / "eval" / "probe_questions.jsonl"
RESULTS = ROOT / "data" / "eval" / "retrieval_probe_results.json"
# BGE v1.5 recommends this prefix for short queries (not for the passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
K = 5
BATCH = 32


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def is_hit(payload: dict, answers: list[dict]) -> bool:
    return any(
        payload["doc_id"] == a["doc_id"] and payload["page_start"] <= a["page"] <= payload["page_end"]
        for a in answers
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-embed even if vectors exist")
    args = ap.parse_args()

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embed_model, device="cpu")
    limit = model.max_seq_length
    client = QdrantClient(path=str(QDRANT_DIR))
    probes = load_jsonl(PROBES)
    query_vecs = model.encode(
        [QUERY_PREFIX + p["question"] for p in probes], normalize_embeddings=True, batch_size=BATCH
    )

    results, misses = {}, {}
    for strategy in STRATEGIES:
        chunks = load_jsonl(PROCESSED / f"chunks_{strategy}.jsonl")
        name = f"probe_{strategy}"
        tokens = [len(t) for t in model.tokenizer([c["embed_text"] for c in chunks])["input_ids"]]
        over = sum(1 for t in tokens if t > limit)

        exists = client.collection_exists(name)
        if args.rebuild or not exists or client.count(name).count != len(chunks):
            t0 = time.perf_counter()
            vecs = model.encode(
                [c["embed_text"] for c in chunks],
                normalize_embeddings=True,
                batch_size=BATCH,
                show_progress_bar=True,
            )
            if exists:
                client.delete_collection(name)
            client.create_collection(
                name, vectors_config=VectorParams(size=vecs.shape[1], distance=Distance.COSINE)
            )
            for i in range(0, len(chunks), 256):
                client.upsert(
                    name,
                    points=[
                        PointStruct(id=j, vector=vecs[j].tolist(), payload=chunks[j])
                        for j in range(i, min(i + 256, len(chunks)))
                    ],
                )
            print(f"{strategy}: embedded {len(chunks)} chunks in {time.perf_counter() - t0:.0f}s")

        hits = {1: 0, 3: 0, 5: 0}
        rr = 0.0
        misses[strategy] = []
        for probe, qv in zip(probes, query_vecs, strict=True):
            found = client.query_points(name, query=qv.tolist(), limit=K, with_payload=True).points
            rank = next((r for r, pt in enumerate(found, 1) if is_hit(pt.payload, probe["answers"])), None)
            for k in hits:
                hits[k] += bool(rank and rank <= k)
            rr += 1 / rank if rank else 0.0
            if not rank or rank > 1:
                top = found[0].payload
                misses[strategy].append(
                    f"{probe['id']} rank={rank or '>5'} top1={top['doc_id']} p{top['page_start']}"
                    f"-{top['page_end']}"
                )
        n = len(probes)
        results[strategy] = {
            "chunks": len(chunks),
            "max_tokens": max(tokens),
            "over_limit": over,
            **{f"hit@{k}": round(v / n, 3) for k, v in hits.items()},
            "mrr@5": round(rr / n, 3),
        }

    print(f"\n{len(probes)} probe questions, top-{K}, model {settings.embed_model}\n")
    print(f"{'strategy':14} {'chunks':>6} {'max tok':>7} {'>512':>5} {'hit@1':>6} {'hit@3':>6} "
          f"{'hit@5':>6} {'MRR@5':>6}")
    for s, r in results.items():
        print(f"{s:14} {r['chunks']:>6} {r['max_tokens']:>7} {r['over_limit']:>5} {r['hit@1']:>6.0%} "
              f"{r['hit@3']:>6.0%} {r['hit@5']:>6.0%} {r['mrr@5']:>6.2f}")
    print("\nNot at rank 1:")
    for s, m in misses.items():
        print(f"  {s}: " + ("; ".join(m) if m else "none"))

    RESULTS.write_text(json.dumps({"results": results, "misses": misses}, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
