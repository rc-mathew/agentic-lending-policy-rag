"""Build the `policy_chunks` Qdrant collection from the section-aware chunks.

    python scripts/build_index.py

If eval_retrieval.py has already embedded the section-aware chunks (collection
`probe_section_aware`) and the chunk file has not changed since, the vectors are copied over
instead of re-embedded, which takes seconds. Otherwise everything is embedded again (~3 min).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_rag.config import QDRANT_DIR, settings
from policy_rag.retrieve import COLLECTION

CHUNKS = ROOT / "data" / "processed" / "chunks_section_aware.jsonl"
PROBE = "probe_section_aware"


def main() -> int:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    ids = [c["chunk_id"] for c in chunks]
    client = QdrantClient(path=str(QDRANT_DIR))
    t0 = time.perf_counter()

    points: list[PointStruct] = []
    if client.collection_exists(PROBE) and client.count(PROBE).count == len(chunks):
        records, offset = [], None
        while True:
            batch, offset = client.scroll(
                PROBE, limit=512, offset=offset, with_vectors=True, with_payload=True
            )
            records.extend(batch)
            if offset is None:
                break
        if sorted(r.payload["chunk_id"] for r in records) == sorted(ids):
            points = [PointStruct(id=r.id, vector=r.vector, payload=r.payload) for r in records]
            how = f"copied from {PROBE}"

    if not points:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.embed_model, device="cpu")
        vecs = model.encode(
            [c["embed_text"] for c in chunks],
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True,
        )
        points = [PointStruct(id=i, vector=v.tolist(), payload=c) for i, (v, c) in
                  enumerate(zip(vecs, chunks, strict=True))]
        how = "embedded"

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION, vectors_config=VectorParams(size=len(points[0].vector), distance=Distance.COSINE)
    )
    for i in range(0, len(points), 256):
        client.upsert(COLLECTION, points=points[i : i + 256])
    print(f"{COLLECTION}: {len(points)} chunks ({how}) in {time.perf_counter() - t0:.1f}s")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
