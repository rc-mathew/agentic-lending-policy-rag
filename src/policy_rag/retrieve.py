"""Dense retrieval over the section-aware index (BGE-small + local Qdrant).

The index is the `policy_chunks` collection, built by scripts/build_index.py. Qdrant local mode
allows one client per storage folder at a time, so keep a single Retriever per process.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import QDRANT_DIR, settings

COLLECTION = "policy_chunks"
# BGE v1.5 recommends this prefix for short queries (not for passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Hit:
    score: float
    chunk_id: str
    doc_id: str
    title: str
    issuer: str
    status: str
    page_start: int
    page_end: int
    context: str
    text: str

    @property
    def pages(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"

    @property
    def label(self) -> str:
        """Human citation: issuer, short title, pages."""
        return f"{self.title} ({self.issuer}), {self.pages}"


class Retriever:
    def __init__(self, collection: str = COLLECTION):
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embed_model, device="cpu")
        self.client = QdrantClient(path=str(QDRANT_DIR))
        self.collection = collection
        if not self.client.collection_exists(collection):
            raise RuntimeError(
                f"Qdrant collection {collection!r} not found. Run: python scripts/build_index.py"
            )

    def search(self, query: str, k: int = 5) -> list[Hit]:
        vec = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        points = self.client.query_points(
            self.collection, query=vec.tolist(), limit=k, with_payload=True
        ).points
        return [
            Hit(
                score=p.score,
                **{f: p.payload[f] for f in Hit.__dataclass_fields__ if f != "score"},
            )
            for p in points
        ]
