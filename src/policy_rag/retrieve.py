"""Retrieval over the section-aware index: dense, hybrid (BM25 + dense) and re-ranked.

Modes (Retriever(mode=...)):
- dense:  BGE-small embeddings, cosine search in local Qdrant.
- hybrid: dense + BM25 keyword search, merged by Reciprocal Rank Fusion (RRF). BM25 catches exact
          wording that embeddings blur: "SBI" vs "State Bank of India", "8:00 a.m.", "0700 hrs".
- rerank: hybrid candidates (top CANDIDATES) re-scored by a cross-encoder that reads the query
          and passage together; slower but sharper at rank 1. Default.

The index is the `policy_chunks` collection built by scripts/build_index.py; BM25 is built in
memory from the same payloads at start-up (~1,750 chunks, under a second). Qdrant local mode
allows one client per storage folder, so keep a single Retriever per process.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import QDRANT_DIR, settings

COLLECTION = "policy_chunks"
# BGE v1.5 recommends this prefix for short queries (not for passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~90 MB, CPU
CANDIDATES = 20
RRF_K = 60
MODES = ("dense", "hybrid", "rerank")

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "shall", "that", "the", "their", "this", "to", "was", "were", "will", "with", "which"]
)


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens without stop words ("8:00 a.m." -> ["8", "00", "m"])."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion: score(id) = sum over rankings of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda c: -scores[c])


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


def _hit(payload: dict, score: float) -> Hit:
    return Hit(score=score, **{f: payload[f] for f in Hit.__dataclass_fields__ if f != "score"})


class Retriever:
    def __init__(self, collection: str = COLLECTION, mode: str = "rerank"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        self.mode = mode
        self.model = SentenceTransformer(settings.embed_model, device="cpu")
        self.client = QdrantClient(path=str(QDRANT_DIR))
        self.collection = collection
        if not self.client.collection_exists(collection):
            raise RuntimeError(
                f"Qdrant collection {collection!r} not found. Run: python scripts/build_index.py"
            )
        self.payloads: dict[str, dict] = {}
        self.bm25 = None
        self.reranker = None
        if mode in ("hybrid", "rerank"):
            self._build_bm25()
        if mode == "rerank":
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(RERANK_MODEL, device="cpu")

    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        offset = None
        while True:
            batch, offset = self.client.scroll(self.collection, limit=512, offset=offset,
                                               with_payload=True)
            for r in batch:
                self.payloads[r.payload["chunk_id"]] = r.payload
            if offset is None:
                break
        self.ids = list(self.payloads)
        self.bm25 = BM25Okapi([tokenize(self.payloads[i]["embed_text"]) for i in self.ids])

    def dense(self, query: str, k: int) -> list[Hit]:
        vec = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        points = self.client.query_points(
            self.collection, query=vec.tolist(), limit=k, with_payload=True
        ).points
        return [_hit(p.payload, p.score) for p in points]

    def keyword(self, query: str, k: int) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [_hit(self.payloads[self.ids[i]], float(scores[i])) for i in top if scores[i] > 0]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if self.mode == "dense":
            return self.dense(query, k)
        n = max(k, CANDIDATES)
        dense, kw = self.dense(query, n), self.keyword(query, n)
        by_id = {h.chunk_id: h for h in dense + kw}
        fused = rrf([[h.chunk_id for h in dense], [h.chunk_id for h in kw]])
        if self.mode == "hybrid":
            return [by_id[c] for c in fused[:k]]
        cands = [by_id[c] for c in fused[:n]]
        # The cross-encoder reads query and passage together (title + section + text).
        scores = self.reranker.predict(
            [(query, self.payloads[h.chunk_id]["embed_text"]) for h in cands]
        )
        ranked = sorted(zip(cands, scores, strict=True), key=lambda x: -x[1])[:k]
        for h, s in ranked:
            h.score = float(s)
        return [h for h, _ in ranked]
