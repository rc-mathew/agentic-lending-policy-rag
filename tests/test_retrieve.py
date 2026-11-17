"""Offline tests for hybrid retrieval: tokenizer, RRF fusion, BM25 path and re-rank ordering."""

import pytest

from policy_rag import retrieve


def test_tokenize_keeps_numbers_and_drops_stopwords():
    assert retrieve.tokenize("Calling the borrower before 8:00 a.m. (0700 hrs)") == [
        "calling", "borrower", "before", "8", "00", "m", "0700", "hrs"]


def test_rrf_rewards_agreement():
    fused = retrieve.rrf([["a", "b", "c"], ["c", "a", "d"]])
    assert fused[0] == "a"            # rank 1 + rank 2
    assert fused.index("c") < fused.index("b") < fused.index("d")


def _payload(cid, text, doc="d"):
    return {"chunk_id": cid, "doc_id": doc, "title": "T", "issuer": "RBI", "status": "current",
            "page_start": 1, "page_end": 1, "context": "ctx", "text": text, "embed_text": text}


class _Fake(retrieve.Retriever):
    """Retriever without Qdrant or models: dense results are scripted."""
    def __init__(self, payloads, dense_order, mode="hybrid", rerank_scores=None):
        from rank_bm25 import BM25Okapi
        self.mode = mode
        self.payloads = {p["chunk_id"]: p for p in payloads}
        self.ids = list(self.payloads)
        self.bm25 = BM25Okapi([retrieve.tokenize(p["embed_text"]) for p in payloads])
        self._dense = dense_order
        self.reranker = None
        if rerank_scores is not None:
            self.reranker = type("R", (), {"predict": lambda _s, pairs: [
                rerank_scores[self._cid_for(t)] for _, t in pairs]})()

    def _cid_for(self, text):
        return next(c for c, p in self.payloads.items() if p["embed_text"] == text)

    def dense(self, query, k):
        return [retrieve._hit(self.payloads[c], 1.0) for c in self._dense[:k]]


DOCS = [
    _payload("rbi_hours", "No calling the borrower before 8:00 a.m. and after 7:00 p.m."),
    _payload("kgb_hours", "Bank representatives contact the borrower between 0700 hrs and 1900 hrs"),
    _payload("ltv", "LTV ratio for housing loans up to 80 per cent"),
    _payload("kyc", "Periodic updation of KYC for high risk customers"),
]


def test_keyword_search_finds_exact_wording_dense_missed():
    r = _Fake(DOCS, dense_order=["ltv", "kyc", "rbi_hours", "kgb_hours"])
    hits = r.search("contact borrower 0700 hrs", k=2)
    assert "kgb_hours" in [h.chunk_id for h in hits]


def test_rerank_reorders_candidates():
    r = _Fake(DOCS, dense_order=["ltv", "kyc", "rbi_hours", "kgb_hours"], mode="rerank",
              rerank_scores={"rbi_hours": 9.0, "kgb_hours": 5.0, "ltv": 1.0, "kyc": 0.5})
    hits = r.search("recovery call hours", k=2)
    assert [h.chunk_id for h in hits] == ["rbi_hours", "kgb_hours"]
    assert hits[0].score == 9.0


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        retrieve.Retriever(mode="magic")
