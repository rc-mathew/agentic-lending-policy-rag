"""Chunk data/processed/pages.jsonl with every strategy and print a comparison.

    python scripts/chunk_corpus.py
    python scripts/chunk_corpus.py --show section_aware:rbi_cb_credit_facilities:0040

Writes data/processed/chunks_<strategy>.jsonl. The stats show size spread and how often a chunk
starts mid-paragraph or mixes two sections, which is what section_aware is meant to fix.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_rag.chunking import STRATEGIES, assemble, chunk_documents

PROCESSED = ROOT / "data" / "processed"
_PARA_START = re.compile(r"^(\d{1,3}[A-Z]{0,2}\.\s|Chapter\s+[IVXLC]+|[A-Z]{1,2}\.\s+[A-Z])")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="strategy:doc_id:index, e.g. section_aware:rbi_cb_kyc:0012")
    args = ap.parse_args()

    pages = [json.loads(line) for line in (PROCESSED / "pages.jsonl").open(encoding="utf-8")]
    docs = assemble(pages)
    print(f"{len(docs)} documents, {sum(len(d.text) for d in docs):,} chars\n")
    print(f"{'strategy':14} {'chunks':>6} {'mean':>6} {'p50':>6} {'p95':>6} {'max':>6} "
          f"{'~max tok':>8} {'multi-page':>10} {'clean start':>11}")
    for strategy in STRATEGIES:
        chunks = chunk_documents(docs, strategy)
        sizes = [len(c.embed_text) for c in chunks]
        rbi = [c for c in chunks if c.issuer == "RBI"]
        clean = sum(1 for c in rbi if _PARA_START.match(c.text)) / max(1, len(rbi))
        multi = sum(1 for c in chunks if c.page_end > c.page_start) / len(chunks)
        p95 = sorted(sizes)[int(len(sizes) * 0.95)]
        print(f"{strategy:14} {len(chunks):>6} {statistics.mean(sizes):>6.0f} "
              f"{statistics.median(sizes):>6.0f} {p95:>6} {max(sizes):>6} {max(sizes) // 4:>8} "
              f"{multi:>10.0%} {clean:>11.0%}")
        out = PROCESSED / f"chunks_{strategy}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({**c.to_dict(), "embed_text": c.embed_text}, ensure_ascii=False) + "\n")

    print("\nclean start = share of RBI chunks that begin at a chapter, section or numbered paragraph")
    if args.show:
        strategy, doc_id, idx = args.show.split(":")
        want = f"{doc_id}:{strategy}:{idx}"
        for line in (PROCESSED / f"chunks_{strategy}.jsonl").open(encoding="utf-8"):
            c = json.loads(line)
            if c["chunk_id"] == want:
                print(f"\n===== {want}  pages {c['page_start']}-{c['page_end']} =====\n{c['embed_text']}")
                break
        else:
            print(f"\n{want} not found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
