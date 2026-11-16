"""Extract and clean every PDF in data/sources.csv into data/processed/pages.jsonl.

    python scripts/extract_pages.py
    python scripts/extract_pages.py --show rbi_cb_kyc:12   # print one cleaned page

One JSON record per page: document metadata, 1-based page number, cleaned text, a TOC flag.
Prints a per-document summary so problems (empty pages, odd headers) are visible at once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_rag.ingest import load_document, read_manifest

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "pages.jsonl"
MANIFEST = ROOT / "data" / "sources.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="doc_id:page to print after extraction, e.g. rbi_cb_kyc:12")
    args = ap.parse_args()

    t0 = time.perf_counter()
    rows = read_manifest(MANIFEST)
    pages = []
    print(f"{'doc_id':40} {'pages':>5} {'toc':>4} {'empty':>5} {'chars/pg':>8} {'hdr/ftr cut':>11}")
    for meta in rows:
        pdf = RAW / meta["filename"]
        if not pdf.exists():
            print(f"{meta['doc_id']:40} MISSING {pdf.name}")
            continue
        doc = load_document(pdf, meta)
        pages.extend(doc)
        body = [p for p in doc if not p.is_toc]
        empty = sum(1 for p in body if len(p.text) < 100)
        avg = sum(len(p.text) for p in body) // max(1, len(body))
        cut = sum(p.lines_removed for p in doc)
        toc = sum(p.is_toc for p in doc)
        print(f"{meta['doc_id']:40} {len(doc):>5} {toc:>4} {empty:>5} {avg:>8} {cut:>11}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    body = [p for p in pages if not p.is_toc]
    chars = sum(len(p.text) for p in body)
    print(
        f"\n{len(pages)} pages ({len(pages) - len(body)} TOC skipped later), "
        f"{chars:,} chars of body text (~{chars // 4:,} tokens) -> {OUT.relative_to(ROOT)} "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    if args.show:
        doc_id, _, n = args.show.partition(":")
        hit = next((p for p in pages if p.doc_id == doc_id and p.page == int(n or 1)), None)
        print(f"\n===== {args.show} =====\n{hit.text if hit else 'not found'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
