"""Download every PDF listed in data/sources.csv into data/raw/.

    python scripts/download_corpus.py          # skip files already downloaded
    python scripts/download_corpus.py --force  # re-download everything

Standard library only, so it runs before `pip install`. For each row it checks the file
really is a PDF (bank sites often return an HTML error page with status 200), then writes
bytes, sha256, download time and, if pypdf is installed, the page count back into the
manifest. Rows that fail are listed at the end for manual download.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "sources.csv"
RAW = ROOT / "data" / "raw"

# Several bank sites reject the default Python user agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*;q=0.8",
}


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # follows redirects
        return resp.read()


def page_count(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        return str(len(PdfReader(path).pages))
    except Exception:  # noqa: BLE001 - a broken PDF should not stop the run
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    with MANIFEST.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    RAW.mkdir(parents=True, exist_ok=True)
    failed: list[tuple[str, str]] = []

    for row in rows:
        dest = RAW / row["filename"]
        label = f"{row['doc_id']:42}"
        if dest.exists() and not args.force:
            print(f"  skip  {label} (already downloaded)")
        else:
            try:
                data = fetch(row["url"])
                if not data.startswith(b"%PDF"):
                    raise ValueError(f"not a PDF (got {data[:15]!r}); link has probably moved")
                dest.write_bytes(data)
                row["downloaded_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                print(f"  ok    {label} {len(data) / 1024:,.0f} KB")
                time.sleep(1)  # be polite to the servers
            except Exception as e:  # noqa: BLE001 - report and continue
                failed.append((row["doc_id"], f"{type(e).__name__}: {e}"))
                print(f"  FAIL  {label} {type(e).__name__}: {e}")
                continue
        blob = dest.read_bytes()
        row["bytes"] = str(len(blob))
        row["sha256"] = hashlib.sha256(blob).hexdigest()
        row["pages"] = page_count(dest) or row.get("pages", "")

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = len(rows) - len(failed)
    pages = sum(int(r["pages"]) for r in rows if r.get("pages", "").isdigit())
    print(f"\n{ok}/{len(rows)} documents in data/raw/" + (f", {pages:,} pages" if pages else ""))
    if failed:
        print("\nDownload these by hand (open the page_url or search the title), save under the")
        print("filename in sources.csv, then re-run this script to fill in pages and sha256:")
        for doc_id, err in failed:
            print(f"  - {doc_id}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
