"""PDF -> cleaned pages, one record per page, with the page number every citation needs.

Cleaning is generic (no per-document rules), so a new PDF needs no code change:

1. Strip trailing spaces and collapse runs of blank lines.
2. Drop page-number furniture near the top or bottom of a page: a bare "12", or "Page" / "|" /
   "6" split across lines (ICICI's "Page 2 | 6").
3. Drop running headers and footers: a line that appears near the top or bottom of at least half
   of a document's pages (digits ignored, so "Ver1.0 - Apr 2026" and page counters still match).
4. Re-join words hyphenated across a line break ("borrow-\\ner" -> "borrower"), and drop the
   footnote number glued to RBI amendment brackets ("1[(ia) ..." -> "[(ia) ..."). The footnote
   lines themselves ("1 Inserted with effect from ...") stay: they carry amendment dates.
5. Fall back to pdfplumber for a document whose pypdf output is broken into word fragments
   (ICICI's PDF: "non\\n acceptance", "pre\\n -\\n payment"). pypdf stays the default: it is
   several times faster and clean on the other documents.
6. Flag table-of-contents pages (several dot-leader lines). They are kept in the output, so the
   page numbering stays true, but chunking skips them.

Line breaks are kept: section-aware chunking splits on numbered paragraphs at line starts.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

EDGE_LINES = 4  # how many non-blank lines at the top/bottom count as "header/footer zone"
REPEAT_SHARE = 0.5  # a line in that zone on >= this share of pages is a running header/footer
MIN_PAGES_FOR_REPEATS = 3

_DOT_LEADER = re.compile(r"(\.\s?){5,}\s*\d+\s*$")
_PAGE_FURNITURE = re.compile(r"^(page|\||\d{1,4}|page\s*\d{1,4}(\s*(\||of)\s*\d{1,4})?)$", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"([a-z])-\n([a-z])")
_AMENDMENT_MARK = re.compile(r"(?<![\w.(])\d{1,3}\[")
FRAGMENT_SHARE = 0.05  # above this share of fragment lines, re-extract with pdfplumber


@dataclass
class Page:
    doc_id: str
    title: str
    issuer: str
    doc_type: str
    status: str
    updated_date: str
    page: int  # 1-based PDF page number
    text: str
    extractor: str
    is_toc: bool
    lines_removed: int

    def to_dict(self) -> dict:
        return asdict(self)


def _norm_for_repeat(line: str) -> str:
    """Key for spotting running headers/footers: case-folded, digits and spaces squashed."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.lower())).strip()


def _edge_indices(lines: list[str]) -> set[int]:
    """Indices of the first and last EDGE_LINES non-blank lines."""
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    return set(nonblank[:EDGE_LINES]) | set(nonblank[-EDGE_LINES:])


def find_running_lines(pages: list[list[str]]) -> set[str]:
    """Normalised lines that recur in the header/footer zone of most pages of one document."""
    if len(pages) < MIN_PAGES_FOR_REPEATS:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        keys = {_norm_for_repeat(lines[i]) for i in _edge_indices(lines)}
        counts.update(k for k in keys if k)
    threshold = max(2, int(len(pages) * REPEAT_SHARE + 0.999))
    return {k for k, c in counts.items() if c >= threshold}


def is_toc_page(text: str) -> bool:
    return sum(1 for ln in text.splitlines() if _DOT_LEADER.search(ln)) >= 3


def clean_page(lines: list[str], running: set[str]) -> tuple[str, int]:
    """Return the cleaned text of one page and how many non-blank lines were removed."""
    lines = [ln.rstrip() for ln in lines]
    edge = _edge_indices(lines)
    kept: list[str] = []
    removed = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s and i in edge and (_PAGE_FURNITURE.match(s) or _norm_for_repeat(s) in running):
            removed += 1
            continue
        kept.append(ln)
    text = "\n".join(kept)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _AMENDMENT_MARK.sub("[", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, removed


def fragment_share(pages: list[list[str]]) -> float:
    """Share of non-blank lines that look like broken-off word fragments: short, leading space."""
    lines = [ln for page in pages for ln in page if ln.strip()]
    if not lines:
        return 0.0
    return sum(1 for ln in lines if ln.startswith(" ") and len(ln.strip()) < 15) / len(lines)


def extract_pdf(path: Path) -> tuple[list[list[str]], str]:
    """Raw text of every page, split into lines, and the extractor used. Lazy imports keep the
    tests free of PDF libraries."""
    from pypdf import PdfReader

    pages = [(p.extract_text() or "").splitlines() for p in PdfReader(path).pages]
    if fragment_share(pages) <= FRAGMENT_SHARE:
        return pages, "pypdf"
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return [(p.extract_text() or "").splitlines() for p in pdf.pages], "pdfplumber"


def load_document(path: Path, meta: dict) -> list[Page]:
    raw, extractor = extract_pdf(path)
    running = find_running_lines(raw)
    out = []
    for n, lines in enumerate(raw, start=1):
        text, removed = clean_page(lines, running)
        out.append(
            Page(
                doc_id=meta["doc_id"],
                title=meta["title"],
                issuer=meta["issuer"],
                doc_type=meta["doc_type"],
                status=meta["status"],
                updated_date=meta.get("updated_date", ""),
                page=n,
                text=text,
                extractor=extractor,
                is_toc=is_toc_page("\n".join(lines)),
                lines_removed=removed,
            )
        )
    return out


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
