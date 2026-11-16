"""Three chunking strategies over the cleaned pages, compared on the same retrieval probe set.

Every strategy works on one document at a time: its body pages (TOC pages skipped) joined into
one string, with a char-offset -> page map, so each chunk records the page range it came from.

- fixed:          1,200-char windows, 150-char overlap, cut at the nearest space. Ignores structure.
- recursive:      LangChain RecursiveCharacterTextSplitter, same size/overlap, prefers to cut at
                  blank lines, then line breaks, then sentences.
- section_aware:  splits at RBI structure (Chapter / lettered section / numbered paragraph such as
                  "60K."), packs whole paragraphs up to 1,500 chars without crossing a section, and
                  records "Chapter > Section" as the chunk's context. Over-long paragraphs fall back
                  to the recursive splitter.

All chunks stay well under BGE-small's 512-token limit (~4 chars per token in this corpus).
`context` is prefixed to the text at embedding time: the document title for every strategy, plus
chapter and section for section_aware.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass

SIZE = 1200
OVERLAP = 150
SECTION_MAX = 1500

_CHAPTER = re.compile(r"^(Chapter\s+[IVXLC]+[A-Z]?\b.*)$")
_SECTION = re.compile(r"^([A-Z]{1,2}\.\s+[A-Z][^.]{2,85})$")
_PARA = re.compile(r"^\d{1,3}[A-Z]{0,2}\.\s+\S")


@dataclass
class Chunk:
    chunk_id: str
    strategy: str
    doc_id: str
    title: str
    issuer: str
    status: str
    updated_date: str
    page_start: int
    page_end: int
    context: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def embed_text(self) -> str:
        return f"{self.context}\n\n{self.text}" if self.context else self.text


@dataclass
class DocText:
    meta: dict
    text: str
    starts: list[int]  # char offset where each page starts
    pages: list[int]  # page number for each entry in starts

    def page_at(self, offset: int) -> int:
        return self.pages[max(0, bisect.bisect_right(self.starts, offset) - 1)]


def assemble(pages: Iterable[dict]) -> list[DocText]:
    """Group page records by document (in input order), skipping TOC pages."""
    docs: dict[str, DocText] = {}
    for p in pages:
        if p["is_toc"] or not p["text"].strip():
            continue
        d = docs.get(p["doc_id"])
        if d is None:
            d = docs[p["doc_id"]] = DocText(meta=p, text="", starts=[], pages=[])
        elif d.text:
            d.text += "\n\n"
        d.starts.append(len(d.text))
        d.pages.append(p["page"])
        d.text += p["text"]
    return list(docs.values())


def _make(doc: DocText, strategy: str, i: int, start: int, end: int, context: str) -> Chunk:
    m = doc.meta
    return Chunk(
        chunk_id=f"{m['doc_id']}:{strategy}:{i:04d}",
        strategy=strategy,
        doc_id=m["doc_id"],
        title=m["title"],
        issuer=m["issuer"],
        status=m["status"],
        updated_date=m.get("updated_date", ""),
        page_start=doc.page_at(start),
        page_end=doc.page_at(max(start, end - 1)),
        context=context,
        text=doc.text[start:end].strip(),
    )


def _title_context(doc: DocText) -> str:
    return f"{doc.meta['title']} ({doc.meta['issuer']})"


def fixed_spans(text: str, size: int = SIZE, overlap: int = OVERLAP) -> list[tuple[int, int]]:
    spans, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut > 0 else end
        spans.append((start, end))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        nxt = text.find(" ", start, end)  # don't start mid-word
        start = nxt + 1 if nxt != -1 else start
    return spans


def recursive_spans(text: str, size: int = SIZE, overlap: int = OVERLAP) -> list[tuple[int, int]]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    return [
        (d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content))
        for d in splitter.create_documents([text])
    ]


def section_units(text: str) -> list[tuple[int, int, str, str]]:
    """Split into (start, end, chapter, section) units: one per numbered paragraph, with
    headings opening a new unit. Text before the first marker is its own unit."""
    units: list[tuple[int, int, str, str]] = []
    chapter = section = ""
    unit_start, unit_ctx = 0, ("", "")
    offset = 0
    for line in text.splitlines(keepends=True):
        s = line.strip()
        is_ch, is_sec, is_para = bool(_CHAPTER.match(s)), bool(_SECTION.match(s)), bool(_PARA.match(s))
        if (is_ch or is_sec or is_para) and offset > unit_start:
            units.append((unit_start, offset, *unit_ctx))
            unit_start = offset
        if is_ch:
            chapter, section = s, ""
        elif is_sec:
            section = s
        if is_ch or is_sec or is_para:
            unit_ctx = (chapter, section)
        offset += len(line)
    if offset > unit_start:
        units.append((unit_start, offset, *unit_ctx))
    return units


def section_spans(text: str, max_size: int = SECTION_MAX) -> list[tuple[int, int, str]]:
    """Pack whole units up to max_size within one (chapter, section); returns (start, end, ctx)."""
    out: list[tuple[int, int, str]] = []
    cur: tuple[int, int, tuple[str, str]] | None = None
    for start, end, ch, sec in section_units(text):
        ctx = (ch, sec)
        if cur and cur[2] == ctx and end - cur[0] <= max_size:
            cur = (cur[0], end, ctx)
            continue
        if cur:
            out.append((cur[0], cur[1], " > ".join(x for x in cur[2] if x)))
        cur = (start, end, ctx)
    if cur:
        out.append((cur[0], cur[1], " > ".join(x for x in cur[2] if x)))
    # Over-long units: split inside, keeping the unit's context.
    final: list[tuple[int, int, str]] = []
    for start, end, ctx in out:
        if end - start <= max_size:
            final.append((start, end, ctx))
        else:
            final.extend((start + a, start + b, ctx) for a, b in recursive_spans(text[start:end]))
    return final


def chunk_documents(docs: list[DocText], strategy: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        title = _title_context(doc)
        if strategy == "fixed":
            spans = [(a, b, title) for a, b in fixed_spans(doc.text)]
        elif strategy == "recursive":
            spans = [(a, b, title) for a, b in recursive_spans(doc.text)]
        elif strategy == "section_aware":
            spans = [(a, b, f"{title} > {c}" if c else title) for a, b, c in section_spans(doc.text)]
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
        for i, (a, b, ctx) in enumerate(spans):
            if doc.text[a:b].strip():
                chunks.append(_make(doc, strategy, i, a, b, ctx))
    return chunks


STRATEGIES = ("fixed", "recursive", "section_aware")
