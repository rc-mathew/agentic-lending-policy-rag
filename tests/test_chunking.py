"""Offline tests for the chunking strategies on a small synthetic RBI-style document."""

import itertools

import pytest

from policy_rag.chunking import (
    SECTION_MAX,
    assemble,
    chunk_documents,
    fixed_spans,
    section_units,
)

PAGE1 = """Chapter III - Digital Lending Guidelines
A. General Requirements
6. A bank shall ensure that the LSP discloses the Key Fact Statement before the loan.
7. A bank shall provide a cooling-off period of at least three days."""
PAGE2 = """B. Conduct Requirements
8. Recovery agents shall not call the borrower before 08:00 or after 19:00.
Chapter IV - Housing Finance
9. The LTV ratio for loans up to Rs 30 lakh shall not exceed 90 percent."""


def _pages():
    meta = {"doc_id": "d", "title": "T", "issuer": "RBI", "doc_type": "md", "status": "current",
            "updated_date": "", "is_toc": False, "extractor": "pypdf", "lines_removed": 0}
    toc = {**meta, "page": 1, "text": "Contents ..... 2", "is_toc": True}
    return [toc, {**meta, "page": 2, "text": PAGE1}, {**meta, "page": 3, "text": PAGE2}]


def test_assemble_skips_toc_and_maps_pages():
    (doc,) = assemble(_pages())
    assert doc.pages == [2, 3]
    assert doc.page_at(0) == 2
    assert doc.page_at(len(doc.text) - 1) == 3


def test_section_units_track_chapter_and_section():
    (doc,) = assemble(_pages())
    units = section_units(doc.text)
    ctx = {doc.text[a:b].strip()[:2]: (ch[:11], sec[:2]) for a, b, ch, sec in units}
    assert ctx["6."] == ("Chapter III", "A.")
    assert ctx["8."] == ("Chapter III", "B.")
    assert ctx["9."] == ("Chapter IV ", "")


def test_section_aware_never_mixes_sections_and_keeps_pages():
    (doc,) = assemble(_pages())
    chunks = chunk_documents([doc], "section_aware")
    def having(words):
        return next(c for c in chunks if words in c.text)

    assert "Recovery agents" not in having("cooling-off").text  # sections A and B kept apart
    assert having("cooling-off").context.endswith("A. General Requirements")
    assert having("LTV ratio").page_start == 3
    assert having("LTV ratio").context.endswith("Chapter IV - Housing Finance")
    assert all(len(c.text) <= SECTION_MAX for c in chunks)


def test_fixed_spans_cover_text_with_overlap():
    text = " ".join(f"word{i}" for i in range(600))
    spans = fixed_spans(text, size=300, overlap=50)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(b2 > a2 and a2 < b1 for (a1, b1), (a2, b2) in itertools.pairwise(spans))


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "section_aware"])
def test_every_strategy_keeps_all_words(strategy):
    (doc,) = assemble(_pages())
    joined = " ".join(c.text for c in chunk_documents([doc], strategy))
    for word in ("cooling-off", "08:00", "90 percent"):
        assert word in joined
