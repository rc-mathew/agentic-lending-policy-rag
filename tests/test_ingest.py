"""Offline tests for page cleaning. Synthetic pages, no PDFs needed."""

from policy_rag.ingest import clean_page, find_running_lines, fragment_share, is_toc_page

WORDS = ["loan", "interest", "borrower", "tenor", "charges", "collateral", "repayment"]


def _pages(n: int) -> list[list[str]]:
    """Realistic page: page number, header, 10 distinct body lines, footer."""
    return [
        [" ", f"{i} ", " ", "Fair Practices Code "]
        + [f"{WORDS[(i + j) % 7]} clause {'abcdefghij'[j]} body text." for j in range(10)]
        + ["Fair Practices Code - Ver1.0 - Apr - 2026 "]
        for i in range(1, n + 1)
    ]


def test_running_header_footer_and_page_number_removed():
    pages = _pages(5)
    running = find_running_lines(pages)
    text, removed = clean_page(pages[2], running)
    lines = text.splitlines()
    assert len(lines) == 10 and lines[0] == "tenor clause a body text."
    assert removed == 3  # page number, header, footer


def test_split_page_counter_removed():
    lines = ["Page ", "2 ", "| ", "6 ", "Real content here.", "More content."]
    text, removed = clean_page(lines, set())
    assert text == "Real content here.\nMore content."
    assert removed == 4


def test_number_in_body_is_kept():
    lines = ["Header", "a", "b", "c", "d", "42", "e", "f", "g", "h", "Footer"]
    text, _ = clean_page(lines, set())
    assert "\n42\n" in text  # only edge-zone numbers are page furniture


def test_hyphen_rejoined_only_between_lowercase():
    text, _ = clean_page(["The borrow-", "er shall pay.", "Kumbakonam-", "612001 x", "y", "z", "w", "v"], set())
    assert "borrower shall pay." in text
    assert "Kumbakonam-\n612001" in text


def test_toc_detection():
    toc = "\n".join(f"Chapter {i} - Something ........................ {i * 3}" for i in range(5))
    assert is_toc_page(toc)
    assert not is_toc_page("60K. A bank may fix a repayment tenor.\nE. Disclosures")


def test_short_docs_have_no_running_lines():
    assert find_running_lines(_pages(2)) == set()


def test_fragment_share_flags_broken_extraction():
    clean = [["A whole line of text.", "Another whole line."]]
    broken = [["processing fees", " 1", "non", " acceptance", "pre", " -", " payment"]]
    assert fragment_share(clean) == 0.0
    assert fragment_share(broken) > 0.05


def test_amendment_footnote_number_dropped():
    lines = ["a", "b", "c", "d", "1[(ia) 'Acquisition Finance' shall mean", "e", "f", "g", "h"]
    text, _ = clean_page(lines, set())
    assert "[(ia) 'Acquisition Finance'" in text and "1[" not in text
