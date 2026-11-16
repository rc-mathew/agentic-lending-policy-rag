# Data

Public documents only. PDFs go in `data/raw/` and are **not committed** (size, and the source
site stays the source of truth). What *is* committed is `sources.csv`, so anyone can rebuild
the corpus.

## Read this before downloading: RBI consolidated its rules

On **28 Nov 2025** RBI issued ~244 consolidated, entity-specific Master Directions and
repealed or withdrew ~9,400 older circulars, including most Master Circulars. A further round
in **July 2026** consolidated 628 more circulars into 64 Master Directions.

So for scheduled commercial banks:

- Download the **consolidated "Commercial Banks – …" Directions**, not the old Master
  Circulars (e.g. the old "Loans and Advances – Statutory and Other Restrictions" Master
  Circular is superseded).
- Check each document's page on rbi.org.in for an "UPDATED" date and download the latest
  version.
- If you keep one or two **repealed** documents on purpose, set `status=repealed` in the
  manifest. That gives the gold set good trap questions ("what does the current rule say?")
  and lets the retriever filter on `status`.

## The corpus (18 documents)

`sources.csv` is the list. To fetch everything:

```bash
python scripts/download_corpus.py
```

It checks each file is a real PDF, then records size, sha256 and page count in the manifest.
Anything that fails is printed with what to do.

- **11 RBI consolidated Directions for commercial banks** (28 Nov 2025, latest updates to
  Jul 2026): Credit Facilities, Responsible Business Conduct, Interest Rates on Advances,
  IRACP, Credit Risk Management, Concentration Risk Management, KYC, Resolution of Stressed
  Assets, Wilful Defaulters, Credit Information Reporting, Credit and Debit Cards.
- **7 bank policies**: Fair Practices Codes from SBI, ICICI, Axis, IDFC FIRST, Bandhan and CSB
  (public, large private, small private and small finance banks), plus a regional rural
  bank's collection and repossession policy. They restate RBI rules in their own words,
  which is what multi-hop questions need.

Left out on purpose:

- **The 2026 IRACP Directions** (expected-credit-loss based, effective later). Two versions
  of the NPA rules in one index would muddy week 1. Worth adding in week 3 as an
  "effective date" test.
- **Repealed circulars.** RBI has removed the old PDFs (e.g. the 2003 Fair Practices Code
  guidelines link now redirects to the homepage). To add a trap document, open the old
  circular's HTML page on rbi.org.in, print it to PDF, and add a row with `status=repealed`.

Size check: check the total page count the script prints. Under ~1,500 pages keeps
re-indexing quick on an 8 GB laptop; if it is far above that, drop Credit Cards and Wilful
Defaulters first.

## Manifest columns (`sources.csv`)

| Column | Example |
|---|---|
| `doc_id` | `rbi_md_cb_credit_facilities` (stable, snake_case; becomes the citation key) |
| `filename` | `rbi_md_cb_credit_facilities_2025.pdf` |
| `title` | Reserve Bank of India (Commercial Banks – Credit Facilities) Directions, 2025 |
| `issuer` | `RBI` or the bank name |
| `doc_type` | `master_direction`, `circular`, `bank_policy`, `faq` |
| `issued_date` | `2025-11-28` |
| `updated_date` | latest "updated as on" date, if any |
| `status` | `current` or `repealed` |
| `url` | direct PDF link used by the download script |
| `page_url` | RBI page for the document, to check for newer versions |
| `pages`, `bytes`, `sha256`, `downloaded_at` | filled in by `download_corpus.py` |
| `notes` | why it is in the corpus, or caveats about the link |
