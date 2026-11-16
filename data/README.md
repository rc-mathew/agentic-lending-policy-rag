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

## Target corpus (15–20 PDFs, Tue 17 Nov)

Titles below are what to look for; confirm the exact name on rbi.org.in → Regulations →
Commercial Banking → Master Directions, since I could not open the site from my workspace.

**RBI: consolidated Master Directions for commercial banks (aim for 8–10)**

| Topic | Why it's in the corpus |
|---|---|
| Credit Facilities | Core lending rules: loan restrictions, limits, sanction |
| Responsible Business Conduct (fair practices, grievance) | Fair Practices Code, penal charges, release of property documents |
| Interest Rate on Advances | Floating-rate resets, EMI vs tenure choice, benchmark linkage |
| Asset Classification, Provisioning and Income Recognition | NPA rules: SMA/NPA days-past-due, provisioning |
| Know Your Customer | Onboarding and due diligence questions |
| Digital Lending | LSPs, Key Fact Statement, cooling-off, default loss guarantee |
| Credit Risk Management / Concentration or exposure norms | Exposure limits, multi-hop questions |
| Capital Adequacy (credit risk weights) | Risk weights on consumer credit |
| Co-lending (if consolidated separately) | Multi-party lending questions |
| Gold-loan / lending against gold collateral | LTV caps; good for calculation questions |

**Bank-published policies (aim for 5–8)**: look on each bank's site for "Fair Practices Code",
"Loan Policy", "Interest Rate Policy", "Grievance Redressal Policy", "Recovery / Collection
Policy". Mix public- and private-sector banks (e.g. SBI, Bank of Baroda, HDFC Bank, ICICI Bank,
Axis Bank). Bank documents restate RBI rules in their own words, which is what makes
multi-hop questions possible.

**Optional (1–2)**: a CIBIL/credit-bureau consumer FAQ or an NHB/housing-finance FAQ, to test
how the system handles a non-RBI source.

Keep the total under ~1,500 pages. Embedding is fine at that size on CPU; past that, ingestion
iterations get slow on an 8 GB laptop.

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
| `url` | page you downloaded it from |
| `pages` | filled in automatically by the ingestion script |
