# Agentic Lending Policy RAG

An assistant that answers credit and lending policy questions from RBI Master Directions and
published bank lending policies, **with a citation for every claim**. An agent decides when and
how to retrieve, rewrites vague questions, checks its own evidence before answering, and calls
tools for EMI and FOIR calculations.

The project compares three systems on the same 60-question gold set: plain RAG, a ReAct agent
written by hand in plain Python, and a LangGraph agent.

> **Status: week 1 of 3.** Corpus ingested (18 PDFs, 829 pages) and chunking chosen. Plain-RAG
> answers and the gold-set results table come later this week.

## Results

| System | Faithfulness | Answer relevance | Context recall | Citation accuracy | Abstention (10 unanswerable) | Cost / question | Latency p50 |
|---|---|---|---|---|---|---|---|
| Plain RAG | – | – | – | – | – | – | – |
| Hand-written ReAct agent | – | – | – | – | – | – | – |
| LangGraph agent | – | – | – | – | – | – | – |

## Chunking: section-aware, chosen on a 15-question probe

Three strategies were embedded with BGE-small and scored on 15 questions, each pinned to the page
that answers it ([`data/eval/probe_questions.jsonl`](data/eval/probe_questions.jsonl)). A hit means
a top-k chunk comes from the answer document and covers the answer page.

| Strategy | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|
| Fixed 1,200 chars | 1,384 | 40% | 60% | 67% | 0.51 |
| Recursive (LangChain) | 1,691 | 40% | 60% | 73% | 0.51 |
| **Section-aware** | 1,751 | **47%** | **73%** | **93%** | **0.62** |

Section-aware chunking splits at RBI's own structure (Chapter / lettered section / numbered
paragraph such as `60K.`), never mixes two sections, and prefixes each chunk with its location
("Credit Facilities > Chapter VII - Project Finance > D. Other Provisions"). 59–66% of its RBI
chunks start at a paragraph boundary, against 0% (fixed) and 17% (recursive).

What this does and does not show:

- 15 questions is a directional check (14 vs 11 vs 10 found in the top 5), not a significant
  result. The 60-question gold set re-tests it.
- **Same-sounding banks.** "Will SBI acknowledge my loan application?" fails under every strategy:
  every bank's Fair Practices Code says much the same thing, and the embedding ignores the bank's
  name. The fix is issuer filtering in the agent's query step and hybrid BM25 search (week 3).
- **Rank 1 is weak** (47%): the answer is usually retrieved but not first. That is the
  cross-encoder re-ranker's job (week 3).
- A first run found chunks up to 670 real tokens (number-heavy tables run ~2.5 chars per token),
  which BGE silently truncates at 512. All strategies now re-split by the model's own tokenizer
  (max 500 tokens); the scores did not change.

Reproduce: `python scripts/extract_pages.py`, `python scripts/chunk_corpus.py`,
`python scripts/eval_retrieval.py`.

## Agents

Two agents wrap the same retriever, calculators and cited-answer format, so the gold set scores
all three systems the same way.

**Hand-written ReAct loop** ([`react_agent.py`](src/policy_rag/react_agent.py), plain Python):
each step the model writes a thought and picks one action (search, a calculator, or finish);
the result is appended to its history. Capped at 5 steps and 20k tokens.

**LangGraph agent** ([`graph_agent.py`](src/policy_rag/graph_agent.py)): the control flow is fixed
in the graph and the model answers only small, checkable questions.

```mermaid
graph TD;
  START([start]) --> plan;
  plan -.->|search| retrieve;
  plan -.->|clearly off-topic| generate;
  retrieve --> grade;
  grade -.->|weak evidence, retries left| retrieve;
  grade -.->|needs arithmetic| calculate;
  grade -.->|enough evidence| generate;
  calculate --> generate;
  generate --> END([end]);
```

- **plan**: routes the question; writes 1-3 searches in regulatory wording (splits two-part
  questions, expands "SBI" to "State Bank of India"); flags arithmetic.
- **grade**: keeps only relevant passages; if they do not answer the question, writes one better
  query and retrieves again (corrective RAG), at most 2 rounds.
- **calculate**: the model picks calculator calls; Python runs them (EMI, tenure, FOIR, LTV).
- **generate**: the same cited-claims answer as plain RAG, over graded evidence only.

First live runs of the ReAct agent: "max home loan on a Rs 40 lakh flat" went search, then
`max_loan_for_ltv(4000000, 80)`, giving Rs 32 lakh, cited (4 calls). "Will SBI acknowledge my loan
application?", missed by every chunking strategy, worked after the agent rewrote "SBI" to "State
Bank of India" in a second search.

## Guardrails

Every system runs behind the same wrapper ([`guardrails.py`](src/policy_rag/guardrails.py)), and
the gold-set scores are measured with it on.

- **Input:** personal identifiers (PAN, Aadhaar, card and account numbers, phone, email) are
  redacted before any API call or trace; empty, over-long and prompt-injection questions
  ("ignore previous instructions", "reveal your system prompt") are refused.
- **Output:** every claim must cite a retrieved source; uncited claims are dropped, and an answer
  with nothing cited left becomes "I can't answer that from the documents".

## Observability

Opt-in LangSmith tracing ([`tracing.py`](src/policy_rag/tracing.py)): set `LANGSMITH_TRACING=true`
in `.env`. Each question becomes one trace: guardrail, agent steps or LangGraph nodes, every
search with its passages, and every Gemini call with tokens and latency. Traces carry the
redacted question, never the raw one.

## Stack

Runs on an 8 GB laptop at zero cost.

| Layer | Choice |
|---|---|
| Ingestion | `pypdf` (pdfplumber fallback for fragmented PDFs), page numbers kept for citations; section-aware chunking |
| Embeddings | BGE-small (`BAAI/bge-small-en-v1.5`, CPU) vs Gemini embeddings |
| Vector store | Qdrant, local mode (no server) |
| Generation | Gemini (free tier), structured answers with citations |
| Agent | Hand-written ReAct loop, then LangGraph: router, query rewriter, retrieval grader, EMI/FOIR tools |
| Evals | RAGAS + 60-question gold set |
| Observability | LangSmith (opt-in), per-call tokens and latency |
| Demo | Streamlit on Hugging Face Spaces |

### Working inside the free tier

Gemini's free-tier quotas are per project, change without notice, and are shown in AI Studio.
Every LLM call goes through `src/policy_rag/llm.py`, which:

- caches each response on disk, so re-running an eval only pays for what changed;
- throttles to `LLM_MAX_RPM` to stay under the per-minute limit;
- backs off on 429/5xx and stops cleanly when the daily quota is gone.

## Setup

Python 3.11–3.13.

```bash
git clone https://github.com/rc-mathew/agentic-lending-policy-rag
cd agentic-lending-policy-rag
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate

# CPU-only PyTorch first (avoids a ~2 GB CUDA download on Windows/Linux)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

cp .env.example .env   # then add GEMINI_API_KEY
python scripts/check_setup.py
```

## Data

See [`data/README.md`](data/README.md). PDFs are not committed; `data/sources.csv` lists
every document with its source URL and status (current or repealed).

## Layout

```
src/policy_rag/   config, llm, ingest, chunking, retrieve, rag, tools, react_agent, graph_agent,
                  guardrails, tracing, evaluate
scripts/          check_setup, download_corpus, extract_pages, chunk_corpus, eval_retrieval,
                  build_index, ask, agent, graph_agent, evaluate
data/             manifest, corpus notes, eval sets and results (PDFs and processed files git-ignored)
tests/            offline tests (no API calls)
```

## License

MIT
