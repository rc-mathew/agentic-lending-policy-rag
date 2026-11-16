# Agentic Lending Policy RAG

An assistant that answers credit and lending policy questions from RBI Master Directions and
published bank lending policies, **with a citation for every claim**. An agent decides when and
how to retrieve, rewrites vague questions, checks its own evidence before answering, and calls
tools for EMI and FOIR calculations.

The project compares three systems on the same 60-question gold set: plain RAG, a ReAct agent
written by hand in plain Python, and a LangGraph agent.

> **Status: week 1 of 3, setup.** Results table coming at the end of week 1.

## Results

| System | Faithfulness | Answer relevance | Context recall | Citation accuracy | Abstention (10 unanswerable) | Cost / question | Latency p50 |
|---|---|---|---|---|---|---|---|
| Plain RAG | – | – | – | – | – | – | – |
| Hand-written ReAct agent | – | – | – | – | – | – | – |
| LangGraph agent | – | – | – | – | – | – | – |

## Stack

Runs on an 8 GB laptop at zero cost.

| Layer | Choice |
|---|---|
| Ingestion | `pypdf` (page-accurate citations) + LangChain text splitters, 2–3 chunking strategies compared |
| Embeddings | BGE-small (`BAAI/bge-small-en-v1.5`, CPU) vs Gemini embeddings |
| Vector store | Qdrant, local mode (no server) |
| Generation | Gemini (free tier), structured answers with citations |
| Agent | Hand-written ReAct loop, then LangGraph: router, query rewriter, retrieval grader, EMI/FOIR tools |
| Evals | RAGAS + 60-question gold set |
| Observability | LangSmith |
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
src/policy_rag/   library code (config, LLM wrapper; ingestion, retrieval, agents to come)
scripts/          entry points (setup check; ingest, eval to come)
data/             manifest and corpus notes (PDFs git-ignored)
tests/            offline tests (no API calls)
```

## License

MIT
