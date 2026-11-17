# AGENTS.md

Rules for AI coding assistants (Claude Code, Copilot, Cursor) working in this repo. Humans: the
README is the place to start.

## What this is

A lending-policy question-answering system over 18 public PDFs (11 RBI Directions, 7 bank
policies). Three systems answer the same questions: plain RAG (`rag.py`), a hand-written ReAct
agent (`react_agent.py`) and a LangGraph agent (`graph_agent.py`). They are compared on a
60-question gold set that was checked by hand against the PDFs.

## Commands

```bash
pip install -e ".[agent,app,dev]"
pytest -q                     # offline: no API calls, no index needed. Must pass before a commit.
ruff check .                  # must pass before a commit
streamlit run app.py          # demo (needs .qdrant/ and GEMINI_API_KEY)
```

## Hard rules

1. **Never run anything that calls Gemini or Groq without asking the person first.** That means
   `scripts/evaluate.py`, `scripts/ask.py`, `scripts/agent.py`, `scripts/graph_agent.py` and
   the app. The free tiers are small: Gemini 3.5 Flash-Lite allows 500 requests a day (resets
   13:30 IST) and 15 a minute; Groq's judge allows about 200K tokens a day (resets 05:30 IST).
   One full scoring run of an agent costs about 200 Gemini calls.
2. **Never tune on the gold set.** `data/eval/gold_candidates.jsonl` is the test set. Do not
   change prompts, tools or retrieval to fix a specific gold question and then report the gold
   score as if it were unseen. Fixes are judged on fresh questions (for example
   `data/eval/calc_holdout.jsonl`) and reported separately.
3. **Never edit verified evaluation data.** Items with `status: verified` were checked by a
   person against the PDFs. Propose changes; do not make them.
4. **Every answer claim must cite a source.** Keep the citation checks, the abstention path and
   the guardrails (`guardrails.py`) on every system, including in evaluation.
5. **Secrets stay in `.env`.** Never print, log or commit API keys or tokens.
6. **Keep it free and small.** No paid services. The target machine has 8 GB of RAM: no model
   larger than the current BGE-small embedder and MiniLM re-ranker.

## Conventions

- Every LLM call goes through `llm.generate()` (cache, rate limiter, quota errors). A change to
  a prompt or schema changes the cache key, so the next run pays for those calls again.
- A bug found by testing gets a regression test that names the case (see `tests/`).
- Commit messages say what changed and why, with the evidence (numbers, question ids).
- Changes made after the scored runs are labelled as such in the README.

## Where things are

| Path | What |
|---|---|
| `src/policy_rag/` | ingestion, chunking, retrieval, the three systems, tools, guardrails, scoring |
| `scripts/` | command-line entry points |
| `data/eval/` | probe set, gold set, held-out calculation set, results |
| `docs/DECISIONS.md` | why each design choice was made, with the evidence |
| `app.py` | Streamlit demo |
