# Design decisions

Each entry records what was chosen, what it was compared with, the evidence, and the cost.
Numbers come from files in `data/eval/`; nothing here is from intuition alone. Entries are in
the order the decisions were made.

## 1. Corpus: consolidated RBI Directions plus seven banks' own policies

- **Chose:** 11 RBI "Commercial Banks" Directions (2025 consolidations) and 7 banks' fair
  practices or collection policies. 18 PDFs, 829 pages.
- **Why:** questions in lending compliance mix the regulator's rule with what one bank promises
  (for example, "is this bank's 07:00 calling hour allowed?"). Both kinds of document are needed
  to ask questions that span documents.
- **Cost:** bank policies say similar things in similar words, which makes "which bank?"
  retrieval hard (see 4). RBI blocks scripted downloads, so the PDFs were fetched by hand.

## 2. PDF text: pypdf, with pdfplumber when the text comes out in fragments

- **Chose:** pypdf by default; switch a document to pdfplumber when more than 5% of its lines
  are fragments. Strip running headers and footers, page numbers and contents pages.
- **Why:** pypdf is fast and good on most pages; a few documents came out as broken one-word
  lines, which pdfplumber reads correctly. The 5% threshold separates the two groups.
- **Cost:** two extractors to maintain.

## 3. Chunking: section-aware, capped at 500 real tokens

- **Compared:** fixed 1,200 characters, LangChain recursive, and splitting at RBI's own
  structure (chapter, lettered section, numbered paragraph) with a location prefix.
- **Evidence (15-question probe):** right page in the top 5: fixed 67%, recursive 73%,
  section-aware **93%**; MRR@5 0.51, 0.51, **0.62**.
- **Also found:** chunks of up to 670 real tokens under a characters/4 estimate; BGE silently
  cuts at 512. All strategies now split by the embedding model's own tokenizer.
- **Cost:** a parser tied to RBI's layout. 15 questions is directional, not significant.

## 4. Search: hybrid (BM25 + dense, merged with RRF) then a cross-encoder re-ranker

- **Compared:** dense only; hybrid; hybrid plus re-ranking the top 20.
- **Evidence (same probe):** right page first: 47%, 53%, **67%**; in the top 5: 93%, 93%,
  **100%**; MRR@5 0.62, 0.69, **0.80**.
- **Why RRF with k = 60:** it merges by rank, so BM25 and cosine scores, which are on different
  scales, never need calibrating; 60 is the value from the original RRF paper and was not tuned.
- **Cost:** about 1.3 s per search on a laptop CPU (41 ms for dense alone). Agents search
  several times, so this is most of their latency.

## 5. Answers: structured output, a citation on every claim, and an explicit abstain path

- **Chose:** the model returns JSON (Pydantic schema): `answerable` plus a list of claims, each
  with the source numbers that support it. Citations outside the source list are dropped and
  counted; claims left without a source are removed by the guardrail.
- **Why:** a free-text answer cannot be checked claim by claim, and the use case needs "the
  documents don't say" to be a normal answer.
- **Evidence:** all three systems abstained correctly on 10 of 10 unanswerable questions;
  citation accuracy 92-95%.

## 6. Two agents: one written by hand, one in LangGraph

- **Chose:** a ReAct loop in plain Python (about 200 lines, step and token caps) and a LangGraph
  state machine (plan, retrieve, grade, re-search, calculate, generate).
- **Why:** the hand-written loop shows the mechanism without a framework; the graph shows the
  same job with explicit state, a grader and routing. Comparing them on the same questions
  answers "what does the framework buy?".
- **Evidence (60-question gold set):** plain RAG 82% correct, ReAct 90%, LangGraph 91%.
  Multi-hop 65%, 90%, 95%. False refusals 6%, 0%, 0%. Tokens per question 2,081, 4,528, 3,284.
  Median latency 3.6 s, 10.1 s, 13.2 s.
- **Reading:** the agents' gain comes from searching again when the first search is not enough.
  The two agents are equal within noise (one question); LangGraph spends 27% fewer tokens
  because the grader passes on only the passages it keeps, and is slower because it makes more,
  smaller sequential calls.

## 7. Agents read whole chunks

- **Was:** agents saw the first 600-700 characters of each passage.
- **Evidence:** the Kerala Grameena Bank question failed because RBI's "before 8:00 a.m." sits
  at character 1,109 of its chunk and the bank's "0700 hrs" at character 951, so the grader
  rejected the very passages that answered it. Regression test:
  `test_grader_sees_text_late_in_a_long_chunk`.
- **Cost:** more input tokens per step (cap raised from 20k to 30k per question).

## 8. Working before verdict

- **Was:** the answer had to open with Yes/No/Partly. Flash-Lite then said "Yes, 07:00 is
  allowed" against an 08:00 limit.
- **Chose:** a hidden `reasoning` field placed first in the schema (generated first), where the
  model writes each limit and value and compares them. The same question then answered
  "Partly", which is right.
- **Also:** the verdict follows the question's polarity ("Can a bank charge X?" gets "No, it
  cannot", not "Yes, it cannot"), found in the demo.

## 9. A judge from a different model family, on a different quota

- **Chose:** answers by Gemini 3.5 Flash-Lite; grading by `openai/gpt-oss-120b` on Groq; one
  judge call per answered question plus deterministic checks (retrieval hit, page cited,
  abstention), instead of the full RAGAS suite.
- **Why:** a model grading its own answers is biased toward them, and RAGAS would cost several
  calls per question on a 500-a-day quota.
- **Cost:** one judge, checked by hand on a few verdicts only. Groq's token limit spreads a
  full run's judging over more than one day.

## 10. A gold set checked by a person, and no tuning on it

- **Chose:** 60 questions (30 direct, 10 multi-hop, 10 calculation, 10 unanswerable), drafted
  by the assistant, then every answer and page checked against the PDF by the project owner.
- **Rule:** a failure found on the gold set is fixed and then judged on new questions, not by
  re-scoring the gold set. With 10 questions per type, one question is 10 points.

## 11. Calculator results as citable sources (v2, after the scored runs)

- **Evidence from the traces:** on g50 the LangGraph agent computed ₹17,995, ₹18,643 and 268.8
  months with its tools, then left every figure out. The answer rule allows only claims that
  cite a numbered source, and calculator results were unnumbered notes, so the model obeyed the
  rule and dropped them. Same cause for g47 and g49.
- **Chose:** results become numbered "Calculator" sources; two tools added (`max_loan_by_slabs`
  for LTV slabs that depend on the loan itself, `arithmetic` parsed with `ast`, never `eval`);
  the agents' action schemas now come from the tool registry (they had listed only the
  original four tools).
- **Judged on:** 10 new calculation questions (`data/eval/calc_holdout.jsonl`), before
  (`--calc-mode v1`) and after. Citation accuracy 70% to 90% (LangGraph) and 72% to 100%
  (ReAct); correctness 75% to 75% and 65% to 80%. Full table in the README.
- **Cost of the change:** the new slab tool read a misnamed key (`"limit"`, `"max"`) as "no
  upper bound" and returned the highest-LTV slab, breaking 3 answers (f01 in both agents, f02 in
  LangGraph). The agents never saw the key names: the tool list in their prompt shows only
  parameter names and the description. Fix: the keys are in the description, and any other key
  is an error, not a default. A tool that guesses at bad input fails silently and fluently.
  Re-run after the fix (a bug check, since the bug was found on these questions): f01 and f02
  right in both agents, nothing else changed.
- **Still open:** f08 (EMI reset). The agent computed the figures and still answered with the
  list of options only, the g50 pattern. Not fixed on the holdout; needs new questions to judge.

## 12. Live-data questions are routed away before searching (after the scored runs)

- **Evidence:** "What is RBI's repo rate today?" cost the LangGraph agent 4 calls and 24 s
  before it abstained correctly. Held back until after the scored runs because the gold set
  has questions of exactly this kind.

## 13. Computed figures must appear in "options" answers (after the holdout)

- **Evidence:** holdout f08 ("My ₹35 lakh, 15-year loan resets from 8.5% to 9%. What are my
  EMI and tenure options?"). The LangGraph agent computed both EMIs and the new tenure, then
  answered with RBI's list of options only; ReAct never called a calculator. The answer rule
  asked for computed figures only when the question "asks for an amount, a period or a ratio",
  and an "options" question does not read as one.
- **Chose:** the answer rule now says every figure a Calculator source gives goes in a claim,
  and an options question gets each option with its figures. ReAct and the LangGraph planner
  now call a calculator when the question gives numbers to work with, "options" included.
- **Judged on:** 5 new questions (`data/eval/calc_holdout2.jsonl`: rate-reset options and
  two-figure LTV questions), written before the fix. Not on f08, where the failure was found.
