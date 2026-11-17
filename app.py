"""Streamlit demo: ask a lending-policy question, pick a system, see the cited answer and the trace.

    streamlit run app.py

Needs the Qdrant index (.qdrant/, built by scripts/build_index.py) and GEMINI_API_KEY in .env
(or in the Space's secrets). Example questions are answered from the cache when it is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from policy_rag import demo
from policy_rag.config import QDRANT_DIR, settings
from policy_rag.llm import DailyQuotaExhausted

st.set_page_config(page_title="Lending Policy Assistant", page_icon="📑", layout="wide")


@st.cache_resource(show_spinner="Loading the search index and models (first run only)...")
def get_retriever():
    from policy_rag.retrieve import Retriever

    return Retriever(mode="rerank")


@st.cache_resource
def get_budget() -> demo.DailyBudget:
    return demo.DailyBudget()


# ---------- sidebar ----------
with st.sidebar:
    st.header("System")
    label = st.radio("Answer with", list(demo.SYSTEMS), index=2, label_visibility="collapsed")
    system = demo.SYSTEMS[label]
    st.caption(demo.SYSTEM_NOTES[system])
    show_trace = st.toggle("Show the agent's steps", value=True, disabled=system == "plain")

    st.divider()
    st.subheader("What it knows")
    st.markdown(
        "18 public documents, 829 pages: 11 RBI Directions for commercial banks (2025) and "
        "the fair-practices or collection policies of SBI, ICICI, Axis, IDFC FIRST, City Union "
        "Bank, DBS India and Kerala Grameena Bank."
    )
    st.caption(
        "Search: BGE-small embeddings + BM25, merged with RRF, then a cross-encoder re-ranker. "
        f"Answers: {settings.gen_model}. Every claim must cite a source; the assistant says so "
        "when the documents do not answer."
    )
    st.divider()
    budget = get_budget()
    asked = st.session_state.get("asked", 0)
    st.caption(
        f"Free-tier demo: {max(0, demo.SESSION_LIMIT - asked)} questions left in this session, "
        f"{budget.left()} model calls left today. Examples are served from cache."
    )
    st.caption("Not legal or financial advice. Check the cited page before relying on an answer.")

# ---------- main ----------
st.title("Lending Policy Assistant")
st.write(
    "Questions about Indian bank lending rules, answered only from RBI Directions and banks' "
    "own policies, with page-level citations."
)

if not QDRANT_DIR.exists():
    st.error("The search index is missing. Run `python scripts/build_index.py` first.")
    st.stop()
if not settings.gemini_api_key:
    st.error("GEMINI_API_KEY is not set (add it to .env, or to the Space's secrets).")
    st.stop()

st.write("Try one:")
cols = st.columns(len(demo.EXAMPLES))
for col, ex in zip(cols, demo.EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state["question"] = ex

with st.form("ask"):
    question = st.text_area(
        "Your question", key="question", height=80, max_chars=1000,
        placeholder="e.g. Can a bank charge interest on penal charges?",
    )
    go = st.form_submit_button("Ask", type="primary")

if go and question.strip():
    if asked >= demo.SESSION_LIMIT:
        st.warning("This session has used its free questions. Reload the page later to ask more.")
        st.stop()
    if not budget.can_spend(demo.MAX_CALLS[system]):
        st.warning("Today's free model quota for this demo is used up. It resets at 13:30 IST.")
        st.stop()
    try:
        with st.spinner(f"{label} is working..."):
            result = demo.run(system, question, get_retriever())
    except DailyQuotaExhausted:
        st.warning("Today's free Gemini quota is used up. It resets at 13:30 IST.")
        st.stop()
    except Exception as e:  # noqa: BLE001  visitors get a short message, the log gets the trace
        print(f"[app] {type(e).__name__}: {e}", file=sys.stderr)
        st.error(f"Something went wrong: {type(e).__name__}. Please try again.")
        st.stop()
    spent = demo.calls_spent(result)
    budget.spend(spent)
    st.session_state["asked"] = asked + (1 if spent else 0)
    st.session_state["last"] = (label, demo.answer_view(result))

if "last" in st.session_state:
    shown_label, v = st.session_state["last"]
    st.subheader("Answer")
    if v["blocked"]:
        st.warning(f"Not answered: {v['guardrail'].removeprefix('blocked: ')}.")
    elif not v["answerable"]:
        st.info("The documents I have don't answer this, so I won't guess.")
    else:
        for c in v["claims"]:
            st.markdown(f"- {c['text']} **{c['cites']}**")
    if v["guardrail"] and not v["blocked"]:
        st.caption(f"Guardrails: {v['guardrail']}")

    s = v["stats"]
    meta = [shown_label, f"{s['calls']} model call(s)",
            f"{s['tokens_in']:,} in / {s['tokens_out']:,} out tokens", f"{s['seconds']}s"]
    if s["stopped_by"]:
        meta.append(f"stopped: {s['stopped_by']}")
    if s["cached"]:
        meta.append("from cache")
    st.caption(" · ".join(meta))

    if v["sources"]:
        st.subheader("Sources")
        for src in v["sources"]:
            tag = " · REPEALED" if src["repealed"] else ""
            with st.expander(f"[{src['n']}] {src['label']}{tag}"):
                st.caption(src["section"])
                st.text(src["text"])
    if v["other_hits"]:
        with st.expander("Also retrieved, not cited"):
            for h in v["other_hits"]:
                st.markdown(f"- {h}")

    if show_trace and v["steps"]:
        st.subheader("How the agent got there")
        for i, step in enumerate(v["steps"], 1):
            with st.expander(f"{i}. {step['action']}: {step['thought'][:90]}"):
                if step["detail"]:
                    st.markdown(f"**Input:** {step['detail']}")
                if step["observation"]:
                    st.text(step["observation"][:3000])
