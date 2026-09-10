import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="EduMind Agents", page_icon="🎓", layout="wide")

st.title("EduMind Agents")
st.caption("MLRIT AI Agent evaluation — two different LangGraph agents in one submission")

left, right = st.columns(2)
with left:
    st.subheader("Agent 1 · EduMind Study Coach")
    st.markdown(
        """
**Problem:** Students upload notes but get generic chatbot answers that ignore the PDF.

**Workflow:** `classify intent → tool: search_study_notes (RAG) → generate`

**Different because:** user PDF embeddings, quiz/summary/flashcards router, grounded citations.
"""
    )
    try:
        st.page_link("pages/1_EduMind_Study_Coach.py", label="Open Study Coach", icon="📚")
    except AttributeError:
        st.info("👈 Open Study Coach from the sidebar")

with right:
    st.subheader("Agent 2 · CampusPath Advisor")
    st.markdown(
        """
**Problem:** Students know they need internships/placements but not a sequenced plan.

**Workflow:** `classify goal → RAG over career KB → tool: estimate_prep_hours → compose plan`

**Different because:** built-in knowledge base (not user PDF), planning tool, week-by-week career output.
"""
    )
    try:
        st.page_link("pages/2_CampusPath_Advisor.py", label="Open CampusPath", icon="🎯")
    except AttributeError:
        st.info("👈 Open CampusPath Advisor from the sidebar")

st.info("Add your Gemini API key in the sidebar of each agent page, or in Streamlit Secrets as `GOOGLE_API_KEY`.")
st.write("Repo:", "https://github.com/anjali-singh15/EduMind_AI")
st.write("Knowledge files:", ", ".join(p.name for p in Path("knowledge").glob("*.md")) or "missing")
st.write("Secrets available:", bool(st.secrets.get("GOOGLE_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")))
