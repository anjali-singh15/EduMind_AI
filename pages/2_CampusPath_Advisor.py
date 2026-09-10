import os

import streamlit as st

from agents.career_agent import build_career_graph
from core.rag import career_kb_retriever

st.set_page_config(page_title="CampusPath Advisor", page_icon="🎯")
st.title("🎯 CampusPath Advisor")
st.caption("Agent 2 — career knowledge RAG + hours tool + planner graph")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Google Gemini API Key", type="password")
    active_key = api_key or st.secrets.get("GOOGLE_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    st.markdown("This agent does **not** use your lecture PDF. It retrieves from the built-in campus career knowledge base.")

if not active_key:
    st.info("Enter your Gemini API key in the sidebar to run the agent.")
    st.stop()

year = st.selectbox("Year", ["2nd year", "3rd year", "4th year / final"])
branch = st.selectbox("Branch", ["CSE", "AIML", "IT", "DS", "ECE"])
goal = st.selectbox("Primary goal", ["AI/ML internship", "Campus placement", "Build a GitHub project"])
extra = st.text_area(
    "Tell the agent about you",
    placeholder="Example: Weak in DSA, finished one ML notebook, want a summer internship",
    height=90,
)

profile = f"Year: {year}. Branch: {branch}. Goal: {goal}. Background: {extra}"

@st.cache_resource
def index_career_kb():
    return career_kb_retriever()


if st.button("Run Agent 2", type="primary"):
    with st.spinner("Indexing career knowledge base..."):
        retriever = index_career_kb()
    graph = build_career_graph()
    with st.spinner("LangGraph: classify → RAG → estimate hours → plan"):
        try:
            result = graph.invoke(
                {
                    "profile": profile,
                    "retriever_ref": retriever,
                    "api_key": active_key,
                }
            )
            st.caption(f"Workflow: `{result.get('tool_trace', '')}`")
            st.markdown("### Plan")
            st.markdown(result["final_output"])
        except Exception as exc:
            st.error(f"Execution error: {exc}")
