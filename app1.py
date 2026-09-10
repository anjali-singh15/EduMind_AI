import os
from pathlib import Path

import streamlit as st

from agents.study_agent import build_study_graph
from core.rag import pdf_retriever

st.set_page_config(page_title="EduMind Study Coach", page_icon="📚")
st.title("📚 EduMind Study Coach")
st.caption("Agent 1 — PDF RAG + LangGraph router + notes search tool")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Google Gemini API Key", type="password")
    active_key = api_key or st.secrets.get("GOOGLE_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    st.markdown("Upload a unit PDF, then ask for an explanation, quiz, summary, or flashcards.")

if not active_key:
    st.info("Enter your Gemini API key in the sidebar to run the agent.")
    st.stop()

uploaded = st.file_uploader("Upload study PDF", type=["pdf"])
if not uploaded:
    st.stop()


@st.cache_resource
def index_pdf(file_bytes: bytes, filename: str):
    tmp = Path("temp.pdf")
    tmp.write_bytes(file_bytes)
    return pdf_retriever(str(tmp))


with st.spinner("Embedding PDF with MiniLM + FAISS..."):
    retriever = index_pdf(uploaded.getvalue(), uploaded.name)

st.success("Notes indexed. Ask from the PDF only.")
question = st.text_area(
    "Your request",
    placeholder="Example: Make a 5-question quiz on this unit",
    height=90,
)

if st.button("Run Agent 1", type="primary") and question.strip():
    graph = build_study_graph()
    with st.spinner("LangGraph: classify → retrieve tool → generate"):
        try:
            result = graph.invoke(
                {
                    "question": question.strip(),
                    "retriever_ref": retriever,
                    "api_key": active_key,
                }
            )
            st.caption(f"Workflow: `{result.get('tool_trace', '')}`")
            st.markdown("### Output")
            st.markdown(result["final_output"])
        except Exception as exc:
            st.error(f"Execution error: {exc}")
            st.caption("If the model name fails, try another Gemini key that can call gemini-2.0-flash or gemini-1.5-flash.")
