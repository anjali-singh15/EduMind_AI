from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from core.llm import invoke_with_retry

Intent = Literal["explain", "quiz", "summary", "flashcards"]


class StudyState(TypedDict):
    question: str
    intent: Intent
    pdf_context: str
    sources: str
    tool_trace: str
    final_output: str
    retriever_ref: Any
    api_key: str


def classify_intent(state: StudyState) -> dict:
    q = state["question"].lower()
    if any(word in q for word in ["quiz", "mcq", "test me", "questions"]):
        intent: Intent = "quiz"
    elif any(word in q for word in ["summary", "summarize", "overview", "revise unit"]):
        intent = "summary"
    elif any(word in q for word in ["flashcard", "flash card", "revise fast"]):
        intent = "flashcards"
    else:
        intent = "explain"
    return {"intent": intent, "tool_trace": f"router -> {intent}"}


def _search_notes(retriever, query: str) -> tuple[str, str]:
    docs = retriever.invoke(query)
    context_parts = []
    source_parts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Chunk {i} | page {page}]\n{doc.page_content}")
        source_parts.append(f"Chunk {i} (page {page})")
    return "\n\n".join(context_parts), ", ".join(source_parts)


def retrieve_with_tool(state: StudyState) -> dict:
    retriever = state["retriever_ref"]

    @tool
    def search_study_notes(query: str) -> str:
        """Search the uploaded PDF notes and return the most relevant chunks."""
        context, _ = _search_notes(retriever, query)
        return context or "No relevant notes found."

    query = state["question"]
    if state["intent"] in {"quiz", "summary", "flashcards"}:
        query = f"main topics, definitions, important points: {state['question']}"

    context, sources = _search_notes(retriever, query)
    tool_result = search_study_notes.invoke(query)
    return {
        "pdf_context": tool_result or context,
        "sources": sources,
        "tool_trace": state.get("tool_trace", "") + " -> tool:search_study_notes",
    }


def generate_output(state: StudyState) -> dict:
    intent = state["intent"]
    format_rules = {
        "explain": "Explain clearly for a college student. Use short sections and a tiny example if useful. If notes are insufficient, say what is missing.",
        "quiz": "Create 5 MCQs from the notes only. For each: question, options A-D, correct answer, 1-line explanation.",
        "summary": "Write a revision summary: key terms, 5 bullets, and 3 probable exam questions.",
        "flashcards": "Create 8 flashcards as Q / A pairs from the notes only.",
    }
    prompt = f"""You are EduMind, a grounded study coach.
Use ONLY the retrieved notes. Do not invent syllabus facts.

Intent: {intent}
Format: {format_rules[intent]}

Retrieved notes:
{state['pdf_context']}

Student request:
{state['question']}

End with a line: Sources: {state['sources']}
"""
    output = invoke_with_retry(state["api_key"], [HumanMessage(content=prompt)])
    return {"final_output": output, "tool_trace": state.get("tool_trace", "") + " -> generate"}


def build_study_graph():
    workflow = StateGraph(StudyState)
    workflow.add_node("classify", classify_intent)
    workflow.add_node("retrieve", retrieve_with_tool)
    workflow.add_node("generate", generate_output)
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()
