from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from core.llm import invoke_with_retry

Goal = Literal["internship", "placement", "project", "general"]


class CareerState(TypedDict):
    profile: str
    goal: Goal
    kb_context: str
    sources: str
    plan_hours: str
    tool_trace: str
    final_output: str
    retriever_ref: Any
    api_key: str


def classify_goal(state: CareerState) -> dict:
    text = f"{state['profile']}".lower()
    if any(word in text for word in ["intern", "internship", "summer"]):
        goal: Goal = "internship"
    elif any(word in text for word in ["placement", "package", "dsa", "oa"]):
        goal = "placement"
    elif any(word in text for word in ["project", "github", "portfolio"]):
        goal = "project"
    else:
        goal = "general"
    return {"goal": goal, "tool_trace": f"router -> {goal}"}


def retrieve_kb(state: CareerState) -> dict:
    retriever = state["retriever_ref"]
    query = f"{state['goal']} plan for: {state['profile']}"
    docs = retriever.invoke(query)
    parts = []
    sources = []
    for i, doc in enumerate(docs, start=1):
        name = Path(str(doc.metadata.get("source", "kb"))).name
        parts.append(f"[KB {i} | {name}]\n{doc.page_content}")
        sources.append(name)
    return {
        "kb_context": "\n\n".join(parts),
        "sources": ", ".join(dict.fromkeys(sources)),
        "tool_trace": state.get("tool_trace", "") + " -> rag:career_kb",
    }


@tool
def estimate_prep_hours(goal: str, weeks: int = 6) -> str:
    """Estimate weekly effort for internship, placement, or project prep."""
    weekly = {"internship": 10, "placement": 14, "project": 8, "general": 9}.get(goal, 9)
    total = weekly * max(weeks, 1)
    return (
        f"Goal={goal}, weeks={weeks}, about {weekly} focused hours/week, "
        f"total ~{total} hours. Split: 50% practice, 30% project, 20% applications."
    )


def call_planning_tool(state: CareerState) -> dict:
    hours = estimate_prep_hours.invoke({"goal": state["goal"], "weeks": 6})
    return {"plan_hours": hours, "tool_trace": state.get("tool_trace", "") + " -> tool:estimate_prep_hours"}


def compose_plan(state: CareerState) -> dict:
    prompt = f"""You are CampusPath, a campus career advisor for engineering students in India.
Use the knowledge-base context. Be specific and practical. Do not invent company names that are not implied.

Student profile / request:
{state['profile']}

Detected goal: {state['goal']}
Effort tool output: {state['plan_hours']}

Knowledge base:
{state['kb_context']}

Write:
1. 1-line diagnosis
2. 6-week week-by-week plan
3. 5 resume / GitHub actions
4. 3 risks and how to avoid them
End with Sources: {state['sources']}
"""
    output = invoke_with_retry(state["api_key"], [HumanMessage(content=prompt)], temperature=0.4)
    return {"final_output": output, "tool_trace": state.get("tool_trace", "") + " -> compose_plan"}


def build_career_graph():
    workflow = StateGraph(CareerState)
    workflow.add_node("classify", classify_goal)
    workflow.add_node("retrieve", retrieve_kb)
    workflow.add_node("plan_tool", call_planning_tool)
    workflow.add_node("compose", compose_plan)
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "plan_tool")
    workflow.add_edge("plan_tool", "compose")
    workflow.add_edge("compose", END)
    return workflow.compile()
