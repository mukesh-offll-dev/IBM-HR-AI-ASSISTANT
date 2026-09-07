"""
LangGraph Workflow Definition for AI HR Recruitment Assistant.
Wires state machine nodes with conditional branching and error handling.
"""

from langgraph.graph import StateGraph, END
from agent.state import HRState
from agent.router import route_intent
from agent.nodes import (
    resume_parse_node,
    jd_retrieve_node,
    match_score_node,
    rank_node,
    question_gen_node,
    general_chat_node,
    memory_update_node,
    error_handler_node
)
from memory.session import session_manager


def select_route(state: HRState) -> str:
    """Conditional edge decision after routing."""
    intent = state.get("intent", "general_chat")
    if intent == "screen_resume":
        return "resume_parse_node"
    elif intent == "match_jd":
        return "jd_retrieve_node"
    elif intent == "rank_candidates":
        return "rank_node"
    elif intent == "generate_questions":
        return "question_gen_node"
    else:
        return "general_chat_node"


def check_error_resume(state: HRState) -> str:
    if state.get("error"):
        return "error_handler"
    return "memory_update"


def check_error_match(state: HRState) -> str:
    if state.get("error"):
        return "error_handler"
    return "memory_update"


def check_error_rank(state: HRState) -> str:
    if state.get("error"):
        return "error_handler"
    return "memory_update"


def check_error_questions(state: HRState) -> str:
    if state.get("error"):
        return "error_handler"
    return "memory_update"


def build_recruitment_graph():
    """Builds and compiles the recruitment agent workflow graph."""
    workflow = StateGraph(HRState)

    # Add Nodes
    workflow.add_node("router", route_intent)
    workflow.add_node("resume_parse_node", resume_parse_node)
    workflow.add_node("jd_retrieve_node", jd_retrieve_node)
    workflow.add_node("match_score_node", match_score_node)
    workflow.add_node("rank_node", rank_node)
    workflow.add_node("question_gen_node", question_gen_node)
    workflow.add_node("general_chat_node", general_chat_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("memory_update", memory_update_node)

    # Set Entry Point
    workflow.set_entry_point("router")

    # Conditional Branching from Router
    workflow.add_conditional_edges(
        "router",
        select_route,
        {
            "resume_parse_node": "resume_parse_node",
            "jd_retrieve_node": "jd_retrieve_node",
            "rank_node": "rank_node",
            "question_gen_node": "question_gen_node",
            "general_chat_node": "general_chat_node"
        }
    )

    # Resume screening branch
    workflow.add_conditional_edges(
        "resume_parse_node",
        check_error_resume,
        {
            "error_handler": "error_handler",
            "memory_update": "memory_update"
        }
    )

    # JD Matching branch (RAG retrieval -> score node)
    workflow.add_edge("jd_retrieve_node", "match_score_node")
    workflow.add_conditional_edges(
        "match_score_node",
        check_error_match,
        {
            "error_handler": "error_handler",
            "memory_update": "memory_update"
        }
    )

    # Ranking branch
    workflow.add_conditional_edges(
        "rank_node",
        check_error_rank,
        {
            "error_handler": "error_handler",
            "memory_update": "memory_update"
        }
    )

    # Question generation branch
    workflow.add_conditional_edges(
        "question_gen_node",
        check_error_questions,
        {
            "error_handler": "error_handler",
            "memory_update": "memory_update"
        }
    )

    # General chat branch
    workflow.add_edge("general_chat_node", "memory_update")

    # Error handler flows into memory update
    workflow.add_edge("error_handler", "memory_update")

    # Memory update flows to END
    workflow.add_edge("memory_update", END)

    return workflow.compile()


# Pre-compiled application graph
recruitment_graph = build_recruitment_graph()


def execute_agent_workflow(
    user_input: str = "",
    intent: str = None,
    target_resume_path: str = None,
    target_candidate_name: str = None,
    active_jd_title: str = None,
    active_jd_text: str = None
) -> HRState:
    """
    Convenience wrapper to run the compiled LangGraph workflow with current session context.
    """
    initial_state: HRState = {
        "user_input": user_input,
        "intent": intent,
        "target_resume_path": target_resume_path,
        "target_candidate_name": target_candidate_name,
        "active_jd_title": active_jd_title or session_manager.active_jd_title,
        "active_jd_text": active_jd_text or session_manager.active_jd_text,
        "parsed_candidates": dict(session_manager.parsed_candidates),
        "scored_candidates": dict(session_manager.scored_candidates),
        "ranked_candidates": list(session_manager.ranked_candidates),
        "interview_questions": dict(session_manager.interview_questions),
        "messages": list(session_manager.chat_history),
        "final_output": "",
        "error": None
    }

    result = recruitment_graph.invoke(initial_state)
    return result
