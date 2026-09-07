"""
Tests for LangGraph Multi-Node Recruitment Workflow.
"""

from agent.graph import execute_agent_workflow


def test_agent_graph_intent_routing():
    # Test router to general chat
    res = execute_agent_workflow(user_input="Hello, can you help me with hiring?")
    assert res.get("final_output") != ""
    assert res.get("intent") in ["general_chat", "screen_resume", "match_jd", "rank_candidates"]


def test_agent_graph_ranking_flow():
    # Test rank node execution
    res = execute_agent_workflow(intent="rank_candidates")
    assert "final_output" in res
