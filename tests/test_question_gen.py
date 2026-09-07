"""
Tests for Interview Question Generator Tool.
"""

from tools.question_gen import generate_interview_questions


def test_generate_interview_questions_structure():
    result = generate_interview_questions(
        candidate_name="Priya Sharma",
        role="Senior Backend AI Engineer",
        candidate_skills=["Python", "Django", "PostgreSQL"],
        skill_gaps=["LangGraph", "ChromaDB RAG", "Model Context Protocol (MCP)"]
    )

    assert result["candidate_name"] == "Priya Sharma"
    assert "questions" in result
    assert len(result["questions"]) >= 3
    first_q = result["questions"][0]
    assert "question" in first_q
    assert "objective" in first_q
    assert "what_to_listen_for" in first_q
