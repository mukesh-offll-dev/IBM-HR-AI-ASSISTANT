"""
Tests for Candidate Scorer Tool.
"""

from tools.scorer import score_candidate


def test_score_candidate_structure():
    candidate_json = {
        "name": "Alex Morgan",
        "title": "Senior AI Engineer",
        "skills": ["Python", "FastAPI", "LangChain", "LangGraph", "ChromaDB", "Docker"],
        "years_of_experience": 6.0,
        "education": ["B.S. in Computer Science"],
        "certifications": ["AWS Certified"],
        "raw_summary": "Experienced AI engineer building multi-agent systems and RAG pipelines."
    }

    jd_text = """
    Job Title: Senior Backend AI Engineer
    Requirements:
    - 4+ years Python backend experience (FastAPI)
    - LangChain, LangGraph, and RAG architectures
    - Vector databases (Chroma, FAISS)
    """

    result = score_candidate(candidate_json=candidate_json, jd_text=jd_text)

    assert "match_score" in result
    assert 0 <= result["match_score"] <= 100
    assert "fit_level" in result
    assert "strengths" in result
    assert "gaps" in result
    assert "recommendation" in result
    assert result["candidate_name"] == "Alex Morgan"
