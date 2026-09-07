"""
Tests for RAG Ingestion and Semantic Retrieval.
"""

from rag.ingestor import rag_manager
from rag.retriever import get_jd_context


def test_rag_ingest_and_retrieve():
    jd_content = """
    Job Title: AI Solutions Architect
    Core Requirements:
    - Designing enterprise agentic workflows with LangGraph and LangChain.
    - Implementing high-throughput vector search with Chroma and FAISS.
    - Model Context Protocol (MCP) tool integration.
    """

    chunks = rag_manager.ingest_jd_text(jd_content, title="AI Solutions Architect")
    assert chunks >= 1

    # Query matching requirements
    context = get_jd_context("LangGraph agentic workflows vector search")
    assert len(context) > 20
    assert "LangGraph" in context or "AI Solutions Architect" in context
