"""
LangChain Tool for Job Requirements Retrieval.
"""

from typing import List, Dict, Any
from rag.retriever import get_jd_context
from rag.ingestor import rag_manager


def retrieve_job_requirements(query: str, k: int = 4) -> str:
    """
    LangChain tool: Vector search over ingested Job Descriptions / hiring-criteria documents.
    Returns the most relevant requirements, skills, and qualifications matching the query.
    """
    return get_jd_context(query, k=k)
