"""
RAG Retriever Module.
Provides helper interfaces for querying the ingested Job Description knowledge base.
"""

from typing import List, Dict, Any
from rag.ingestor import rag_manager
from config import RAG_TOP_K, logger


def get_jd_context(query: str, k: int = RAG_TOP_K) -> str:
    """
    Retrieves and combines the most relevant JD passages into a coherent context string.
    """
    results = rag_manager.retrieve(query, k=k)
    if not results:
        return rag_manager.active_jd_full_text or "No Job Description currently indexed."

    passages = []
    for i, res in enumerate(results, start=1):
        passages.append(f"[JD Section {i}]:\n{res['text'].strip()}")

    return "\n\n".join(passages)


def get_active_jd() -> Dict[str, Any]:
    """Returns the title and full text of the currently indexed JD."""
    return {
        "title": rag_manager.active_jd_title,
        "full_text": rag_manager.active_jd_full_text
    }
