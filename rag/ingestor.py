"""
RAG Ingestion and Vector Storage for Job Descriptions & Hiring Criteria.
Chunks documents, indexes them in ChromaDB or lightweight local vector index, and provides semantic retrieval.
"""

import math
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_TOP_K,
    logger
)

# Lightweight, robust local vector store using TF-IDF + Cosine Similarity fallback
# ensures 100% reliability if external embedding servers or heavy models are unreachable.
class LightweightVectorStore:
    """In-memory or disk-backed semantic index with TF-IDF cosine ranking."""
    def __init__(self):
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.vocabulary: Dict[str, int] = {}
        self.doc_vectors: List[Dict[int, float]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w{2,}\b", text.lower())

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        if not texts:
            return
        if metadatas is None:
            metadatas = [{} for _ in texts]

        start_idx = len(self.documents)
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

        # Build vocabulary
        for text in texts:
            words = set(self._tokenize(text))
            for w in words:
                if w not in self.vocabulary:
                    self.vocabulary[w] = len(self.vocabulary)

        # Build vectors (TF-IDF)
        N = len(self.documents)
        # Calculate DF
        dfs = {w: 0 for w in self.vocabulary}
        for doc in self.documents:
            words = set(self._tokenize(doc))
            for w in words:
                if w in dfs:
                    dfs[w] += 1

        self.doc_vectors = []
        for doc in self.documents:
            tokens = self._tokenize(doc)
            total = len(tokens) or 1
            tf = {}
            for t in tokens:
                idx = self.vocabulary.get(t)
                if idx is not None:
                    tf[idx] = tf.get(idx, 0) + 1 / total
            # Compute TF-IDF
            vec = {}
            norm = 0.0
            for idx, freq in tf.items():
                w = list(self.vocabulary.keys())[idx]
                idf = math.log((1 + N) / (1 + dfs.get(w, 0))) + 1.0
                val = freq * idf
                vec[idx] = val
                norm += val * val
            norm = math.sqrt(norm) or 1.0
            for idx in vec:
                vec[idx] /= norm
            self.doc_vectors.append(vec)

    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        if not self.doc_vectors:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return [{"text": d, "metadata": m, "score": 0.5} for d, m in zip(self.documents[:k], self.metadatas[:k])]

        q_tf = {}
        for t in q_tokens:
            if t in self.vocabulary:
                idx = self.vocabulary[t]
                q_tf[idx] = q_tf.get(idx, 0) + 1
        q_norm = math.sqrt(sum(v * v for v in q_tf.values())) or 1.0
        q_vec = {k: v / q_norm for k, v in q_tf.items()}

        scores = []
        for i, doc_vec in enumerate(self.doc_vectors):
            dot = sum(q_vec[k] * doc_vec[k] for k in q_vec if k in doc_vec)
            scores.append((dot, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scores[:k]:
            results.append({
                "text": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": float(score)
            })
        return results


# Global RAG Manager combining ChromaDB and fallback
class RAGManager:
    def __init__(self):
        self.chroma_client = None
        self.chroma_collection = None
        self.fallback_index = LightweightVectorStore()
        self.active_jd_title = "General Job Description"
        self.active_jd_full_text = ""
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            persist_dir = Path(CHROMA_PERSIST_DIRECTORY)
            persist_dir.mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=str(persist_dir))
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME
            )
            logger.info("ChromaDB initialized successfully.")
        except Exception as e:
            logger.warning(f"ChromaDB initialization notice ({e}). Using robust built-in vector store.")
            self.chroma_client = None
            self.chroma_collection = None

    def ingest_jd_text(self, jd_text: str, title: str = "Job Description") -> int:
        """Chunks and stores job description in vector database."""
        self.active_jd_title = title
        self.active_jd_full_text = jd_text

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(jd_text)
        logger.info(f"Ingesting JD '{title}': generated {len(chunks)} chunks.")

        # Ingest in fallback index
        metadatas = [{"source": title, "chunk_id": i} for i in range(len(chunks))]
        self.fallback_index = LightweightVectorStore()
        self.fallback_index.add_texts(chunks, metadatas)

        # Ingest in ChromaDB if available
        if self.chroma_collection:
            try:
                ids = [f"jd_chunk_{i}" for i in range(len(chunks))]
                self.chroma_collection.upsert(
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                logger.warning(f"ChromaDB upsert fallback notice: {e}")

        return len(chunks)

    def retrieve(self, query: str, k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
        """Retrieves top-k relevant JD chunks for a query."""
        if self.chroma_collection:
            try:
                results = self.chroma_collection.query(
                    query_texts=[query],
                    n_results=min(k, max(1, self.chroma_collection.count()))
                )
                if results and results.get("documents") and results["documents"][0]:
                    docs = results["documents"][0]
                    metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                    return [{"text": d, "metadata": m, "score": 1.0} for d, m in zip(docs, metas)]
            except Exception as e:
                logger.warning(f"Chroma retrieval notice: {e}")

        # Fallback to local vector search
        return self.fallback_index.similarity_search(query, k=k)


# Singleton instance
rag_manager = RAGManager()
