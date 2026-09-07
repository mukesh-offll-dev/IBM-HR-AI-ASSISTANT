"""
Configuration module for the AI HR Recruitment Assistant.
Loads environment variables and initializes core services.
"""

import os
from pathlib import Path
from typing import List, Optional, Any
from dotenv import load_dotenv
import logging

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("HR_Assistant")

# Ollama LLM Settings
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://api.ollama.com").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b").strip()

# Vector Store / RAG Settings
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "chroma_db"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "hr_job_descriptions")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "700"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

# Web Server Settings
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8001"))

# Safe Uploads Directory (transient storage)
TEMP_UPLOAD_DIR = BASE_DIR / "temp_resumes"
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)


# LangChain Custom LLM Wrapper for Ollama Cloud
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
import ollama

class OllamaCloudChat(BaseChatModel):
    """Custom LangChain chat model wrapper for Ollama Cloud."""
    client: Any = None
    model: str = "gpt-oss:120b"
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "ollama-cloud"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ChatResult:
        ollama_msgs = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, HumanMessage):
                role = "user"
            ollama_msgs.append({"role": role, "content": str(m.content)})

        try:
            options = {"temperature": self.temperature}
            if stop:
                options["stop"] = stop
            resp = self.client.chat(
                model=self.model,
                messages=ollama_msgs,
                options=options
            )
            content = resp.get("message", {}).get("content", "")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
        except Exception as e:
            logger.error(f"Ollama invocation error: {e}")
            raise e

def get_ollama_client() -> ollama.Client:
    """Returns an authenticated ollama.Client."""
    headers = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return ollama.Client(host=OLLAMA_BASE_URL, headers=headers)

def get_llm(temperature: float = 0.2) -> OllamaCloudChat:
    """Returns a LangChain-compatible LLM instance."""
    client = get_ollama_client()
    return OllamaCloudChat(client=client, model=OLLAMA_MODEL, temperature=temperature)
