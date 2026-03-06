"""
Configuration loader for Multimodal Math Mentor.
Reads from .env file and provides centralized access to all settings.
"""

import os
from dotenv import load_dotenv

# Load .env file from project root, overriding any stale shell variables
load_dotenv(override=True)


# ── LLM ──────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "openai", "groq", or "gemini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-pro")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


def get_llm_client():
    """
    Return an OpenAI-compatible client for the configured LLM provider.
    Groq's API is OpenAI-compatible, so we use the openai SDK with a custom base_url.
    """
    from openai import OpenAI

    if LLM_PROVIDER == "groq":
        return OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
    elif LLM_PROVIDER == "gemini":
        return OpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    else:
        return OpenAI(api_key=OPENAI_API_KEY)

# ── Whisper ──────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# ── RAG ──────────────────────────────────────────
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base")

# ── Thresholds ───────────────────────────────────
VERIFIER_CONFIDENCE_THRESHOLD = float(
    os.getenv("VERIFIER_CONFIDENCE_THRESHOLD", "0.7")
)
OCR_CONFIDENCE_THRESHOLD = float(
    os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.6")
)

# ── Server ───────────────────────────────────────
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))

# ── Paths ────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQLITE_DB_PATH = os.path.join(PROJECT_ROOT, "data", "memory.db")
