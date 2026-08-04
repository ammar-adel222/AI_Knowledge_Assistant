# config.py
import os
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()


class Config:
    # ── API Keys ──────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")




    LLM_MODEL: str = "llama-3.3-70b-versatile"  # Main model for Q&A
    SUMMARIZER_MODEL: str = "llama-3.1-8b-instant"  # Lighter model for summarization

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Chunking ──────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # ── Vector Store ──────────────────────────────
    VECTOR_STORE_PATH: str = "./vector_store"

    # ── YouTube Summarization ─────────────────────
    YOUTUBE_MAX_SUMMARY_LENGTH: int = 5000

    @classmethod
    def get_api_key(cls) -> str:
        load_dotenv(override=True)
        return os.getenv("GROQ_API_KEY", cls.GROQ_API_KEY)

    @classmethod
    def get_qa_llm(cls) -> ChatGroq:
        api_key = cls.get_api_key()
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing! Please provide a valid Groq API key.")
        return ChatGroq(
            api_key=api_key,
            model_name=cls.LLM_MODEL,
            temperature=0.7,
            max_tokens=1024,
        )

    @classmethod
    def get_summarizer_llm(cls) -> ChatGroq:
        api_key = cls.get_api_key()
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing! Please provide a valid Groq API key.")
        return ChatGroq(
            api_key=api_key,
            model_name=cls.SUMMARIZER_MODEL,
            temperature=0.3,
            max_tokens=2048
        )

    @classmethod
    def validate(cls):
        if not cls.GROQ_API_KEY:
            raise ValueError("❌ GROQ_API_KEY missing in .env file")
        print("✅ Config ready — Groq LLaMA")