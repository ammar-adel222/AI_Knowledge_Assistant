# config.py
import os
import this

from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()


class Config:
    # ── API Keys ──────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")




    LLM_MODEL: str = "llama3-70b-8192"  # Main model for Q&A
    SUMMARIZER_MODEL: str = "llama3-8b-8192"  # Lighter model for summarization

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
    def get_qa_llm(cls) -> ChatGroq:
        return ChatGroq(
            api_key=cls.GROQ_API_KEY,
            model_name=cls.LLM_MODEL,
            temperature=0.2,
            max_tokens=1024,
        )


    @classmethod
    def get_summarizer_llm(cls) -> ChatGroq:
        return ChatGroq(
            api_key=cls.GROQ_API_KEY,
            model_name=cls.SUMMARIZER_MODEL,
            temperature=0.3,
            max_tokens=2048
        )

    @classmethod
    def validate(cls):
        if not cls.GROQ_API_KEY:
            raise ValueError("❌ GROQ_API_KEY missing in .env file")
        print("✅ Config ready — Groq LLaMA")