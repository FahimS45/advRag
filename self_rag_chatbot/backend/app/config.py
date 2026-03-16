"""
Centralised configuration loaded from environment variables / .env file.
All tuneable knobs live here so you never hard-code values elsewhere.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── OpenAI ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str 
    # ── PostgreSQL (LangGraph checkpointer + session table) ──────────────────
    DB_URI: str = "postgresql://postgres:postgres@localhost:5442/postgres"
    DB_POOL_MAX_SIZE: int = 20

    # ── PostgreSQL (LangChain PGVectorStore) ─────────────────────────────────
    # Uses the psycopg3 dialect (postgresql+psycopg, NOT psycopg2)
    VECTOR_DB_URI: str = "postgresql+psycopg://postgres:postgres@localhost:5442/postgres"
    VECTOR_TABLE_NAME: str = "company_docs"
    VECTOR_SIZE: int = 3072
    EMBEDDING_MODEL: str = "text-embedding-3-large"

    # ── LLM ──────────────────────────────────────────────────────────────────
    CHAT_MODEL: str = "gpt-4o-mini"       # Used for answer generation (streaming)
    INSPECT_MODEL: str = "gpt-4o-mini"    # Used for routing / verification (non-streaming)
    CHAT_TEMPERATURE: float = 0.7
    RETRIEVAL_K: int = 5

    # ── Graph limits ─────────────────────────────────────────────────────────
    MAX_RETRIES: int = 3          # Max times to revise an answer for factual grounding
    MAX_REWRITE_TRIES: int = 3    # Max times to rewrite the retrieval query
    SUMMARIZE_AFTER: int = 3      # Summarise chat history after N turns
    KEEP_LAST_N: int = 2          # Keep last N turns in memory after summarising
    RECURSION_LIMIT: int = 80

    # ── Session management ───────────────────────────────────────────────────
    SESSION_TTL_MINUTES: int = 30   # Sessions idle longer than this are purged
    CLEANUP_INTERVAL_SECONDS: int = 300  # How often the cleanup task runs (5 min)

    # ── CORS ─────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
