"""
State schema for the Self-RAG LangGraph.

All fields are typed explicitly so LangGraph can serialise them to Postgres.

Note on reducers:
  - `chat_history` and `summary` use a *replace* reducer so that we can write
    back a trimmed/updated list without accidentally appending duplicates.
  - Everything else uses the default (last-write-wins) behaviour.
"""
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document


def _replace(old, new):
    """Simple replacement reducer – new value completely replaces old."""
    return new


class State(TypedDict):
    # ── Per-turn inputs ───────────────────────────────────────────────────────
    question: str
    retrieval_query: str          # Rewritten query for vector retrieval

    # ── Routing decisions ─────────────────────────────────────────────────────
    need_retrieval: bool

    # ── Retrieval ─────────────────────────────────────────────────────────────
    docs: list[Document]          # Raw retrieved documents
    relevant_docs: list[Document] # Filtered relevant documents
    context: str                  # Joined page_content of relevant_docs

    # ── Generation ────────────────────────────────────────────────────────────
    answer: str

    # ── Verification ──────────────────────────────────────────────────────────
    is_sup: Literal["fully_supported", "partially_supported", "not_supported"]
    evidence: list[str]
    retries: int                  # Times we have revised the answer

    # ── Usefulness check ──────────────────────────────────────────────────────
    is_use: Literal["useful", "not_useful"]
    useful_reason: str
    rewrite_tries: int            # Times we have rewritten the retrieval query

    # ── Persistent memory (replacement reducer so trimming works correctly) ──
    chat_history: Annotated[list[dict], _replace]  # [{question, answer}, …]
    summary: Annotated[list[str], _replace]        # [latest_summary_text]
