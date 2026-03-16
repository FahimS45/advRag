"""
All LangGraph nodes for the Self-RAG pipeline.

Every node is an async function so LangGraph can execute the graph with
`astream_events`, giving us fine-grained SSE progress + token streaming.

Token streaming tags
──────────────────
We tag the `chat_llm` with `"answer_generation"` so the SSE layer can
filter `on_chat_model_stream` events specifically from generation nodes
(not from routing / verification LLMs).
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal

from app.config import settings
from app.core.prompts import (
    decide_retrieval_prompt,
    direct_generation_prompt,
    is_relevant_prompt,
    is_sup_prompt,
    is_use_prompt,
    rag_generation_prompt,
    revise_answer_prompt,
    rewrite_query_prompt,
)
from app.core.state import State

# ── LLM instances ─────────────────────────────────────────────────────────────

# Generation LLM – streaming enabled, tagged for SSE token filtering
chat_llm = ChatOpenAI(
    model=settings.CHAT_MODEL,
    temperature=settings.CHAT_TEMPERATURE,
    streaming=True,
    tags=["answer_generation"],
    api_key=settings.OPENAI_API_KEY,
)

# Inspection LLM – non-streaming, used for routing & verification
inspect_llm = ChatOpenAI(
    model=settings.INSPECT_MODEL,
    temperature=0,
    tags=["inspection"],
    api_key=settings.OPENAI_API_KEY,
)

# ── Pydantic output schemas ────────────────────────────────────────────────────

class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ..., description="True if external documents are needed to answer reliably, else False."
    )


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ..., description="True if the document helps answer the question, else False"
    )


class IsSupDecision(BaseModel):
    is_sup: Literal["fully_supported", "partially_supported", "not_supported"]
    evidence: list[str]


class IsUse(BaseModel):
    is_use: Literal["useful", "not_useful"]
    useful_reason: str = Field(..., description="Explain why the answer is useful in one sentence")


class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ..., description="Rewritten query optimized for vector retrieval against internal company PDF."
    )


# ── Structured-output chains ──────────────────────────────────────────────────
should_retrieve_llm = inspect_llm.with_structured_output(RetrieveDecision)
is_relevant_llm = inspect_llm.with_structured_output(RelevanceDecision)
is_sup_llm = inspect_llm.with_structured_output(IsSupDecision)
is_use_llm = inspect_llm.with_structured_output(IsUse)
rewrite_llm = inspect_llm.with_structured_output(RewriteDecision)

# ── Vectorstore retriever (set during app startup) ────────────────────────────
_retriever = None  # Injected at startup via set_retriever()


def set_retriever(retriever) -> None:
    """Called once at app startup after the vectorstore is initialised."""
    global _retriever
    _retriever = retriever


# ── Helper ────────────────────────────────────────────────────────────────────

def format_history(state: State) -> str:
    """Build a single string from summary + recent turns."""
    parts: list[str] = []

    summary_list = state.get("summary", [])
    summary = summary_list[0] if summary_list else ""
    if summary:
        parts.append(f"[Conversation Summary]\n{summary}")

    recent = state.get("chat_history", [])
    if recent:
        parts.append(
            "[Recent Turns]\n"
            + "\n".join(
                f"User: {h['question']}\nAssistant: {h['answer']}"
                for h in recent
            )
        )

    return "\n\n".join(parts) if parts else "No previous conversation."


# ── Graph nodes ────────────────────────────────────────────────────────────────

async def decide_retrieval(state: State) -> dict:
    """Decide whether we need to hit the vector store."""
    decision: RetrieveDecision = await should_retrieve_llm.ainvoke(
        decide_retrieval_prompt.format_messages(
            question=state["question"],
            chat_history=format_history(state),
        )
    )
    return {"need_retrieval": decision.should_retrieve}


async def generate_direct(state: State) -> dict:
    """Answer from chat history / general knowledge (no retrieval)."""
    out = await chat_llm.ainvoke(
        direct_generation_prompt.format_messages(
            question=state["question"],
            chat_history=format_history(state),
        )
    )
    return {"answer": out.content}


async def retrieve(state: State) -> dict:
    """Retrieve documents from the vector store."""
    if _retriever is None:
        raise RuntimeError("Retriever has not been initialised. Call set_retriever() first.")
    query = state.get("retrieval_query") or state["question"]
    docs = await _retriever.ainvoke(query)
    return {"docs": docs}


async def is_relevant(state: State) -> dict:
    """Filter retrieved documents to those relevant to the question."""
    relevant_docs: list[Document] = []

    for doc in state["docs"]:
        decision: RelevanceDecision = await is_relevant_llm.ainvoke(
            is_relevant_prompt.format_messages(
                question=state["question"],
                document=doc.page_content,
            )
        )
        if decision.is_relevant:
            relevant_docs.append(doc)

    return {"relevant_docs": relevant_docs}


async def generate_from_context(state: State) -> dict:
    """Generate an answer grounded in the retrieved context."""
    context = "\n\n---\n\n".join(
        d.page_content for d in state.get("relevant_docs", [])
    ).strip()

    out = await chat_llm.ainvoke(
        rag_generation_prompt.format_messages(
            question=state["question"],
            context=context,
        )
    )
    return {"answer": out.content, "context": context}


async def no_answer_found(state: State) -> dict:
    """Fallback when we have no good answer after exhausting all options."""
    return {
        "answer": (
            "I don't have enough information on that. "
            "Please contact us at info@bluebug.com for further assistance!"
        ),
        "context": "",
    }


async def check_is_sup(state: State) -> dict:
    """Verify that the generated answer is factually grounded in the context."""
    decision: IsSupDecision = await is_sup_llm.ainvoke(
        is_sup_prompt.format_messages(
            question=state["question"],
            answer=state["answer"],
            context=state["context"],
        )
    )
    return {"is_sup": decision.is_sup, "evidence": decision.evidence}


async def check_is_use(state: State) -> dict:
    """Check whether the (possibly revised) answer is useful to the user."""
    decision: IsUse = await is_use_llm.ainvoke(
        is_use_prompt.format_messages(
            question=state["question"],
            answer=state["answer"],
        )
    )
    return {"is_use": decision.is_use, "useful_reason": decision.useful_reason}


async def revise_answer(state: State) -> dict:
    """Rewrite the answer using strict direct quotes from the context."""
    out = await inspect_llm.ainvoke(
        revise_answer_prompt.format_messages(
            question=state["question"],
            answer=state["answer"],
            context=state["context"],
        )
    )
    return {
        "answer": out.content,
        "retries": state.get("retries", 0) + 1,
    }


async def rewrite_question(state: State) -> dict:
    """Produce a better retrieval query when the first answer wasn't useful."""
    decision: RewriteDecision = await rewrite_llm.ainvoke(
        rewrite_query_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        # Reset retrieval state so next pass is clean
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }


async def save_memory(state: State) -> dict:
    """Append current Q/A pair to chat history."""
    new_turn = {"question": state["question"], "answer": state["answer"]}
    updated = list(state.get("chat_history", []) or []) + [new_turn]
    return {"chat_history": updated}


async def summarize_conversation(state: State) -> dict:
    """Summarise older turns and trim the in-memory history."""
    chat_history = state.get("chat_history", [])
    existing_summary_list = state.get("summary", [])
    existing_summary = existing_summary_list[0] if existing_summary_list else ""

    history_str = "\n".join(
        f"User: {h['question']}\nAssistant: {h['answer']}"
        for h in chat_history
    )

    if existing_summary:
        prompt = (
            f"You are maintaining a running summary of a RAG assistant conversation.\n\n"
            f"Existing summary:\n{existing_summary}\n\n"
            f"New conversation turns:\n{history_str}\n\n"
            "Update and extend the summary to include the new turns. "
            "Focus on: topics discussed, key facts retrieved, and decisions made. "
            "Be concise but preserve important context."
        )
    else:
        prompt = (
            f"Summarize the following RAG assistant conversation.\n\n"
            f"{history_str}\n\n"
            "Focus on: topics discussed, key facts retrieved, and decisions made. "
            "Be concise but preserve important context."
        )

    response = await inspect_llm.ainvoke([{"role": "user", "content": prompt}])

    trimmed = chat_history[-settings.KEEP_LAST_N:]
    return {
        "summary": [response.content],
        "chat_history": trimmed,
    }


# ── Conditional routing functions ─────────────────────────────────────────────

def route_after_decide(state: State) -> str:
    return "retrieve" if state["need_retrieval"] else "generate_direct"


def route_after_relevance(state:State):
    if len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    else:
        return "no_answer_found"


def route_after_issup(state: State) -> str:
    if state["is_sup"] == "fully_supported":
        return "is_use"
    if state.get("retries", 0) >= settings.MAX_RETRIES:
        return "is_use"  # give up revising; pass to usefulness check anyway
    return "revise_answer"


def route_after_isuse(state: State) -> str:
    if state["is_use"] == "useful":
        return "save_memory"
    if state.get("rewrite_tries", 0) >= settings.MAX_REWRITE_TRIES:
        return "no_answer_found"
    return "rewrite_question"


def should_summarize(state: State) -> bool:
    return len(state.get("chat_history", []) or []) > settings.SUMMARIZE_AFTER
