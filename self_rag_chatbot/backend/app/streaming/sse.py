"""
Server-Sent Events streaming layer.

Taps into LangGraph's astream_events (v2) API to produce a structured
event stream the frontend can consume token-by-token.

Event protocol
--------------
progress  {"type": "progress", "node": "<node_name>", "message": "<human label>"}
token     {"type": "token",    "content": "<text chunk>"}
done      {"type": "done",     "answer": "<full answer>", "session_id": "<id>",
                               "need_retrieval": bool}
error     {"type": "error",    "content": "<message>"}
"""

import json
import logging
from typing import AsyncGenerator, Any

from langgraph.graph.state import CompiledStateGraph as CompiledGraph

logger = logging.getLogger(__name__)

# ── Human-readable labels for each graph node ────────────────────────────────

NODE_LABELS: dict[str, str] = {
    "decide_retrieval":      "Deciding whether to search the knowledge base…",
    "retrieve":              "Searching the knowledge base…",
    "is_relevant":           "Checking document relevance…",
    "generate_from_context": "Generating answer from context…",
    "generate_direct":       "Generating answer…",
    "check_is_sup":          "Verifying answer is grounded in sources…",
    "check_is_use":          "Checking if answer is useful…",
    "revise_answer":         "Revising answer for accuracy…",
    "rewrite_question":      "Rewriting query for better retrieval…",
    "no_answer_found":       "No relevant answer found in knowledge base.",
    "save_memory":           "Saving conversation…",
    "summarize_conversation":"Summarising conversation history…",
}

# Nodes that should NOT produce a visible progress event (internal plumbing)
_SILENT_NODES = {"save_memory", "summarize_conversation"}


def _format_sse(data: dict[str, Any], event: str | None = None) -> str:
    """Serialise a dict as an SSE frame."""
    payload = f"data: {json.dumps(data, ensure_ascii=False)}\n"
    if event:
        payload = f"event: {event}\n" + payload
    return payload + "\n"


async def stream_rag_response(
    rag_graph: CompiledGraph,
    question: str,
    thread_id: str,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted strings.

    Lifecycle:
        1. Emit `progress` events as each graph node starts.
        2. Stream `token` events for every chunk produced by the answer-
           generation LLM (identified by the "answer_generation" tag).
        3. Emit a single `done` event carrying the final answer and metadata.
        4. Yield a `keepalive` comment every ~15 s if no event arrives
           (proxies and browsers close idle SSE connections).
        5. On any exception, emit an `error` event.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 80,
    }

    initial_state = {
        "question": question,
        "need_retrieval": None,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "is_sup": None,
        "evidence": "",
        "retries": 0,
        "is_use": None,
        "useful_reason": "",
        "retrieval_query": question,
        "rewrite_tries": 0,
    }

    full_answer: list[str] = []
    final_state: dict = {}

    try:
        async for event in rag_graph.astream_events(
            initial_state, config=config, version="v2"
        ):
            kind = event.get("event", "")
            name = event.get("name", "")
            tags = event.get("tags", [])

            # ── Graph node started ──────────────────────────────────────────
            if kind == "on_chain_start" and name in NODE_LABELS and name not in _SILENT_NODES:
                yield _format_sse({
                    "type": "progress",
                    "node": name,
                    "message": NODE_LABELS[name],
                })

            # ── Answer generation token ─────────────────────────────────────
            elif kind == "on_chat_model_stream" and "answer_generation" in tags:
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    full_answer.append(chunk.content)
                    yield _format_sse({"type": "token", "content": chunk.content})

            # ── Graph finished ──────────────────────────────────────────────
            elif kind == "on_chain_end" and name == "LangGraph":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    final_state = output

        # Assemble final answer (prefer state field, fall back to streamed tokens)
        answer = final_state.get("answer") or "".join(full_answer)
        need_retrieval = final_state.get("need_retrieval", False)

        yield _format_sse({
            "type": "done",
            "answer": answer,
            "session_id": thread_id,
            "need_retrieval": bool(need_retrieval),
        })

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error streaming RAG response for thread %s", thread_id)
        yield _format_sse({"type": "error", "content": str(exc)})
