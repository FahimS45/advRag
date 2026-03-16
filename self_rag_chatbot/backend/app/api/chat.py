"""
Chat endpoint — streams the Self-RAG response as Server-Sent Events.

POST /api/chat/stream
  Body:  { "question": "...", "thread_id": "..." }
  Returns: text/event-stream

The client MUST have already called POST /api/sessions to obtain a thread_id.
"""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.db import sessions as session_db
from app.streaming.sse import stream_rag_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    thread_id: str = Field(..., description="Session ID from POST /api/sessions")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/stream",
    summary="Stream a chat response (SSE)",
    response_description="Server-Sent Event stream",
)
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """
    Accepts a question + thread_id and returns a live SSE stream.

    Event types
    -----------
    progress  – a graph node has started (shows thinking steps)
    token     – one text chunk from the answer LLM
    done      – final answer + metadata
    error     – something went wrong
    """
    # ── Guard: session must exist ────────────────────────────────────────────
    session = await session_db.get_session(body.thread_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Session '{body.thread_id}' not found. "
                "Call POST /api/sessions first."
            ),
        )

    # ── Guard: compiled graph must be available ──────────────────────────────
    rag_graph = getattr(request.app.state, "rag_graph", None)
    if rag_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG graph is not yet initialised. Please retry shortly.",
        )

    # ── Bump last-active timestamp ───────────────────────────────────────────
    await session_db.update_session_activity(body.thread_id)

    # ── Return SSE stream ────────────────────────────────────────────────────
    return StreamingResponse(
        stream_rag_response(rag_graph, body.question, body.thread_id),
        media_type="text/event-stream",
        headers={
            # Disable buffering in nginx / proxies
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
