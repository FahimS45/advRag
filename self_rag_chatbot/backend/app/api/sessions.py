"""
Session management endpoints.

POST   /api/sessions              – Create (or reactivate) a session
GET    /api/sessions/{thread_id}  – Check session status
DELETE /api/sessions/{thread_id}  – Terminate session + erase all checkpoint data
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.db import sessions as session_db

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class SessionCreateResponse(BaseModel):
    thread_id: str
    created_at: datetime
    last_active_at: datetime
    is_active: bool


class SessionStatusResponse(BaseModel):
    thread_id: str
    created_at: datetime
    last_active_at: datetime
    is_active: bool
    exists: bool


class SessionDeleteResponse(BaseModel):
    thread_id: str
    deleted: bool
    message: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat session",
)
async def create_session() -> SessionCreateResponse:
    """
    Generate a unique thread_id and persist the session row.
    Returns the thread_id the client must include in every subsequent request.
    """
    thread_id = str(uuid.uuid4())
    row = await session_db.create_session(thread_id)
    return SessionCreateResponse(**row)


@router.get(
    "/{thread_id}",
    response_model=SessionStatusResponse,
    summary="Get session status",
)
async def get_session(thread_id: str) -> SessionStatusResponse:
    """Returns session metadata. Use this to check whether a session is still alive."""
    row = await session_db.get_session(thread_id)
    if row is None:
        return SessionStatusResponse(
            thread_id=thread_id,
            created_at=datetime.utcnow(),
            last_active_at=datetime.utcnow(),
            is_active=False,
            exists=False,
        )
    return SessionStatusResponse(**row, exists=True)


@router.delete(
    "/{thread_id}",
    response_model=SessionDeleteResponse,
    summary="Terminate session and erase all data",
)
async def delete_session(thread_id: str) -> SessionDeleteResponse:
    """
    Immediately removes the session and ALL associated LangGraph checkpoint
    data (checkpoint_blobs, checkpoint_writes, checkpoints).
    Call this when the user ends a conversation.
    """
    deleted = await session_db.delete_session(thread_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{thread_id}' not found.",
        )
    return SessionDeleteResponse(
        thread_id=thread_id,
        deleted=True,
        message="Session and all associated data have been erased.",
    )
