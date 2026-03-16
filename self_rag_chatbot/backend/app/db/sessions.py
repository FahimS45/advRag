"""
Session lifecycle management.

Responsibilities:
- Maintain a `chat_sessions` table (created at startup).
- Track last-active timestamps so the background cleanup task can prune
  idle sessions together with all LangGraph checkpoint data.
- Provide explicit delete helper used by DELETE /api/sessions/{thread_id}.

Cleanup strategy (industry standard TTL + explicit delete):
  1. Background task runs every CLEANUP_INTERVAL_SECONDS and removes
     sessions that have been idle longer than SESSION_TTL_MINUTES.
  2. A client-triggered DELETE call wipes the session immediately.
  Both paths delete from the LangGraph checkpoint tables first, then
  from chat_sessions — preserving FK integrity.
"""

import logging
from datetime import datetime
from typing import Optional

from psycopg.rows import dict_row

from app.db.pool import get_pool

logger = logging.getLogger(__name__)

# ── DDL ─────────────────────────────────────────────────────────────────────

CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    thread_id       TEXT PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active       BOOLEAN     NOT NULL DEFAULT TRUE
);
"""

CREATE_LAST_ACTIVE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_chat_sessions_last_active
    ON chat_sessions (last_active_at);
"""

# ── Setup ────────────────────────────────────────────────────────────────────

async def create_sessions_table() -> None:
    """Run at startup to ensure the sessions table exists."""
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(CREATE_SESSIONS_TABLE)
        await conn.execute(CREATE_LAST_ACTIVE_INDEX)
    logger.info("chat_sessions table ready.")


# ── CRUD ─────────────────────────────────────────────────────────────────────

async def create_session(thread_id: str) -> dict:
    """Insert a new session row and return it."""
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO chat_sessions (thread_id)
                VALUES (%s)
                ON CONFLICT (thread_id) DO UPDATE
                    SET last_active_at = NOW(),
                        is_active      = TRUE
                RETURNING thread_id, created_at, last_active_at, is_active
                """,
                (thread_id,),
            )
            row = await cur.fetchone()
    return dict(row)


async def get_session(thread_id: str) -> Optional[dict]:
    """Return session metadata, or None if not found / inactive."""
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT thread_id, created_at, last_active_at, is_active
                FROM   chat_sessions
                WHERE  thread_id = %s
                """,
                (thread_id,),
            )
            row = await cur.fetchone()
    return dict(row) if row else None


async def update_session_activity(thread_id: str) -> None:
    """Bump last_active_at; called at the start of every chat turn."""
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE chat_sessions
            SET    last_active_at = NOW()
            WHERE  thread_id = %s
            """,
            (thread_id,),
        )


async def delete_session(thread_id: str) -> bool:
    """
    Hard-delete a session and all associated LangGraph checkpoint data.
    Returns True if a session was actually deleted.
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        # Remove checkpoint data first (no FK constraints, but logical order)
        for table in ("checkpoint_blobs", "checkpoint_writes", "checkpoints"):
            await conn.execute(
                f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,)  # noqa: S608
            )
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM chat_sessions WHERE thread_id = %s",
                (thread_id,),
            )
            deleted = cur.rowcount > 0
    if deleted:
        logger.info("Session %s deleted.", thread_id)
    return deleted


# ── Background cleanup ────────────────────────────────────────────────────────

async def cleanup_expired_sessions(ttl_minutes: int) -> int:
    """
    Delete sessions idle longer than `ttl_minutes`.
    Returns the number of sessions purged.
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        # Collect expired thread IDs
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT thread_id
                FROM   chat_sessions
                WHERE  last_active_at < NOW() - INTERVAL '%s minutes'
                  AND  is_active = TRUE
                """,
                (ttl_minutes,),
            )
            rows = await cur.fetchall()

        if not rows:
            return 0

        expired_ids = [r["thread_id"] for r in rows]
        placeholders = ",".join(["%s"] * len(expired_ids))

        for table in ("checkpoint_blobs", "checkpoint_writes", "checkpoints"):
            await conn.execute(
                f"DELETE FROM {table} WHERE thread_id IN ({placeholders})",  # noqa: S608
                expired_ids,
            )

        async with conn.cursor() as cur:
            await cur.execute(
                f"DELETE FROM chat_sessions WHERE thread_id IN ({placeholders})",  # noqa: S608
                expired_ids,
            )
            count = cur.rowcount

    logger.info("Cleanup: purged %d expired session(s).", count)
    return count
