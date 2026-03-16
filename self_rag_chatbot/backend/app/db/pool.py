"""
Async PostgreSQL connection pool shared across the application.

Uses `psycopg_pool.AsyncConnectionPool` which is compatible with both:
  - LangGraph's `AsyncPostgresSaver` (checkpointer)
  - Direct DB operations (session tracking, cleanup)
"""
from __future__ import annotations

import logging

from psycopg_pool import AsyncConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None


async def get_pool() -> AsyncConnectionPool:
    """Return the shared connection pool, creating it if necessary."""
    global _pool
    if _pool is None:
        logger.info("Opening Postgres connection pool…")
        _pool = AsyncConnectionPool(
            conninfo=settings.DB_URI,
            max_size=settings.DB_POOL_MAX_SIZE,
            # autocommit=True is required by LangGraph's AsyncPostgresSaver
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,  # We call .open() explicitly below
        )
        await _pool.open()
        logger.info("Postgres connection pool ready.")
    return _pool


async def close_pool() -> None:
    """Gracefully close the pool at shutdown."""
    global _pool
    if _pool is not None:
        logger.info("Closing Postgres connection pool…")
        await _pool.close()
        _pool = None
        logger.info("Postgres connection pool closed.")
