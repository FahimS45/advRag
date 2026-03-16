"""
FastAPI application – entry point.

Startup sequence (lifespan):
  1. Open async DB connection pool.
  2. Initialise LangGraph AsyncPostgresSaver (creates checkpoint tables).
  3. Create chat_sessions table.
  4. Initialise PGVectorStore retriever.
  5. Inject retriever into node module.
  6. Compile the Self-RAG graph and stash on app.state.
  7. Launch background cleanup task.

Shutdown:
  Cancel cleanup task → close DB pool.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

# psycopg3 async is incompatible with Windows ProactorEventLoop (Python 3.8+ default).
# Force SelectorEventLoop on Windows before anything else starts.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.api import chat, health, sessions
from app.config import settings
from app.core.graph import build_graph
from app.core.nodes import set_retriever
from app.db.pool import close_pool, get_pool
from app.db.sessions import cleanup_expired_sessions, create_sessions_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ── Background cleanup task ───────────────────────────────────────────────────

async def _cleanup_loop() -> None:
    """Periodically purge expired sessions."""
    while True:
        await asyncio.sleep(settings.CLEANUP_INTERVAL_SECONDS)
        try:
            n = await cleanup_expired_sessions(settings.SESSION_TTL_MINUTES)
            if n:
                logger.info("Background cleanup removed %d session(s).", n)
        except Exception:  # noqa: BLE001
            logger.exception("Session cleanup task encountered an error.")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("── Startup ──────────────────────────────────────")

    # 1. DB pool
    pool = await get_pool()
    logger.info("Connection pool opened (max_size=%d).", settings.DB_POOL_MAX_SIZE)

    # 2. LangGraph checkpointer (creates tables if missing)
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    logger.info("LangGraph checkpointer ready.")

    # 3. Sessions table
    await create_sessions_table()

    # 4. Vector store — must use async engine for async retrieval
    from sqlalchemy.ext.asyncio import create_async_engine
    async_engine = create_async_engine(settings.VECTOR_DB_URI)
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=settings.VECTOR_TABLE_NAME,
        connection=async_engine,
        use_jsonb=True,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.RETRIEVAL_K},
    )
    logger.info("Vector store connected (table=%s).", settings.VECTOR_TABLE_NAME)

    # 5. Inject retriever into node module
    set_retriever(retriever)

    # 6. Compile graph
    app.state.rag_graph = build_graph(checkpointer)
    logger.info("Self-RAG graph compiled.")

    # 7. Background cleanup
    cleanup_task = asyncio.create_task(_cleanup_loop())
    logger.info(
        "Cleanup task started (TTL=%dm, interval=%ds).",
        settings.SESSION_TTL_MINUTES,
        settings.CLEANUP_INTERVAL_SECONDS,
    )

    logger.info("── Ready ─────────────────────────────────────────")
    yield  # ── Application runs here ──────────────────────────────

    # Shutdown
    logger.info("── Shutdown ─────────────────────────────────────")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    await close_pool()
    logger.info("Connection pool closed. Goodbye.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="BlueBug Self-RAG API",
        version="1.0.0",
        description=(
            "Company knowledge-base chatbot powered by LangGraph Self-RAG. "
            "Streams answers token-by-token via Server-Sent Events."
        ),
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(chat.router)

    return app


app = create_app()