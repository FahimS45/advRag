"""
Health-check endpoint.
Used by load balancers, container orchestrators, and uptime monitors.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    db: str
    graph: str


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """
    Returns 200 when the service is ready to handle traffic.
    Checks that the DB pool and compiled graph are initialised.
    """
    from app.db.pool import get_pool

    db_status = "ok"
    try:
        pool = await get_pool()
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")
    except Exception:  # noqa: BLE001
        db_status = "unavailable"

    graph_status = "ok" if getattr(request.app.state, "rag_graph", None) else "unavailable"

    overall = "ok" if db_status == "ok" and graph_status == "ok" else "degraded"

    return HealthResponse(status=overall, db=db_status, graph=graph_status)
