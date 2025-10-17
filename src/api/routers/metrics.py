from __future__ import annotations

from fastapi import APIRouter, Response

router = APIRouter(tags=["system"])

# Prometheus exposition format content type
_PROM_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@router.get("/metrics")
async def metrics() -> Response:
    """
    Expose Prometheus metrics collected via MetricsMiddleware and other counters.

    Returns 200 with the latest metrics in Prometheus text format.
    If prometheus_client is unavailable, returns 503 gracefully.
    """
    try:
        from prometheus_client import generate_latest
    except Exception:
        return Response(status_code=503, content=b"prometheus_client not installed")
    payload = generate_latest()  # bytes
    return Response(content=payload, media_type=_PROM_CONTENT_TYPE)