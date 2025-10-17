from __future__ import annotations

import time
from typing import Any

# Soft dependency: prometheus_client is required at runtime for metrics, but we avoid hard import failures.
try:
    from prometheus_client import Counter, Histogram  # type: ignore

    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover
    Counter = None  # type: ignore[assignment]
    Histogram = None  # type: ignore[assignment]
    _PROM_AVAILABLE = False

# Label hygiene: keep labels low-cardinality (endpoint, method, status)
REQUEST_COUNT = (
    Counter("request_count", "Total HTTP requests", labelnames=("endpoint", "method", "status"))
    if _PROM_AVAILABLE
    else None
)

REQUEST_LATENCY_SECONDS = (
    Histogram(
        "request_latency_seconds",
        "HTTP request latency in seconds",
        labelnames=("endpoint", "method"),
        buckets=(
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
        ),
    )
    if _PROM_AVAILABLE
    else None
)

# Optional domain metrics (lightweight, can be used by engine adapters)
EMBEDDINGS_THROUGHPUT = (
    Counter("embeddings_throughput", "Count of embeddings generated", labelnames=("model",))
    if _PROM_AVAILABLE
    else None
)
VECTOR_SEARCH_LATENCY = (
    Histogram("vector_search_latency_seconds", "Vector search latency in seconds")
    if _PROM_AVAILABLE
    else None
)
ANALYSIS_TOKEN_USAGE = (
    Counter("analysis_token_usage", "Estimated tokens used for analysis")
    if _PROM_AVAILABLE
    else None
)


def _extract_endpoint(scope: dict[str, Any]) -> str:
    # Prefer route path template if available; fallback to raw path
    try:
        route = scope.get("route")
        if route is not None:
            # Starlette/FastAPI Route has .path
            path = getattr(route, "path", None)
            if isinstance(path, str) and path:
                return path
    except Exception:
        pass
    path = scope.get("path") or scope.get("raw_path")
    if isinstance(path, bytes):
        try:
            path = path.decode("utf-8", errors="ignore")
        except Exception:
            path = "/unknown"
    if not isinstance(path, str):
        return "/unknown"
    # Avoid high cardinality by trimming trailing slashes
    return path.rstrip("/") or "/"


class MetricsMiddleware:
    """
    ASGI middleware that records request counts and latencies.

    - Avoids recording bodies/headers to prevent PII exposure.
    - Labels: endpoint (path template or path), method, status code.
    - If prometheus_client is unavailable, acts as a no-op pass-through.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        if not _PROM_AVAILABLE or REQUEST_LATENCY_SECONDS is None or REQUEST_COUNT is None:
            # Prometheus not available: pass-through without metrics
            return await self.app(scope, receive, send)

        method = (scope.get("method") or "GET").upper()
        endpoint = _extract_endpoint(scope)
        start = time.monotonic()

        status_code_holder: dict[str, int | None] = {"status": None}

        async def send_wrapper(message: dict[str, Any]):
            if message.get("type") == "http.response.start":
                status = message.get("status")
                try:
                    status_code_holder["status"] = int(status) if status is not None else None
                except Exception:
                    status_code_holder["status"] = None
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = max(0.0, time.monotonic() - start)
            # type: ignore[union-attr]
            REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint, method=method).observe(duration)
            status_label = str(status_code_holder["status"] or 0)
            # type: ignore[union-attr]
            REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status_label).inc()
