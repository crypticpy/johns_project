from __future__ import annotations

import logging
import os

# Import types lazily inside functions to avoid hard failures when optional deps are not installed
# This module is safe to import regardless of whether OpenTelemetry is installed.
from fastapi import FastAPI

from config.settings import AppSettings

_SERVICE_NAME = "sd-onboarding-analyzer"
_LOGGER = logging.getLogger("sd_onboarding")


def _resolve_otlp_http_endpoint() -> str:
    """
    Resolve OTLP HTTP traces endpoint.

    Priority:
    - OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
    - OTEL_EXPORTER_OTLP_ENDPOINT (append '/v1/traces' if not provided)
    - default 'http://localhost:4318/v1/traces'
    """
    traces_ep = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if traces_ep:
        return traces_ep.strip()
    base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if base:
        base = base.strip().rstrip("/")
        # Append v1/traces unless already present
        return base if base.endswith("/v1/traces") else f"{base}/v1/traces"
    return "http://localhost:4318/v1/traces"


def init_tracing(settings: AppSettings) -> None:
    """
    Initialize OpenTelemetry tracing if enabled via settings.

    - Configures TracerProvider with Resource(service.name=_SERVICE_NAME).
    - Adds BatchSpanProcessor with OTLP HTTP exporter.
    - Does not auto-enable metrics; strictly tracing only.

    Safe: No-ops if settings.enable_tracing is False or if OpenTelemetry packages are missing.
    """
    if not getattr(settings, "enable_tracing", False):
        return

    try:
        # Local imports to avoid module import errors if optional deps are not installed
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception as e:
        _LOGGER.warning("OpenTelemetry packages not available; tracing disabled: %s", e)
        return

    # Configure resource with service identity and environment tag
    attributes = {
        "service.name": _SERVICE_NAME,
        "deployment.environment": str(getattr(settings, "environment", "dev")),
    }
    resource = Resource.create(attributes=attributes)

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    endpoint = _resolve_otlp_http_endpoint()
    try:
        exporter = OTLPSpanExporter(endpoint=endpoint, timeout=5)
        span_processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(span_processor)
    except Exception as e:
        _LOGGER.warning("Failed to initialize OTLP exporter; tracing will be local-only: %s", e)


def instrument_fastapi_app(app: FastAPI) -> None:
    """
    Instrument a FastAPI app with OpenTelemetry without auto-metrics.

    Safe: No-ops if OpenTelemetry FastAPI instrumentation is not available.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except Exception as e:
        _LOGGER.debug("FastAPI OpenTelemetry instrumentation not available: %s", e)
        return

    try:
        # Exclude /metrics to keep metrics endpoint clean and avoid recursion
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=trace.get_tracer_provider(),
            excluded_urls="/metrics",
        )
    except Exception as e:
        _LOGGER.warning("Failed to instrument FastAPI with OpenTelemetry: %s", e)


def tracer() -> object | None:
    """
    Convenience accessor to get a tracer for manual spans.

    Returns None if OpenTelemetry is unavailable.
    """
    try:
        from opentelemetry import trace

        return trace.get_tracer(_SERVICE_NAME)
    except Exception:
        return None


def shutdown_tracing() -> None:
    """
    Flush and shutdown the tracer provider gracefully.

    Safe: No-ops if tracing is not initialized or SDK unavailable.
    """
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        # Some providers implement shutdown(); guard with hasattr
        if hasattr(provider, "shutdown"):
            try:
                provider.shutdown()
            except Exception as e:
                _LOGGER.debug("Tracer provider shutdown error: %s", e)
    except Exception:
        # SDK not available or provider not set; nothing to do
        pass
