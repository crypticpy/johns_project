import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from config.env import get_settings
from api.routers.ingest import router as ingest_router
from api.routers.embed import router as embed_router
from api.routers.search import router as search_router
from api.routers.analytics import router as analytics_router
from api.routers.cluster import router as cluster_router
from api.routers.analysis import router as analysis_router
from api.routers.history import router as history_router
from api.routers.reports import router as reports_router
from api.routers.metrics import router as metrics_router
from db.session import init_db
from utils.logging import init_logging
from observability.metrics import MetricsMiddleware
from observability.tracing import init_tracing, instrument_fastapi_app, shutdown_tracing, tracer


def create_app() -> FastAPI:
    settings = get_settings()

    # Initialize structured logging early; safe fallback inside init_logging
    try:
        cfg_path = str(settings.logging_config_path) if settings.logging_config_path else "src/config/logging.yaml"
        init_logging(cfg_path)
    except Exception:
        # Never fail app creation due to logging config
        pass

    # Initialize tracing provider (no-op if disabled or otel missing)
    try:
        init_tracing(settings)
    except Exception:
        # Never fail app creation due to tracing init
        pass

    app = FastAPI(
        title="Service Desk Onboarding Analyzer API",
        description=(
            "Modular backend API derived from reference behaviors in "
            "servicenow_analyzer.py and requirements in plan.md"
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Instrument FastAPI with OpenTelemetry (no auto-metrics)
    try:
        instrument_fastapi_app(app)
    except Exception:
        pass

    # CORS per settings; default open in dev, restrict in prod via APP_ALLOWED_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics middleware (low-cardinality labels; no PII)
    if getattr(settings, "enable_metrics", True):
        try:
            app.add_middleware(MetricsMiddleware)
        except Exception:
            # Do not fail if prometheus client missing
            pass

    # Minimal request logging middleware (no PII): logs path and method at INFO
    @app.middleware("http")
    async def _request_logging_middleware(request: Request, call_next):
        try:
            log = logging.getLogger("sd_onboarding")
            log.info(
                "request",
                extra={
                    "endpoint": request.url.path,
                    "method": request.method,
                },
            )
        except Exception:
            # Do not block requests on logging failures
            pass
        response = await call_next(request)
        return response

    # Initialize tables immediately in dev for test environments (idempotent)
    if not settings.is_prod:
        init_db()

    # Startup: initialize tables in dev (idempotent); migrations will replace later
    @app.on_event("startup")
    async def _startup_init_db() -> None:
        if not settings.is_prod:
            init_db()
        # Log startup message once app is fully initialized
        try:
            log = logging.getLogger("sd_onboarding")
            log.info(
                "api_startup",
                extra={
                    "environment": settings.environment,
                    "version": app.version,
                },
            )
        except Exception:
            pass

        # Emit startup span with version/environment attributes
        try:
            tr = tracer()
            if tr is not None:
                with tr.start_as_current_span("api.startup") as span:
                    # Minimal attributes (no PII)
                    span.set_attributes(
                        {
                            "app.version": app.version,
                            "app.environment": settings.environment,
                        }
                    )
        except Exception:
            # Never fail startup due to tracing
            pass

    # Graceful shutdown: flush tracing
    @app.on_event("shutdown")
    async def _shutdown_tracing() -> None:
        try:
            shutdown_tracing()
        except Exception:
            pass

    # Routers
    app.include_router(ingest_router, prefix="")
    app.include_router(embed_router, prefix="")
    app.include_router(search_router, prefix="")
    app.include_router(analytics_router, prefix="")
    app.include_router(cluster_router, prefix="")
    app.include_router(analysis_router, prefix="")
    app.include_router(history_router, prefix="")
    app.include_router(reports_router, prefix="")
    if getattr(settings, "enable_metrics", True):
        try:
            app.include_router(metrics_router, prefix="")
        except Exception:
            pass

    @app.get("/health", response_class=JSONResponse, tags=["system"])
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "version": app.version})

    return app


app = create_app()