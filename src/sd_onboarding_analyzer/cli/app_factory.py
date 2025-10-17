from __future__ import annotations

import os
from fastapi import FastAPI

# Import the application factory from the API package
from api.main import create_app


def _ensure_offline_defaults() -> None:
    """
    Ensure offline-deterministic defaults for embeddings, rerank, and analyzer backends.

    Defaults (only applied when env vars are missing):
    - APP_EMBED_BACKEND=builtin           (deterministic, no network)
    - APP_RERANK_BACKEND=builtin          (deterministic, no network)
    - ANALYZER_BACKEND=offline            (deterministic, no network)
    - TRANSFORMERS_OFFLINE=1              (prevents model downloads during tests)
    - HF_HUB_OFFLINE=1                    (prevents HuggingFace hub access)

    These defaults make CLI/tests stable in CI and local dev without network access.
    They can be overridden by explicitly setting the environment variables.
    """
    os.environ.setdefault("APP_EMBED_BACKEND", "builtin")
    os.environ.setdefault("APP_RERANK_BACKEND", "builtin")
    os.environ.setdefault("ANALYZER_BACKEND", "offline")
    # Guard against accidental network calls from transformers-backed components
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # Force RBAC off for CLI/TestClient-created apps to avoid token requirements in offline flows.
    # RBAC tests call api.main.create_app() directly and set APP_ENABLE_RBAC=true explicitly.
    os.environ["APP_ENABLE_RBAC"] = "false"


def get_app() -> FastAPI:
    """
    Create and return a FastAPI app instance for in-process CLI/TestClient usage.

    - Applies offline-safe environment defaults before app creation.
    - Delegates to the application factory at
      api.main.create_app()

    Returns:
        FastAPI: initialized application instance
    """
    _ensure_offline_defaults()
    return create_app()


__all__ = ["get_app"]