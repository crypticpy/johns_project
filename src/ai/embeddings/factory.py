from __future__ import annotations

import os
from typing import Literal

from ai.embeddings.builtin_embedder import BuiltinEmbedder
from ai.embeddings.interface import EmbeddingsAdapter
from ai.embeddings.sentence_transformers_embedder import SentenceTransformersEmbedder
from config.settings import get_settings


def _resolve_backend() -> str:
    """
    Determine embedding backend from environment, defaulting to sentence-transformers.
    """
    backend = os.environ.get("APP_EMBED_BACKEND", "").strip().lower()
    return backend or "sentence-transformers"


def select_embedder(
    backend: Literal["sentence-transformers", "builtin"] | None = None,
) -> EmbeddingsAdapter:
    """
    Factory for embeddings adapters.

    - Reads backend from APP_EMBED_BACKEND when not provided.
    - Defaults to sentence-transformers for local/dev.
    - Returns a production-capable adapter; no mocks.

    Returns:
        EmbeddingsAdapter instance
    """
    chosen = (backend or _resolve_backend()).strip().lower()
    if chosen == "builtin":
        return BuiltinEmbedder()
    if chosen == "sentence-transformers":
        # Default to CPU device for portability
        return SentenceTransformersEmbedder(device="cpu")
    # Fallback: be explicit to avoid silent misconfigurations
    raise ValueError(f"Unsupported embedding backend: {chosen}")


def resolve_defaults() -> tuple[str, int]:
    """
    Resolve model name and batch size defaults from application settings.

    Note: settings currently only define the default model; batch size is set to a sane default (64).
    """
    settings = get_settings()
    model_name = settings.embedding_model
    batch_size = 64
    return model_name, batch_size
