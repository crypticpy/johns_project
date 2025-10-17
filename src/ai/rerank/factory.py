from __future__ import annotations

import os
from typing import Literal, Optional

from ai.rerank.interface import RerankAdapter, RerankError


def _normalize_backend(val: Optional[str]) -> Optional[Literal["builtin", "cross-encoder"]]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("builtin", "lexical"):
        return "builtin"
    if s in ("cross-encoder", "crossencoder", "cross_encoder"):
        return "cross-encoder"
    return None


def select_reranker(backend: Optional[Literal["builtin", "cross-encoder"]] = None) -> RerankAdapter:
    """
    Select and instantiate a reranker adapter.

    Priority:
    1) explicit backend arg when provided
    2) environment APP_RERANK_BACKEND (default: "builtin")

    Returns:
        RerankAdapter instance.

    Raises:
        RerankError: when requested backend is unavailable or offline-guarded.
        ValueError: when an invalid backend is provided.
    """
    env_backend = _normalize_backend(os.environ.get("APP_RERANK_BACKEND", "builtin"))
    choice = backend or env_backend or "builtin"

    if choice == "builtin":
        # Import locally to avoid circular imports and keep linters happy
        from ai.rerank.builtin_lexical import BuiltinLexicalReranker

        return BuiltinLexicalReranker()

    if choice == "cross-encoder":
        from ai.rerank.cross_encoder import CrossEncoderReranker

        # Let CrossEncoderReranker handle offline guards and dependency checks
        return CrossEncoderReranker()

    raise ValueError(f"Unknown rerank backend: {backend!r}")


__all__ = ["select_reranker", "RerankError", "RerankAdapter"]