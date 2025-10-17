from __future__ import annotations

"""
SentenceTransformers embedder implementation with clear error messages and
strict typing suitable for offline/CI environments.
"""

import os
from typing import List

import numpy as np
from ai.embeddings.interface import EmbeddingsAdapter, EmbeddingError

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover
    raise EmbeddingError(
        f"sentence-transformers is not installed or failed to import: {e}"
    ) from e


def _map_model_name(model: str) -> str:
    """
    Map generic or non-ST model names to a sensible SentenceTransformers model.

    If an OpenAI-style name is provided (e.g., 'text-embedding-3-large'), default to
    'sentence-transformers/all-MiniLM-L6-v2' which is a widely available local model.
    """
    m = (model or "").strip()
    if not m:
        return "sentence-transformers/all-MiniLM-L6-v2"
    lower = m.lower()
    if "text-embedding" in lower or "openai" in lower:
        return "sentence-transformers/all-MiniLM-L6-v2"
    return m


class SentenceTransformersEmbedder(EmbeddingsAdapter):
    """
    Sentence-Transformers-based embedder with batching and CPU default.

    Behavior:
    - Uses CPU by default to avoid GPU dependency in CI/dev.
    - Honors offline mode: if TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE are set and model is not cached,
      raises a clear EmbeddingError instructing to pre-download the model.
    - Returns unit-length vectors (normalize_embeddings=True).
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device = device

    def _load_model(self, model_name: str) -> SentenceTransformer:
        resolved = _map_model_name(model_name)

        # Detect offline mode envs
        offline = (
            os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            or os.environ.get("HF_HUB_OFFLINE") == "1"
        )
        try:
            # Attempt to load model; if offline and not cached, ST will throw OSError/ConnectionError.
            return SentenceTransformer(resolved, device=self._device)
        except Exception as e:
            if offline:
                raise EmbeddingError(
                    (
                        f"Model '{resolved}' not available in local cache while offline. "
                        "Please pre-download the SentenceTransformers model or "
                        "switch backend to 'builtin'. "
                        f"Original error: {e}"
                    )
                ) from e
            # Online or unknown error: surface a clear message
            raise EmbeddingError(
                f"Failed to load SentenceTransformers model '{resolved}': {e}"
            ) from e

    def embed_texts(self, items: List[str], model: str, batch_size: int) -> List[List[float]]:
        if items is None:
            raise EmbeddingError("items must not be None")
        # Defensive batch sizing
        bs = max(1, min(batch_size or 32, 256))

        st_model = self._load_model(model)

        # SentenceTransformers can encode all items at once with internal batching
        try:
            result = st_model.encode(
                [x or "" for x in items],
                batch_size=bs,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # ensure unit-length vectors
            )
            # Guarantee numpy ndarray type for typing and downstream code
            emb: np.ndarray = np.asarray(result)
        except Exception as e:
            raise EmbeddingError(f"SentenceTransformers encode failed: {e}") from e

        # Convert to list of lists[float]
        return [row.astype(float).tolist() for row in emb]
