from __future__ import annotations

import hashlib
import math

from ai.embeddings.interface import EmbeddingError, EmbeddingsAdapter


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector; return zero vector unchanged if norm == 0."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    inv = 1.0 / norm
    return [x * inv for x in vec]


def _hash_to_index(token: str, dim: int) -> int:
    """
    Deterministically map a token to an index in [0, dim).

    Uses SHA-256 for stability across Python versions and runs.
    """
    h = hashlib.sha256(token.encode("utf-8")).digest()
    # Take first 8 bytes as big-endian integer for speed; mod by dim
    idx = int.from_bytes(h[:8], byteorder="big", signed=False) % dim
    return idx


def _embed_text(text: str, dim: int) -> list[float]:
    """
    Deterministic embedding:
    - Lowercase, simple whitespace tokenization
    - Bucket tokens into dim-length vector via stable hashing
    - Weight by token length to slightly emphasize informative tokens
    - L2-normalize output
    """
    if not text:
        return [0.0] * dim

    vec = [0.0] * dim
    # Simple, stable tokenization; avoid external deps
    tokens = text.lower().strip().split()
    if not tokens:
        return vec

    for tok in tokens:
        # Basic token cleanup
        token = tok.strip()
        if not token:
            continue
        idx = _hash_to_index(token, dim)
        # Weight contribution by sqrt(length) to damp very long tokens
        weight = math.sqrt(max(1, len(token)))
        vec[idx] += weight

    return _normalize(vec)


class BuiltinEmbedder(EmbeddingsAdapter):
    """
    Deterministic, lightweight embedder for CI/offline use.

    - No network or filesystem access.
    - Stable across runs and environments.
    - Produces unit-length float vectors.

    Model name controls dimension:
    - "builtin-256" -> 256 dims
    - "builtin-384" -> 384 dims (default)
    - "builtin-512" -> 512 dims

    Any other model string falls back to 384 dims.
    """

    _DEFAULT_DIM = 384

    @staticmethod
    def _dim_from_model(model: str) -> int:
        m = (model or "").strip().lower()
        if m == "builtin-256":
            return 256
        if m == "builtin-512":
            return 512
        # Default and "builtin-384"
        return 384

    def embed_texts(self, items: list[str], model: str, batch_size: int) -> list[list[float]]:
        if items is None:
            raise EmbeddingError("items must not be None")
        dim = self._dim_from_model(model)
        # Defensive batch sizing
        bs = max(1, min(int(batch_size) if batch_size else 32, 1024))

        results: list[list[float]] = []
        for i in range(0, len(items), bs):
            batch = items[i : i + bs]
            for text in batch:
                results.append(_embed_text(text or "", dim))
        return results
