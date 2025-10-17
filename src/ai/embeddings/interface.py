from __future__ import annotations

from typing import List, Protocol, runtime_checkable


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend fails or is unavailable."""


@runtime_checkable
class EmbeddingsAdapter(Protocol):
    """
    Adapter interface for text embeddings backends.

    Contract:
    - Pure Engine modules must not perform I/O; adapters may access network/FS.
    - Implementations should return normalized vectors (unit length) as floats.
    - Batching/backpressure should be handled inside the adapter.

    Args:
        items: list of input strings to embed
        model: model identifier/name (backend-specific)
        batch_size: desired batch size; adapter may clamp to safe bounds

    Returns:
        List of embedding vectors, each a list[float] of equal dimension.
    """

    def embed_texts(self, items: List[str], model: str, batch_size: int) -> List[List[float]]:
        ...