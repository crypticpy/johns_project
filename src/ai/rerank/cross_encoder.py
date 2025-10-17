from __future__ import annotations

import os
from typing import List, Tuple, Optional

from ai.rerank.interface import RerankAdapter, RerankError


def _truthy(val: Optional[str]) -> bool:
    if val is None:
        return False
    return str(val).strip().lower() in ("1", "true", "yes", "on")


class CrossEncoderReranker(RerankAdapter):
    """
    Optional HuggingFace sentence-transformers CrossEncoder backend.

    Constraints:
    - CPU device only
    - Strictly offline-safe: if offline flags are set or deps unavailable, raises RerankError
    - Batch scoring for candidates
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> None:
        # Guard against offline mode
        if _truthy(os.environ.get("TRANSFORMERS_OFFLINE")) or _truthy(os.environ.get("HF_HUB_OFFLINE")):
            raise RerankError(
                "CrossEncoder backend disabled in offline mode (TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE set). "
                "Use builtin reranker or unset offline flags if a local model is available."
            )

        # Lazy imports with clear error messages
        try:
            # Prefer sentence-transformers CrossEncoder
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
            except Exception as e_st:
                # Fallback: alt import path
                try:
                    from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore
                except Exception:
                    raise e_st
        except Exception as e:
            raise RerankError(f"sentence-transformers not available for CrossEncoder backend: {e}") from e

        # Torch is usually required by sentence-transformers
        try:
            import torch  # type: ignore
        except Exception as e:
            raise RerankError(f"PyTorch is required for CrossEncoder backend: {e}") from e

        # Initialize model with Sigmoid activation to ensure [0, 1] normalized scores
        try:
            self._model = CrossEncoder(model_name, activation_fn=torch.nn.Sigmoid(), device="cpu")
        except Exception as e:
            raise RerankError(f"Failed to initialize CrossEncoder model '{model_name}': {e}") from e

    def rerank(self, query: str, candidates: List[Tuple[int, str]]) -> List[Tuple[int, float]]:
        if not candidates:
            return []
        pairs = [(query or "", c if isinstance(c, str) else "") for _tid, c in candidates]
        try:
            scores = self._model.predict(pairs)  # type: ignore[attr-defined]
        except Exception as e:
            raise RerankError(f"CrossEncoder scoring failed: {e}") from e

        # Ensure python floats and normalization bounds [0,1]
        out: List[Tuple[int, float]] = []
        for (tid, _), s in zip(candidates, list(scores)):
            try:
                sf = float(s)
            except Exception:
                sf = 0.0
            if sf < 0.0:
                sf = 0.0
            elif sf > 1.0:
                sf = 1.0
            out.append((int(tid), sf))

        # Sort by score desc; stable tie-break by ascending ticket_id for determinism
        out.sort(key=lambda x: (-x[1], x[0]))
        return out


__all__ = ["CrossEncoderReranker"]