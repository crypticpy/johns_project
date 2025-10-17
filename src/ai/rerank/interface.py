from __future__ import annotations

from typing import List, Protocol, Tuple, runtime_checkable


class RerankError(Exception):
    """Raised when rerank backend is unavailable or fails deterministically."""
    pass


@runtime_checkable
class RerankAdapter(Protocol):
    """Adapter protocol for rerank backends."""

    def rerank(self, query: str, candidates: List[Tuple[int, str]]) -> List[Tuple[int, float]]:
        """
        Score and order candidates given a query.

        Args:
            query: The query string to match against.
            candidates: List of (ticket_id, summary_text).

        Returns:
            List of (ticket_id, score) sorted by descending score.
            Scores must be normalized to [0.0, 1.0] for comparability.
        """
        ...
        

__all__ = ["RerankError", "RerankAdapter"]