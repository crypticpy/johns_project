from __future__ import annotations

from typing import Protocol, runtime_checkable


class AnalyzerError(RuntimeError):
    """
    Raised when an analyzer backend fails, is unavailable, or misconfigured.

    - Do not include secrets in error messages.
    - Map provider-specific errors to this type in adapters.
    """


@runtime_checkable
class AnalyzerAdapter(Protocol):
    """
    Adapter interface for LLM analysis backends.

    Contract:
    - Engine code must remain pure; adapters may perform I/O (network/FS).
    - Implementations must be production-capable (no mocks).
    - Deterministic behavior is required for offline analyzer backend.

    Args:
        context: Prompt context constructed from stratified sampling and summaries
        question: User question or analysis focus
        prompt_version: Version identifier of the prompt template to use
        comparison_mode: When True, perform comparison-oriented analysis

    Returns:
        Structured markdown string as the analysis result
    """

    def analyze(self, context: str, question: str, prompt_version: str, comparison_mode: bool = False) -> str:
        ...