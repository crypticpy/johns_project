from __future__ import annotations

# Package init for ai.llm to satisfy import resolution and linters.
# Expose core types without importing optional adapters to avoid dependency cycles.
from .interface import AnalyzerAdapter, AnalyzerError  # noqa: F401
