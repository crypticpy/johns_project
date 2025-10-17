"""
Rerank adapters package.

Modules:
- interface: Protocol and error type for rerank adapters
- builtin_lexical: Deterministic offline lexical reranker
- cross_encoder: Optional HuggingFace cross-encoder backend (guarded)
- factory: Selector for rerank backend based on env or explicit arg
"""

# Import submodules so names exist for linters and explicit exports
from . import builtin_lexical as builtin_lexical
from . import cross_encoder as cross_encoder
from . import factory as factory
from . import interface as interface

__all__ = ["interface", "builtin_lexical", "cross_encoder", "factory"]
