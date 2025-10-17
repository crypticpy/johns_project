from __future__ import annotations

"""
AI Tools Registry package.

Exports:
- ToolSpec: metadata describing a tool contract (name, description, IO models, RBAC, audit flag)
- ToolContext: per-call context (subject, roles, request_id, budgets)
- ToolRegistry: registry and executor with validation, RBAC, audit, and observability
- build_tool_context_from_claims: helper to derive ToolContext fields from JWT claims
"""

from .registry import ToolContext, ToolRegistry, ToolSpec, build_tool_context_from_claims

__all__ = ["ToolSpec", "ToolContext", "ToolRegistry", "build_tool_context_from_claims"]
