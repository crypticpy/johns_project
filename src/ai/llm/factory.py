from __future__ import annotations

import os
from typing import Literal, Optional

from ai.llm.interface import AnalyzerAdapter, AnalyzerError
from ai.llm.offline_analyzer import OfflineAnalyzer
from config.env import get_settings


def _env_backend_default() -> str:
    """
    Resolve analyzer backend default from environment.

    ANALYZER_BACKEND:
      - "openai" | "offline"
      - Default: "offline" (safer for CI/offline tests)
    """
    val = os.environ.get("ANALYZER_BACKEND", "").strip().lower()
    return val or "offline"


def _has_openai_creds() -> bool:
    """
    Detect presence of OpenAI or Azure OpenAI credentials using settings first, then environment,
    without revealing their values.
    """
    try:
        s = get_settings()
        if getattr(s, "openai_api_key", None):
            return True
        if getattr(s, "azure_openai_endpoint", None) and getattr(s, "azure_openai_key", None):
            return True
    except Exception:
        # Fall back to environment inspection
        pass

    # Standard OpenAI keys
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("APP_OPENAI_API_KEY"):
        return True

    # Azure-style envs (prefer APP_* names)
    azure_endpoint = os.environ.get("APP_AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_key = (
        os.environ.get("APP_AZURE_OPENAI_KEY")
        or os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("AZURE_OPENAI_KEY")
    )
    return bool(azure_endpoint and azure_key)


def _env_flag(name: str, default: bool = False) -> bool:
    """
    Resolve a boolean feature flag from environment in a safe, conservative way.
    Accepts truthy values: '1', 'true', 'yes', 'on'. Missing -> default.
    """
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def select_analyzer(backend: Literal["openai", "offline"] | None = None) -> AnalyzerAdapter:
    """
    Factory for analyzer adapters.

    Selection precedence:
    1) Explicit backend arg
    2) ANALYZER_BACKEND env (default "offline")
    3) If backend is None and OpenAI creds exist -> openai, else offline

    Returns:
        AnalyzerAdapter
    """
    # Prime settings to ensure .env is loaded and config evaluated early
    _ = get_settings()
    chosen = (backend or _env_backend_default()).strip().lower()

    if chosen == "openai":
        # Guardrails: require credentials
        if not _has_openai_creds():
            raise AnalyzerError("OpenAI analyzer requested but credentials are not configured in environment")
        # Import lazily to avoid import-time deps in offline CI
        try:
            from ai.llm.openai_analyzer import OpenAIAnalyzer  # type: ignore
        except Exception as e:
            raise AnalyzerError(f"OpenAI analyzer module unavailable: {e}")
        # Minimal factory wiring: optionally inject ToolRegistry and enable tools when flag is set
        enable_tools = _env_flag("APP_ENABLE_TOOLS", False)
        if enable_tools:
            try:
                # Construct a default registry singleton for this process via dynamic import
                import importlib
                mod = importlib.import_module("ai.llm.tools.registry")
                ToolRegistry = getattr(mod, "ToolRegistry")
                registry = ToolRegistry()  # type: ignore[call-arg]
                analyzer = OpenAIAnalyzer()
                # Minimal wiring without changing constructor signature expectations
                analyzer._registry = registry  # type: ignore[attr-defined]
                analyzer._tools_enabled = True  # type: ignore[attr-defined]
                return analyzer
            except Exception:
                # If registry import/construct fails, fall back to no-tool mode for safety
                return OpenAIAnalyzer()
        # Default: no tool-calling
        return OpenAIAnalyzer()

    # Fallback/default: offline deterministic backend
    return OfflineAnalyzer()


__all__ = ["select_analyzer"]