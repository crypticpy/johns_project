from __future__ import annotations

import os
from typing import Optional

# Optional: lightweight python-dotenv fallback for local dev.
# Pydantic Settings already reads .env via env_file, but this ensures early availability if imported before settings.
try:
    if os.path.exists(".env"):
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(".env")
except Exception:
    # Do not fail if dotenv is not installed; BaseSettings will load .env
    pass

# Re-export Settings and accessor from the existing settings module
from config.settings import AppSettings as Settings, get_settings


def get_azure_openai_endpoint() -> Optional[str]:
    """
    Return Azure OpenAI endpoint or None.
    """
    s = get_settings()
    try:
        return str(s.azure_openai_endpoint) if s.azure_openai_endpoint else None
    except Exception:
        return None


def get_azure_openai_key() -> Optional[str]:
    """
    Return Azure OpenAI API key or None.
    """
    s = get_settings()
    return s.azure_openai_key or None


def get_azure_openai_deployment() -> str:
    """
    Return Azure OpenAI deployment name, defaulting to 'gpt-5'.
    """
    s = get_settings()
    return (s.azure_openai_deployment or "gpt-5").strip()


def get_openai_api_key() -> Optional[str]:
    """
    Return standard OpenAI API key or None.
    """
    s = get_settings()
    return s.openai_api_key or None


def get_openai_model() -> str:
    """
    Return standard OpenAI model name, defaulting to 'gpt-5'.
    """
    s = get_settings()
    # In case older settings lack this field, default to 'gpt-5'
    val = getattr(s, "openai_model", "gpt-5") or "gpt-5"
    return str(val).strip()


def is_tracing_enabled() -> bool:
    """
    Return tracing flag (APP_ENABLE_TRACING), default False.
    """
    s = get_settings()
    return bool(getattr(s, "enable_tracing", False))


__all__ = [
    "Settings",
    "get_settings",
    "get_azure_openai_endpoint",
    "get_azure_openai_key",
    "get_azure_openai_deployment",
    "get_openai_api_key",
    "get_openai_model",
    "is_tracing_enabled",
]