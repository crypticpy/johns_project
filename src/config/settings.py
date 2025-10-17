from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    Centralized application configuration.

    Environment variables override defaults. This module must be imported only by API startup or
    top-level services. Pure Engine modules should receive concrete values via function arguments.
    """

    # General
    environment: Literal["dev", "prod"] = Field(default="dev", description="Runtime environment")
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins; restrict in production",
    )

    # Storage and vector backends
    database_url: str = Field(
        default="sqlite:///./data/app.db",
        description="SQLAlchemy-style DB URL (sqlite/postgres)",
    )
    vector_backend: Literal["faiss", "milvus", "weaviate"] = Field(
        default="faiss", description="Vector store backend selection"
    )
    embedding_model: str = Field(
        default="text-embedding-3-large", description="Default embedding model name"
    )

    # Prompt templates
    prompt_template_dir: Path = Field(
        default=Path("src/ai/llm/prompts/templates"),
        description="Filesystem path for versioned prompt templates",
    )

    # Observability
    logging_config_path: Path | None = Field(
        default=Path("src/config/logging.yaml"),
        description="Path to logging configuration YAML",
    )
    enable_tracing: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    enable_metrics: bool = Field(default=True, description="Expose Prometheus metrics")

    # Azure/OpenAI configuration (one of these providers may be used)
    azure_openai_endpoint: AnyHttpUrl | None = Field(
        default=None, description="Azure OpenAI endpoint base URL"
    )
    azure_openai_key: str | None = Field(
        default=None, description="Azure OpenAI API key (use env/secret manager)"
    )
    azure_openai_deployment: str = Field(
        default="gpt-5", description="Azure OpenAI deployment name"
    )

    openai_api_key: str | None = Field(
        default=None, description="OpenAI API key (use env/secret manager)"
    )
    openai_model: str = Field(default="gpt-5", description="Default OpenAI chat model name")

    # Security and governance
    enable_rbac: bool = Field(default=False, description="Enable RBAC for admin/history routes")
    audit_log_enabled: bool = Field(default=True, description="Record admin actions in audit log")
    pii_redaction_enabled: bool = Field(
        default=True, description="Redact PII in preprocess pipeline"
    )
    jwt_secret: str | None = Field(
        default=None,
        description="JWT HMAC secret for verifying Bearer tokens (env: APP_JWT_SECRET)",
    )
    jwt_algorithms: list[str] = Field(
        default_factory=lambda: ["HS256"],
        description="Accepted JWT algorithms (e.g., HS256)",
    )

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("allowed_origins")
    @classmethod
    def sanitize_allowed_origins(cls, v: list[str]) -> list[str]:
        return [origin.strip() for origin in v if origin and origin.strip()]

    @field_validator("prompt_template_dir")
    @classmethod
    def ensure_prompt_dir(cls, v: Path) -> Path:
        # Do not create directories here; only validate type and normalize
        return v.resolve() if not v.is_absolute() else v

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v:
            raise ValueError("database_url must not be empty")
        return v

    @field_validator("azure_openai_key", "openai_api_key")
    @classmethod
    def warn_plaintext_secrets(cls, v: str | None) -> str | None:
        # Placeholder for secret scanning; leave as-is, but ensure non-empty strings are trimmed
        if v is None:
            return v
        trimmed = v.strip()
        if not trimmed:
            return None
        return trimmed

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"

    @property
    def cors_origins(self) -> list[str]:
        if self.is_prod and self.allowed_origins == ["*"]:
            # Safety: default deny-all in prod if not configured
            return []
        return self.allowed_origins


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Cached settings accessor for application modules.
    """
    return AppSettings()
