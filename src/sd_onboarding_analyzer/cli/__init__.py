"""
CLI package for Service Desk Onboarding Analyzer.

Exports convenience helpers for in-process usage and orchestration:
- get_app(): FastAPI app factory with offline-safe defaults
- run_ingest(), run_embed(), run_analysis(), run_report(): step functions
- run_pipeline(): end-to-end orchestration
- to_json(): compact JSON serializer for CLI output
"""

from __future__ import annotations

from fastapi import FastAPI

from .app_factory import get_app
from .run_pipeline import (
    run_ingest,
    run_embed,
    run_analysis,
    run_report,
    run_pipeline,
    to_json,
)

__all__ = [
    "FastAPI",
    "get_app",
    "run_ingest",
    "run_embed",
    "run_analysis",
    "run_report",
    "run_pipeline",
    "to_json",
]