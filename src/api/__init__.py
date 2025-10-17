"""
API package initialization for the modular Service Desk Onboarding Analyzer.
Exposes the FastAPI application factory and default app for convenience.
"""

from .main import app, create_app  # noqa: F401
