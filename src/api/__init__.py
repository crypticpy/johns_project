"""
API package initialization for the modular Service Desk Onboarding Analyzer.
Exposes the FastAPI application factory and default app for convenience.
"""
from .main import create_app, app  # noqa: F401