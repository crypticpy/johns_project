import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Set

import pytest

# Make 'src' importable when running tests without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Ensure offline, deterministic environment BEFORE importing app modules
os.environ.setdefault("APP_DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("ANALYZER_BACKEND", "offline")
os.environ.setdefault("APP_ENABLE_RBAC", "true")
os.environ.setdefault("APP_ENABLE_METRICS", "false")
os.environ.setdefault("APP_EMBED_BACKEND", "builtin")
# Keep FAISS disabled to use numpy fallback (stable offline behavior)
os.environ.setdefault("APP_FAISS_ENABLED", "0")
os.environ.setdefault("FAISS_ENABLED", "0")

# Now import DB/session and initialize schema
from db.session import init_db, SessionLocal  # noqa: E402
from ai.llm.tools.registry import (  # noqa: E402
    ToolRegistry,
    ToolContext,
    build_tool_context_from_claims,
)

init_db()


@pytest.fixture(scope="session")
def db_session():
    """Yield a SQLAlchemy Session bound to the in-memory test engine."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session")
def registry() -> ToolRegistry:
    """Fresh ToolRegistry instance for tests."""
    return ToolRegistry()


@pytest.fixture(scope="session")
def ctx_builder():
    """Factory to build ToolContext from a set of roles."""
    def _make(roles: Iterable[str], request_id: str = "test-req") -> ToolContext:
        claims: Dict[str, object] = {
            "sub": "test-user",
            "roles": [str(r).strip().lower() for r in roles],
        }
        return build_tool_context_from_claims(
            claims,
            request_id=request_id,
            token_budget=2000,
            step_limit=8,
            dataset_id=None,
        )
    return _make


@pytest.fixture(scope="session")
def ctx_viewer(ctx_builder) -> ToolContext:
    return ctx_builder({"viewer"})


@pytest.fixture(scope="session")
def ctx_analyst(ctx_builder) -> ToolContext:
    return ctx_builder({"analyst"})


@pytest.fixture(scope="session")
def ctx_admin(ctx_builder) -> ToolContext:
    return ctx_builder({"admin"})


@pytest.fixture(scope="session")
def small_csv_path() -> str:
    """
    Create a tiny CSV file for ingest/upload integration tests.
    Uses canonical columns to match engine expectations.
    """
    data_dir = Path("tests/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "tickets_small.csv"
    if not csv_path.exists():
        lines = [
            "Department,extract_product,summarize_ticket,ticket_quality,resolution_complexity,Reassignment group count tracking_index",
            "IT,Email,Password reset failing on SSO portal,medium,medium,1",
            "HR,Onboarding,New hire cannot access benefits portal,low,low,0",
            "Finance,ERP,Invoice approval workflow stuck,medium,high,2",
            "IT,VPN,Users disconnected intermittently,low,medium,3",
            "Sales,CRM,Lead import duplicates detected,medium,medium,1",
            "Support,Helpdesk,High volume of ticket reassignments,high,high,4",
            "IT,Endpoint,Software install blocked by policy,low,medium,2",
            "HR,Payroll,Overtime not calculated correctly,medium,high,1",
            "Finance,Expense,Receipt OCR misclassification,low,low,0",
            "Support,Knowledge Base,Articles out of date,medium,low,0",
        ]
        csv_path.write_text("\n".join(lines), encoding="utf-8")
    return str(csv_path)