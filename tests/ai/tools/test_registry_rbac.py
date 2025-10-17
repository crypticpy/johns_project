# pylint: disable=import-error,no-name-in-module
import sys
from pathlib import Path

# Ensure 'src' is importable for tests
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pytest

from ai.llm.tools.registry import ToolContext, ToolRegistry


@pytest.mark.unit
def test_analysis_run_rbac_denied_for_viewer(registry: ToolRegistry, ctx_viewer: ToolContext):
    """
    tool.analysis.run requires roles {'analyst','admin'} when RBAC enabled.
    Viewer-only context should be denied.
    """
    res = registry.execute(
        name="analysis.run",
        args={"dataset_id": 1, "question": "Test", "analyzer_backend": "offline"},
        context=ctx_viewer,
    )
    assert "error" in res, f"expected error payload, got: {res}"
    assert res["error"]["category"] == "rbac_denied"


@pytest.mark.unit
def test_prompts_save_rbac_denied_for_analyst(registry: ToolRegistry, ctx_analyst: ToolContext):
    """
    tool.prompts.save requires {'admin'} per registry; analyst-only should be denied.
    """
    res = registry.execute(
        name="prompts.save",
        args={"version": "v_test", "template": "Hello", "metadata": {"note": "x"}},
        context=ctx_analyst,
    )
    assert "error" in res, f"expected error payload, got: {res}"
    assert res["error"]["category"] == "rbac_denied"


@pytest.mark.unit
def test_history_list_allowed_for_viewer(registry: ToolRegistry, ctx_viewer: ToolContext):
    """
    tool.history.list requires {'viewer','admin'}; viewer should succeed.
    """
    res = registry.execute(
        name="history.list",
        args={"limit": 5, "offset": 0},
        context=ctx_viewer,
    )
    assert "error" not in res, f"unexpected error payload: {res}"
    # Basic shape assertions
    assert isinstance(res.get("limit"), int)
    assert isinstance(res.get("offset"), int)
    assert isinstance(res.get("total"), int)
    assert isinstance(res.get("items"), list)
