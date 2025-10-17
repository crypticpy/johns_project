# pylint: disable=import-error,no-name-in-module
import sys
from pathlib import Path

# Ensure 'src' is importable for test modules (without installing the package)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pytest

from ai.llm.tools.registry import ToolRegistry
from ai.llm.tools.registry import ToolContext


@pytest.mark.unit
def test_search_nn_invalid_args_validation_error(registry: ToolRegistry, ctx_viewer: ToolContext):
    """
    Invalid args for tool.search.nn should return category='validation_error'.
    Cases:
      - Empty query_text
      - k <= 0
    """
    # Empty query_text
    res1 = registry.execute(
        name="search.nn",
        args={"dataset_id": 1, "query_text": "", "k": 5},
        context=ctx_viewer,
    )
    assert "error" in res1, f"expected error payload, got: {res1}"
    assert res1["error"]["category"] == "validation_error"

    # k <= 0 (invalid)
    res2 = registry.execute(
        name="search.nn",
        args={"dataset_id": 1, "query_text": "reset", "k": 0},
        context=ctx_viewer,
    )
    assert "error" in res2
    assert res2["error"]["category"] == "validation_error"


@pytest.mark.unit
def test_analysis_run_missing_required_field_validation_error(registry: ToolRegistry, ctx_analyst: ToolContext):
    """
    Missing required 'question' for tool.analysis.run should return category='validation_error'.
    """
    res = registry.execute(
        name="analysis.run",
        args={"dataset_id": 1, "prompt_version": "v1", "analyzer_backend": "offline"},
        context=ctx_analyst,
    )
    assert "error" in res, f"expected error payload, got: {res}"
    assert res["error"]["category"] == "validation_error"