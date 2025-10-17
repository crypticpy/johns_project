# pylint: disable=import-error,no-name-in-module
import sys
from pathlib import Path

# Ensure 'src' is importable for tests
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pytest

import ai.llm.tools.registry as reg_mod  # type: ignore
from ai.llm.tools.registry import ToolContext, ToolRegistry


@pytest.mark.unit
def test_tool_call_count_increments_when_metrics_available(
    registry: ToolRegistry, ctx_viewer: ToolContext
):
    """
    If Prometheus metrics are available, tool_call_count should increment on registry.execute.
    Guard the assertion when counter is not present.
    """
    counter = getattr(reg_mod, "_TOOL_CALL_COUNT", None)
    if counter is None:
        # Metrics disabled or unavailable; nothing to assert
        return

    # Read initial count via label usage (labels: tool_name)
    labels = counter.labels(tool_name="prompts.list")  # type: ignore[attr-defined]
    # Counter from prometheus_client supports ._value.get()
    initial = float(labels._value.get())  # type: ignore[attr-defined]

    # Execute a few tool calls
    for _ in range(3):
        res = registry.execute("prompts.list", {}, ctx_viewer)
        assert "error" not in res, f"unexpected error from prompts.list: {res}"

    final = float(labels._value.get())  # type: ignore[attr-defined]
    assert (
        final - initial == 3
    ), f"Expected counter to increase by 3, initial={initial}, final={final}"
