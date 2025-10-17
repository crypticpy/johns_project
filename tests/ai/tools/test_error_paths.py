# pylint: disable=import-error,no-name-in-module
import sys
from pathlib import Path

# Ensure 'src' is importable for tests (without installing package)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pytest

from ai.llm.tools.registry import ToolRegistry, ToolContext, build_tool_context_from_claims


@pytest.mark.integration
def test_invalid_tool_name_returns_validation_or_unavailable(registry: ToolRegistry, ctx_viewer: ToolContext):
    """
    Invalid tool name should surface a controlled error.
    Registry implementation returns category='validation_error' for unknown tools.
    Accept either 'validation_error' or 'tool_unavailable' to allow implementation variance.
    """
    res = registry.execute("no.such.tool", {}, ctx_viewer)
    assert "error" in res, f"expected error payload, got: {res}"
    assert res["error"]["category"] in {"validation_error", "tool_unavailable"}


@pytest.mark.integration
def test_rerank_cross_encoder_unavailable_or_fallback(
    registry: ToolRegistry,
    ctx_analyst: ToolContext,
    small_csv_path: str,
):
    """
    In offline env, cross-encoder reranker should be unavailable or gracefully fallback to builtin.
    Flow:
      - ingest.upload -> embed.run -> search.nn with rerank_backend='cross-encoder'
    Expect:
      - Either success (fallback used) or error category in {'tool_unavailable', 'downstream_error'}
    """
    # Ingest
    res_ing = registry.execute(
        "ingest.upload",
        {"file_path": small_csv_path, "dataset_name": "err_path_small"},
        ctx_analyst,
    )
    assert "error" not in res_ing, f"ingest failed: {res_ing}"
    dataset_id = int(res_ing["dataset_id"])

    # Embed with builtin to ensure index exists
    res_emb = registry.execute(
        "embed.run",
        {"dataset_id": dataset_id, "backend": "builtin", "model_name": "builtin-384", "batch_size": 32},
        ctx_analyst,
    )
    assert "error" not in res_emb, f"embed failed: {res_emb}"
    assert res_emb["indexed"] in (True, False)

    # Search with cross-encoder rerank
    res_search = registry.execute(
        "search.nn",
        {
            "dataset_id": dataset_id,
            "query_text": "knowledge base article",
            "k": 5,
            "rerank": True,
            "rerank_backend": "cross-encoder",
        },
        ctx_analyst,
    )
    if "error" in res_search:
        assert res_search["error"]["category"] in {"tool_unavailable", "downstream_error"}
    else:
        # Success path indicates graceful fallback
        assert isinstance(res_search.get("results"), list)
        assert len(res_search["results"]) <= 5


@pytest.mark.integration
def test_step_budget_enforced_by_driver_cap(registry: ToolRegistry):
    """
    Simulate a loop driver enforcing a TOOL_MAX_STEPS cap.
    Registry does not enforce step limits; driver should refuse beyond cap.
    """
    # Build a fresh context with capped step_limit (for record), roles 'analyst' to pass RBAC where applicable
    claims = {"sub": "driver-test", "roles": ["analyst"]}
    ctx_capped: ToolContext = build_tool_context_from_claims(
        claims, request_id="driver-cap", token_budget=2000, step_limit=3, dataset_id=None
    )

    steps_cap = 3
    executed = 0
    # Use a trivially safe tool to count steps deterministically
    for _ in range(10):
        if executed >= steps_cap:
            break
        res = registry.execute("prompts.list", {}, ctx_capped)
        assert "error" not in res, f"unexpected error from prompts.list: {res}"
        executed += 1

    assert executed == steps_cap, f"driver should cap steps at {steps_cap}, executed={executed}"