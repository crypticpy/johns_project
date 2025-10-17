# pylint: disable=import-error,no-name-in-module
import sys
from pathlib import Path

# Ensure 'src' is importable for tests
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pytest

from ai.llm.tools.registry import (
    ToolRegistry,
    ToolContext,
    IngestUploadOutput,
    EmbedRunOutput,
    SearchNNOutput,
    ClusterRunOutput,
    AnalysisRunOutput,
    ReportsGetOutput,
)


@pytest.mark.integration
def test_offline_sequence_end_to_end(
    registry: ToolRegistry,
    ctx_analyst: ToolContext,
    small_csv_path: str,
):
    """
    Deterministic offline sequence:
      1) ingest.upload -> dataset_id
      2) embed.run backend=builtin -> embedded_count >= 0, indexed in numpy fallback
      3) search.nn k=5 rerank=builtin -> ordered results length <= k
      4) cluster.run algorithm=kmeans params={"n_clusters": 3} -> run_id, cluster_counts keys length=3 (if enough data)
      5) analysis.run analyzer_backend=offline prompt_version="v1" max_tickets=10 token_budget=2000
      6) reports.get -> report_markdown contains analysis summaries; analysis_count >= 1
    All outputs validated via registry output models.
    """
    # 1) ingest.upload
    res_ing = registry.execute(
        "ingest.upload",
        {"file_path": small_csv_path, "dataset_name": "tickets_small"},
        ctx_analyst,
    )
    assert "error" not in res_ing, f"ingest failed: {res_ing}"
    ingest_out = IngestUploadOutput.model_validate(res_ing)
    dataset_id = int(ingest_out.dataset_id)
    assert dataset_id >= 1
    assert ingest_out.row_count >= 8

    # 2) embed.run (builtin)
    res_emb = registry.execute(
        "embed.run",
        {
            "dataset_id": dataset_id,
            "backend": "builtin",
            "model_name": "builtin-384",
            "batch_size": 32,
        },
        ctx_analyst,
    )
    assert "error" not in res_emb, f"embed failed: {res_emb}"
    emb_out = EmbedRunOutput.model_validate(res_emb)
    assert emb_out.dataset_id == dataset_id
    assert emb_out.embedded_count >= 0
    assert emb_out.vector_dim in (256, 384, 512)

    # 3) search.nn (builtin rerank)
    res_search = registry.execute(
        "search.nn",
        {
            "dataset_id": dataset_id,
            "query_text": "onboarding portal",
            "k": 5,
            "filters": {"department": ["HR"]},
            "rerank": True,
            "rerank_backend": "builtin",
        },
        ctx_analyst,
    )
    assert "error" not in res_search, f"search failed: {res_search}"
    search_out = SearchNNOutput.model_validate(res_search)
    assert search_out.dataset_id == dataset_id
    assert search_out.k == 5
    assert len(search_out.results) <= 5

    # 4) cluster.run (kmeans, k=3)
    res_cluster = registry.execute(
        "cluster.run",
        {"dataset_id": dataset_id, "algorithm": "kmeans", "params": {"n_clusters": 3}},
        ctx_analyst,
    )
    assert "error" not in res_cluster, f"cluster failed: {res_cluster}"
    cluster_out = ClusterRunOutput.model_validate(res_cluster)
    assert cluster_out.dataset_id == dataset_id
    assert cluster_out.algorithm == "kmeans"
    # cluster_counts may have fewer keys if small data; assert 1..3 range
    assert 1 <= len(cluster_out.cluster_counts.keys()) <= 3

    # 5) analysis.run (offline analyzer)
    res_analysis = registry.execute(
        "analysis.run",
        {
            "dataset_id": dataset_id,
            "question": "Where are onboarding gaps?",
            "prompt_version": "v1",
            "analyzer_backend": "offline",
            "max_tickets": 10,
            "token_budget": 2000,
        },
        ctx_analyst,
    )
    assert "error" not in res_analysis, f"analysis failed: {res_analysis}"
    analysis_out = AnalysisRunOutput.model_validate(res_analysis)
    assert analysis_out.dataset_id == dataset_id
    assert analysis_out.ticket_count <= 10
    # created_at optional; if present must be non-empty string
    if analysis_out.created_at is not None:
        assert isinstance(analysis_out.created_at, str)
        assert analysis_out.created_at.strip() != ""

    # 6) reports.get
    res_report = registry.execute(
        "reports.get",
        {"dataset_id": dataset_id},
        ctx_analyst,
    )
    assert "error" not in res_report, f"report failed: {res_report}"
    report_out = ReportsGetOutput.model_validate(res_report)
    assert report_out.dataset_id == dataset_id
    assert isinstance(report_out.report_markdown, str)
    assert len(report_out.report_markdown) > 0
    assert report_out.analysis_count >= 1