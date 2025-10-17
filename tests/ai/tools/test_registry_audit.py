import json
from typing import List

import pytest
from sqlalchemy import select

from ai.llm.tools.registry import ToolRegistry, ToolContext, ReportsGetOutput
from db.models import AuditLog, Dataset, Ticket


@pytest.mark.unit
def test_cluster_run_emits_audit(registry: ToolRegistry, ctx_analyst: ToolContext, small_csv_path: str):
    """
    Execute tool.cluster.run with roles {'analyst'} on a small synthetic dataset.
    After execution, assert an audit entry persisted.
    """
    # 1) ingest/upload
    res_ing = registry.execute(
        "ingest.upload",
        {"file_path": small_csv_path, "dataset_name": "small"},
        ctx_analyst,
    )
    assert "error" not in res_ing, f"ingest failed: {res_ing}"
    dataset_id = int(res_ing["dataset_id"])

    # 2) embed/run
    res_emb = registry.execute(
        "embed.run",
        {"dataset_id": dataset_id, "backend": "builtin", "model_name": "builtin-384", "batch_size": 32},
        ctx_analyst,
    )
    assert "error" not in res_emb, f"embed failed: {res_emb}"
    assert res_emb["embedded_count"] >= 0
    assert res_emb["indexed"] in (True, False)

    # 3) cluster/run (kmeans)
    res_clus = registry.execute(
        "cluster.run",
        {"dataset_id": dataset_id, "algorithm": "kmeans", "params": {"n_clusters": 3}},
        ctx_analyst,
    )
    assert "error" not in res_clus, f"cluster failed: {res_clus}"
    assert res_clus["dataset_id"] == dataset_id
    assert res_clus["algorithm"] == "kmeans"
    assert isinstance(res_clus["cluster_counts"], dict)

    # Verify audit log persisted
    from db.session import SessionLocal

    db = SessionLocal()
    try:
        rows = list(
            db.execute(
                select(AuditLog).where(
                    AuditLog.action == "cluster.run",
                    AuditLog.resource == f"dataset:{dataset_id}",
                )
            ).scalars().all()
        )
        assert len(rows) >= 1, "expected at least one audit row for cluster.run"
        meta = rows[-1].metadata_ or {}
        assert isinstance(meta, dict)
        assert int(meta.get("run_id", 0)) == int(res_clus["run_id"])
        assert str(meta.get("model_name", "")).strip() != ""
    finally:
        db.close()


@pytest.mark.unit
def test_prompts_save_emits_audit_and_metadata_sane(registry: ToolRegistry, ctx_admin: ToolContext):
    """
    tool.prompts.save with roles {'admin'} should emit an audit entry; metadata contains version and excludes secrets.
    """
    version = "v_test_audit"
    res = registry.execute(
        "prompts.save",
        {"version": version, "template": "Hello World", "metadata": {"note": "audit-check"}},
        ctx_admin,
    )
    assert "error" not in res, f"prompts.save failed: {res}"
    assert bool(res.get("ok", False)) is True

    from db.session import SessionLocal

    db = SessionLocal()
    try:
        rows = list(
            db.execute(
                select(AuditLog).where(
                    AuditLog.action == "prompts.save",
                    AuditLog.resource == f"prompt:{version}",
                )
            ).scalars().all()
        )
        assert len(rows) >= 1, "expected at least one audit row for prompts.save"
        meta = rows[-1].metadata_ or {}
        assert isinstance(meta, dict)
        assert meta.get("version") == version
        # Sanity: metadata must not include secrets keys
        for key in ("openai_api_key", "azure_openai_key", "jwt_secret"):
            assert key not in meta
    finally:
        db.close()