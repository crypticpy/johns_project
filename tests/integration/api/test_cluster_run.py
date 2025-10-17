from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from db.models import Base, ClusterAssignment, ClusterTerm
from db.session import engine, SessionLocal
from db.repositories.clusters_repo import ClustersRepository


def _make_csv_bytes() -> bytes:
    # Construct a small DataFrame with canonical and variant columns
    df = pd.DataFrame(
        [
            {
                "Department": "IT",
                "Assignment Group": "Service Desk",
                "extract_product": "Laptop",
                "summarize_ticket": "User cannot login",
                "ticket_quality": "good",
                "resolution_complexity": "low",
                "Reassignment group count tracking_index": 0,
            },
            {
                "Department": "HR",
                "Assignment Group": "Onboarding",
                "extract_product": "HRIS",
                "summarize_ticket": "Provision new employee account",
                "ticket_quality": "average",
                "resolution_complexity": "medium",
                "Reassignment group count tracking_index": 2,
            },
            # Include some variant columns to test normalization behavior
            {
                "department": "Finance",
                "Assignment_Group": "Accounts",
                "Product": "ERP",
                "Summary": "Request access to ERP module",
                "Quality": "poor",
                "Resolution Complexity": "high",
                "Reassignment_group_count_tracking_index": 3,
            },
        ]
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def test_cluster_run_kmeans_persists_assignments_metrics_and_terms_offline():
    # Ensure a clean database for this test run (idempotent)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    app = create_app()
    client = TestClient(app)

    # Upload dataset
    csv_bytes = _make_csv_bytes()
    files = {"file": ("tickets.csv", csv_bytes, "text/csv")}
    resp_ingest = client.post("/ingest/upload", files=files)
    assert resp_ingest.status_code == 200, resp_ingest.text
    payload_ingest = resp_ingest.json()

    dataset_id = int(payload_ingest["dataset_id"])
    row_count = int(payload_ingest["row_count"])
    assert row_count == 3

    # Run embeddings using builtin backend to avoid any network
    req_embed = {
        "dataset_id": dataset_id,
        "backend": "builtin",
        "model_name": "builtin-384",
        "batch_size": 32,
    }
    resp_embed = client.post("/embed/run", json=req_embed)
    assert resp_embed.status_code == 200, resp_embed.text
    payload_embed = resp_embed.json()
    vector_dim = int(payload_embed["vector_dim"])
    assert vector_dim == 384

    # Assert FAISS index files exist to allow model_name inference
    index_dir = Path("data/faiss")
    index_path = index_dir / f"{dataset_id}.index"
    meta_path = index_dir / f"{dataset_id}.meta.json"
    assert index_path.exists(), "FAISS index file missing"
    assert meta_path.exists(), "FAISS meta file missing"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert int(meta.get("dim", 0)) == vector_dim
    assert meta.get("model_name") == "builtin-384"

    # Run clustering (K-Means, n_clusters=2)
    req_cluster = {
        "dataset_id": dataset_id,
        "algorithm": "kmeans",
        "params": {"n_clusters": 2},
    }
    resp_cluster = client.post("/cluster/run", json=req_cluster)
    assert resp_cluster.status_code == 200, resp_cluster.text
    payload_cluster = resp_cluster.json()

    assert payload_cluster["dataset_id"] == dataset_id
    assert payload_cluster["algorithm"] == "kmeans"
    assert payload_cluster["model_name"] in ("builtin-384", meta.get("model_name"))
    run_id_1 = int(payload_cluster["run_id"])
    silhouette = payload_cluster.get("silhouette", None)
    # Silhouette may be a number (float) or None per constraints
    assert (silhouette is None) or isinstance(silhouette, (int, float))

    # Cluster counts sum equals row_count
    counts: Dict[int, int] = payload_cluster.get("cluster_counts", {})
    assert isinstance(counts, dict)
    assert sum(int(v) for v in counts.values()) == row_count

    # Ensure TF-IDF top terms exist for clusters >= 0
    db = SessionLocal()
    try:
        terms: List[ClusterTerm] = list(
            db.query(ClusterTerm).filter(ClusterTerm.run_id == run_id_1).all()  # type: ignore[attr-defined]
        )
        # There should be at least one term recorded
        assert len(terms) >= 1
        # All terms have cluster_id >= 0 and non-empty term strings
        assert all((t.cluster_id >= 0 and isinstance(t.term, str) and t.term.strip() != "") for t in terms)
    finally:
        db.close()

    # Re-run clustering to verify idempotency (new run_id, no duplication constraint violations)
    resp_cluster_2 = client.post("/cluster/run", json=req_cluster)
    assert resp_cluster_2.status_code == 200, resp_cluster_2.text
    payload_cluster_2 = resp_cluster_2.json()
    run_id_2 = int(payload_cluster_2["run_id"])
    assert run_id_2 > run_id_1

    # Repository latest run should match second run_id
    db = SessionLocal()
    try:
        latest = ClustersRepository.get_latest_run(db, dataset_id=dataset_id, model_name="builtin-384", algorithm="kmeans")
        assert int(latest or 0) == run_id_2

        # Each run should have assignments count == row_count
        for run_id in (run_id_1, run_id_2):
            count_rows = (
                db.query(ClusterAssignment)
                .filter(ClusterAssignment.run_id == run_id)  # type: ignore[attr-defined]
                .count()
            )
            assert count_rows == row_count
    finally:
        db.close()


def test_cluster_run_invalid_params_returns_400():
    # Ensure a clean database for this test run (idempotent)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    app = create_app()
    client = TestClient(app)

    # Upload dataset
    csv_bytes = _make_csv_bytes()
    files = {"file": ("tickets.csv", csv_bytes, "text/csv")}
    resp_ingest = client.post("/ingest/upload", files=files)
    assert resp_ingest.status_code == 200, resp_ingest.text
    payload_ingest = resp_ingest.json()

    dataset_id = int(payload_ingest["dataset_id"])

    # Run embeddings
    req_embed = {
        "dataset_id": dataset_id,
        "backend": "builtin",
        "model_name": "builtin-384",
        "batch_size": 32,
    }
    resp_embed = client.post("/embed/run", json=req_embed)
    assert resp_embed.status_code == 200, resp_embed.text

    # Invalid K-Means params: n_clusters = 1 -> 400
    req_cluster_bad = {
        "dataset_id": dataset_id,
        "algorithm": "kmeans",
        "params": {"n_clusters": 1},
    }
    resp_cluster_bad = client.post("/cluster/run", json=req_cluster_bad)
    assert resp_cluster_bad.status_code == 400, resp_cluster_bad.text