from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import Select, and_, func, select
from sqlalchemy.orm import Session

from db.models import (
    ClusterAssignment,
    ClusterMetric,
    ClusterRun,
    ClusterTerm,
    Ticket,
)


class ClustersRepository:
    """
    Repository for clustering runs, assignments, metrics, and terms.

    Responsibilities:
    - Create clustering runs with parameters (idempotency at higher level).
    - Persist per-ticket assignments and per-cluster top terms.
    - Store run-level metrics (e.g., silhouette).
    - Query latest run id and fetch summaries for reporting.
    """

    @staticmethod
    def create_run(
        db: Session,
        dataset_id: int,
        model_name: str,
        algorithm: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> int:
        run = ClusterRun(
            dataset_id=int(dataset_id),
            model_name=str(model_name),
            algorithm=str(algorithm),
            params=params or {},
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return int(run.id)

    @staticmethod
    def store_assignments(
        db: Session,
        run_id: int,
        assignments: Iterable[Tuple[int, int]],
        batch_size: int = 1000,
    ) -> int:
        """
        Persist (ticket_id, cluster_id) pairs for a run. Returns number of rows inserted.
        Unique constraint on (run_id, ticket_id) prevents duplication.
        """
        rows: List[ClusterAssignment] = []
        inserted = 0
        for ticket_id, cluster_id in assignments:
            rows.append(
                ClusterAssignment(
                    run_id=int(run_id),
                    ticket_id=int(ticket_id),
                    cluster_id=int(cluster_id),
                )
            )
            if len(rows) >= batch_size:
                db.bulk_save_objects(rows)
                db.commit()
                inserted += len(rows)
                rows.clear()
        if rows:
            db.bulk_save_objects(rows)
            db.commit()
            inserted += len(rows)
            rows.clear()
        return inserted

    @staticmethod
    def store_metrics(db: Session, run_id: int, metrics: Dict[str, Any]) -> int:
        """
        Store metrics for a run. Values are wrapped into {'value': ...} for JSON stability.
        Returns number of metrics rows inserted.
        """
        if not metrics:
            return 0
        rows: List[ClusterMetric] = []
        for name, val in metrics.items():
            rows.append(
                ClusterMetric(
                    run_id=int(run_id),
                    name=str(name),
                    value={"value": val} if val is not None else {"value": None},
                )
            )
        db.bulk_save_objects(rows)
        db.commit()
        return len(rows)

    @staticmethod
    def store_top_terms(
        db: Session,
        run_id: int,
        top_terms: Dict[int, List[Tuple[str, float]]],
        batch_size: int = 1000,
    ) -> int:
        """
        top_terms: {cluster_id -> [(term, score), ...]}
        Returns number of term rows inserted.
        """
        if not top_terms:
            return 0
        rows: List[ClusterTerm] = []
        inserted = 0
        for cid, pairs in top_terms.items():
            for term, score in pairs:
                rows.append(
                    ClusterTerm(
                        run_id=int(run_id),
                        cluster_id=int(cid),
                        term=str(term),
                        score=float(score),
                    )
                )
                if len(rows) >= batch_size:
                    db.bulk_save_objects(rows)
                    db.commit()
                    inserted += len(rows)
                    rows.clear()
        if rows:
            db.bulk_save_objects(rows)
            db.commit()
            inserted += len(rows)
            rows.clear()
        return inserted

    @staticmethod
    def get_latest_run(
        db: Session,
        dataset_id: int,
        model_name: str,
        algorithm: str,
    ) -> Optional[int]:
        stmt: Select = (
            select(ClusterRun.id)
            .where(
                and_(
                    ClusterRun.dataset_id == int(dataset_id),
                    ClusterRun.model_name == str(model_name),
                    ClusterRun.algorithm == str(algorithm),
                )
            )
            .order_by(ClusterRun.created_at.desc(), ClusterRun.id.desc())
            .limit(1)
        )
        row = db.execute(stmt).first()
        if not row:
            return None
        return int(row[0])

    @staticmethod
    def fetch_run_summary(db: Session, run_id: int) -> Dict[str, Any]:
        """
        Return summary payload with silhouette (if present) and cluster counts:
          {
            "run_id": int,
            "silhouette": float | None,
            "cluster_counts": {cluster_id: count, ...},
            "total_assigned": int
          }
        """
        # Silhouette metric
        sil_stmt: Select = select(ClusterMetric.value).where(
            and_(ClusterMetric.run_id == int(run_id), ClusterMetric.name == "silhouette")
        )
        sil_row = db.execute(sil_stmt).first()
        silhouette: Optional[float] = None
        if sil_row and isinstance(sil_row[0], dict):
            val = sil_row[0].get("value", None)
            if isinstance(val, (int, float)):
                silhouette = float(val)

        # Cluster counts
        counts_stmt: Select = (
            select(ClusterAssignment.cluster_id, func.count(ClusterAssignment.id))
            .where(ClusterAssignment.run_id == int(run_id))
            .group_by(ClusterAssignment.cluster_id)
        )
        counts_rows = db.execute(counts_stmt).all()
        cluster_counts: Dict[int, int] = {int(cid): int(cnt) for cid, cnt in counts_rows}
        total_assigned = sum(cluster_counts.values())

        return {
            "run_id": int(run_id),
            "silhouette": silhouette,
            "cluster_counts": cluster_counts,
            "total_assigned": int(total_assigned),
        }