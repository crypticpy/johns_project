from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Select, and_, select
from sqlalchemy.orm import Session

from db.models import Analysis


class AnalysesRepository:
    """
    Repository for LLM Analyses persistence and history queries.

    Responsibilities:
    - save_analysis: persist a new analysis record and return its ID.
    - list_analyses: paginated listing with optional filters.
    - count_analyses: count analyses matching filters for pagination.
    """

    @staticmethod
    def save_analysis(
        db: Session,
        *,
        dataset_id: int,
        prompt_version: str,
        question: str,
        result_markdown: str,
        ticket_count: int,
        metrics: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """
        Persist an analysis record and return its primary key.

        Args:
            dataset_id: FK to datasets table
            prompt_version: prompt template version used
            question: analysis question or focus
            result_markdown: structured markdown produced by analyzer
            ticket_count: number of tickets considered/sampled
            metrics: sampling/analysis metrics payload (JSON-serializable dict)
            filters: filter context used during analysis (JSON-serializable dict)

        Returns:
            int: newly created analysis ID
        """
        row = Analysis(
            dataset_id=int(dataset_id),
            prompt_version=str(prompt_version),
            question=str(question),
            result_markdown=str(result_markdown),
            metrics=metrics or None,
            ticket_count=int(ticket_count),
            filters=filters or None,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return int(row.id)

    @staticmethod
    def list_analyses(
        db: Session,
        *,
        limit: int = 50,
        offset: int = 0,
        dataset_id: int | None = None,
        prompt_version: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[Analysis]:
        """
        List analyses with optional filters and pagination.

        Filters:
            - dataset_id: exact match
            - prompt_version: exact match
            - date_from/date_to: inclusive range on created_at

        Ordering:
            - created_at DESC, id DESC (stable, deterministic)
        """
        stmt: Select = select(Analysis)

        conditions = []
        if dataset_id is not None:
            conditions.append(Analysis.dataset_id == int(dataset_id))
        if prompt_version:
            conditions.append(Analysis.prompt_version == str(prompt_version))
        if date_from is not None:
            conditions.append(Analysis.created_at >= date_from)
        if date_to is not None:
            conditions.append(Analysis.created_at <= date_to)

        if conditions:
            stmt = stmt.where(and_(*conditions))

        stmt = (
            stmt.order_by(Analysis.created_at.desc(), Analysis.id.desc())
            .limit(int(limit))
            .offset(int(offset))
        )
        return list(db.execute(stmt).scalars().all())

    @staticmethod
    def count_analyses(
        db: Session,
        *,
        dataset_id: int | None = None,
        prompt_version: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> int:
        """
        Count analyses matching the given filters.
        """
        stmt: Select = select(Analysis.id)

        conditions = []
        if dataset_id is not None:
            conditions.append(Analysis.dataset_id == int(dataset_id))
        if prompt_version:
            conditions.append(Analysis.prompt_version == str(prompt_version))
        if date_from is not None:
            conditions.append(Analysis.created_at >= date_from)
        if date_to is not None:
            conditions.append(Analysis.created_at <= date_to)

        if conditions:
            stmt = stmt.where(and_(*conditions))

        # Scalar count via len of IDs list to retain SQL portability without count(*)
        ids = [int(row[0]) for row in db.execute(stmt).all()]
        return len(ids)
