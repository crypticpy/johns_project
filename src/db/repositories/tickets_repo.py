from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd
from sqlalchemy import Select, and_, select
from sqlalchemy.orm import Session

from db.models import Ticket


class TicketsRepository:
    """
    Repository for Ticket entities.

    - bulk_insert: insert tickets for a dataset from a normalized DataFrame.
    - query_filtered: retrieve tickets by optional filters.
    """

    @staticmethod
    def _to_int_or_none(value: object) -> Optional[int]:
        try:
            if value is None:
                return None
            if isinstance(value, (int,)):
                return value
            if isinstance(value, float):
                # pandas may provide NaN
                if pd.isna(value):
                    return None
                return int(value)
            s = str(value).strip()
            if s == "" or s.lower() == "nan":
                return None
            return int(float(s))
        except Exception:
            return None

    @staticmethod
    def bulk_insert(
        db: Session,
        *,
        dataset_id: int,
        df: pd.DataFrame,
        normalized_texts: Optional[List[Optional[str]]] = None,
        batch_size: int = 1000,
    ) -> int:
        """
        Insert tickets from a DataFrame that already has canonical columns:
          - Department
          - Assignment Group
          - extract_product
          - summarize_ticket
          - ticket_quality
          - resolution_complexity
          - Reassignment group count tracking_index

        normalized_texts: optional list (len == len(df)) to populate Ticket.normalized_text.
        Returns number of inserted rows.
        """
        rows: List[Ticket] = []
        n = len(df)
        if normalized_texts is not None and len(normalized_texts) != n:
            raise ValueError("normalized_texts length must match DataFrame length")

        # Use dict records to avoid pandas itertuples name-mangling issues with spaces/casing
        records = df.to_dict(orient="records")
        for i, rec in enumerate(records):
            department = rec.get("Department")
            assignment_group = rec.get("Assignment Group") or rec.get("Assignment_Group")
            product = rec.get("extract_product")
            summary = rec.get("summarize_ticket")
            quality = rec.get("ticket_quality")
            complexity = rec.get("resolution_complexity")
            reassignment = rec.get("Reassignment group count tracking_index")
            if reassignment is None:
                reassignment = rec.get("Reassignment_group_count_tracking_index")

            ticket = Ticket(
                dataset_id=dataset_id,
                department=None if (department is None or pd.isna(department)) else str(department),
                assignment_group=None if (assignment_group is None or pd.isna(assignment_group)) else str(assignment_group),
                product=None if (product is None or pd.isna(product)) else str(product),
                summary=None if (summary is None or pd.isna(summary)) else str(summary),
                quality=None if (quality is None or pd.isna(quality)) else str(quality),
                complexity=None if (complexity is None or pd.isna(complexity)) else str(complexity),
                reassignment_count=TicketsRepository._to_int_or_none(reassignment),
                normalized_text=(
                    None
                    if normalized_texts is None
                    else (normalized_texts[i] if normalized_texts[i] else None)
                ),
            )
            rows.append(ticket)

            if len(rows) >= batch_size:
                db.bulk_save_objects(rows)
                db.commit()
                rows.clear()

        if rows:
            db.bulk_save_objects(rows)
            db.commit()
            rows.clear()

        return n

    @staticmethod
    def query_filtered(
        db: Session,
        *,
        dataset_id: Optional[int] = None,
        departments: Optional[Iterable[str]] = None,
        assignment_groups: Optional[Iterable[str]] = None,
        products: Optional[Iterable[str]] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[Ticket]:
        """
        Fetch tickets using optional filters. All filters are combined with AND.
        """
        stmt: Select = select(Ticket)

        conditions = []
        if dataset_id is not None:
            conditions.append(Ticket.dataset_id == dataset_id)
        if departments:
            conditions.append(Ticket.department.in_(list(departments)))
        if assignment_groups:
            conditions.append(Ticket.assignment_group.in_(list(assignment_groups)))
        if products:
            conditions.append(Ticket.product.in_(list(products)))

        if conditions:
            stmt = stmt.where(and_(*conditions))

        stmt = stmt.limit(limit).offset(offset)

        return list(db.execute(stmt).scalars().all())