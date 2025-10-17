from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db.models import Dataset, Ticket


class DatasetsRepository:
    """
    Repository for Dataset entities.

    Responsibilities:
    - Unique guard on file_hash for idempotent dataset creation.
    - Lookup by file_hash.
    - Recompute department_count from persisted tickets.
    """

    @staticmethod
    def get_by_hash(db: Session, file_hash: str) -> Optional[Dataset]:
        return (
            db.execute(select(Dataset).where(Dataset.file_hash == file_hash))
            .scalars()
            .first()
        )

    @staticmethod
    def create_or_get(
        db: Session,
        *,
        name: str,
        file_hash: str,
        row_count: int,
        department_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> Dataset:
        """
        Create a dataset row if one does not already exist for the given file_hash.
        Returns the existing row when re-uploading the same file.
        """
        existing = DatasetsRepository.get_by_hash(db, file_hash)
        if existing is not None:
            return existing

        ds = Dataset(
            name=name,
            file_hash=file_hash,
            row_count=row_count,
            department_count=department_count,
            metadata_=metadata or None,
        )
        db.add(ds)
        try:
            db.commit()
        except IntegrityError:
            # Another concurrent request may have inserted the same hash; fetch it.
            db.rollback()
            existing = DatasetsRepository.get_by_hash(db, file_hash)
            if existing is not None:
                return existing
            raise

        db.refresh(ds)
        return ds

    @staticmethod
    def recompute_department_count(db: Session, dataset_id: int) -> int:
        """
        Compute distinct non-null department count from tickets for the dataset and persist it.
        """
        count: int = int(
            db.execute(
                select(func.count(func.distinct(Ticket.department))).where(
                    Ticket.dataset_id == dataset_id, Ticket.department.is_not(None)
                )
            ).scalar_one()
            or 0
        )

        ds = db.get(Dataset, dataset_id)
        if ds is not None:
            ds.department_count = count
            db.add(ds)
            db.commit()
            db.refresh(ds)

        return count