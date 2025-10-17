from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from db.models import AuditLog


class AuditRepository:
    """
    Repository for append-only audit logging.
    """

    @staticmethod
    def record(
        db: Session,
        *,
        subject: str,
        action: str,
        resource: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Persist an audit log entry and return its primary key.

        Notes:
        - Do not include PII in metadata. Keep payload minimal.
        """
        row = AuditLog(
            subject=str(subject),
            action=str(action),
            resource=str(resource),
            metadata_=metadata or None,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return int(row.id)
