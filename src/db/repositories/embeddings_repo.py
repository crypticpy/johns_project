from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from sqlalchemy import Select, and_, select
from sqlalchemy.orm import Session

from db.models import Embedding, Ticket


def _vector_to_blob(vec: Sequence[float]) -> bytes:
    """
    Encode a vector as float32 bytes for compact storage.
    """
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes(order="C")


def _blob_to_vector(blob: bytes) -> list[float]:
    """
    Decode float32 bytes back to a Python list[float].
    """
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr.astype(float).tolist()


class EmbeddingsRepository:
    """
    Repository for per-ticket embeddings.

    Responsibilities:
    - Idempotent upsert for (dataset_id, ticket_id, model_name).
    - Bulk fetch of vectors by dataset and model.
    - Existence checks for dataset/model embeddings.
    """

    @staticmethod
    def exists_for_dataset(db: Session, dataset_id: int, model_name: str) -> bool:
        stmt: Select = (
            select(Embedding.id)
            .where(and_(Embedding.dataset_id == dataset_id, Embedding.model_name == model_name))
            .limit(1)
        )
        return db.execute(stmt).first() is not None

    @staticmethod
    def fetch_by_dataset(
        db: Session, dataset_id: int, model_name: str
    ) -> tuple[list[int], list[list[float]]]:
        """
        Return (ids, vectors) where ids are Ticket IDs for the dataset/model.
        """
        stmt: Select = select(Embedding.ticket_id, Embedding.vector).where(
            and_(Embedding.dataset_id == dataset_id, Embedding.model_name == model_name)
        )
        rows = db.execute(stmt).all()
        ids: list[int] = []
        vectors: list[list[float]] = []
        for ticket_id, blob in rows:
            ids.append(int(ticket_id))
            vectors.append(_blob_to_vector(blob))
        return ids, vectors

    @staticmethod
    def upsert_for_dataset(
        db: Session,
        dataset_id: int,
        model_name: str,
        records: Iterable[tuple[int, Sequence[float]]],
        batch_size: int = 1000,
    ) -> dict:
        """
        Idempotently insert embeddings for the given dataset and model.

        - Skips existing (dataset_id, ticket_id, model_name) rows.
        - Inserts only missing rows in batches.
        - Returns counts: {"inserted": int, "skipped": int}

        Notes:
        - This method avoids duplicate rows without relying on DB-specific UPSERT syntax,
          ensuring portability across SQL backends configured for this app.
        """
        # Collect candidate ticket_ids and map for quick lookup
        recs = list(records)
        if not recs:
            return {"inserted": 0, "skipped": 0}

        ticket_ids = [int(tid) for tid, _ in recs]

        # Determine which tickets already have embeddings for this dataset/model
        existing_stmt: Select = select(Embedding.ticket_id).where(
            and_(
                Embedding.dataset_id == dataset_id,
                Embedding.model_name == model_name,
                Embedding.ticket_id.in_(ticket_ids),
            )
        )
        existing_ids_set = {int(row[0]) for row in db.execute(existing_stmt).all()}

        to_insert: list[Embedding] = []
        skipped = 0
        for ticket_id, vec in recs:
            tid_int = int(ticket_id)
            if tid_int in existing_ids_set:
                skipped += 1
                continue
            blob = _vector_to_blob(vec)
            to_insert.append(
                Embedding(
                    dataset_id=dataset_id,
                    ticket_id=tid_int,
                    model_name=model_name,
                    vector=blob,
                )
            )
            if len(to_insert) >= batch_size:
                db.bulk_save_objects(to_insert)
                db.commit()
                to_insert.clear()

        inserted = 0
        if to_insert:
            db.bulk_save_objects(to_insert)
            db.commit()
            inserted += len(to_insert)
            to_insert.clear()

        return {"inserted": inserted, "skipped": skipped}

    @staticmethod
    def fetch_candidate_texts(
        db: Session, dataset_id: int, limit: int = 100_000
    ) -> tuple[list[int], list[str]]:
        """
        Helper to retrieve candidate ticket texts for embedding:
        - Use normalized_text if present; otherwise fall back to summary.
        Returns aligned (ticket_ids, texts).
        """
        stmt: Select = (
            select(Ticket.id, Ticket.normalized_text, Ticket.summary)
            .where(Ticket.dataset_id == dataset_id)
            .limit(limit)
        )
        ids: list[int] = []
        texts: list[str] = []
        for tid, norm, summ in db.execute(stmt).all():
            ids.append(int(tid))
            text = (norm or summ or "").strip()
            texts.append(text)
        return ids, texts
