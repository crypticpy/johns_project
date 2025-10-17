from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for all ORM models."""

    pass


class Dataset(Base):
    """
    Dataset metadata persisted for each uploaded file.

    Notes:
    - Database column is named "metadata" to satisfy spec, but Python attribute uses "metadata_"
      to avoid clashing with SQLAlchemy's Base.metadata attribute.
    """

    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("file_hash", name="uq_datasets_file_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_hash: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # md5 hex (32) padded future-proof
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    department_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    upload_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Column named "metadata" in DB; attribute name "metadata_"
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    # Relationships
    tickets: Mapped[list[Ticket]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"Dataset(id={self.id!r}, name={self.name!r}, file_hash={self.file_hash!r})"


class Ticket(Base):
    """
    Ticket rows associated with a Dataset.

    Canonical columns aligned with charts/analysis naming in docs/reference_behavior.md:
      - Department
      - Assignment Group
      - extract_product
      - summarize_ticket
      - ticket_quality
      - resolution_complexity
      - Reassignment group count tracking_index
    Plus:
      - normalized_text (cleaned/redacted text for downstream features)
    """

    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False
    )

    # Canonical features (nullable to allow partial datasets)
    department: Mapped[str | None] = mapped_column(String(255), nullable=True)
    assignment_group: Mapped[str | None] = mapped_column(String(255), nullable=True)
    product: Mapped[str | None] = mapped_column(String(255), nullable=True)  # extract_product
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)  # summarize_ticket
    quality: Mapped[str | None] = mapped_column(String(100), nullable=True)  # ticket_quality
    complexity: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )  # resolution_complexity
    reassignment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    normalized_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationship
    dataset: Mapped[Dataset] = relationship(back_populates="tickets")

    def __repr__(self) -> str:
        return f"Ticket(id={self.id!r}, dataset_id={self.dataset_id!r})"


class Embedding(Base):
    """
    Per-ticket embedding vectors.

    Idempotency:
    - Unique constraint on (dataset_id, ticket_id, model_name) to prevent duplicates for same dataset+model.
    """

    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint(
            "dataset_id", "ticket_id", "model_name", name="uq_embeddings_dataset_ticket_model"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    ticket_id: Mapped[int] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships (optional navigation)
    dataset: Mapped[Dataset] = relationship()
    ticket: Mapped[Ticket] = relationship()

    def __repr__(self) -> str:
        return f"Embedding(id={self.id!r}, ticket_id={self.ticket_id!r}, model_name={self.model_name!r})"


class ClusterRun(Base):
    """
    Clustering run metadata for a dataset/model and algorithm.
    """

    __tablename__ = "cluster_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    algorithm: Mapped[str] = mapped_column(Text, nullable=False)
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"ClusterRun(id={self.id!r}, dataset_id={self.dataset_id!r}, algorithm={self.algorithm!r}, model_name={self.model_name!r})"


class ClusterAssignment(Base):
    """
    Per-ticket cluster assignment for a specific run.
    """

    __tablename__ = "cluster_assignments"
    __table_args__ = (
        UniqueConstraint("run_id", "ticket_id", name="uq_cluster_assignments_run_ticket"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("cluster_runs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    ticket_id: Mapped[int] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)

    def __repr__(self) -> str:
        return f"ClusterAssignment(run_id={self.run_id!r}, ticket_id={self.ticket_id!r}, cluster_id={self.cluster_id!r})"


class ClusterMetric(Base):
    """
    Metrics for a clustering run (e.g., silhouette).
    """

    __tablename__ = "cluster_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("cluster_runs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    def __repr__(self) -> str:
        return f"ClusterMetric(run_id={self.run_id!r}, name={self.name!r})"


class ClusterTerm(Base):
    """
    Top TF-IDF terms per cluster for a given run.
    """

    __tablename__ = "cluster_terms"
    __table_args__ = (
        UniqueConstraint("run_id", "cluster_id", "term", name="uq_cluster_terms_run_cluster_term"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("cluster_runs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    term: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)

    def __repr__(self) -> str:
        return f"ClusterTerm(run_id={self.run_id!r}, cluster_id={self.cluster_id!r}, term={self.term!r}, score={self.score!r})"


class Analysis(Base):
    """
    Persisted LLM analysis runs and results.

    Fields:
      - id: primary key
      - dataset_id: FK to datasets
      - prompt_version: version identifier of the prompt template used
      - question: user-provided question or analysis focus
      - result_markdown: structured markdown produced by analyzer
      - metrics: JSON blob for sampling/analysis metrics (e.g., segment counts)
      - ticket_count: number of tickets sampled/analyzed
      - filters: JSON blob for filter context (e.g., departments included)
      - created_at: timestamp of creation
    """

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    prompt_version: Mapped[str] = mapped_column(Text, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    result_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ticket_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    filters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"Analysis(id={self.id!r}, dataset_id={self.dataset_id!r}, prompt_version={self.prompt_version!r})"


class AuditLog(Base):
    """
    Append-only audit log for administrative and sensitive actions.

    Columns:
      - id: primary key
      - timestamp: event time (server default now)
      - subject: actor identifier (e.g., JWT sub or 'anonymous')
      - action: action verb (e.g., 'analysis.run')
      - resource: resource identifier (e.g., 'dataset:{id}')
      - metadata: optional JSON payload (PII-free)
    """

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    resource: Mapped[str] = mapped_column(Text, nullable=False)
    # Column named "metadata" in DB; attribute name "metadata_" to avoid Base.metadata conflict
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    def __repr__(self) -> str:
        return f"AuditLog(id={self.id!r}, subject={self.subject!r}, action={self.action!r}, resource={self.resource!r})"
