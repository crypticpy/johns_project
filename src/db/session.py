from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings

try:
    # SQLAlchemy 2.x utility to parse URLs robustly
    from sqlalchemy.engine import make_url  # type: ignore
except Exception:  # pragma: no cover
    make_url = None  # fallback if unavailable


def _is_sqlite_url(url: str) -> bool:
    """
    Determine if the given SQLAlchemy URL uses a SQLite scheme.
    Robust to odd inputs by string-casting and scheme extraction.
    """
    try:
        scheme = str(url).split(":")[0].lower()
        return scheme.startswith("sqlite")
    except Exception:
        return False


def _ensure_sqlite_dir(database_url: str) -> None:
    """
    Ensure directory for SQLite file exists in dev when using a file-based URL.

    Handles common forms:
      - sqlite:///./data/app.db
      - sqlite+pysqlite:///./data/app.db
      - sqlite:////absolute/path/app.db
    """
    if not _is_sqlite_url(database_url):
        return

    # Best-effort parse: prefer make_url; fallback to manual parsing
    db_path: Path | None = None
    try:
        if make_url is not None:
            url = make_url(database_url)
            # url.database returns the filesystem path (may be relative)
            if url.database:
                db_path = Path(url.database)
    except Exception:
        db_path = None

    if db_path is None:
        # Manual fallback: strip scheme and possible driver
        # e.g., sqlite+pysqlite:///./data/app.db -> ./data/app.db
        parts = database_url.split(":///")
        if len(parts) == 2:
            db_path = Path(parts[1])
        else:
            # absolute path variant sqlite:////path/to/app.db
            parts_abs = database_url.split(":////")
            if len(parts_abs) == 2:
                db_path = Path("/" + parts_abs[1])

    if db_path is None:
        return

    # Resolve relative path against current working directory for directory creation
    parent_dir = (Path.cwd() / db_path).parent if not db_path.is_absolute() else db_path.parent
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)


def _create_engine() -> Engine:
    """
    Create a SQLAlchemy Engine bound to settings, with SQLite-safe connect args.
    """
    settings = get_settings()
    database_url: str = cast(str, settings.database_url)
    _ensure_sqlite_dir(database_url)

    connect_args: dict = {}
    if _is_sqlite_url(database_url):
        # FastAPI runs with multiple threads; SQLite needs check_same_thread=False
        connect_args = {"check_same_thread": False}

    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        future=True,
        echo=False,
        connect_args=connect_args,
    )
    return engine


# Singleton Engine and Session factory
engine: Engine = _create_engine()
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def get_session() -> Iterator[Session]:
    """
    FastAPI dependency that yields a DB session and ensures proper cleanup.

    Usage:
        @router.get("/route")
        def handler(db: Session = Depends(get_session)): ...
    """
    # Defensive: ensure tables exist in dev (idempotent). This guards tests that
    # construct an app without running startup lifespan yet.
    try:
        init_db()
    except Exception:
        # If initialization fails, allow request to proceed; DB operations will surface errors.
        pass

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database tables for dev by calling SQLAlchemy metadata.create_all.

    This is safe and idempotent. In production, migrations should replace this.
    """
    # Import inside function to avoid circular dependencies during app startup
    from db import Base  # type: ignore

    Base.metadata.create_all(bind=engine)


__all__ = ["engine", "SessionLocal", "get_session", "init_db"]
