"""
TradeSense V2 — SQLite Database Connection & Session Management

Uses SQLAlchemy with SQLite backend.
Database path: data/tradesense.db (relative to project root).
Designed for future PostgreSQL migration — swap DATABASE_URL to use psycopg2.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# Resolve absolute path for SQLite database
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "tradesense.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH}")

# Create SQLite engine with WAL mode for better concurrent reads
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,  # Set True for SQL query logging during development
)

# Enable WAL journal mode for SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragmas(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all SQLAlchemy ORM models."""
    pass


def get_db():
    """
    FastAPI dependency — yields a database session and guarantees cleanup.
    Usage: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables defined in models.py if they don't already exist."""
    from backend.database import models  # noqa: F401 — ensures models are registered
    Base.metadata.create_all(bind=engine)
