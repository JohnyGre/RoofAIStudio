"""
This module provides utilities for managing SQLAlchemy database sessions.
"""

from typing import Generator

from sqlalchemy.orm import Session

from app.database.database import db_manager

def get_db_session() -> Generator[Session, None, None]:
    """
    Dependency for getting a database session.
    Yields a session and ensures it's closed after use.
    """
    db = db_manager.SessionLocal()
    try:
        yield db
    finally:
        db.close()
