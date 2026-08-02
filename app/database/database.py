"""
This module provides the core database functionality for Roof AI Studio,
including engine creation and session management.
"""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import config
from app.core.logger import setup_logging
from app.database.base import Base
# No direct import of models here; they are imported via app.database.__init__

# Setup logging for this module
logger = setup_logging()

class DatabaseManager:
    """
    Manages the SQLAlchemy engine and session for the application.
    """
    def __init__(self):
        """
        Initializes the database manager, creating the engine and session factory.
        """
        self.engine = create_engine(
            f"sqlite:///{config.database_path}",
            echo=config.debug_mode,  # Echo SQL statements if in debug mode
            connect_args={"check_same_thread": False} # Required for SQLite with multiple threads
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        logger.info(f"Database initialized at: {config.database_path}")

    def create_all_tables(self) -> None:
        """
        Creates all tables defined in the SQLAlchemy Base metadata.
        """
        logger.info("Attempting to create all database tables...")
        try:
            # Importing the database package ensures all models defined in its submodules
            # and imported into __init__.py are registered with Base.metadata.
            import app.database
            Base.metadata.create_all(bind=self.engine)
            logger.info("All database tables created successfully or already exist.")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise

# Instantiate the DatabaseManager to be used throughout the application
db_manager = DatabaseManager()
