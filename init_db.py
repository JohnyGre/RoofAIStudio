from app.database.database import db_manager
from app.core.logger import setup_logging

# Setup logging for this script
logger = setup_logging()

if __name__ == "__main__":
    logger.info("Initializing database schema...")
    try:
        db_manager.create_all_tables()
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")
