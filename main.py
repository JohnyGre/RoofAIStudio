import sys
from PySide6.QtWidgets import QApplication
from app.ui import MainWindow
from app.database.database import db_manager
from app.core.logger import setup_logging

logger = setup_logging()

def main():
    """
    Main entry point for the Roof AI Studio application.
    """
    # Initialize database schema
    try:
        db_manager.create_all_tables()
        logger.info("Database schema ensured.")
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}")
        sys.exit(1) # Exit if database cannot be initialized

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
