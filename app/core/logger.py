"""
This module sets up professional logging for the Roof AI Studio application,
including console output and rotating file logs.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from app.core.app_paths import app_paths
from app.core.app_info import APP_NAME

def setup_logging(log_level: int = logging.INFO) -> None:
    """
    Configures the application's logging system.

    - Logs to console (StreamHandler).
    - Logs to a rotating file (RotatingFileHandler) in the 'logs' directory.
    - Automatically creates the 'logs' directory if it doesn't exist.
    - Uses a formatter with a timestamp.

    Args:
        log_level (int): The minimum logging level to capture (e.g., logging.INFO, logging.DEBUG).
    """
    # Ensure the logs directory exists
    log_dir: Path = app_paths.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file: Path = log_dir / f"{APP_NAME.lower().replace(' ', '_')}.log"

    # Create a logger
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(log_level)

    # Prevent adding multiple handlers if setup_logging is called multiple times
    if not logger.handlers:
        # Formatter for both console and file
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO) # Console usually shows INFO and above
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler (Rotating)
        # Max 5 MB per file, keep 5 backup files
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Set propagate to False to prevent messages from being passed to the root logger
        # if the root logger also has handlers configured elsewhere.
        logger.propagate = False

# Initialize logging when the module is imported
setup_logging()

# Example usage (can be removed after verification)
# if __name__ == "__main__":
#     logger = logging.getLogger(APP_NAME)
#     logger.debug("This is a debug message.")
#     logger.info("This is an info message.")
#     logger.warning("This is a warning message.")
#     logger.error("This is an error message.")
#     logger.critical("This is a critical message.")
