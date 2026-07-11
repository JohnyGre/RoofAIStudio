"""
This module defines application-wide constants, grouped into classes for better organization.
"""

from typing import Final

class UI:
    """Constants related to the User Interface."""
    MAIN_WINDOW_TITLE: Final[str] = "Roof AI Studio"
    DEFAULT_WIDTH: Final[int] = 1280
    DEFAULT_HEIGHT: Final[int] = 800
    MIN_WIDTH: Final[int] = 800
    MIN_HEIGHT: Final[int] = 600

class DATABASE:
    """Constants related to the database."""
    DATABASE_NAME: Final[str] = "roof_ai_studio.db"
    # Add other database-related constants here

class FILES:
    """Constants related to file operations and types."""
    PROJECT_FILE_EXTENSION: Final[str] = ".raip" # Roof AI Project
    # Add other file-related constants here

class IMAGES:
    """Constants related to image processing and assets."""
    DEFAULT_LOGO_PATH: Final[str] = "assets/icons/logo.png"
    # Add other image-related constants here

class AI:
    """Constants related to Artificial Intelligence modules."""
    DEFAULT_MODEL_EXTENSION: Final[str] = ".h5" # Example for Keras models
    # Add other AI-related constants here

class CALIBRATION:
    """Constants related to calibration processes."""
    DEFAULT_CALIBRATION_PROFILE_EXTENSION: Final[str] = ".json"
    # Add other calibration-related constants here
