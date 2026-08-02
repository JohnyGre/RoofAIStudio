"""
This module defines the central configuration class for the Roof AI Studio application.
"""

from pathlib import Path
from app.core.app_paths import app_paths
from app.core.constants import DATABASE

class Config:
    """
    Central configuration class for the Roof AI Studio application.
    Manages various application settings and paths.
    """

    def __init__(self):
        """
        Initializes the configuration settings.
        """
        self._debug_mode: bool = False

        # Paths
        self._database_path: Path = app_paths.data_dir / DATABASE.DATABASE_NAME
        self._projects_path: Path = app_paths.projects_dir
        self._assets_path: Path = app_paths.assets_dir
        self._export_path: Path = app_paths.data_dir / "exports" # Default export location
        self._ai_models_path: Path = app_paths.ai_models_dir

        # Ensure export directory exists
        self._export_path.mkdir(parents=True, exist_ok=True)

    @property
    def debug_mode(self) -> bool:
        """
        Gets the current debug mode status.
        """
        return self._debug_mode

    @debug_mode.setter
    def debug_mode(self, value: bool) -> None:
        """
        Sets the debug mode status.
        """
        if not isinstance(value, bool):
            raise TypeError("debug_mode must be a boolean.")
        self._debug_mode = value

    @property
    def database_path(self) -> Path:
        """
        Gets the path to the SQLite database file.
        """
        return self._database_path

    @property
    def projects_path(self) -> Path:
        """
        Gets the base path for user projects.
        """
        return self._projects_path

    @property
    def assets_path(self) -> Path:
        """
        Gets the base path for application assets.
        """
        return self._assets_path

    @property
    def export_path(self) -> Path:
        """
        Gets the default path for exported files.
        """
        return self._export_path

    @property
    def ai_models_path(self) -> Path:
        """
        Gets the base path for AI models.
        """
        return self._ai_models_path

# Instantiate Config to be easily imported and used throughout the application
config = Config()
