"""
This module centralizes all application folder paths using pathlib,
and ensures that necessary directories are created automatically.
"""

from pathlib import Path
import sys

class AppPaths:
    """
    Manages all application-specific file system paths.
    Ensures that required directories exist.
    """

    def __init__(self):
        """
        Initializes AppPaths and creates necessary directories.
        """
        self._base_dir: Path = self._get_base_directory()
        self._app_dir: Path = self._base_dir / "app"
        self._assets_dir: Path = self._base_dir / "assets"
        self._config_dir: Path = self._base_dir / "config"
        self._data_dir: Path = self._base_dir / "data"
        self._docs_dir: Path = self._base_dir / "docs"
        self._projects_dir: Path = self._base_dir / "projects"
        self._logs_dir: Path = self._base_dir / "logs"
        self._ai_models_dir: Path = self._base_dir / "ai_models" # Dedicated for AI models

        self._create_directories()

    def _get_base_directory(self) -> Path:
        """
        Determines the base directory of the application.
        Handles cases where the application is run as a script or a bundled executable.
        """
        if getattr(sys, 'frozen', False):
            # Running as a bundled executable (e.g., PyInstaller)
            return Path(sys.executable).parent
        else:
            # Running as a script
            return Path(__file__).parent.parent.parent # RoofAIStudio/app/core -> RoofAIStudio

    def _create_directories(self) -> None:
        """
        Creates all necessary directories if they do not already exist.
        """
        for path in [
            self._app_dir,
            self._assets_dir,
            self._config_dir,
            self._data_dir,
            self._docs_dir,
            self._projects_dir,
            self._logs_dir,
            self._ai_models_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        """Returns the base directory of the application."""
        return self._base_dir

    @property
    def app_dir(self) -> Path:
        """Returns the application source directory."""
        return self._app_dir

    @property
    def assets_dir(self) -> Path:
        """Returns the assets directory."""
        return self._assets_dir

    @property
    def config_dir(self) -> Path:
        """Returns the configuration directory."""
        return self._config_dir

    @property
    def data_dir(self) -> Path:
        """Returns the data directory."""
        return self._data_dir

    @property
    def docs_dir(self) -> Path:
        """Returns the documentation directory."""
        return self._docs_dir

    @property
    def projects_dir(self) -> Path:
        """Returns the projects directory where user projects are stored."""
        return self._projects_dir

    @property
    def logs_dir(self) -> Path:
        """Returns the directory for application logs."""
        return self._logs_dir

    @property
    def ai_models_dir(self) -> Path:
        """Returns the directory for AI models."""
        return self._ai_models_dir

# Instantiate AppPaths to ensure directories are created on import
app_paths = AppPaths()
