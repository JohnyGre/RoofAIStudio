"""
This module defines the StatusBar for the Roof AI Studio application.
"""

from PySide6.QtWidgets import QStatusBar, QLabel
from PySide6.QtCore import Qt

class StatusBar(QStatusBar):
    """
    Custom QStatusBar for the Roof AI Studio application.
    Displays application status, current project, and version.
    """

    def __init__(self, parent=None):
        """
        Initializes the StatusBar.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self._status_label = QLabel("Ready")
        self._project_label = QLabel("No Project Open")
        self._version_label = QLabel("vX.Y.Z") # Placeholder, will be updated dynamically

        self.addWidget(self._status_label)
        self.addPermanentWidget(self._project_label)
        self.addPermanentWidget(self._version_label)

        self.setStyleSheet("QStatusBar { padding-left: 8px; }") # Add some padding

    def set_status_message(self, message: str) -> None:
        """
        Sets the main status message.
        """
        self._status_label.setText(message)

    def set_current_project(self, project_name: str) -> None:
        """
        Sets the current project name displayed.
        """
        self._project_label.setText(f"Project: {project_name}")

    def set_application_version(self, version: str) -> None:
        """
        Sets the application version displayed.
        """
        self._version_label.setText(f"v{version}")
