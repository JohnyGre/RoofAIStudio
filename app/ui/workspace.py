"""
This module defines the main workspace widget for the Roof AI Studio application,
integrating the RoofCanvas and placeholder panels.
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame, QSizePolicy
from PySide6.QtCore import Qt

from app.ui.roof_canvas import RoofCanvas

class Workspace(QWidget):
    """
    The main workspace widget for Roof AI Studio, featuring a central RoofCanvas
    and placeholder panels for project info, properties, and status.
    """

    def __init__(self, parent: QWidget = None):
        """
        Initializes the Workspace.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """
        Sets up the layout and widgets for the workspace.
        """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5) # Small spacing between major sections

        # Top section: Left Panel, Central Canvas, Right Panel
        top_section_layout = QHBoxLayout()
        top_section_layout.setContentsMargins(0, 0, 0, 0)
        top_section_layout.setSpacing(5)

        # Left Panel (Placeholder for Project Info)
        self._left_panel = self._create_panel("Project Information", "LeftPanel")
        self._left_panel.setMinimumWidth(200)
        self._left_panel.setMaximumWidth(300)
        top_section_layout.addWidget(self._left_panel)

        # Central Roof Canvas
        self.roof_canvas = RoofCanvas(self)
        self.roof_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        top_section_layout.addWidget(self.roof_canvas, 1) # Stretch factor 1 for canvas

        # Right Panel (Placeholder for Properties)
        self._right_panel = self._create_panel("Properties", "RightPanel")
        self._right_panel.setMinimumWidth(200)
        self._right_panel.setMaximumWidth(300)
        top_section_layout.addWidget(self._right_panel)

        main_layout.addLayout(top_section_layout, 1) # Stretch factor 1 for top section

        # Bottom Panel (Placeholder for Status/Logs)
        self._bottom_panel = self._create_panel("Status / Logs", "BottomPanel")
        self._bottom_panel.setMinimumHeight(100)
        self._bottom_panel.setMaximumHeight(150)
        main_layout.addWidget(self._bottom_panel)

    def _create_panel(self, title: str, object_name: str) -> QFrame:
        """
        Helper method to create a styled placeholder panel.
        """
        panel = QFrame(self)
        panel.setObjectName(object_name)
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setFrameShadow(QFrame.Shadow.Raised)
        panel.setStyleSheet(f"#{object_name} {{ background-color: #3E5060; border: 1px solid #2C3E50; border-radius: 5px; }}")

        layout = QVBoxLayout(panel)
        label = QLabel(title, panel)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-weight: bold; color: #ECF0F1;")
        layout.addWidget(label)
        layout.addStretch(1) # Push title to top

        return panel

    def load_image_to_canvas(self, image_path: str) -> None:
        """
        Loads an image into the central RoofCanvas.
        """
        self.roof_canvas.load_image(image_path)
