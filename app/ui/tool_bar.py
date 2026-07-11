"""
This module defines the ToolBar for the Roof AI Studio application.
"""

from PySide6.QtWidgets import QToolBar, QToolButton
from PySide6.QtGui import QIcon
from PySide6.QtCore import Signal, QSize

class ToolBar(QToolBar):
    """
    Custom QToolBar for the Roof AI Studio application.
    Emits signals for each tool button action.
    """

    # Tool Bar Signals
    new_triggered = Signal()
    open_triggered = Signal()
    save_triggered = Signal()
    analyze_triggered = Signal()
    measure_triggered = Signal()
    export_triggered = Signal()

    def __init__(self, parent=None):
        """
        Initializes the ToolBar.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setIconSize(QSize(24, 24)) # Set a default icon size
        self.setWindowTitle("Main Toolbar")
        self._create_tools()

    def _create_tools(self) -> None:
        """
        Creates and adds tool buttons to the toolbar.
        Icons are placeholders and should be replaced with actual assets.
        """
        # New Project
        new_button = QToolButton(self)
        new_button.setIcon(QIcon(":/icons/new.png")) # Placeholder icon
        new_button.setText("New")
        new_button.setToolTip("Create a New Project")
        new_button.clicked.connect(self.new_triggered)
        self.addWidget(new_button)

        # Open Project
        open_button = QToolButton(self)
        open_button.setIcon(QIcon(":/icons/open.png")) # Placeholder icon
        open_button.setText("Open")
        open_button.setToolTip("Open an Existing Project")
        open_button.clicked.connect(self.open_triggered)
        self.addWidget(open_button)

        # Save Project
        save_button = QToolButton(self)
        save_button.setIcon(QIcon(":/icons/save.png")) # Placeholder icon
        save_button.setText("Save")
        save_button.setToolTip("Save Current Project")
        save_button.clicked.connect(self.save_triggered)
        self.addWidget(save_button)

        self.addSeparator()

        # Analyze Roof
        analyze_button = QToolButton(self)
        analyze_button.setIcon(QIcon(":/icons/analyze.png")) # Placeholder icon
        analyze_button.setText("Analyze")
        analyze_button.setToolTip("Analyze Roof Geometry with AI")
        analyze_button.clicked.connect(self.analyze_triggered)
        self.addWidget(analyze_button)

        # Measure Tool
        measure_button = QToolButton(self)
        measure_button.setIcon(QIcon(":/icons/measure.png")) # Placeholder icon
        measure_button.setText("Measure")
        measure_button.setToolTip("Activate Measurement Tools")
        measure_button.clicked.connect(self.measure_triggered)
        self.addWidget(measure_button)

        self.addSeparator()

        # Export
        export_button = QToolButton(self)
        export_button.setIcon(QIcon(":/icons/export.png")) # Placeholder icon
        export_button.setText("Export")
        export_button.setToolTip("Export Project Data (e.g., PDF)")
        export_button.clicked.connect(self.export_triggered)
        self.addWidget(export_button)
