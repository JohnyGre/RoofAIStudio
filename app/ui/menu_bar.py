"""
This module defines the MenuBar for the Roof AI Studio application.
"""

from PySide6.QtWidgets import QMenuBar, QMenu
from PySide6.QtCore import Signal

class MenuBar(QMenuBar):
    """
    Custom QMenuBar for the Roof AI Studio application.
    Emits signals for each menu action.
    """

    # File Menu Signals
    new_project_triggered = Signal()
    open_project_triggered = Signal()
    save_project_triggered = Signal()
    export_pdf_triggered = Signal()
    exit_triggered = Signal()

    # Project Menu Signals
    project_settings_triggered = Signal()

    # AI Menu Signals
    analyze_roof_triggered = Signal()
    ai_models_triggered = Signal()

    # Tools Menu Signals
    geometry_editor_triggered = Signal()
    materials_triggered = Signal()

    # Help Menu Signals
    about_triggered = Signal()

    def __init__(self, parent=None):
        """
        Initializes the MenuBar.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self._file_menu: Optional[QMenu] = None # Initialize _file_menu
        self._create_menus()

    @property
    def file_menu(self) -> QMenu:
        """
        Returns the 'File' QMenu object.
        """
        if self._file_menu is None:
            raise RuntimeError("File menu has not been created yet.")
        return self._file_menu

    def _create_menus(self) -> None:
        """
        Creates and populates all menus and their actions.
        """
        self._create_file_menu()
        self._create_project_menu()
        self._create_ai_menu()
        self._create_tools_menu()
        self._create_help_menu()

    def _create_file_menu(self) -> None:
        """
        Creates the 'File' menu and its actions.
        """
        self._file_menu = self.addMenu("&File") # Assign the created menu
        
        new_project_action = self._file_menu.addAction("New Project")
        new_project_action.triggered.connect(self.new_project_triggered)

        open_project_action = self._file_menu.addAction("Open Project")
        open_project_action.triggered.connect(self.open_project_triggered)

        save_project_action = self._file_menu.addAction("Save Project")
        save_project_action.triggered.connect(self.save_project_triggered)

        self._file_menu.addSeparator()

        export_pdf_action = self._file_menu.addAction("Export PDF")
        export_pdf_action.triggered.connect(self.export_pdf_triggered)

        self._file_menu.addSeparator()

        exit_action = self._file_menu.addAction("Exit")
        exit_action.triggered.connect(self.exit_triggered)

    def _create_project_menu(self) -> None:
        """
        Creates the 'Project' menu and its actions.
        """
        project_menu = self.addMenu("&Project")

        project_settings_action = project_menu.addAction("Project Settings")
        project_settings_action.triggered.connect(self.project_settings_triggered)

    def _create_ai_menu(self) -> None:
        """
        Creates the 'AI' menu and its actions.
        """
        ai_menu = self.addMenu("&AI")

        analyze_roof_action = ai_menu.addAction("Analyze Roof")
        analyze_roof_action.triggered.connect(self.analyze_roof_triggered)

        ai_models_action = ai_menu.addAction("AI Models")
        ai_models_action.triggered.connect(self.ai_models_triggered)

    def _create_tools_menu(self) -> None:
        """
        Creates the 'Tools' menu and its actions.
        """
        tools_menu = self.addMenu("&Tools")

        geometry_editor_action = tools_menu.addAction("Geometry Editor")
        geometry_editor_action.triggered.connect(self.geometry_editor_triggered)

        materials_action = tools_menu.addAction("Materials")
        materials_action.triggered.connect(self.materials_triggered)

    def _create_help_menu(self) -> None:
        """
        Creates the 'Help' menu and its actions.
        """
        help_menu = self.addMenu("&Help")

        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.about_triggered)
