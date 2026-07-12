"""
This module defines the main application window for Roof AI Studio.
"""

from pathlib import Path
from typing import List, Union
from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QToolButton, QMenu, QInputDialog # Added QInputDialog
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QIcon

from app.core import APP_NAME, UI, VERSION
from app.ui.menu_bar import MenuBar
from app.ui.tool_bar import ToolBar
from app.ui.status_bar import StatusBar
from app.ui.styles import DarkTheme
from app.ui.workspace import Workspace
from app.controllers.image_controller import ImageController
from app.controllers.geometry_controller import GeometryController
from app.controllers.ai_controller import AIController
from app.core.image.image_model import ImageInfo
from app.geometry.calibration import CalibrationModel, CalibrationService
from app.materials.material_repository import SQLAlchemyMaterialRepository
from app.database.session import get_db_session
from app.materials.calculation_result import MaterialCalculationResult
from app.geometry.roof_geometry import RoofGeometry
from app.ai.ai_result import DetectionResult, SegmentationResult
from app.geometry.point import Point2D # Added import for Point2D

class MainWindow(QMainWindow):
    """
    The main application window for Roof AI Studio.
    Integrates the menu bar, tool bar, status bar, and a central workspace widget.
    """

    def __init__(self):
        """
        Initializes the MainWindow.
        """
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setGeometry(100, 100, UI.DEFAULT_WIDTH, UI.DEFAULT_HEIGHT)
        self.setMinimumSize(UI.MIN_WIDTH, UI.MIN_HEIGHT)

        self._apply_styles()
        self._create_menu_bar()
        self._create_tool_bar()
        self._create_status_bar()
        self._create_central_widget()

        # Initialize database session for repository
        self._db_session = next(get_db_session()) # Get a session for the lifetime of the app
        self._material_repository = SQLAlchemyMaterialRepository(self._db_session)

        self.image_controller = ImageController(self)
        self.geometry_controller = GeometryController(self._material_repository, self)
        
        # AIController needs AIEngine and GeometryConverter
        # AIEngine is a singleton, GeometryConverter is stateless
        from app.ai.ai_engine import AIEngine
        from app.ai.geometry_converter import GeometryConverter
        self.ai_engine = AIEngine() # Get the singleton instance
        self.geometry_converter = GeometryConverter()
        self.ai_controller = AIController(self.ai_engine, self.geometry_converter, self) # Instantiate AIController

        self._connect_signals()

    def _apply_styles(self) -> None:
        """
        Applies the defined QSS styles to the main window.
        """
        self.setStyleSheet(DarkTheme.MAIN_WINDOW_QSS)

    def _create_menu_bar(self) -> None:
        """
        Creates and sets up the custom menu bar.
        """
        self.menu_bar = MenuBar(self)
        self.setMenuBar(self.menu_bar)

        # Add "Open Image" action to File menu
        self.menu_bar.file_menu.addSeparator()
        open_image_action = self.menu_bar.file_menu.addAction("Open Image...")
        open_image_action.triggered.connect(self._on_open_image)

        # Add AI Overlay toggle to AI menu
        if self.menu_bar.ai_menu:
            self.ai_overlay_action = self.menu_bar.ai_menu.addAction("Toggle AI Overlay")
            self.ai_overlay_action.setCheckable(True)
            self.ai_overlay_action.setChecked(False) # Default to off
            self.ai_overlay_action.triggered.connect(self._on_toggle_ai_overlay)
        else:
            print("Warning: AI menu not found in MenuBar.")


    def _create_tool_bar(self) -> None:
        """
        Creates and sets up the custom tool bar.
        """
        self.tool_bar = ToolBar(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.tool_bar)

    def _create_status_bar(self) -> None:
        """
        Creates and sets up the custom status bar.
        """
        self.status_bar = StatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.set_application_version(VERSION)

    def _create_central_widget(self) -> None:
        """
        Creates and sets the Workspace widget as the central widget.
        """
        self.workspace = Workspace(self)
        self.setCentralWidget(self.workspace)

    def _connect_signals(self) -> None:
        """
        Connects signals from menu bar and tool bar to main window slots (or placeholder methods).
        Connects ImageController, GeometryController, and AIController signals to UI elements.
        """
        # File Menu
        self.menu_bar.new_project_triggered.connect(self._on_new_project)
        self.menu_bar.open_project_triggered.connect(self._on_open_project)
        self.menu_bar.save_project_triggered.connect(self._on_save_project)
        self.menu_bar.export_pdf_triggered.connect(self._on_export_pdf)
        self.menu_bar.exit_triggered.connect(self.close)

        # Project Menu
        self.menu_bar.project_settings_triggered.connect(self._on_project_settings)

        # AI Menu
        self.menu_bar.analyze_roof_triggered.connect(self._on_analyze_roof_action)
        self.menu_bar.ai_models_triggered.connect(self._on_ai_models)

        # Tools Menu
        self.menu_bar.geometry_editor_triggered.connect(self._on_geometry_editor_action)
        self.menu_bar.materials_triggered.connect(self._on_materials)
        self.menu_bar.calibrate_image_triggered.connect(self._on_calibrate_image_action) # Connect new menu action

        # Help Menu
        self.menu_bar.about_triggered.connect(self._on_about)

        # Tool Bar
        self.tool_bar.new_triggered.connect(self._on_new_project)
        self.tool_bar.open_triggered.connect(self._on_open_project)
        self.tool_bar.save_triggered.connect(self._on_save_project)
        self.tool_bar.analyze_triggered.connect(self._on_analyze_roof_action)
        self.tool_bar.measure_triggered.connect(self._on_measure)
        self.tool_bar.calibrate_triggered.connect(self._on_calibrate_image_action) # Connect new toolbar action
        self.tool_bar.export_triggered.connect(self._on_export_pdf)

        # ImageController Signals
        self.image_controller.image_loaded.connect(self.workspace.roof_canvas.display_qpixmap)
        self.image_controller.image_cleared.connect(self.workspace.roof_canvas.clear_canvas)
        self.image_controller.error_occurred.connect(self.status_bar.set_status_message)
        self.image_controller.status_message.connect(self.status_bar.set_status_message)
        self.image_controller.image_loaded.connect(self.ai_controller.set_current_image)

        # RoofCanvas Signals
        self.workspace.roof_canvas.image_displayed.connect(self._on_image_displayed_on_canvas)
        self.workspace.roof_canvas.image_cleared.connect(self._on_image_cleared_from_canvas)
        self.workspace.roof_canvas.point_added_to_drawing.connect(self.geometry_controller.add_point)
        self.workspace.roof_canvas.point_moved_in_drawing.connect(self.geometry_controller.move_point)
        self.workspace.roof_canvas.polygon_drawing_finished.connect(self.geometry_controller.finalize_polygon)
        self.workspace.roof_canvas.drawing_mode_changed.connect(self._on_drawing_mode_changed)
        self.workspace.roof_canvas.calibration_points_selected.connect(self._on_calibration_points_selected) # Connect new signal

        # GeometryController Signals
        self.geometry_controller.polygon_drawing_updated.connect(self.workspace.roof_canvas.update_drawing_visuals)
        self.geometry_controller.drawing_cleared.connect(self.workspace.roof_canvas.clear_drawing_visuals)
        self.geometry_controller.status_message.connect(self.status_bar.set_status_message)
        self.geometry_controller.error_occurred.connect(self.status_bar.set_status_message)
        self.geometry_controller.roof_geometry_created.connect(self._on_roof_geometry_created)
        self.geometry_controller.measurements_calculated.connect(self._on_measurements_calculated)
        self.geometry_controller.materials_calculated.connect(self._on_materials_calculated)

        # AIController Signals
        self.ai_controller.analysis_started.connect(lambda: self.status_bar.set_status_message("AI Analysis in progress..."))
        self.ai_controller.analysis_completed.connect(self._on_ai_analysis_completed)
        self.ai_controller.error_occurred.connect(self.status_bar.set_status_message)
        self.ai_controller.status_message.connect(self.status_bar.set_status_message)


    # --- Slots for Menu/Toolbar Actions ---
    def _on_new_project(self) -> None:
        self.status_bar.set_status_message("Action: New Project")
        print("New Project triggered")

    def _on_open_project(self) -> None:
        self.status_bar.set_status_message("Action: Open Project")
        print("Open Project triggered")

    def _on_save_project(self) -> None:
        self.status_bar.set_status_message("Action: Save Project")
        print("Save Project triggered")

    def _on_export_pdf(self) -> None:
        self.status_bar.set_status_message("Action: Export PDF")
        print("Export PDF triggered")

    def _on_project_settings(self) -> None:
        self.status_bar.set_status_message("Action: Project Settings")
        print("Project Settings triggered")

    def _on_analyze_roof_action(self) -> None:
        """
        Slot to handle the "Analyze Roof" menu/toolbar action.
        Triggers AI analysis.
        """
        if self.image_controller.current_image_data is None:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first to perform AI analysis.")
            return
        if self.geometry_controller._current_calibration is None:
            QMessageBox.warning(self, "No Calibration Set", "Please set calibration (e.g., by loading an image) before performing AI analysis.")
            return

        # Use AIController's default model selection (will use registered RoofDetector by default)
        self.ai_controller.analyze_roof()
        print("Analyze Roof triggered.")

    def _on_ai_models(self) -> None:
        self.status_bar.set_status_message("Action: AI Models")
        print("AI Models triggered")

    def _on_geometry_editor_action(self) -> None:
        """
        Slot to handle the "Geometry Editor" menu action.
        Activates the geometry drawing mode.
        """
        if self.image_controller.current_image_data is None:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first to use the Geometry Editor.")
            return

        # Toggle drawing mode
        if self.geometry_controller.is_drawing_active:
            self.geometry_controller.stop_drawing()
            self.workspace.roof_canvas.set_drawing_mode(False)
        else:
            self.geometry_controller.start_drawing()
            self.workspace.roof_canvas.set_drawing_mode(True)
        print(f"Geometry Editor triggered. Drawing mode active: {self.geometry_controller.is_drawing_active}")

    def _on_calibrate_image_action(self) -> None:
        """
        Slot to handle the "Calibrate Image" menu/toolbar action.
        Activates the calibration mode on the canvas.
        """
        if self.image_controller.current_image_data is None:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first to calibrate.")
            return
        
        # Deactivate other modes
        self.geometry_controller.stop_drawing()
        self.workspace.roof_canvas.set_drawing_mode(False)

        # Activate calibration mode
        self.workspace.roof_canvas.set_calibration_mode(True)
        self.status_bar.set_status_message("Calibration mode active. Click two points on the image to define a known distance.")
        print("Calibrate Image triggered. Calibration mode active.")

    def _on_calibration_points_selected(self, p1_pixel: Point2D, p2_pixel: Point2D) -> None:
        """
        Slot to handle the selection of two calibration points on the canvas.
        Prompts user for real-world distance and creates CalibrationModel.
        """
        self.status_bar.set_status_message("Two calibration points selected. Enter real-world distance.")
        print(f"Calibration points selected: {p1_pixel}, {p2_pixel}")

        # Prompt user for real-world distance
        # QInputDialog.getDouble in PySide6 may not accept keyword arguments; use positional args:
        # (parent, title, label, value=1.0, min=0.01, max=10000.0, decimals=2)
        distance_meters, ok = QInputDialog.getDouble(
            self,
            "Enter Real-World Distance",
            "Enter the real-world distance (in meters) between the two selected points:",
            1.0,  # default value (meters)
            0.01,  # min
            10000.0,  # max
            2  # decimals
        )

        if ok and distance_meters > 0:
            try:
                calibration = CalibrationService.calibrate_from_distance(
                    p1_pixel, p2_pixel, distance_meters
                )
                self.geometry_controller.set_calibration_model(calibration)
                self.ai_controller.set_calibration_model(calibration)
                self.status_bar.set_status_message(f"Calibration successful: {calibration.scale_factor_pixels_per_meter:.2f} px/m.")
                print(f"Calibration successful: {calibration}")
            except ValueError as e:
                self.status_bar.set_status_message(f"Calibration failed: {e}")
                QMessageBox.critical(self, "Calibration Error", f"Failed to calibrate: {e}")
        else:
            self.status_bar.set_status_message("Calibration cancelled or invalid distance entered.")
            QMessageBox.warning(self, "Calibration Cancelled", "Calibration was cancelled or an invalid distance was entered.")
        
        self.workspace.roof_canvas.set_calibration_mode(False) # Deactivate calibration mode

    def _on_materials(self) -> None:
        self.status_bar.set_status_message("Action: Materials")
        print("Materials triggered")

    def _on_about(self) -> None:
        self.status_bar.set_status_message("Action: About")
        print("About triggered")

    def _on_measure(self) -> None:
        self.status_bar.set_status_message("Action: Measure Tool Activated")
        print("Measure tool triggered")

    def _on_open_image(self) -> None:
        """
        Slot to handle the "Open Image" menu action.
        Opens a file dialog and passes the selected image path to the ImageController.
        """
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.tiff *.tif)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                image_path = Path(selected_files[0])
                self.image_controller.load_image(image_path)

    def _on_image_displayed_on_canvas(self, image_info: ImageInfo) -> None:
        """
        Slot to react when an image is successfully displayed on the canvas.
        Updates the status bar and sets up a placeholder calibration.
        """
        self.status_bar.set_status_message(f"Image '{image_info.file_path.name}' loaded. Dimensions: {image_info.width}x{image_info.height}")
        print(f"Image displayed: {image_info.file_path.name}")

        # Placeholder calibration is removed, now user must calibrate manually
        self.status_bar.set_status_message(f"Image loaded. Please calibrate the image using 'Calibrate Image' tool.")
        # Clear any previous calibration in controllers
        self.geometry_controller.set_calibration_model(None)
        self.ai_controller.set_calibration_model(None)


    def _on_image_cleared_from_canvas(self) -> None:
        """
        Slot to react when the image is cleared from the canvas.
        Updates the status bar and clears drawing visuals.
        """
        self.status_bar.set_status_message("Canvas cleared.")
        self.geometry_controller.clear_drawing()
        self.workspace.roof_canvas.set_drawing_mode(False)
        self.workspace.roof_canvas.set_calibration_mode(False) # Ensure calibration mode is off
        self.workspace.roof_canvas.clear_ai_overlay_visuals()
        print("Image cleared from canvas.")

    def _on_drawing_mode_changed(self, active: bool) -> None:
        """
        Slot to react when the drawing mode is activated/deactivated on the canvas.
        """
        self.status_bar.set_status_message(f"Drawing mode {'activated' if active else 'deactivated'}.")
        print(f"Drawing mode changed to: {active}")

    def _on_roof_geometry_created(self, roof_geometry: RoofGeometry) -> None:
        """
        Slot to react when a RoofGeometry object is created by the GeometryController.
        """
        self.status_bar.set_status_message(f"Roof Geometry created. Total area: {roof_geometry.calculate_total_area():.2f} sq m.")
        print(f"Received RoofGeometry: {roof_geometry}")

    def _on_measurements_calculated(self, measurements) -> None:
        """
        Slot to react when roof measurements are calculated.
        """
        self.status_bar.set_status_message(f"Measurements: Total Area = {measurements.total_area_m2:.2f} sq m, Perimeter = {measurements.total_perimeter_m:.2f} m.")
        print(f"Received Measurements: {measurements}")

    def _on_materials_calculated(self, material_results: List[MaterialCalculationResult]) -> None:
        """
        Slot to react when material calculations are performed.
        """
        total_cost = sum(res.estimated_cost for res in material_results)
        self.status_bar.set_status_message(f"Materials calculated. Total estimated cost: ${total_cost:.2f}")
        print(f"Received Material Calculations: {material_results}")

    def _on_ai_analysis_completed(self, roof_geometry: RoofGeometry, raw_ai_results: List[Union[DetectionResult, SegmentationResult]]) -> None:
        """
        Slot to react when AI analysis is completed.
        Displays the AI results as an overlay on the canvas.
        """
        self.status_bar.set_status_message("AI Analysis completed. Displaying results.")
        self.workspace.roof_canvas.display_ai_results_overlay(raw_ai_results)
        print(f"AI Analysis completed. Generated RoofGeometry: {roof_geometry}")

    def _on_toggle_ai_overlay(self, checked: bool) -> None:
        """
        Slot to toggle the visibility of the AI overlay.
        """
        self.workspace.roof_canvas.set_ai_overlay_mode(checked)
        if not checked:
            self.workspace.roof_canvas.clear_ai_overlay_visuals()
        self.status_bar.set_status_message(f"AI Overlay: {'ON' if checked else 'OFF'}")
        print(f"AI Overlay toggled: {checked}")
