"""
This module defines the GeometryController, responsible for managing interactive
geometry editing and mediating between the UI and the geometry engine.
"""

from typing import List, Optional, Tuple
import uuid

from PySide6.QtCore import QObject, Signal, QPointF

from app.geometry.point import Point2D, Point3D
from app.geometry.edge import Edge
from app.geometry.polygon import Polygon2D
from app.geometry.plane import RoofPlane
from app.geometry.roof_geometry import RoofGeometry
from app.geometry.calibration import CalibrationModel, CalibrationService
from app.services.measurement_service import RoofMeasurementService, RoofMeasurementResult
from app.materials.material_calculator import MaterialCalculator
from app.materials.calculation_result import MaterialCalculationResult
from app.materials.material_repository import MaterialRepository # New import
from app.materials.roof_system_model import RoofSystem, RoofLayer # New import for placeholder
from app.database.enums import MaterialUnit # New import for placeholder material
from app.core.logger import setup_logging

logger = setup_logging()

class GeometryController(QObject):
    """
    Controller for handling interactive geometry creation and manipulation on the canvas.
    It manages the current drawing state and converts UI pixel coordinates
    into domain-specific geometry objects.
    """

    # Signals to update the UI (RoofCanvas) with current drawing state
    polygon_drawing_updated = Signal(list) # Emits List[Point2D] (pixel coordinates)
    drawing_cleared = Signal()

    # Signals to notify other parts of the application about finalized geometry
    polygon_finalized = Signal(Polygon2D) # Emits Polygon2D (real-world coordinates)
    roof_geometry_created = Signal(RoofGeometry)
    measurements_calculated = Signal(RoofMeasurementResult)
    materials_calculated = Signal(list) # New signal: Emits List[MaterialCalculationResult]

    # Signals for status/error messages
    status_message = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, material_repository: MaterialRepository, parent: Optional[QObject] = None):
        """
        Initializes the GeometryController.

        Args:
            material_repository (MaterialRepository): Repository for material data.
            parent (Optional[QObject]): The parent QObject.
        """
        super().__init__(parent)
        self._current_pixel_points: List[Point2D] = []
        self._current_calibration: Optional[CalibrationModel] = None
        self._is_drawing_active: bool = False
        self._measurement_service = RoofMeasurementService()
        self._material_calculator = MaterialCalculator(material_repository) # Instantiate material calculator
        logger.info("GeometryController initialized.")

    @property
    def is_drawing_active(self) -> bool:
        """Returns True if drawing mode is currently active."""
        return self._is_drawing_active

    def start_drawing(self) -> None:
        """
        Activates the drawing mode.
        """
        self._is_drawing_active = True
        self._current_pixel_points.clear()
        self.drawing_cleared.emit() # Clear any previous drawing visuals
        self.status_message.emit("Drawing mode activated. Click to add points.")
        logger.info("Geometry drawing mode started.")

    def stop_drawing(self) -> None:
        """
        Deactivates the drawing mode without finalizing the current polygon.
        """
        self._is_drawing_active = False
        self._current_pixel_points.clear()
        self.drawing_cleared.emit()
        self.status_message.emit("Drawing mode deactivated.")
        logger.info("Geometry drawing mode stopped.")

    def set_calibration_model(self, calibration: CalibrationModel) -> None:
        """
        Sets the calibration model to be used for converting pixel to real-world coordinates.

        Args:
            calibration (CalibrationModel): The active calibration model.
        """
        self._current_calibration = calibration
        self.status_message.emit(f"Calibration set: {calibration.scale_factor_pixels_per_meter:.2f} px/m")
        logger.info(f"Calibration model set in GeometryController: {calibration.scale_factor_pixels_per_meter} px/m")

    def add_point(self, pixel_point_qf: QPointF) -> None:
        """
        Adds a new point to the current polygon being drawn.

        Args:
            pixel_point_qf (QPointF): The point in pixel coordinates from the canvas.
        """
        if not self._is_drawing_active:
            self.error_occurred.emit("Drawing mode is not active.")
            return

        new_point = Point2D(pixel_point_qf.x(), pixel_point_qf.y())
        self._current_pixel_points.append(new_point)
        self.polygon_drawing_updated.emit(self._current_pixel_points)
        self.status_message.emit(f"Point added: ({new_point.x:.1f}, {new_point.y:.1f}). Total: {len(self._current_pixel_points)}")
        logger.debug(f"Added point {new_point} to drawing.")

    def move_point(self, index: int, new_pixel_point_qf: QPointF) -> None:
        """
        Moves an existing point in the current polygon.

        Args:
            index (int): The index of the point to move.
            new_pixel_point_qf (QPointF): The new position of the point in pixel coordinates.
        """
        if not self._is_drawing_active:
            self.error_occurred.emit("Drawing mode is not active.")
            return
        if not (0 <= index < len(self._current_pixel_points)):
            self.error_occurred.emit(f"Invalid point index: {index}")
            return

        new_point = Point2D(new_pixel_point_qf.x(), new_pixel_point_qf.y())
        self._current_pixel_points[index] = new_point
        self.polygon_drawing_updated.emit(self._current_pixel_points)
        self.status_message.emit(f"Point {index} moved to ({new_point.x:.1f}, {new_point.y:.1f})")
        logger.debug(f"Moved point {index} to {new_point}.")

    def finalize_polygon(self) -> None:
        """
        Finalizes the current polygon drawing, converts it to real-world coordinates,
        creates a RoofGeometry object, and calculates measurements.
        """
        if not self._is_drawing_active:
            self.error_occurred.emit("Drawing mode is not active.")
            return
        if len(self._current_pixel_points) < 3:
            self.error_occurred.emit("A polygon must have at least 3 points.")
            return
        if self._current_calibration is None:
            self.error_occurred.emit("Cannot finalize polygon: No calibration model set.")
            return

        logger.info("Finalizing polygon and creating RoofGeometry.")
        pixel_polygon = Polygon2D(vertices=self._current_pixel_points)

        # Convert pixel points to real-world (meter) points
        real_world_vertices: List[Point2D] = []
        for p_pixel in pixel_polygon.vertices:
            x_meter = CalibrationService.pixel_to_meter(p_pixel.x, self._current_calibration)
            y_meter = CalibrationService.pixel_to_meter(p_pixel.y, self._current_calibration)
            real_world_vertices.append(Point2D(x_meter, y_meter))

        real_world_polygon = Polygon2D(vertices=real_world_vertices)
        self.polygon_finalized.emit(real_world_polygon)
        self.status_message.emit(f"Polygon finalized. Area: {real_world_polygon.area:.2f} sq m.")

        plane_name = f"RoofPlane_{uuid.uuid4().hex[:8]}"
        heights = [0.0] * len(real_world_polygon.vertices)
        roof_plane = RoofPlane(
            name=plane_name,
            polygon=real_world_polygon,
            slope=30.0,
            orientation=0.0,
            height_at_vertices=heights
        )
        
        vertices_3d = [Point3D(p.x, p.y, h) for p, h in zip(real_world_polygon.vertices, heights)]

        roof_geometry = RoofGeometry(
            vertices=vertices_3d,
            edges=[],
            planes=[roof_plane],
            ridges=[],
            valleys=[],
            openings=[]
        )
        self.roof_geometry_created.emit(roof_geometry)
        self.status_message.emit("RoofGeometry created from polygon.")
        logger.info(f"RoofGeometry created: {roof_geometry}")

        # Calculate measurements
        measurements = self.calculate_measurements(roof_geometry)

        # Calculate materials if measurements are available
        if measurements:
            self.calculate_materials(measurements)

        self.stop_drawing() # Deactivate drawing mode after finalizing

    def clear_drawing(self) -> None:
        """
        Clears the current drawing points and notifies the UI.
        """
        self._current_pixel_points.clear()
        self.drawing_cleared.emit()
        self.status_message.emit("Current drawing cleared.")
        logger.debug("Current geometry drawing cleared.")

    def calculate_measurements(self, roof_geometry: RoofGeometry) -> Optional[RoofMeasurementResult]:
        """
        Calculates real-world measurements for the given RoofGeometry and emits the result.
        """
        if self._current_calibration is None:
            self.error_occurred.emit("Cannot calculate measurements: No calibration model set.")
            return None

        try:
            measurements = self._measurement_service.calculate_roof_statistics(
                roof_geometry=roof_geometry,
                calibration=self._current_calibration # Pass calibration if geometry was pixel-derived
            )
            self.measurements_calculated.emit(measurements)
            self.status_message.emit(f"Measurements calculated: Total Area = {measurements.total_area_m2:.2f} sq m.")
            logger.info(f"Measurements calculated: {measurements}")
            return measurements
        except Exception as e:
            self.error_occurred.emit(f"Error calculating measurements: {e}")
            logger.error(f"Error calculating measurements: {e}")
            return None

    def calculate_materials(self, roof_measurement: RoofMeasurementResult) -> None:
        """
        Calculates material quantities and costs based on roof measurements.
        """
        try:
            # Placeholder: Create some dummy materials and a roof system for testing
            # In a real application, these would come from the database or user selection.
            dummy_category = MaterialCategory(id=uuid.uuid4(), name="Roofing Tiles")
            dummy_manufacturer = MaterialManufacturer(id=uuid.uuid4(), name="Generic Mfg")

            # Ensure the material repository has these materials
            # For a real app, you'd fetch existing materials or create them if needed.
            # For this example, we'll create them directly if they don't exist.
            # This part would typically be handled by a MaterialService.
            
            # Example: Create a dummy material for covering
            covering_material_id = uuid.uuid4()
            covering_material = Material(
                id=covering_material_id,
                name="Standard Roof Tile",
                category=dummy_category,
                manufacturer=dummy_manufacturer,
                unit=MaterialUnit.SQUARE_FOOT,
                price=0.5, # $0.5 per sq ft
                coverage=1.0, # 1 sq ft per unit
                waste_factor=0.10, # 10% waste
                is_active=True
            )
            # Add to repository (this would normally be done once, e.g., on app startup or via admin UI)
            try:
                self._material_calculator._material_repository.add_material(covering_material)
            except ValueError:
                pass # Already exists, ignore

            # Example: Create a dummy material for membrane
            membrane_material_id = uuid.uuid4()
            membrane_material = Material(
                id=membrane_material_id,
                name="Underlayment Membrane",
                category=MaterialCategory(id=uuid.uuid4(), name="Underlayment"),
                manufacturer=dummy_manufacturer,
                unit=MaterialUnit.SQUARE_FOOT,
                price=0.2, # $0.2 per sq ft
                coverage=1.0, # 1 sq ft per unit
                waste_factor=0.05, # 5% waste
                is_active=True
            )
            try:
                self._material_calculator._material_repository.add_material(membrane_material)
            except ValueError:
                pass # Already exists, ignore

            # Example: Create a dummy roof system
            roof_system = RoofSystem(
                id=uuid.uuid4(),
                name="Standard Pitched Roof System",
                layers=[
                    RoofLayer(name="Roof Covering", material=covering_material, order=1),
                    RoofLayer(name="Underlayment", material=membrane_material, order=2)
                ]
            )

            material_results = self._material_calculator.calculate_roof_system_materials(
                roof_measurement=roof_measurement,
                roof_system=roof_system
            )
            total_cost = self._material_calculator.calculate_total_cost(material_results)

            self.materials_calculated.emit(material_results)
            self.status_message.emit(f"Materials calculated. Total estimated cost: ${total_cost:.2f}")
            logger.info(f"Material calculation results: {material_results}, Total Cost: ${total_cost:.2f}")
        except Exception as e:
            self.error_occurred.emit(f"Error calculating materials: {e}")
            logger.error(f"Error calculating materials: {e}")
