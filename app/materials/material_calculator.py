"""
This module provides the MaterialCalculator service for estimating material quantities and costs.
"""

from typing import List, Optional
import uuid

from app.materials.material_model import Material, MaterialCategory
from app.materials.material_repository import MaterialRepository
from app.materials.calculation_result import MaterialCalculationResult
from app.materials.roof_system_model import RoofSystem, RoofLayer
from app.services.measurement_service import RoofMeasurementResult
from app.database.enums import MaterialUnit
from app.core.logger import setup_logging

logger = setup_logging()

class MaterialCalculator:
    """
    Service for calculating required material quantities and estimated costs
    based on roof measurements and material properties.
    """

    def __init__(self, material_repository: MaterialRepository):
        """
        Initializes the MaterialCalculator with a material repository.

        Args:
            material_repository (MaterialRepository): Repository for accessing material data.
        """
        self._material_repository = material_repository

    def _calculate_single_material(
        self,
        material: Material,
        required_area_sq_m: Optional[float] = None,
        required_length_m: Optional[float] = None,
        required_pieces: Optional[int] = None
    ) -> MaterialCalculationResult:
        """
        Calculates the quantity and cost for a single material.
        """
        if material.coverage is None or material.coverage <= 0:
            logger.warning(f"Material {material.name} (ID: {material.id}) has no defined coverage. Skipping calculation.")
            return MaterialCalculationResult(
                material_id=material.id,
                material_name=material.name,
                quantity=0.0, unit=material.unit, waste_quantity=0.0, total_quantity=0.0, estimated_cost=0.0
            )

        base_quantity = 0.0
        if material.unit == MaterialUnit.SQUARE_FOOT:
            if required_area_sq_m is None:
                raise ValueError(f"Required area (sq m) is needed for {material.name} (unit: {material.unit}).")
            required_area_sq_ft = required_area_sq_m * 10.7639
            base_quantity = required_area_sq_ft / material.coverage
        elif material.unit == MaterialUnit.LINEAR_FOOT:
            if required_length_m is None:
                raise ValueError(f"Required length (m) is needed for {material.name} (unit: {material.unit}).")
            required_length_ft = required_length_m * 3.28084
            base_quantity = required_length_ft / material.coverage
        elif material.unit == MaterialUnit.EACH:
            if required_pieces is None:
                raise ValueError(f"Required pieces is needed for {material.name} (unit: {material.unit}).")
            base_quantity = float(required_pieces)
        elif material.unit == MaterialUnit.BUNDLE:
            if required_area_sq_m is None:
                raise ValueError(f"Required area (sq m) is needed for {material.name} (unit: {material.unit}).")
            required_area_sq_ft = required_area_sq_m * 10.7639
            base_quantity = required_area_sq_ft / material.coverage
        else:
            logger.warning(f"Unsupported material unit {material.unit} for {material.name}. Skipping calculation.")
            return MaterialCalculationResult(
                material_id=material.id,
                material_name=material.name,
                quantity=0.0, unit=material.unit, waste_quantity=0.0, total_quantity=0.0, estimated_cost=0.0
            )

        waste_quantity = base_quantity * material.waste_factor
        total_quantity = base_quantity + waste_quantity
        estimated_cost = total_quantity * material.price

        return MaterialCalculationResult(
            material_id=material.id,
            material_name=material.name,
            quantity=base_quantity,
            unit=material.unit,
            waste_quantity=waste_quantity,
            total_quantity=total_quantity,
            estimated_cost=estimated_cost
        )

    def calculate_covering(
        self,
        roof_measurement: RoofMeasurementResult,
        covering_material_id: uuid.UUID
    ) -> MaterialCalculationResult:
        """
        Calculates the quantity and cost for the main roof covering material.

        Args:
            roof_measurement (RoofMeasurementResult): The measured roof statistics.
            covering_material_id (uuid.UUID): The ID of the covering material.

        Returns:
            MaterialCalculationResult: The calculation result for the covering material.
        """
        material = self._material_repository.get_material(covering_material_id)
        if not material:
            raise ValueError(f"Covering material with ID {covering_material_id} not found.")
        
        # Assuming covering material is typically measured by area
        return self._calculate_single_material(
            material,
            required_area_sq_m=roof_measurement.total_area_m2
        )

    def calculate_membrane(
        self,
        roof_measurement: RoofMeasurementResult,
        membrane_material_id: uuid.UUID
    ) -> MaterialCalculationResult:
        """
        Calculates the quantity and cost for the roof membrane material.

        Args:
            roof_measurement (RoofMeasurementResult): The measured roof statistics.
            membrane_material_id (uuid.UUID): The ID of the membrane material.

        Returns:
            MaterialCalculationResult: The calculation result for the membrane material.
        """
        material = self._material_repository.get_material(membrane_material_id)
        if not material:
            raise ValueError(f"Membrane material with ID {membrane_material_id} not found.")
        
        # Membrane usually covers the same area as the roof covering
        return self._calculate_single_material(
            material,
            required_area_sq_m=roof_measurement.total_area_m2
        )

    def calculate_battens(
        self,
        roof_measurement: RoofMeasurementResult,
        batten_material_id: uuid.UUID,
        spacing_m: float = 0.3 # Example spacing in meters
    ) -> MaterialCalculationResult:
        """
        Calculates the quantity and cost for roof battens.

        Args:
            roof_measurement (RoofMeasurementResult): The measured roof statistics.
            batten_material_id (uuid.UUID): The ID of the batten material.
            spacing_m (float): The spacing between battens in meters.

        Returns:
            MaterialCalculationResult: The calculation result for the batten material.
        """
        material = self._material_repository.get_material(batten_material_id)
        if not material:
            raise ValueError(f"Batten material with ID {batten_material_id} not found.")
        
        if spacing_m <= 0:
            raise ValueError("Batten spacing must be positive.")

        # Simplified calculation: assume battens run horizontally across the roof planes
        # Total length of battens = sum(plane_width / spacing) * plane_length
        # This is a very rough estimate and would need more sophisticated geometry for accuracy.
        # For now, let's use total perimeter as a proxy for linear materials.
        # A more accurate calculation would involve iterating over roof planes and their dimensions.
        
        # Placeholder: Assume total linear feet needed is proportional to total perimeter
        # Or, if we have a total area, and a typical batten length per area unit.
        # For simplicity, let's assume battens are needed for the total area,
        # and we need to cover the area with battens at a certain spacing.
        # Required linear meters = (total_area_m2 / spacing_m) + some overlap/edge factor
        
        # Let's use a simpler approach: assume battens are needed for the total perimeter
        # and some additional for internal structure.
        # For now, a simple multiplier of the total perimeter.
        required_linear_m = roof_measurement.total_perimeter_m * 1.5 # Arbitrary multiplier

        return self._calculate_single_material(
            material,
            required_length_m=required_linear_m
        )

    def calculate_accessories(
        self,
        roof_measurement: RoofMeasurementResult,
        accessory_material_ids: List[uuid.UUID]
    ) -> List[MaterialCalculationResult]:
        """
        Calculates the quantity and cost for various accessory materials.

        Args:
            roof_measurement (RoofMeasurementResult): The measured roof statistics.
            accessory_material_ids (List[uuid.UUID]): List of IDs for accessory materials.

        Returns:
            List[MaterialCalculationResult]: A list of calculation results for accessories.
        """
        results: List[MaterialCalculationResult] = []
        for material_id in accessory_material_ids:
            material = self._material_repository.get_material(material_id)
            if not material:
                logger.warning(f"Accessory material with ID {material_id} not found. Skipping.")
                continue
            
            # This is highly dependent on the accessory type.
            # For example, ridge caps might be based on ridge length,
            # vents might be a fixed number, flashing based on perimeter.
            # For this placeholder, we'll make a very simple assumption.
            
            if material.unit == MaterialUnit.EACH:
                # Assume 1 piece per 100 sq m of roof area, or a fixed number
                required_pieces = max(1, int(roof_measurement.total_area_m2 / 100)) # At least one
                results.append(self._calculate_single_material(material, required_pieces=required_pieces))
            elif material.unit == MaterialUnit.LINEAR_FOOT:
                # Assume linear accessories are proportional to total perimeter
                required_length_m = roof_measurement.total_perimeter_m * 0.2 # 20% of perimeter
                results.append(self._calculate_single_material(material, required_length_m=required_length_m))
            else:
                logger.warning(f"Accessory material {material.name} has unsupported unit {material.unit} for generic calculation.")
                results.append(MaterialCalculationResult(
                    material_id=material.id, material_name=material.name,
                    quantity=0.0, unit=material.unit, waste_quantity=0.0, total_quantity=0.0, estimated_cost=0.0
                ))
        return results

    def calculate_roof_system_materials(
        self,
        roof_measurement: RoofMeasurementResult,
        roof_system: RoofSystem
    ) -> List[MaterialCalculationResult]:
        """
        Calculates all materials required for a given roof system based on measurements.

        Args:
            roof_measurement (RoofMeasurementResult): The measured roof statistics.
            roof_system (RoofSystem): The defined roof system with its layers.

        Returns:
            List[MaterialCalculationResult]: A list of calculation results for all materials in the system.
        """
        all_results: List[MaterialCalculationResult] = []

        for layer in roof_system.layers:
            if layer.material:
                # This logic needs to be much more sophisticated in a real app,
                # mapping layer type to how its material quantity is derived from roof measurements.
                # For now, a simplified approach:
                if layer.material.unit in [MaterialUnit.SQUARE_FOOT, MaterialUnit.BUNDLE]:
                    all_results.append(self._calculate_single_material(
                        layer.material, required_area_sq_m=roof_measurement.total_area_m2
                    ))
                elif layer.material.unit == MaterialUnit.LINEAR_FOOT:
                    # For linear materials, use total perimeter as a rough estimate
                    all_results.append(self._calculate_single_material(
                        layer.material, required_length_m=roof_measurement.total_perimeter_m
                    ))
                elif layer.material.unit == MaterialUnit.EACH:
                    # For individual pieces, assume a fixed number or based on area
                    required_pieces = max(1, int(roof_measurement.total_area_m2 / 50)) # 1 piece per 50 sq m
                    all_results.append(self._calculate_single_material(
                        layer.material, required_pieces=required_pieces
                    ))
                else:
                    logger.warning(f"Material {layer.material.name} in roof system has unsupported unit {layer.material.unit}. Skipping.")
            else:
                logger.debug(f"Roof system layer '{layer.name}' has no material defined.")

        return all_results

    def calculate_total_cost(self, calculation_results: List[MaterialCalculationResult]) -> float:
        """
        Calculates the total estimated cost from a list of material calculation results.

        Args:
            calculation_results (List[MaterialCalculationResult]): List of individual material calculation results.

        Returns:
            float: The total estimated cost.
        """
        return sum(res.estimated_cost for res in calculation_results)
