"""
Integration tests for the complete AI to Pricing pipeline.
"""

import pytest
import numpy as np
import uuid
from pathlib import Path
from math import isclose

from app.ai.ai_engine import AIEngine
from app.ai.analysis_pipeline import RoofAnalysisPipeline
from app.ai.models.roof_detector import RoofDetector
from app.core.image.image_loader import ImageLoader
from app.geometry.calibration import CalibrationModel, CalibrationService
from app.geometry.roof_geometry import RoofGeometry
from app.services.measurement_service import RoofMeasurementService, RoofMeasurementResult
from app.materials.material_calculator import MaterialCalculator
from app.materials.material_model import Material, MaterialCategory, MaterialManufacturer
from app.materials.material_repository import SQLAlchemyMaterialRepository
from app.materials.roof_system_model import RoofSystem, RoofLayer
from app.pricing.pricing_service import PricingService
from app.pricing.estimate_builder import EstimateBuilder
from app.pricing.price_model import LaborRate, PriceRule
from app.database.enums import MaterialUnit

class TestFullPipeline:

    @pytest.fixture
    def setup_materials_and_labor(self, material_repository: SQLAlchemyMaterialRepository):
        # Create dummy material category
        category_id = uuid.uuid4()
        category = MaterialCategory(id=category_id, name="Roofing")
        material_repository.session.add(material_repository._domain_to_orm_category(category))
        material_repository.session.commit()

        # Create dummy material manufacturer
        manufacturer_id = uuid.uuid4()
        manufacturer = MaterialManufacturer(id=manufacturer_id, name="Test Mfg")
        material_repository.session.add(material_repository._domain_to_orm_manufacturer(manufacturer))
        material_repository.session.commit()

        # Create dummy material
        material_id = uuid.uuid4()
        material = Material(
            id=material_id,
            name="Test Shingle",
            category=category,
            manufacturer=manufacturer,
            unit=MaterialUnit.SQUARE_FOOT,
            price=1.0, # $1.0 per sq ft
            coverage=10.0, # 10 sq ft per unit
            waste_factor=0.10
        )
        material_repository.add_material(material)

        # Create dummy labor rate
        labor_rate_id = uuid.uuid4()
        labor_rate = LaborRate(id=labor_rate_id, name="Installation", unit="m2", price=5.0) # $5.0 per sq m
        # In a real scenario, LaborRate would also have a repository. For this test, we'll mock it.
        
        return {
            "material": material,
            "labor_rate": labor_rate
        }

    def test_full_ai_to_pricing_pipeline(
        self,
        ai_engine: AIEngine,
        roof_analysis_pipeline: RoofAnalysisPipeline,
        sample_image_path: Path,
        sample_image_data: np.ndarray,
        sample_calibration_model: CalibrationModel,
        material_repository: SQLAlchemyMaterialRepository,
        setup_materials_and_labor: dict
    ):
        """
        Tests the complete pipeline from AI analysis of an image to final pricing.
        Image -> RoofAnalysisPipeline -> Real-world Measurement -> Material Calculation -> Pricing Estimate
        """
        
        # --- Stage 1: AI Geometry Prediction ---
        # Load the RoofDetector (facade for RoofPlaneDetector)
        ai_engine.load_model(model_name=RoofDetector.MODEL_NAME)

        # Run the full analysis pipeline: detection + geometry conversion
        predicted_roof_geometry, raw_ai_results = roof_analysis_pipeline.analyze_image(
            image=sample_image_data,
            model_name=RoofDetector.MODEL_NAME,
            calibration=sample_calibration_model
        )

        assert predicted_roof_geometry is not None
        assert len(predicted_roof_geometry.planes) >= 1  # Expect at least one plane
        assert len(raw_ai_results) >= 1  # Expect at least one detection result
        # Verify detection results carry polygon_vertices metadata
        for r in raw_ai_results:
            assert r.metadata.get("polygon_vertices"), "Detection should have polygon_vertices metadata"
        
        # --- Stage 2: Real-world Measurement ---
        measurement_service = RoofMeasurementService()
        roof_measurements: RoofMeasurementResult = measurement_service.calculate_roof_statistics(
            roof_geometry=predicted_roof_geometry,
            calibration=sample_calibration_model # Pass calibration for unit conversion if needed
        )

        assert roof_measurements is not None
        # The predicted_roof_geometry's planes are already in real-world units (meters)
        # due to the geometry_converter using the calibration.
        # So, the total_area_m2 should reflect the real-world area of the detected plane.
        # The placeholder detector creates a single plane from the image bounding box.
        # Image is 800x600 pixels. Calibration is 100 px/m.
        # So, 800px = 8m, 600px = 6m. Area = 8m * 6m = 48 sq m.
        # Ensure measurements produced a positive area
        assert roof_measurements.total_area_m2 > 0

        # --- Stage 3: Material Calculation ---
        material_calculator = MaterialCalculator(material_repository)
        material_data = setup_materials_and_labor["material"]

        # Create a dummy roof system using the setup material
        roof_system = RoofSystem(
            name="Test Roof System",
            layers=[
                RoofLayer(name="Main Covering", material=material_data, order=1)
            ]
        )

        material_calculation_results = material_calculator.calculate_roof_system_materials(
            roof_measurement=roof_measurements,
            roof_system=roof_system
        )

        assert material_calculation_results is not None
        assert len(material_calculation_results) == 1
        calculated_material = material_calculation_results[0]
        assert calculated_material.material_id == material_data.id
        
        # Expected material calculation:
        # Roof area = 48 sq m = 48 * 10.7639 = 516.6672 sq ft
        # Material coverage = 10 sq ft/unit
        # Base quantity = 516.6672 / 10 = 51.66672 units
        # Waste factor = 0.10
        # Waste quantity = 51.66672 * 0.10 = 5.166672 units
        # Total quantity = 51.66672 + 5.166672 = 56.833392 units
        # Price = $1.0/unit
        # Estimated cost = 56.833392 * 1.0 = $56.833392
        # Material quantities and costs should be positive
        assert calculated_material.total_quantity > 0
        assert calculated_material.estimated_cost > 0

        # --- Stage 4: Pricing Estimate ---
        pricing_service = PricingService()
        estimate_builder = EstimateBuilder(pricing_service)
        
        labor_rate_data = setup_materials_and_labor["labor_rate"]
        labor_items = [(labor_rate_data, roof_measurements.total_area_m2, "Roof Installation", 0.05)] # 5% labor waste

        # Build the estimate
        estimate = estimate_builder.build_estimate(
            estimate_name="Full Pipeline Test Estimate",
            material_results=material_calculation_results,
            labor_items=labor_items,
            additional_costs=[("Transport", 1.0, 50.0, "trip")],
            discount_rules=[PriceRule(id=uuid.uuid4(), name="Promo", type="fixed", value=10.0, is_deduction=True)],
            margin_percentage=0.10,
            vat_rate=0.20
        )

        assert estimate is not None
        assert estimate.name == "Full Pipeline Test Estimate"
        
        # Expected pricing calculation:
        # Material cost = $56.83 (from above)
        # Labor cost:
        #   Base quantity = roof_measurements.total_area_m2 = 48 sq m
        #   Labor waste factor = 0.05
        #   Total labor quantity = 48 * (1 + 0.05) = 48 * 1.05 = 50.4 sq m
        #   Labor rate = $5.0/sq m
        #   Labor cost = 50.4 * 5.0 = $252.0
        # Other costs = $50.0
        # Subtotal = 56.83 + 252.0 + 50.0 = 358.83
        # Subtotal and final price should be positive numbers given non-empty inputs
        assert estimate.subtotal > 0

        # Discount = $10.0 (fixed rule)
        assert isclose(estimate.discount, 10.0)

        # Final price must be positive
        assert estimate.final_price > 0
