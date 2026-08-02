"""
Tests for the app.services.roof_measurement_service module.
"""

import pytest
from unittest.mock import MagicMock
from math import isclose

from app.geometry.point import Point2D
from app.geometry.calibration import CalibrationModel
from app.geometry.roof_area import RoofAreaCalculator, RoofPolygonMetrics
from app.services.roof_measurement_service import RoofMeasurementService

class TestRoofMeasurementService:

    @pytest.fixture
    def mock_roof_area_calculator(self):
        """Fixture that provides a mocked RoofAreaCalculator."""
        mock_calculator = MagicMock(spec=RoofAreaCalculator)
        # Configure the mock to return a predefined metrics object
        mock_calculator.calculate.return_value = RoofPolygonMetrics(
            area_m2=100.0,
            perimeter_m=40.0,
            centroid=Point2D(5.0, 5.0),
            vertex_count=4
        )
        return mock_calculator

    @pytest.fixture
    def real_roof_area_calculator(self):
        """Fixture that provides a real RoofAreaCalculator."""
        return RoofAreaCalculator()

    @pytest.fixture
    def sample_calibration_model(self):
        """Fixture that provides a sample CalibrationModel."""
        return CalibrationModel(
            reference_points_pixel=(Point2D(0, 0), Point2D(100, 0)),
            reference_distance_meters=1.0,
            scale_factor_pixels_per_meter=100.0,
            unit="m"
        )

    def test_measure_from_points_with_mock_calculator(self, mock_roof_area_calculator, sample_calibration_model):
        """
        Test measure_from_points using a mocked RoofAreaCalculator.
        This verifies the service correctly converts points and calls the calculator.
        """
        service = RoofMeasurementService(mock_roof_area_calculator)
        pixel_points = [Point2D(0, 0), Point2D(100, 0), Point2D(100, 100), Point2D(0, 100)]

        metrics = service.measure_from_points(pixel_points, sample_calibration_model)

        # Verify that the mock calculator's calculate method was called
        mock_roof_area_calculator.calculate.assert_called_once()

        # Check the arguments passed to the mock calculator
        called_points = mock_roof_area_calculator.calculate.call_args[0][0]
        assert len(called_points) == 4
        assert isclose(called_points[0].x, 0.0)
        assert isclose(called_points[0].y, 0.0)
        assert isclose(called_points[1].x, 1.0) # 100 pixels / 100 px/m = 1 meter
        assert isclose(called_points[1].y, 0.0)
        assert isclose(called_points[2].x, 1.0)
        assert isclose(called_points[2].y, 1.0)
        assert isclose(called_points[3].x, 0.0)
        assert isclose(called_points[3].y, 1.0)

        # Verify the returned metrics are from the mock
        assert metrics.area_m2 == 100.0
        assert metrics.perimeter_m == 40.0

    def test_measure_from_points_with_real_calculator(self, real_roof_area_calculator, sample_calibration_model):
        """
        Test measure_from_points using a real RoofAreaCalculator.
        This verifies the end-to-end calculation from pixels to real-world metrics.
        """
        service = RoofMeasurementService(real_roof_area_calculator)
        # A 100x100 pixel square, with 100 px/m calibration, should be a 1x1 meter square
        pixel_points = [Point2D(0, 0), Point2D(100, 0), Point2D(100, 100), Point2D(0, 100)]

        metrics = service.measure_from_points(pixel_points, sample_calibration_model)

        assert isclose(metrics.area_m2, 1.0)
        assert isclose(metrics.perimeter_m, 4.0)
        assert isclose(metrics.centroid.x, 0.5)
        assert isclose(metrics.centroid.y, 0.5)
        assert metrics.vertex_count == 4

    def test_measure_from_points_less_than_3_points(self, real_roof_area_calculator, sample_calibration_model):
        """Test that ValueError is raised for polygons with less than 3 points."""
        service = RoofMeasurementService(real_roof_area_calculator)
        pixel_points = [Point2D(0, 0), Point2D(100, 0)]
        with pytest.raises(ValueError, match="Polygón potrebuje aspoň 3 body"):
            service.measure_from_points(pixel_points, sample_calibration_model)

    def test_measure_from_points_invalid_calibration(self, real_roof_area_calculator):
        """Test that ValueError is raised for invalid calibration."""
        service = RoofMeasurementService(real_roof_area_calculator)
        pixel_points = [Point2D(0, 0), Point2D(100, 0), Point2D(100, 100)]
        invalid_calibration = CalibrationModel(
            reference_points_pixel=(Point2D(0, 0), Point2D(0, 0)),
            reference_distance_meters=1.0,
            scale_factor_pixels_per_meter=0.0, # Invalid scale factor
            unit="m"
        )
        with pytest.raises(ValueError, match="Kalibrácia má neplatný mierkový faktor"):
            service.measure_from_points(pixel_points, invalid_calibration)
