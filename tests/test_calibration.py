"""
Tests for the app.geometry.calibration module.
"""

import pytest
from math import isclose

from app.geometry.point import Point2D
from app.geometry.calibration import CalibrationModel, CalibrationService

class TestCalibrationService:
    def test_calibrate_from_distance_success(self):
        p1_pixel = Point2D(0, 0)
        p2_pixel = Point2D(100, 0)
        real_world_distance_meters = 1.0
        
        calibration = CalibrationService.calibrate_from_distance(
            p1_pixel, p2_pixel, real_world_distance_meters
        )
        
        assert isinstance(calibration, CalibrationModel)
        assert calibration.reference_points_pixel == (p1_pixel, p2_pixel)
        assert isclose(calibration.reference_distance_meters, 1.0)
        assert isclose(calibration.scale_factor_pixels_per_meter, 100.0)
        assert calibration.unit == "m"

    def test_calibrate_from_distance_zero_real_world_distance(self):
        p1_pixel = Point2D(0, 0)
        p2_pixel = Point2D(100, 0)
        with pytest.raises(ValueError, match="Real-world distance must be positive"):
            CalibrationService.calibrate_from_distance(p1_pixel, p2_pixel, 0.0)

    def test_calibrate_from_distance_identical_pixel_points(self):
        p1_pixel = Point2D(0, 0)
        p2_pixel = Point2D(0, 0)
        with pytest.raises(ValueError, match="Reference pixel points are identical"):
            CalibrationService.calibrate_from_distance(p1_pixel, p2_pixel, 1.0)

    def test_pixel_to_meter(self, sample_calibration_model: CalibrationModel):
        pixels = 200.0
        meters = CalibrationService.pixel_to_meter(pixels, sample_calibration_model)
        assert isclose(meters, 2.0) # 200 pixels / 100 px/m = 2 meters

    def test_meter_to_pixel(self, sample_calibration_model: CalibrationModel):
        meters = 3.0
        pixels = CalibrationService.meter_to_pixel(meters, sample_calibration_model)
        assert isclose(pixels, 300.0) # 3 meters * 100 px/m = 300 pixels

    def test_convert_unit_m_to_cm(self):
        value_m = 1.5
        value_cm = CalibrationService.convert_unit(value_m, "m", "cm")
        assert isclose(value_cm, 150.0)

    def test_convert_unit_cm_to_mm(self):
        value_cm = 10.0
        value_mm = CalibrationService.convert_unit(value_cm, "cm", "mm")
        assert isclose(value_mm, 100.0)

    def test_convert_unit_mm_to_m(self):
        value_mm = 500.0
        value_m = CalibrationService.convert_unit(value_mm, "mm", "m")
        assert isclose(value_m, 0.5)

    def test_convert_unit_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported unit conversion"):
            CalibrationService.convert_unit(1.0, "m", "inch")
