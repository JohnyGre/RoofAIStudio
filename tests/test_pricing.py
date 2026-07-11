"""
Tests for the app.pricing module.
"""

import pytest
import uuid
from math import isclose

from app.pricing.price_model import LaborRate, PriceRule, EstimateLine, Estimate
from app.pricing.pricing_service import PricingService
from app.pricing.estimate_builder import EstimateBuilder
from app.materials.calculation_result import MaterialCalculationResult
from app.database.enums import MaterialUnit

class TestPriceModels:
    def test_labor_rate_creation(self):
        rate = LaborRate(id=uuid.uuid4(), name="Installation", unit="m2", price=15.0)
        assert rate.name == "Installation"
        assert rate.unit == "m2"
        assert isclose(rate.price, 15.0)

    def test_labor_rate_invalid_price(self):
        with pytest.raises(ValueError, match="price cannot be negative"):
            LaborRate(id=uuid.uuid4(), name="Invalid", unit="hour", price=-5.0)

    def test_price_rule_creation(self):
        rule_percent = PriceRule(id=uuid.uuid4(), name="Discount 10%", type="percentage", value=0.10, is_deduction=True)
        assert rule_percent.type == "percentage"
        assert isclose(rule_percent.value, 0.10)
        assert rule_percent.is_deduction is True

        rule_fixed = PriceRule(id=uuid.uuid4(), name="Fixed Fee", type="fixed", value=50.0)
        assert rule_fixed.type == "fixed"
        assert isclose(rule_fixed.value, 50.0)
        assert rule_fixed.is_deduction is False

    def test_price_rule_invalid_percentage(self):
        with pytest.raises(ValueError, match="Percentage value must be between 0 and 1"):
            PriceRule(id=uuid.uuid4(), name="Invalid", type="percentage", value=1.5)

    def test_estimate_line_creation(self):
        line = EstimateLine(description="Material A", quantity=10.0, unit_price=5.0, unit="pcs")
        assert line.description == "Material A"
        assert isclose(line.quantity, 10.0)
        assert isclose(line.unit_price, 5.0)
        assert isclose(line.total_price, 50.0)

    def test_estimate_creation(self):
        mat_line = EstimateLine(description="Shingles", quantity=100.0, unit_price=1.0, unit="sqft")
        lab_line = EstimateLine(description="Install", quantity=10.0, unit_price=20.0, unit="hr")
        other_line = EstimateLine(description="Transport", quantity=1.0, unit_price=50.0)

        estimate = Estimate(
            name="Test Estimate",
            material_lines=[mat_line],
            labor_lines=[lab_line],
            other_lines=[other_line],
            discount=10.0,
            margin=0.10,
            vat_rate=0.20
        )
        # Subtotal = 100*1 + 10*20 + 50 = 100 + 200 + 50 = 350
        assert isclose(estimate.subtotal, 350.0)
        # Price after discount = 350 - 10 = 340
        # Price with margin = 340 * (1 + 0.10) = 340 * 1.1 = 374
        # Final price = 374 * (1 + 0.20) = 374 * 1.2 = 448.8
        assert isclose(estimate.final_price, 448.8)

class TestPricingService:
    @pytest.fixture
    def pricing_service(self):
        return PricingService()

    def test_calculate_material_cost(self, pricing_service: PricingService):
        material_results = [
            MaterialCalculationResult(uuid.uuid4(), "Shingle A", 100.0, MaterialUnit.SQUARE_FOOT, 10.0, 110.0, 110.0),
            MaterialCalculationResult(uuid.uuid4(), "Nails", 5.0, MaterialUnit.EACH, 0.5, 5.5, 55.0)
        ]
        estimate_lines = pricing_service.calculate_material_cost(material_results)
        assert len(estimate_lines) == 2
        assert estimate_lines[0].description.startswith("Shingle A")
        assert isclose(estimate_lines[0].total_price, 110.0)

    def test_calculate_labor_cost(self, pricing_service: PricingService):
        labor_rate = LaborRate(id=uuid.uuid4(), name="Installation", unit="m2", price=15.0)
        labor_line = pricing_service.calculate_labor_cost(labor_rate, 10.0, "Roof Installation", waste_factor=0.1)
        # Quantity = 10 * (1 + 0.1) = 11
        # Total price = 11 * 15 = 165
        assert labor_line.description == "Roof Installation (Installation)"
        assert isclose(labor_line.quantity, 11.0)
        assert isclose(labor_line.total_price, 165.0)

    def test_apply_price_rules(self, pricing_service: PricingService):
        base_amount = 100.0
        rules = [
            PriceRule(id=uuid.uuid4(), name="Discount 10%", type="percentage", value=0.10, is_deduction=True),
            PriceRule(id=uuid.uuid4(), name="Fixed Surcharge", type="fixed", value=20.0, is_deduction=False)
        ]
        # 100 - (100 * 0.10) + 20 = 100 - 10 + 20 = 110
        adjusted_amount = pricing_service.apply_price_rules(base_amount, rules)
        assert isclose(adjusted_amount, 110.0)

    def test_calculate_total(self, pricing_service: PricingService):
        material_cost = 1000.0
        labor_cost = 500.0
        other_costs = 100.0
        discount_rules = [PriceRule(id=uuid.uuid4(), name="Promo", type="percentage", value=0.05, is_deduction=True)]
        margin_percentage = 0.10
        vat_rate = 0.20

        total_summary = pricing_service.calculate_total(
            material_cost, labor_cost, other_costs,
            discount_rules=discount_rules,
            margin_percentage=margin_percentage,
            vat_rate=vat_rate
        )
        # Subtotal = 1000 + 500 + 100 = 1600
        assert isclose(total_summary["subtotal"], 1600.0)
        # Discount = 1600 * 0.05 = 80
        assert isclose(total_summary["discount_amount"], 80.0)
        # Price after discount = 1600 - 80 = 1520
        assert isclose(total_summary["price_after_discount"], 1520.0)
        # Price with margin = 1520 * 1.10 = 1672
        assert isclose(total_summary["price_with_margin"], 1672.0)
        # VAT = 1672 * 0.20 = 334.4
        assert isclose(total_summary["vat_amount"], 334.4)
        # Final price = 1672 + 334.4 = 2006.4
        assert isclose(total_summary["final_price"], 2006.4)

class TestEstimateBuilder:
    @pytest.fixture
    def pricing_service_for_builder(self):
        return PricingService()

    def test_build_estimate(self, pricing_service_for_builder: PricingService):
        builder = EstimateBuilder(pricing_service_for_builder)

        material_results = [
            MaterialCalculationResult(uuid.uuid4(), "Shingle A", 100.0, MaterialUnit.SQUARE_FOOT, 10.0, 110.0, 110.0),
            MaterialCalculationResult(uuid.uuid4(), "Nails", 5.0, MaterialUnit.EACH, 0.5, 5.5, 55.0)
        ]
        labor_rate_install = LaborRate(id=uuid.uuid4(), name="Installation", unit="m2", price=15.0)
        labor_items = [(labor_rate_install, 10.0, "Roof Installation", 0.1)] # 10 m2, 10% waste

        additional_costs = [("Transport Fee", 1.0, 50.0, "trip")]

        discount_rules = [PriceRule(id=uuid.uuid4(), name="Early Bird", type="percentage", value=0.05, is_deduction=True)]

        estimate = builder.build_estimate(
            estimate_name="Comprehensive Roof Job",
            material_results=material_results,
            labor_items=labor_items,
            additional_costs=additional_costs,
            discount_rules=discount_rules,
            margin_percentage=0.15,
            vat_rate=0.20
        )

        # Material subtotal = 110 + 55 = 165
        # Labor subtotal = 10 * (1+0.1) * 15 = 11 * 15 = 165
        # Other subtotal = 50
        # Total subtotal = 165 + 165 + 50 = 380
        assert isclose(estimate.subtotal, 380.0)
        
        # Discount = 380 * 0.05 = 19
        assert isclose(estimate.discount, 19.0)

        # Price after discount = 380 - 19 = 361
        # Price with margin = 361 * (1 + 0.15) = 361 * 1.15 = 415.15
        # Final price = 415.15 * (1 + 0.20) = 415.15 * 1.2 = 498.18
        assert isclose(estimate.final_price, 498.18)
        assert len(estimate.material_lines) == 2
        assert len(estimate.labor_lines) == 1
        assert len(estimate.other_lines) == 1
