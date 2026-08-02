"""
This module defines the EstimateBuilder service for constructing comprehensive
customer estimates.
"""

from typing import List, Optional, Tuple
import uuid

from app.materials.calculation_result import MaterialCalculationResult
from app.pricing.price_model import LaborRate, PriceRule, EstimateLine, Estimate
from app.pricing.pricing_service import PricingService
from app.core.logger import setup_logging

logger = setup_logging()

class EstimateBuilder:
    """
    Builds a comprehensive Estimate object from various cost components,
    including materials, labor, and additional costs, applying pricing rules.
    """
    def __init__(self, pricing_service: PricingService):
        """
        Initializes the EstimateBuilder with a PricingService instance.

        Args:
            pricing_service (PricingService): The service to use for cost calculations
                                              and rule application.
        """
        self._pricing_service = pricing_service
        logger.info("EstimateBuilder initialized.")

    def build_estimate(
        self,
        estimate_name: str,
        material_results: List[MaterialCalculationResult],
        labor_items: Optional[List[Tuple[LaborRate, float, str, float]]] = None, # (labor_rate, quantity, description, waste_factor)
        additional_costs: Optional[List[Tuple[str, float, float, Optional[str]]]] = None, # (description, quantity, unit_price, unit)
        project_id: Optional[uuid.UUID] = None,
        customer_id: Optional[uuid.UUID] = None,
        discount_rules: Optional[List[PriceRule]] = None,
        margin_percentage: float = 0.0,
        vat_rate: float = 0.0
    ) -> Estimate:
        """
        Constructs a complete Estimate object based on provided inputs.

        Args:
            estimate_name (str): The name of the estimate.
            material_results (List[MaterialCalculationResult]): Calculated material quantities and costs.
            labor_items (Optional[List[Tuple[LaborRate, float, str, float]]]): List of labor items,
                                                                                each as (LaborRate, quantity, description, waste_factor).
            additional_costs (Optional[List[Tuple[str, float, float, Optional[str]]]]): List of additional costs,
                                                                                         each as (description, quantity, unit_price, unit).
            project_id (Optional[uuid.UUID]): Optional ID of the associated project.
            customer_id (Optional[uuid.UUID]): Optional ID of the associated customer.
            discount_rules (Optional[List[PriceRule]]): List of discount rules to apply.
            margin_percentage (float): Profit margin as a percentage (e.g., 0.15 for 15%).
            vat_rate (float): Value Added Tax rate (e.g., 0.20 for 20%).

        Returns:
            Estimate: The fully constructed Estimate object.
        """
        logger.info(f"Building estimate: {estimate_name}")

        # 1. Convert material results to EstimateLine items
        material_lines = self._pricing_service.calculate_material_cost(material_results)

        # 2. Calculate labor costs and convert to EstimateLine items
        labor_lines: List[EstimateLine] = []
        if labor_items:
            for rate, qty, desc, waste in labor_items:
                labor_lines.append(self._pricing_service.calculate_labor_cost(rate, qty, desc, waste))

        # 3. Convert additional costs to EstimateLine items
        other_lines: List[EstimateLine] = []
        if additional_costs:
            for desc, qty, unit_price, unit in additional_costs:
                other_lines.append(EstimateLine(description=desc, quantity=qty, unit_price=unit_price, unit=unit))

        # 4. Calculate total discount amount from rules
        total_subtotal_for_discount_calc = sum(line.total_price for line in material_lines + labor_lines + other_lines)
        calculated_discount_amount = 0.0
        if discount_rules:
            for rule in discount_rules:
                if rule.type == "percentage":
                    calculated_discount_amount += total_subtotal_for_discount_calc * rule.value
                elif rule.type == "fixed":
                    calculated_discount_amount += rule.value
                else:
                    logger.warning(f"Unknown discount rule type: {rule.type}. Skipping rule '{rule.name}'.")
        
        # Ensure discount is not more than subtotal
        calculated_discount_amount = min(calculated_discount_amount, total_subtotal_for_discount_calc)
        calculated_discount_amount = max(0.0, calculated_discount_amount) # Ensure non-negative

        # 5. Construct the Estimate object
        estimate = Estimate(
            name=estimate_name,
            project_id=project_id,
            customer_id=customer_id,
            material_lines=material_lines,
            labor_lines=labor_lines,
            other_lines=other_lines,
            discount=calculated_discount_amount,
            margin=margin_percentage,
            vat_rate=vat_rate
        )
        logger.info(f"Estimate '{estimate_name}' built successfully. Final Price: {estimate.final_price:.2f}")
        return estimate
