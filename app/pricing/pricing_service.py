"""
This module defines the PricingService for calculating costs and applying pricing rules.
"""

from typing import List, Optional, Dict
import uuid

from app.materials.calculation_result import MaterialCalculationResult
from app.pricing.price_model import LaborRate, PriceRule, EstimateLine
from app.database.enums import MaterialUnit
from app.core.logger import setup_logging

logger = setup_logging()

class PricingService:
    """
    Service for calculating various costs (materials, labor) and applying pricing rules
    to generate components of an estimate.
    """

    def __init__(self):
        pass

    def calculate_material_cost(self, material_results: List[MaterialCalculationResult]) -> List[EstimateLine]:
        """
        Converts material calculation results into estimate line items.

        Args:
            material_results (List[MaterialCalculationResult]): List of calculated material quantities and costs.

        Returns:
            List[EstimateLine]: A list of estimate line items for materials.
        """
        material_lines: List[EstimateLine] = []
        for res in material_results:
            material_lines.append(EstimateLine(
                description=f"{res.material_name} (incl. {res.waste_quantity:.2f} waste)",
                quantity=res.total_quantity,
                unit_price=res.estimated_cost / res.total_quantity if res.total_quantity > 0 else 0.0,
                unit=res.unit.value # Use the string value of the enum
            ))
        return material_lines

    def calculate_labor_cost(
        self,
        labor_rate: LaborRate,
        quantity: float,
        description: str,
        waste_factor: float = 0.0
    ) -> EstimateLine:
        """
        Calculates the cost for a specific labor item.

        Args:
            labor_rate (LaborRate): The labor rate to apply.
            quantity (float): The base quantity of labor (e.g., m2, hours).
            description (str): A description for the labor line item.
            waste_factor (float): Additional factor for labor (e.g., for complex tasks).

        Returns:
            EstimateLine: An estimate line item for the labor.
        """
        if quantity < 0:
            raise ValueError("Labor quantity cannot be negative.")
        if not (0 <= waste_factor < 1):
            raise ValueError("Waste factor must be between 0 and 1 (exclusive of 1).")

        total_quantity = quantity * (1 + waste_factor)
        total_price = total_quantity * labor_rate.price

        return EstimateLine(
            description=f"{description} ({labor_rate.name})",
            quantity=total_quantity,
            unit_price=labor_rate.price,
            unit=labor_rate.unit,
            total_price=total_price
        )

    def apply_price_rules(self, base_amount: float, rules: List[PriceRule]) -> float:
        """
        Applies a list of pricing rules (discounts, surcharges) to a base amount.

        Args:
            base_amount (float): The initial amount to apply rules to.
            rules (List[PriceRule]): A list of PriceRule objects.

        Returns:
            float: The adjusted amount after applying all rules.
        """
        current_amount = base_amount
        for rule in rules:
            if rule.type == "percentage":
                change = current_amount * rule.value
            elif rule.type == "fixed":
                change = rule.value
            else:
                logger.warning(f"Unknown price rule type: {rule.type}. Skipping rule '{rule.name}'.")
                continue

            if rule.is_deduction:
                current_amount -= change
            else:
                current_amount += change
            
            current_amount = max(0.0, current_amount) # Ensure amount doesn't go negative

        return round(current_amount, 2)

    def calculate_total(
        self,
        material_cost: float,
        labor_cost: float,
        other_costs: float = 0.0,
        discount_rules: Optional[List[PriceRule]] = None,
        margin_percentage: float = 0.0,
        vat_rate: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculates the final total price of an estimate, applying discounts, margin, and VAT.

        Args:
            material_cost (float): Total cost of materials.
            labor_cost (float): Total cost of labor.
            other_costs (float): Any additional costs (e.g., transport).
            discount_rules (Optional[List[PriceRule]]): List of discount rules to apply.
            margin_percentage (float): Profit margin as a percentage (e.g., 0.15 for 15%).
            vat_rate (float): Value Added Tax rate (e.g., 0.20 for 20%).

        Returns:
            Dict[str, float]: A dictionary containing subtotal, discount_amount,
                              price_after_discount, price_with_margin, vat_amount, final_price.
        """
        subtotal = material_cost + labor_cost + other_costs
        
        discount_amount = 0.0
        if discount_rules:
            # Calculate total discount amount
            for rule in discount_rules:
                if rule.type == "percentage":
                    discount_amount += subtotal * rule.value
                elif rule.type == "fixed":
                    discount_amount += rule.value
            
            price_after_discount = max(0.0, subtotal - discount_amount)
        else:
            price_after_discount = subtotal

        price_with_margin = price_after_discount * (1 + margin_percentage)
        vat_amount = price_with_margin * vat_rate
        final_price = price_with_margin + vat_amount

        return {
            "subtotal": round(subtotal, 2),
            "discount_amount": round(discount_amount, 2),
            "price_after_discount": round(price_after_discount, 2),
            "price_with_margin": round(price_with_margin, 2),
            "vat_amount": round(vat_amount, 2),
            "final_price": round(final_price, 2)
        }
