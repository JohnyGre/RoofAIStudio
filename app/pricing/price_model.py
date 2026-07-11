"""
This module defines data models for pricing components like labor rates, price rules,
and estimate line items.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List
import uuid

@dataclass(frozen=True)
class LaborRate:
    """
    Represents a labor rate for a specific task or skill.
    """
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    name: str
    unit: Literal["m2", "linear_m", "hour", "km", "piece"] # Unit of measurement for labor
    price: float # Price per unit

    def __post_init__(self):
        if self.price < 0:
            raise ValueError("Labor rate price cannot be negative.")

@dataclass(frozen=True)
class PriceRule:
    """
    Represents a pricing rule that can be applied as a percentage or fixed amount.
    """
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    name: str
    type: Literal["percentage", "fixed"]
    value: float # Percentage (e.g., 0.10 for 10%) or fixed amount
    is_deduction: bool = False # True if this rule subtracts from the total (e.g., discount)

    def __post_init__(self):
        if self.type == "percentage" and not (0 <= self.value <= 1):
            raise ValueError("Percentage value must be between 0 and 1.")
        if self.type == "fixed" and self.value < 0:
            raise ValueError("Fixed amount cannot be negative.")

@dataclass(frozen=True)
class EstimateLine:
    """
    Represents a single line item in an estimate.
    """
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    description: str
    quantity: float
    unit_price: float
    total_price: float = field(init=False) # Calculated property
    unit: Optional[str] = None # Optional unit for display

    def __post_init__(self):
        object.__setattr__(self, 'total_price', round(self.quantity * self.unit_price, 2))
        if self.quantity < 0 or self.unit_price < 0:
            raise ValueError("Quantity and unit price cannot be negative.")

@dataclass(frozen=True)
class Estimate:
    """
    Represents a complete customer estimate.
    """
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    name: str
    project_id: Optional[uuid.UUID] = None
    customer_id: Optional[uuid.UUID] = None

    material_lines: List[EstimateLine] = field(default_factory=list)
    labor_lines: List[EstimateLine] = field(default_factory=list)
    other_lines: List[EstimateLine] = field(default_factory=list) # For transport, additional costs etc.

    subtotal: float = field(init=False)
    discount: float = 0.0
    margin: float = 0.0
    vat_rate: float = 0.0 # e.g., 0.20 for 20% VAT
    final_price: float = field(init=False)

    def __post_init__(self):
        # Calculate subtotal
        total_materials = sum(line.total_price for line in self.material_lines)
        total_labor = sum(line.total_price for line in self.labor_lines)
        total_other = sum(line.total_price for line in self.other_lines)
        object.__setattr__(self, 'subtotal', round(total_materials + total_labor + total_other, 2))

        # Apply discount
        price_after_discount = self.subtotal - self.discount
        if price_after_discount < 0:
            price_after_discount = 0 # Ensure price doesn't go negative

        # Apply margin
        price_with_margin = price_after_discount * (1 + self.margin)

        # Apply VAT
        final_price_calc = price_with_margin * (1 + self.vat_rate)
        object.__setattr__(self, 'final_price', round(final_price_calc, 2))

        if self.discount < 0 or self.margin < 0 or self.vat_rate < 0:
            raise ValueError("Discount, margin, and VAT rate cannot be negative.")
