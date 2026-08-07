"""
This module defines data models for PDF report generation.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Dict, Optional
import uuid

from app.services.measurement_service import RoofMeasurementResult
from app.materials.calculation_result import MaterialCalculationResult
from app.pricing.price_model import Estimate

@dataclass(frozen=True)
class CompanyInfo:
    """
    Represents the company's information for branding in PDF reports.
    """
    company_name: str
    address: str
    phone: str
    email: str
    logo_path: Optional[Path] = None
    website: Optional[str] = None

@dataclass(frozen=True)
class CustomerReport:
    """
    Comprehensive data model holding all information required to generate a customer-facing PDF report.
    """
    # Non-default fields (required)
    customer_name: str
    customer_address: str
    project_name: str
    project_address: str

    # Default fields (optional)
    customer_phone: Optional[str] = None
    customer_email: Optional[str] = None
    project_description: Optional[str] = None
    roof_summary: str = "Detailed analysis of the roof structure."
    roof_image_path: Optional[Path] = None
    geometry_summary_text: Optional[str] = None
    geometry_image_path: Optional[Path] = None
    measurements: Optional[RoofMeasurementResult] = None
    materials_breakdown: List[MaterialCalculationResult] = field(default_factory=list)
    estimate: Optional[Estimate] = None
    notes: Optional[str] = None
    signature_area_text: str = "Approved by:"
    roof_material_name: Optional[str] = None
    roof_material_supplier: Optional[str] = None
    roof_material_price_per_m2: Optional[float] = None
    roof_material_total_price: Optional[float] = None
    roof_material_waste_pct: Optional[float] = None
    report_date: date = field(default_factory=date.today)
    report_id: uuid.UUID = field(default_factory=uuid.uuid4)

    # Material cost estimate
    roof_material_name: Optional[str] = None
    roof_material_supplier: Optional[str] = None
    roof_material_price_per_m2: Optional[float] = None
    roof_material_total_price: Optional[float] = None
    roof_material_waste_pct: Optional[float] = None
