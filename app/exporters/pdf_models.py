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
    report_id: uuid.UUID = field(default_factory=uuid.uuid4)
    report_date: date = field(default_factory=date.today)

    # Customer and Project Information
    customer_name: str
    customer_address: str
    customer_phone: Optional[str] = None
    customer_email: Optional[str] = None

    project_name: str
    project_address: str
    project_description: Optional[str] = None

    # Roof Analysis Data
    roof_summary: str = "Detailed analysis of the roof structure."
    roof_image_path: Optional[Path] = None # Path to the analyzed roof image
    geometry_summary_text: Optional[str] = None # Textual description of geometry
    geometry_image_path: Optional[Path] = None # Path to an image of the generated geometry

    # Measurements
    measurements: Optional[RoofMeasurementResult] = None

    # Materials
    materials_breakdown: List[MaterialCalculationResult] = field(default_factory=list)

    # Estimate
    estimate: Optional[Estimate] = None

    # Additional sections
    notes: Optional[str] = None
    signature_area_text: str = "Approved by:"
