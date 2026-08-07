"""Clean Architecture pricing service for roofing materials."""
import uuid
from typing import List, Optional
from dataclasses import dataclass
from sqlalchemy.orm import Session
from app.database.models.material import Material as ORMMaterial, MaterialCategory
from app.database.models.supplier import Supplier
from app.database.models.price_list import PriceList, PriceItem


@dataclass
class MaterialPriceResult:
    material_id: uuid.UUID
    material_name: str
    category_name: str
    supplier_name: str
    price_per_m2: float
    waste_factor: float
    roof_area_m2: float
    waste_area_m2: float
    total_area_m2: float
    total_price_eur: float
    currency: str = "EUR"


class RoofPricingService:
    def __init__(self, session: Session):
        self.session = session

    def get_materials_by_slope(self, slope_deg: float = 25.0):
        return (self.session.query(ORMMaterial)
            .where(ORMMaterial.is_active == True)
            .where((ORMMaterial.min_slope_deg == None) |
                   (ORMMaterial.min_slope_deg <= slope_deg))
            .order_by(ORMMaterial.name).all())

    def get_suppliers_for_material(self, material_id: uuid.UUID):
        items = (self.session.query(PriceItem, PriceList, Supplier)
            .join(PriceList, PriceItem.price_list_id == PriceList.id)
            .join(Supplier, PriceList.supplier_id == Supplier.id)
            .where(PriceItem.material_id == material_id)
            .order_by(PriceItem.unit_price).all())
        return [{"supplier_id": s.id, "supplier_name": s.name,
                 "price_per_unit": pi.unit_price,
                 "price_list_name": pl.name, "currency": pl.currency}
                for pi, pl, s in items]

    def get_all_suppliers(self):
        return self.session.query(Supplier).order_by(Supplier.name).all()

    def calculate_price(self, material_id: uuid.UUID, supplier_id, roof_area_m2: float):
        material = self.session.get(ORMMaterial, material_id)
        if not material:
            return None
        unit_price = material.unit_cost
        supplier_name = "Neuvedeny"
        if supplier_id:
            item = (self.session.query(PriceItem).join(PriceList)
                    .where(PriceItem.material_id == material_id,
                           PriceList.supplier_id == supplier_id).first())
            if item:
                unit_price = item.unit_price
                supplier_name = self.session.get(Supplier, supplier_id).name
        waste_area = roof_area_m2 * material.waste_factor
        total_area = roof_area_m2 + waste_area
        total_price = unit_price * total_area
        return MaterialPriceResult(
            material_id=material.id, material_name=material.name,
            category_name=material.category.name if material.category else "",
            supplier_name=supplier_name, price_per_m2=unit_price,
            waste_factor=material.waste_factor,
            roof_area_m2=round(roof_area_m2, 1),
            waste_area_m2=round(waste_area, 1),
            total_area_m2=round(total_area, 1),
            total_price_eur=round(total_price, 2),
        )
