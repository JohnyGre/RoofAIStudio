"""
CSV importer for roof accessories — reads cennik_doplnky.csv into DB.
"""
import os, csv, re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import List, Dict, Optional

from app.database.models.accessory import RoofAccessory


def _parse_decimal(val: str) -> Optional[float]:
    """Parse Slovak decimal format (comma) to float. Returns None for empty."""
    if not val or not str(val).strip():
        return None
    try:
        return float(Decimal(str(val).strip().replace(",", ".")))
    except (InvalidOperation, ValueError):
        return None


def _parse_date(val: str) -> Optional[date]:
    """Parse YYYY-MM-DD date. Returns None for empty/invalid."""
    if not val or not str(val).strip():
        return None
    try:
        return datetime.strptime(str(val).strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def import_csv(session, csv_path: str, clear_first: bool = True) -> dict:
    """
    Import cennik_doplnky.csv into the roof_accessories table.
    
    CSV format (; delimiter, , decimal, UTF-8):
    kategoria;vyrobca;vyrobca_web;nazov_materialu;jednotka;sklon_min_stupne;
    sirka_m;dlzka_m;hmotnost_kg_m2;odpad_percent;dodavatel;dodavatel_web;
    dodavatel_region;cena_bez_dph;mena;platnost_od;poznamka
    
    Returns dict with stats.
    """
    if clear_first:
        session.query(RoofAccessory).delete()
        session.commit()
    
    if not os.path.exists(csv_path):
        return {"error": f"File not found: {csv_path}", "imported": 0, "skipped": 0}
    
    stats = {"imported": 0, "skipped": 0, "errors": [], "categories": set()}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        # Skip comment lines
        reader = csv.reader(
            (line for line in f if not line.startswith("#")),
            delimiter=";"
        )
        
        for row in reader:
            if len(row) < 17:
                stats["skipped"] += 1
                continue
            
            try:
                (
                    kategoria, vyrobca, vyrobca_web, nazov, jednotka,
                    sklon_str, sirka_str, dlzka_str, hmotnost_str, odpad_str,
                    dodavatel, dodavatel_web, dodavatel_region,
                    cena_str, mena, platnost_str, poznamka
                ) = row[:17]
                
                # Validate required fields
                if not kategoria.strip() or not nazov.strip() or not cena_str.strip():
                    stats["skipped"] += 1
                    continue
                
                price = _parse_decimal(cena_str)
                if price is None:
                    stats["skipped"] += 1
                    continue
                
                accessory = RoofAccessory(
                    category=kategoria.strip(),
                    manufacturer=vyrobca.strip(),
                    manufacturer_web=vyrobca_web.strip() or None,
                    name=nazov.strip(),
                    unit=jednotka.strip() or "ks",
                    min_slope_deg=_parse_decimal(sklon_str),
                    width_m=_parse_decimal(sirka_str),
                    length_m=_parse_decimal(dlzka_str),
                    weight_kg_per_m2=_parse_decimal(hmotnost_str),
                    waste_percent=_parse_decimal(odpad_str) or 5.0,
                    supplier=dodavatel.strip() or None,
                    supplier_web=dodavatel_web.strip() or None,
                    supplier_region=dodavatel_region.strip() or None,
                    price=price,
                    currency=mena.strip().upper() or "EUR",
                    valid_from=_parse_date(platnost_str),
                    notes=poznamka.strip() or None,
                    is_active=True
                )
                session.add(accessory)
                stats["imported"] += 1
                stats["categories"].add(kategoria.strip())
            except Exception as e:
                stats["errors"].append(f"Row error: {e}")
                stats["skipped"] += 1
    
    session.commit()
    stats["categories"] = sorted(stats["categories"])
    return stats


CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "cennik_doplnky.csv"
)
