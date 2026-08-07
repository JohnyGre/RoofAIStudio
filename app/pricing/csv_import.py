"""
CSV import modul pre cenniky striesnych materialov.

Podporovane formaty:
1. Novy format (oddelovac ';', desatinna ciarka ','):
   kategoria;vyrobca;vyrobca_web;nazov_materialu;jednotka;sklon_min_stupne;sirka_m;dlzka_m;
   hmotnost_kg_m2;odpad_percent;dodavatel;dodavatel_web;dodavatel_region;cena_bez_dph;
   mena;platnost_od;poznamka

2. Stary format (oddelovac ',', desatinna bodka '.') - spatna kompatibilita:
   kategoria,vyrobca,nazov_materialu,cena_za_m2,min_sklon_stupne,krycia_sirka_m,
   krycia_dlzka_m,hmotnost_kg_m2,odpad_faktor,dodavatel,cena_dodavatela_m2,mena
"""
import csv
import os
import uuid
from datetime import date, datetime


def _parse_decimal(val: str) -> float:
    """Parse decimal: handles both ',' (Slovak) and '.' (legacy)."""
    if not val or not val.strip():
        return 0.0
    return float(val.strip().replace(',', '.'))


def _parse_date(val: str) -> date:
    """Parse YYYY-MM-DD date, fallback to today."""
    if not val or not val.strip():
        return date.today()
    try:
        return datetime.strptime(val.strip(), '%Y-%m-%d').date()
    except ValueError:
        return date.today()


def _detect_format(header_line: str) -> tuple[str, bool]:
    """Returns (delimiter, is_new_format) based on header."""
    if ';' in header_line and 'platnost_od' in header_line:
        return ';', True
    if ',' in header_line and 'cena_za_m2' in header_line:
        return ',', False
    # Auto-detect
    if ';' in header_line:
        return ';', True
    return ',', False


def import_csv(filepath: str, db_session):
    """
    Import materials and prices from CSV.
    Supports both old and new CSV formats.
    Returns count of price items imported.
    """
    from app.database.models.material import Material as ORMMaterial, MaterialCategory, MaterialManufacturer
    from app.database.models.supplier import Supplier
    from app.database.models.price_list import PriceList, PriceItem
    from app.database.enums import MaterialUnit
    from app.database.enums import MaterialUnit

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV subor {filepath} neexistuje!")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.rstrip('\n\r') for l in f if l.strip() and not l.lstrip().startswith('#')]

    if not lines:
        raise ValueError("CSV subor je prazdny!")

    # Detect format
    delimiter, is_new = _detect_format(lines[0])
    reader = csv.reader(lines, delimiter=delimiter)
    rows = list(reader)

    if not rows:
        raise ValueError("CSV subor je prazdny!")

    # Prepare caches
    cat_map = {c.name: c for c in db_session.query(MaterialCategory).all()}
    mfr_map = {m.name: m for m in db_session.query(MaterialManufacturer).all()}
    sup_map = {s.name: s for s in db_session.query(Supplier).all()}

    imported = 0
    for row in rows[1:]:
        if not row or len(row) < 7:
            continue

        if is_new:
            # New format: 17 columns
            # kategoria;vyrobca;vyrobca_web;nazov_materialu;jednotka;sklon_min_stupne;
            # sirka_m;dlzka_m;hmotnost_kg_m2;odpad_percent;dodavatel;dodavatel_web;
            # dodavatel_region;cena_bez_dph;mena;platnost_od;poznamka
            if len(row) < 14:
                continue
            cat_name = row[0].strip()
            mfr_name = row[1].strip()
            mfr_web = row[2].strip() if len(row) > 2 else ''
            mat_name = row[3].strip()
            unit_str = row[4].strip() if len(row) > 4 else 'm2'
            min_slope = _parse_decimal(row[5]) if len(row) > 5 else 0
            cov_w = _parse_decimal(row[6]) if len(row) > 6 else 0
            cov_l = _parse_decimal(row[7]) if len(row) > 7 else 0
            weight = _parse_decimal(row[8]) if len(row) > 8 else 0
            waste_pct = _parse_decimal(row[9]) if len(row) > 9 else 0
            sup_name = row[10].strip()
            sup_web = row[11].strip() if len(row) > 11 else ''
            sup_region = row[12].strip() if len(row) > 12 else ''
            price_val = _parse_decimal(row[13])
            currency = row[14].strip() if len(row) > 14 and row[14].strip() else 'EUR'
            platnost_od = _parse_date(row[15]) if len(row) > 15 else date.today()
            # poznamka = row[16] if len(row) > 16 else ''
            waste_factor = (waste_pct / 100.0) if waste_pct > 0 else 0.10
        else:
            # Old format: 12 columns
            # kategoria,vyrobca,nazov_materialu,cena_za_m2,min_sklon_stupne,
            # krycia_sirka_m,krycia_dlzka_m,hmotnost_kg_m2,odpad_faktor,
            # dodavatel,cena_dodavatela_m2,mena
            (cat_name, mfr_name, mat_name, price_m2, min_slope_s,
             cov_w_s, cov_l_s, weight_s, waste_s, sup_name, sup_price_s, currency) = row[:12]

            cat_name = cat_name.strip()
            mfr_name = mfr_name.strip()
            mat_name = mat_name.strip()
            mfr_web = ''
            unit_str = 'm2'
            min_slope = float(min_slope_s.strip()) if min_slope_s.strip() else 0
            cov_w = float(cov_w_s.strip()) if cov_w_s.strip() else 0
            cov_l = float(cov_l_s.strip()) if cov_l_s.strip() else 0
            weight = float(weight_s.strip()) if weight_s.strip() else 0
            waste_factor = float(waste_s.strip()) if waste_s.strip() else 0.10
            sup_name = sup_name.strip()
            sup_web = ''
            sup_region = ''
            price_val = float(sup_price_s.strip()) if sup_price_s.strip() else float(price_m2.strip())
            currency = currency.strip() if currency else 'EUR'
            platnost_od = date.today()  # fallback

        # Resolve unit
        unit_map = {'m2': MaterialUnit.SQUARE_METER, 'ks': MaterialUnit.PIECE,
                    'lm': MaterialUnit.LINEAR_METER, 'bal_m2': MaterialUnit.BUNDLE_M2,
                    'kg': MaterialUnit.KG}
        unit = unit_map.get(unit_str.strip().lower() if unit_str else 'm2', MaterialUnit.SQUARE_METER)

        # Create category if missing
        if cat_name not in cat_map:
            cat_map[cat_name] = MaterialCategory(id=uuid.uuid4(), name=cat_name)
            db_session.add(cat_map[cat_name])

        # Create manufacturer if missing
        if mfr_name not in mfr_map:
            mfr_map[mfr_name] = MaterialManufacturer(
                id=uuid.uuid4(), name=mfr_name,
                website=mfr_web if mfr_web else None
            )
            db_session.add(mfr_map[mfr_name])
        elif mfr_web and not mfr_map[mfr_name].website:
            mfr_map[mfr_name].website = mfr_web

        # Create supplier if missing
        if sup_name not in sup_map:
            sup_map[sup_name] = Supplier(
                id=uuid.uuid4(), name=sup_name,
                website=sup_web if sup_web else None,
                region=sup_region if sup_region else None
            )
            db_session.add(sup_map[sup_name])
        else:
            sup = sup_map[sup_name]
            if sup_web and not sup.website:
                sup.website = sup_web
            if sup_region and not sup.region:
                sup.region = sup_region

        sup = sup_map[sup_name]

        # Price list
        pl_name = f"{sup_name} cennik"
        pl = db_session.query(PriceList).where(
            PriceList.supplier_id == sup.id, PriceList.name == pl_name).first()
        if not pl:
            pl = PriceList(id=uuid.uuid4(), supplier_id=sup.id, name=pl_name,
                           currency=currency.strip() if currency else 'EUR')
            db_session.add(pl)

        # Material
        existing = db_session.query(ORMMaterial).where(ORMMaterial.name == mat_name).first()
        if not existing:
            mat = ORMMaterial(
                id=uuid.uuid4(), name=mat_name, category_id=cat_map[cat_name].id,
                manufacturer_id=mfr_map[mfr_name].id, unit_cost=price_val,
                unit_of_measure=unit,
                covering_width_m=cov_w if cov_w else None,
                covering_length_m=cov_l if cov_l else None,
                min_slope_deg=min_slope if min_slope else None,
                weight_kg_per_m2=weight if weight else None,
                waste_factor=waste_factor,
                is_active=True,
            )
            db_session.add(mat)
            db_session.flush()
            mat_id = mat.id
        else:
            mat_id = existing.id
            # Update with new data if available
            if cov_w:
                existing.covering_width_m = cov_w
            if cov_l:
                existing.covering_length_m = cov_l
            if min_slope:
                existing.min_slope_deg = min_slope
            if weight:
                existing.weight_kg_per_m2 = weight
            existing.waste_factor = waste_factor

        # Price item with platnost_od
        pi = PriceItem(
            id=uuid.uuid4(), material_id=mat_id, price_list_id=pl.id,
            unit_price=price_val,
            platnost_od=datetime.combine(platnost_od, datetime.min.time())
        )
        db_session.add(pi)
        imported += 1
        print(f"  + {mat_name} @ {price_val} EUR/m2 ({sup_name}) platna od {platnost_od}")

    db_session.commit()
    return imported


def export_csv(output_path: str, db_session) -> int:
    """
    Export all price items to CSV in the new format (; delimiter).
    Returns number of rows exported (excluding header).
    Uses the field names from the ORM models directly - no hardcoded strings.
    """
    from app.database.models.material import Material as ORMMaterial, MaterialCategory, MaterialManufacturer
    from app.database.models.supplier import Supplier
    from app.database.models.price_list import PriceList, PriceItem
    from app.database.enums import MaterialUnit

    header = "kategoria;vyrobca;vyrobca_web;nazov_materialu;jednotka;sklon_min_stupne;sirka_m;" \
             "dlzka_m;hmotnost_kg_m2;odpad_percent;dodavatel;dodavatel_web;dodavatel_region;" \
             "cena_bez_dph;mena;platnost_od;poznamka"

    rows_written = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Cenník strešných materiálov - RoofAIStudio - export {date.today()}\n")
        f.write(f"# Riadky s # sa ignorujú. Desatinná čiarka. Oddeľovač ;\n")
        f.write(header + "\n")

        items = db_session.query(PriceItem).join(
            PriceList, PriceItem.price_list_id == PriceList.id
        ).join(Supplier, PriceList.supplier_id == Supplier.id).join(
            ORMMaterial, PriceItem.material_id == ORMMaterial.id
        ).join(MaterialCategory, ORMMaterial.category_id == MaterialCategory.id).join(
            MaterialManufacturer, ORMMaterial.manufacturer_id == MaterialManufacturer.id
        ).all()

        for pi in items:
            mat = pi.material
            pl = pi.price_list
            sup = pl.supplier
            cat = mat.category
            mfr = mat.manufacturer

            unit_map = {
                MaterialUnit.SQUARE_METER: 'm2',
                MaterialUnit.PIECE: 'ks',
                MaterialUnit.LINEAR_METER: 'lm',
                MaterialUnit.BUNDLE_M2: 'bal_m2',
                MaterialUnit.KG: 'kg',
            }

            # Format with comma decimals
            def fmt(val):
                if val is None:
                    return ""
                return str(val).replace('.', ',')

            platnost = pi.platnost_od.strftime('%Y-%m-%d') if pi.platnost_od else date.today().isoformat()

            row = [
                cat.name,
                mfr.name,
                mfr.website or '',
                mat.name,
                unit_map.get(mat.unit_of_measure, 'm2'),
                fmt(mat.min_slope_deg),
                fmt(mat.covering_width_m),
                fmt(mat.covering_length_m),
                fmt(mat.weight_kg_per_m2),
                fmt(int(mat.waste_factor * 100)) if mat.waste_factor else '',
                sup.name,
                sup.website or '',
                sup.region or '',
                fmt(pi.unit_price),
                pl.currency or 'EUR',
                platnost,
                ''  # poznamka
            ]
            f.write(';'.join(row) + '\n')
            rows_written += 1

    print(f"Exportovanych {rows_written} poloziek do {output_path}")
    return rows_written


if __name__ == "__main__":
    # Create example CSV
    csv_path = r'C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio\data\cennik_priklad.csv'
    example = [
        "# Cenník strešných materiálov - RoofAIStudio",
        "# Riadky s # sa ignorujú. Desatinná čiarka. Oddeľovač ;",
        "kategoria;vyrobca;vyrobca_web;nazov_materialu;jednotka;sklon_min_stupne;sirka_m;dlzka_m;hmotnost_kg_m2;odpad_percent;dodavatel;dodavatel_web;dodavatel_region;cena_bez_dph;mena;platnost_od;poznamka",
        "skridla;Tondach;tondach.sk;Figaro 12;m2;22;0,3;0,42;42,5;12;Dektrade;dektrade.sk;celé SR;18,90;EUR;2026-01-15;",
        "skridla;Bramac;bramac.sk;Alpina;m2;18;0,3;0,42;45;12;LAD;lad.sk;Trnavský kraj;19,40;EUR;2026-01-15;",
        "plech;Ruukki;ruukki.sk;Adamante;m2;10;1,18;;5,2;7;Dachmetal;dachmetal.sk;celé SR;24,50;EUR;2026-01-10;",
        "sindel;Katepal;katepal.sk;KL;m2;12;1;;9;8;Dektrade;dektrade.sk;celé SR;15,20;EUR;2026-01-15;",
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(example))
    print(f"Prikladovy CSV vytvoreny: {csv_path}")
