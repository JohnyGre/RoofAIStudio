"""
CSV import modul pre cenniky striesnych materialov.

Format CSV suboru (kodovanie UTF-8):
kategoria,vyrobca,nazov_materialu,cena_za_m2,min_sklon_stupne,krycia_sirka_m,krycia_dlzka_m,hmotnost_kg_m2,odpad_faktor,dodavatel,cena_dodavatela_m2,mena

Priklad: Skridla,Tondach,Tondach Figaro 11,14.50,22,0.300,0.340,45,0.12,LAD,14.50,EUR
"""
import csv
import os
import uuid

def import_csv(filepath: str, db_session):
    """Import materials and prices from CSV. Returns count of materials imported."""
    from app.database.models.material import Material as ORMMaterial, MaterialCategory, MaterialManufacturer
    from app.database.models.supplier import Supplier
    from app.database.models.price_list import PriceList, PriceItem
    from app.database.enums import MaterialUnit

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV subor {filepath} neexistuje!")

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows or not rows[0][0].startswith('kategoria'):
        raise ValueError("CSV musi mat hlavicku: kategoria,vyrobca,nazov_materialu,...")

    cat_map = {c.name: c for c in db_session.query(MaterialCategory).all()}
    mfr_map = {m.name: m for m in db_session.query(MaterialManufacturer).all()}
    sup_map = {s.name: s for s in db_session.query(Supplier).all()}

    imported = 0
    for row in rows[1:]:
        if not row or len(row) < 7 or row[0].startswith('#'):
            continue
        (cat_name, mfr_name, mat_name, price_m2, min_slope,
         cov_w, cov_l, weight, waste, sup_name, sup_price, currency) = row[:12]

        if cat_name not in cat_map:
            cat_map[cat_name] = MaterialCategory(id=uuid.uuid4(), name=cat_name)
            db_session.add(cat_map[cat_name])

        if mfr_name not in mfr_map:
            mfr_map[mfr_name] = MaterialManufacturer(id=uuid.uuid4(), name=mfr_name)
            db_session.add(mfr_map[mfr_name])

        if sup_name not in sup_map:
            sup_map[sup_name] = Supplier(id=uuid.uuid4(), name=sup_name)
            db_session.add(sup_map[sup_name])

        sup = sup_map[sup_name]

        pl_name = f"{sup_name} cennik"
        pl = db_session.query(PriceList).where(
            PriceList.supplier_id == sup.id, PriceList.name == pl_name).first()
        if not pl:
            pl = PriceList(id=uuid.uuid4(), supplier_id=sup.id, name=pl_name,
                           currency=currency.strip() if currency else 'EUR')
            db_session.add(pl)

        existing = db_session.query(ORMMaterial).where(ORMMaterial.name == mat_name).first()
        if not existing:
            mat = ORMMaterial(
                id=uuid.uuid4(), name=mat_name, category_id=cat_map[cat_name].id,
                manufacturer_id=mfr_map[mfr_name].id, unit_cost=float(price_m2),
                unit_of_measure=MaterialUnit.SQUARE_METER,
                covering_width_m=float(cov_w) if cov_w else None,
                covering_length_m=float(cov_l) if cov_l else None,
                min_slope_deg=float(min_slope) if min_slope else None,
                weight_kg_per_m2=float(weight) if weight else None,
                waste_factor=float(waste) if waste else 0.10,
                is_active=True,
            )
            db_session.add(mat)
            db_session.flush()
            mat_id = mat.id
        else:
            mat_id = existing.id

        price_val = float(sup_price) if sup_price.strip() else float(price_m2)
        pi = PriceItem(id=uuid.uuid4(), material_id=mat_id, price_list_id=pl.id, unit_price=price_val)
        db_session.add(pi)
        imported += 1
        print(f"  + {mat_name} @ {price_val} EUR/m2 ({sup_name})")

    db_session.commit()
    return imported


if __name__ == "__main__":
    csv_path = r'C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio\data\cennik_priklad.csv'
    example = [
        "kategoria,vyrobca,nazov_materialu,cena_za_m2,min_sklon_stupne,krycia_sirka_m,krycia_dlzka_m,hmotnost_kg_m2,odpad_faktor,dodavatel,cena_dodavatela_m2,mena",
        "Skridla,Tondach,Tondach Figaro 11,14.50,22,0.300,0.340,45,0.12,LAD,14.50,EUR",
        "Plechova krytina,Ruukki,Ruukki Classic,11.20,14,1.100,2.000,5,0.08,Dektrade,11.80,EUR",
        "# Toto je komentar - ignoruje sa",
        "Bitumenovy sindel,Katepal,Katepal Ambient,9.80,12,0.300,1.000,9,0.10,LAD,9.80,EUR",
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(example))
    print(f"Prikladovy CSV vytvoreny: {csv_path}")