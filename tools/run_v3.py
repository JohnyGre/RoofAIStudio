#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_v3 — vylepšená v3 pipeline (deterministická, bez LLM v geometrii).

Reťaz:
    GPS (alebo adresa) ─┬─ LiDAR (LAZ, trieda 6) ──> ENGINE (segmentácia → polygóny → hrany)
                        └─ GIS (obrys budovy)     ──┘
                                                 ↓
                          KONTRAKT v3 + REGISTRÁCIA + QA  →  JSON / PNG / report

Spustenie:
    .venv\\Scripts\\python.exe tools\\run_v3.py                       # default lokalita
    .venv\\Scripts\\python.exe tools\\run_v3.py --lat 48.3954 --lon 17.5860
    .venv\\Scripts\\python.exe tools\\run_v3.py --address "Átriová 9309/16, Trnava"
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from app.core import contract, engine, gis, ortho, preprocess, qa, registration, vision

DEFAULT = {"lat": 48.39559280067209, "lon": 17.585647957122642, "name": "Triova_7751_16A_Trnava"}
LAZ_DIR = ROOT / "data" / "laz"
MODEL = ROOT / "ai_models" / "roof_gmaps_v2_last.pt"
OUT = ROOT / "output" / "v3"


def geocode(address: str):
    import ssl
    import urllib.parse
    import urllib.request
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": address, "format": "json", "limit": 1})
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "RoofAIStudio/3.0"})
    with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
        data = json.loads(r.read())
    if not data:
        raise SystemExit(f"Adresa sa nenašla: {address}")
    return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", address)


def load_lidar(lat: float, lon: float, radius_m: float, laz_dir: Path):
    """Načíta klasifikované body budov (trieda 6) a terén (trieda 2) v okolí."""
    import laspy
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", "EPSG:8353", always_xy=True)
    ex, ny = t.transform(lon, lat)

    bpts, gpts, used = [], [], []
    for f in sorted(glob.glob(str(laz_dir / "*.laz"))):
        if os.path.getsize(f) < 50_000:
            continue
        try:
            las = laspy.read(f)
        except Exception:
            continue
        xs, ys, zs = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        cls = np.asarray(las.classification, dtype=np.uint8)
        d = np.sqrt((xs - ex) ** 2 + (ys - ny) ** 2)
        near = d < radius_m
        if not near.any():
            continue
        m6 = near & (cls == 6)
        m2 = near & (cls == 2)
        if m6.any():
            bpts.append(np.column_stack([xs[m6], ys[m6], zs[m6]]))
            used.append(os.path.basename(f))
        if m2.any():
            gpts.append(np.column_stack([xs[m2], ys[m2], zs[m2]]))
    if not bpts:
        raise SystemExit(f"Žiadne body triedy 6 v okolí {radius_m} m (laicky: budova nenájdená v LAZ).")
    building = np.vstack(bpts)
    ground = np.vstack(gpts) if gpts else np.zeros((0, 3))
    return building, ground, used, (ex, ny)


def downsample(pts: np.ndarray, voxel: float = 0.25) -> np.ndarray:
    if voxel <= 0 or len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def topdown_png(planes, footprint_ring, lat, lon, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, pl in enumerate(planes):
        p = pl["points"]
        ax.scatter(p[:, 0], p[:, 1], s=1, color=colors[i % 10], alpha=0.55)
        v2 = pl.get("vertices_2d")
        if v2:
            xs = [v[0] for v in v2] + [v2[0][0]]
            ys = [v[1] for v in v2] + [v2[0][1]]
            ax.plot(xs, ys, color=colors[i % 10], lw=0.9, alpha=0.45)
            ax.plot([], [], color=colors[i % 10], lw=2, label=f"R{i+1} {pl['slope_deg']:.0f}° / {pl['area_m2']:.0f} m²")
        # kreslíme len nosné hrany — nie každý krátky úsek (menej čiar)
        for e in pl.get("edges", []):
            if e["type"] == "s" or e["length_m"] < 2.5:
                continue
            c = {"h": "#4da3ff", "n": "#38d39f", "u": "#ff5a5a", "o": "#ffd23f"}.get(e["type"], "#ffffff")
            ax.plot([e["start"][0], e["end"][0]], [e["start"][1], e["end"][1]], color=c, lw=3.2,
                    solid_capstyle="round", zorder=5)
    if footprint_ring:
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:8353", always_xy=True)
        xs, ys = [], []
        for lo, la in footprint_ring:
            x, y = t.transform(lo, la)
            xs.append(x); ys.append(y)
        xs.append(xs[0]); ys.append(ys[0])
        ax.plot(xs, ys, color="#ffffff", lw=1.6, ls="--", label="GIS obrys (OSM)")
    ax.set_aspect("equal")
    ax.set_title("RoofAIStudio v3 — roviny a klasifikované hrany", color="#e0e1dd")
    ax.tick_params(colors="#a0a8b4")
    for s in ax.spines.values():
        s.set_color("#2d3e50")
    leg = ax.legend(facecolor="#1b263b", edgecolor="#2d3e50", labelcolor="#e0e1dd", fontsize=8)
    ax.set_xlabel("X [m] (S-JTSK/JTSK03)", color="#a0a8b4")
    ax.set_ylabel("Y [m]", color="#a0a8b4")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--address")
    ap.add_argument("--name")
    ap.add_argument("--radius", type=float, default=22.0)
    ap.add_argument("--voxel", type=float, default=0.15)
    ap.add_argument("--height-band", type=float, default=7.0, help="výškový pás okolo stredu strechy (m)")
    ap.add_argument("--vision-model", type=str, default="roof_gmaps_v2_last.pt",
                    help="názov modelu v ai_models/ (determinismus)")
    ap.add_argument("--ortho", action="store_true", help="stiahni ortofoto v max. rozlíšení a prekry geometriu")
    ap.add_argument("--ortho-size", type=int, default=4096)
    ap.add_argument("--ortho-margin", type=float, default=0.08, help="okraj výrezu (podiel rozsahu)")
    ap.add_argument("--preprocess", action="store_true",
                    help="vyrob GeoTIFF + dlaždicové masky + GeoJSON (zrozumiteľný formát)")
    args = ap.parse_args()

    if args.address:
        lat, lon, display = geocode(args.address)
        name = args.name or args.address
    else:
        lat = args.lat if args.lat is not None else DEFAULT["lat"]
        lon = args.lon if args.lon is not None else DEFAULT["lon"]
        display = name = args.name or DEFAULT["name"]

    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)[:48]
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"RoofAIStudio v3 — {display}")
    print(f"GPS: {lat:.7f}, {lon:.7f} | rádius {args.radius:.0f} m")
    print("=" * 78)

    # 1) GIS obrys (planimetrická autorita) — potrebujeme ho pred orezaním mračna
    fp = gis.footprint_for(lat, lon, prefer="osm")
    print(f"[1] GIS obrys: {fp.get('footprint_m2')} m² (zdroj: {fp.get('source')})")
    if fp.get("accuracy_note"):
        print(f"      pozn.: {fp['accuracy_note']}")
    clip_ring = engine.reproject_ring(fp["ring_lonlat"]) if fp.get("ring_lonlat") else None

    # 2) LiDAR
    building, ground, used, xy = load_lidar(lat, lon, args.radius, LAZ_DIR)
    print(f"[2] LiDAR: {len(building):,} bodov triedy 6 | terén {len(ground):,} | súborov: {len(used)}")
    for u in used:
        print("      -", u)
    # Výškový pás: Y-strecha v rade domov — susedné budovy majú inú výšku strechy
    d_seed = np.sqrt(((building[:, :2] - np.asarray(xy)) ** 2).sum(axis=1))
    seed = building[d_seed < 5.0]
    if len(seed) > 50:
        z0 = float(np.median(seed[:, 2]))
        keep_mask = np.abs(building[:, 2] - z0) <= args.height_band
        print(f"      výškový pás: stred strechy z={z0:.2f} m, ±{args.height_band:.1f} m → "
              f"{int(keep_mask.sum()):,} z {len(building):,} bodov")
        building = building[keep_mask]

    pts = downsample(building, args.voxel)
    print(f"      po voxel downsample ({args.voxel} m): {len(pts):,} bodov")
    pts_sep = engine.separate_buildings(pts, xy)
    if len(pts_sep) != len(pts):
        print(f"      oddelenie budovy: {len(pts)} → {len(pts_sep):,} bodov (zvyšok patrí susedom)")
    pts = pts_sep
    if clip_ring:
        pts_clip = engine.clip_to_polygon(pts, clip_ring, buffer_m=2.0)
        if len(pts_clip) > 200:
            print(f"      orezanie GIS obrysom (+2 m): {len(pts):,} → {len(pts_clip):,} bodov")
            pts = pts_clip
        else:
            print("      orezanie GIS obrysom preskočené (málo bodov v obryse)")
    elif fp.get("footprint_m2"):
        # Bez geometrie obrysu neorezávame kruhom — Y-strecha má krídla ďaleko od
        # stredu a kruh by ich odrezal. Použijeme celý klaster budovy a spoľahne­me
        # sa na QA kontrolu plochy voči pôdorysu.
        print(f"      POZOR: GIS obrys bez geometrie (sieť) → používam celý klaster budovy "
              f"({len(pts):,} bodov); plochu stráži QA kontrola")
        print("      FLAG: FOOTPRINT_GEOMETRY_APPROXIMATED (doplniť geometrický obrys ZBGIS/OSM)")
    else:
        print("      POZOR: GIS obrys nedostupný (sieť aj cache) → FLAG NO_GIS_FOOTPRINT")


    # 2b) Ortofoto + vycvičený segmentačný model → obrys a masky strechy
    ortho_bytes, ortho_bbox, vision_masks = None, None, None
    try:
        if MODEL.exists():
            ortho_bbox = ortho.bbox_3857_from_points(building)
            ortho_bytes = ortho.fetch_zbgis_bbox(ortho_bbox, size=args.ortho_size)
            if ortho_bytes:
                # Bod 3: vyskúšaj všetky natrénované modely a vyber ten, ktorý sedí s mračnom
                from shapely.geometry import MultiPoint as _MP2, Polygon as _Pg2
                from shapely import concave_hull as _ch2
                _hull = _ch2(_MP2([tuple(b[:2]) for b in building]), ratio=0.4)
                det, best_iou, best_name = None, -1.0, None
                _pref = args.vision_model
                _cands = []
                for _m in ([_pref] if _pref else []) + ["roof_gmaps_v2_last.pt"]:
                    if _m and _m not in _cands:
                        _cands.append(_m)
                _best = None
                for _mname in _cands:
                    _mp = ROOT / "ai_models" / _mname
                    if not _mp.exists():
                        print(f"      VISION: model {_mname} sa nenašiel")
                        continue
                    try:
                        _d = vision.detect_from_ortho(ortho_bytes, ortho_bbox, str(_mp), conf=0.15)
                        _fp = _d.get("footprint_xy")
                        _iou = -1.0
                        if _fp and len(_fp) >= 3:
                            _p = _Pg2([(x, y) for x, y in _fp]).buffer(0)
                            _iou = _p.intersection(_hull).area / max(_p.union(_hull).area, 1e-6)
                        print(f"      VISION {_mname}: obrys {(_d.get('area_m2') or 0):.0f} m², "
                              f"masky {_d.get('masks')}, IoU voči mračnu {_iou:.3f}")
                        if _best is None or _iou > best_iou:
                            _best, best_iou, best_name = _d, _iou, _mname
                    except Exception as _me:
                        print(f"      VISION {_mname}: zlyhalo ({_me})")
                det = _best
                if det is not None:
                    print(f"      VISION použitý model: {best_name} (IoU {best_iou:.3f})")
                print(f"      VISION model: masky {det.get('masks')}, triedy {det.get('classes')}, "
                      f"obrys {det.get('area_m2')} m²")
                if det.get("masks_detail"):
                    # co-registrácia masiek voči mračnu (deterministicky, max IoU)
                    from shapely import concave_hull as _ch
                    from shapely.geometry import MultiPoint as _MP
                    _hull = _ch(_MP([tuple(p[:2]) for p in pts]), ratio=0.4)
                    hull_ring = list(_hull.simplify(0.6).exterior.coords)
                    dx, dy, iou = engine.register_offset(
                        [[float(x), float(y)] for x, y in hull_ring],
                        [m["polygon_xy"] for m in det["masks_detail"]])
                    if (abs(dx) > 0.01 or abs(dy) > 0.01) and iou >= 0.45:
                        for m in det["masks_detail"]:
                            m["polygon_xy"] = [[x + dx, y + dy] for x, y in m["polygon_xy"]]
                        if det.get("footprint_xy"):
                            det["footprint_xy"] = [[x + dx, y + dy] for x, y in det["footprint_xy"]]
                    elif abs(dx) > 0.01 or abs(dy) > 0.01:
                        print(f"      registrácia zamietnutá (IoU {iou:.3f} < 0,45) → "
                              f"FLAG REGISTRATION_LOW_CONFIDENCE, masky ostávajú bez posunu")
                        dx, dy = 0.0, 0.0
                    vision_masks = det["masks_detail"]
                    print(f"      VISION: {len(vision_masks)} masiek | co-registrácia: posun "
                          f"({dx:+.2f}, {dy:+.2f}) m, IoU {iou:.3f} → segmentácia riadená modelom")
                if det.get("footprint_xy"):
                    pts_v = engine.clip_to_polygon(pts, det["footprint_xy"], buffer_m=1.5)
                    if len(pts_v) > 300:
                        print(f"      orezanie obrysom z modelu (+1,5 m): {len(pts):,} → {len(pts_v):,} bodov")
                        print("      FLAG: FOOTPRINT_FROM_VISION (GIS geometria chýba; zdroj je model)")
                        pts = pts_v
                else:
                    print("      VISION: obrys sa nedal zostaviť —", det.get("note"))
        else:
            print(f"      VISION: model {MODEL.name} sa nenašiel")
    except Exception as _e:
        print("      VISION zlyhalo:", _e)

    # 3) ENGINE — roviny z mračna (LiDAR je autorita geometrie).
    # Model sa používa ako planimetria (obrys) a typológia, nie ako definícia plôch:
    # jeho masky sú TYPY striech (slope_flat/min/poly/trap/tri), nie jednotlivé roviny
    # — fit na masku preto dáva nezmyselné sklony (namerané 25–65°).
    planes = engine.build_roof_geometry(pts)
    if vision_masks:
        print(f"      roviny z mračna (RANSAC); model prispel obrysom a typológiou "
              f"({len(vision_masks)} typových masiek)")
    print(f"[3] ENGINE: {len(planes)} rovín")
    # kratky vypis
    n_edges = sum(len(pl.get("edges", [])) for pl in planes)
    for i, pl in enumerate(planes, 1):
        types = {}
        for e in pl.get("edges", []):
            types[e["type"]] = types.get(e["type"], 0) + 1
        print(f"      R{i}: {pl['slope_deg']}° / {pl['area_m2']:.0f} m² / {len(pl.get('edges', []))} hrán {types}")
    print(f"      spolu: {len(planes)} rovín, {n_edges} hrán")
    from collections import Counter
    et = Counter(e["type"] for pl in planes for e in pl["edges"])
    print(f"      hrany podľa typu: " + ", ".join(f"{k}={et.get(k, 0)}" for k in ("o", "h", "n", "u", "s")))
    st = Counter(engine.classify_plane_subtype(pl) for pl in planes)
    print(f"      podtypy rovín: " + ", ".join(f"{k}={v}" for k, v in st.items()))

    # 4) KONTRAKT
    # odkvapy z obrysu CELEJ strechy (nie z obrysu každej roviny)
    o_edges = engine.roof_outline_edges(planes)
    for e in o_edges:
        planes_mid = None
        for pl in planes:
            v = pl.get("vertices_2d") or []
            if len(v) >= 3:
                from shapely.geometry import Point as _Pt, Polygon as _Pg
                if _Pg(v).distance(_Pt(e["start"][0], e["start"][1])) < 0.05:
                    planes_mid = pl
                    break
        if planes_mid is not None:
            # nepridávaj, ak na tej istej čiare už hrana existuje (inak vznikne duplicita)
            def _same(e1, e2, dist_tol=0.30, ang_tol=8.0):
                import math as _m
                dx1, dy1 = e1["end"][0] - e1["start"][0], e1["end"][1] - e1["start"][1]
                n1 = _m.hypot(dx1, dy1) or 1.0
                dx1, dy1 = dx1 / n1, dy1 / n1
                dx2, dy2 = e2["end"][0] - e2["start"][0], e2["end"][1] - e2["start"][1]
                n2 = _m.hypot(dx2, dy2) or 1.0
                dx2, dy2 = dx2 / n2, dy2 / n2
                ang = _m.degrees(_m.acos(min(1.0, abs(dx1 * dx2 + dy1 * dy2))))
                if ang > ang_tol:
                    return False
                mx = (e2["start"][0] + e2["end"][0]) / 2.0
                my = (e2["start"][1] + e2["end"][1]) / 2.0
                return abs(-dy1 * (mx - e1["start"][0]) + dx1 * (my - e1["start"][1])) <= dist_tol

            if any(_same(x, e) for x in pl["edges"]):
                continue
            pl["edges"] = [x for x in pl["edges"] if x["type"] != "o" or not _same(x, e)]
            pl["edges"].append({"id": f"o{len(pl['edges'])+1}", "type": e["type"],
                                "length_m": e["length_m"],
                                "start": e["start"], "end": e["end"], "exact": False})
    print(f"      odkvapy z obrysu strechy: {len(o_edges)} úsekov")
    _ring = engine.roof_outline_ring(planes)
    if _ring:
        import math as _m
        _len = sum(_m.hypot(_ring[i][0] - _ring[i - 1][0], _ring[i][1] - _ring[i - 1][1])
                   for i in range(len(_ring)))
        print(f"      odkvapový obvod (obrys strechy): uzavretý, dĺžka {_len:.2f} m, bodov {len(_ring)}")
    _loops = engine.eave_loop_summary([e for pl in planes for e in (pl.get("edges") or [])])
    print(f"      rozpadnuté úseky odkvapu: {sum(1 for _l in _loops if not _l['closed'])}")

    # Komíny a potenciálne solárne panely (bod 4)
    chimneys, panels = engine.detect_features_from_cloud(planes, pts)
    print(f"      komíny: {len(chimneys)} | potenciálne solárne panely: {len(panels)}")
    for c in chimneys[:3]:
        where = ("R%d" % (c["plane"] + 1)) if "plane" in c else ("(%.1f, %.1f)" % (c["x"], c["y"]))
        print(f"        komín @ {where}: {c['area_m2']} m², {c['points']} bodov")
    for c in panels[:3]:
        where = ("R%d" % (c["plane"] + 1)) if "plane" in c else ("(%.1f, %.1f)" % (c["x"], c["y"]))
        print(f"        panel @ {where}: {c['area_m2']} m², {c['points']} bodov")

    # presné hrany z priesečníc rovín (hrebeň / nárožie / úžľabie)
    ex_edges = engine.intersection_edges(planes)
    exact_by_plane = {}
    for e in ex_edges:
        for k in e.get("pair", []):
            exact_by_plane.setdefault(k, []).append(e)
    from collections import Counter as _C
    print(f"      presné hrany z priesečníc: " + ", ".join(
        f"{t}={n}" for t, n in _C(e['type'] for e in ex_edges).items()) or "      presné hrany: 0")

    # Deduplikácia: ak existuje presná hrana z priesečnice, polygónová hrana na tej istej
    # čiare sa odstráni (audit: tie isté hrany boli v kontrakte 2-3x)
    def _same_line_lite(e1, e2, dist_tol=0.30, ang_tol=8.0, min_overlap=1.0):
        import math as _m

        def d(e):
            dx, dy = e["end"][0] - e["start"][0], e["end"][1] - e["start"][1]
            n = _m.hypot(dx, dy) or 1.0
            return dx / n, dy / n
        d1, d2 = d(e1), d(e2)
        ang = _m.degrees(_m.acos(min(1.0, abs(d1[0] * d2[0] + d1[1] * d2[1]))))
        if ang > ang_tol:
            return False
        mx = (e2["start"][0] + e2["end"][0]) / 2.0
        my = (e2["start"][1] + e2["end"][1]) / 2.0
        dist = abs(-d1[1] * (mx - e1["start"][0]) + d1[0] * (my - e1["start"][1]))
        if dist > dist_tol:
            return False
        ts = [d1[0] * (p[0] - e1["start"][0]) + d1[1] * (p[1] - e1["start"][1])
              for p in (e2["start"], e2["end"])]
        L1 = _m.hypot(e1["end"][0] - e1["start"][0], e1["end"][1] - e1["start"][1])
        return max(0.0, min(max(ts), L1) - max(min(ts), 0.0)) >= min_overlap

    for _i, _pl in enumerate(planes):
        _ex = exact_by_plane.get(_i, [])
        if not _ex:
            continue
        _keep = []
        for _e in _pl.get("edges", []):
            # kanonický záznam je X (z priesečnice); engine hrany na tej istej čiare
            # sa vyhodia — aj keď majú exact=True (inak vznikali duplicity v kontrakte)
            if any(_same_line_lite(_x, _e) for _x in _ex):
                continue
            _keep.append(_e)
        _pl["edges"] = _keep

    # ── finálna konzistencia typov hrán (audit 2026-10-05) ─────────────────
    # 1) odkvap musí byť vodorovný (inak je to štít) — Z sa berie z vlastnej roviny
    for _pi, _pl in enumerate(planes):
        for _e in _pl.get("edges", []):
            if _e.get("type") != "o":
                continue
            _L = math.hypot(_e["end"][0] - _e["start"][0], _e["end"][1] - _e["start"][1])
            _sl = math.degrees(math.atan2(abs(_e["end"][2] - _e["start"][2]), max(_L, 1e-9)))
            if _sl > 6.0:
                _e["type"] = "s"
                _e["id"] = "s" + _e["id"][1:]

    # 1b) 's' hrana ležiaca na PRESNEJ hrane (X) inej roviny prevezme jej typ —
    #     spoločná hrana nemôže byť štít ani v jednom z páru (audit, Beluj R7/R8)
    for _i, _pl in enumerate(planes):
        _others = [e for _j, _ex in exact_by_plane.items() if _j != _i for e in _ex]
        if not _others:
            continue
        for _e in _pl.get("edges", []):
            if _e.get("type") != "s":
                continue
            for _f in _others:
                if _same_line_lite(_e, _f):
                    _e["type"] = _f["type"]
                    _e["id"] = _f["type"] + _e["id"][1:]
                    break

    # 2) spoločná hrana nesmie byť 's' → pretypuj podľa konvexnosti oboch rovín
    import numpy as _np
    _seen_pairs = set()
    for _i, _pl in enumerate(planes):
        for _e in _pl.get("edges", []):
            if _e.get("type") != "s":
                continue
            for _j, _q in enumerate(planes):
                if _i >= _j or (_i, _j) in _seen_pairs:
                    continue
                for _f in _q.get("edges", []):
                    if _f.get("type") != "s":
                        continue
                    if not _same_line_lite(_e, _f):
                        continue
                    _seen_pairs.add((_i, _j))
                    _mid = _np.array([(_e["start"][0] + _e["end"][0]) / 2.0,
                                      (_e["start"][1] + _e["end"][1]) / 2.0,
                                      (_e["start"][2] + _e["end"][2]) / 2.0])
                    _dirv = _np.array([_e["end"][0] - _e["start"][0],
                                       _e["end"][1] - _e["start"][1],
                                       _e["end"][2] - _e["start"][2]])
                    _conv = engine._fold_convex(_pl, _q, _mid, _dirv)
                    if _conv is True:
                        _t = engine._ridge_or_hip(_pl, _q)
                    elif _conv is False:
                        _t = "u"
                    else:
                        _t = "n"
                    _e["type"] = _t
                    _e["id"] = _t + _e["id"][1:]
                    _f["type"] = _t
                    _f["id"] = _t + _f["id"][1:]

    crec = []
    for i, pl in enumerate(planes, 1):
        slope = pl["slope_deg"]
        ptype = engine.classify_plane_subtype(pl)
        _cos = math.cos(math.radians(min(89.0, slope))) if slope else 1.0
        crec.append(contract.PlaneRecord(
            id=f"R{i}", type=ptype, pitch_deg=slope, area_m2=round(pl["area_m2"], 2),
            area_true_m2=round(pl["area_m2"] / _cos, 2) if _cos > 0.01 else round(pl["area_m2"], 2),
            azimuth_deg=pl["azimuth_deg"], rmse_m=pl["rmse_m"],
            vertices=[list(v) for v in pl.get("vertices_3d", [])],
            edges=[contract.EdgeRecord(**{k: e[k] for k in ("id", "type", "length_m", "start", "end", "exact")})
                   for e in pl.get("edges", [])] + [
                      contract.EdgeRecord(id=f"X{e['type']}{n+1}", type=e["type"], length_m=round(e["length_m"], 3),
                                          start=e["start"], end=e["end"], exact=True)
                      for n, e in enumerate(exact_by_plane.get(i - 1, []))],
        ))
    ground_z = float(ground[:, 2].mean()) if len(ground) else None
    roof_area = round(sum(p.area_m2 for p in crec), 2)
    model = contract.RoofModel(
        address=display, gps={"lat": lat, "lon": lon}, planes=crec,
        sources=[
            contract.SourceRecord(id="lidar_lls", role="lidar", crs="EPSG:8353", acquired_year=2018,
                                  license="CC BY 4.0", note="ZBGIS/MAPKA LLS 1. cyklus"),
            contract.SourceRecord(id="osm_buildings", role="gis", crs="EPSG:4326", acquired_year=2024,
                                  license="ODbL", note=fp.get("accuracy_note", "")),
        ],
        ground_mnm=ground_z, footprint_m2=fp.get("footprint_m2"), roof_area_m2=roof_area,
    )
    base = OUT / f"{safe}"
    (OUT).mkdir(parents=True, exist_ok=True)
    Path(str(base) + "_roofmodel_v3.json").write_text(model.to_json(), encoding="utf-8")

    # 5) REGISTRÁCIA + QA
    conv = registration.normalize(lat, lon, "EPSG:4326")
    reg = registration.registration_report([s.__dict__ for s in model.sources])
    qa_res = qa.run_all_checks(model)
    errs = contract.validate(model)
    Path(str(base) + "_qa.json").write_text(json.dumps(
        {"contract_errors": errs, "qa": qa_res, "registration": reg,
         "vision": {"masks": len(vision_masks) if vision_masks else 0,
                    "classes": (det.get("classes") if isinstance(vision_masks, list) and 'det' in dir() else None)},
         "local_xy": {"x": conv["x"], "y": conv["y"], "crs": conv["crs"]}},
        indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[4] KONTRAKT: {len(crec)} rovín, plocha {roof_area} m², pôdorys {fp.get('footprint_m2')} m²")
    print(f"      chyby kontraktu: {len(errs)} → {errs if errs else 'žiadne'}")
    print(f"[5] REGISTRÁCIA: {conv['crs']} = ({conv['x']}, {conv['y']}); "
          f"časové flagy: {len(reg['temporal_flags'])}")
    for t in reg["temporal_flags"]:
        print(f"      TEMPORAL: {t['a']} vs {t['b']} → {t['gap_years']} rokov")
    print(f"[6] QA: {qa_res['verdict']} (errors={qa_res['counts']['errors']}, warnings={qa_res['counts']['warnings']})")
    for i in (qa_res["errors"] + qa_res["warnings"])[:3]:
        print(f"      [{i.get('severity')}] {i.get('check')}: {i.get('detail', '')}")
    extra = len(qa_res["errors"] + qa_res["warnings"]) - 3
    if extra > 0:
        print(f"      ... a {extra} ďalších (celé v *_qa.json)")

    # 6) NÁHĽAD
    png = Path(str(base) + "_topdown.png")
    try:
        topdown_png(planes, fp.get("ring_lonlat"), lat, lon, png)
        print(f"[7] náhľad: {png}")
    except Exception as e:
        print(f"[7] náhľad zlyhal: {e}")

    # 7) ORTOFOTO (max. rozlíšenie) + prekrytie geometrie
    if args.ortho:
        try:
            allp = np.vstack([pl["points"] for pl in planes])
            span = (float(allp[:, 0].max() - allp[:, 0].min()), float(allp[:, 1].max() - allp[:, 1].min()))
            bbox = ortho.bbox_3857_from_points([p for pl in planes for p in pl["points"]],
                                               margin_frac=args.ortho_margin)
            side = bbox[2] - bbox[0]
            k_merc = 1.0 / math.cos(math.radians(lat))   # Web Mercator nafukuje metre
            side_ground = side / k_merc
            print(f"[8] ORTOFOTO: rozsah strechy {span[0]:.1f} × {span[1]:.1f} m → výrez "
                  f"{side_ground:.1f} × {side_ground:.1f} m pri {args.ortho_size} px "
                  f"({side_ground / args.ortho_size * 100:.2f} cm/px)")
            if ortho_bytes:
                img, bbox = ortho_bytes, (ortho_bbox or bbox)
                src = "ZBGIS (+Vision obrys)"
            else:
                img = ortho.fetch_zbgis_bbox(bbox, size=args.ortho_size)
                src = "ZBGIS"
            if img is None:
                img = ortho.fetch_osm_bbox(bbox, zoom=19)
                src = "OSM (fallback)"
            if img is None:
                print("      ortofoto sa nepodarilo stiahnuť (sieť) → FLAG NO_ORTHOPHOTO")
            else:
                jpg = str(base) + "_ortofoto_max.jpg"
                Path(jpg).write_bytes(img)
                ov = ortho.save_overlay(img, bbox, planes, clip_ring, str(base) + "_kontrola_hran_na_ortofote.png")
                print(f"      zdroj: {src} | uložené: {Path(jpg).name}, {Path(ov).name}")
            # izometrický výkres z kontraktu
            import subprocess
            try:
                subprocess.run([sys.executable, str(ROOT / "tools" / "render_iso.py"),
                                str(base) + "_roofmodel_v3.json"], check=False, cwd=str(ROOT))
            except Exception as _e:
                print("      výkres zlyhal:", _e)

            # Predspracovanie do zrozumiteľného formátu (GeoTIFF + masky + GeoJSON)
            if args.preprocess and ortho_bytes:
                try:
                    _pp = preprocess.run_preprocess(ortho_bytes, bbox, str(MODEL), safe,
                                                    OUT)
                    print("      preprocess: " + ", ".join(_pp.get("outputs", {}).values()))
                    print(f"      preprocess triedy: {_pp.get('class_counts')}")
                except Exception as _pe:
                    print("      preprocess zlyhal:", _pe)

            # DXF pre projektanta + HTML report (a otvor ho)
            try:
                subprocess.run([sys.executable, str(ROOT / "tools" / "v3_export.py"),
                                str(base) + "_roofmodel_v3.json"], check=False, cwd=str(ROOT))
                import webbrowser
                webbrowser.open(Path(str(base) + "_report.html").resolve().as_uri())
            except Exception as _e:
                print("      export/report zlyhal:", _e)
        except Exception as e:
            print(f"      ortofoto zlyhalo: {e}")

    print(f"\nVýstupy:\n  {base}_roofmodel_v3.json\n  {base}_qa.json\n  {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
