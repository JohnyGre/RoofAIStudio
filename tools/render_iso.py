#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""render_iso — izometrický technický výkres strechy (ako referencia).

Vstup: v3 kontrakt (`*_roofmodel_v3.json`)
Výstup: SVG + PNG (cez headless prehliadač), štýl:
  * biely podklad, tenké čierne kóty, hrubé hrany strechy,
  * svetlosivé plochy s jemnou mriežkou,
  * slovenské popisky s vodiacimi čiarami (Hrebeň, Úžľabie, Nárožia, Okapy),
  * kóty rozmerov a sklon strechy, severka vpravo dole.

Spustenie:
    .venv\\Scripts\\python.exe tools\\render_iso.py output\\v3\\X_roofmodel_v3.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

C = math.cos(math.radians(30.0))
S = math.sin(math.radians(30.0))

EDGE_STYLE = {
    "o": dict(w=2.6, dash="", name="Okap"),
    "h": dict(w=3.0, dash="", name="Hrebeň"),
    "n": dict(w=2.6, dash="", name="Nárožie"),
    "u": dict(w=2.6, dash="", name="Úžľabie"),
    "s": dict(w=1.6, dash="6 4", name="Štít"),
}


def iso(x, y, z):
    """Izometrická projekcia (x, y vodorovné, z výška)."""
    u = (x - y) * C
    v = (x + y) * S - z * 1.25
    return u, v


def main() -> int:
    if len(sys.argv) < 2:
        print("Použitie: python tools/render_iso.py <roofmodel_v3.json> [vystup.svg]")
        return 2
    src = Path(sys.argv[1])
    if not src.exists():
        print("CHYBA: neexistuje", src)
        return 2
    model = json.loads(src.read_text(encoding="utf-8-sig"))
    planes = [p for p in model.get("planes", []) if p.get("vertices")]
    if not planes:
        print("CHYBA: model nemá roviny s vrcholmi")
        return 2

    # 1) projekcia všetkých vrcholov
    proj = []
    for p in planes:
        vv = [iso(v[0], v[1], v[2]) for v in p["vertices"]]
        proj.append(vv)
    xs = [u for vv in proj for u, _ in vv]
    ys = [v for vv in proj for _, v in vv]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    W = maxx - minx
    H = maxy - miny
    pad = 150.0
    scale = min((1400.0 - 2 * pad) / max(W, 1e-6), (900.0 - 2 * pad) / max(H, 1e-6))
    scale = max(min(scale, 6.0), 0.05)

    def px(u, v):
        return (pad + (u - minx) * scale, 880.0 - pad - (v - miny) * scale)

    # 2) telo výkresu
    out = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append('<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">')
    out.append('<defs><marker id="a" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#111"/></marker><pattern id="mesh" width="10" height="10" patternUnits="userSpaceOnUse">'
               '<path d="M10 0 L0 0 0 10" fill="none" stroke="#dcdcdc" stroke-width="0.6"/></pattern></defs>')
    out.append('<rect width="1400" height="900" fill="#ffffff"/>')
    out.append(f'<text x="40" y="46" font-family="Arial" font-size="20" font-weight="bold">{model.get("address","")}</text>')
    out.append(f'<text x="40" y="70" font-family="Arial" font-size="13" fill="#555">'
               f'Izometrický výkres strechy — roviny a hrany (S-JTSK/JTSK03, Bpv) · výška z LiDARu</text>')

    # plochy
    for vv in proj:
        pts = " ".join(f"{px(u, v)[0]:.1f},{px(u, v)[1]:.1f}" for u, v in vv)
        out.append(f'<polygon points="{pts}" fill="#f4f4f4" fill-opacity="0.95" stroke="#b9b9b9" stroke-width="1"/>')
    for vv in proj:
        pts = " ".join(f"{px(u, v)[0]:.1f},{px(u, v)[1]:.1f}" for u, v in vv)
        out.append(f'<polygon points="{pts}" fill="url(#mesh)" stroke="none"/>')

    # hrany
    labelled = {}
    for p, vv in zip(planes, proj):
        for e in p.get("edges", []):
            t = e.get("type", "s")
            # kreslíme len presné hrany (z priesečníc) a odkvapy — nie zubaté "štíty"
            if not (e.get("exact") or (t == "o" and float(e.get("length_m") or 0) >= 2.0)):
                continue
            st = EDGE_STYLE.get(t, EDGE_STYLE["s"])
            a = iso(*e["start"])
            b = iso(*e["end"])
            x1, y1 = px(*a)
            x2, y2 = px(*b)
            dash = f' stroke-dasharray="{st["dash"]}"' if st["dash"] else ""
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                       f'stroke="#111" stroke-width="{st["w"]}" stroke-linecap="round"{dash}/>')
            # zapamätaj najdlhšiu hranu daného typu pre popisok
            L = float(e.get("length_m") or 0)
            if t in ("h", "u", "n", "o") and L > labelled.get(t, (0,))[0]:
                labelled[t] = (L, (x1, y1, x2, y2))

    # popisky s vodiacimi čiarami (Hrebeň, Úžľabie, Nárožie, Okapy)
    positions = {"h": (0.5, -1.0), "u": (1.0, -0.6), "n": (-1.0, 0.4), "o": (0.4, 1.0)}
    for t, (L, (x1, y1, x2, y2)) in labelled.items():
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = positions.get(t, (0, -1))
        lx, ly = mx + dx * 120, my + dy * 90
        lx = min(max(lx, 90), 1310)
        ly = min(max(ly, 90), 830)
        out.append(f'<line x1="{mx:.1f}" y1="{my:.1f}" x2="{lx:.1f}" y2="{ly:.1f}" stroke="#111" stroke-width="0.9"/>')
        out.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="2.6" fill="#111"/>')
        out.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-family="Arial" font-size="16" font-weight="bold" '
                   f'text-anchor="middle">{EDGE_STYLE[t]["name"]}</text>')

    # kóty: najdlhšie hrany (do 6) s dĺžkou
    dim_edges = []
    for p in planes:
        for e in p.get("edges", []):
            dim_edges.append((float(e.get("length_m") or 0), e))
    dim_edges.sort(key=lambda kv: -kv[0])
    for L, e in dim_edges[:6]:
        if L < 3.0:
            continue
        a = iso(*e["start"]); b = iso(*e["end"])
        x1, y1 = px(*a); x2, y2 = px(*b)
        vx, vy = x2 - x1, y2 - y1
        n = math.hypot(vx, vy) or 1.0
        ox, oy = -vy / n * 22, vx / n * 22
        out.append(f'<line x1="{x1+ox:.1f}" y1="{y1+oy:.1f}" x2="{x2+ox:.1f}" y2="{y2+oy:.1f}" '
                   f'stroke="#111" stroke-width="0.9" marker-start="url(#a)" marker-end="url(#a)"/>')
        out.append(f'<text x="{(x1+x2)/2+ox*1.9:.1f}" y="{(y1+y2)/2+oy*1.9:.1f}" font-family="Arial" '
                   f'font-size="13" text-anchor="middle">{L:.2f} m</text>')

    # sklon najstrmšej roviny
    steep = max(planes, key=lambda p: float(p.get("pitch_deg") or 0))
    if steep.get("vertices"):
        cx = sum(v[0] for v in steep["vertices"]) / len(steep["vertices"])
        cy = sum(v[1] for v in steep["vertices"]) / len(steep["vertices"])
        cz = sum(v[2] for v in steep["vertices"]) / len(steep["vertices"])
        u, v = iso(cx, cy, cz)
        X, Y = px(u, v)
        out.append(f'<text x="{X:.1f}" y="{Y:.1f}" font-family="Arial" font-size="15" text-anchor="middle" '
                   f'fill="#111">{float(steep["pitch_deg"]):.0f}°</text>')

    # severka + mierka
    out.append('<g transform="translate(1300,790)">'
               '<line x1="0" y1="0" x2="0" y2="-46" stroke="#111" stroke-width="1.6"/>'
               '<polygon points="0,-56 -6,-40 6,-40" fill="#111"/>'
               '<text x="0" y="18" font-family="Arial" font-size="13" text-anchor="middle">S</text></g>')
    out.append('<g transform="translate(60,830)">'
               '<line x1="0" y1="0" x2="140" y2="0" stroke="#111" stroke-width="1.2"/>'
               '<line x1="0" y1="-5" x2="0" y2="5" stroke="#111" stroke-width="1.2"/>'
               '<line x1="140" y1="-5" x2="140" y2="5" stroke="#111" stroke-width="1.2"/>'
               '<text x="70" y="-8" font-family="Arial" font-size="12" text-anchor="middle">10 m</text></g>')
    out.append('</svg>')

    svg_path = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem.replace("_roofmodel_v3", "") + "_izometria.svg")
    svg_path.write_text("\n".join(out), encoding="utf-8")
    print("SVG:", svg_path)

    # PNG cez headless prehliadač
    png_path = svg_path.with_suffix(".png")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch(headless=True)
            page = b.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1.5)
            page.goto(svg_path.resolve().as_uri(), wait_until="load")
            page.wait_for_timeout(400)
            page.screenshot(path=str(png_path))
            b.close()
        print("PNG:", png_path)
    except Exception as e:
        print("PNG zlyhalo:", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
