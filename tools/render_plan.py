#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""render_plan — pohľad zhora v štýle rekonštrukcie z mračna:
farebné roviny (id, sklon, plocha), hrany s dĺžkami, legenda, OBVOD budovy.

Spustenie:
    .venv\\Scripts\\python.exe tools\\render_plan.py output\\v3\\X_roofmodel_v3.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

EDGE_STYLE = {
    "o": ("odkvap", "#e8a33d", 2.6),
    "h": ("hrebeň", "#2f6fd0", 3.0),
    "n": ("nárožie", "#2f9e6f", 2.6),
    "u": ("úžľabie", "#d33b3b", 2.6),
    "s": ("štít/voľná hrana", "#9b7fd4", 2.0),
}

PASTEL = ["#f6c9a0", "#f8b7b7", "#bcd7f0", "#d9c7ef", "#c9e7c9", "#f3e6a8",
          "#e8e8e8", "#f7d3ea", "#d4f0ea", "#f0dcc0", "#dbe7c9", "#e0d5f5"]


def polygon_area(v):
    s = 0.0
    n = len(v)
    for i in range(n):
        x1, y1 = v[i][0], v[i][1]
        x2, y2 = v[(i + 1) % n][0], v[(i + 1) % n][1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def main() -> int:
    if len(sys.argv) < 2:
        print("Použitie: python tools/render_plan.py <roofmodel_v3.json>")
        return 2
    src = Path(sys.argv[1])
    model = json.loads(src.read_text(encoding="utf-8-sig"))
    planes = [p for p in model.get("planes", []) if len(p.get("vertices") or []) >= 3]
    if not planes:
        print("model nemá roviny")
        return 2

    # obvod budovy = vonkajší obrys zjednotenia plôch
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = [Polygon([(v[0], v[1]) for v in p["vertices"]]) for p in planes]
    u = unary_union([p for p in polys if p.is_valid and p.area > 0])
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda g: g.area)
    ring = list(u.simplify(0.35).exterior.coords)

    fig, ax = plt.subplots(figsize=(11, 11), dpi=120)
    fig.patch.set_facecolor("white")

    # roviny
    for i, p in enumerate(planes):
        v = p["vertices"]
        xs = [t[0] for t in v] + [v[0][0]]
        ys = [t[1] for t in v] + [v[0][1]]
        col = PASTEL[i % len(PASTEL)]
        ax.fill(xs, ys, color=col, alpha=0.75, zorder=1)
        ax.plot(xs, ys, color="#999999", lw=0.8, zorder=2)
        cx = sum(t[0] for t in v) / len(v)
        cy = sum(t[1] for t in v) / len(v)
        ax.text(cx, cy, f"{p['id']}\n{p['pitch_deg']:.1f}°\n{p['area_m2']:.1f} m²",
                ha="center", va="center", fontsize=8.5, color="#222", zorder=6)

    # hrany + dĺžky
    for p in planes:
        for e in p.get("edges", []):
            t = e.get("type", "s")
            if not (e.get("exact") or t in ("o", "h", "n", "u")):
                # voľné "s" hrany kreslíme len ak sú dlhšie (inak je výkres preplnený)
                if e.get("length_m", 0) < 2.0:
                    continue
            name, col, lw = EDGE_STYLE.get(t, EDGE_STYLE["s"])
            x1, y1, x2, y2 = e["start"][0], e["start"][1], e["end"][0], e["end"][1]
            ax.plot([x1, x2], [y1, y2], color=col, lw=lw, solid_capstyle="round", zorder=4)
            if e.get("length_m", 0) >= 2.0 and t != "s":
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if ang > 90:
                    ang -= 180
                if ang < -90:
                    ang += 180
                ax.text(mx, my, f"{e['length_m']:.2f}", fontsize=6.8, color="#333",
                        rotation=ang, rotation_mode="anchor", ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75), zorder=7)

    # OBVOD budovy
    if ring:
        rx = [c[0] for c in ring]
        ry = [c[1] for c in ring]
        ax.plot(rx, ry, color="#111111", lw=2.0, ls="--", zorder=5, label="obvod budovy")

    # legenda
    handles = [Patch(facecolor=c, edgecolor="none", label=n) for n, (n2, c, _) in EDGE_STYLE.items()]
    handles.append(plt.Line2D([0], [0], color="#111111", lw=2, ls="--", label="obvod budovy"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.95)

    ax.set_aspect("equal")
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"{model.get('address', '')} — rekonštrukcia z mračna: roviny, hrany (dĺžky 3D, m), obvod budovy",
                 fontsize=11)
    ax.tick_params(labelsize=8)

    out_png = src.with_name(src.stem.replace("_roofmodel_v3", "") + "_plan_rekonstrukcia.png")
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("PNG:", out_png)
    print(f"obvod: {sum(math.hypot(ring[i][0]-ring[i-1][0], ring[i][1]-ring[i-1][1]) for i in range(len(ring))):.2f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
