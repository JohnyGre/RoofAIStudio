#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3_export — DXF (pre projektanta) a HTML report (na prezeranie) z v3 kontraktu.

Spustenie:
    .venv\\Scripts\\python.exe tools\\v3_export.py output\\v3\\X_roofmodel_v3.json
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LAYERS = {"o": "ODKVAP", "h": "HREBEN", "n": "NAROZIE", "u": "UZLABIE", "s": "STIT"}


def write_dxf(model: dict, out: Path) -> Path:
    """Minimalny DXF R12: hrany podla typov + obrysy rovin."""
    L = []

    def line(x1, y1, z1, x2, y2, z2, layer):
        L.extend(["0", "LINE", "8", layer,
                  "10", f"{x1:.3f}", "20", f"{y1:.3f}", "30", f"{z1:.3f}",
                  "11", f"{x2:.3f}", "21", f"{y2:.3f}", "31", f"{z2:.3f}"])

    L += ["0", "SECTION", "2", "ENTITIES"]
    for pl in model.get("planes", []):
        v = pl.get("vertices") or []
        if len(v) >= 3:
            for i in range(len(v)):
                a, b = v[i], v[(i + 1) % len(v)]
                line(a[0], a[1], a[2], b[0], b[1], b[2], "ROVINY")
        for e in pl.get("edges", []):
            layer = LAYERS.get(e.get("type", "s"), "STIT")
            if not (e.get("exact") or e.get("type") == "o"):
                continue
            s, t = e["start"], e["end"]
            line(s[0], s[1], s[2], t[0], t[1], t[2], layer)
    L += ["0", "ENDSEC", "0", "EOF"]
    out.write_text("\n".join(L) + "\n", encoding="ascii", errors="ignore")
    return out


def write_html(model: dict, out: Path, images: list | None = None) -> Path:
    rows = ""
    for i, pl in enumerate(model.get("planes", []), 1):
        types = {}
        for e in pl.get("edges", []):
            types[e["type"]] = types.get(e["type"], 0) + 1
        rows += (f"<tr><td>R{i}</td><td>{pl.get('type','')}</td>"
                 f"<td>{pl.get('pitch_deg',0):.1f}°</td>"
                 f"<td>{pl.get('area_m2',0):.1f} m²</td>"
                 f"<td>{', '.join(f'{k}={v}' for k, v in sorted(types.items())) or '—'}</td></tr>")
    imgs = ""
    for p in images or []:
        fp = Path(p)
        if fp.exists() and fp.suffix.lower() in (".png", ".jpg", ".jpeg"):
            raw = fp.read_bytes()
            try:                                   # veľké obrázky zmenšiť (ľahký report)
                import io as _io

                from PIL import Image as _Im
                im = _Im.open(_io.BytesIO(raw)).convert("RGB")
                if max(im.size) > 1800:
                    im.thumbnail((1800, 1800), _Im.LANCZOS)
                    buf = _io.BytesIO()
                    im.save(buf, "JPEG", quality=88)
                    raw = buf.getvalue()
                    mime = "jpeg"
                else:
                    mime = "png" if fp.suffix.lower() == ".png" else "jpeg"
            except Exception:
                mime = "png" if fp.suffix.lower() == ".png" else "jpeg"
            b64 = base64.b64encode(raw).decode()
            imgs += (f'<figure><img src="data:image/{mime};base64,{b64}" alt="{fp.name}"/>'
                     f'<figcaption>{fp.name}</figcaption></figure>')
    html = f"""<!DOCTYPE html><html lang="sk"><head><meta charset="utf-8">
<title>RoofAIStudio v3 — {model.get('address','')}</title>
<style>
 body{{font-family:Segoe UI,system-ui,sans-serif;background:#0d1b2a;color:#e0e1dd;margin:0;padding:24px}}
 h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#a0a8b4;font-size:13px;margin-bottom:18px}}
 table{{border-collapse:collapse;width:100%;margin:14px 0 24px}}
 th,td{{border-bottom:1px solid #2d3e50;padding:8px 10px;font-size:13px;text-align:left}}
 th{{color:#a0a8b4;text-transform:uppercase;font-size:11px;letter-spacing:.5px}}
 figure{{margin:0 0 22px}} img{{max-width:100%;border:1px solid #2d3e50;border-radius:8px}}
 figcaption{{color:#778da9;font-size:12px;margin-top:6px}}
 .meta{{display:flex;gap:26px;flex-wrap:wrap;color:#778da9;font-size:13px;margin-bottom:10px}}
 .meta b{{color:#e0e1dd}}
</style></head><body>
<h1>RoofAIStudio v3 — {model.get('address','')}</h1>
<div class="sub">kontrakt {model.get('schema_version','')} · {model.get('crs','')} · vytvorené {model.get('created','')}</div>
<div class="meta">
 <span>Rovín: <b>{len(model.get('planes',[]))}</b></span>
 <span>Plocha strechy: <b>{model.get('roof_area_m2','—')} m²</b></span>
 <span>Pôdorys: <b>{model.get('footprint_m2','—')} m²</b></span>
 <span>Zdroje: <b>{len(model.get('sources',[]))}</b></span>
</div>
<table><thead><tr><th>#</th><th>Typ</th><th>Sklon</th><th>Plocha</th><th>Hrany</th></tr></thead>
<tbody>{rows}</tbody></table>
{imgs}
</body></html>"""
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("Použitie: python tools/v3_export.py <roofmodel_v3.json>")
        return 2
    src = Path(sys.argv[1])
    model = json.loads(src.read_text(encoding="utf-8-sig"))
    base = src.with_name(src.stem.replace("_roofmodel_v3", ""))

    dxf = write_dxf(model, Path(str(base) + "_hrany.dxf"))
    def pick(*cands):
        for c in cands:
            if Path(c).exists():
                return c
        return None

    imgs = [x for x in (
        pick(str(base) + "_izometria.png"),
        pick(str(base) + "_kontrola_final_preview.jpg", str(base) + "_kontrola_hran_na_ortofote.png"),
        pick(str(base) + "_topdown.png"),
        pick(str(base) + "_ortofoto_max.jpg"),
    ) if x]
    html = write_html(model, Path(str(base) + "_report.html"), imgs)
    print("DXF:", dxf)
    print("HTML report:", html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
