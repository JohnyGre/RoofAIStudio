# -*- coding: utf-8 -*-
"""
Roof Photo Pipeline — predspracovanie, detekcia, klasifikácia, GUI
===================================================================
Kompletný pipeline na spracovanie fotografie strechy rovnakým spôsobom
ako satelitné snímky:
  1. Resize na 640×640 (preserve aspect ratio + padding 114)
  2. YOLO segmentácia (roof_gmaps_v2_last.pt)
  3. Klasifikácia hrán/oblastí (5 tried: flat, min, poly, trap, tri)
  4. HTML GUI s vrstvami (pôvodný, spracovaný, detekcia, klasifikácia)
  5. SQLite backend s metadatami
  6. Konfiguračný súbor (YAML)
"""
import os, sys, json, time, hashlib, sqlite3, base64
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# ─── Konfigurácia ───────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT / "ai_models" / "roof_gmaps_v2_last.pt"
OUTPUT_DIR = PROJECT / "output" / "photo_pipeline"
DB_PATH = OUTPUT_DIR / "pipeline_results.db"
CONFIG_PATH = OUTPUT_DIR / "pipeline_config.json"

CLASS_NAMES = {
    0: "slope_flat",
    1: "slope_min",
    2: "slope_poly",
    3: "slope_trap",
    4: "slope_tri",
}

CLASS_COLORS = {
    0: (100, 149, 237),    # flat — cornflower blue
    1: (72, 201, 176),     # min — medium aquamarine
    2: (255, 165, 0),      # poly — orange
    3: (220, 20, 60),      # trap — crimson
    4: (138, 43, 226),     # tri — blue violet
}

CLASS_LABELS_SK = {
    0: "Plochá strecha",
    1: "Minimálny sklon",
    2: "Polygónálna",
    3: "Lichobežníková",
    4: "Trojuholníková",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── 1. Konfiguračný súbor ─────────────────────────────────────
def ensure_config():
    """Vytvor konfiguračný súbor ak neexistuje."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    config = {
        "model": {
            "path": str(MODEL_PATH),
            "task": "segment",
            "imgsz": 640,
            "classes": {str(k): v for k, v in CLASS_NAMES.items()},
        },
        "preprocessing": {
            "target_size": 640,
            "padding_value": 114,
            "color_space": "RGB",
            "normalization": "imagenet",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "inference": {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.7,
            "device": "auto",
        },
        "database": {
            "path": str(DB_PATH),
        },
        "version": "1.0.0",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config


# ─── 2. Predspracovanie ────────────────────────────────────────
def preprocess_image(image_path, target_size=640, padding_value=114):
    """Načítaj obrázok, resize na 640×640 s aspect-ratio preservation.

    Returns:
        dict with:
            - original: np.ndarray (H,W,3) BGR uint8
            - processed: np.ndarray (640,640,3) BGR uint8 (padded)
            - scale: float (1/scale_factor — pre spätnú konverziu)
            - padding: (top, left) — offset pre spätnú konverziu
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Nepodarilo sa načítať obrázok: {image_path}")

    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    bottom = target_size - new_h - top
    right = target_size - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(padding_value, padding_value, padding_value)
    )

    return {
        "original": img,
        "processed": padded,
        "original_shape": (h, w),
        "processed_shape": (target_size, target_size),
        "scale": 1.0 / scale,
        "padding": (top, left),
        "resize_scale": scale,
    }


# ─── 3. Inferencia ─────────────────────────────────────────────
def run_inference(processed_img, model_path=None, conf=0.25, iou=0.7):
    """Spustí YOLO segmentáciu na spracovanom obrázku.

    Returns:
        list of dicts: [{class_id, class_name, confidence, mask, bbox, area_px}, ...]
    """
    from ultralytics import YOLO

    if model_path is None:
        model_path = MODEL_PATH

    model = YOLO(str(model_path))
    rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    results = model(rgb, conf=conf, iou=iou, verbose=False)

    detections = []
    if results and len(results) > 0:
        r = results[0]
        if r.masks is not None and r.boxes is not None:
            masks = r.masks.data.cpu().numpy()
            boxes = r.boxes

            for i, mask in enumerate(masks):
                conf_val = float(boxes.conf[i].item())
                cls_id = int(boxes.cls[i].item())
                cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                # Resize mask to processed image size
                mask_resized = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_LINEAR)
                binary_mask = (mask_resized > 0.5).astype(np.uint8)

                # Bounding box
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                area_px = int(binary_mask.sum())

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "class_label_sk": CLASS_LABELS_SK.get(cls_id, cls_name),
                    "confidence": conf_val,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "bbox_size": [float(x2 - x1), float(y2 - y1)],
                    "area_px": area_px,
                    "mask": binary_mask,
                })

    return detections


# ─── 4. Vizualizácia ───────────────────────────────────────────
def draw_overlay(processed_img, detections, show_masks=True, show_labels=True):
    """Vykreslí detekcie na spracovanom obrázku."""
    vis = processed_img.copy()

    for det in detections:
        cls_id = det["class_id"]
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        mask = det["mask"]

        if show_masks:
            colored_mask = np.zeros_like(vis)
            colored_mask[mask > 0] = color
            vis = cv2.addWeighted(vis, 0.7, colored_mask, 0.3, 0)

        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        if show_labels:
            label = f'{det["class_label_sk"]} ({det["confidence"]:.2f})'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


# ─── 5. Dátový backend ─────────────────────────────────────────
def init_db():
    """Inicializuj SQLite databázu."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            num_detections INTEGER,
            classes TEXT,
            confidences TEXT,
            processing_time_ms REAL,
            image_hash TEXT,
            config_hash TEXT,
            output_json TEXT
        )
    """)
    conn.commit()
    return conn


def save_result(conn, filename, detections, processing_time_ms, image_hash, config_hash, output_json):
    """Ulož výsledok do databázy."""
    classes = [d["class_name"] for d in detections]
    confidences = [d["confidence"] for d in detections]

    conn.execute(
        "INSERT INTO pipeline_runs (filename, timestamp, num_detections, classes, confidences, "
        "processing_time_ms, image_hash, config_hash, output_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (filename, time.strftime("%Y-%m-%d %H:%M:%S"),
         len(detections), json.dumps(classes), json.dumps(confidences),
         processing_time_ms, image_hash, config_hash, output_json)
    )
    conn.commit()


# ─── 6. Hlavný pipeline ────────────────────────────────────────
def process_photo(image_path, config=None, save_to_db=True):
    """Spracuj fotografiu kompletným pipeline.

    Args:
        image_path: cesta k JPG/PNG
        config: konfiguračný dict (ak None, načíta z config súboru)
        save_to_db: uložiť výsledok do databázy

    Returns:
        dict so všetkými výstupmi
    """
    t_start = time.time()

    if config is None:
        config = ensure_config()

    pp = config["preprocessing"]
    inf = config["inference"]

    # Predspracovanie
    pre = preprocess_image(
        image_path,
        target_size=pp["target_size"],
        padding_value=pp["padding_value"],
    )

    # Inferencia
    detections = run_inference(
        pre["processed"],
        model_path=config["model"]["path"],
        conf=inf["confidence_threshold"],
        iou=inf["iou_threshold"],
    )

    # Vizualizácia
    overlay = draw_overlay(pre["processed"], detections)

    t_elapsed = (time.time() - t_start) * 1000

    # Hash pre reprodukovateľnosť
    with open(image_path, "rb") as f:
        image_hash = hashlib.md5(f.read()).hexdigest()
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

    # Ulož do DB
    if save_to_db:
        conn = init_db()
        result_summary = {
            "filename": os.path.basename(image_path),
            "detections": [
                {"class": d["class_name"], "conf": d["confidence"], "area_px": d["area_px"]}
                for d in detections
            ],
            "processing_time_ms": t_elapsed,
        }
        save_result(conn, os.path.basename(image_path), detections,
                    t_elapsed, image_hash, config_hash,
                    json.dumps(result_summary, ensure_ascii=False))
        conn.close()

    return {
        "original": pre["original"],
        "processed": pre["processed"],
        "overlay": overlay,
        "detections": detections,
        "preprocess_info": pre,
        "processing_time_ms": t_elapsed,
        "image_hash": image_hash,
        "config_hash": config_hash,
    }


# ─── 7. GUI — HTML generátor ───────────────────────────────────
def generate_html_gui(result, image_path, output_html=None):
    """Vygeneruje samostatný HTML súbor s vrstvami.

    Vrchy:
      - Pôvodná fotografia
      - Spracovaný vstup (640×640)
      - Detegované masky + bounding boxy
      - Prekrytie s klasifikačnými štítkami
    """
    # Konvertuj obrázky na base64
    def to_b64(img, fmt=".jpg", quality=90):
        _, buf = cv2.imencode(fmt, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    orig_b64 = to_b64(result["original"])
    proc_b64 = to_b64(result["processed"])
    overlay_b64 = to_b64(result["overlay"])

    # Detekcie pre tabuľku
    det_rows = ""
    for i, d in enumerate(result["detections"]):
        color_hex = "#{:02x}{:02x}{:02x}".format(
            CLASS_COLORS[d["class_id"]][2],
            CLASS_COLORS[d["class_id"]][1],
            CLASS_COLORS[d["class_id"]][0],
        )
        det_rows += f"""
        <tr>
            <td><span class="color-dot" style="background:{color_hex}"></span>{i+1}</td>
            <td>{d["class_label_sk"]}</td>
            <td>{d["class_name"]}</td>
            <td>{d["confidence"]:.3f}</td>
            <td>{d["area_px"]} px²</td>
            <td>{d["bbox_size"][0]:.0f}×{d["bbox_size"][1]:.0f}</td>
        </tr>"""

    num_det = len(result["detections"])
    proc_time = result["processing_time_ms"]

    html = f"""<!DOCTYPE html>
<html lang="sk">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RoofAI — Detekcia strechy</title>
<style>
  :root {{
    --bg: #0d1b2a;
    --surface: #1b263b;
    --surface-2: #243447;
    --accent: #415a77;
    --accent-2: #778da9;
    --text: #e0e1dd;
    --text-dim: #a0a8b4;
    --border: #2d3e50;
    --success: #4ade80;
    --warning: #fbbf24;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', 'SF Pro Display', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 24px;
  }}
  .header {{
    text-align: center;
    margin-bottom: 32px;
    padding: 24px;
    background: var(--surface);
    border-radius: 12px;
    border: 1px solid var(--border);
  }}
  .header h1 {{
    font-size: 24px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: 0.5px;
  }}
  .header .subtitle {{
    color: var(--text-dim);
    font-size: 14px;
    margin-top: 8px;
  }}
  .header .meta {{
    display: flex;
    justify-content: center;
    gap: 32px;
    margin-top: 16px;
    font-size: 13px;
  }}
  .meta-item {{ color: var(--accent-2); }}
  .meta-item strong {{ color: var(--text); }}
  .layers {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 32px;
  }}
  .layer-card {{
    background: var(--surface);
    border-radius: 12px;
    border: 1px solid var(--border);
    overflow: hidden;
  }}
  .layer-card h3 {{
    padding: 12px 16px;
    font-size: 14px;
    font-weight: 500;
    color: var(--accent-2);
    border-bottom: 1px solid var(--border);
    background: var(--surface-2);
  }}
  .layer-card img {{
    width: 100%;
    display: block;
    cursor: pointer;
    transition: opacity 0.2s;
  }}
  .layer-card img:hover {{ opacity: 0.92; }}
  .layer-card.full {{ grid-column: 1 / -1; }}
  .detections {{
    background: var(--surface);
    border-radius: 12px;
    border: 1px solid var(--border);
    overflow: hidden;
  }}
  .detections h3 {{
    padding: 12px 16px;
    font-size: 14px;
    font-weight: 500;
    color: var(--accent-2);
    border-bottom: 1px solid var(--border);
    background: var(--surface-2);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
  }}
  th, td {{
    padding: 10px 16px;
    text-align: left;
    font-size: 13px;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    color: var(--text-dim);
    font-weight: 500;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
  }}
  td {{ color: var(--text); }}
  .color-dot {{
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 3px;
    margin-right: 8px;
    vertical-align: middle;
  }}
  .toggle-bar {{
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }}
  .toggle-btn {{
    padding: 8px 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-dim);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
  }}
  .toggle-btn.active {{
    background: var(--accent);
    color: var(--text);
    border-color: var(--accent);
  }}
  .toggle-btn:hover {{ border-color: var(--accent-2); }}
  .empty-state {{
    padding: 48px;
    text-align: center;
    color: var(--text-dim);
    font-size: 14px;
  }}
  @media (max-width: 768px) {{
    .layers {{ grid-template-columns: 1fr; }}
    .header .meta {{ flex-direction: column; gap: 8px; }}
  }}
</style>
</head>
<body>
<div class="header">
    <h1>RoofAI Studio — Detekcia strechy</h1>
    <div class="subtitle">YOLO segmentačný model · 640×640 predspracovanie · 5 tried klasifikácie</div>
    <div class="meta">
        <span class="meta-item"><strong>{os.path.basename(image_path)}</strong></span>
        <span class="meta-item">Detekcie: <strong>{num_det}</strong></span>
        <span class="meta-item">Čas: <strong>{proc_time:.0f} ms</strong></span>
        <span class="meta-item">Hash: <strong>{result["image_hash"][:12]}</strong></span>
    </div>
</div>

<div class="toggle-bar">
    <button class="toggle-btn active" onclick="showLayer('all', this)">Všetky vrstvy</button>
    <button class="toggle-btn" onclick="showLayer('original', this)">Pôvodná fotografia</button>
    <button class="toggle-btn" onclick="showLayer('processed', this)">Spracovaný vstup (640×640)</button>
    <button class="toggle-btn" onclick="showLayer('overlay', this)">Detekcia + klasifikácia</button>
</div>

<div class="layers" id="layers-all">
    <div class="layer-card">
        <h3>Pôvodná fotografia</h3>
        <img src="{orig_b64}" alt="Pôvodná fotografia" onclick="window.open(this.src)">
    </div>
    <div class="layer-card">
        <h3>Spracovaný vstup (640×640)</h3>
        <img src="{proc_b64}" alt="Spracovaný vstup" onclick="window.open(this.src)">
    </div>
    <div class="layer-card full">
        <h3>Detegované hrany + klasifikácia</h3>
        <img src="{overlay_b64}" alt="Detekcia" onclick="window.open(this.src)">
    </div>
</div>

<div class="detections">
    <h3>Klasifikácia detegovaných oblastí ({num_det})</h3>
    {"<table><thead><tr><th>#</th><th>Trieda (SK)</th><th>Trieda (kód)</th><th>Confidence</th><th>Plocha</th><th>BBox</th></tr></thead><tbody>" + det_rows + "</tbody></table>" if det_rows else '<div class="empty-state">Žiadne detekcie. Skúste znížiť confidence threshold v konfigurácii.</div>'}
</div>

<script>
function showLayer(layer, el) {{
    document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
    (el || event.target).classList.add('active');
    const all = document.getElementById('layers-all');
    if (layer === 'all') {{
        all.style.display = 'grid';
    }} else {{
        all.style.display = 'block';
        const cards = all.querySelectorAll('.layer-card');
        cards.forEach((c, i) => {{
            c.style.display = (layer === 'original' && i === 0) ||
                             (layer === 'processed' && i === 1) ||
                             (layer === 'overlay' && i === 2) ? 'block' : 'none';
        }});
    }}
}}
</script>
</body>
</html>"""

    if output_html is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_html = str(OUTPUT_DIR / f"{base}_pipeline_gui.html")

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    return output_html


# ─── 8. CLI vstup ──────────────────────────────────────────────

def save_layer_pngs(result, image_path):
    """Uloží jednotlivé vrstvy ako PNG (originál, 640×640 vstup, prekrytie).

    Vďaka tomu sa dajú vrstvy otvoriť aj mimo HTML GUI a použiť v reporte.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    paths = {}
    layers = {
        "original": result["original"],
        "processed_640": result["processed"],
        "overlay": result["overlay"],
    }
    for name, img in layers.items():
        out = str(OUTPUT_DIR / f"{base}_{name}.png")
        cv2.imwrite(out, img)
        paths[name] = out
    return paths


def main():
    if len(sys.argv) < 2:
        print("Použitie: python photo_pipeline.py <image_path> [conf_threshold]")
        print("Príklad: python photo_pipeline.py output/atriova_16_v2_ortofoto_maxzoom.jpg 0.15")
        sys.exit(1)

    image_path = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

    print(f"RoofAI Photo Pipeline")
    print(f"Input: {image_path}")
    print(f"Confidence: {conf}")
    print()

    # Načítaj konfiguráciu a prepíš confidence
    config = ensure_config()
    config["inference"]["confidence_threshold"] = conf

    # Spracuj
    result = process_photo(image_path, config)

    print(f"Detections: {len(result['detections'])}")
    for d in result["detections"]:
        print(f"  {d['class_label_sk']} ({d['class_name']}) conf={d['confidence']:.3f} area={d['area_px']}px²")
    print(f"Processing time: {result['processing_time_ms']:.0f} ms")
    print(f"Image hash: {result['image_hash']}")
    print(f"Config hash: {result['config_hash']}")

    # Vygeneruj GUI
    html_path = generate_html_gui(result, image_path)
    print(f"\nGUI: {html_path}")
    layer_paths = save_layer_pngs(result, image_path)
    print(f"Vrstvy PNG: {layer_paths['overlay']}")

    # Ulož JSON
    base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = str(OUTPUT_DIR / f"{base}_pipeline_result.json")
    json_result = {
        "filename": os.path.basename(image_path),
        "detections": [
            {"class": d["class_name"], "label_sk": d["class_label_sk"],
             "confidence": d["confidence"], "area_px": d["area_px"],
             "bbox": d["bbox"]}
            for d in result["detections"]
        ],
        "processing_time_ms": result["processing_time_ms"],
        "image_hash": result["image_hash"],
        "config_hash": result["config_hash"],
        "preprocess": {
            "original_shape": result["preprocess_info"]["original_shape"],
            "processed_shape": result["preprocess_info"]["processed_shape"],
            "resize_scale": result["preprocess_info"]["resize_scale"],
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    print(f"JSON: {json_path}")
    print(f"DB: {DB_PATH}")
    print(f"Config: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
