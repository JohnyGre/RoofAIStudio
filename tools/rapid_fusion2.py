#!/usr/bin/env python3
"""Optimized rapid fusion: center crop + LiDAR fusion in one pass."""
import time, numpy as np, glob, os, json, math
import cv2
from ultralytics import YOLO
from pyproj import Transformer
from shapely.geometry import Polygon, Point, box
import laspy
from scipy.spatial import ConvexHull

MODEL_PATH = r'C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio\ai_models\roof_gmaps_v2_last.pt'
ORTHO = '.cluster/task/DELIVERY/kopernika_7004_38_ortofoto.jpg'
LAZ_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'laz')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')

lat, lon = 48.3934036, 17.5892253
IMGSZ = 640

t_total = time.time()

# --- Load model ---
print('Loading model...')
model = YOLO(MODEL_PATH)
print(f'Model: {model.names}')

# --- Load ortho, crop center ---
img = cv2.imread(ORTHO)
h, w = img.shape[:2]
ch, cw = 2048, 2048
cx, cy = w//2, h//2
crop = img[cy-ch//2:cy+ch//2, cx-cw//2:cx+cw//2]

# --- Tile and detect ---
t1 = time.time()
detections = []
for y in range(0, ch, IMGSZ):
    for x in range(0, cw, IMGSZ):
        tile = crop[y:y+IMGSZ, x:x+IMGSZ]
        if tile.shape[0] < 100 or tile.shape[1] < 100:
            continue
        results = model.predict(tile, imgsz=IMGSZ, conf=0.2, verbose=False)
        for r in results:
            if r.boxes is None: continue
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_name = model.names[int(box.cls.item())]
                gx1 = (cx-ch//2) + x + x1
                gy1 = (cy-ch//2) + y + y1
                gx2 = (cx-ch//2) + x + x2
                gy2 = (cy-ch//2) + y + y2
                detections.append({
                    'cls': cls_name, 'conf': box.conf.item(),
                    'bbox': [gx1, gy1, gx2, gy2],
                })

print(f'Detection: {len(detections)} raw in {time.time()-t1:.1f}s')

if not detections:
    print('No detections found!')
    exit()

# --- NMS merge ---
boxes = np.array([d['bbox'] for d in detections])
scores = np.array([d['conf'] for d in detections])
order = scores.argsort()[::-1]
keep = []
suppressed = set()
for i in order:
    if i in suppressed: continue
    keep.append(i)
    for j in range(len(order)):
        if j in suppressed: continue
        xx1 = max(boxes[i][0], boxes[j][0]); yy1 = max(boxes[i][1], boxes[j][1])
        xx2 = min(boxes[i][2], boxes[j][2]); yy2 = min(boxes[i][3], boxes[j][3])
        wi = max(0, xx2-xx1); hi = max(0, yy2-yy1)
        inter = wi*hi
        ai = (boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1])
        aj = (boxes[j][2]-boxes[j][0])*(boxes[j][3]-boxes[j][1])
        iou = inter / (ai + aj - inter + 1e-6)
        if iou > 0.3: suppressed.add(j)

merged = [detections[i] for i in keep]
print(f'NMS: {len(merged)} buildings')

# --- Georeference ---
t_wgs_3857 = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
t_3857_8353 = Transformer.from_crs('EPSG:3857', 'EPSG:8353', always_xy=True)

d = 120 / 111000
lat_min, lat_max = lat-d, lat+d
lon_min, lon_max = lon-d, lon+d
x_min, y_min = t_wgs_3857.transform(lon_min, lat_min)
x_max, y_max = t_wgs_3857.transform(lon_max, lat_max)

# Convert detections to EPSG:8353 polygons
buildings = []
for det in merged:
    x1, y1, x2, y2 = det['bbox']
    corners = []
    for px, py in [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]:
        mx = x_min + (px/w)*(x_max-x_min)
        my = y_max - (py/h)*(y_max-y_min)
        east, north = t_3857_8353.transform(mx, my)
        corners.append((east, north))
    poly = Polygon(corners)
    if poly.area < 20: continue
    buildings.append({'cls': det['cls'], 'conf': det['conf'], 'poly': poly})

print(f'Georeferenced: {len(buildings)} with area > 20m2')

# --- LiDAR Fusion ---
print('\nFusing with LiDAR...')
results = []

for b in buildings[:10]:  # top 10
    poly = b['poly']
    minx, miny, maxx, maxy = poly.bounds
    
    bpts = []; gpts = []
    for f in glob.glob(os.path.join(LAZ_DIR, '*.laz')):
        if '.copc.' in f: continue
        try:
            las = laspy.read(f)
            xs = np.array(las.x); ys = np.array(las.y); zs = np.array(las.z)
            cls = np.array(las.classification, dtype=np.uint8)
            bbox_f = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
            for i in np.where(bbox_f)[0]:
                if poly.contains(Point(float(xs[i]), float(ys[i]))):
                    if cls[i] == 6: bpts.append([xs[i], ys[i], zs[i]])
                    elif cls[i] == 2: gpts.append([xs[i], ys[i], zs[i]])
        except: pass
    
    if len(bpts) < 20:
        print(f'  {b["cls"]}: not enough LiDAR points ({len(bpts)})')
        continue
    
    bp = np.array(bpts)
    gp = np.array(gpts) if gpts else bp
    gz = float(np.median(gp[:, 2])) if len(gp) > 5 else float(np.min(bp[:, 2]) - 1)
    
    bp_r = bp.copy(); bp_r[:, 2] -= gz
    zmax = float(np.max(bp_r[:, 2])); zmin = float(np.min(bp_r[:, 2]))
    
    eave_m = (bp_r[:, 2] > 0.5) & (bp_r[:, 2] < zmax - 1.5)
    eave_z = float(np.median(bp_r[eave_m, 2])) if np.sum(eave_m) > 5 else zmin
    
    cv_a = poly.area
    try:
        hull = ConvexHull(bp_r[:, :2]); lidar_a = hull.volume
    except: lidar_a = cv_a
    
    centered = bp_r[:, :2] - bp_r[:, :2].mean(axis=0)
    _, eigvecs = np.linalg.eigh(np.cov(centered.T))
    proj = centered @ eigvecs[:, 1]
    hs = (proj.max() - proj.min()) / 2
    pitch = math.degrees(math.atan((zmax - eave_z) / hs)) if hs > 0.5 else 0
    
    rtype = 'flat' if (zmax-zmin)<2 else 'low_pitch' if (zmax-zmin)<4 else 'pitched'
    
    res = {
        'class': b['cls'], 'confidence': round(b['conf'], 3),
        'area_cv_m2': round(cv_a, 1), 'area_lidar_m2': round(lidar_a, 1),
        'area_final_m2': round(cv_a, 1),  # CV edges for XY
        'height_ridge_m': round(zmax, 2), 'height_eave_m': round(eave_z, 2),
        'pitch_deg': round(pitch, 1), 'roof_type': rtype,
        'ground_mnm': round(gz, 2), 'lidar_points': len(bp),
    }
    results.append(res)
    
    print(f'  {b["cls"]} conf={b["conf"]:.2f} area={cv_a:.0f}m2 h={zmax:.1f}m pitch={pitch:.1f}deg {rtype} ({len(bp)} pts)')

results.sort(key=lambda r: -r['area_final_m2'])

# --- Report ---
print(f'\n{"="*60}')
print(f'FUSION RESULTS ({len(results)} buildings) | {time.time()-t_total:.1f}s total')
print(f'{"="*60}')

for i, r in enumerate(results[:5]):
    print(f'\n#{i+1} {r["class"]} (conf={r["confidence"]:.2f})')
    print(f'  Area:     {r["area_final_m2"]:.1f} m2 (CV: {r["area_cv_m2"]:.1f}, LiDAR: {r["area_lidar_m2"]:.1f})')
    print(f'  Ridge:    {r["height_ridge_m"]:.2f} m above ground')
    print(f'  Eave:     {r["height_eave_m"]:.2f} m')
    print(f'  Pitch:    {r["pitch_deg"]:.1f} deg | Type: {r["roof_type"]}')
    print(f'  Ground:   {r["ground_mnm"]:.2f} m n.m.')
    print(f'  Points:   {r["lidar_points"]}')

if results:
    json_path = os.path.join(OUT_DIR, 'rapid_fusion_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nJSON: {json_path}')
