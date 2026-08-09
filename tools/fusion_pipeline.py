#!/usr/bin/env python3
"""
CV + LiDAR Fusion Module
=========================
Combines ZBGIS orthophoto detection (YOLO segmentation) with LAZ point cloud
for centimeter-accurate building measurements.

Pipeline:
  ortofoto (4096×4096) → YOLO 640×640 tiles → mask
  LAZ class=6 points → mask filter → precise 3D
  Fuse: CV edges × LiDAR Z → final metrics
"""
import os, sys, json, glob, math, ssl, urllib.request, urllib.parse
import numpy as np
import cv2
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import Delaunay, ConvexHull
from shapely.geometry import Polygon, mapping

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAZ_DIR = os.path.join(PROJECT_ROOT, 'data', 'laz')
OUT_DIR = os.path.join(PROJECT_ROOT, 'output')
DEFAULT_MODEL_PATH = 'yolov8n-seg.pt'  # or roof-specific model
YOLO_IMGSZ = 640
TILE_OVERLAP = 64  # pixels overlap between tiles

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. ORTHOPHOTO → CV MASKS
# ============================================================

class CVDetector:
    """Wraps YOLO segmentation for building/roof detection on orthophotos."""
    
    def __init__(self, model_path=DEFAULT_MODEL_PATH, imgsz=YOLO_IMGSZ, device='cpu'):
        self.model_path = model_path
        self.imgsz = imgsz
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f'[CV] Loaded model: {self.model_path} on {self.device}')
        except Exception as e:
            print(f'[CV] Model not available: {e}')
            print('[CV] Will use fallback: LiDAR-only segmentation')
            self.model = None
    
    def detect(self, image_path, conf=0.25, classes=None):
        """
        Run detection on orthophoto.
        Returns list of dicts: {mask, bbox, confidence, class_name}
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f'Cannot read: {image_path}')
        
        h, w = img.shape[:2]
        print(f'[CV] Image: {w}x{h} -> tiling {self.imgsz}x{self.imgsz}')
        
        if self.model is None:
            return self._fallback_detection(img)
        
        results = []
        tiles = self._tile_image(img)
        
        for tx, ty, tile in tiles:
            th, tw = tile.shape[:2]
            if th < 32 or tw < 32:
                continue
            
            # Run YOLO
            preds = self.model.predict(
                tile, imgsz=self.imgsz, conf=conf,
                classes=classes, verbose=False, device=self.device
            )
            
            for pred in preds:
                if pred.masks is None:
                    continue
                for i, mask_tensor in enumerate(pred.masks.data):
                    mask = mask_tensor.cpu().numpy()
                    mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_LINEAR)
                    mask = (mask > 0.5).astype(np.uint8)
                    
                    box = pred.boxes[i]
                    cls_id = int(box.cls.item())
                    cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else str(cls_id)
                    
                    # Place mask in full image coordinates
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    full_mask[ty:ty+th, tx:tx+tw] = mask * (full_mask[ty:ty+th, tx:tx+tw] == 0) + mask
                    
                    results.append({
                        'mask': full_mask,
                        'bbox': [tx + box.xyxy[0][0].item(), ty + box.xyxy[0][1].item(),
                                tx + box.xyxy[0][2].item(), ty + box.xyxy[0][3].item()],
                        'confidence': box.conf.item(),
                        'class': cls_name,
                    })
        
        print(f'[CV] Detected {len(results)} objects')
        return results
    
    def _tile_image(self, img):
        """Split large image into 640x640 tiles with overlap."""
        h, w = img.shape[:2]
        step = self.imgsz - TILE_OVERLAP
        tiles = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                tile = img[y:y+self.imgsz, x:x+self.imgsz]
                if tile.shape[0] > 0 and tile.shape[1] > 0:
                    tiles.append((x, y, tile))
        return tiles
    
    def _fallback_detection(self, img):
        """LiDAR-only fallback: cluster by color/texture."""
        print('[CV] Fallback: color-based building detection')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Building roofs typically: low saturation, mid-high value
        lower = np.array([0, 0, 40])
        upper = np.array([180, 80, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # min 500 px (~1.8 m2 at 6cm/px)
                continue
            cnt_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 1, -1)
            x, y, w, h = cv2.boundingRect(cnt)
            results.append({
                'mask': cnt_mask,
                'bbox': [x, y, x+w, y+h],
                'confidence': 0.5,
                'class': 'building_fallback',
            })
        
        print(f'[CV] Fallback: {len(results)} objects')
        return results

# ============================================================
# 2. GEO-REFERENCE MASKS
# ============================================================

def georeference_mask(mask, ortho_bbox, target_epsg=8353):
    """
    Convert pixel mask coordinates to EPSG:8353 (S-JTSK).
    
    ortho_bbox: (lat_min, lon_min, lat_max, lon_max) — area covered by orthophoto
    Returns: list of (easting, northing) polygon vertices
    """
    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Take largest contour
    cnt = max(contours, key=cv2.contourArea)
    
    # Simplify to polygon
    epsilon = 0.005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    h, w = mask.shape
    
    # Ortho bbox in EPSG:3857 (Web Mercator)
    t_wgs_to_3857 = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    t_3857_to_8353 = Transformer.from_crs('EPSG:3857', 'EPSG:8353', always_xy=True)
    
    lat_min, lon_min, lat_max, lon_max = ortho_bbox
    x_min, y_min = t_wgs_to_3857.transform(lon_min, lat_min)
    x_max, y_max = t_wgs_to_3857.transform(lon_max, lat_max)
    
    # Convert pixel coordinates to EPSG:3857, then to EPSG:8353
    points_8353 = []
    for pt in approx:
        px, py = pt[0]
        # Pixel to EPSG:3857
        ex = x_min + (px / w) * (x_max - x_min)
        ey = y_min + (1 - py / h) * (y_max - y_min)  # flip Y
        # EPSG:3857 to EPSG:8353
        east, north = t_3857_to_8353.transform(ex, ey)
        points_8353.append((east, north))
    
    return points_8353

# ============================================================
# 3. LiDAR + CV FUSION
# ============================================================

class LidarCVFusion:
    """Combines CV-detected building footprint with LiDAR point cloud."""
    
    def __init__(self):
        self.t_wgs_to_8353 = Transformer.from_crs('EPSG:4326', 'EPSG:8353', always_xy=True)
        self.t_8353_to_wgs = Transformer.from_crs('EPSG:8353', 'EPSG:4326', always_xy=True)
    
    def fuse(self, cv_mask_polygon_8353, laz_dir=LAZ_DIR):
        """
        Fuse CV footprint with LiDAR points for precise measurements.
        
        Returns dict with all fused metrics.
        """
        # Create shapely polygon from CV mask
        if len(cv_mask_polygon_8353) < 3:
            return None
        
        footprint = Polygon(cv_mask_polygon_8353)
        if not footprint.is_valid:
            footprint = footprint.buffer(0)
        
        # Load LiDAR building points
        bpoints, gpoints = self._load_lidar_points(laz_dir, footprint)
        
        if len(bpoints) < 10:
            print('[FUSION] Not enough LiDAR points in footprint')
            return None
        
        # Ground elevation
        ground_z = float(np.median(gpoints[:, 2])) if len(gpoints) > 10 else float(np.min(bpoints[:, 2]) - 1)
        
        # LiDAR (precise Z)
        bp = bpoints.copy()
        bp[:, 2] -= ground_z
        
        z_max = float(np.max(bp[:, 2]))
        z_min = float(np.min(bp[:, 2]))
        z_range = z_max - z_min
        
        # CV (precise XY) - area from polygon
        cv_area = footprint.area
        
        # LiDAR XY - convex hull area
        try:
            hull = ConvexHull(bp[:, :2])
            lidar_area = hull.volume  # 2D area
        except:
            lidar_area = cv_area
        
        # FUSED metrics
        # Area: use CV for edges (more precise XY), cross-check with LiDAR
        area_final = cv_area if cv_area > 20 else lidar_area
        
        # Height: LiDAR is authoritative for Z
        eave_mask = (bp[:, 2] > 0.5) & (bp[:, 2] < z_max - z_range * 0.5)
        eave_z = float(np.median(bp[eave_mask, 2])) if np.sum(eave_mask) > 5 else z_min
        
        # Pitch: LiDAR profile
        # Simplified: take Z range along longest axis
        centered = bp[:, :2] - bp[:, :2].mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        main_axis = eigvecs[:, 1]
        proj = centered @ main_axis
        half_span = (proj.max() - proj.min()) / 2
        
        pitch_rad = math.atan((z_max - eave_z) / half_span) if half_span > 0.5 else 0
        pitch_deg = math.degrees(pitch_rad)
        
        roof_type = 'flat' if z_range < 2.0 else 'low_pitch' if z_range < 4.0 else 'pitched'
        
        # Ridge line: max Z points along main axis
        ridge_mask = bp[:, 2] > (z_max - 0.3)
        ridge_pts = bp[ridge_mask]
        
        # CV gives ridge line orientation
        # LiDAR gives ridge line Z
        ridge_length = 0
        if len(ridge_pts) > 3:
            ridge_proj = ridge_pts[:, :2] @ main_axis
            ridge_length = ridge_proj.max() - ridge_proj.min()
        
        # Confidence scores
        cv_confidence = 0.8  # CV is good for XY
        lidar_confidence = 0.95  # LiDAR is great for Z
        
        result = {
            'area_m2': round(area_final, 2),
            'cv_area_m2': round(cv_area, 2),
            'lidar_area_m2': round(lidar_area, 2),
            
            'height_ridge_m': round(z_max, 2),
            'height_eave_m': round(eave_z, 2),
            'height_range_m': round(z_range, 2),
            
            'pitch_deg': round(pitch_deg, 1),
            'roof_type': roof_type,
            
            'ridge_length_m': round(ridge_length, 2),
            
            'ground_mnm': round(ground_z, 2),
            'lidar_points': len(bp),
            
            'footprint_geojson': mapping(footprint),
            
            'confidence': {
                'area': round(0.5 * cv_confidence + 0.5 * lidar_confidence, 2),
                'height': round(lidar_confidence, 2),
                'pitch': round(lidar_confidence, 2),
            }
        }
        
        # Create fused 3D model
        self._create_fused_mesh(bp, result)
        
        return result
    
    def _load_lidar_points(self, laz_dir, footprint):
        """Load LiDAR points inside the footprint polygon."""
        import laspy
        
        bpoints_all = []
        gpoints_all = []
        
        # Get polygon bounds for quick filtering
        minx, miny, maxx, maxy = footprint.bounds
        
        for f in glob.glob(os.path.join(laz_dir, '*.laz')):
            if '.copc.' in f: continue
            try:
                las = laspy.read(f)
                xs = np.array(las.x); ys = np.array(las.y); zs = np.array(las.z)
                cls = np.array(las.classification, dtype=np.uint8)
                
                # Quick bbox filter
                bbox_mask = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
                if not np.any(bbox_mask):
                    continue
                
                # Precise polygon filter for building points
                for i in np.where(bbox_mask)[0]:
                    from shapely.geometry import Point
                    if footprint.contains(Point(float(xs[i]), float(ys[i]))):
                        if cls[i] == 6:
                            bpoints_all.append([xs[i], ys[i], zs[i]])
                        elif cls[i] == 2:
                            gpoints_all.append([xs[i], ys[i], zs[i]])
            except Exception as e:
                pass
        
        return np.array(bpoints_all), np.array(gpoints_all)
    
    def _create_fused_mesh(self, bp, result):
        """Create 3D mesh from fused data."""
        try:
            tri = Delaunay(bp[:, :2])
            
            # PLY
            ply_path = os.path.join(OUT_DIR, 'fused_building.ply')
            with open(ply_path, 'w') as f:
                f.write(f'ply\nformat ascii 1.0\nelement vertex {len(bp)}\n')
                f.write('property float x\nproperty float y\nproperty float z\n')
                f.write(f'element face {len(tri.simplices)}\nproperty list uchar int vertex_indices\nend_header\n')
                for v in bp: f.write(f'{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n')
                for fc in tri.simplices: f.write(f'3 {fc[0]} {fc[1]} {fc[2]}\n')
            
            # OBJ
            obj_path = os.path.join(OUT_DIR, 'fused_building.obj')
            with open(obj_path, 'w') as f:
                f.write('# CV + LiDAR fused model\n')
                for v in bp: f.write(f'v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n')
                for fc in tri.simplices: f.write(f'f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n')
            
            result['files'] = {'ply': ply_path, 'obj': obj_path}
        except Exception as e:
            print(f'[MESH] Error: {e}')

# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def run_fusion_pipeline(orthophoto_path, ortho_bbox, model_path=None, conf=0.25):
    """
    Full CV + LiDAR fusion pipeline.
    
    Args:
        orthophoto_path: Path to ZBGIS orthophoto JPEG
        ortho_bbox: (lat_min, lon_min, lat_max, lon_max) of the orthophoto area
        model_path: Optional custom model path
        conf: Detection confidence threshold
    
    Returns:
        dict with all fused metrics and file paths
    """
    print('='*60)
    print('CV + LiDAR FUSION PIPELINE')
    print('='*60)
    
    # Step 1: CV detection
    print('\n[1/4] CV Detection (orthophoto)...')
    detector = CVDetector(model_path or DEFAULT_MODEL_PATH)
    detections = detector.detect(orthophoto_path, conf=conf)
    
    if not detections:
        print('[ERROR] No buildings detected')
        return None
    
    # Step 2: Select best building (largest area, highest confidence)
    best = max(detections, key=lambda d: np.sum(d['mask']) * d['confidence'])
    print(f'  Selected: {best["class"]} (conf={best["confidence"]:.2f}, area={np.sum(best["mask"])} px)')
    
    # Step 3: Georeference mask
    print('\n[2/4] Georeferencing mask...')
    footprint = georeference_mask(best['mask'], ortho_bbox)
    if footprint is None:
        print('[ERROR] Could not extract footprint polygon')
        return None
    print(f'  Polygon: {len(footprint)} vertices')
    
    # Step 4: Fuse with LiDAR
    print('\n[3/4] LiDAR + CV fusion...')
    fusion = LidarCVFusion()
    result = fusion.fuse(footprint)
    
    if result is None:
        print('[ERROR] Fusion failed')
        return None
    
    # Step 5: Report
    print('\n[4/4] Results:')
    print(f'  Area:   {result["area_m2"]:.2f} m2 (CV: {result["cv_area_m2"]:.1f}, LiDAR: {result["lidar_area_m2"]:.1f})')
    print(f'  Height: {result["height_ridge_m"]:.2f} m (ridge), {result["height_eave_m"]:.2f} m (eave)')
    print(f'  Pitch:  {result["pitch_deg"]:.1f} deg')
    print(f'  Type:   {result["roof_type"]}')
    print(f'  Ridge:  {result["ridge_length_m"]:.1f} m')
    print(f'  Ground: {result["ground_mnm"]:.2f} m n.m.')
    print(f'  Points: {result["lidar_points"]:,}')
    
    if 'files' in result:
        print(f'\n  Files:')
        for k, v in result['files'].items():
            print(f'    {v}')
    
    # Save full result as JSON
    json_path = os.path.join(OUT_DIR, 'fusion_result.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        # Remove non-serializable geometry
        clean = {k: v for k, v in result.items() if k != 'footprint_geojson'}
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f'\n  JSON: {json_path}')
    
    return result

# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='CV + LiDAR Fusion Pipeline')
    p.add_argument('orthophoto', help='Path to ZBGIS orthophoto JPEG')
    p.add_argument('--lat', type=float, required=True, help='Center latitude')
    p.add_argument('--lon', type=float, required=True, help='Center longitude')
    p.add_argument('--extent', type=float, default=120, help='Orthophoto half-extent in meters (default 120)')
    p.add_argument('--model', help='Custom model path')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = p.parse_args()
    
    # Compute ortho bbox
    lat, lon = args.lat, args.lon
    d = args.extent / 111000  # approximate meters to degrees
    
    run_fusion_pipeline(
        args.orthophoto,
        ortho_bbox=(lat-d, lon-d, lat+d, lon+d),
        model_path=args.model,
        conf=args.conf,
    )
