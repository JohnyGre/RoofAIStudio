# -*- coding: utf-8 -*-
"""
manual_outline.py - manuálne naklikávanie obrysu strechy na ortofoto + dopočet meshu.
Self-contained: download ortofoto (ZBGIS WMS), klikacie okno, LiDAR roviny, mesh.
"""
import os, sys, math, json, ssl, urllib.request, urllib.parse, time
import numpy as np
import cv2
from pyproj import Transformer
from shapely.geometry import Polygon as ShPolygon, Point, MultiPoint
from shapely import concave_hull
from scipy.spatial import Delaunay, cKDTree

from PySide6.QtWidgets import (QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
                               QMessageBox, QScrollArea)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

T52 = Transformer.from_crs('EPSG:4326', 'EPSG:8353', always_xy=True)
T25 = Transformer.from_crs('EPSG:8353', 'EPSG:4326', always_xy=True)
T3857_8353 = Transformer.from_crs('EPSG:3857', 'EPSG:8353', always_xy=True)
T8353_3857 = Transformer.from_crs('EPSG:8353', 'EPSG:3857', always_xy=True)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LAZ_DIR = os.path.join(PROJECT_DIR, 'data', 'laz')
OUT_DIR = os.path.join(PROJECT_DIR, 'output')
ORTHO_DIR = os.path.join(PROJECT_DIR, 'data', 'ortho')
os.makedirs(ORTHO_DIR, exist_ok=True)

WMS_URL = 'https://zbgisws.skgeodesy.sk/zbgis_ortofoto_wms/service.svc/get'


# ============================================================
# Ortofoto download + transformácia pixel <-> S-JTSK
# ============================================================
def download_ortho(lat, lon, half=30, size=1000):
    """Stiahne ortofoto (ZBGIS WMS) okolo (lat, lon) v S-JTSK bboxe half metrov.
    Vráti dict: path, xmin, ymax, scale (meter_sjtsk/px)."""
    cx, cy = T52.transform(lon, lat)  # S-JTSK
    xmin, xmax = cx - half, cx + half
    ymin, ymax = cy - half, cy + half
    # S-JTSK bbox -> EPSG:3857 (Web Mercator)
    mx_min, my_min = T8353_3857.transform(xmin, ymin)
    mx_max, my_max = T8353_3857.transform(xmax, ymax)
    if mx_min > mx_max:
        mx_min, mx_max = mx_max, mx_min
    if my_min > my_max:
        my_min, my_max = my_max, my_min
    ar = (ymax - ymin) / max((xmax - xmin), 1e-6)
    W = size
    H = max(100, int(round(size * ar)))
    for layers in ('1', '0', 'ortofoto'):
        url = (f'{WMS_URL}?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS={layers}'
               f'&STYLES=&CRS=EPSG:3857&BBOX={mx_min},{my_min},{mx_max},{my_max}'
               f'&WIDTH={W}&HEIGHT={H}&FORMAT=image/jpeg&TRANSPARENT=false')
        try:
            ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = False
            req = urllib.request.Request(url, headers={'User-Agent': 'RoofAI/2.0'})
            with urllib.request.urlopen(req, timeout=45, context=ctx) as r:
                data = r.read()
            if len(data) > 5000:
                path = os.path.join(ORTHO_DIR, 'manual_ortho.jpg')
                with open(path, 'wb') as f:
                    f.write(data)
                meta = {'path': path,
                        'mx_min': mx_min, 'my_min': my_min, 'mx_max': mx_max, 'my_max': my_max,
                        'W': W, 'H': H}
                return meta
        except Exception:
            continue
    return None


def px_to_sjtsk(px, py, meta):
    """Pixel (px, py) -> S-JTSK (x, y). Cez Web Mercator (nelineárny) + pyproj."""
    mx = meta['mx_min'] + (px / meta['W']) * (meta['mx_max'] - meta['mx_min'])
    my = meta['my_max'] - (py / meta['H']) * (meta['my_max'] - meta['my_min'])
    return T3857_8353.transform(mx, my)


def sjtsk_to_px(x, y, meta):
    mx, my = T8353_3857.transform(x, y)
    px = (mx - meta['mx_min']) / (meta['mx_max'] - meta['mx_min']) * meta['W']
    py = (meta['my_max'] - my) / (meta['my_max'] - meta['my_min']) * meta['H']
    return px, py


# ============================================================
# LiDAR body + roviny
# ============================================================
def fit_lsq(points):
    A = np.c_[points[:, 0], points[:, 1], np.ones(len(points))]
    coef, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    n = np.array([-coef[0], -coef[1], 1.0]); n /= np.linalg.norm(n)
    dists = np.abs(A @ coef - points[:, 2])
    return n, coef, dists


def ransac_plane(points, thresh=0.08, iters=1500, min_inl=100, seed=7):
    rng = np.random.default_rng(seed)
    best = None
    n = len(points)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p = points[idx]
        normal = np.cross(p[1]-p[0], p[2]-p[0])
        nl = np.linalg.norm(normal)
        if nl < 1e-9:
            continue
        normal /= nl
        d = -np.dot(normal, p[0])
        dist = np.abs(points @ normal + d)
        inl = dist < thresh
        if inl.sum() >= min_inl and (best is None or inl.sum() > best[0]):
            best = (int(inl.sum()), normal.copy(), inl.copy())
    return best


def extract_planes(points, max_planes=12):
    remaining = points.copy()
    planes = []
    for _ in range(max_planes):
        if len(remaining) < 200:
            break
        res = ransac_plane(remaining)
        if res is None or res[0] < 200:
            break
        cnt, n0, inl = res
        in_pts = remaining[inl]
        n, coef, dists = fit_lsq(in_pts)
        slope = math.degrees(math.acos(abs(n[2])))
        az = (math.degrees(math.atan2(-n[0], -n[1])) + 360) % 360
        planes.append({'n': n, 'coef': coef, 'pts': in_pts, 'slope': slope, 'az': az,
                       'rmse': float(np.sqrt(np.mean(dists**2))), 'cnt': cnt})
        remaining = remaining[~inl]
    return planes


def load_laz_points(lat, lon, radius=60):
    import laspy, glob
    cx, cy = T52.transform(lon, lat)
    out = []
    for f in glob.glob(os.path.join(LAZ_DIR, '*.laz')):
        if '.copc.' in f:
            continue
        try:
            las = laspy.read(f)
            xs, ys, zs = np.array(las.x), np.array(las.y), np.array(las.z)
            cls = np.array(las.classification, dtype=np.uint8)
            m = (np.abs(xs - cx) < radius) & (np.abs(ys - cy) < radius) & (cls == 6)
            if np.any(m):
                out.append(np.column_stack([xs[m], ys[m], zs[m]]))
        except Exception:
            pass
    if not out:
        return None
    pts = np.vstack(out)
    _, idx = np.unique(np.round(pts, 2), axis=0, return_index=True)
    return pts[np.sort(idx)]


# ============================================================
# Mesh z naklikaného obrysu + LiDAR rovín (top-down grid v obryse)
# ============================================================
def build_mesh_from_outline(outline, points, grid=0.35):
    """outline: list[(x,y) S-JTSK] obrys strechy. points: LiDAR body (N,3).
    Vráti (verts, faces, plane_areas, total)."""
    if points is None or len(points) < 200:
        return None, None, [], 0.0
    ob = ShPolygon(outline)
    # auto-kalibrácia: ZBGIS ortofoto má offset ~3-5 m voči LiDAR -> zarovnaj centroidy
    ob_c = np.array([ob.centroid.x, ob.centroid.y])
    lidar_c = np.median(points[:, :2], axis=0)
    shift = ob_c - lidar_c
    pts = points.copy()
    pts[:, :2] += shift
    # body len vo vnútri obrysu (+ buffer)
    buf = ob.buffer(2.0)
    inc = np.array([buf.contains(Point(p)) for p in pts[:, :2]])
    pts = pts[inc]
    if len(pts) < 200:
        return None, None, [], 0.0
    # odrež len skutočný terén (najnižšie 5%)
    z = pts[:, 2]
    zcut = z.min() + (z.max() - z.min()) * 0.05
    pts = pts[pts[:, 2] > zcut]

    planes = extract_planes(pts)
    main = [pl for pl in planes if pl['slope'] > 4 and pl['cnt'] > 50]
    if not main:
        main = planes[:6]
    print(f'[manual] RANSAC {len(planes)} rovín -> {len(main)} hlavných')

        # ---- merge koplanárnych susedných zhlukov (jedna fyzická rovina = jeden zhluk) ----
    def merge_coplanar_clusters(pls):
        import roof_geometry_reconstruction as _rg
        planes_r = [_rg.fit_plane(pl["pts"]) for pl in pls]
        adj = _rg.detect_adjacency(planes_r)
        parent = list(range(len(pls)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for (i, j) in adj:
            ni = planes_r[i]["normal"] / np.linalg.norm(planes_r[i]["normal"])
            nj = planes_r[j]["normal"] / np.linalg.norm(planes_r[j]["normal"])
            angle = float(np.degrees(np.arccos(np.clip(float(np.dot(ni, nj)), -1.0, 1.0))))
            if angle < 4.0:
                parent[find(j)] = find(i)
        groups = {}
        for k, pl in enumerate(pls):
            groups.setdefault(find(k), []).append(pl)
        out = []
        for root, pls_g in groups.items():
            if len(pls_g) == 1:
                out.append(pls_g[0])
                continue
            pts_all = np.vstack([pl["pts"] for pl in pls_g])
            pr = _rg.fit_plane(pts_all)
            slope = float(np.degrees(np.arccos(abs(pr["normal"][2]))))
            az = (float(np.degrees(np.arctan2(-pr["normal"][0], -pr["normal"][1]))) + 360) % 360
            rmse = float(np.sqrt(np.mean((pts_all @ pr["normal"] - pr["d"]) ** 2)))
            out.append({"normal": pr["normal"], "coef": pr["coef"], "pts": pts_all,
                        "slope": slope, "az": az, "rmse": rmse, "cnt": len(pts_all),
                        "d": float(pr["normal"] @ pr["centroid"])})
        return out

    main = merge_coplanar_clusters(main)
    print(f"[manual] po merge koplanárnych: {len(main)} rovín")

    # priesečníkové hrany rovín (hrebeň/nárožie/úžľabie) cez rgr modul
    import roof_geometry_reconstruction as _rgr
    _planes_rgr = []
    for _pl in main:
        _pr = _rgr.fit_plane(_pl['pts'])
        _pr['coef'] = _pl['coef']
        _pr['d'] = float(_pr['normal'] @ _pr['centroid'])
        _planes_rgr.append(_pr)
    _adj = _rgr.detect_adjacency(_planes_rgr)
    _edges_rgr = _rgr.plane_edges(_planes_rgr, _adj, ob)

    # ---- grid: majoritné hlasovanie LiDAR bodov per bunka (partition bez prekryvov) ----
    xs, ys = pts[:, 0], pts[:, 1]
    gx0, gy0 = xs.min() - 1, ys.min() - 1
    gx1, gy1 = xs.max() + 1, ys.max() + 1
    grid = 0.5
    nx = int(math.ceil((gx1 - gx0) / grid))
    ny = int(math.ceil((gy1 - gy0) / grid))
    counts = np.zeros((len(main), ny, nx), dtype=np.int32)
    for k, pl in enumerate(main):
        b = pl["pts"]
        ix = np.clip(((b[:, 0] - gx0) / grid).astype(np.int64), 0, nx - 1)
        iy = np.clip(((b[:, 1] - gy0) / grid).astype(np.int64), 0, ny - 1)
        np.add.at(counts, (k, iy, ix), 1)
    labels = counts.argmax(axis=0).astype(np.int16)
    labels[np.max(counts, axis=0) == 0] = -1
    # vyplniť prázdne bunky najbližšou hodnotou (partition bez dier)
    from scipy import ndimage as _ndi
    if (labels < 0).any():
        ind = _ndi.distance_transform_edt(labels < 0, return_distances=False, return_indices=True)
        labels = labels[tuple(ind)]

    # regióny per rovina -> kontúry -> approxPolyDP (rovné hrany)
    results = []
    for k, pl in enumerate(main):
        mask = (labels == k).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 3:
                continue
            poly_pts = np.array([cnt[i][0] for i in range(len(cnt))], dtype=float)
            ap = cv2.approxPolyDP(poly_pts.astype(np.float32).reshape(-1, 1, 2), 0.5, True)
            xy_s = [(gx0 + (p[0][0] + 0.5) * grid, gy0 + (p[0][1] + 0.5) * grid) for p in ap]
            if len(xy_s) < 3:
                continue
            results.append((xy_s, pl))

    # zlúčenie regiónov rovnakej roviny (1 rovina = 1 polygón)
    from shapely.ops import unary_union as _uuo
    from collections import defaultdict as _dd
    by_plane = _dd()
    for xy_s, pl in results:
        by_plane.setdefault(id(pl), []).append((xy_s, pl))
    results2 = []
    for k2, items in by_plane.items():
        pl = items[0][1]
        if len(items) == 1:
            results2.append(items[0])
            continue
        gs = [ShPolygon(it[0]) for it in items if len(it[0]) >= 3]
        if not gs:
            results2.append(items[0])
            continue
        u = _uuo(gs)
        if u.geom_type == "Polygon":
            results2.append((list(u.exterior.coords)[:-1], pl))
        else:
            for g in u.geoms:
                if g.geom_type == "Polygon" and g.area > 1.0:
                    results2.append((list(g.exterior.coords)[:-1], pl))
    results = results2

    verts, faces = [], []
    plane_areas = []

    def emit(xy, pl):
        xy = np.asarray(xy, dtype=float)
        if len(xy) < 3:
            return
        x2, y2 = xy[:, 0], xy[:, 1]
        a2d = 0.5 * abs(np.dot(x2, np.roll(y2, -1)) - np.dot(y2, np.roll(x2, -1)))
        if a2d < 0.5:
            return
        coef = pl['coef']
        a3d = a2d / math.cos(math.radians(pl['slope']))
        pts3d = np.column_stack([xy[:, 0], xy[:, 1], coef[0]*xy[:, 0] + coef[1]*xy[:, 1] + coef[2]])
        z_pts = pl['pts'][:, 2]
        zmin, zmax = float(z_pts.min()), float(z_pts.max())
        edges = []
        kk = len(xy)
        for t in range(kk):
            A = pts3d[t]; B = pts3d[(t+1) % kk]
            M = (A + B) / 2
            length = float(np.linalg.norm(B - A))
            dz_edge = abs(float(B[2] - A[2]))
            on_outline = ob.boundary.distance(Point(M[0], M[1])) < 1.0
            if not on_outline:
                # priesečníková priamka susednej roviny blízko? -> hrebeň/nárožie/úžľabie
                typ2 = None
                try:
                    _ki = main.index(pl)
                except ValueError:
                    _ki = -1
                for (ka, kb), e2 in _edges_rgr.items():
                    if _ki in (ka, kb):
                        if _rgr._dist_point_line_2d(M[:2], e2['p'][:2], e2['s'][:2]) < 2.0:
                            typ2 = e2['type']
                            break
                if typ2 is not None:
                    typ = typ2
                else:
                    zrel = (M[2] - zmin) / (zmax - zmin + 1e-9)
                    typ = 'o' if zrel < 0.35 else 'f'
            else:
                if dz_edge > 0.3:
                    typ = 'n'
                else:
                    zrel = (M[2] - zmin) / (zmax - zmin + 1e-9)
                    typ = 'o' if zrel < 0.35 else 'f'
            edges.append({'typ': typ, 'dlzka_m': round(length, 2)})
        spadnica = (zmax - zmin) / math.sin(math.radians(pl['slope']))
        base = len(verts)
        for p in xy:
            verts.append((float(p[0]), float(p[1]), float(coef[0]*p[0] + coef[1]*p[1] + coef[2])))
        try:
            tri = Delaunay(xy)
            shp = ShPolygon(xy)
            for t in tri.simplices:
                c = xy[t].mean(axis=0)
                if shp.contains(Point(c)) or shp.boundary.distance(Point(c)) < 0.05:
                    faces.append((int(base + t[0]), int(base + t[1]), int(base + t[2])))
        except Exception:
            ci = len(verts); verts.append(tuple(float(x) for x in np.mean(xy, axis=0)))
            for t in range(len(xy)):
                faces.append((base + t, base + (t+1) % len(xy), ci))
        plane_areas.append({'slope': round(pl['slope'], 1), 'az': round(pl['az'], 1),
                            'area_m2': round(a3d, 1), 'rmse_m': round(pl['rmse'], 4),
                            'hrany': edges, 'spadnica_m': round(spadnica, 2),
                            'z_min': round(zmin, 2), 'z_max': round(zmax, 2)})

    for xy_s, pl in results:
        emit(np.array(xy_s), pl)

    total = round(sum(p['area_m2'] for p in plane_areas), 1)
    return verts, faces, plane_areas, total


# ============================================================
# Klikacie okno
# ============================================================
class OutlineCanvas(QLabel):
    """QLabel s obrázkom; kliknutie pridá roh, pravé tlačidlo zmaže posledný."""
    changed = Signal()

    def __init__(self, meta):
        super().__init__()
        self.meta = meta
        self.points_px = []  # pôvodné pixely (nie zobrazené)
        self._pixmap = None
        self._scale = 1.0
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setMouseTracking(True)
        self.setStyleSheet('background:#111;')

    def set_image(self, path):
        img = QImage(path)
        self._orig = img
        # scale na šírku max 900px
        self._scale = min(1.0, 900.0 / img.width())
        w = int(img.width() * self._scale)
        h = int(img.height() * self._scale)
        self._pixmap = QPixmap.fromImage(img.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setPixmap(self._pixmap)
        self.setFixedSize(self._pixmap.size())

    def mousePressEvent(self, event):
        if self._pixmap is None:
            return
        pos = event.position()
        if event.button() == Qt.LeftButton:
            # zobrazené -> pôvodné px
            orig_x = pos.x() / self._scale
            orig_y = pos.y() / self._scale
            self.points_px.append((orig_x, orig_y))
            self.changed.emit()
        elif event.button() == Qt.RightButton:
            if self.points_px:
                self.points_px.pop()
                self.changed.emit()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.points_px:
            return
        p = QPainter(self)
        p.setPen(QPen(QColor(255, 80, 80), 3))
        for i, (px, py) in enumerate(self.points_px):
            x = px * self._scale
            y = py * self._scale
            p.drawEllipse(int(x)-5, int(y)-5, 10, 10)
            p.drawText(int(x)+8, int(y)-8, str(i+1))
        if len(self.points_px) > 1:
            p.setPen(QPen(QColor(255, 200, 60), 2))
            for i in range(len(self.points_px)):
                x1, y1 = self.points_px[i]
                x2, y2 = self.points_px[(i+1) % len(self.points_px)]
                p.drawLine(int(x1*self._scale), int(y1*self._scale),
                           int(x2*self._scale), int(y2*self._scale))
        p.end()

    def outline_sjtsk(self):
        return [px_to_sjtsk(px, py, self.meta) for px, py in self.points_px]


class ManualOutlineDialog(QDialog):
    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.meta = meta
        self.setWindowTitle('Naklikať obrys strechy')
        self.resize(900, 740)
        lay = QVBoxLayout(self)
        hint = QLabel('ĽAVÉ tlačidlo = pridať roh obrysu (v smere hodinových ručičiek)  |  '
                      'PRAVÉ tlačidlo = zmazať posledný  |  Dokončiť = zatvoriť')
        hint.setWordWrap(True)
        hint.setStyleSheet('color:#ddd; font-size:12px;')
        lay.addWidget(hint)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setStyleSheet('QScrollArea { background:#111; border:1px solid #333; }')
        self.canvas = OutlineCanvas(meta)
        self.canvas.set_image(meta['path'])
        self.scroll.setWidget(self.canvas)
        lay.addWidget(self.scroll, 1)
        btn = QHBoxLayout()
        self.clear_btn = QPushButton('Vyčistiť')
        self.clear_btn.clicked.connect(self._clear)
        self.done_btn = QPushButton('✅ Dokončiť')
        self.done_btn.clicked.connect(self._finish)
        self.done_btn.setStyleSheet('QPushButton { background:#2d7dd2; color:white; padding:8px 20px; border-radius:4px; }')
        btn.addWidget(self.clear_btn)
        btn.addStretch(1)
        btn.addWidget(self.done_btn)
        lay.addLayout(btn)
        self.outline = None

    def showEvent(self, event):
        super().showEvent(event)
        # posun scroll na stred obrázka (kde je strecha)
        QTimer.singleShot(50, lambda: self.scroll.ensureVisible(
            int(self.canvas.width()/2), int(self.canvas.height()/2),
            int(self.canvas.width()/2), int(self.canvas.height()/2)))

    def _clear(self):
        self.canvas.points_px.clear()
        self.canvas.update()

    def _finish(self):
        pts = self.canvas.points_px
        if len(pts) < 3:
            QMessageBox.warning(self, 'Málo bodov', 'Naklikaj aspoň 3 rohy obrysu.')
            return
        self.outline = self.canvas.outline_sjtsk()
        self.accept()


def build_flow(outline, points, meta):
    """Dopocet meshu z naklikaneho obrysu (bez dialogu - bezi vo workeri).
    Vrati dict s verts/faces/plane_areas/total/outline/ortho/meta alebo None."""
    verts, faces, plane_areas, total = build_mesh_from_outline(outline, points)
    if verts is None or len(verts) < 3:
        return None
    return {'verts': verts, 'faces': faces, 'plane_areas': plane_areas, 'total': total,
            'outline': outline, 'ortho': meta['path'], 'meta': meta}


def run_manual_flow(lat, lon, parent=None):
    """Kompletny flow: stiahni ortofoto -> naklikaj obrys -> dopocitaj mesh.
    POZOR: dialog MUSI bezat v hlavnom vlakne, nie v QThread!"""
    meta = download_ortho(lat, lon)
    if meta is None:
        QMessageBox.warning(parent, 'Ortofoto', 'Nepodarilo sa stiahnut ortofoto (ZBGIS WMS).')
        return None
    dlg = ManualOutlineDialog(meta, parent)
    if dlg.exec() != QDialog.Accepted or dlg.outline is None:
        return None
    outline = dlg.outline
    points = load_laz_points(lat, lon)
    if points is None:
        QMessageBox.warning(parent, 'LiDAR', 'Nenasli sa LiDAR body (class 6) pre tuto lokalitu.')
        return None
    return build_flow(outline, points, meta)
