# -*- coding: utf-8 -*-
"""
RoofAI Desktop v2 — GUI pre kompletnú pipeline (adresa → roviny → 3D mesh → viewer).
Rozšírenie pôvodného roofai_desktop.py o:
  - OSM adresy (Beluj 49+50) — Overpass addr:housenumber fallback
  - RANSAC roviny + koplanárny merge (presná segmentácia strechy)
  - top-down grid mesh s hrebeňmi (roviny ako plochy)
  - point cloud PLY/OBJ s farbami podľa rovín
  - self-contained 3D viewer (three.js inline)
"""
import sys, os, json, threading, time, ssl, urllib.request, urllib.parse, math, glob, io, re
import zipfile, imaplib, email
from email.header import decode_header
import numpy as np
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import Polygon as ShPolygon, Point, MultiPoint
from shapely import concave_hull

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTextEdit, QLabel, QProgressBar, QFileDialog,
    QGroupBox, QCheckBox, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QTextCursor

import manual_outline

# ============================================================
# CONFIG
# ============================================================
EMAIL = 'jangrexa@gmail.com'
APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD', 'kxnzeoijbfrfaywh')
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LAZ_DIR = os.path.join(PROJECT_DIR, 'data', 'laz')
OUT_DIR = os.path.join(PROJECT_DIR, 'output')
os.makedirs(LAZ_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

T52 = Transformer.from_crs('EPSG:4326', 'EPSG:8353', always_xy=True)
T25 = Transformer.from_crs('EPSG:8353', 'EPSG:4326', always_xy=True)


# ============================================================
# POMOCNÉ: RANSAC roviny + top-down mesh + viewer
# ============================================================
def fit_lsq(points):
    A = np.c_[points[:, 0], points[:, 1], np.ones(len(points))]
    coef, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    n = np.array([-coef[0], -coef[1], 1.0]); n /= np.linalg.norm(n)
    dists = np.abs(A @ coef - points[:, 2])
    return n, coef, dists


def ransac_plane(points, thresh=0.08, iters=1000, min_inl=50, seed=7):
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


def extract_planes(points, max_planes=14):
    """RANSAC + koplanárny merge -> zoznam rovín."""
    remaining = points.copy()
    raw = []
    for _ in range(max_planes):
        if len(remaining) < 50:
            break
        res = ransac_plane(remaining)
        if res is None or res[0] < 50:
            break
        cnt, n0, inl = res
        in_pts = remaining[inl]
        n, coef, dists = fit_lsq(in_pts)
        slope = math.degrees(math.acos(abs(n[2])))
        az = (math.degrees(math.atan2(-n[0], -n[1])) + 360) % 360
        raw.append({"n": n, "coef": coef, "pts": in_pts, "slope": slope, "az": az,
                    "rmse": float(np.sqrt(np.mean(dists**2))), "cnt": cnt})
        remaining = remaining[~inl]
    # merge
    merged = []
    for pl in raw:
        placed = False
        for m in merged:
            if abs(np.dot(pl["n"], m["n"])) > math.cos(math.radians(10)):
                navg = (pl["n"] + m["n"]); navg /= np.linalg.norm(navg)
                if abs(np.mean(pl["pts"] @ navg) - np.mean(m["pts"] @ navg)) < 0.3:
                    m["pts"] = np.vstack([m["pts"], pl["pts"]])
                    m["n"], m["coef"], dists = fit_lsq(m["pts"])
                    m["slope"] = math.degrees(math.acos(abs(m["n"][2])))
                    m["az"] = (math.degrees(math.atan2(-m["n"][0], -m["n"][1])) + 360) % 360
                    m["rmse"] = float(np.sqrt(np.mean(dists**2)))
                    m["cnt"] += pl["cnt"]
                    placed = True
                    break
        if not placed:
            merged.append(dict(pl))
    return merged


def topdown_mesh(planes, grid=0.5):
    """Top-down grid: každá bunka = rovina s najvyšším Z -> polygóny -> mesh."""
    allpts = np.vstack([pl["pts"] for pl in planes])
    xs, ys = allpts[:, 0], allpts[:, 1]
    gx0, gy0 = xs.min() - 1, ys.min() - 1
    gx1, gy1 = xs.max() + 1, ys.max() + 1
    nx = int(math.ceil((gx1 - gx0) / grid))
    ny = int(math.ceil((gy1 - gy0) / grid))
    if nx > 200 or ny > 200:
        grid = max((gx1-gx0)/200.0, (gy1-gy0)/200.0)
        nx = int(math.ceil((gx1 - gx0) / grid))
        ny = int(math.ceil((gy1 - gy0) / grid))

    h2d = []
    for pl in planes:
        h = concave_hull(MultiPoint(pl["pts"][:, :2]), ratio=0.05).simplify(0.35, preserve_topology=True)
        if h.geom_type == "MultiPolygon":
            h = max(h.geoms, key=lambda g: g.area)
        pl["hull2d"] = h
        if h.geom_type == "Polygon" and h.area > 4:
            h2d.append((h, pl))

    cx_g = gx0 + (np.arange(nx) + 0.5) * grid
    cy_g = gy0 + (np.arange(ny) + 0.5) * grid
    labels = np.full((ny, nx), -1, dtype=np.int16)
    z_top = np.full((ny, nx), -np.inf)
    for k, (h, pl) in enumerate(h2d):
        bx0, by0, bx1, by1 = h.bounds
        i0 = max(0, int((bx0-gx0)/grid)); i1 = min(nx, int((bx1-gx0)/grid)+1)
        j0 = max(0, int((by0-gy0)/grid)); j1 = min(ny, int((by1-gy0)/grid)+1)
        coef = pl["coef"]
        for j in range(j0, j1):
            yy = cy_g[j]
            for i in range(i0, i1):
                xx = cx_g[i]
                if h.contains(Point(xx, yy)):
                    zz = coef[0]*xx + coef[1]*yy + coef[2]
                    if zz > z_top[j, i]:
                        z_top[j, i] = zz
                        labels[j, i] = k

    import cv2
    verts, faces = [], []
    plane_areas = []
    for k, (h, pl) in enumerate(h2d):
        mask = (labels == k).astype(np.uint8)
        if mask.sum() == 0:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if len(cnt) < 3:
                continue
            poly = cv2.approxPolyDP(cnt, 0.8, True)
            pts = poly[:, 0, :].astype(float)
            if len(pts) < 3:
                continue
            coef = pl["coef"]
            xy = np.column_stack([gx0 + (pts[:, 0]+0.5)*grid, gy0 + (pts[:, 1]+0.5)*grid])
            zz = coef[0]*xy[:, 0] + coef[1]*xy[:, 1] + coef[2]
            p3d = np.column_stack([xy[:, 0], xy[:, 1], zz])
            x2, y2 = xy[:, 0], xy[:, 1]
            a2d = 0.5 * abs(np.dot(x2, np.roll(y2, -1)) - np.dot(y2, np.roll(x2, -1)))
            a3d = a2d / math.cos(math.radians(pl["slope"]))
            if a2d < 1.5:
                continue
            base = len(verts)
            for p in p3d:
                verts.append(tuple(p))
            ci = len(verts); verts.append(tuple(p3d.mean(axis=0)))
            kk = len(p3d)
            for j in range(kk):
                faces.append((base + j, base + (j+1) % kk, ci))
            plane_areas.append({"slope": round(pl["slope"], 1), "az": round(pl["az"], 1),
                                "area_m2": round(a3d, 1), "rmse_m": round(pl["rmse"], 4)})
    return verts, faces, plane_areas


def assign_planes(points, planes):
    """Každý bod -> najbližšia rovina (dist < 0.1)."""
    A = np.c_[points[:, 0], points[:, 1], np.ones(len(points))]
    assign = np.full(len(points), -1, dtype=np.int32)
    for i, pl in enumerate(planes):
        d = np.abs(A @ pl["coef"] - points[:, 2])
        better = np.full(len(points), True)
        for j, pj in enumerate(planes):
            if j == i:
                continue
            d2 = np.abs(A @ pj["coef"] - points[:, 2])
            better &= d <= d2 + 1e-9
        assign[(d < 0.1) & better] = i
    return assign


def make_viewer_html(verts, faces, title, area_m2, plane_areas):
    """Self-contained three.js viewer (inline, vlastné orbit ovládanie)."""
    three_path = os.path.join(OUT_DIR, 'three.min.js')
    if not os.path.exists(three_path):
        try:
            urllib.request.urlretrieve(
                'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js', three_path)
        except Exception:
            pass
    if os.path.exists(three_path):
        three_js = open(three_path, encoding='utf-8').read()
    else:
        three_js = ''  # bez three.js -> viewer nebude fungovať, ale súbor vznikne

    if len(verts) < 3:
        return None
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    cz = sum(v[2] for v in verts) / len(verts)
    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
    maxdim = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs), 1.0)

    planes_txt = " · ".join(f"{p['slope']:.0f}°/{p['az']:.0f}°" for p in plane_areas[:13])
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>%TITLE%</title>
<style>body{margin:0;overflow:hidden;font-family:Segoe UI,sans-serif;background:#14141f}
#info{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.82);color:#fff;padding:12px 16px;border-radius:10px;font-size:13px;z-index:10;max-width:400px;line-height:1.6;pointer-events:none}
#info b{color:#7CFC00}#info .r{color:#ff9f43}</style></head><body>
<div id="info"><b>🏠 %TITLE%</b><br>📐 Strešný plášť: <b>%AREA% m²</b> · %NPL% rovín<br>
<span class="r">%PLANES%</span><br><i style="color:#aaa">ťahaj — otáčanie · koliesko — zoom · pravé tlačidlo — posun</i></div>
<script>
%THREEJS%
</script>
<script>
(function(){
  var V = %VERTS%;
  var F = %FACES%;
  var scene = new THREE.Scene(); scene.background = new THREE.Color(0x14141f);
  var camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.1, 5000);
  var renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(innerWidth, innerHeight); renderer.setPixelRatio(devicePixelRatio||1);
  document.body.appendChild(renderer.domElement);
  var positions = new Float32Array(V.length*3);
  for (var i=0;i<V.length;i++){ positions[i*3]=V[i][0]-(%CX%); positions[i*3+1]=V[i][1]-(%CY%); positions[i*3+2]=V[i][2]-(%CZ%); }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions,3));
  var idx=[]; for (var j=0;j<F.length;j++){idx.push(F[j][0],F[j][1],F[j][2]);}
  geo.setIndex(idx); geo.computeVertexNormals();
  var pal=[[0.80,0.32,0.32],[0.32,0.72,0.85],[0.55,0.75,0.40],[0.85,0.62,0.25],[0.70,0.42,0.70],[0.35,0.75,0.65],[0.85,0.75,0.30],[0.55,0.45,0.75],[0.85,0.50,0.30],[0.45,0.65,0.85]];
  var colors=new Float32Array(V.length*3); var nrm=geo.getAttribute('normal');
  for (var i=0;i<V.length;i++){ var nx=nrm.getX(i),ny=nrm.getY(i),nz=nrm.getZ(i); var k=Math.floor(Math.abs(nx*3+nz*5+ny*7))%10;
    colors[i*3]=pal[k][0]; colors[i*3+1]=pal[k][1]; colors[i*3+2]=pal[k][2]; }
  geo.setAttribute('color', new THREE.BufferAttribute(colors,3));
  var mat=new THREE.MeshPhongMaterial({vertexColors:true,side:THREE.DoubleSide,flatShading:true});
  var mesh=new THREE.Mesh(geo,mat); scene.add(mesh);
  scene.add(new THREE.GridHelper(%MAXDIM%*2.5,30,0x666666,0x333333));
  scene.add(new THREE.AmbientLight(0xffffff,0.55));
  var dl=new THREE.DirectionalLight(0xffffff,0.9); dl.position.set(1,2,1); scene.add(dl);
  var theta=Math.PI/4, phi=Math.PI/5, dist=%MAXDIM%*2.0, panX=0, panY=0;
  var dragging=false, draggingR=false, lastX=0, lastY=0, autoRotate=true, lastMove=Date.now();
  function upd(){ var cp=Math.cos(phi);
    camera.position.set(dist*Math.sin(theta)*cp+panX, dist*Math.cos(phi)+panY, dist*Math.cos(theta)*cp);
    camera.lookAt(panX,panY,0); }
  renderer.domElement.addEventListener('mousedown',function(e){ if(e.button===2){draggingR=true;}else{dragging=true;} lastX=e.clientX;lastY=e.clientY;autoRotate=false;lastMove=Date.now(); });
  addEventListener('mousemove',function(e){ if(dragging){theta-=(e.clientX-lastX)*0.008; phi-=(e.clientY-lastY)*0.008; phi=Math.max(-1.5,Math.min(1.5,phi));} else if(draggingR){panX+=(e.clientX-lastX)*0.0004*dist; panY-=(e.clientY-lastY)*0.0004*dist;} lastX=e.clientX;lastY=e.clientY; });
  addEventListener('mouseup',function(){dragging=false;draggingR=false;lastMove=Date.now();});
  renderer.domElement.addEventListener('contextmenu',function(e){e.preventDefault();});
  addEventListener('wheel',function(e){dist*=(1+Math.sign(e.deltaY)*0.1);dist=Math.max(2,Math.min(5000,dist));},{passive:true});
  function anim(){requestAnimationFrame(anim); if(autoRotate && Date.now()-lastMove>1500){theta+=0.003;} upd(); renderer.render(scene,camera);} anim();
  addEventListener('resize',function(){camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
})();
</script></body></html>"""
    return (html.replace('%TITLE%', title).replace('%AREA%', str(area_m2)).replace('%NPL%', str(len(plane_areas)))
            .replace('%PLANES%', planes_txt).replace('%THREEJS%', three_js)
            .replace('%VERTS%', json.dumps([[round(v[0],2), round(v[1],2), round(v[2],2)] for v in verts]))
            .replace('%FACES%', json.dumps(faces))
            .replace('%CX%', str(round(cx,2))).replace('%CY%', str(round(cy,2))).replace('%CZ%', str(round(cz,2)))
            .replace('%MAXDIM%', str(round(maxdim,2))))


# ============================================================
# WORKER THREAD
# ============================================================
class PipelineWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    done_signal = Signal(bool, str, object)  # (success, message, result_dict)

    def __init__(self, address):
        super().__init__()
        self.address = address
        self.cancelled = False

    def log(self, msg):
        self.log_signal.emit(msg)
        time.sleep(0.005)

    def run(self):
        try:
            self.log('--- [1/6] Geocoding ---')
            lat, lon, display, footprints = self._geocode()
            self.log(f'  GPS: {lat:.7f}N, {lon:.7f}E')

            self.log('\n--- [2/6] LAZ data ---')
            self.progress_signal.emit(20)
            if not self._ensure_laz(lat, lon):
                self.done_signal.emit(False, 'No LAZ data available', None)
                return

            self.log('\n--- [3/6] Extracting building ---')
            self.progress_signal.emit(40)
            points, ground_z, info = self._extract_building(lat, lon, footprints)
            if points is None:
                self.done_signal.emit(False, 'No building found', None)
                return

            self.log('\n--- [4/6] RANSAC planes ---')
            self.progress_signal.emit(55)
            planes = extract_planes(points)
            self.log(f'  {len(planes)} roof planes found:')
            for i, pl in enumerate(planes):
                self.log(f'    P{i+1}: {pl["slope"]:.1f}° / {pl["az"]:.0f}°  (RMSE {pl["rmse"]*100:.1f} cm, {pl["cnt"]} pts)')

            self.log('--- [5/6] 3D mesh (priesecniky rovin) ---')
            self.progress_signal.emit(70)
            outline_sjtsk = None
            for fp in (footprints or []):
                if fp.get('outline') and len(fp['outline']) >= 4:
                    outline_sjtsk = fp['outline']
                    break
            if outline_sjtsk is not None:
                try:
                    import manual_outline as _mo
                    verts, faces, plane_areas, total = _mo.build_mesh_from_outline(outline_sjtsk, points)
                    total = round(float(total), 1)
                    self.log('  Obrys z OSM -> pravidelne roviny')
                except Exception as _ex:
                    self.log(f'  build_mesh_from_outline zlyhal ({_ex}), fallback top-down')
                    verts, faces, plane_areas = topdown_mesh(planes)
                    total = round(sum(p['area_m2'] for p in plane_areas), 1)
            else:
                verts, faces, plane_areas = topdown_mesh(planes)
                total = round(sum(p['area_m2'] for p in plane_areas), 1)
            self.log(f'  Mesh: {len(verts)} v, {len(faces)} f, plocha {total} m2')

            self.log('\n--- [6/6] Export (PLY / OBJ / viewer) ---')
            self.progress_signal.emit(85)
            result = self._create_outputs(points, ground_z, info, display, planes, verts, faces, plane_areas, total)
            self.progress_signal.emit(100)
            self.done_signal.emit(True, result['summary'], result)

        except Exception as e:
            import traceback
            self.log(f'\nERROR: {e}\n{traceback.format_exc()}')
            self.done_signal.emit(False, str(e), None)

    # ---------- geocoding ----------
    def _geocode(self):
        url = f'https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(self.address)}&format=json&limit=1'
        ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = False
        req = urllib.request.Request(url, headers={'User-Agent': 'RoofAI/2.0'})
        with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
            data = json.loads(r.read())
        if not data:
            raise ValueError('Address not found')
        lat, lon = float(data[0]['lat']), float(data[0]['lon'])
        display = data[0].get('display_name', self.address)
        footprints = []
        # ak adresa obsahuje číslo domu a Nominatim nenašiel presný dom -> OSM addr lookup
        mnum = re.search(r'\d+', self.address)
        if mnum and ('house' not in data[0].get('type', '') or True):
            footprints = self._osm_buildings(lat, lon, mnum.group(0))
            if footprints:
                lat = sum(f['lat'] for f in footprints) / len(footprints)
                lon = sum(f['lon'] for f in footprints) / len(footprints)
                self.log(f'  OSM: {len(footprints)} budov(y) s číslom {mnum.group(0)}')
        return lat, lon, display, footprints

    def _osm_buildings(self, lat, lon, number):
        """Overpass: budovy s addr:housenumber=number v okolí 1 km."""
        try:
            D = 0.01
            q = f"""[out:json][timeout:60];(way["building"]["addr:housenumber"="{number}"]({lat-D},{lon-D},{lat+D},{lon+D}););out geom tags;"""
            ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = False
            req = urllib.request.Request('https://overpass-api.de/api/interpreter',
                                         data=q.encode(), headers={'User-Agent': 'RoofAI/2.0'})
            with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                data = json.loads(r.read())
            out = []
            for e in data.get('elements', []):
                g = e.get('geometry')
                if not g or len(g) < 3:
                    c = e.get('center')
                    if c:
                        out.append({'lat': c['lat'], 'lon': c['lon'], 'id': e.get('id')})
                    continue
                xs = [p['lon'] for p in g]
                ys = [p['lat'] for p in g]
                outline = [T52.transform(x, y) for x, y in zip(xs, ys)]
                if outline and outline[0] != outline[-1]:
                    outline = outline + [outline[0]]
                out.append({'lat': sum(ys)/len(ys), 'lon': sum(xs)/len(xs),
                            'id': e.get('id'), 'outline': outline})
            return out
        except Exception:
            return []

    # ---------- LAZ ----------
    def _ensure_laz(self, lat, lon):
        e, n = T52.transform(lon, lat)
        import laspy, glob
        existing = []
        laz_files = glob.glob(os.path.join(LAZ_DIR, '*.laz'))
        self.log(f'  Checking {len(laz_files)} local LAZ files for coverage...')
        for f in laz_files:
            if '.copc.' in f:
                continue
            try:
                with laspy.open(f) as las:
                    if las.header.x_min <= e <= las.header.x_max and las.header.y_min <= n <= las.header.y_max:
                        existing.append(os.path.basename(f))
            except Exception as ex:
                self.log(f"  - Warning: Could not read LAZ header for {os.path.basename(f)}: {ex}")
                pass
        if existing:
            self.log(f'  Found coverage in {len(existing)} existing LAZ file(s): {", ".join(existing)}')
            return True
        self.log('  No local LAZ coverage. Checking Gmail for MAPKA export...')
        return self._download_from_gmail(e, n)

    def _download_from_gmail(self, e, n):
        self.log(f'  S-JTSK target: E={e:.0f}, N={n:.0f}')
        self.log('  Waiting for ZBGIS MAPKA export email...')
        start = time.time()
        while time.time() - start < 180 and not self.cancelled:
            try:
                mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
                mail.login(EMAIL, APP_PASSWORD)
                mail.select('INBOX')
                _, msgs = mail.search(None, 'FROM', 'skgeodesy.sk', 'SINCE', time.strftime('%d-%b-%Y'))
                mail_ids = msgs[0].split() if msgs and msgs[0] else []
                download_links = []
                for mid in reversed(mail_ids[-20:]):
                    try:
                        _, msg_data = mail.fetch(mid, '(RFC822)')
                        for resp in msg_data:
                            if isinstance(resp, tuple):
                                msg = email.message_from_bytes(resp[1])
                                body = self._get_body(msg)
                                for url in re.findall(r'https?://[^\s<>"\']+', body):
                                    if any(x in url.lower() for x in ['stiahn', 'download', 'export', '.zip', 'zbgis', 'mapka']):
                                        download_links.append(url)
                    except Exception:
                        pass
                mail.logout()
                if download_links:
                    self.log(f'  Found {len(download_links)} download link(s)!')
                    for url in download_links[:3]:
                        try:
                            ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = False
                            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                            with urllib.request.urlopen(req, timeout=180, context=ctx) as r:
                                data = r.read()
                            fname = f'export_{int(time.time())}.zip'
                            with open(os.path.join(LAZ_DIR, fname), 'wb') as f:
                                f.write(data)
                            self.log(f'  Saved: {fname} ({len(data)/1024/1024:.1f} MB)')
                            with zipfile.ZipFile(os.path.join(LAZ_DIR, fname)) as zf:
                                zf.extractall(LAZ_DIR)
                                for lf in [n for n in zf.namelist() if n.endswith('.laz') and '.copc.' not in n]:
                                    self.log(f'    Extracted: {lf}')
                            return True
                        except Exception as ex:
                            self.log(f'  Download failed: {ex}')
                    return True
                self.log(f'  Waiting... {int(time.time()-start)}s (complete MAPKA export now)')
                time.sleep(10)
            except Exception as ex:
                self.log(f'  Gmail error: {ex}')
                time.sleep(15)
        self.log('  Timeout — no download email received.')
        return False

    def _get_body(self, msg):
        body = ''
        if msg.is_multipart():
            for part in msg.walk():
                try:
                    p = part.get_payload(decode=True)
                    if p:
                        body += p.decode('utf-8', errors='replace')
                except Exception:
                    pass
        else:
            try:
                body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
            except Exception:
                pass
        return body

    # ---------- extraction ----------
    def _extract_building(self, lat, lon, footprints):
        import laspy
        te, tn = T52.transform(lon, lat)
        bp_list, gp_list = [], []
        for f in glob.glob(os.path.join(LAZ_DIR, '*.laz')):
            if '.copc.' in f:
                continue
            try:
                las = laspy.read(f)
                xs, ys, zs = np.array(las.x), np.array(las.y), np.array(las.z)
                cls = np.array(las.classification, dtype=np.uint8)
                dist = np.sqrt((xs-te)**2 + (ys-tn)**2)
                n = dist < 60
                bm, gm = n & (cls == 6), n & (cls == 2)
                if np.any(bm):
                    bp_list.append(np.column_stack([xs[bm], ys[bm], zs[bm]]))
                if np.any(gm):
                    gp_list.append(np.column_stack([xs[gm], ys[gm], zs[gm]]))
            except Exception:
                pass
        if not bp_list:
            return None, None, None
        bp = np.vstack(bp_list)
        gp = np.vstack(gp_list)
        gmed = float(np.median(gp[:, 2]))
        self.log(f'  {len(bp):,} building, {len(gp):,} ground points within 60m; ground {gmed:.2f} m')

        # ak máme OSM footprint, vyber body v ňom (+ buffer 6 m)
        if footprints:
            t_inv = T25
            polys = []
            for fp in footprints:
                polys.append(Point(fp['lon'], fp['lat']))
            c = MultiPoint(polys).centroid
            # zjednodušený výber: buffer okolo centroidov footprintov
            sel = np.zeros(len(bp), dtype=bool)
            for fp in footprints:
                ex, en = T52.transform(fp['lon'], fp['lat'])
                sel |= (np.abs(bp[:, 0]-ex) < 10) & (np.abs(bp[:, 1]-en) < 10)
            cand = bp[sel]
            if len(cand) > 50:
                bp = cand

        gr = 0.5
        xi = ((bp[:, 0]-bp[:, 0].min())/gr).astype(int)
        yi = ((bp[:, 1]-bp[:, 1].min())/gr).astype(int)
        grid = np.zeros((yi.max()+2, xi.max()+2), dtype=bool)
        grid[yi, xi] = True
        labeled, nc = ndimage.label(grid)
        best, best_pts = None, None
        for cl in range(1, nc+1):
            mask = labeled == cl
            area = np.sum(mask)*0.25
            if area < 30:
                continue
            rows, cols = np.where(mask)
            cx = bp[:, 0].min() + np.mean(cols)*gr
            cy = bp[:, 1].min() + np.mean(rows)*gr
            dist = np.sqrt((cx-te)**2 + (cy-tn)**2)
            if dist > 30:
                continue
            r = max(area**0.5, 8)*0.8
            in_c = (np.abs(bp[:, 0]-cx) < r) & (np.abs(bp[:, 1]-cy) < r)
            pts = bp[in_c]
            zmin, zmax = float(np.min(pts[:, 2])), float(np.max(pts[:, 2]))
            if best is None or dist < best['dist']:
                clon, clat = T25.transform(cx, cy)
                best = {'area': area, 'zmin': zmin, 'zmax': zmax, 'n': len(pts), 'dist': dist, 'lat': clat, 'lon': clon}
                best_pts = pts
        return best_pts, gmed, best

    # ---------- outputs ----------
    def _create_outputs(self, points, ground_z, info, display, planes, verts, faces, plane_areas, total):
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', self.address)[:40]
        base = os.path.join(OUT_DIR, safe)

        # OBJ mesh z rovín
        obj = base + '_mesh.obj'
        with open(obj, 'w', encoding='utf-8') as f:
            f.write(f'# {display} — roviny\n')
            for v in verts:
                f.write(f'v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n')
            for fc in faces:
                f.write(f'f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n')

        # point cloud PLY (farby podľa rovín)
        assign = assign_planes(points, planes)
        clean = points[assign >= 0]
        lab = assign[assign >= 0]
        palette = np.array([[255,80,80],[80,200,255],[120,255,120],[255,200,60],[255,120,255],
                            [60,255,220],[220,220,80],[180,120,255],[255,160,60],[80,160,255]], dtype=np.uint8)
        ply = base + '_cloud.ply'
        import struct
        with open(ply, 'wb') as f:
            f.write(b'ply\nformat binary_little_endian 1.0\n')
            f.write(f'element vertex {len(clean)}\n'.encode())
            f.write(b'property float x\nproperty float y\nproperty float z\n')
            f.write(b'property uchar red\nproperty uchar green\nproperty uchar blue\n')
            f.write(b'end_header\n')
            for i in range(len(clean)):
                f.write(struct.pack('<fff', clean[i,0], clean[i,1], clean[i,2]))
                c = palette[lab[i] % 10]
                f.write(struct.pack('<BBB', int(c[0]), int(c[1]), int(c[2])))

        # viewer HTML
        viewer = None
        try:
            html = make_viewer_html(verts, faces, display, total, plane_areas)
            if html:
                viewer = base + '_3d.html'
                with open(viewer, 'w', encoding='utf-8') as f:
                    f.write(html)
        except Exception as ex:
            self.log(f'  Viewer: {ex}')

        # meta JSON
        meta = {'address': display, 'gps': {'lat': info['lat'], 'lon': info['lon']},
                'ground_mnm': round(float(ground_z), 3),
                'area_m2': round(info['area'], 0),
                'plast_m2': total,
                'roviny': plane_areas,
                'points': len(points)}
        with open(base + '_meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        summary = (f'Address: {display}\n'
                   f'GPS: {info["lat"]:.6f}N, {info["lon"]:.6f}E\n'
                   f'Plocha plášťa: {total} m²\n'
                   f'Rovín: {len(plane_areas)}\n'
                   f'Terén: {ground_z:.2f} m n.m.\n\n'
                   f'Súbory:\n  {obj}\n  {ply}\n'
                   + (f'  {viewer}\n' if viewer else ''))
        return {'summary': summary, 'viewer': viewer, 'obj': obj, 'ply': ply, 'meta': meta,
                'roviny': plane_areas, 'plast_m2': total}


# ============================================================
# GUI
# ============================================================
class OutlineWorker(QThread):
    done_signal = Signal(object)
    log_signal = Signal(str)

    def __init__(self, lat, lon, parent_widget):
        super().__init__()
        self.lat = lat
        self.lon = lon
        self.parent_widget = parent_widget

    def run(self):
        try:
            result = manual_outline.run_manual_flow(self.lat, self.lon, self.parent_widget)
            self.done_signal.emit(result)
        except Exception as ex:
            self.log_signal.emit(f'Naklikávanie zlyhalo: {ex}')
            self.done_signal.emit(None)

class RoofAIWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('RoofAI Desktop v2')
        self.setMinimumSize(760, 640)
        self.last_result = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)

        title = QLabel('RoofAI — adresa → strešné roviny → 3D model')
        title.setFont(QFont('Segoe UI', 15, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        input_group = QGroupBox('Adresa')
        il = QHBoxLayout(input_group)
        self.address_input = QLineEdit()
        self.address_input.setPlaceholderText('napr. Beluj 50, 969 01 Beluj')
        self.address_input.setFont(QFont('Segoe UI', 12))
        self.address_input.returnPressed.connect(self.start_pipeline)
        self.go_btn = QPushButton('GO')
        self.go_btn.setFont(QFont('Segoe UI', 12, QFont.Bold))
        self.go_btn.setMinimumWidth(90)
        self.go_btn.clicked.connect(self.start_pipeline)
        self.go_btn.setStyleSheet('QPushButton { background: #2d7dd2; color: white; border-radius: 4px; padding: 6px 20px; } QPushButton:hover { background: #3d8de2; }')
        il.addWidget(self.address_input)
        il.addWidget(self.go_btn)
        layout.addWidget(input_group)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont('Consolas', 10))
        self.log_output.setStyleSheet('QTextEdit { background: #1a1a2e; color: #ddd; border: 1px solid #333; border-radius: 4px; }')
        layout.addWidget(self.log_output, 1)

        btn_row = QHBoxLayout()
        self.viewer_btn = QPushButton('🔍 Otvoriť 3D viewer')
        self.viewer_btn.setEnabled(False)
        self.viewer_btn.clicked.connect(self.open_viewer)
        self.folder_btn = QPushButton('📁 Otvoriť priečinok')
        self.folder_btn.clicked.connect(self.open_folder)
        self.outline_btn = QPushButton('Naklikat obrys')
        self.outline_btn.setEnabled(False)
        self.outline_btn.clicked.connect(self.open_outline)
        btn_row.addWidget(self.viewer_btn)
        btn_row.addWidget(self.folder_btn)
        btn_row.addWidget(self.outline_btn)
        layout.addLayout(btn_row)

        self.status_label = QLabel('Ready — enter an address and click GO')
        self.status_label.setStyleSheet('color: #888; padding: 4px;')
        layout.addWidget(self.status_label)

        quick_group = QGroupBox('Rýchly test')
        ql = QHBoxLayout(quick_group)
        for addr in ['Beluj 50, 969 01 Beluj', 'Slnečná 988/60, 917 01 Trnava']:
            b = QPushButton(addr)
            b.clicked.connect(lambda checked, a=addr: self.quick_test(a))
            ql.addWidget(b)
        layout.addWidget(quick_group)

        self.setStyleSheet('''
            QMainWindow { background: #0f0f1a; }
            QWidget { color: #ddd; }
            QGroupBox { color: #aaa; border: 1px solid #333; border-radius: 6px; margin-top: 8px; padding-top: 16px; }
            QGroupBox::title { padding: 0 8px; }
            QLineEdit { background: #1a1a2e; border: 1px solid #444; border-radius: 4px; padding: 8px; color: #fff; }
            QPushButton { background: #252540; border: 1px solid #444; border-radius: 4px; padding: 6px 14px; }
            QPushButton:hover { background: #353560; }
            QProgressBar { border: 1px solid #333; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background: #2d7dd2; border-radius: 4px; }
        ''')

    def quick_test(self, addr):
        self.address_input.setText(addr)
        self.start_pipeline()

    def open_viewer(self):
        if self.last_result and self.last_result.get('viewer'):
            os.startfile(self.last_result['viewer'])

    def open_folder(self):
        os.startfile(OUT_DIR)

    def start_pipeline(self):
        addr = self.address_input.text().strip()
        if not addr:
            self.log_output.append('[ERROR] Enter an address first.')
            return
        self.go_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.log_output.clear()
        self.log_output.append(f'RoofAI Pipeline\n{"="*50}\nAddress: {addr}')
        self.status_label.setText('Working...')
        self.worker = PipelineWorker(addr)
        self.worker.log_signal.connect(self.on_log)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.done_signal.connect(self.on_done)
        self.worker.start()

    def on_log(self, msg):
        self.log_output.append(msg)
        c = self.log_output.textCursor(); c.movePosition(QTextCursor.End); self.log_output.setTextCursor(c)

    def on_progress(self, val):
        self.progress.setValue(val)

    def on_done(self, success, message, result):
        self.go_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.last_result = result
        if success:
            self.log_output.append(f'\n{"="*50}\nDONE!\n{message}')
            if result and result.get('roviny'):
                self.log_output.append('\nVýmer rovín:')
                for r in result['roviny']:
                    self.log_output.append(f"  {r['slope']:.0f}°/{r['az']:.0f}° → {r['area_m2']} m²")
            self.status_label.setText('Complete')
            self.status_label.setStyleSheet('color: #4caf50; padding: 4px;')
            if result and result.get('viewer'):
                self.viewer_btn.setEnabled(True)
            if result and result.get('meta') and result['meta'].get('gps'):
                self.outline_btn.setEnabled(True)
                self._last_gps = (result['meta']['gps']['lat'], result['meta']['gps']['lon'])
        else:
            self.log_output.append(f'\nFAILED: {message}')
            self.status_label.setText(f'Failed: {message}')
            self.status_label.setStyleSheet('color: #e74c3c; padding: 4px;')

    def open_outline(self):
        gps = getattr(self, '_last_gps', None)
        if gps is None:
            self.log_output.append('Najprv spusti pipeline (GO) pre adresu, potom naklikaj obrys.')
            return
        lat, lon = gps
        self.status_label.setText('Sťahujem ortofoto (ZBGIS WMS)...')
        QApplication.processEvents()

        self.outline_worker = OutlineWorker(lat, lon, self)
        self.outline_worker.log_signal.connect(self.on_log)
        self.outline_worker.done_signal.connect(self.on_outline_done)
        self.outline_worker.start()

    def on_outline_done(self, result):
        if result is None:
            self.status_label.setText('Naklikávanie zrušené alebo zlyhalo')
            return

        verts = result['verts']; faces = result['faces']
        plane_areas = result['plane_areas']; total = result['total']
        outline = result['outline']; meta = result['meta']
        self.log_output.append(f'\n=== Manuálny obrys: {total} m², {len(plane_areas)} rovín ===')
        for r in plane_areas:
            self.log_output.append(f"  {r['slope']:.0f}°/{r['az']:.0f}°  {r['area_m2']} m²  "
                                   f"spádnica {r.get('spadnica_m','-')} m")
        # súčtová tabuľka hrán
        edge_names = {'h': 'hrebeň', 'o': 'odkvap', 'u': 'úžľabie', 'f': 'štít', 'n': 'nárožie'}
        edge_totals = {}
        for r in plane_areas:
            for e in r.get('hrany', []):
                edge_totals[e['typ']] = edge_totals.get(e['typ'], 0.0) + e['dlzka_m']
        if edge_totals:
            self.log_output.append('\nSúčtová tabuľka hrán:')
            for t in ['h', 'o', 'u', 'f', 'n']:
                if edge_totals.get(t, 0) > 0:
                    self.log_output.append(f"  {edge_names[t]}: {edge_totals[t]:.1f} m")
            zmax = max((r.get('z_max', 0) for r in plane_areas), default=0)
            zmin = min((r.get('z_min', 9e9) for r in plane_areas), default=0)
            self.log_output.append(f'  Výška strechy: {zmin:.2f}–{zmax:.2f} m n.m. (Δ {zmax-zmin:.2f} m)')
        # ulož mesh + viewer + 2D prekrytie
        base = os.path.join(OUT_DIR, 'manual_outline')
        try:
            with open(base + '_mesh.obj', 'w', encoding='utf-8') as f:
                for v in verts:
                    f.write(f'v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n')
                for fc in faces:
                    f.write(f'f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n')
            html = make_viewer_html(verts, faces, 'Manuálny obrys', total, plane_areas)
            viewer = base + '_3d.html'
            if html:
                with open(viewer, 'w', encoding='utf-8') as f:
                    f.write(html)
            # 2D prekrytie
            import cv2
            img = cv2.imread(result['ortho'])
            if img is not None:
                pts = np.array([manual_outline.sjtsk_to_px(x, y, meta) for x, y in outline], dtype=np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255), 3)
                for i, p in enumerate(pts):
                    cv2.circle(img, tuple(p), 6, (0, 0, 255), -1)
                    cv2.putText(img, str(i+1), (p[0]+8, p[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # obrysy rovín (okrajové hrany meshu) cez ortofoto
                from collections import Counter
                _ec = Counter()
                for fc in faces:
                    for _a, _b in [(fc[0], fc[1]), (fc[1], fc[2]), (fc[2], fc[0])]:
                        _ec[tuple(sorted((_a, _b)))] += 1
                for (_a, _b), _c in _ec.items():
                    if _c == 1:
                        p1 = manual_outline.sjtsk_to_px(verts[_a][0], verts[_a][1], meta)
                        p2 = manual_outline.sjtsk_to_px(verts[_b][0], verts[_b][1], meta)
                        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 200, 255), 2)
                cv2.imwrite(base + '_2d.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.last_result = {'viewer': viewer}
            self.viewer_btn.setEnabled(True)
            self.log_output.append(f'\nSúbory:\n  {base}_mesh.obj\n  {viewer}\n  {base}_2d.jpg')
            self.status_label.setText(f'Manuálny obrys: {total} m² hotový')
            self.status_label.setStyleSheet('color: #4caf50; padding: 4px;')
            try:
                os.startfile(viewer)
            except Exception:
                pass
        except Exception as ex:
            self.log_output.append(f'Uloženie zlyhalo: {ex}')


def main():
    app = QApplication(sys.argv)
    window = RoofAIWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
