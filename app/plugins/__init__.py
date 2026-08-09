"""
RoofAI Studio — LiDAR + CV pipeline integration plugin.
Registered via register_plugin(main_window).
Adds: Tools > Download LiDAR (ZBGIS)...
"""
import os, sys, json, time, threading, ssl, urllib.request, urllib.parse
import zipfile, imaplib, email, re, glob, math
import numpy as np
from email.header import decode_header
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import Delaunay

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QTextEdit, QLabel, QProgressBar, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal

EMAIL = 'jangrexa@gmail.com'
APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD', 'kxnzeoijbfrfaywh')
LAZ_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'laz')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output')
os.makedirs(LAZ_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


class LidarWorker(QThread):
    log = Signal(str)
    progress = Signal(int)
    done = Signal(bool, str, object)

    def __init__(self, address):
        super().__init__()
        self.address = address

    def run(self):
        try:
            self.log.emit('[1/4] Geocoding...')
            lat, lon, display = self._geocode()
            self.log.emit(f'  GPS: {lat:.7f}N, {lon:.7f}E')
            self.progress.emit(10)

            self.log.emit('[2/4] LAZ data...')
            self.progress.emit(25)
            if not self._ensure_laz(lat, lon):
                self.done.emit(False, 'No LAZ coverage. Run MAPKA export first.', None)
                return

            self.log.emit('[3/4] Extracting building...')
            self.progress.emit(50)
            points, ground_z, info = self._extract_building(lat, lon)
            if points is None:
                self.done.emit(False, 'No building found', None)
                return

            self.log.emit('[4/4] Creating 3D model...')
            self.progress.emit(75)
            result = self._create_outputs(points, ground_z, info, display, lat, lon)
            self.progress.emit(100)
            self.done.emit(True, 'Complete', result)
        except Exception as e:
            self.log.emit(f'ERROR: {e}')
            self.done.emit(False, str(e), None)

    def _geocode(self):
        url = f'https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(self.address)}&format=json&limit=1'
        ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = False
        req = urllib.request.Request(url, headers={'User-Agent': 'RoofAI/3.0'})
        with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
            data = json.loads(r.read())
        if not data: raise ValueError('Address not found')
        return float(data[0]['lat']), float(data[0]['lon']), data[0].get('display_name', self.address)

    def _ensure_laz(self, lat, lon):
        t = Transformer.from_crs('EPSG:4326', 'EPSG:8353', always_xy=True)
        e, n = t.transform(lon, lat)
        for f in glob.glob(os.path.join(LAZ_DIR, '*.laz')):
            if '.copc.' in f: continue
            try:
                import laspy; las = laspy.read(f)
                xs, ys = np.array(las.x), np.array(las.y)
                if xs.min() <= e <= xs.max() and ys.min() <= n <= ys.max():
                    return True
            except: pass
        return self._gmail_download()

    def _gmail_download(self):
        self.log.emit('  Checking Gmail for ZBGIS download...')
        try:
            mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
            mail.login(EMAIL, APP_PASSWORD)
            mail.select('INBOX')
            date_str = time.strftime('%d-%b-%Y')
            _, msgs = mail.search(None, 'SINCE', date_str, 'OR FROM skgeodesy.sk FROM gku.sk')
            ids = msgs[0].split() if msgs[0] else []
            links = []
            for mid in reversed(ids[-20:]):
                try:
                    _, data = mail.fetch(mid, '(RFC822)')
                    for r in data:
                        if isinstance(r, tuple):
                            body = ''
                            msg = email.message_from_bytes(r[1])
                            if msg.is_multipart():
                                for p in msg.walk():
                                    try:
                                        pl = p.get_payload(decode=True)
                                        if pl: body += pl.decode('utf-8', errors='replace')
                                    except: pass
                            else:
                                try: body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
                                except: pass
                            for u in re.findall(r'https?://[^\s<>"\']+', body):
                                if any(x in u.lower() for x in ['stiahn', 'download', 'export', '.zip', 'zbgis']):
                                    links.append(u)
                except: pass
            mail.logout()
            if links:
                self.log.emit(f'  Found download link!')
                ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = False
                req = urllib.request.Request(links[0], headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=180, context=ctx) as r:
                    data = r.read()
                path = os.path.join(LAZ_DIR, f'export_{int(time.time())}.zip')
                with open(path, 'wb') as f: f.write(data)
                with zipfile.ZipFile(path) as zf:
                    for n in zf.namelist():
                        if n.endswith('.laz') and '.copc.' not in n:
                            zf.extract(n, LAZ_DIR)
                self.log.emit(f'  Downloaded + extracted!')
                return True
        except Exception as e:
            self.log.emit(f'  Gmail error: {e}')
        return False

    def _extract_building(self, lat, lon):
        import laspy
        t = Transformer.from_crs('EPSG:4326', 'EPSG:8353', always_xy=True)
        t_inv = Transformer.from_crs('EPSG:8353', 'EPSG:4326', always_xy=True)
        te, tn = t.transform(lon, lat)
        bp_list, gp_list = [], []
        for f in glob.glob(os.path.join(LAZ_DIR, '*.laz')):
            if '.copc.' in f: continue
            try:
                las = laspy.read(f)
                xs = np.array(las.x); ys = np.array(las.y); zs = np.array(las.z)
                cls = np.array(las.classification, dtype=np.uint8)
                d = np.sqrt((xs - te)**2 + (ys - tn)**2)
                n = d < 60
                b, g = n & (cls == 6), n & (cls == 2)
                if np.any(b): bp_list.append(np.column_stack([xs[b], ys[b], zs[b]]))
                if np.any(g): gp_list.append(np.column_stack([xs[g], ys[g], zs[g]]))
            except: pass
        if not bp_list: return None, None, None
        bp = np.vstack(bp_list)
        gp = np.vstack(gp_list) if gp_list else bp
        gmed = float(np.median(gp[:, 2]))
        self.log.emit(f'  {len(bp):,} pts, terrain={gmed:.2f}m')
        gr = 0.5
        xi = ((bp[:, 0] - bp[:, 0].min()) / gr).astype(int)
        yi = ((bp[:, 1] - bp[:, 1].min()) / gr).astype(int)
        grid = np.zeros((yi.max() + 2, xi.max() + 2), dtype=bool)
        grid[yi, xi] = True
        labeled, nc = ndimage.label(grid)
        best, best_pts = None, None
        for cl in range(1, nc + 1):
            mask = labeled == cl
            area = np.sum(mask) * 0.25
            if area < 30: continue
            r, c = np.where(mask)
            cx = bp[:, 0].min() + np.mean(c) * gr
            cy = bp[:, 1].min() + np.mean(r) * gr
            if np.sqrt((cx - te)**2 + (cy - tn)**2) > 30: continue
            rad = max(area**0.5, 8) * 0.8
            in_c = (np.abs(bp[:, 0] - cx) < rad) & (np.abs(bp[:, 1] - cy) < rad)
            pts = bp[in_c]
            if best is None or np.sqrt((cx - te)**2 + (cy - tn)**2) < best['dist']:
                clon, clat = t_inv.transform(cx, cy)
                best = {'area': area, 'zmin': float(np.min(pts[:, 2])),
                        'zmax': float(np.max(pts[:, 2])), 'n': len(pts),
                        'dist': np.sqrt((cx - te)**2 + (cy - tn)**2),
                        'lat': clat, 'lon': clon}
                best_pts = pts
        return best_pts, gmed, best

    def _create_outputs(self, points, gz, info, display, lat, lon):
        pts = points.copy()
        c = pts[:, :2].mean(axis=0)
        pts[:, 0] -= c[0]; pts[:, 1] -= c[1]; pts[:, 2] -= gz
        xd = pts[:, 0].max() - pts[:, 0].min()
        yd = pts[:, 1].max() - pts[:, 1].min()
        zmx = pts[:, 2].max(); zmn = pts[:, 2].min(); zr = zmx - zmn
        try: tri = Delaunay(pts[:, :2])
        except: 
            pts[:, 0] += np.random.normal(0, 0.01, len(pts))
            pts[:, 1] += np.random.normal(0, 0.01, len(pts))
            tri = Delaunay(pts[:, :2])
        em = (pts[:, 2] > 0.5) & (pts[:, 2] < zmx - zr * 0.5)
        ez = float(np.median(pts[em, 2])) if np.sum(em) > 5 else zmn
        pitch = math.degrees(math.atan((zmx - ez) / (yd / 2))) if yd > 0 else 0
        safe = re.sub(r'[^a-z0-9]', '_', self.address.lower())[:30]
        # PLY
        ply = os.path.join(OUT_DIR, f'{safe}.ply')
        with open(ply, 'w') as f:
            f.write(f'ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(tri.simplices)}\nproperty list uchar int vertex_indices\nend_header\n')
            for v in pts: f.write(f'{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n')
            for fc in tri.simplices: f.write(f'3 {fc[0]} {fc[1]} {fc[2]}\n')
        # OBJ
        obj = os.path.join(OUT_DIR, f'{safe}.obj')
        with open(obj, 'w') as f:
            f.write(f'# {display}\n\n')
            for v in pts: f.write(f'v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n')
            for fc in tri.simplices: f.write(f'f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n')
        # Orthophoto
        t = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        x, y = t.transform(info['lon'], info['lat'])
        ext = 120
        url = 'https://zbgisws.skgeodesy.sk/zbgis_ortofoto_wms/service.svc/get?' + urllib.parse.urlencode({
            'SERVICE': 'WMS', 'VERSION': '1.3.0', 'REQUEST': 'GetMap',
            'LAYERS': '1', 'CRS': 'EPSG:3857',
            'BBOX': f'{x-ext},{y-ext},{x+ext},{y+ext}',
            'WIDTH': '4096', 'HEIGHT': '4096', 'FORMAT': 'image/jpeg', 'STYLES': 'default',
        })
        ctx_s = ssl.create_default_context(); ctx_s.check_hostname = False; ctx_s.verify_mode = False
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60, context=ctx_s) as r:
            jpg = os.path.join(OUT_DIR, f'{safe}_ortofoto.jpg')
            with open(jpg, 'wb') as f: f.write(r.read())
        return {
            'address': display, 'gps': {'lat': info['lat'], 'lon': info['lon']},
            'footprint': f'{xd:.1f} x {yd:.1f} m',
            'height_ridge': round(float(zmx), 2), 'height_eave': round(float(ez), 2),
            'pitch': round(float(pitch), 1), 'roof_type': 'flat' if zr < 2 else 'pitched',
            'area': round(info['area'], 0), 'ground': round(float(gz), 2), 'points': info['n'],
            'files': {'ply': ply, 'obj': obj, 'orthophoto': jpg},
        }


class LidarDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('LiDAR 3D — Address to Model')
        self.setMinimumSize(550, 420)
        self.setAttribute(Qt.WA_DeleteOnClose)
        l = QVBoxLayout(self)
        g = QGroupBox('Address')
        gl = QHBoxLayout(g)
        self.inp = QLineEdit(); self.inp.setPlaceholderText('e.g. Slnecna 988/60, 917 01 Trnava')
        self.inp.returnPressed.connect(self._go)
        gl.addWidget(self.inp)
        self.go = QPushButton('GO')
        self.go.clicked.connect(self._go)
        self.go.setStyleSheet('QPushButton{background:#2d7dd2;color:white;font-weight:bold;padding:6px 20px}')
        gl.addWidget(self.go); l.addWidget(g)
        self.pb = QProgressBar(); self.pb.setVisible(False); l.addWidget(self.pb)
        self.lg = QTextEdit(); self.lg.setReadOnly(True)
        self.lg.setStyleSheet('QTextEdit{background:#1a1a2e;color:#ddd;font-family:Consolas}'); l.addWidget(self.lg)

    def _go(self):
        a = self.inp.text().strip()
        if not a: return
        self.go.setEnabled(False); self.pb.setVisible(True); self.pb.setValue(0); self.lg.clear()
        self.lg.append(f'Address: {a}\n')
        self.w = LidarWorker(a)
        self.w.log.connect(self.lg.append)
        self.w.progress.connect(self.pb.setValue)
        self.w.done.connect(self._done)
        self.w.start()

    def _done(self, ok, msg, res):
        self.go.setEnabled(True); self.pb.setVisible(False)
        if ok and res:
            self.lg.append(f'\nDONE!')
            self.lg.append(f'{res["footprint"]} | {res["height_ridge"]}m | {res["pitch"]}deg | {res["roof_type"]}')
            self.lg.append(f'Files: {res["files"]["ply"]}')
            self.lg.append(f'       {res["files"]["orthophoto"]}')
        else:
            self.lg.append(f'\nFAILED: {msg}')


def register_plugin(main_window):
    mb = main_window.menu_bar
    mb.tools_menu.addSeparator()
    a = mb.tools_menu.addAction('Download LiDAR (ZBGIS)...')
    a.triggered.connect(lambda: LidarDialog(main_window).show())
    print('LiDAR plugin registered: Tools > Download LiDAR (ZBGIS)...')
