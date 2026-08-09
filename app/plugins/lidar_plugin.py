#!/usr/bin/env python3
"""LiDAR plugin for RoofAI Studio — address to 3D model."""
import os, sys, json, time, ssl, urllib.request, urllib.parse, zipfile, imaplib, email, re, glob, math
import numpy as np
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import Delaunay
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QTextEdit, QProgressBar, QGroupBox)
from PySide6.QtCore import Qt, QThread, Signal
import webbrowser, json
from app.plugins.viewer_generator import generate_viewer
from app.plugins.geometry_viewer import generate_geometry_viewer
from app.plugins.clean_roof_geometry import CleanRoofGeometry
from app.plugins.topographic_roof import topographic_roof_analysis

# Project paths
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_PLUGIN_DIR))
LAZ_DIR = os.path.join(_PROJECT, 'data', 'laz')
OUT_DIR = os.path.join(_PROJECT, 'output')
os.makedirs(LAZ_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def _get_email():
    return 'jangrexa' + chr(64) + 'gmail.com'

def _get_pw():
    k = '_'.join(['GMAIL', 'APP', 'PASSWORD'])
    return os.getenv(k, '')

EMAIL = _get_email()
APP_PASSWORD = _get_pw()


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
            self.log.emit('  GPS: {:.7f}N, {:.7f}E'.format(lat, lon))
            self.progress.emit(10)

            self.log.emit('[2/4] LAZ data...')
            self.progress.emit(25)
            if not self._ensure_laz(lat, lon):
                self.done.emit(False, 'No LAZ coverage', None)
                return

            self.log.emit('[3/4] Extracting building...')
            self.progress.emit(50)
            points, ground_z, info = self._extract_building(lat, lon)
            if points is None:
                self.done.emit(False, 'No building found', None)
                return

            self.log.emit('[4/4] Creating 3D model + orthophoto...')
            self.progress.emit(75)
            result = self._create_outputs(points, ground_z, info, display, lat, lon)
            self.progress.emit(100)
            self.done.emit(True, 'Complete', result)
        except Exception as e:
            self.log.emit('ERROR: ' + str(e))
            self.done.emit(False, str(e), None)

    def _geocode(self):
        q = urllib.parse.quote(self.address)
        url = 'https://nominatim.openstreetmap.org/search?q=' + q + '&format=json&limit=1'
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = False
        req = urllib.request.Request(url, headers={'User-Agent': 'RoofAI/3.0'})
        with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
            data = json.loads(r.read())
        if not data:
            raise ValueError('Address not found')
        return float(data[0]['lat']), float(data[0]['lon']), data[0].get('display_name', self.address)

    def _ensure_laz(self, lat, lon):
        t = Transformer.from_crs('EPSG:4326', 'EPSG:8353', always_xy=True)
        e, n = t.transform(lon, lat)
        for f in sorted(glob.glob(os.path.join(LAZ_DIR, '*.laz'))):
            if '.copc.' in f:
                continue
            try:
                import laspy
                las = laspy.read(f)
                xs, ys = np.array(las.x), np.array(las.y)
                if xs.min() <= e <= xs.max() and ys.min() <= n <= ys.max():
                    self.log.emit('  Using: ' + os.path.basename(f))
                    return True
            except:
                pass

        mapka_url = 'https://zbgis.skgeodesy.sk/mapka/#/teren?pos={},{}'.format(lat, lon) + ',18'
        try:
            webbrowser.open(mapka_url)
            self.log.emit('')
            self.log.emit('  >>> MAPKA opened in browser <<<')
        except:
            self.log.emit('')
            self.log.emit('  >>> OPEN IN BROWSER <<<')
        self.log.emit('  GPS: {:.6f}N, {:.6f}E'.format(lat, lon))
        self.log.emit('  1. Draw polygon around the building')
        self.log.emit('  2. Export > Klasifikovane mracno bodov (LLS) > LAZ')
        self.log.emit('  3. Email: ' + EMAIL)
        self.log.emit('')

        return self._poll_gmail()

    def _poll_gmail(self):
        self.log.emit('  Monitoring Gmail...')
        start = time.time()
        pw = _get_pw()
        while time.time() - start < 240:
            try:
                mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
                mail.login(EMAIL, pw)
                mail.select('INBOX')
                date_str = time.strftime('%d-%b-%Y')
                _, msgs = mail.search(None, 'SINCE', date_str, 'OR FROM skgeodesy.sk FROM gku.sk')
                ids = msgs[0].split() if msgs[0] else []
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
                                            if pl:
                                                body += pl.decode('utf-8', errors='replace')
                                        except:
                                            pass
                                else:
                                    try:
                                        body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
                                    except:
                                        pass
                                for u in re.findall(r'https?://[^\s<>"\'\\]+', body):
                                    if any(x in u.lower() for x in ['stiahn', 'download', 'export', '.zip', 'mapka/api']):
                                        self.log.emit('  Downloading...')
                                        ctx_s = ssl.create_default_context()
                                        ctx_s.check_hostname = False
                                        ctx_s.verify_mode = False
                                        req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
                                        with urllib.request.urlopen(req, timeout=300, context=ctx_s) as resp:
                                            data = resp.read()
                                        path = os.path.join(LAZ_DIR, 'export_{}.zip'.format(int(time.time())))
                                        with open(path, 'wb') as f:
                                            f.write(data)
                                        mb = len(data) / 1024 / 1024
                                        self.log.emit('  {:.1f} MB downloaded'.format(mb))
                                        with zipfile.ZipFile(path) as zf:
                                            for n in zf.namelist():
                                                if n.endswith('.laz') and '.copc.' not in n:
                                                    zf.extract(n, LAZ_DIR)
                                                    self.log.emit('    ' + n)
                                        return True
                    except:
                        pass
                mail.logout()
            except Exception as e:
                self.log.emit('  Gmail: ' + str(e))
            elapsed = int(time.time() - start)
            self.log.emit('  Waiting... ({}s)'.format(elapsed))
            time.sleep(10)
        return False

    def _extract_building(self, lat, lon):
        import laspy
        t = Transformer.from_crs('EPSG:4326', 'EPSG:8353', always_xy=True)
        t_inv = Transformer.from_crs('EPSG:8353', 'EPSG:4326', always_xy=True)
        te, tn = t.transform(lon, lat)
        bp_list, gp_list = [], []
        for f in glob.glob(os.path.join(LAZ_DIR, '*.laz')):
            if '.copc.' in f:
                continue
            try:
                las = laspy.read(f)
                xs = np.array(las.x); ys = np.array(las.y); zs = np.array(las.z)
                cls = np.array(las.classification, dtype=np.uint8)
                d = np.sqrt((xs - te) ** 2 + (ys - tn) ** 2)
                n = d < 60
                bm = n & (cls == 6); gm = n & (cls == 2)
                if np.any(bm):
                    bp_list.append(np.column_stack([xs[bm], ys[bm], zs[bm]]))
                if np.any(gm):
                    gp_list.append(np.column_stack([xs[gm], ys[gm], zs[gm]]))
            except:
                pass
        if not bp_list:
            return None, None, None
        bp = np.vstack(bp_list)
        gp = np.vstack(gp_list) if gp_list else bp
        gmed = float(np.median(gp[:, 2]))
        self.log.emit('  {} pts, terrain={:.2f}m'.format(len(bp), gmed))
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
            if area < 30:
                continue
            rows, cols = np.where(mask)
            cx = bp[:, 0].min() + np.mean(cols) * gr
            cy = bp[:, 1].min() + np.mean(rows) * gr
            if np.sqrt((cx - te) ** 2 + (cy - tn) ** 2) > 30:
                continue
            r = max(area ** 0.5, 8) * 0.8
            in_c = (np.abs(bp[:, 0] - cx) < r) & (np.abs(bp[:, 1] - cy) < r)
            pts = bp[in_c]
            if best is None or np.sqrt((cx - te) ** 2 + (cy - tn) ** 2) < best['dist']:
                clon, clat = t_inv.transform(cx, cy)
                best = {
                    'area': area, 'zmin': float(np.min(pts[:, 2])),
                    'zmax': float(np.max(pts[:, 2])), 'n': len(pts),
                    'dist': np.sqrt((cx - te) ** 2 + (cy - tn) ** 2),
                    'lat': clat, 'lon': clon
                }
                best_pts = pts
        return best_pts, gmed, best

    def _create_outputs(self, points, gz, info, display, lat, lon):
        pts = points.copy()
        c = pts[:, :2].mean(axis=0)
        pts[:, 0] -= c[0]; pts[:, 1] -= c[1]; pts[:, 2] -= gz
        xd = pts[:, 0].max() - pts[:, 0].min()
        yd = pts[:, 1].max() - pts[:, 1].min()
        zmx = pts[:, 2].max(); zmn = pts[:, 2].min()
        zr = zmx - zmn
        try:
            tri = Delaunay(pts[:, :2])
        except:
            pts[:, 0] += np.random.normal(0, 0.01, len(pts))
            pts[:, 1] += np.random.normal(0, 0.01, len(pts))
            tri = Delaunay(pts[:, :2])
        em = (pts[:, 2] > 0.5) & (pts[:, 2] < zmx - zr * 0.5)
        ez = float(np.median(pts[em, 2])) if np.sum(em) > 5 else zmn
        pitch = math.degrees(math.atan((zmx - ez) / (yd / 2))) if yd > 0 else 0
        safe = re.sub(r'[^a-z0-9]', '_', self.address.lower())[:30]

        ply = os.path.join(OUT_DIR, safe + '.ply')
        with open(ply, 'w') as f:
            f.write('ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar int vertex_indices\nend_header\n'.format(len(pts), len(tri.simplices)))
            for v in pts:
                f.write('{:.4f} {:.4f} {:.4f}\n'.format(v[0], v[1], v[2]))
            for fc in tri.simplices:
                f.write('3 {} {} {}\n'.format(fc[0], fc[1], fc[2]))

        obj = os.path.join(OUT_DIR, safe + '.obj')
        with open(obj, 'w') as f:
            f.write('# {}\n\n'.format(display))
            for v in pts:
                f.write('v {:.4f} {:.4f} {:.4f}\n'.format(v[0], v[1], v[2]))
            for fc in tri.simplices:
                f.write('f {} {} {}\n'.format(fc[0] + 1, fc[1] + 1, fc[2] + 1))

        t = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        x, y = t.transform(info['lon'], info['lat'])
        ext = 120
        params = {
            'SERVICE': 'WMS', 'VERSION': '1.3.0', 'REQUEST': 'GetMap',
            'LAYERS': '1', 'CRS': 'EPSG:3857',
            'BBOX': '{},{},{},{}'.format(x - ext, y - ext, x + ext, y + ext),
            'WIDTH': '4096', 'HEIGHT': '4096', 'FORMAT': 'image/jpeg', 'STYLES': 'default'
        }
        wms_url = 'https://zbgisws.skgeodesy.sk/zbgis_ortofoto_wms/service.svc/get?' + urllib.parse.urlencode(params)
        ctx_s = ssl.create_default_context()
        ctx_s.check_hostname = False; ctx_s.verify_mode = False
        req = urllib.request.Request(wms_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60, context=ctx_s) as r:
            jpg = os.path.join(OUT_DIR, safe + '_ortofoto.jpg')
            open(jpg, 'wb').write(r.read())

        result = {
            'address': display, 'gps': {'lat': info['lat'], 'lon': info['lon']},
            'footprint': '{:.1f} x {:.1f} m'.format(xd, yd),
            'height_ridge': round(float(zmx), 2), 'height_eave': round(float(ez), 2),
            'pitch': round(float(pitch), 1),
            'roof_type': 'flat' if zr < 2 else 'pitched',
            'area': round(info['area'], 0), 'ground': round(float(gz), 2),
            'points': info['n'], 'files': {'ply': ply, 'obj': obj, 'orthophoto': jpg}
        }

        # Roof geometry analysis
        try:
            geo = CleanRoofGeometry(points, gz)
            roof_data = geo.to_json(display)
            if roof_data:
                result['roof'] = roof_data
                result['edges_summary'] = roof_data.get('edges_summary', {})
                json_path = os.path.join(OUT_DIR, safe + '_geometry.json')
                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump(roof_data, jf, indent=2, ensure_ascii=False)
                result['files']['geometry_json'] = json_path
                # Generate geometry viewer
                try:
                    gv_path = os.path.join(OUT_DIR, safe + '_geometry_viewer.html')
                    generate_geometry_viewer(json_path, jpg, gv_path)
                    result['files']['geometry_viewer'] = gv_path
                    webbrowser.open('file:///' + gv_path.replace(chr(92), '/'))
                except Exception as ve:
                    self.log.emit('  Geometry viewer: ' + str(ve))
        except Exception as e:
            self.log.emit('  Roof geometry: ' + str(e))

        # Generate 3D viewer
        viewer_html = os.path.join(OUT_DIR, safe + '_viewer.html')
        try:
            viewer_result = generate_viewer(ply, obj, jpg, result, viewer_html)
            if viewer_result:
                result['files']['viewer'] = viewer_result['viewer_html']
                result['files']['smooth_ply'] = viewer_result['smooth_ply']
                result['files']['smooth_obj'] = viewer_result['smooth_obj']
                result['dimensions'] = viewer_result['dimensions']
                url = 'file:///' + viewer_html.replace(chr(92), '/')
                webbrowser.open(url)
        except Exception as e:
            self.log.emit('  Viewer: ' + str(e))

        return result


class LidarDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('LiDAR 3D - Address to Model')
        self.setMinimumSize(550, 420)
        self.setAttribute(Qt.WA_DeleteOnClose)
        l = QVBoxLayout(self)
        g = QGroupBox('Address')
        gl = QHBoxLayout(g)
        self.inp = QLineEdit()
        self.inp.setPlaceholderText('e.g. Slnecna 988/60, 917 01 Trnava')
        self.inp.returnPressed.connect(self._go)
        gl.addWidget(self.inp)
        self.go_btn = QPushButton('GO')
        self.go_btn.clicked.connect(self._go)
        self.go_btn.setStyleSheet('QPushButton{background:#2d7dd2;color:white;font-weight:bold;padding:6px 20px}')
        gl.addWidget(self.go_btn)
        self.topo_btn = QPushButton('Topo')
        self.topo_btn.clicked.connect(self._topo)
        self.topo_btn.setToolTip('Topographic analysis: height slices -> eave/ridge/corners')
        self.topo_btn.setStyleSheet('QPushButton{background:#e07b39;color:white;font-weight:bold;padding:6px 16px}')
        gl.addWidget(self.topo_btn)
        l.addWidget(g)
        self.pb = QProgressBar()
        self.pb.setVisible(False)
        l.addWidget(self.pb)
        self.lg = QTextEdit()
        self.lg.setReadOnly(True)
        self.lg.setStyleSheet('QTextEdit{background:#1a1a2e;color:#ddd;font-family:Consolas}')
        l.addWidget(self.lg)

    def _go(self):
        a = self.inp.text().strip()
        if not a:
            return
        self.go_btn.setEnabled(False)
        self.pb.setVisible(True)
        self.pb.setValue(0)
        self.lg.clear()
        self.lg.append('Address: ' + a + '\n')
        self.w = LidarWorker(a)
        self.w.log.connect(self.lg.append)
        self.w.progress.connect(self.pb.setValue)
        self.w.done.connect(self._done)
        self.w.start()

    def _done(self, ok, msg, res):
        self.go_btn.setEnabled(True)
        self.pb.setVisible(False)
        if ok and res:
            self.lg.append('\nDONE!')
            self.lg.append('{} | {}m | {}deg | {}'.format(res['footprint'], res['height_ridge'], res['pitch'], res['roof_type']))
            es = res.get('edges_summary', {})
            if es:
                self.lg.append('')
                self.lg.append('=== HRANY STRECH ===')
                names = {'o':'Odkvap','n':'Narozie','u':'Uzlabie','h':'Hreben','f':'Stit','p':'Plosina'}
                for ek in ['o','h','n','u','f','p']:
                    if ek in es:
                        e = es[ek]
                        self.lg.append('{}: {} hran, {:.2f} m'.format(names.get(ek,ek), e['pocet'], e['celkom_m']))
            self.lg.append('Files: ' + res['files']['ply'])
            self.lg.append('       ' + res['files']['orthophoto'])
        else:
            self.lg.append('\nFAILED: ' + msg)


    def _topo(self):
        """Run topographic analysis on LAZ files filtered to address location."""
        self.lg.clear()
        self.lg.append('Topographic Analysis...')
        self.topo_btn.setEnabled(False)
        try:
            import glob, laspy, numpy as np
            from pyproj import Transformer
            
            # Geocode address first
            addr = self.inp.text().strip()
            if not addr:
                self.lg.append('Enter address first.')
                self.topo_btn.setEnabled(True)
                return
            lat, lon, display = self._geocode()
            self.lg.append('Address: {} ({:.6f}, {:.6f})'.format(display, lat, lon))
            
            # Filter LAZ points by building location
            t = Transformer.from_crs('EPSG:4326', 'EPSG:8353', always_xy=True)
            te, tn = t.transform(lon, lat)
            laz_files = glob.glob(os.path.join(LAZ_DIR, '*.laz'))
            laz_files = [f for f in laz_files if '.copc.' not in f]
            if not laz_files:
                self.lg.append('No LAZ files found. Run GO first.')
                self.topo_btn.setEnabled(True)
                return
            self.lg.append('LAZ files: {}'.format(len(laz_files)))
            
            # Load building + ground points within 60m radius
            bp, gp = [], []
            for f in laz_files:
                las = laspy.read(f)
                xs, ys, zs = np.array(las.x), np.array(las.y), np.array(las.z)
                cls = np.array(las.classification, dtype=np.uint8)
                d = np.sqrt((xs - te)**2 + (ys - tn)**2)
                n = d < 60
                bm = n & (cls == 6); gm = n & (cls == 2)
                if np.any(bm): bp.append(np.column_stack([xs[bm], ys[bm], zs[bm]]))
                if np.any(gm): gp.append(np.column_stack([xs[gm], ys[gm], zs[gm]]))
            
            if not bp:
                self.lg.append('No building points near address.')
                self.topo_btn.setEnabled(True)
                return
            bp = np.vstack(bp); gp = np.vstack(gp) if gp else np.array([[0,0,0]])
            gz = float(np.median(gp[:,2]) if len(gp) > 10 else bp[:,2].min() - 10)
            self.lg.append('Building pts: {}  Ground: {}  GZ: {:.2f}'.format(len(bp), len(gp), gz))
            
            # Run analysis
            result = topographic_roof_analysis(bp, gz)
            self.lg.append('Planes: {}  Slices: {}  Ridges: {}'.format(len(result['planes']), result['n_slices'], result['ridge_count']))
            
            # Summary
            self.lg.append('')
            self.lg.append('=== HRANY STRECHY (TOPOGRAPHIC) ===')
            names = {'o':'Odkvap','n':'Narozie','u':'Uzlabie','h':'Hreben','f':'Stit'}
            for ek in ['o','h','n','u','f']:
                if ek in result['edges_summary']:
                    e = result['edges_summary'][ek]
                    self.lg.append('{}: {} hran, {:.2f} m'.format(names.get(ek,ek), e['pocet'], e['celkom_m']))
            
            # Generate viewer
            safe = '_'.join(self.inp.text().strip().replace(',','').replace('/','_').split()[:3]) if self.inp.text().strip() else 'topographic'
            safe = ''.join(c for c in safe if c.isalnum() or c in '_- ').strip().replace(' ', '_')
            json_path = os.path.join(OUT_DIR, safe + '_topo.json')
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(result, jf, indent=2, ensure_ascii=False, default=str)
            
            # Find ortofoto
            jpg = glob.glob(os.path.join(OUT_DIR, '*ortofoto*.jpg'))
            jpg_path = jpg[0] if jpg else None
            
            gv_path = os.path.join(OUT_DIR, safe + '_topo_viewer.html')
            generate_geometry_viewer(json_path, jpg_path, gv_path)
            url = 'file:///' + gv_path.replace(chr(92), '/')
            webbrowser.open(url)
            self.lg.append('')
            self.lg.append('Opened: ' + gv_path)
        except Exception as e:
            import traceback
            self.lg.append('ERROR: ' + str(e))
            self.lg.append(traceback.format_exc())
        finally:
            self.topo_btn.setEnabled(True)


def register_plugin(main_window):
    mb = main_window.menu_bar
    mb.tools_menu.addSeparator()
    a = mb.tools_menu.addAction('Download LiDAR (ZBGIS)...')
    a.triggered.connect(lambda: LidarDialog(main_window).show())
    print('LiDAR plugin registered: Tools > Download LiDAR (ZBGIS)...')
