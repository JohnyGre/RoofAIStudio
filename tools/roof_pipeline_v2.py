#!/usr/bin/env python3
"""
RoofAI Pipeline v2 — Full Auto Mode
====================================
Address → GPS → LAZ (auto-download from Gmail) → 3D Mesh → Metrics → Orthophoto

Setup (first time only):
  1. Generate Gmail app password: https://myaccount.google.com/apppasswords
     Select app: "Mail", device: "Windows Computer"
  2. Set environment: setx GMAIL_APP_PASSWORD "your-16-char-password"
  3. Restart terminal

Usage:
  python roof_pipeline_v2.py "Slnečná 988/60, 917 01 Trnava"
  python roof_pipeline_v2.py --skip-gmail "Koperníkova 7004/38, Trnava"
"""
import sys, os, json, time, re, ssl, urllib.request, urllib.parse
import zipfile, io, imaplib, email, glob
import numpy as np
from email.header import decode_header
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import Delaunay

# ============================================================
# CONFIGURATION
# ============================================================
EMAIL = 'jangrexa@gmail.com'
LAZ_DIR = r'C:\Users\jangr\Downloads\ExportMB'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DELIVERY')
os.makedirs(LAZ_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: GEOCODE
# ============================================================
def geocode(address):
    url = f'https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(address)}&format=json&limit=1'
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = False
    req = urllib.request.Request(url, headers={'User-Agent': 'RoofAI/2.0'})
    with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
        data = json.loads(r.read())
    if not data: raise ValueError(f'Not found: {address}')
    lat, lon = float(data[0]['lat']), float(data[0]['lon'])
    display = data[0].get('display_name', address)
    print(f'[OK] GPS: {lat:.7f}N, {lon:.7f}E  ({display})')
    return lat, lon, display

# ============================================================
# STEP 2: LAZ CHECK + GMAIL AUTO-DOWNLOAD
# ============================================================
def assert_laz_coverage(lat, lon, skip_gmail=False):
    t = Transformer.from_crs('EPSG:4326', 'EPSG:8353', always_xy=True)
    e, n = t.transform(lon, lat)
    
    existing = []
    for f in glob.glob(os.path.join(LAZ_DIR, '*.laz')):
        if '.copc.' in f: continue
        try:
            import laspy; las = laspy.read(f)
            xs, ys = np.array(las.x), np.array(las.y)
            if xs.min() <= e <= xs.max() and ys.min() <= n <= ys.max():
                existing.append(os.path.basename(f))
        except: pass
    
    if existing:
        print(f'[OK] LAZ coverage: {len(existing)} file(s)')
        return True
    
    # No coverage — print instructions and optionally monitor Gmail
    print(f'\n{"="*60}')
    print('NO LAZ COVERAGE — DOWNLOAD REQUIRED')
    print(f'{"="*60}')
    print(f'S-JTSK: E={e:.0f}, N={n:.0f}')
    print(f'''
MAPKA EXPORT STEPS:
  1. Open https://zbgis.skgeodesy.sk/mapka/
  2. Menu → Témy → Terén
  3. Draw polygon around GPS: {lat:.6f}N, {lon:.6f}E
  4. Export → Klasifikované mračno bodov (LLS) → LAZ
  5. Enter email: {EMAIL}
  6. Click "Odoslať" / "Generovať"
''')
    
    if skip_gmail:
        print('[PAUSE]  Skipping Gmail monitor (--skip-gmail). Save LAZ files to:')
        print(f'   {LAZ_DIR}')
        print(f'   Then re-run this script.')
        return False
    
    # Auto-monitor Gmail
    app_pw = os.environ.get('GMAIL_APP_PASSWORD', '')
    if not app_pw:
        print('[WARN]️  GMAIL_APP_PASSWORD not set. Cannot auto-download.')
        print('   See: https://myaccount.google.com/apppasswords')
        print(f'   Or save LAZ to: {LAZ_DIR} and re-run.')
        return False
    
    print(f'\n[EMAIL] MONITORING GMAIL for ZBGIS download link...')
    print(f'   Account: {EMAIL}')
    print(f'   Waiting up to 5 minutes...')
    print(f'   (Complete MAPKA export now!)')
    
    start = time.time()
    while time.time() - start < 300:
        try:
            mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
            mail.login(EMAIL, app_pw)
            mail.select('INBOX')
            
            # Search for recent unread emails from common ZBGIS domains or with download keywords
            for query in ['FROM skgeodesy.sk', 'FROM gku.sk', 'SUBJECT stiahnut', 
                         'SUBJECT export', 'SUBJECT download', 'SUBJECT ZBGIS', 
                         'SUBJECT MAPKA', 'TEXT "odkaz na stiahnutie"', 'TEXT "stiahnutie dát"']:
                try:
                    _, msgs = mail.search(None, '(UNSEEN)', query)
                    if msgs[0]:
                        break
                except: pass
            
            mail_ids = msgs[0].split() if msgs[0] else []
            if not mail_ids:
                # Also check SEEN from last hour
                _, msgs = mail.search(None, f'SINCE {time.strftime("%d-%b-%Y")}', 
                                     'OR FROM skgeodesy.sk FROM gku.sk')
                mail_ids = msgs[0].split() if msgs[0] else []
            
            download_links = []
            for mid in reversed(mail_ids[-10:]):
                _, msg_data = mail.fetch(mid, '(RFC822)')
                for resp in msg_data:
                    if isinstance(resp, tuple):
                        msg = email.message_from_bytes(resp[1])
                        body = get_body(msg)
                        urls = re.findall(r'https?://[^\s<>"\']+', body)
                        for url in urls:
                            if any(x in url.lower() for x in ['stiahn', 'download', 'export', '.zip', 'zbgis']):
                                download_links.append(url)
            
            mail.logout()
            
            if download_links:
                print(f'\n[OK] Found {len(download_links)} download link(s)!')
                for url in download_links:
                    print(f'  {url[:120]}')
                    filename = download_laz_zip(url, LAZ_DIR)
                    if filename:
                        return True
            
            elapsed = int(time.time() - start)
            print(f'\r   Waiting... {elapsed}s elapsed', end='', flush=True)
            time.sleep(10)
            
        except Exception as ex:
            print(f'\n   IMAP error: {ex}')
            time.sleep(15)
    
    print(f'\n[TIMEOUT] Timeout — no download email received.')
    return False

def get_body(msg):
    body = ''
    if msg.is_multipart():
        for part in msg.walk():
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode('utf-8', errors='replace')
            except: pass
    else:
        try:
            body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
        except: pass
    return body

def download_laz_zip(url, target_dir):
    try:
        filename = f'export_{int(time.time())}.zip'
        path = os.path.join(target_dir, filename)
        
        ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = False
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=180, context=ctx) as r:
            data = r.read()
        
        with open(path, 'wb') as f: f.write(data)
        print(f'  Downloaded: {path} ({len(data)/1024/1024:.1f} MB)')
        
        # Extract
        with zipfile.ZipFile(path) as zf:
            laz_files = [n for n in zf.namelist() if n.endswith('.laz') and '.copc.' not in n]
            zf.extractall(target_dir)
            for lf in laz_files:
                print(f'    [OK] {lf}')
        
        return filename
    except Exception as e:
        print(f'  Download failed: {e}')
        return None

# ============================================================
# STEP 3: EXTRACT BUILDING
# ============================================================
def extract_building(lat, lon):
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
            dist = np.sqrt((xs-te)**2 + (ys-tn)**2)
            n = dist < 60
            bm, gm = n & (cls==6), n & (cls==2)
            if np.any(bm): bp_list.append(np.column_stack([xs[bm], ys[bm], zs[bm]]))
            if np.any(gm): gp_list.append(np.column_stack([xs[gm], ys[gm], zs[gm]]))
        except Exception as e:
            print(f'  Warning reading {os.path.basename(f)}: {e}')
    
    if not bp_list:
        return None, None, None, None
    
    bp = np.vstack(bp_list)
    gp = np.vstack(gp_list)
    gmed = float(np.median(gp[:, 2]))
    
    # Cluster
    gr = 0.5
    xi = ((bp[:,0]-bp[:,0].min())/gr).astype(int)
    yi = ((bp[:,1]-bp[:,1].min())/gr).astype(int)
    grid = np.zeros((yi.max()+2, xi.max()+2), dtype=bool)
    grid[yi, xi] = True
    labeled, nc = ndimage.label(grid)
    
    best = None
    best_pts = None
    for cl in range(1, nc+1):
        mask = labeled == cl
        area = np.sum(mask)*0.25
        if area < 30: continue
        rows, cols = np.where(mask)
        cx = bp[:,0].min() + np.mean(cols)*gr
        cy = bp[:,1].min() + np.mean(rows)*gr
        dist = np.sqrt((cx-te)**2 + (cy-tn)**2)
        if dist > 30: continue
        r = max(area**0.5,8)*0.8
        in_c = (np.abs(bp[:,0]-cx)<r) & (np.abs(bp[:,1]-cy)<r)
        pts = bp[in_c]
        zmin, zmax = float(np.min(pts[:,2])), float(np.max(pts[:,2]))
        if best is None or dist < best['dist']:
            clon, clat = t_inv.transform(cx, cy)
            best = {'area':area, 'zmin':zmin, 'zmax':zmax, 'n':len(pts), 'dist':dist, 'lat':clat, 'lon':clon}
            best_pts = pts
    
    return best_pts, gmed, best, t_inv

# ============================================================
# STEP 4: CREATE 3D MESH
# ============================================================
def create_mesh(points, ground_z, info, prefix, address):
    pts = points.copy()
    c = pts[:,:2].mean(axis=0)
    pts[:,0] -= c[0]; pts[:,1] -= c[1]; pts[:,2] -= ground_z
    
    xd = pts[:,0].max()-pts[:,0].min()
    yd = pts[:,1].max()-pts[:,1].min()
    zmax = pts[:,2].max()
    zmin = pts[:,2].min()
    zr = zmax-zmin
    
    try:
        tri = Delaunay(pts[:,:2])
    except:
        pts[:,0] += np.random.normal(0,0.01,len(pts))
        pts[:,1] += np.random.normal(0,0.01,len(pts))
        tri = Delaunay(pts[:,:2])
    
    eave_m = (pts[:,2]>0.5) & (pts[:,2]<zmax-zr*0.5)
    eave_z = float(np.median(pts[eave_m,2])) if np.sum(eave_m)>5 else zmin
    pitch = np.degrees(np.arctan((zmax-eave_z)/(yd/2))) if yd>0 else 0
    roof_type = 'flat' if zr<2 else 'low_pitch' if zr<4 else 'pitched'
    
    # Save
    ply = os.path.join(OUT_DIR, f'{prefix}.ply')
    with open(ply,'w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(tri.simplices)}\nproperty list uchar int vertex_indices\nend_header\n')
        for v in pts: f.write(f'{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n')
        for fc in tri.simplices: f.write(f'3 {fc[0]} {fc[1]} {fc[2]}\n')
    
    obj = os.path.join(OUT_DIR, f'{prefix}.obj')
    with open(obj,'w') as f:
        f.write(f'# {address}\n\n')
        for v in pts: f.write(f'v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n')
        for fc in tri.simplices: f.write(f'f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n')
    
    # Orthophoto
    t = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    x, y = t.transform(info['lon'], info['lat'])
    ext = 120
    params = {'SERVICE':'WMS','VERSION':'1.3.0','REQUEST':'GetMap','LAYERS':'1','CRS':'EPSG:3857',
              'BBOX':f'{x-ext},{y-ext},{x+ext},{y+ext}','WIDTH':'4096','HEIGHT':'4096',
              'FORMAT':'image/jpeg','STYLES':'default'}
    url = 'https://zbgisws.skgeodesy.sk/zbgis_ortofoto_wms/service.svc/get?' + urllib.parse.urlencode(params)
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = False
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        jpg = os.path.join(OUT_DIR, f'{prefix}_ortofoto.jpg')
        with open(jpg,'wb') as f: f.write(r.read())
    
    # Report
    print(f'\n{"="*60}')
    print(f'BUILDING: {address}')
    print(f'{"="*60}')
    print(f'  GPS:        {info["lat"]:.6f}N, {info["lon"]:.6f}E')
    print(f'  Footprint:  {xd:.1f} x {yd:.1f} m')
    print(f'  Height:     {zmax:.2f} m (ridge), {eave_z:.2f} m (eave)')
    print(f'  Pitch:      {pitch:.1f}°')
    print(f'  Type:       {roof_type}')
    print(f'  Area:       {info["area"]:.0f} m2')
    print(f'  Ground:     {ground_z:.2f} m n.m.')
    print(f'  Points:     {len(pts):,}')
    print(f'\nFiles:')
    print(f'  {ply}')
    print(f'  {obj}')
    print(f'  {jpg}')
    
    meta = {'address':address,'gps':{'lat':info['lat'],'lon':info['lon']},
            'footprint_m':{'length':round(xd,1),'width':round(yd,1)},
            'height_ridge_m':round(float(zmax),2),'height_eave_m':round(float(eave_z),2),
            'pitch_deg':round(float(pitch),1),'roof_type':roof_type,
            'area_m2':round(info['area'],0),'points':len(pts),
            'ground_mnm':round(float(ground_z),3)}
    with open(os.path.join(OUT_DIR,f'{prefix}_meta.json'),'w',encoding='utf-8') as f:
        json.dump(meta,f,indent=2,ensure_ascii=False)
    return meta

# ============================================================
# MAIN
# ============================================================
def main():
    skip_gmail = '--skip-gmail' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    
    if not args:
        print('RoofAI Pipeline v2')
        print('Usage: python roof_pipeline_v2.py "Address, City"')
        print('       python roof_pipeline_v2.py --skip-gmail "Address"')
        return
    
    address = args[0]
    print(f'RoofAI Pipeline v2')
    print(f'{"="*60}\n')
    
    # Step 1
    print('[1/4] Geocoding...')
    lat, lon, display = geocode(address)
    
    # Step 2
    print('\n[2/4] LAZ coverage...')
    if not assert_laz_coverage(lat, lon, skip_gmail):
        return
    
    # Step 3
    print('\n[3/4] Extracting building...')
    points, ground_z, info, _ = extract_building(lat, lon)
    if points is None:
        print('  ERROR: No building points found!')
        return
    print(f'  Building: {info["area"]:.0f} m2, {info["n"]:,} points')
    
    # Step 4
    print('\n[4/4] Creating 3D model + orthophoto...')
    safe = address.lower().replace(' ','_').replace('/','_').replace(',','')[:40]
    create_mesh(points, ground_z, info, safe, display)
    
    print(f'\n[OK] DONE! Outputs in: {OUT_DIR}')

if __name__ == '__main__':
    main()
