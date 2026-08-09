#!/usr/bin/env python3
"""
MAPKA Export via direct API — no UI automation needed.
Uses Playwright to capture the API endpoint, then calls it directly.
"""
import asyncio, json, time, os, ssl, urllib.request, urllib.parse
import imaplib, email, re, zipfile
from playwright.async_api import async_playwright

EMAIL = 'jangrexa@gmail.com'
APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD', 'kxnzeoijbfrfaywh')
MAPKA_BASE = 'https://zbgis.skgeodesy.sk/mapka'

# Known API patterns (from earlier observations)
# Export generates a job, then sends result via email
# We need: POST /api/Export/Generate with polygon + email

async def mapka_export_v2(lat, lon, output_dir):
    """
    V2: Use Playwright to inject drawing + trigger export via ArcGIS JS API.
    More direct approach: access ArcGIS MapView and SketchViewModel.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1920, 'height': 1080})
        
        try:
            # Navigate with terrain theme + position
            url = f'{MAPKA_BASE}/#/teren?pos={lat},{lon},18'
            print(f'  Loading MAPKA...')
            await page.goto(url, timeout=30000, wait_until='networkidle')
            await asyncio.sleep(5)
            
            # Accept cookies
            try:
                await page.click('button:has-text("Pokračovať")', timeout=5000)
                await asyncio.sleep(2)
            except: pass
            
            # MONITOR network for API calls
            api_calls = []
            page.on('request', lambda req: _log_api(req, api_calls))
            page.on('response', lambda resp: _log_response(resp, api_calls))
            
            # Direct approach: use JS to access ArcGIS MapView
            # MAPKA stores the map view as a global or in a service
            js_draw = """
            (async function() {
                try {
                    // Try to find ArcGIS MapView via DOM
                    const views = document.querySelectorAll('.esri-view');
                    if (!views.length) return 'ERR: no esri-view found';
                    
                    // The MapView is typically attached as a property of the view div
                    // or accessible via the Angular component tree
                    
                    // Try window.__mapView or similar global
                    for (const key of Object.keys(window)) {
                        if (key.includes('map') || key.includes('view') || key.includes('esri')) {
                            const val = window[key];
                            if (val && val.map && val.graphics) {
                                // Found a MapView!
                                const view = val;
                                
                                // Add polygon
                                const d = 0.0006;
                                const polygon = {
                                    type: 'polygon',
                                    rings: [[
                                        [''' + str(lon) + ''' - d, ''' + str(lat) + ''' - d],
                                        [''' + str(lon) + ''' + d, ''' + str(lat) + ''' - d],
                                        [''' + str(lon) + ''' + d, ''' + str(lat) + ''' + d],
                                        [''' + str(lon) + ''' - d, ''' + str(lat) + ''' + d],
                                        [''' + str(lon) + ''' - d, ''' + str(lat) + ''' - d]
                                    ]]
                                };
                                
                                view.graphics.removeAll();
                                view.graphics.add({
                                    geometry: polygon,
                                    symbol: {
                                        type: 'simple-fill',
                                        color: [255, 0, 0, 0.3],
                                        outline: { color: [255, 0, 0], width: 2 }
                                    }
                                });
                                
                                await view.goTo(polygon);
                                return 'OK: polygon drawn via ' + key;
                            }
                        }
                    }
                    
                    // Try Angular's ng global
                    const ng = document.querySelector('[ng-version]');
                    if (ng && window.ng) {
                        try {
                            const comp = window.ng.getComponent(ng);
                            return 'FOUND: angular component: ' + (comp?.constructor?.name || 'unknown');
                        } catch(e) {
                            return 'ERR: ng.getComponent failed: ' + e.message;
                        }
                    }
                    
                    return 'ERR: no map view found in window globals';
                } catch(e) {
                    return 'ERR: ' + e.message;
                }
            })()
            """
            
            result = await page.evaluate(js_draw)
            print(f'  JS draw result: {result}')
            
            # If JS injection worked, now try to find export button
            await asyncio.sleep(2)
            await page.screenshot(path=os.path.join(output_dir, '..', 'output', 'mapka_debug2.png'))
            
            # Try clicking various export buttons
            export_selectors = [
                'button:has-text("Export")',
                'button:has-text("Sťahovanie")',
                '[aria-label*="Export"]',
                '[aria-label*="Stiahnut"]',
                '[title*="Export"]',
                'text=Export dát',
            ]
            
            for sel in export_selectors:
                els = page.locator(sel)
                cnt = await els.count()
                if cnt > 0:
                    print(f'  Found export: {sel} (count={cnt})')
                    await els.first.click()
                    await asyncio.sleep(2)
                    
                    # Check for email input
                    email_input = page.locator('input[type="email"], input[placeholder*="mail"], input[placeholder*="email"]')
                    if await email_input.count() > 0:
                        await email_input.first.fill(EMAIL)
                        await asyncio.sleep(0.5)
                        submit = page.locator('button:has-text("Odoslať"), button:has-text("OK"), button:has-text("Generovať")')
                        if await submit.count() > 0:
                            await submit.first.click()
                            print(f'  Export submitted to {EMAIL}')
                            break
            
            # Print captured API calls
            export_apis = [c for c in api_calls if 'Export' in c.get('url', '') or 'export' in c.get('url', '')]
            if export_apis:
                print(f'\n  Captured export API calls:')
                for c in export_apis:
                    print(f'    {c["method"]} {c["url"][:150]}')
            
            await browser.close()
            
            # Monitor Gmail
            print(f'\n  Waiting for email...')
            return await _wait_for_email(output_dir)
            
        except Exception as e:
            try:
                await page.screenshot(path=os.path.join(output_dir, '..', 'output', 'mapka_error2.png'))
            except: pass
            await browser.close()
            raise RuntimeError(f'MAPKA API export failed: {e}')

def _log_api(request, store):
    url = request.url
    if any(x in url for x in ['api/', 'Export', 'export', 'Generate', 'Job']):
        store.append({'method': request.method, 'url': url, 'type': 'request'})

def _log_response(response, store):
    url = response.url
    if any(x in url for x in ['api/', 'Export', 'export', 'Generate', 'Job']):
        try:
            body = response.text()
            if len(body) < 5000:
                store.append({'method': 'RESP', 'url': url, 'status': response.status, 'body': body[:500]})
        except: pass

def _wait_for_email(output_dir, timeout=180):
    print(f'  Monitoring {EMAIL}...')
    start = time.time()
    while time.time() - start < timeout:
        try:
            mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
            mail.login(EMAIL, APP_PASSWORD)
            mail.select('INBOX')
            _, msgs = mail.search(None, 'SINCE', time.strftime('%d-%b-%Y'), 'OR FROM skgeodesy.sk FROM gku.sk')
            ids = msgs[0].split() if msgs[0] else []
            for mid in reversed(ids[-5:]):
                _, msg_data = mail.fetch(mid, '(RFC822)')
                for resp in msg_data:
                    if isinstance(resp, tuple):
                        msg = email.message_from_bytes(resp[1])
                        body = ''
                        if msg.is_multipart():
                            for part in msg.walk():
                                try:
                                    p = part.get_payload(decode=True)
                                    if p: body += p.decode('utf-8', errors='replace')
                                except: pass
                        else:
                            try: body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
                            except: pass
                        urls = re.findall(r'https?://[^\s<>"\']+', body)
                        for url in urls:
                            if any(x in url.lower() for x in ['stiahn', 'download', 'export', '.zip', 'mapka/api']):
                                print(f'  Downloading: {url[:120]}...')
                                return _download(url, output_dir)
            mail.logout()
        except Exception as e:
            print(f'  Gmail: {e}')
        time.sleep(10)
    return False

def _download(url, output_dir):
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = False
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=300, context=ctx) as r:
        data = r.read()
    path = os.path.join(output_dir, f'mapka_{int(time.time())}.zip')
    with open(path, 'wb') as f: f.write(data)
    print(f'  {len(data)/1024/1024:.1f} MB -> {path}')
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith('.laz') and '.copc.' not in name:
                zf.extract(name, output_dir)
                print(f'    {name}')
    return True

def mapka_export(lat, lon, output_dir):
    return asyncio.run(mapka_export_v2(lat, lon, output_dir))
