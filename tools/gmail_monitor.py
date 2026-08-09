#!/usr/bin/env python3
"""
Gmail/ZBGIS download monitor for RoofAI Pipeline.

Two modes:
1. Gmail API (autoclaw-productivity) — clean, OAuth-based
2. IMAP fallback — direct, needs app-specific password

Usage:
    python gmail_monitor.py --setup        # Show setup instructions
    python gmail_monitor.py --check        # Check for ZBGIS download emails
    python gmail_monitor.py --watch 120    # Watch for 120 seconds, download when found
"""
import sys, os, json, time, urllib.request, urllib.parse, ssl, re, zipfile, io, imaplib, email
from email.header import decode_header

# Config
EMAIL_ADDRESS = 'jangrexa@gmail.com'
DOWNLOAD_DIR = r'C:\Users\jangr\Downloads\ExportMB'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ZBGIS email patterns
ZBGIS_PATTERNS = [
    r'noreply@skgeodesy\.sk',
    r'@skgeodesy\.sk',
    r'@gku\.sk',
    r'stiahnut.*dát',
    r'export.*dát',
    r'ZBGIS.*export',
    r'MAPKA.*export',
    r'odkaz na stiahnutie',
    r'download.*link',
    r'lls.*export',
    r'mračno.*bodov',
]

def show_setup():
    """Show Gmail setup instructions."""
    print("""
Gmail Integration Setup
=======================

Option A: autoclaw-productivity Gmail (recommended)
  1. Open AutoClaw settings
  2. Go to Productivity → Gmail
  3. Click "Connect" and authorize
  4. Done! Use --check to test.

Option B: IMAP App Password
  1. Go to https://myaccount.google.com/apppasswords
  2. Generate app password for "Mail" on "Windows Computer"
  3. Set env var: set GMAIL_APP_PASSWORD=xxxx
  4. Run with --imap flag

Option C: Manual download monitoring
  - Script monitors Downloads folder for new LAZ ZIPs
  - You manually download from MAPKA email link
  - Script auto-detects and processes new files
""")

def check_gmail_api():
    """Check Gmail via autoclaw API — returns list of download links."""
    print('[API] Gmail check not available — needs autoclaw-productivity connection.')
    print('  Use --setup for instructions.')
    return []

def check_imap():
    """Check Gmail via IMAP for ZBGIS download links."""
    app_password = os.environ.get('GMAIL_APP_PASSWORD', '')
    if not app_password:
        print('[IMAP] GMAIL_APP_PASSWORD not set. Run with --setup for instructions.')
        return []
    
    print(f'[IMAP] Connecting as {EMAIL_ADDRESS}...')
    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
        mail.login(EMAIL_ADDRESS, app_password)
        mail.select('INBOX')
        
        # Search for recent ZBGIS emails (last 3 days)
        _, messages = mail.search(None, 'UNSEEN', 'SINCE', 
                                   time.strftime('%d-%b-%Y', time.gmtime(time.time()-86400*3)))
        
        links = []
        mail_ids = messages[0].split()
        print(f'[IMAP] Found {len(mail_ids)} recent unread emails')
        
        for mid in reversed(mail_ids[-20:]):  # Check last 20
            _, msg_data = mail.fetch(mid, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_header_str(msg['subject'])
                    sender = msg['from']
                    
                    # Check if ZBGIS-related
                    body = get_email_body(msg)
                    is_zbgis = any(
                        re.search(pat, f'{sender} {subject} {body[:500]}', re.I)
                        for pat in ZBGIS_PATTERNS
                    ) or 'zbgis' in sender.lower() or 'gku' in sender.lower() or 'skgeodesy' in sender.lower()
                    
                    if is_zbgis:
                        print(f'\n  ZBGIS Email Found!')
                        print(f'  From: {sender}')
                        print(f'  Subject: {subject}')
                        
                        # Extract download links
                        urls = re.findall(r'https?://[^\s<>"]+', body)
                        for url in urls:
                            if any(ext in url.lower() for ext in ['.zip', '.laz', 'download', 'export', 'stiahn']):
                                print(f'  Download link: {url}')
                                links.append(url)
        
        mail.logout()
        return links
    except Exception as e:
        print(f'[IMAP] Error: {e}')
        return []

def get_email_body(msg):
    """Extract body text from email message."""
    body = ''
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == 'text/plain':
                try:
                    body += part.get_payload(decode=True).decode('utf-8', errors='replace')
                except: pass
            elif ctype == 'text/html':
                try:
                    body += part.get_payload(decode=True).decode('utf-8', errors='replace')
                except: pass
    else:
        try:
            body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
        except: pass
    return body

def decode_header_str(header):
    """Decode email header to string."""
    if header is None: return ''
    parts = decode_header(header)
    result = ''
    for part, charset in parts:
        if isinstance(part, bytes):
            result += part.decode(charset or 'utf-8', errors='replace')
        else:
            result += part
    return result

def download_file(url, save_dir):
    """Download file from URL to directory."""
    filename = url.split('/')[-1].split('?')[0]
    if not filename.endswith('.zip'):
        filename = f'export_{int(time.time())}.zip'
    
    path = os.path.join(save_dir, filename)
    print(f'  Downloading: {url[:100]}...')
    
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = False
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
        data = r.read()
    
    with open(path, 'wb') as f:
        f.write(data)
    print(f'  Saved: {path} ({len(data)/1024/1024:.1f} MB)')
    
    # Auto-extract if ZIP
    if path.endswith('.zip'):
        extract_dir = save_dir
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            laz_files = [n for n in names if n.endswith('.laz')]
            print(f'  Extracting {len(laz_files)} LAZ files...')
            zf.extractall(extract_dir)
            for lf in laz_files:
                print(f'    {lf}')
    
    return path

def watch_mode(timeout_seconds=120):
    """Watch for new ZBGIS emails and auto-download."""
    print(f'[WATCH] Monitoring for {timeout_seconds}s...')
    print(f'  Email: {EMAIL_ADDRESS}')
    print(f'  Waiting for ZBGIS MAPKA export email...')
    
    start = time.time()
    check_count = 0
    attempts_api = 0
    attempts_imap = 0
    
    while time.time() - start < timeout_seconds:
        check_count += 1
        elapsed = int(time.time() - start)
        
        # Try IMAP first (faster)
        links = check_imap()
        attempts_imap += 1
        if links:
            for link in links:
                download_file(link, DOWNLOAD_DIR)
            print(f'\n[DONE] Downloaded {len(links)} files in {elapsed}s')
            return True
        
        # If autoclaw API available, use it too
        api_links = check_gmail_api()
        attempts_api += 1
        if api_links:
            for link in api_links:
                download_file(link, DOWNLOAD_DIR)
            print(f'\n[DONE] Downloaded {len(api_links)} files via API in {elapsed}s')
            return True
        
        time.sleep(10)  # Poll every 10 seconds
    
    print(f'\n[TIMEOUT] No ZBGIS email received in {timeout_seconds}s')
    print(f'  Checked {check_count} times (IMAP: {attempts_imap})')
    return False

if __name__ == '__main__':
    if '--setup' in sys.argv:
        show_setup()
    elif '--check' in sys.argv:
        links = check_imap() or check_gmail_api()
        if links:
            print(f'\nFound {len(links)} download links!')
            action = input('Download now? (y/n): ')
            if action.lower() == 'y':
                for link in links:
                    download_file(link, DOWNLOAD_DIR)
    elif '--watch' in sys.argv:
        timeout = int(sys.argv[sys.argv.index('--watch')+1]) if len(sys.argv) > sys.argv.index('--watch')+1 else 120
        watch_mode(timeout)
    elif '--imap' in sys.argv:
        links = check_imap()
        if links:
            for link in links:
                download_file(link, DOWNLOAD_DIR)
    else:
        show_setup()
