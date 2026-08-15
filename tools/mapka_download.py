# -*- coding: utf-8 -*-
"""Stiahni MB1 LAZ - opraveny parser (text/plain cast)."""
import imaplib, os, re, time, zipfile, sys
from email.header import decode_header
from email.parser import BytesParser
from email.utils import parseaddr
import requests

user = "jangrexa@gmail.com"
pw = os.getenv("GMAIL_APP_PASSWORD", "")
LAZ_DIR = r"C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio\data\laz"
os.makedirs(LAZ_DIR, exist_ok=True)

mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
mail.login(user, pw)
mail.select("INBOX")
status, data = mail.search(None, "ON", time.strftime("%d-%b-%Y"))
ids = data[0].split()

target_url = None
for mid in reversed(ids):
    status, msg_data = mail.fetch(mid, "(RFC822)")
    msg = BytesParser().parsebytes(msg_data[0][1])
    subject = decode_header(msg.get("Subject", ""))[0]
    if isinstance(subject[0], bytes):
        subject = subject[0].decode(subject[1] or "utf-8", errors="replace")
    else:
        subject = str(subject[0])
    sender = parseaddr(msg.get("From", ""))[1]

    if "skgeodesy" in sender and "MB1" in subject:
        print(f"Email: {subject}")
        # Zbieraj text zo VŠETKÝCH častí
        all_text = []
        if msg.is_multipart():
            for part in msg.walk():
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    all_text.append(payload.decode(charset, errors="replace"))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                all_text.append(payload.decode(msg.get_content_charset() or "utf-8", errors="replace"))

        body = "\n".join(all_text)
        urls = re.findall(r"https?://[^\s\"'<>\)\]]+", body)
        print(f"URL: {len(urls)}")
        for u in urls:
            print(f"  {u[:160]}")
        for u in urls:
            if "Export/Result" in u:
                target_url = u
                break
        if target_url:
            break

mail.logout()

if not target_url:
    print("Ziadny download URL!")
    sys.exit(1)

print(f"\nStahujem: {target_url[:160]}")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Referer": "https://zbgis.skgeodesy.sk/mapka/sk/teren",
    "Accept": "application/json, text/plain, */*",
}
for attempt in range(6):
    try:
        r = requests.get(target_url, headers=headers, timeout=180, allow_redirects=True)
        print(f"Pokus {attempt+1}: {r.status_code} | CT: {r.headers.get('Content-Type')} | len: {len(r.content)}")
        if r.status_code == 200:
            ct = r.headers.get("Content-Type", "")
            if "zip" in ct or r.content[:2] == b"PK":
                out = os.path.join(LAZ_DIR, "mb1_atriova_9309.zip")
                with open(out, "wb") as f:
                    f.write(r.content)
                print(f"ZIP ulozeny: {out} ({len(r.content)/1e6:.1f} MB)")
                with zipfile.ZipFile(out) as zf:
                    print(f"  Obsah: {zf.namelist()}")
                    zf.extractall(LAZ_DIR)
                    for root, dirs, files in os.walk(LAZ_DIR):
                        for fn in files:
                            if fn.endswith((".laz", ".las")):
                                full = os.path.join(root, fn)
                                print(f"  ✅ LAZ: {full} ({os.path.getsize(full)/1e6:.1f} MB)")
                break
            elif "json" in ct:
                print(f"  JSON: {r.text[:800]}")
                break
            else:
                print(f"  Body: {r.content[:300]}")
        else:
            print(f"  Chyba: {r.text[:150]}")
    except Exception as e:
        print(f"  EXC: {e}")
    time.sleep(10)
