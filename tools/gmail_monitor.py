# -*- coding: utf-8 -*-
"""
Gmail monitoring pre MAPKA LAZ export.

Kontroluje IMAP schránku (jangrexa@gmail.com), hľadá email od ZBGIS/MAPKA
s odkazom na stiahnutie LAZ, stiahne ZIP a rozbalí .laz do data/laz/.

POUŽITIE:
    python tools/gmail_monitor.py              # jednorazová kontrola
    python tools/gmail_monitor.py --watch 300  # kontrola každých 300s
"""
from __future__ import annotations

import argparse
import imaplib
import json
import os
import re
import sys
import time
import zipfile
from email.header import decode_header
from email.parser import BytesParser
from email.utils import parseaddr
from typing import Optional

import requests

# Projektové cesty
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LAZ_DIR = os.path.join(_PROJECT, "data", "laz")
_ORTHO_DIR = os.path.join(_PROJECT, "data", "ortho")
_STATE_FILE = os.path.join(_PROJECT, "data", "cache", "gmail_monitor_state.json")

# Gmail
GMAIL_USER = "jangrexa@gmail.com"
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

# Filtre - hladame emaily ohladom exportu LLS/LAZ
KEYWORDS = ["lls", "laz", "las", "mračno", "mracno", "bodov", "export", "skgeodesy", "zbgis", "mapka", "úgkk", "ugkk", "geodetick"]
FROM_FILTERS = ["skgeodesy", "ugkk", "gku", "zbgis", "mapka"]

# Download timeout (LAZ moze byt velky)
DOWNLOAD_TIMEOUT = 300


def _decode_header_value(val) -> str:
    if not val:
        return ""
    parts = decode_header(val)
    out = ""
    for data, charset in parts:
        if isinstance(data, bytes):
            out += data.decode(charset or "utf-8", errors="replace")
        else:
            out += data
    return out


def _load_state() -> dict:
    if os.path.exists(_STATE_FILE):
        try:
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
    with open(_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _extract_urls(text: str):
    """Najdi vsetky URL v texte (https://...)."""
    if not text:
        return []
    return re.findall(r"https?://[^\s\"'<>\)\]]+", text)


def _is_lls_email(subject: str, body: str, sender: str) -> bool:
    """Je tento email o LAZ/LLS exporte?"""
    haystack = f"{subject} {body} {sender}".lower()
    from_ok = any(f in sender.lower() for f in FROM_FILTERS) or not FROM_FILTERS
    kw_hits = sum(1 for kw in KEYWORDS if kw in haystack)
    return from_ok and kw_hits >= 2


def _parse_body(msg) -> str:
    """Extrahuj textovu cast emailu (plain + html)."""
    texts = []

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        texts.append(payload.decode(charset, errors="replace"))
                except Exception:
                    pass
            elif ct == "text/html":
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        texts.append(payload.decode(charset, errors="replace"))
                except Exception:
                    pass
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                texts.append(payload.decode(charset, errors="replace"))
        except Exception:
            pass

    return "\n".join(texts)


def check_gmail_once(verbose: bool = True) -> bool:
    """
    Jednorazova kontrola Gmailu. Vrati True ak sa nasiel a stiahol novy LAZ.
    """
    if not GMAIL_APP_PASSWORD:
        print("ERROR: GMAIL_APP_PASSWORD nie je nastavena v environmente")
        return False

    state = _load_state()
    seen_ids = list(state.get("seen_ids", []))
    downloaded = list(state.get("downloaded", []))

    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        mail.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        mail.select("INBOX")
    except Exception as e:
        print(f"IMAP pripojenie zlyhalo: {e}")
        return False

    try:
        # Hladaj v poslednych 7 dnoch
        status, data = mail.search(None, 'SINCE', (time.strftime("%d-%b-%Y", time.localtime(time.time() - 7 * 86400))))
        if status != "OK":
            print("Search zlyhal")
            return False

        ids = data[0].split()
        print(f"Najdenych emailov za 7 dni: {len(ids)}")

        found_lls = False
        for mid in reversed(ids):  # najnovsie prve
            mid_str = mid.decode()
            if mid_str in seen_ids:
                continue

            status, msg_data = mail.fetch(mid, "(RFC822)")
            if status != "OK":
                continue

            msg = BytesParser().parsebytes(msg_data[0][1])
            subject = _decode_header_value(msg.get("Subject", ""))
            sender = _decode_header_value(msg.get("From", ""))
            _, sender_email = parseaddr(sender)
            body = _parse_body(msg)

            seen_ids.append(mid_str)

            if _is_lls_email(subject, body, sender_email):
                found_lls = True
                print(f"\n=== LLS EMAIL NÁJDENÝ ===")
                print(f"  Od: {sender_email}")
                print(f"  Predmet: {subject}")

                # Najdi URL odkazy
                urls = _extract_urls(body)
                print(f"  URL odkazov: {len(urls)}")
                for u in urls[:10]:
                    print(f"    {u[:150]}")

                # Stiahni ZIP z prvého vhodného odkazu
                for url in urls:
                    if any(k in url.lower() for k in ["download", "laz", "export", "stiah", "zip", "file", "attachment", "get"]):
                        print(f"\n  Stahujem: {url[:150]}")
                        ok = _download_and_extract(url, verbose=verbose)
                        if ok:
                            downloaded.append(mid_str)
                            print("  ✅ LAZ stiahnutý a rozbalený!")
                            break
                        else:
                            print("  ❌ Download zlyhal, skúšam ďalší odkaz")
                else:
                    # Ziadny download odkaz - skus stiahnut prilohu
                    print("  Žiadny download URL - hľadám prílohu...")
                    for part in msg.walk():
                        fn = part.get_filename()
                        if fn:
                            fn_decoded = _decode_header_value(fn)
                            print(f"    Príloha: {fn_decoded}")
                            if fn_decoded.lower().endswith((".zip", ".laz", ".las")):
                                payload = part.get_payload(decode=True)
                                if payload:
                                    out_path = os.path.join(_LAZ_DIR, fn_decoded)
                                    os.makedirs(_LAZ_DIR, exist_ok=True)
                                    with open(out_path, "wb") as f:
                                        f.write(payload)
                                    print(f"    ✅ Príloha uložená: {out_path}")
                                    if fn_decoded.endswith(".zip"):
                                        _extract_zip(out_path)
                                    downloaded.append(mid_str)
                                    break

        # Uloz stav
        state["seen_ids"] = seen_ids[-500:]
        state["downloaded"] = downloaded[-100:]
        _save_state(state)
        return found_lls

    finally:
        mail.logout()


def _download_and_extract(url: str, verbose: bool = True) -> bool:
    """Stiahni ZIP/LAZ z URL a rozbal."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (RoofAIStudio/2.0)"}
        if verbose:
            print(f"    GET {url[:120]}...")
        resp = requests.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()

        # Detekcia nazvu z URL
        fname = url.split("?")[0].split("/")[-1]
        if not fname or "." not in fname:
            fname = f"mapka_export_{int(time.time())}.zip"

        os.makedirs(_LAZ_DIR, exist_ok=True)
        out_path = os.path.join(_LAZ_DIR, fname)

        total = 0
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                total += len(chunk)
        print(f"    Uložené: {out_path} ({total/1e6:.1f} MB)")

        if fname.endswith(".zip"):
            return _extract_zip(out_path)
        return True

    except Exception as e:
        print(f"    Download chyba: {e}")
        return False


def _extract_zip(zip_path: str) -> bool:
    """Rozbal ZIP do data/laz/."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            print(f"    ZIP obsahuje {len(names)} súborov: {names[:5]}")
            os.makedirs(_LAZ_DIR, exist_ok=True)
            zf.extractall(_LAZ_DIR)
            # Vypis LAZ subory
            for root, dirs, files in os.walk(_LAZ_DIR):
                for fn in files:
                    if fn.endswith((".laz", ".las")):
                        full = os.path.join(root, fn)
                        print(f"    ✅ LAZ: {full} ({os.path.getsize(full)/1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"    ZIP extrakcia chyba: {e}")
        return False


def watch_loop(interval: int = 300):
    """Kontroluj email kazdych interval sekund."""
    print(f"Gmail monitor spustený — kontrola každých {interval}s")
    print(f"Čakám na LAZ export email od ZBGIS/MAPKA...")
    while True:
        try:
            found = check_gmail_once()
            if found:
                print("\n✅ LAZ stiahnutý! Monitor môže byť ukončený.")
                return
        except Exception as e:
            print(f"Chyba pri kontrole: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gmail monitoring pre MAPKA LAZ export")
    parser.add_argument("--watch", type=int, default=0, help="kontrola každých N sekúnd (0 = jednorazovo)")
    args = parser.parse_args()

    if args.watch > 0:
        watch_loop(args.watch)
    else:
        ok = check_gmail_once()
        sys.exit(0 if ok else 1)
