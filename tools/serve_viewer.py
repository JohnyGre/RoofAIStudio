"""
tools/serve_viewer.py
Spusti lokálny HTTP server a automaticky otvorí zadaný alebo najnovší HTML viewer v prehliadači.

Použitie:
    .venv\Scripts\python.exe tools/serve_viewer.py [nazov_suboru.html] [port]

Príklad:
    .venv\Scripts\python.exe tools/serve_viewer.py beluj_50_viewer.html 8080
"""

import http.server
import socketserver
import os
import sys
import webbrowser

PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
VIEWER_FILE = sys.argv[1] if len(sys.argv) > 1 else "beluj_50_viewer.html"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=OUTPUT_DIR, **kwargs)

    def log_message(self, format, *args):
        print(f"[HTTP] {self.address_string()} - {format % args}")

def main():
    target_url = f"http://localhost:{PORT}/{VIEWER_FILE}"
    print(f"============================================================")
    print(f"  RoofAIStudio Local Viewer Server")
    print(f"  Adresar: {OUTPUT_DIR}")
    print(f"  URL:     {target_url}")
    print(f"============================================================")
    print(f"Otváram prehliadač...")
    webbrowser.open(target_url)

    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Server beží na porte {PORT}. Ukonči stlačením Ctrl+C.")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer zastavený.")

if __name__ == "__main__":
    main()
