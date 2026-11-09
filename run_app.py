#!/usr/bin/env python
"""Start GeoAI Detect as a desktop application.

Opens the work area in its own window (Windows: Edge WebView2, built into Windows 10/11).
If pywebview is not installed, the work area opens in the default web browser instead.

    python run_app.py                 # desktop window
    python run_app.py --browser       # use the web browser
    python run_app.py --device cuda   # force GPU / cpu
"""
import argparse
import os
import socket
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

RASTER_TYPES = ("Raster images (*.tif;*.tiff;*.jp2;*.img;*.vrt;*.png;*.jpg)", "All files (*.*)")


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def serve(port: int):
    import uvicorn
    uvicorn.run("server.app:app", host="127.0.0.1", port=port, log_level="warning")


def wait_ready(url: str, timeout: float = 180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url + "/api/config", timeout=2)
            return True
        except Exception:
            time.sleep(0.3)
    return False


class DesktopApi:
    """Functions the web page can call through window.pywebview.api (native dialogs)."""

    def __init__(self):
        self.window = None

    def open_file(self):
        import webview
        r = self.window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=RASTER_TYPES)
        return r[0] if r else None

    def choose_folder(self):
        import webview
        r = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        return r[0] if r else None

    def open_folder(self, path):
        if os.path.isdir(path) or os.path.exists(path):
            target = path if os.path.isdir(path) else os.path.dirname(path)
            if sys.platform == "win32":
                os.startfile(target)  # noqa
            elif sys.platform == "darwin":
                os.system(f'open "{target}"')
            else:
                os.system(f'xdg-open "{target}"')
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", action="store_true")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    a = ap.parse_args()
    os.environ["GEOAI_DEVICE"] = a.device
    port = a.port or free_port()
    url = f"http://127.0.0.1:{port}"

    use_window = not a.browser
    if use_window:
        try:
            import webview  # noqa: F401
        except Exception:
            print("pywebview not installed -> opening in the web browser")
            use_window = False
    os.environ["GEOAI_DESKTOP"] = "1" if use_window else "0"

    threading.Thread(target=serve, args=(port,), daemon=True).start()
    if not wait_ready(url):
        print("Server did not start; see the console output above.")
        sys.exit(1)

    if use_window:
        import webview
        api = DesktopApi()
        api.window = webview.create_window("GeoAI Detect", url, js_api=api, width=1500, height=950,
                                           min_size=(1000, 650), text_select=True)
        webview.start()            # blocks until the window is closed
    else:
        import webbrowser
        webbrowser.open(url)
        print(f"GeoAI Detect is running at {url}  (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
