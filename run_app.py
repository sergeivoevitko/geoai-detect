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
import subprocess
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

RASTER_TYPES = ("Raster images (*.tif;*.tiff;*.jp2;*.img;*.vrt;*.png;*.jpg)", "All files (*.*)")


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


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
    """Functions the web page can call through window.pywebview.api (native dialogs).

    Keep non-function attributes private (leading underscore): pywebview walks public attributes
    recursively, and walking into the window object deadlocks the Windows UI thread."""

    def __init__(self):
        self._window = None

    def open_file(self):
        import webview
        r = self._window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=RASTER_TYPES)
        return r[0] if r else None

    def choose_folder(self):
        import webview
        r = self._window.create_file_dialog(webview.FOLDER_DIALOG)
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


LOADING_HTML = """<html><body style="font-family:Segoe UI,sans-serif;display:flex;align-items:center;
justify-content:center;height:100vh;margin:0;background:#eef0f3;color:#1d2330">
<div style="text-align:center"><h2>GeoAI Detect</h2><p>Starting the AI engine&hellip; (first start can take a minute)</p></div>
</body></html>"""


def start_server(port: int) -> subprocess.Popen:
    """The AI engine runs in its own process, so the window never freezes while it works."""
    os.makedirs(os.path.join(ROOT, "workspace"), exist_ok=True)
    log = open(os.path.join(ROOT, "workspace", "server.log"), "w", encoding="utf-8", errors="replace")
    flags = 0x08000000 if sys.platform == "win32" else 0          # CREATE_NO_WINDOW
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1",
                             "--port", str(port), "--log-level", "warning"],
                            cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT,
                            creationflags=flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", action="store_true")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    a = ap.parse_args()
    os.environ["GEOAI_DEVICE"] = a.device
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # Windows without developer mode
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

    server = start_server(port)
    try:
        if use_window:
            import webview
            api = DesktopApi()
            api._window = webview.create_window("GeoAI Detect", html=LOADING_HTML, js_api=api, width=1500,
                                               height=950, min_size=(1000, 650), text_select=True)

            def when_ready():
                if wait_ready(url, timeout=600):
                    api._window.load_url(url)
                else:
                    api._window.load_html("<h3 style='font-family:sans-serif'>The AI engine did not start. "
                                         "Send the file workspace\\server.log from the app folder.</h3>")
            webview.start(when_ready)        # window runs on this thread; engine in its own process
        else:
            if not wait_ready(url, timeout=600):
                print("Server did not start; see workspace/server.log")
                sys.exit(1)
            import webbrowser
            webbrowser.open(url)
            print(f"GeoAI Detect is running at {url}  (Ctrl+C to stop)")
            while server.poll() is None:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(10)
            except Exception:
                server.kill()


if __name__ == "__main__":
    main()
