from __future__ import annotations

import os
import socket
import threading
import time
import urllib.error
import urllib.request
import webbrowser

import uvicorn

from .api import app
from .runtime import load_settings


HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def find_free_localhost_port(preferred_port: int = DEFAULT_PORT) -> int:
    if preferred_port > 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((HOST, preferred_port))
                return preferred_port
            except OSError:
                pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def _wait_for_health(url: str, timeout_seconds: float = 20.0) -> bool:
    health_url = f"{url.rstrip('/')}/api/health"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.5) as response:
                if 200 <= int(response.status) < 300:
                    return True
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(0.25)
    return False


def _open_browser_after_start(url: str) -> None:
    if not _wait_for_health(url):
        return
    try:
        webbrowser.open(url)
    except Exception:
        # Keep packaged startup resilient when browser launch is blocked.
        pass


def main() -> int:
    settings = load_settings()
    preferred_port = int(os.getenv("INTEL_LINT_PORT", str(DEFAULT_PORT)).strip() or DEFAULT_PORT)
    port = find_free_localhost_port(preferred_port=preferred_port)
    app_url = f"http://{HOST}:{port}/"

    print(f"URL: {app_url}")
    print(f"Data dir: {settings['data_dir']}")
    print(f"Engine: {settings['engine']} | Model: {settings['model']}")

    browser_thread = threading.Thread(target=_open_browser_after_start, args=(app_url,), daemon=True)
    browser_thread.start()

    uvicorn.run(
        app,
        host=HOST,
        port=port,
        access_log=False,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
