"""
visualbondweb.__main__
~~~~~~~~~~~~~~~~~~~~~~
Entry point: start the FastAPI server and open the browser.

Usage:
    python -m visualbondweb          # default port 8000
    python -m visualbondweb --port 8080
    visualbond                       # if installed via pip
"""
import argparse
import socket
import sys
import threading
import time
import webbrowser

import uvicorn


def find_free_port(preferred: int) -> int:
    """Return `preferred` if free, otherwise the next available port."""
    for port in range(preferred, preferred + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("Could not find a free port in range "
                       f"{preferred}–{preferred + 99}.")


def wait_for_server(host: str, port: int, timeout: float = 10.0) -> bool:
    """Poll until the server accepts connections or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def open_browser(url: str, delay: float = 0.5) -> None:
    """Open the browser in a thread after a short delay."""
    def _open():
        if wait_for_server("127.0.0.1", int(url.split(":")[-1].rstrip("/")), timeout=15):
            time.sleep(delay)
            webbrowser.open(url)
        else:
            print(f"[visualbond] Server did not respond at {url}. "
                  "Open it manually in your browser.", file=sys.stderr)
    threading.Thread(target=_open, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="visualbond",
        description="Visualbond — web interface for spectrojotometer",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8000,
        help="TCP port to listen on (default: 8000; auto-increments if busy)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Start server without opening the browser",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload (development mode)",
    )
    args = parser.parse_args()

    port = find_free_port(args.port)
    url  = f"http://{args.host}:{port}"

    if port != args.port:
        print(f"[visualbond] Port {args.port} busy — using {port} instead.")

    print(f"[visualbond] Starting server at {url}")
    print(f"[visualbond] Press Ctrl+C to stop.")

    if not args.no_browser:
        open_browser(url)

    # Use the installed package's api module so that STATIC_DIR resolves
    # relative to the package, not the cwd.
    uvicorn.run(
        "visualbondweb.api:app",
        host=args.host,
        port=port,
        reload=args.reload,
        log_level="warning",   # keep console clean; errors still show
    )


if __name__ == "__main__":
    main()
