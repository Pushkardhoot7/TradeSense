#!/usr/bin/env python
"""
TradeSense V2 — Server Entry Point

Usage:
    python run.py                    # Development mode with auto-reload
    python run.py --prod             # Production mode (no reload)
    python run.py --port 8080        # Custom port
"""

import sys
import uvicorn


def main():
    reload = "--prod" not in sys.argv
    port = 8000
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            try:
                port = int(sys.argv[i + 1])
            except ValueError:
                pass

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import socket
    def _get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    local_ip = _get_local_ip()

    print(f"""
+----------------------------------------------------------+
|                    TradeSense V2                         |
|      Intelligent Market Insights                         |
|      Discrete Mathematics Powered Stock Analysis         |
+----------------------------------------------------------+
|  Local PC:   http://127.0.0.1:{port:<29}|
|  Any Device: http://{local_ip}:{port:<29}|
|  (Open 'Any Device' URL on phone / tablet on same Wi-Fi) |
|  API Health: http://127.0.0.1:{port}/api/health{'':<15}|
|  Mode:       {'Development (auto-reload)' if reload else 'Production':<29}|
+----------------------------------------------------------+
""")

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
