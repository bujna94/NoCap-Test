#!/usr/bin/env python3
"""Serves the experiment dashboard on localhost:8080 (threaded to avoid hangs)."""
import http.server
import json
import os
import subprocess
import socketserver

PORT = 8080
SERVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root


def get_system_stats():
    """Get live GPU and RAM usage."""
    stats = {"gpu_util": None, "gpu_mem_used": None, "gpu_mem_total": None, "gpu_temp": None,
             "ram_used_gb": None, "ram_total_gb": None}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], timeout=5, text=True).strip()
        parts = [p.strip() for p in out.split(",")]
        stats["gpu_util"] = int(parts[0])
        stats["gpu_mem_used"] = int(parts[1])
        stats["gpu_mem_total"] = int(parts[2])
        stats["gpu_temp"] = int(parts[3])
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                k, v = line.split(":")
                info[k.strip()] = int(v.strip().split()[0])
            total = info.get("MemTotal", 0) / 1048576
            avail = info.get("MemAvailable", 0) / 1048576
            stats["ram_total_gb"] = round(total, 1)
            stats["ram_used_gb"] = round(total - avail, 1)
    except Exception:
        pass
    return stats


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVE_DIR, **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/system"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(get_system_stats()).encode())
            return
        return super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # suppress access logs


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    with ThreadedHTTPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Dashboard: http://localhost:{PORT}/dashboard/index.html")
        print(f"Serving from: {SERVE_DIR}")
        print("Press Ctrl+C to stop")
        httpd.serve_forever()
