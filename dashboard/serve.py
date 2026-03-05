#!/usr/bin/env python3
"""Serve the experiment dashboard (threaded to avoid hangs)."""
import base64
import http.server
import json
import os
from pathlib import Path
import subprocess
import socketserver
from urllib.parse import parse_qs, urlsplit

PORT = int(os.environ.get("DASHBOARD_PORT", os.environ.get("PORT", "8081")))
SERVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
EXPERIMENTS_JSON = Path(SERVE_DIR) / "experiments.json"
EXPERIMENTS_DIR = Path(SERVE_DIR) / "experiments"
RERUN_QUEUE_JSON = Path(SERVE_DIR) / "rerun_queue.json"
IDEA_MD = Path(SERVE_DIR) / "IDEA.md"
INDEX_HTML = Path(SERVE_DIR) / "dashboard" / "index.html"
DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "admin")
DASHBOARD_PASS = os.environ.get("DASHBOARD_PASS", "nocap2026")
ALLOW_CREDENTIALS = os.environ.get("DASHBOARD_CORS_ALLOW_CREDENTIALS", "1").lower() in {"1", "true", "yes"}


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


def check_auth(handler):
    """Check HTTP Basic Auth. Returns True if authorized."""
    auth_header = handler.headers.get("Authorization")
    if auth_header and auth_header.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            user, passwd = decoded.split(":", 1)
            if user == DASHBOARD_USER and passwd == DASHBOARD_PASS:
                return True
        except Exception:
            pass
    handler.send_response(401)
    handler.send_header("WWW-Authenticate", 'Basic realm="Dashboard"')
    handler.send_header("Content-Type", "text/html")
    handler.end_headers()
    handler.wfile.write(b"Unauthorized")
    return False


def load_experiments_payload():
    if not EXPERIMENTS_JSON.exists():
        return {"target_val_loss": 3.3821, "baseline_time_seconds": None, "experiments": []}
    try:
        return json.loads(EXPERIMENTS_JSON.read_text())
    except Exception:
        return {"target_val_loss": 3.3821, "baseline_time_seconds": None, "experiments": []}


def load_rerun_queue():
    if not RERUN_QUEUE_JSON.exists():
        return []
    try:
        payload = json.loads(RERUN_QUEUE_JSON.read_text())
    except Exception:
        return []
    if isinstance(payload, list):
        return [str(x) for x in payload if isinstance(x, str)]
    return []


def load_idea_payload():
    if not IDEA_MD.exists():
        return {"content": "", "exists": False, "updated_at": None}
    try:
        stat = IDEA_MD.stat()
        return {
            "content": IDEA_MD.read_text(encoding="utf-8"),
            "exists": True,
            "updated_at": int(stat.st_mtime),
        }
    except Exception:
        return {"content": "", "exists": False, "updated_at": None}


def save_rerun_queue(queue):
    tmp = RERUN_QUEUE_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(queue, indent=2))
    tmp.replace(RERUN_QUEUE_JSON)


def enqueue_rerun(exp_name):
    queue = load_rerun_queue()
    if exp_name not in queue:
        queue.append(exp_name)
        save_rerun_queue(queue)
    return queue


def render_index_with_bootstrap():
    html = INDEX_HTML.read_text(encoding="utf-8")
    payload = {"experiments": load_experiments_payload(), "system": get_system_stats()}
    payload_json = json.dumps(payload).replace("</", "<\\/")
    marker = "</head>"
    inject = f'<script>window.__DASHBOARD_BOOTSTRAP__ = {payload_json};</script>\n'
    if marker in html:
        return html.replace(marker, inject + marker, 1).encode("utf-8")
    return (inject + html).encode("utf-8")


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVE_DIR, **kwargs)

    def do_OPTIONS(self):
        # Allow CORS preflight requests without auth challenge.
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if not check_auth(self):
            return

        parsed = urlsplit(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # Serve index with embedded JSON so dashboards still work when proxies
        # rewrite API paths to index.html.
        if path.endswith("/dashboard/index.html") and "json" not in query:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(render_index_with_bootstrap())
            return

        # Proxy-proof fallback: allow JSON over the known-working index route.
        if path.endswith("/dashboard/index.html") and "json" in query:
            kind = query.get("json", [""])[0]
            if kind == "system":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(get_system_stats()).encode())
                return
            if kind == "experiments":
                if not EXPERIMENTS_JSON.exists():
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error":"experiments.json not found"}')
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(EXPERIMENTS_JSON.read_bytes())
                return
            if kind == "idea":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(load_idea_payload()).encode())
                return

        # Additional proxy-proof fallback for setups that strip query params
        # but preserve path prefixes.
        if path.endswith("/dashboard/index.html/system"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(get_system_stats()).encode())
            return
        if path.endswith("/dashboard/index.html/experiments"):
            if not EXPERIMENTS_JSON.exists():
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"experiments.json not found"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(EXPERIMENTS_JSON.read_bytes())
            return
        if path.endswith("/dashboard/index.html/idea"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(load_idea_payload()).encode())
            return

        # Support both root and /dashboard-prefixed API paths for reverse proxies
        if path.startswith("/api/system") or path.startswith("/dashboard/api/system"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(get_system_stats()).encode())
            return

        if path.startswith("/experiments.json") or path.startswith("/dashboard/experiments.json"):
            if not EXPERIMENTS_JSON.exists():
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"experiments.json not found"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(EXPERIMENTS_JSON.read_bytes())
            return

        if path.startswith("/api/idea") or path.startswith("/dashboard/api/idea"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(load_idea_payload()).encode())
            return

        return super().do_GET()

    def do_POST(self):
        if not check_auth(self):
            return

        parsed = urlsplit(self.path)
        path = parsed.path
        if path in {"/api/rerun", "/dashboard/api/rerun"}:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            body = self.rfile.read(content_length) if content_length > 0 else b""
            try:
                payload = json.loads(body.decode("utf-8") if body else "{}")
            except Exception:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"invalid JSON body"}')
                return

            exp_name = str(payload.get("exp_name", "")).strip()
            if not exp_name:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"exp_name is required"}')
                return

            run_sh = EXPERIMENTS_DIR / exp_name / "run.sh"
            if not run_sh.exists():
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"experiment run.sh not found"}')
                return

            queue = enqueue_rerun(exp_name)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "queued": exp_name, "queue_size": len(queue)}).encode())
            return

        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error":"not found"}')

    def end_headers(self):
        origin = self.headers.get("Origin")
        if origin:
            # Reflect request origin so browser fetch with Authorization can pass CORS.
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            if ALLOW_CREDENTIALS:
                self.send_header("Access-Control-Allow-Credentials", "true")
        else:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, Accept, Cache-Control")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # suppress access logs


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    print(f"Auth: user={DASHBOARD_USER} (set DASHBOARD_USER/DASHBOARD_PASS env to change)")
    print(f"Port: {PORT} (override with DASHBOARD_PORT or PORT)")
    with ThreadedHTTPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Dashboard: http://localhost:{PORT}/dashboard/index.html")
        print(f"Serving from: {SERVE_DIR}")
        print("Press Ctrl+C to stop")
        httpd.serve_forever()
