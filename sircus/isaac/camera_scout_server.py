"""Serve the static camera scout on loopback; no simulation or hardware commands."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def handler(data):
    here = Path(__file__).parent
    vendor = here.parent / "simulation/web/vendor"
    frontend = here / "camera_scout"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            fixed = {
                "/": frontend / "index.html",
                "/index.html": frontend / "index.html",
                "/scout.js": frontend / "scout.js",
                "/scout.css": frontend / "scout.css",
                "/scene.json": data / "scene.json",
            }
            if path in fixed:
                file = fixed[path]
            elif (
                path.startswith("/vendor/")
                and path.split("/")[-1] in ("three.module.js", "OrbitControls.js")
                and path.count("/") == 2
            ):
                file = vendor / path.split("/")[-1]
            elif (
                path.startswith("/meshes/")
                and path.count("/") == 2
                and Path(path).name.endswith(".stl")
            ):
                file = data / "meshes" / Path(path).name
            else:
                self.send_error(404)
                return
            if not file.is_file():
                self.send_error(404)
                return
            kind = {
                ".html": "text/html; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".json": "application/json",
                ".stl": "application/octet-stream",
            }[file.suffix]
            raw = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

    return Handler


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--port", type=int, default=18784)
    a = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), handler(a.data))
    print(f"Camera scout: http://127.0.0.1:{server.server_port}", flush=True)
    server.serve_forever()
