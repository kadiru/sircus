"""Separate loopback GET/HEAD viewer for the frozen-policy TRAIN comparison.

No control endpoint, simulator import, native launcher or existing-reader change.
Only static UI files and the explicit published report are served.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

STATIC = {
    "/": ("candidate_comparison.html", "text/html; charset=utf-8"),
    "/candidate_comparison_panel.js": ("candidate_comparison_panel.js", "text/javascript; charset=utf-8"),
}


def published_file(root, request):
    """The report is the entire output allowlist; raw files stay private."""
    name = unquote(urlsplit(request).path)
    if name != "/report.json":
        return None
    root = Path(root).resolve()
    path = root / "report.json"
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        return None
    return path


def handler(root):
    root = Path(root)

    class Viewer(BaseHTTPRequestHandler):
        def do_GET(self):
            self.content(False)

        def do_HEAD(self):
            self.content(True)

        def content(self, head):
            route = urlsplit(self.path).path
            if route in STATIC:
                name, mime = STATIC[route]
                path = Path(__file__).with_name(name)
            else:
                path = published_file(root, self.path)
                if path is None:
                    self.send_error(404)
                    return
                mime = "application/json; charset=utf-8"
            try:
                raw = path.read_bytes()
            except OSError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            if not head:
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    return Viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18802)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose an unprivileged loopback port")
    if published_file(args.output, "/report.json") is None:
        parser.error("An explicit nonsymlink published report is required")
    ThreadingHTTPServer(("127.0.0.1", args.port), handler(args.output)).serve_forever()


if __name__ == "__main__":
    main()
