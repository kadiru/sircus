"""Read-only, loopback-only viewer for one explicit continuous-pilot export.

No simulator, policy, hardware command or arbitrary filesystem endpoint exists.
This starts a new viewer; it never restarts an existing dashboard.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import re
from pathlib import Path


def resource(root, request):
    path = request.split("?", 1)[0]
    source = Path(__file__).parent
    fixed = {"/": (source/"continuous_pilot.html", "text/html; charset=utf-8"),
             "/continuous_pilot.js": (source/"continuous_pilot.js", "application/javascript"),
             "/report.json": (Path(root)/"report.json", "application/json")}
    if path in fixed:
        result = fixed[path]
    elif re.fullmatch(r"/(trials|trial-\d{3})\.mp4", path):
        result = (Path(root)/path[1:], "video/mp4")
    else:
        return None
    file, mime = result
    if not file.is_file() or file.is_symlink():
        return None
    return file, mime


def byte_range(header, size):
    if not header:
        return 0, size-1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()):
        raise ValueError("Single byte range required")
    start, end = match.groups()
    if start:
        start = int(start)
        end = min(int(end), size-1) if end else size-1
    else:
        length = int(end)
        if length <= 0:
            raise ValueError("Invalid suffix length")
        start, end = max(0, size-length), size-1
    if not 0 <= start <= end < size:
        raise ValueError("Unsatisfiable range")
    return start, end, True


def handler(root):
    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_content(head=True)

        def do_GET(self):
            self.send_content()

        def send_content(self, head=False):
            result = resource(root, self.path)
            if result is None:
                self.send_error(404)
                return
            file, mime = result
            size = file.stat().st_size
            try:
                start, end, partial = byte_range(self.headers.get("Range"), size)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end-start+1))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not head:
                try:
                    with file.open("rb") as source:
                        source.seek(start)
                        remaining = end-start+1
                        while remaining:
                            data = source.read(min(remaining, 1024*1024))
                            if not data:
                                break
                            self.wfile.write(data)
                            remaining -= len(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, *_):
            pass
    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18794)
    args = parser.parse_args()
    if not (args.output/"report.json").is_file():
        parser.error("A pilot report export is required")
    ThreadingHTTPServer(("127.0.0.1", args.port), handler(args.output)).serve_forever()
