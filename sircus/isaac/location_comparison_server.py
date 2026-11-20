"""Loopback GET/HEAD viewer for an explicitly published location comparison.

The published report and its byte-verified encoded videos are the entire output
allowlist. Raw evidence and model files are never served. No control endpoint.
"""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from shot_benchmark_server import byte_range

SCHEMA = 'sircus-location-comparison-publication-v1'
STATIC = {
    '/': ('location_comparison.html', 'text/html; charset=utf-8'),
    '/location_comparison_panel.js': ('location_comparison_panel.js', 'text/javascript; charset=utf-8'),
    '/location_comparison.css': ('location_comparison.css', 'text/css; charset=utf-8'),
}


def _safe_file(root, relative):
    root = Path(root).absolute()
    if root.is_symlink():
        return None
    root = root.resolve()
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        return None
    if any(parent.is_symlink() for parent in path.parents if parent != root and parent.is_relative_to(root)):
        return None
    return path


def _metrics(row, count):
    return (isinstance(row, dict) and all(type(row.get(k)) is int and row[k] >= 0
            for k in ('requests', 'captures', 'clean', 'misses', 'rejected'))
            and row['requests'] == count and row['clean'] <= row['captures']
            and row['captures'] + row['misses'] + row['rejected'] == count)


def _results_complete(report):
    return (report.get('model_promoted') is False
            and all(_metrics(report.get('train', {}).get(model), 8960)
                    for model in ('original_usable', 'broader'))
            and all(_metrics(report.get('final_tests', {}).get(model, {}).get(phase), count)
                    for model in ('parent', 'original_usable', 'broader')
                    for phase, count in (('test_broad', 70), ('test_local', 25))))


def published_file(root, request):
    """Return only the report or one encoded video bound by an audited report."""
    name = unquote(urlsplit(request).path)
    report_path = _safe_file(root, 'report.json')
    if report_path is None:
        return None
    if name in ('/report.json', '/api/location-comparison'):
        return report_path
    if not re.fullmatch(r'/media/[A-Za-z0-9_-][A-Za-z0-9_.-]*\.mp4', name):
        return None
    if report_path.stat().st_size > 16 * 1024**2:
        return None
    report = json.loads(report_path.read_text())
    audit = report.get('audit', {})
    if not (report.get('schema') == SCHEMA and report.get('status') == 'completed_audited'
            and report.get('synthetic') is False and audit.get('status') == 'passed'
            and audit.get('complete') is True and audit.get('synthetic') is False
            and _results_complete(report)):
        return None
    media = [video.get('media') for video in report.get('videos', [])
             if isinstance(video, dict) and isinstance(video.get('media'), dict)
             and video['media'].get('url') == name]
    if len(media) != 1 or media[0].get('verified') is not True:
        return None
    duration = media[0].get('duration_s')
    if type(duration) not in (float, int) or not math.isfinite(duration) or duration <= 0:
        return None
    expected = media[0].get('sha256')
    if not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{64}', expected):
        return None
    path = _safe_file(root, name.lstrip('/'))
    if path is None:
        return None
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            checksum.update(block)
    return path if checksum.hexdigest() == expected else None


def handler(root):
    root = Path(root)

    class Viewer(BaseHTTPRequestHandler):
        def do_GET(self):
            self.content(False)

        def do_HEAD(self):
            self.content(True)

        def content(self, head):
            route = urlsplit(self.path).path
            try:
                if route in STATIC:
                    name, mime = STATIC[route]
                    path = Path(__file__).with_name(name)
                else:
                    path = published_file(root, self.path)
                    if path is None:
                        self.send_error(404)
                        return
                    mime = 'application/json; charset=utf-8' if path.name == 'report.json' else 'video/mp4'
                size = path.stat().st_size
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                self.send_error(404)
                return
            try:
                first, last, partial = byte_range(self.headers.get('Range'), size)
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.end_headers()
                return
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; media-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(max(0, last - first + 1)))
            if partial:
                self.send_header('Content-Range', f'bytes {first}-{last}/{size}')
            self.end_headers()
            if head:
                return
            try:
                with path.open('rb') as stream:
                    stream.seek(first)
                    remaining = last - first + 1
                    while remaining > 0:
                        block = stream.read(min(1024**2, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (OSError, BrokenPipeError, ConnectionResetError):
                pass

    return Viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--port', type=int, default=18803)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Choose an unprivileged loopback port')
    if published_file(args.output, '/report.json') is None:
        parser.error('An explicit nonsymlink published report is required')
    ThreadingHTTPServer(('127.0.0.1', args.port), handler(args.output)).serve_forever()


if __name__ == '__main__':
    main()
