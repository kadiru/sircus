"""Loopback-only GET/HEAD viewer for a separately published benchmark report."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

STATIC = {'/': ('shot_benchmark.html','text/html; charset=utf-8'),
          '/shot_benchmark_panel.js': ('shot_benchmark_panel.js','text/javascript; charset=utf-8')}


def published_file(root, request):
    """Only explicit report and encoded-media allowlist; never raw-run browsing."""
    name = unquote(urlsplit(request).path)
    if name == '/report.json':
        p = Path(root)/'report.json'
    else:
        report = json.loads((Path(root)/'report.json').read_text())
        allowed = {'/'+v['file'] for r in report['runs'] for v in r['videos']
                   if v.get('available') and v.get('file')}
        if name not in allowed: return None
        p = Path(root)/name.lstrip('/')
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(Path(root).resolve()): return None
    return p


def byte_range(header, length):
    if header is None: return 0,length-1,False
    m = re.fullmatch(r'bytes=(\d*)-(\d*)',header)
    if not m or not any(m.groups()) or length <= 0: raise ValueError('Invalid range')
    first,last=m.groups()
    if not first:
        suffix=int(last)
        if suffix <= 0: raise ValueError('Invalid suffix')
        return max(0,length-suffix),length-1,True
    first=int(first);last=min(length-1,int(last)) if last else length-1
    if first > last or first >= length: raise ValueError('Range outside file')
    return first,last,True


def handler(root):
    root=Path(root)
    class Viewer(BaseHTTPRequestHandler):
        def do_GET(self): self.content(False)
        def do_HEAD(self): self.content(True)
        def content(self,head):
            route=urlsplit(self.path).path
            if route in STATIC:
                name,mime=STATIC[route];path=Path(__file__).with_name(name)
            else:
                try: path=published_file(root,self.path)
                except (OSError,ValueError,KeyError): path=None
                if path is None: self.send_error(404);return
                mime='application/json' if path.suffix=='.json' else 'video/mp4' if path.suffix=='.mp4' else 'video/webm'
            size=path.stat().st_size
            try: first,last,partial=byte_range(self.headers.get('Range'),size)
            except ValueError:
                self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('Accept-Ranges','bytes')
            if partial:self.send_header('Content-Range',f'bytes {first}-{last}/{size}')
            self.send_header('Content-Length',str(max(0,last-first+1)));self.end_headers()
            if head:return
            try:
                with path.open('rb') as stream:
                    stream.seek(first);remaining=last-first+1
                    while remaining>0:
                        block=stream.read(min(1024*1024,remaining))
                        if not block:break
                        self.wfile.write(block);remaining-=len(block)
            except (BrokenPipeError,ConnectionResetError):pass
    return Viewer


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=18797)
    args=parser.parse_args()
    if not (args.output/'report.json').is_file():parser.error('Explicit published report required')
    ThreadingHTTPServer(('127.0.0.1',args.port),handler(args.output)).serve_forever()
