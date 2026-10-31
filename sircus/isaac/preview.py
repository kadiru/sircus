"""Read-only loopback preview. Serves only named latest outputs, no controls."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = r"""<!doctype html><html><meta charset="utf-8"><title>Sircus · Isaac Lab</title>
<style>body{background:#10262b;color:#f9f3df;font:17px system-ui;margin:32px auto;max-width:1200px}h1{margin-bottom:4px}p{color:#b7d1d1}main{display:grid;grid-template-columns:2fr 1fr;gap:20px}img{width:100%;border-radius:12px}section{padding:20px;border:1px solid #376169;border-radius:15px}pre{white-space:pre-wrap;font-size:14px}b{color:#ffbb59}</style>
<h1>SIRCUS <span style="font-weight:400">/ Isaac Lab</span></h1><p>Two UF850 arms · rigid board mount · open tapered bucket · synthetic RGB-D</p>
<main><section><img id="rgb" alt="Latest synthetic RGB camera view"><p id="fresh">Waiting for simulator…</p></section><section><b>Depth camera</b><img id="depth" alt="Axial depth preview"><pre id="status"></pre><b>Learning</b><pre id="training">Stationary foundation</pre></section></main>
<p>Fixed-scene cooperative shot learning. Policy uses known simulation state; grasping, compliant mounting and perception-driven control remain future work. Bucket reported height: 175 mm; its interpretation is unconfirmed.</p>
<script>async function update(){try{const s=await(await fetch('/status.json',{cache:'no-store'})).json();const age=(Date.now()-Date.parse(s.timestamp_utc))/1000;document.getElementById('fresh').textContent=(age<5?'Live':'Stale')+' · frame '+s.frame+' · '+s.timestamp_utc;document.getElementById('rgb').src='/latest.jpg?t='+s.frame;document.getElementById('depth').src='/depth-preview.png?t='+s.frame;document.getElementById('status').textContent=s.gpu+'\n\n'+s.camera.valid_depth_pixels.toLocaleString()+' depth points\n640 × 480 · axial metres\n\n'+Object.entries(s.arms).map(([n,a])=>n+': '+a.joint_names.length+' joints').join('\n')+'\n\nJoint target error: '+s.max_joint_target_error_rad.toFixed(4)+' rad';;document.getElementById('training').textContent=s.training?JSON.stringify(s.training,null,2):'Stationary foundation';}catch(e){document.getElementById('fresh').textContent='Waiting for a valid simulator observation';}}setInterval(update,1000);update();</script></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--port", type=int, default=18860)
    a = p.parse_args()
    out = Path(a.output)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                payload = PAGE.encode()
                mime = "text/html; charset=utf-8"
            elif path in ("/latest.jpg", "/depth-preview.png", "/status.json"):
                f = out / path[1:]
                if not f.exists():
                    self.send_error(503)
                    return
                payload = f.read_bytes()
                mime = {
                    ".jpg": "image/jpeg",
                    ".png": "image/png",
                    ".json": "application/json",
                }[f.suffix]
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
