"""Read-only loopback preview. Serves only named latest outputs, no controls."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = r"""<!doctype html><html><meta charset="utf-8"><title>Sircus · Isaac Lab</title>
<style>body{background:#10262b;color:#f9f3df;font:17px system-ui;margin:32px auto;max-width:1200px}h1{margin-bottom:4px}p{color:#b7d1d1}main{display:grid;grid-template-columns:2fr 1fr;gap:20px}img{width:100%;border-radius:12px}section{padding:20px;border:1px solid #376169;border-radius:15px}pre{white-space:pre-wrap;font-size:14px}b{color:#ffbb59}</style>
<h1>SIRCUS <span style="font-weight:400">/ Isaac Lab</span></h1><p>Two UF850 arms · rigid board mount · open tapered bucket · synthetic RGB-D</p>
<main><section><img id="rgb" alt="Latest synthetic RGB camera view"><p id="fresh">Waiting for simulator…</p></section><section><b>Depth camera</b><img id="depth" alt="Axial depth preview"><pre id="status"></pre><b>Learning</b><pre id="training">Stationary foundation</pre></section></main>
<p>Fixed-scene cooperative shot learning. Policy uses known simulation state; grasping, compliant mounting and perception-driven control remain future work. Bucket reported height: 175 mm; its interpretation is unconfirmed.</p>
<section><h2>Randomized perception validation</h2><p id="validationStatus">No randomized study yet</p><p>Frozen detector · paired lighting/noise conditions across seeded layouts · ground truth used only for scoring. Static snapshots, not drop success.</p><img id="validationChart" alt="Completed-study detection availability chart" style="display:none"><div style="overflow:auto"><table id="validationTable" style="width:100%;text-align:left;border-spacing:10px"></table></div><p>Cells: detections / observations; mean error when detected. Table = plane distance; other objects = center distance. Missing detections stay in the availability denominator.</p><p id="visibility"></p><details><summary>Ranges, error percentiles and exclusions</summary><pre id="validationDetails"></pre></details><img id="validationImage" alt="Most recent validation observation with estimates" style="max-width:640px"></section>
<section><h2>RGB-D perception</h2><p>Observed geometry from RGB + depth. Color/shape priors; synthetic camera calibration. No object ground truth is used by the estimator.</p><img id="perception" alt="RGB-D object estimates"><p id="perceptionFresh">Waiting for observations</p><pre id="objects"></pre><p>Capture outcome: unknown from perception. The RL scorer is separate and uses simulator truth.</p><details><summary>Simulator evaluation only</summary><pre id="evaluation"></pre></details></section>
<script>
async function validationUpdate(){
 try {
 const r=await(await fetch('/validation.json',{cache:'no-store'})).json();
 const age=(Date.now()-Date.parse(r.timestamp_utc))/1000;
 const chart=document.getElementById('validationChart');chart.style.display='none';if(r.state==='completed'){try{const m=await(await fetch('/validation-chart.json',{cache:'no-store'})).json();if(m.run===r.run){chart.onload=()=>{chart.style.display='block';};chart.onerror=()=>{chart.style.display='none';};chart.src='/validation-chart.png?t='+encodeURIComponent(r.run);}}catch(e){}}
 document.getElementById('validationStatus').textContent=r.state+' · '+r.completed_observations+'/'+r.planned_observations+' observations · seed '+r.seed+(r.current?' · '+r.current:'')+((r.state==='running'&&age>30)?' · STALE PROGRESS':'');
 const table=document.getElementById('validationTable');table.replaceChildren();
 const header=document.createElement('tr');for(const t of ['Condition','Table','Board','Bucket','Ball','Visible ball within 10 mm']){const cell=document.createElement('th');cell.textContent=t;header.appendChild(cell);}table.appendChild(header);
 for(const [name,g] of Object.entries(r.conditions||{})){
 const row=document.createElement('tr');const label=document.createElement('td');label.textContent=name;row.appendChild(label);
 for(const n of ['table','board','bucket','ball']){const o=g.objects[n];const cell=document.createElement('td');cell.textContent=o.estimated+'/'+o.total+' · '+(o.mean_mm===null?'—':o.mean_mm.toFixed(1)+' mm');row.appendChild(cell);}
 const score=document.createElement('td');score.textContent=g.ball_localized_10mm+'/'+g.ball_visible;row.appendChild(score);table.appendChild(row);
 }
 document.getElementById('visibility').textContent='Visible-ball denominator: at least 20 clean-depth pixels agreeing with the true sphere surface within 5 mm. This evaluation-only geometric label never enters the detector. Excluded layouts: '+(r.excluded_layouts||[]).length;
 document.getElementById('validationDetails').textContent=JSON.stringify(r,null,2);
 document.getElementById('validationImage').src='/validation-example.jpg?t='+r.completed_observations;
 }catch(e){document.getElementById('validationStatus').textContent='No randomized validation report available';}
}
setInterval(validationUpdate,1500);validationUpdate();
async function perceptionUpdate(){
 try {
 const p=await(await fetch('/perception.json',{cache:'no-store'})).json();
 const age=(Date.now()-Date.parse(p.timestamp_utc))/1000;
 document.getElementById('perceptionFresh').textContent=(age<3?'Fresh':'STALE — not current')+' · '+age.toFixed(1)+' s old · '+p.processing_ms.toFixed(0)+' ms processing';
 document.getElementById('perception').src='/perception.jpg?t='+encodeURIComponent(p.frame_id);
 document.getElementById('objects').textContent=Object.entries(p.objects).map(([name,o])=>name.toUpperCase()+': '+o.status+(o.status==='estimated'?' | '+o.quality+' | center mm: '+o.center_m.map(v=>(v*1000).toFixed(1)).join(', ')+(o.normal?' | normal: '+o.normal.map(v=>v.toFixed(3)).join(', '):''):'')+'\n'+(o.caveat||o.reason||'')).join('\n\n');
 } catch(e){document.getElementById('perceptionFresh').textContent='Perception unavailable';}
 try {const e=await(await fetch('/perception-evaluation.json',{cache:'no-store'})).json();const age=(Date.now()-Date.parse(e.timestamp_utc))/1000;document.getElementById('evaluation').textContent=(age<3?'Fresh':'STALE')+' matched-frame evaluation · '+age.toFixed(1)+' s old\n'+JSON.stringify(e.errors,null,2);}catch(e){document.getElementById('evaluation').textContent='No matching frame evaluation available';}
}
setInterval(perceptionUpdate,1000);perceptionUpdate();
async function update(){try{const s=await(await fetch('/status.json',{cache:'no-store'})).json();const age=(Date.now()-Date.parse(s.timestamp_utc))/1000;document.getElementById('fresh').textContent=(age<5?'Live':'Stale')+' · frame '+s.frame+' · '+s.timestamp_utc;document.getElementById('rgb').src='/latest.jpg?t='+s.frame;document.getElementById('depth').src='/depth-preview.png?t='+s.frame;document.getElementById('status').textContent=s.gpu+'\n\n'+s.camera.valid_depth_pixels.toLocaleString()+' depth points\n640 × 480 · axial metres\n\n'+Object.entries(s.arms).map(([n,a])=>n+': '+a.joint_names.length+' joints').join('\n')+'\n\nJoint target error: '+s.max_joint_target_error_rad.toFixed(4)+' rad';;document.getElementById('training').textContent=s.training?JSON.stringify(s.training,null,2):'Stationary foundation';}catch(e){document.getElementById('fresh').textContent='Waiting for a valid simulator observation';}}setInterval(update,1000);update();</script></html>"""


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
            elif path in (
                "/latest.jpg",
                "/depth-preview.png",
                "/status.json",
                "/perception.jpg",
                "/perception.json",
                "/perception-evaluation.json",
                "/validation.json",
                "/validation-example.jpg",
                "/validation-chart.png",
                "/validation-chart.json",
            ):
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
