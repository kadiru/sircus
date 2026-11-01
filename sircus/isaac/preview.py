"""Read-only loopback preview. Serves only named latest outputs, no controls."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = r"""<!doctype html><html><meta charset="utf-8"><title>Sircus · Isaac Lab</title>
<style>body{background:#10262b;color:#f9f3df;font:17px system-ui;margin:32px auto;max-width:1200px}h1{margin-bottom:4px}p{color:#b7d1d1}main{display:grid;grid-template-columns:2fr 1fr;gap:20px}img{width:100%;border-radius:12px}section{padding:20px;border:1px solid #376169;border-radius:15px}.studyPlots{display:grid;grid-template-columns:1fr 1fr;gap:24px}.studyPlots svg{width:100%}@media(max-width:800px){.studyPlots,main{grid-template-columns:1fr}}pre{white-space:pre-wrap;font-size:14px}b{color:#ffbb59}</style>
<h1>SIRCUS <span style="font-weight:400">/ Isaac Lab</span></h1><p>Two UF850 arms · rigid board mount · open tapered bucket · synthetic RGB-D</p>
<section id="study-results"><h2>Completed learning study</h2><p id="comparisonStatus">Loading frozen policy comparison…</p><div id="comparisonSummary" style="overflow:auto"></div><div class="studyPlots"><div><h3>Validation learning curve</h3><svg id="learningCurve" viewBox="0 0 600 270" role="img" aria-label="Validation capture rate by training throws"></svg><p>32 layouts repeated twice at each checkpoint. Checkpoint selection uses validation, not test results.</p></div><div><h3>Unseen test layouts by distance</h3><svg id="distanceChart" viewBox="0 0 600 270" role="img" aria-label="Test captures by bucket distance"></svg><p>Near: 200–267 mm · medium: 267–333 mm · far: 333–400 mm. Distance is horizontal board-center to bucket-center.</p></div></div><div id="distanceSummary" style="overflow:auto"></div><p id="studyDecision"></p><p id="studyLimit"></p><details><summary>Action ranges and failure analysis</summary><pre id="studyDiagnostics"></pre></details></section>

<main><section><img id="rgb" alt="Latest synthetic RGB camera view"><p id="fresh">Waiting for simulator…</p></section><section><b>Depth camera</b><img id="depth" alt="Axial depth preview"><pre id="status"></pre><b>Learning</b><pre id="training">Stationary foundation</pre></section></main>
<p>Fixed-scene cooperative shot learning. Policy uses known simulation state; grasping, compliant mounting and perception-driven control remain future work. Bucket reported height: 175 mm; its interpretation is unconfirmed.</p>
<section><h2>Cooperative learning study</h2><p>Privileged geometry → shared PPO → board tilt, heading offset and release height. Endpoint-initialized shots; no grasp or travel learning. Validation selects checkpoints; test layouts remain held out.</p><div id="learningRuns">Waiting for study status…</div><p>Training success uses exploratory actions. Validation uses a frozen deterministic policy; repeated contact outcomes may differ. A completed run is not a physical deployment claim.</p></section>
<section><h2>Randomized perception validation</h2><p id="validationStatus">No randomized study yet</p><p>Frozen detector · paired lighting/noise conditions across seeded layouts · ground truth used only for scoring. Static snapshots, not drop success.</p><img id="validationChart" alt="Completed-study detection availability chart" style="display:none"><div style="overflow:auto"><table id="validationTable" style="width:100%;text-align:left;border-spacing:10px"></table></div><p>Cells: detections / observations; mean error when detected. Table = plane distance; other objects = center distance. Missing detections stay in the availability denominator.</p><p id="visibility"></p><details><summary>Ranges, error percentiles and exclusions</summary><pre id="validationDetails"></pre></details><img id="validationImage" alt="Most recent validation observation with estimates" style="max-width:640px"></section>
<section><h2>RGB-D perception</h2><p>Observed geometry from RGB + depth. Color/shape priors; synthetic camera calibration. No object ground truth is used by the estimator.</p><img id="perception" alt="RGB-D object estimates"><p id="perceptionFresh">Waiting for observations</p><pre id="objects"></pre><p>Capture outcome: unknown from perception. The RL scorer is separate and uses simulator truth.</p><details><summary>Simulator evaluation only</summary><pre id="evaluation"></pre></details></section>
<script>
const studyColors=['#ffc35d','#56d3ca'];
function studyTable(parent,headers,rows){parent.replaceChildren();const table=document.createElement('table');table.style.cssText='border-spacing:14px;text-align:left;width:100%';parent.appendChild(table);for(const [i,row] of [headers,...rows].entries()){const tr=document.createElement('tr');for(const value of row){const cell=document.createElement(i===0?'th':'td');cell.textContent=value;tr.appendChild(cell);}table.appendChild(tr);}}
function studySvg(root,tag,attrs,text){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))node.setAttribute(k,v);if(text!==undefined)node.textContent=text;root.appendChild(node);return node;}
function studyAxes(root){root.replaceChildren();for(const rate of [0,.25,.5,.75,1]){const y=220-180*rate;studySvg(root,'line',{x1:50,x2:570,y1:y,y2:y,stroke:'#35545b'});studySvg(root,'text',{x:43,y:y+5,fill:'#c5d9db','text-anchor':'end','font-size':12},Math.round(rate*100)+'%');}}
async function comparisonUpdate(){
 try{
 const r=await(await fetch('/learning-comparison.json',{cache:'no-store'})).json();
 const live=await Promise.all(['/learning-study.json','/learning-study-4070.json'].map(async u=>{try{return (await(await fetch(u,{cache:'no-store'})).json()).run;}catch(e){return null;}}));const matched=r.source_runs.every(id=>live.includes(id));document.getElementById('comparisonStatus').textContent=(matched?'Both runs completed':'Archived completed comparison — latest runs differ')+' · 4,096 training throws each · 256 matched unseen test layouts · frozen policies · report '+r.timestamp_utc;
 const pct=x=>(100*x).toFixed(1)+'%';
 const rows=[];for(const run of r.runs){for(const [key,label] of [['test',run.training_layouts+'-layout PPO'],['test_warm_fixed','Fixed warm action ('+run.training_layouts+' run)'],['test_midpoint','Fixed midpoint ('+run.training_layouts+' run)']]){const v=run.evaluation[key];rows.push([label,v.captures+'/'+v.trials,pct(v.success_rate),v.wilson95.map(pct).join('–'),v.invalid_endpoints,key==='test'?run.selected_episode:'—']);}}
 studyTable(document.getElementById('comparisonSummary'),['Policy','Captures','Test rate','95% interval','IK failures','Selected throw'],rows);
 const distanceRows=[];for(const run of r.runs){distanceRows.push([run.training_layouts+'-layout PPO',...['near','medium','far'].map(k=>{const g=run.evaluation.test.by_distance[k];return g.captures+'/'+g.trials+' · '+pct(g.captures/g.trials);})]);}
 studyTable(document.getElementById('distanceSummary'),['Policy','Near','Medium','Far'],distanceRows);
 const line=document.getElementById('learningCurve');studyAxes(line);
 for(const [i,run] of r.runs.entries()){const points=run.learning_curve.map(c=>[50+520*c.episodes/4096,220-180*c.success_rate]);studySvg(line,'polyline',{points:points.map(p=>p.join(',')).join(' '),fill:'none',stroke:studyColors[i],'stroke-width':3});for(const [x,y] of points)studySvg(line,'circle',{cx:x,cy:y,r:3,fill:studyColors[i]});studySvg(line,'text',{x:60+i*260,y:20,fill:studyColors[i],'font-size':14},run.training_layouts+' layouts');}
 for(const e of [0,1024,2048,3072,4096])studySvg(line,'text',{x:50+520*e/4096,y:245,fill:'#c5d9db','text-anchor':'middle','font-size':12},e);
 const bars=document.getElementById('distanceChart');studyAxes(bars);
 for(const [j,key] of ['near','medium','far'].entries()){for(const [i,run] of r.runs.entries()){const g=run.evaluation.test.by_distance[key],h=180*g.captures/g.trials;studySvg(bars,'rect',{x:90+170*j+40*i,y:220-h,width:32,height:h,fill:studyColors[i]});}studySvg(bars,'text',{x:125+170*j,y:245,fill:'#c5d9db','text-anchor':'middle','font-size':13},key);}
 for(const [i,run] of r.runs.entries())studySvg(bars,'text',{x:60+i*260,y:20,fill:studyColors[i],'font-size':14},run.training_layouts+' layouts');
 document.getElementById('studyDecision').textContent=r.recommendation;
 document.getElementById('studyLimit').textContent=r.scope+' Results do not establish that 100 layouts are generally better than 1,000. Confidence intervals describe layout sampling, not training-seed variability.';
 document.getElementById('studyDiagnostics').textContent=JSON.stringify(r.runs.map(x=>({run:x.run,selected_episode:x.selected_episode,elapsed_hours:x.elapsed_hours,diagnostics:x.diagnostics})),null,2);
 }catch(e){document.getElementById('comparisonStatus').textContent='Comparison unavailable or incomplete; no readiness claim.';}
}
setInterval(comparisonUpdate,30000);comparisonUpdate();

async function learningUpdate(){
 const root=document.getElementById('learningRuns');root.replaceChildren();
 for(const [gpu,url] of [['RTX 4090','/learning-study.json'],['RTX 4070','/learning-study-4070.json']]){
 const section=document.createElement('section');root.appendChild(section);
 try{const r=await(await fetch(url,{cache:'no-store'})).json();const title=document.createElement('h3');title.textContent=gpu+' · '+r.phase;section.appendChild(title);
 const age=(Date.now()-Date.parse(r.timestamp_utc))/1000;const p=document.createElement('p');p.textContent=r.episodes+'/'+r.budget+' throws · '+r.train_layouts+' training layouts · '+r.run+' · '+(r.scope||'historical diagnostic')+(age>(r.phase==='training'?120:900)&&!['completed','failed'].includes(r.phase)?' · STALE: check worker':'');section.appendChild(p);
 const table=document.createElement('table');table.style.borderSpacing='12px';section.appendChild(table);
 for(const values of [['Throws','Validation captures','Rate'],...(r.learning_curve||[]).map(c=>[c.episodes,c.captures+'/'+c.trials,(100*c.success_rate).toFixed(1)+'%'])]){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.appendChild(td);}table.appendChild(tr);}
 if(r.summary){const detail=document.createElement('details');const title=document.createElement('summary');title.textContent='Full frozen evaluation';detail.appendChild(title);section.appendChild(detail);const pre=document.createElement('pre');pre.textContent=JSON.stringify(r.summary.evaluation,null,2);detail.appendChild(pre);}
 }catch(e){section.textContent=gpu+' · no current report';}}
}
setInterval(learningUpdate,5000);learningUpdate();
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
    p.add_argument("--secondary-output", help="Read-only second GPU study output")
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
                "/learning-comparison.json",
                "/learning-study.json",
                "/learning-study-4070.json",
                "/validation.json",
                "/validation-example.jpg",
                "/validation-chart.png",
                "/validation-chart.json",
            ):
                f = out / path[1:]
                if path == "/learning-study-4070.json" and a.secondary_output:
                    f = Path(a.secondary_output) / "learning-study.json"
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
