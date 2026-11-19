"""Read-only loopback preview. Serves only named latest outputs, no controls."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = r"""<!doctype html><html><meta charset="utf-8"><title>Sircus · Isaac Lab</title>
<style>body{background:#10262b;color:#f9f3df;font:17px system-ui;margin:32px auto;max-width:1200px}h1{margin-bottom:4px}p{color:#b7d1d1}main{display:grid;grid-template-columns:2fr 1fr;gap:20px}img{width:100%;border-radius:12px}section{padding:20px;border:1px solid #376169;border-radius:15px}.studyPlots{display:grid;grid-template-columns:1fr 1fr;gap:24px}.studyPlots svg{width:100%}@media(max-width:800px){.studyPlots,main{grid-template-columns:1fr}}pre{white-space:pre-wrap;font-size:14px}#planeReplayButtons button{background:#24474c;color:#f9f3df;border:1px solid #65b8b1;border-radius:7px;padding:9px 12px;margin:4px 5px 4px 0;cursor:pointer}#planeReplayButtons button:hover{background:#376169}b{color:#ffbb59}#bucketGridButtons button{margin:4px;padding:8px;background:#24474c;color:#f9f3df;border:1px solid #65b8b1;border-radius:6px}</style>
<h1>SIRCUS <span style="font-weight:400">/ Isaac Lab</span></h1><p>Two UF850 arms · rigid board mount · open tapered bucket · synthetic RGB-D</p>
<section id="bucket-grid"><h2>Bucket grid · board-only routes</h2><p id="bucketGridStatus">Waiting for spatial study…</p><p>Nine positions: X=470/670/870 mm, Y=−400/−550/−700 mm. Board/release XY stay (670,−100) mm; the board faces each bucket. Search height, gap and tilt, then freeze and perturb the best setting. This is a development catalogue, not RL training or held-out policy evaluation.</p><p role="note" style="color:#ffcf8b">New board-route v1: board bounce + whole-ball entry + 1 second retained, one or more board-contact episodes, and no detected table/floor/robot/other contact during flight. Rim-assisted entry is allowed; single-bounce captures are counted separately. Historical capture totals remain separate. Contact forces are sampled at 240 Hz above 0.01 N; this is simulation evidence, not physical certification.</p><div id="bucketGridTable" style="overflow:auto"></div><div class="studyPlots"><svg id="bucketGridMap" viewBox="0 0 550 420" role="img" aria-label="Top view of nine bucket positions and board-only capture results"></svg><div><p>Fresh confirmation: 30 new seeded shots after candidate screening. Board height and gap vary ±2 mm, tilt ±0.5°. No per-shot retuning. Pending and no setting found are not measured zero success rates.</p><p>Board height is its center above the tabletop. Gap is vertical to the ball bottom. Ball-center Z adds the provisional 37.5 mm radius. Arm travel, grasping and perception are not evaluated.</p><div id="bucketGridButtons"></div><p id="bucketGridDetail"></p></div></div><p id="bucketGridDownload" style="display:none"><a href="/bucket-grid-settings.csv" style="color:#65d2d3">Download verified grid settings (CSV)</a></p><details><summary>Contact probes, frozen controls and study provenance</summary><pre id="bucketGridProvenance"></pre></details></section>
<section id="distance-catalogue"><h2>How far can the bucket go?</h2><p id="distanceCatalogueStatus">Waiting for distance study…</p><p>300–650 mm from the fixed release plane, toward −Y. The 400 mm row rechecks our prior best setting; other distances receive broad searches and local refinement. The 650 mm bucket has 25 mm clearance from the modeled table edge.</p><p id="distanceRouteWarning" role="note" style="border-left:3px solid #ffbb59;padding-left:12px;color:#ffcf8b">Direct board-to-bucket travel is not certified: the capture scorer permits intermediate contacts. A saved-trajectory audit will flag low rebounds outside the bucket.</p><div id="distanceCatalogueTable" style="overflow:auto"></div><div class="studyPlots"><div><svg id="distanceCatalogueChart" viewBox="0 0 710 275" role="img" aria-label="Fresh confirmation capture rate by bucket distance, clean and assisted"></svg><p>Fresh confirmation: green = clean, amber = assisted/marginal. Denominators include all attempted confirmation trials. Pending is not zero performance.</p></div><div><p>Board and release XY remain (670, −100) mm; bucket Y = −100 mm minus distance. Heights are above tabletop Z=0. Tilt is from horizontal, facing the bucket.</p><p>Ball-bottom Z = board H + gap. Ball-center Z adds 37.5 mm; provisional flange Z adds another 95 mm.</p><p>Each searched distance: up to three candidates receive 5 unchanged and 20 varied trials. Freeze the winner, then test 50 fresh variations: board H ±2 mm, gap ±2 mm, tilt ±0.5°. No action retuning during confirmation.</p></div></div><h3>Selected setup · side view toward the bucket</h3><div id="distanceSetupButtons"></div><svg id="distanceSetupDiagram" viewBox="0 0 850 650" role="img" aria-label="Selected board height, tilt, ball release and bucket distance" style="width:100%;max-width:850px"></svg><p id="distanceSetupCaption"></p><p id="distanceCatalogueDecision"></p><p>Quality 2 = clean capture; 1 = assisted/marginal capture; 0 = miss. Both capture tiers require a measured board bounce, whole-ball entry and continuous one-second retention. Multiple board contacts remain allowed. This is simulation search with endpoint initialization, not learned motion or physical validation.</p><p id="distanceCatalogueDownload" style="display:none"><a href="/distance-settings.csv" style="color:#65d2d3">Download selected settings and fresh results (CSV)</a></p><details><summary>Compare screening candidates at each distance</summary><div id="distanceCatalogueCandidates" style="overflow:auto"></div></details><details><summary>Search bounds, per-distance contact checks and provenance</summary><pre id="distanceCatalogueDetails"></pre></details></section>
<section id="shot-catalogue"><h2>Fixed-distance shot catalogue · 400 mm</h2><p id="catalogueStatus">Waiting for dense shot catalogue…</p><p><b>2 — clean capture</b>: at least 1 mm mouth clearance, no detected pre-entry/near-rim contact or later partial exit. <b>1 — assisted or marginal capture</b>. <b>0 — miss</b>. Rejected IK/settling is ×, not a released miss. Both capture tiers require a board bounce, whole-ball entry and one continuous second inside.</p><p id="catalogueSearch"></p><h3>Selected settings: repeated confirmation</h3><p>All heights are above the table. Board center XY (670, −100) mm; bucket XY (670, −500) mm. Tilt 0° is horizontal, facing toward −Y. Gap = release ball-bottom height minus board-center height. Flange offset 95 mm and ball radius 37.5 mm are provisional.</p><div id="catalogueFinalists" style="overflow:auto"></div><p>Each selected setting: 20 unchanged repeats + 20 seeded trials varying board H ±2 mm, gap ±2 mm and tilt ±0.5°. Selection is frozen before confirmation. No physical noise, material variation or bucket-position variation in this test.</p><p id="catalogueFresh"></p><p id="catalogueDecision"></p><p><a href="/catalogue-settings.csv" style="color:#65d2d3">Download every searched setting (CSV)</a></p><details><summary>Dense search heatmaps: 9 heights × 5 gaps × 11 tilts</summary><p>Green 2: clean · amber 1: assisted/marginal · coral 0: miss · gray ×: rejected · dark: pending.</p><div id="catalogueHeatmaps" class="studyPlots"></div></details><details><summary>Every successful searched combination — single search trial, not confirmed repeatability</summary><div id="catalogueAll" style="overflow:auto"></div></details><details><summary>Contact probes, settling evidence and immutable contract</summary><pre id="catalogueDetails"></pre></details><p>See the earlier fixed-plane side view below for the geometry. It shows historical selected shots, not this catalogue's confirmations. This is a denser physics search, not new RL training.</p></section>
<section id="plane-sweep"><h2>One plane · one bucket · three shot settings</h2><p id="planeStatus">Waiting for fixed-plane study…</p><p>Bucket 400 mm in front of the release plane. Release XY stays above the board center; only board height, drop gap and tilt vary. Endpoint initialization with verified settling; not collision-safe arm travel or RL training.</p><div class="studyPlots"><div><svg id="planeDiagram" viewBox="0 0 700 510" role="img" aria-label="Side view of fixed release plane, tilted board, bucket and measured ball trajectory"></svg><p id="planeShot"></p><div id="planeReplayButtons"></div></div><div><img id="planeSetup" alt="Actual simulator render of the first selected board/dropper setup before release" style="display:none"><p>Actual Isaac camera image of selected setup 1 before release. Ball is parked during this image; the side view shows its release and measured flight.</p></div></div><p id="planeDecision"></p><p>Green ✓: capture · coral dot: miss · gray ×: rejected endpoint/settling · dark: pending. Grid results are searched candidates, not policy performance.</p><div id="planeHeatmaps" class="studyPlots"></div><h3>Fresh repeats of selected combinations</h3><div id="planeRepeats"></div><details><summary>Contract, measured pose checks and evidence</summary><pre id="planeDetails"></pre></details></section>
<section id="balanced-pilot"><h2>Solver fix &amp; distance-balanced learning pilot</h2><p id="pilotStatus">Waiting for pilot report…</p><p id="pilotPose"></p><div id="pilotCurve"></div><div id="pilotComparison" style="overflow:auto"></div><svg id="pilotDistanceChart" viewBox="0 0 700 275" style="width:100%;max-width:900px" role="img" aria-label="Fresh paired test captures by distance"></svg><p id="pilotDecision"></p><p>384 PPO training throws · 96 training layouts · 18 validation layouts · 36 fresh test layouts, equally split near/medium/far. Fine-tunes the selected 100-layout policy; validation can retain the initial checkpoint. All comparison policies use the same new IK and measured settling controller. These are transferred reference policies, not their old historical scores. No per-scene action search at inference. Only12 test layouts per distance and one training seed; these are pilot results, not a broad robustness claim. Settling uses simulator body readback and model-based gravity compensation.</p><details><summary>Pose checks, action ranges and study details</summary><pre id="pilotDetails"></pre></details></section>
<section id="ik-distance"><h2>IK audit &amp; controlled distance sweep</h2><p id="diagnosticStatus">Loading diagnostic results…</p><p id="ikAuditText"></p><svg id="ikAuditChart" viewBox="0 0 700 75" style="width:100%;max-width:900px" role="img" aria-label="Historical IK failure recovery"></svg><div id="distanceSweepTable" style="overflow:auto"></div><p>Frozen policy cells: original solver → extended solver, one shot each. Grid: 35 tilt/height candidates at each distance, heading fixed toward the bucket. Selected candidates get three fresh repeats. This is diagnostic search, not learned inference or a generalization score.</p><div id="sweepHeatmaps" class="studyPlots"></div><p id="diagnosticConclusion"></p><p>Board center fixed at (670, −100, 400) mm; bucket moves from 200 to 400 mm away along −Y. IK proves only an endpoint solution, not collision-safe travel. An unresolved solve does not prove the target physically unreachable.</p></section>

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
async function pilotUpdate(){
 try{
 const r=await(await fetch('/balanced-pilot.json',{cache:'no-store'})).json();
 const stale=!['completed','smoke_completed','failed','blocked_pose_verification'].includes(r.phase)&&(Date.now()-Date.parse(r.timestamp_utc))>120000;
 document.getElementById('pilotStatus').textContent=r.phase+' · '+r.episodes+'/'+r.budget+' training throws · '+r.run+(stale?' · STALE PROGRESS':'');
 const p=r.pose_verification||{};
 document.getElementById('pilotPose').textContent=p.measured?'Development board verification: '+p.measured+'/'+p.trials+' measured; '+p.invalid+' invalid. Worst position error '+p.initial_max_position_mm.toFixed(4)+' → '+p.final_max_position_mm.toFixed(4)+' mm; final normal error '+p.final_max_normal_deg.toFixed(4)+'°. Release requires ≤2 mm, ≤0.5°, joint speed ≤0.005 rad/s, and ≤0.1 mm drift over250 ms.':'Board pose verification pending.';
 studyTable(document.getElementById('pilotCurve'),['Training throws','Validation captures','Invalid endpoints'],(r.learning_curve||[]).map(c=>[c.episodes,c.captures+'/'+c.trials,c.invalid_endpoints]));
 const labels={pilot_selected:'Pilot selected',ppo_100:'Existing 100-layout',ppo_1000:'Existing 1,000-layout',warm_fixed:'Fixed warm action'};
 studyTable(document.getElementById('pilotComparison'),['Frozen policy','Fresh test captures','Near','Medium','Far','Invalid'],Object.entries(r.evaluation||{}).map(([k,v])=>[labels[k]||k,v.captures+'/'+v.trials,...['near','medium','far'].map(d=>v.by_distance[d].captures+'/'+v.by_distance[d].trials),v.invalid_endpoints]));
 const plot=document.getElementById('pilotDistanceChart');plot.replaceChildren();const entries=Object.entries(r.evaluation||{});const colors=['#ffc35d','#56d3ca','#88b8ff','#b0b6c2'];
 if(entries.length){for(const rate of [0,.5,1]){studySvg(plot,'line',{x1:45,x2:665,y1:210-160*rate,y2:210-160*rate,stroke:'#35545b'});studySvg(plot,'text',{x:40,y:215-160*rate,fill:'#c5d9db','text-anchor':'end'},Math.round(rate*100)+'%');}for(const [i,d]of ['near','medium','far'].entries()){studySvg(plot,'text',{x:135+i*210,y:232,fill:'#c5d9db','text-anchor':'middle'},d);for(const [j,[label,value]]of entries.entries()){const v=value.by_distance[d],height=160*v.captures/v.trials;const rect=studySvg(plot,'rect',{x:65+i*210+j*38,y:210-height,width:32,height,fill:colors[j]});studySvg(rect,'title',{},labels[label]+': '+v.captures+'/'+v.trials);}}for(const [i,[label]]of entries.entries())studySvg(plot,'text',{x:15+i*175,y:264,fill:colors[i],'font-size':12},labels[label]);}
 document.getElementById('pilotDecision').textContent=r.conclusion||(r.phase==='completed'?'Study completed; inspect the paired results before increasing the budget.':'Training and evaluation are incomplete; no improvement claim yet.');
 document.getElementById('pilotDetails').textContent=JSON.stringify(r,null,2);
 }catch(e){document.getElementById('pilotStatus').textContent='Pilot report unavailable';}
}
setInterval(pilotUpdate,5000);pilotUpdate();
async function diagnosticUpdate(){
 try{
 const r=await(await fetch('/ik-distance.json',{cache:'no-store'})).json();
 document.getElementById('diagnosticStatus').textContent=r.state+' · '+r.completed+'/'+r.planned+' PhysX shots · '+r.study+' · no policy updates'+(r.state==='running'&&(Date.now()-Date.parse(r.timestamp_utc))>120000?' · STALE PROGRESS':'');
 const a=r.ik_audit,counts=a.stage_counts||{},n=a.historical_failures||0;
 document.getElementById('ikAuditText').textContent='Historical board-arm failures: '+n+'. Recovered with more iterations: '+(counts.iteration||0)+'; recovered with different seeds: '+(counts.seed||0)+'; unresolved: '+(counts.unresolved||0)+'. Audit '+a.completed+'/'+a.planned+' cases including successful controls.';
 const chart=document.getElementById('ikAuditChart');chart.replaceChildren();let x=0;
 for(const [key,label,color] of [['iteration','More iterations','#ffc35d'],['seed','Different seed','#56d3ca'],['unresolved','Unresolved','#ed8176']]){const v=counts[key]||0,w=680*v/Math.max(n,1);studySvg(chart,'rect',{x,y:5,width:w,height:25,fill:color});studySvg(chart,'text',{x:x+w/2,y:23,fill:'#10262b','text-anchor':'middle','font-size':14},v);x+=w;}
 for(const [i,text] of ['More iterations','Different seed','Unresolved'].entries())studySvg(chart,'text',{x:10+230*i,y:55,fill:['#ffc35d','#56d3ca','#ed8176'][i],'font-size':14},text);
 const rows=r.distances.map(d=>{const policy=label=>[false,true].map(extended=>{const p=d.policies.find(p=>p.label===label&&p.solver===(extended?'extended':'original'));return !p?'pending':p.invalid?'IK fail':p.success?'capture':'miss';}).join(' → ');const best=d.grid_best_action;const warm=d.policies.find(p=>p.label==='warm_fixed');return [(d.distance_m*1000).toFixed(0)+' mm',policy('ppo_100'),policy('ppo_1000'),!warm?'pending':warm.invalid?'IK fail':warm.success?'capture':'miss',d.grid_captures+'/'+d.grid_tested+' ('+d.grid_invalid+' IK fail)',best?best[0]+'° / '+(best[2]*1000).toFixed(0)+' mm':'pending',d.repeats.filter(p=>p.success).length+'/'+d.repeats.length];});
 studyTable(document.getElementById('distanceSweepTable'),['Distance','100-layout PPO','1,000-layout PPO','Fixed warm action','Grid captures','Grid-selected tilt / gap','Fresh repeats'],rows);
 const maps=document.getElementById('sweepHeatmaps');maps.replaceChildren();
 for(const d of r.distances){if(!d.grid_cells)continue;const box=document.createElement('div');maps.appendChild(box);const heading=document.createElement('h3');heading.textContent=(d.distance_m*1000)+' mm · action grid';box.appendChild(heading);const svg=studySvg(box,'svg',{viewBox:'0 0 400 285',role:'img','aria-label':'Tilt and height outcomes at '+d.distance_m+' metres'});const tilts=[12,16,20,24,28,36,45],gaps=[60,100,150,200,250];for(const [i,t]of tilts.entries()){studySvg(svg,'text',{x:40,y:40+30*i,fill:'#c5d9db','text-anchor':'end','font-size':13},t+'°');for(const [j,g]of gaps.entries()){const c=d.grid_cells.find(c=>c.tilt_deg===t&&Math.abs(c.gap_m*1000-g)<.01);studySvg(svg,'rect',{x:50+65*j,y:20+30*i,width:60,height:26,fill:!c?'#35545b':c.invalid?'#87939a':c.success?'#56d3ca':'#ed8176'});}}
 for(const [j,g]of gaps.entries())studySvg(svg,'text',{x:80+65*j,y:250,fill:'#c5d9db','text-anchor':'middle','font-size':13},g+' mm');studySvg(svg,'text',{x:50,y:278,fill:'#c5d9db','font-size':12},'Green capture · coral miss · gray IK failure');}
 document.getElementById('diagnosticConclusion').textContent=r.conclusion||'Diagnostic in progress; do not infer readiness from partial results.';
 }catch(e){document.getElementById('diagnosticStatus').textContent='No current diagnostic report available';}
}
setInterval(diagnosticUpdate,5000);diagnosticUpdate();

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
async function update(){try{const s=await(await fetch('/status.json',{cache:'no-store'})).json();const age=(Date.now()-Date.parse(s.timestamp_utc))/1000;document.getElementById('fresh').textContent=(age<5?'Live':'Stale')+' · frame '+s.frame+' · '+s.timestamp_utc;document.getElementById('rgb').src='/latest.jpg?t='+s.frame;document.getElementById('depth').src='/depth-preview.png?t='+s.frame;document.getElementById('status').textContent=s.gpu+'\n\n'+s.camera.valid_depth_pixels.toLocaleString()+' depth points\n640 × 480 · axial metres\n\n'+Object.entries(s.arms).map(([n,a])=>n+': '+a.joint_names.length+' joints').join('\n')+'\n\nJoint target error: '+s.max_joint_target_error_rad.toFixed(4)+' rad';;document.getElementById('training').textContent=s.training?JSON.stringify(s.training,null,2):'Stationary foundation';}catch(e){document.getElementById('fresh').textContent='Waiting for a valid simulator observation';}}setInterval(update,1000);update();</script><script src="/plane_panel.js"></script><script src="/catalogue_panel.js"></script><script src="/distance_catalogue_panel.js"></script><script src="/bucket_grid_panel.js"></script></html>"""


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
                "/plane_panel.js",
                "/catalogue_panel.js",
                "/distance_catalogue_panel.js",
                "/bucket_grid_panel.js",
            ):
                payload = Path(__file__).with_name(path[1:]).read_bytes()
                mime = "application/javascript"
            elif path in (
                "/latest.jpg",
                "/depth-preview.png",
                "/status.json",
                "/perception.jpg",
                "/perception.json",
                "/perception-evaluation.json",
                "/plane-sweep.json",
                "/shot-catalogue.json",
                "/distance-catalogue.json",
                "/bucket-grid.json",
                "/bucket-grid-settings.csv",
                "/distance-settings.csv",
                "/catalogue-confirmation.json",
                "/catalogue-settings.csv",
                "/plane-setup.jpg",
                "/balanced-pilot.json",
                "/ik-distance.json",
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
                    ".csv": "text/csv; charset=utf-8",
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
