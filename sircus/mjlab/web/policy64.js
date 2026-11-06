// Read-only fixed64 policy evidence. No prediction, solver, training or hardware requests.
export const VERSION='sircus-policy64-bc-ppo-v1';
export const AUDIT_VERSION='sircus-policy64-independent-audit-v1';
const METHODS=['prior','nearest','bc','ppo'];
const NAMES={prior:'Prior',nearest:'Nearest TRAIN success',bc:'Behavior cloning',ppo:'Selected PPO'};
const finiteVector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
const mm=(v,d=1)=>Number.isFinite(v)?(v*1000).toFixed(d):'unknown';
const count=v=>Number.isInteger(v)&&v>=0?String(v):'unknown';
const fraction=(k,n)=>Number.isInteger(k)&&Number.isInteger(n)&&0<=k&&k<=n?(n?`${k} / ${n}`:'not run'):'unknown';
const ratio=(k,n)=>Number.isInteger(k)&&Number.isInteger(n)&&n>0&&0<=k&&k<=n?100*k/n:null;
const rows=v=>Array.isArray(v)?v.filter(r=>r&&typeof r==='object'):[];
const node=(root,name)=>root.querySelector(`[data-policy64="${name}"]`);
const text=(root,name,value)=>{const n=node(root,name);if(n)n.textContent=value;};
export function policy64AuditReady(raw){const a=raw?.audit;return raw?.version===VERSION&&raw.status==='completed'&&raw.phase==='completed'&&typeof raw.contract_fingerprint==='string'&&raw.contract_fingerprint.length>0&&a?.version===AUDIT_VERSION&&a.status==='passed'&&a.contract_fingerprint===raw.contract_fingerprint&&a.full_checks_verified===true&&a.bound_files_verified===true&&Number.isInteger(a.checks_count)&&a.checks_count>0&&/^[a-f0-9]{64}$/.test(a.source_sha256??'');}
export function policy64ComparisonReady(raw){const seen=new Set(),comparison=rows(raw?.test_comparison),coverage=policy64Coverage(raw);return policy64AuditReady(raw)&&raw.gate?.status==='passed'&&raw.gate?.complete===true&&raw.gate?.substeps===64&&raw.protocol?.substeps===64&&raw.control_gates?.pre?.status==='passed'&&raw.control_gates?.post?.status==='passed'&&comparison.length===4&&coverage.length===50&&new Set(coverage.map(r=>r.context_id)).size===50&&coverage.every(r=>r.outcome!=='unknown'&&finiteVector(r.bucket_xy_m,2))&&comparison.every(row=>{const c=row.counts??{};if(!METHODS.includes(row.method)||seen.has(row.method)||c.attempts!==50||!Number.isInteger(c.native_releases)||c.native_releases<0||c.native_releases>50||ratio(c.captures,50)===null||ratio(c.clean,50)===null||c.clean>c.captures||c.captures>c.native_releases)return false;seen.add(row.method);return row.method!=='ppo'||/^[a-f0-9]{64}$/.test(raw.selected?.sha256??'')&&row.artifact_sha256===raw.selected.sha256&&c.native_releases===coverage.filter(p=>p.native_release===true).length&&c.captures===coverage.filter(p=>p.native_release===true&&!p.invalid&&p.invalid_physics!==true&&p.capture===true).length&&c.clean===coverage.filter(p=>p.native_release===true&&!p.invalid&&p.invalid_physics!==true&&p.capture===true&&p.clean===true).length;});}
export function policy64Trajectory(raw,trace) {
  if(!trace)return null;
  const group=null;
  const same=(a,b)=>finiteVector(a,2)&&finiteVector(b,2)&&a.every((v,i)=>Math.abs(v-b[i])<=1e-8);
  const board=trace.board_xy_m??(finiteVector(trace.values,5)?trace.values.slice(0,2):group?.board_xy_m),bucket=trace.bucket_xy_m??group?.bucket_xy_m;
  if(!finiteVector(board,2)||!finiteVector(bucket,2)||
    (trace.board_xy_m!==undefined&&group?.board_xy_m!==undefined&&!same(trace.board_xy_m,group.board_xy_m))||
    (trace.bucket_xy_m!==undefined&&group?.bucket_xy_m!==undefined&&!same(trace.bucket_xy_m,group.bucket_xy_m))||
    !Array.isArray(trace.position_m)||trace.position_m.length<2||!trace.position_m.every(p=>finiteVector(p,3)))return null;
  const distance=Math.hypot(bucket[0]-board[0],bucket[1]-board[1]);
  if(distance<1e-9||[trace.distance_m,group?.distance_m].some(d=>d!==undefined&&(!Number.isFinite(d)||Math.abs(d-distance)>1e-6)))return null;
  const axis=[(bucket[0]-board[0])/distance,(bucket[1]-board[1])/distance],h=trace.board_height_m??(finiteVector(trace.values,5)?trace.values[2]:group?.board_height_m);
  return {distance,board,bucket,boardHeight:Number.isFinite(h)?h:null,
    points:trace.position_m.map(p=>[(p[0]-board[0])*axis[0]+(p[1]-board[1])*axis[1],p[2]]),
    lateral:Math.max(...trace.position_m.map(p=>Math.abs(-(p[0]-board[0])*axis[1]+(p[1]-board[1])*axis[0])))};
}
function table(target,headers,rows) {
  if(!target)return;
  const d=target.ownerDocument,t=d.createElement('table'),head=d.createElement('thead'),body=d.createElement('tbody'),h=d.createElement('tr');
  for(const label of headers){const cell=d.createElement('th');cell.textContent=label;h.append(cell);}head.append(h);
  for(const row of rows){const tr=d.createElement('tr');for(const value of row){const td=d.createElement('td');td.textContent=String(value);tr.append(td);}body.append(tr);}
  if(!rows.length){const tr=d.createElement('tr'),td=d.createElement('td');td.colSpan=headers.length;td.textContent='No observations available yet.';tr.append(td);body.append(tr);}
  t.append(head,body);target.replaceChildren(t);
}
function drawTrace(root,raw,trace) {
  const canvas=node(root,'trace'),c=canvas?.getContext('2d');if(!c)return;
  const w=canvas.width,h=canvas.height;c.clearRect(0,0,w,h);c.fillStyle='#edf0e7';c.fillRect(0,0,w,h);
  const view=policy64Trajectory(raw,trace);c.font='11px sans-serif';c.fillStyle='#496f67';
  if(!view){c.fillText('No unambiguous saved native trace available',40,100);text(root,'trace-caption','A matching record/group geometry and actual saved positions are required; no analytic trajectory is generated.');return;}
  const xs=view.points.map(p=>p[0]),zs=view.points.map(p=>p[1]),xl=Math.min(-.1,...xs),xh=Math.max(view.distance+.15,...xs),zl=Math.min(0,...zs,view.boardHeight??0),zh=Math.max(.3,...zs,view.boardHeight??0)+.05;
  const x=v=>55+(v-xl)/(xh-xl)*(w-80),y=v=>h-42-(v-zl)/(zh-zl)*(h-68);
  c.strokeStyle='#d3ddd0';c.lineWidth=1;c.textAlign='center';for(let i=0;i<=4;i++){const px=xl+(xh-xl)*i/4,pz=zl+(zh-zl)*i/4;c.beginPath();c.moveTo(x(px),y(zl));c.lineTo(x(px),y(zh));c.stroke();c.beginPath();c.moveTo(x(xl),y(pz));c.lineTo(x(xh),y(pz));c.stroke();c.fillText(px.toFixed(2),x(px),h-25);c.fillText(pz.toFixed(2),30,y(pz)+4);}
  c.fillText('Distance along this board → bucket axis (m)',w/2,h-7);c.textAlign='left';c.fillText('Native ball-center Z (m)',55,15);
  c.strokeStyle='#81978a';c.setLineDash([5,4]);c.beginPath();c.moveTo(x(view.distance),y(zl));c.lineTo(x(view.distance),y(zh));c.stroke();c.fillText(`Bucket center axis (${mm(view.distance)} mm)`,Math.min(x(view.distance)+6,w-190),42);c.setLineDash([]);
  if(view.boardHeight!==null){c.fillStyle='#b88a30';c.beginPath();c.arc(x(0),y(view.boardHeight),4,0,2*Math.PI);c.fill();c.fillText('Configured board center',x(0)+8,y(view.boardHeight)-7);}
  const captured=trace.capture??trace.route_success;c.strokeStyle=captured===true?'#128b8c':captured===false?'#ec7657':'#6e8180';c.lineWidth=2;c.beginPath();view.points.forEach((p,i)=>{if(i)c.lineTo(x(p[0]),y(p[1]));else c.moveTo(x(p[0]),y(p[1]));});c.stroke();
  const bool=v=>v===true?'yes':v===false?'no':'unknown',invalid=trace.invalid===false||trace.invalid===''||trace.invalid===null?'none':trace.invalid===undefined?'unknown':String(trace.invalid);
  const oracle=typeof trace.oracle_agreement==='boolean'?trace.oracle_agreement:trace.scalar_score_audit_passed;
  text(root,'trace-caption',`${trace.record_id??trace.id??'Unknown record'} · ${trace.method??'unknown method'} · ${trace.context_id??'unknown layout'} · ${trace.substeps??'?'} substeps · ${view.points.length} actual saved samples · configured board XY ${view.board.map(v=>mm(v)).join(' / ')} mm · observed board center ${finiteVector(trace.board_center_m,3)?trace.board_center_m.map(v=>mm(v)).join(' / ')+' mm':'unknown'} · bucket XY ${view.bucket.map(v=>mm(v)).join(' / ')} mm · whole entry ${bool(trace.whole_ball_entry)} · max retention ${Number.isFinite(trace.max_retention_s)?trace.max_retention_s.toFixed(3)+' s':'unknown'} · invalid ${invalid} · invalid physics ${bool(trace.invalid_physics)} · oracle agreement ${bool(oracle)} · max off-plane ${mm(view.lateral)} mm. The board marker is configured; the sampled ball flight is observed. ${policy64AuditReady(raw)?'Completed matching full audit.':'Full audit unverified.'} ${trace.selection_scope??''}`);
}

export function policy64CurveRows(raw){return rows(raw?.curves).map(row=>{const v=row.validation??{};return {samples:row.samples,attempts:v.attempts,native:v.native_releases,captures:v.captures,clean:v.clean,invalid:v.invalid,meanReward:v.mean_reward,captureRate:ratio(v.captures,v.attempts),cleanRate:ratio(v.clean,v.attempts)};});}
export function policy64Coverage(raw){return rows(raw?.test_positions).filter(r=>r.method==='ppo').map(r=>({...r,outcome:r.native_release===false?'rejected':r.native_release!==true?'unknown':r.invalid||r.invalid_physics===true?'invalid':r.capture===true?(r.clean===true?'clean':'capture'):r.capture===false?'miss':'unknown'}));}

function drawCurves(root,raw){
 const canvas=node(root,'curves'),c=canvas?.getContext('2d');if(!c)return;const w=canvas.width,h=canvas.height;c.clearRect(0,0,w,h);c.fillStyle='#edf0e7';c.fillRect(0,0,w,h);c.font='11px sans-serif';c.fillStyle='#496f67';
 const points=policy64CurveRows(raw).filter(r=>Number.isInteger(r.samples)&&r.samples>=0&&r.captureRate!==null);if(!points.length){c.fillText('No observed validation checkpoints yet',40,100);return;}
 const max=Math.max(1024,...points.map(p=>p.samples)),x=v=>55+(w-90)*v/max,y=v=>h-42-(h-80)*v/100;
 c.strokeStyle='#d3ddd0';for(let v=0;v<=100;v+=25){c.beginPath();c.moveTo(x(0),y(v));c.lineTo(x(max),y(v));c.stroke();c.fillText(String(v)+'%',8,y(v)+4);}
 for(const v of [0,256,512,1024])c.fillText(String(v),x(v)-10,h-23);
 c.fillText('Fresh PPO TRAIN requests after BC warm start',w/2-135,h-7);c.fillText('Fresh VAL capture / requested',55,17);
 for(const [key,color,label] of [['captureRate','#128b8c','Capture'],['cleanRate','#b88a30','Clean']]){c.strokeStyle=color;c.fillStyle=color;c.lineWidth=2;const valid=points.filter(p=>p[key]!==null).sort((a,b)=>a.samples-b.samples);c.beginPath();valid.forEach((p,i)=>{if(i)c.lineTo(x(p.samples),y(p[key]));else c.moveTo(x(p.samples),y(p[key]));});c.stroke();for(const p of valid){c.beginPath();c.arc(x(p.samples),y(p[key]),4,0,2*Math.PI);c.fill();}c.fillText(label,key==='captureRate'?w-135:w-65,17);}
}
const COLORS={clean:'#128b8c',capture:'#bd9635',miss:'#df7256',rejected:'#65747a',invalid:'#9b5e88',unknown:'#b8c1b6'};
function drawCoverage(root,raw){
 const canvas=node(root,'coverage'),c=canvas?.getContext('2d');if(!c)return;const w=canvas.width,h=canvas.height;c.clearRect(0,0,w,h);c.fillStyle='#edf0e7';c.fillRect(0,0,w,h);c.font='11px sans-serif';c.fillStyle='#496f67';
 const all=policy64Coverage(raw),points=all.filter(r=>finiteVector(r.bucket_xy_m,2)),counts={};for(const r of all)counts[r.outcome]=(counts[r.outcome]??0)+1;
 text(root,'coverage-caption',`${all.length} selected-PPO trial records / 50 planned layouts; ${count(counts.clean??0)} clean, ${count(counts.capture??0)} other captures, ${count(counts.miss??0)} misses, ${count(counts.rejected??0)} rejected, ${count(counts.invalid??0)} invalid, ${count(counts.unknown??0)} unknown. ${policy64AuditReady(raw)?'Completed matching full audit.':'Provisional; full audit is pending or unavailable.'}`);
 if(!points.length){c.fillText('No saved PPO test positions yet',40,100);return;}
 let xmin=Math.min(...points.map(p=>p.bucket_xy_m[0]))-.05,xmax=Math.max(...points.map(p=>p.bucket_xy_m[0]))+.05,ymin=Math.min(...points.map(p=>p.bucket_xy_m[1]))-.05,ymax=Math.max(...points.map(p=>p.bucket_xy_m[1]))+.05;
 const x=v=>55+(w-85)*(v-xmin)/(xmax-xmin),y=v=>h-50-(h-90)*(v-ymin)/(ymax-ymin);
 c.strokeStyle='#d3ddd0';for(let i=0;i<=4;i++){const a=xmin+(xmax-xmin)*i/4,b=ymin+(ymax-ymin)*i/4;c.beginPath();c.moveTo(x(a),y(ymin));c.lineTo(x(a),y(ymax));c.stroke();c.beginPath();c.moveTo(x(xmin),y(b));c.lineTo(x(xmax),y(b));c.stroke();c.fillText(a.toFixed(2),x(a)-12,h-32);c.fillText(b.toFixed(2),8,y(b)+4);}
 c.fillText('Bucket world X (m)',w/2-50,h-9);c.fillText('Bucket world Y (m)',55,17);
 for(const [i,p] of points.entries()){c.fillStyle=COLORS[p.outcome];c.beginPath();c.arc(x(p.bucket_xy_m[0]),y(p.bucket_xy_m[1]),5,0,2*Math.PI);c.fill();c.fillStyle='#496f67';c.fillText(String(i+1),x(p.bucket_xy_m[0])+7,y(p.bucket_xy_m[1])-4);}
 let left=180;for(const [key,color] of Object.entries(COLORS)){c.fillStyle=color;c.fillRect(left,8,8,8);c.fillStyle='#496f67';c.fillText(key,left+11,16);left+=95;}
}

export function renderPolicy64(root,raw){
 root.hidden=!raw||typeof raw!=='object';if(root.hidden)return;const audited=policy64AuditReady(raw),complete=policy64ComparisonReady(raw),p=raw.protocol??{},b=raw.bootstrap_summary??{},selected=raw.selected??{};
 text(root,'connection','Read-only snapshot loaded');text(root,'status',`${raw.version??'Fixed64 policy study'} · ${raw.status??'pending'} · ${raw.phase??'pending'}${raw.error?' · '+raw.error:''}${raw.reason?' · '+raw.reason:''}`);
 text(root,'scope',raw.scope??'35 TRAIN layouts bootstrap BC, then fresh fixed64 PPO;12 disjoint VAL and 50 TEST layouts. Privileged geometry and prescribed held endpoints only.');
 text(root,'plan',`${count(p.train_requests)} fresh PPO TRAIN requests / ${count(p.updates)} updates; ${count(p.validation_layouts)} fresh VAL layouts at ${Array.isArray(p.milestones)?p.milestones.join(' / '):'unknown'} samples; ${count(p.test_layouts)} TEST layouts ×4 methods. Maximum ${count(p.maximum_fresh_requests)} fresh requests including controls. Historical bootstrap costs are separate.`);
 table(node(root,'progress'),['Stage','Requested / observed','Other observations'],[['Historical TRAIN source',count(b.requests),`${count(b.layouts)} layouts; ${count(b.successful_layouts)} successful layouts; ${count(b.teacher_count)} teacher labels`],['Fresh PPO TRAIN',`${count(raw.train_requests)} / ${count(p.train_requests)}`,`${count(raw.updates)} / ${count(p.updates)} updates`],['Fresh validation',count(Array.isArray(raw.curves)&&raw.curves.every(r=>Number.isInteger(r.validation?.attempts))?raw.curves.reduce((n,r)=>n+r.validation.attempts,0):null),`${rows(raw.curves).length} saved checkpoints`],['Fresh TEST',count(Array.isArray(raw.test_comparison)&&raw.test_comparison.every(r=>Number.isInteger(r.counts?.attempts))?raw.test_comparison.reduce((n,r)=>n+r.counts.attempts,0):null),`${rows(raw.test_comparison).length} method summaries`],['Controls',`pre ${raw.control_gates?.pre?.status??'pending'} / post ${raw.control_gates?.post?.status??'pending'}`,'Separate from training and test captures']]);
 text(root,'artifacts',`Selected checkpoint: ${count(selected.samples)} fresh training samples; file SHA ${selected.sha256??'unknown'}; weight SHA ${selected.weight_sha256??'unknown'}. Historical source native ${count(b.native_releases)}, captures ${count(b.captures)}, clean ${count(b.clean)}; these are bootstrap evidence, not fresh TEST results. ${b.labels_scope??''}`);
 drawCurves(root,raw);table(node(root,'curve-table'),['Fresh TRAIN samples','VAL requested','VAL native','VAL captures/all','VAL clean/all','VAL invalid/all','Mean VAL reward'],policy64CurveRows(raw).map(r=>[count(r.samples),count(r.attempts),count(r.native),fraction(r.captures,r.attempts),fraction(r.clean,r.attempts),fraction(r.invalid,r.attempts),Number.isFinite(r.meanReward)?r.meanReward.toFixed(3):'unknown']));
 const comparison=rows(raw.test_comparison);table(node(root,'comparison'),['Method','TEST requested','Native/all','Capture/all','Clean/all','Observed rim/all','Invalid/all','Geometry rejects','Mean reward','Checkpoint samples'],METHODS.map(method=>{const r=comparison.find(row=>row.method===method)??{},c=r.counts??{};return[NAMES[method],count(c.attempts),fraction(c.native_releases,c.attempts),fraction(c.captures,c.attempts),fraction(c.clean,c.attempts),fraction(c.rim_assisted,c.attempts),fraction(c.invalid,c.attempts),count(c.geometry_rejections),Number.isFinite(c.mean_reward)?c.mean_reward.toFixed(3):'unknown',count(r.samples)];}));
 const failure=raw.status==='failed'||raw.audit?.status==='failed';
 text(root,'result',failure?'Study or audit failed; final comparison is unverified. Inspect the reported status and error.':!audited?'Live observations are provisional. Final comparison awaits the completed matching full audit.':!complete?'The full audit is present, but complete matched method/coverage/artifact evidence is missing or inconsistent.':`Completed audited fixed64 warm-start ablation: selected PPO captured ${comparison.find(r=>r.method==='ppo').counts.captures} / 50, including ${comparison.find(r=>r.method==='ppo').counts.clean} clean. Each method received 50 nominal TEST shots; historical training exposure differs. This is not a 128-substep or noise qualification.`);
 node(root,'result')?.setAttribute('data-state',failure?'failed':complete?'passed':'unknown');
 drawCoverage(root,raw);table(node(root,'coverage-table'),['#','TEST layout','Bucket X / Y mm','Outcome','Board H mm / tilt ° / gap mm','Board-bucket distance mm'],policy64Coverage(raw).map((r,i)=>[i+1,r.context_id??'unknown',finiteVector(r.bucket_xy_m,2)?r.bucket_xy_m.map(v=>mm(v)).join(' / '):'unknown',r.outcome,finiteVector(r.values,5)?`${mm(r.values[2])} / ${r.values[3].toFixed(2)} / ${mm(r.values[4])}`:'unknown',mm(r.board_bucket_distance_m)]));
 const select=node(root,'trace-select'),previous=select?.value,traces=rows(raw.trace_examples);if(select){select.replaceChildren();for(const trace of traces){const option=select.ownerDocument.createElement('option');option.value=trace.record_id??trace.id??'';option.textContent=`${trace.method??'unknown'} · ${trace.context_id??trace.record_id??trace.id??'?'} · ${trace.substeps??'?'} steps`;select.append(option);}select.disabled=!traces.length;select.value=traces.some(t=>(t.record_id??t.id)===previous)?previous:traces[0]?.record_id??traces[0]?.id??'';select.onchange=()=>drawTrace(root,raw,traces.find(t=>(t.record_id??t.id)===select.value));drawTrace(root,raw,traces.find(t=>(t.record_id??t.id)===select.value));}
 text(root,'audit',`Audit ${raw.audit?.status??'pending'}; matching full checks ${audited?'verified':'unverified'}; ${count(raw.audit?.checks_count)} checks. Study completion is separate from capture performance. Contract ${raw.contract_fingerprint??'unknown'}. ${raw.audit?.verification_reason??''}`);
}
export function startPolicy64(root,{fetchReport=async()=>{const r=await fetch('/api/study',{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json();},schedule=(fn,ms)=>setTimeout(fn,ms),cancel=id=>clearTimeout(id)}={}){let stopped=false,timer=null;const tick=async()=>{if(stopped)return;try{const report=await fetchReport();if(!stopped)renderPolicy64(root,report?.policy64_study);}catch(error){if(!stopped){root.hidden=false;text(root,'connection',`Snapshot unavailable: ${error?.message??'read failed'}`);text(root,'result','Unverified: refresh failed; displayed observations may be stale.');node(root,'result')?.setAttribute('data-state','unknown');}}finally{if(!stopped)timer=schedule(tick,5000);}};void tick();return()=>{stopped=true;if(timer!==null)cancel(timer);};}
if(typeof document!=='undefined'){const mount=()=>{const root=document.getElementById('policy64Study');if(root){const stop=startPolicy64(root);globalThis.addEventListener?.('pagehide',stop,{once:true});}};if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();}
