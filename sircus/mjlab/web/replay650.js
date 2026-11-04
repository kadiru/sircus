// Additive, read-only reader. No shared dashboard state or solver routes.
export const VERSION='sircus-calibrated-650-replay-v1';
const STEPS=['64','128'];
const ROLES={positive_control:'Positive control',negative_control:'Negative control',candidate:'650 mm candidate',window:'Conditional gap window'};
const finiteVector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
const mm=(v,d=1)=>Number.isFinite(v)?(v*1000).toFixed(d):'unknown';
const count=v=>Number.isInteger(v)&&v>=0?String(v):'unknown';
const fraction=(k,n)=>Number.isInteger(k)&&Number.isInteger(n)&&0<=k&&k<=n?(n?`${k} / ${n}`:'pending'):'unknown';
const node=(root,name)=>root.querySelector(`[data-replay650="${name}"]`);
const text=(root,name,value)=>{const n=node(root,name);if(n)n.textContent=value;};

export function replay650View(raw) {
  if(!raw||typeof raw!=='object')return null;
  const groups=Array.isArray(raw.groups)?raw.groups.filter(g=>g&&typeof g.id==='string'&&Object.hasOwn(ROLES,g.role)):[];
  return {raw,groups,initial:groups.filter(g=>g.role!=='window'),window:groups.filter(g=>g.role==='window')};
}
export function replay650AuditReady(raw) {
  const a=raw?.audit;return raw?.version===VERSION&&raw.status==='completed'&&raw.phase==='completed'&&
    typeof raw.contract_fingerprint==='string'&&raw.contract_fingerprint.length>0&&a?.status==='passed'&&a.contract_fingerprint===raw.contract_fingerprint&&
    a.full_checks_verified===true&&Number.isInteger(a.checks_count)&&a.checks_count>0&&/^[a-f0-9]{64}$/.test(a.source_sha256??'');
}
function validStats(s) {
  return s&&Number.isInteger(s.attempts)&&s.attempts===3&&Number.isInteger(s.native_releases)&&0<=s.native_releases&&s.native_releases<=3&&
    Number.isInteger(s.captures)&&0<=s.captures&&s.captures<=s.native_releases&&Number.isInteger(s.invalid)&&0<=s.invalid&&s.invalid<=3;
}
export function replay650Gate(raw) {
  const pending=reason=>({qualified:null,label:'Unverified',reason});
  if(!replay650AuditReady(raw))return pending('Qualification awaits a completed exact-version replay and matching strictly passed full audit.');
  const gate=raw.gate??{},fields=['positive_control','negative_control','candidate','physics_valid','oracle_agreement'];
  if(!fields.every(k=>typeof gate[k]==='boolean')||typeof gate.window_opened!=='boolean')return pending('One or more required gate values are missing; unknown is not a pass.');
  const driftKnown=Number.isFinite(gate.max_drift_m)&&gate.max_drift_m>=0;
  if(gate.status==='failed'&&!gate.window_opened&&(fields.some(k=>gate[k]===false)||(driftKnown&&gate.max_drift_m>.005)))
    return {qualified:false,label:'Failed',reason:gate.reason??'A matched control, candidate, physics/oracle or drift condition failed; the conditional window was not opened.'};
  const groups=replay650View(raw).initial;
  const countsMatch=['positive_control','negative_control','candidate'].every(role=>{
    const matching=groups.filter(g=>g.role===role);return matching.length===1&&STEPS.every(step=>{const s=matching[0].by_substeps?.[step];return validStats(s)&&s.native_releases===3&&s.invalid===0&&s.captures===(role==='negative_control'?0:3);});
  });
  if(gate.status==='passed'&&fields.every(k=>gate[k]===true)&&driftKnown&&gate.max_drift_m<=.005&&gate.window_opened&&countsMatch)
    return {qualified:true,label:'Passed',reason:'Matched controls and candidate passed the frozen gate. The conditional gap window is separate nominal engineering evidence.'};
  return pending('Reported gate and required counts or metadata are incomplete or inconsistent.');
}
export function replay650Trajectory(raw,trace) {
  if(!trace)return null;
  const group=(raw.groups??[]).find(g=>g.id===trace.group_id);
  const same=(a,b)=>finiteVector(a,2)&&finiteVector(b,2)&&a.every((v,i)=>Math.abs(v-b[i])<=1e-8);
  const board=trace.board_xy_m??group?.board_xy_m,bucket=trace.bucket_xy_m??group?.bucket_xy_m;
  if(!finiteVector(board,2)||!finiteVector(bucket,2)||
    (trace.board_xy_m!==undefined&&group?.board_xy_m!==undefined&&!same(trace.board_xy_m,group.board_xy_m))||
    (trace.bucket_xy_m!==undefined&&group?.bucket_xy_m!==undefined&&!same(trace.bucket_xy_m,group.bucket_xy_m))||
    !Array.isArray(trace.position_m)||trace.position_m.length<2||!trace.position_m.every(p=>finiteVector(p,3)))return null;
  const distance=Math.hypot(bucket[0]-board[0],bucket[1]-board[1]);
  if(distance<1e-9||[trace.distance_m,group?.distance_m].some(d=>d!==undefined&&(!Number.isFinite(d)||Math.abs(d-distance)>1e-6)))return null;
  const axis=[(bucket[0]-board[0])/distance,(bucket[1]-board[1])/distance],h=trace.board_height_m??group?.board_height_m;
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
function groupRows(groups) {
  return groups.flatMap(g=>STEPS.map(step=>{const s=g.by_substeps?.[step]??{};return[`${ROLES[g.role]} · ${g.id}${Number.isFinite(g.gap_m)?' · '+mm(g.gap_m,3)+' mm gap':''}`,step,
    fraction(s.native_releases,s.attempts),fraction(s.captures,s.attempts),fraction(s.whole_entries,s.attempts),fraction(s.invalid,s.attempts),fraction(s.clean,s.attempts),s.attempts===0?'pending':s.native_releases===0?'not observed':s.native_releases>0&&Number.isFinite(s.max_retention_s)?s.max_retention_s.toFixed(3)+' s':'unknown'];}));
}
function drawTrace(root,raw,trace) {
  const canvas=node(root,'trace'),c=canvas?.getContext('2d');if(!c)return;
  const w=canvas.width,h=canvas.height;c.clearRect(0,0,w,h);c.fillStyle='#edf0e7';c.fillRect(0,0,w,h);
  const view=replay650Trajectory(raw,trace);c.font='11px sans-serif';c.fillStyle='#496f67';
  if(!view){c.fillText('No unambiguous saved native trace available',40,100);text(root,'trace-caption','A matching record/group geometry and actual saved positions are required; no analytic trajectory is generated.');return;}
  const xs=view.points.map(p=>p[0]),zs=view.points.map(p=>p[1]),xl=Math.min(-.1,...xs),xh=Math.max(view.distance+.15,...xs),zl=Math.min(0,...zs,view.boardHeight??0),zh=Math.max(.3,...zs,view.boardHeight??0)+.05;
  const x=v=>55+(v-xl)/(xh-xl)*(w-80),y=v=>h-42-(v-zl)/(zh-zl)*(h-68);
  c.strokeStyle='#d3ddd0';c.lineWidth=1;c.textAlign='center';for(let i=0;i<=4;i++){const px=xl+(xh-xl)*i/4,pz=zl+(zh-zl)*i/4;c.beginPath();c.moveTo(x(px),y(zl));c.lineTo(x(px),y(zh));c.stroke();c.beginPath();c.moveTo(x(xl),y(pz));c.lineTo(x(xh),y(pz));c.stroke();c.fillText(px.toFixed(2),x(px),h-25);c.fillText(pz.toFixed(2),30,y(pz)+4);}
  c.fillText('Distance along this board → bucket axis (m)',w/2,h-7);c.textAlign='left';c.fillText('Native ball-center Z (m)',55,15);
  c.strokeStyle='#81978a';c.setLineDash([5,4]);c.beginPath();c.moveTo(x(view.distance),y(zl));c.lineTo(x(view.distance),y(zh));c.stroke();c.fillText(`Bucket center axis (${mm(view.distance)} mm)`,Math.min(x(view.distance)+6,w-190),42);c.setLineDash([]);
  if(view.boardHeight!==null){c.fillStyle='#b88a30';c.beginPath();c.arc(x(0),y(view.boardHeight),4,0,2*Math.PI);c.fill();c.fillText('Configured board center',x(0)+8,y(view.boardHeight)-7);}
  c.strokeStyle=trace.route_success===true?'#128b8c':trace.route_success===false?'#ec7657':'#6e8180';c.lineWidth=2;c.beginPath();view.points.forEach((p,i)=>{if(i)c.lineTo(x(p[0]),y(p[1]));else c.moveTo(x(p[0]),y(p[1]));});c.stroke();
  const bool=v=>v===true?'yes':v===false?'no':'unknown',invalid=trace.invalid===false||trace.invalid===''?'none':trace.invalid===undefined||trace.invalid===null?'unknown':String(trace.invalid);
  const oracle=typeof trace.oracle_agreement==='boolean'?trace.oracle_agreement:trace.scalar_score_audit_passed;
  text(root,'trace-caption',`${trace.record_id??'Unknown record'} · ${ROLES[trace.role]??trace.role??'unknown role'} · ${trace.substeps??'?'} substeps · ${view.points.length} actual saved samples · bucket XY ${view.bucket.map(v=>mm(v)).join(' / ')} mm · whole entry ${bool(trace.whole_ball_entry)} · max retention ${Number.isFinite(trace.max_retention_s)?trace.max_retention_s.toFixed(3)+' s':'unknown'} · invalid ${invalid} · invalid physics ${bool(trace.invalid_physics)} · oracle agreement ${bool(oracle)} · max off-plane ${mm(view.lateral)} mm. ${view.boardHeight===null?'Configured board height unknown; marker omitted.':'Board marker is configured geometry.'} ${replay650AuditReady(raw)?'Completed matching full audit.':'Full audit unverified.'} ${trace.selection_scope??''}`);
}
export function renderReplay650(root,raw) {
  const view=replay650View(raw);root.hidden=!view;if(!view)return;
  text(root,'connection','Read-only snapshot loaded');const gate=replay650Gate(raw),candidate=raw.settings?.candidate??{};
  text(root,'status',`${raw.version??'Calibrated replay'} · ${raw.status??'pending'} · ${raw.phase??'pending'}${raw.error?' · '+raw.error:''}${raw.reason?' · '+raw.reason:''}${raw.queue?.reason?' · '+raw.queue.reason:''}`);
  text(root,'scope',raw.scope??'Calibrated native replay and matched nominal engineering controls; no new RL or fresh-noise study.');
  text(root,'candidate',`Candidate: board H ${mm(candidate.board_height_m,3)} mm · tilt ${Number.isFinite(candidate.tilt_deg)?candidate.tilt_deg.toFixed(3):'unknown'}° · gap ${mm(candidate.gap_m,3)} mm · released ball-center H ${mm(candidate.release_center_height_m,3)} mm · bucket distance ${mm(candidate.distance_m,3)} mm.`);
  const metrics=[['All-role requests',count(raw.counts?.requests??raw.counts?.attempts)],['Native releases',count(raw.counts?.native_releases)],['Calibrated captures',count(raw.counts?.captures)],['Invalid requests',count(raw.counts?.invalid)],['Clean captures',count(raw.counts?.clean)],['Full audit',replay650AuditReady(raw)?'Verified':'Unverified / pending']];
  table(node(root,'summary'),['Metric','Observed value'],metrics);text(root,'gate',`${gate.label}: ${gate.reason}`);node(root,'gate')?.setAttribute('data-state',gate.qualified===true?'passed':gate.qualified===false?'failed':'unknown');
  const bool=v=>!replay650AuditReady(raw)?'Unverified':v===true?'Passed':v===false?'Failed':'unknown';
  table(node(root,'conditions'),['Initial gate condition','Status'],[['Positive control · 3/3 at each step',bool(raw.gate?.positive_control)],['Negative control · 0/3 at each step',bool(raw.gate?.negative_control)],['650 mm candidate · 3/3 at each step',bool(raw.gate?.candidate)],['Physics validity',bool(raw.gate?.physics_valid)],['Oracle agreement',bool(raw.gate?.oracle_agreement)],['Candidate matched 350 ms drift',Number.isFinite(raw.gate?.max_drift_m)?mm(raw.gate.max_drift_m,3)+' mm':'unknown']]);
  const headers=['Group / requested gap','Substeps','Native/all','Captures/all','Whole entry/all','Invalid/all','Clean/all','Max retention'];
  table(node(root,'controls'),headers,groupRows(view.initial.filter(g=>g.role!=='candidate')));table(node(root,'candidate-results'),headers,groupRows(view.initial.filter(g=>g.role==='candidate')));
  text(root,'window-status',!replay650AuditReady(raw)?'Conditional window qualification is unverified while the full audit is pending.':raw.gate?.window_opened===true?'Conditional window opened after the initial gate; these are three nominal repeats per gap and step.':raw.gate?.window_opened===false?'Conditional window was not opened.':'Whether the window opened is unknown.');
  table(node(root,'window'),headers,groupRows(view.window));
  const sensitivity=view.groups.map(g=>[`${ROLES[g.role]} · ${g.id}`,fraction(g.sensitivity?.capture_agreement,g.sensitivity?.paired_attempts),Number.isFinite(g.sensitivity?.nominal_position_delta_350ms_max_m)?mm(g.sensitivity.nominal_position_delta_350ms_max_m,3)+' mm':'unknown']);
  table(node(root,'sensitivity'),['Group','64↔128 outcome agreement','Nominal 350 ms drift'],sensitivity);
  const select=node(root,'trace-select'),previous=select?.value,traces=Array.isArray(raw.trace_examples)?raw.trace_examples:[];
  if(select){const doc=select.ownerDocument;select.replaceChildren();for(const trace of traces){const option=doc.createElement('option');option.value=trace.record_id??'';option.textContent=`${ROLES[trace.role]??trace.role??'?'} · ${trace.group_id??'?'} · ${trace.substeps??'?'} steps · ${trace.route_success===true?'capture':trace.route_success===false?'miss':'unknown'}`;select.append(option);}select.disabled=!traces.length;select.value=traces.some(t=>t.record_id===previous)?previous:traces[0]?.record_id??'';select.onchange=()=>drawTrace(root,raw,traces.find(t=>t.record_id===select.value));drawTrace(root,raw,traces.find(t=>t.record_id===select.value));}
  text(root,'audit',`Audit reported ${raw.audit?.status??'pending'} · full checks ${replay650AuditReady(raw)?'verified':'unverified'} · ${count(raw.audit?.checks_count)} checks. Contract ${raw.contract_fingerprint??'unknown'}. ${raw.audit?.verification_reason??''}`);
}
export function startReplay650(root,{fetchReport=async()=>{const response=await fetch('/api/study',{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.json();},schedule=(fn,ms)=>setTimeout(fn,ms),cancel=id=>clearTimeout(id)}={}) {
  let stopped=false,timer=null;
  const tick=async()=>{if(stopped)return;try{const report=await fetchReport();if(!stopped)renderReplay650(root,report?.calibrated650_replay);}catch(error){if(!stopped){root.hidden=false;text(root,'connection',`Snapshot unavailable: ${error?.message??'read failed'}`);text(root,'gate','Unverified: the snapshot could not be refreshed; existing rows may be stale.');node(root,'gate')?.setAttribute('data-state','unknown');}}finally{if(!stopped)timer=schedule(tick,5000);}};
  void tick();return()=>{stopped=true;if(timer!==null)cancel(timer);};
}
if(typeof document!=='undefined') {
  const mount=()=>{const root=document.getElementById('calibrated650Replay');if(root){const stop=startReplay650(root);globalThis.addEventListener?.('pagehide',stop,{once:true});}};
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();
}
