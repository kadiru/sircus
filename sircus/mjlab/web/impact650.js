// Additive read-only evidence view. No solver, policy or robot requests.
export const VERSION='sircus-impact650-study-v1';
export const AUDIT_VERSION='sircus-impact650-independent-audit-v1';
const STEPS=['64','128'];
const finiteVector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
const mm=(v,d=1)=>Number.isFinite(v)?(v*1000).toFixed(d):'unknown';
const count=v=>Number.isInteger(v)&&v>=0?String(v):'unknown';
const fraction=(k,n,finished=false)=>Number.isInteger(k)&&Number.isInteger(n)&&0<=k&&k<=n?(n?`${k} / ${n}`:finished?'not run':'pending'):'unknown';
const node=(root,name)=>root.querySelector(`[data-impact650="${name}"]`);
const text=(root,name,value)=>{const n=node(root,name);if(n)n.textContent=value;};
const rows=raw=>Array.isArray(raw)?raw.filter(g=>g&&typeof g==='object'):[];
const stats=g=>g.statistics??g.stats??g;
export function impact650AuditReady(raw){const a=raw?.audit;return raw?.version===VERSION&&raw.status==='completed'&&raw.phase==='completed'&&
  typeof raw.contract_fingerprint==='string'&&raw.contract_fingerprint.length>0&&a?.version===AUDIT_VERSION&&a.status==='passed'&&a.contract_fingerprint===raw.contract_fingerprint&&
  a.full_checks_verified===true&&a.bound_files_verified===true&&Number.isInteger(a.checks_count)&&a.checks_count>0&&/^[a-f0-9]{64}$/.test(a.source_sha256??'');}
export function impact650Qualification(raw,g){
  if(!impact650AuditReady(raw))return {capture:null,clean:null,label:'Unverified'};
  const q=g.qualification??{},s=g.sensitivity??{};
  const valid=STEPS.every(step=>{const v=g.by_substeps?.[step];return v&&v.attempts===3&&v.native_releases===3&&v.captures===3&&v.invalid===0;});
  const clean=STEPS.every(step=>g.by_substeps?.[step]?.clean===3);
  const common=raw.gate?.controls_pre_passed===true&&raw.gate?.controls_post_passed===true&&q.source_capture===true&&q.physics_valid===true&&q.oracle_agreement===true&&valid&&s.paired_attempts===3&&s.capture_agreement===3&&Number.isFinite(s.nominal_position_delta_350ms_max_m)&&s.nominal_position_delta_350ms_max_m>=0&&s.nominal_position_delta_350ms_max_m<=.005;
  const capture=q.capture_qualified===false?false:q.capture_qualified===true&&q.capture_both===true&&common?true:null;
  const cleanClaim=q.clean_qualified??q.qualified;
  const cleanPassed=cleanClaim===false||g.qualified===false?false:cleanClaim===true&&q.clean_both===true&&capture===true&&clean?true:null;
  return {capture,clean:cleanPassed,label:cleanPassed===true?'Capture + clean gates passed':capture===true?`Capture gate passed; clean ${cleanPassed===false?'failed':'unverified'}`:capture===false?`Capture gate failed; clean ${cleanPassed===false?'failed':'unverified'}`:'Unverified / incomplete gate'};
}
export function impact650View(raw){if(!raw||typeof raw!=='object')return null;return {raw,controls:rows(raw.controls),top:rows(raw.top_settings),verification:rows(raw.verification)};}
export function impact650Trajectory(raw,trace) {
  if(!trace)return null;
  const group=[...(raw.groups??[]),...(raw.controls??[]),...(raw.top_settings??[]),...(raw.verification??[])].find(g=>g.id===trace.group_id);
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
function drawTrace(root,raw,trace) {
  const canvas=node(root,'trace'),c=canvas?.getContext('2d');if(!c)return;
  const w=canvas.width,h=canvas.height;c.clearRect(0,0,w,h);c.fillStyle='#edf0e7';c.fillRect(0,0,w,h);
  const view=impact650Trajectory(raw,trace);c.font='11px sans-serif';c.fillStyle='#496f67';
  if(!view){c.fillText('No unambiguous saved native trace available',40,100);text(root,'trace-caption','A matching record/group geometry and actual saved positions are required; no analytic trajectory is generated.');return;}
  const xs=view.points.map(p=>p[0]),zs=view.points.map(p=>p[1]),xl=Math.min(-.1,...xs),xh=Math.max(view.distance+.15,...xs),zl=Math.min(0,...zs,view.boardHeight??0),zh=Math.max(.3,...zs,view.boardHeight??0)+.05;
  const x=v=>55+(v-xl)/(xh-xl)*(w-80),y=v=>h-42-(v-zl)/(zh-zl)*(h-68);
  c.strokeStyle='#d3ddd0';c.lineWidth=1;c.textAlign='center';for(let i=0;i<=4;i++){const px=xl+(xh-xl)*i/4,pz=zl+(zh-zl)*i/4;c.beginPath();c.moveTo(x(px),y(zl));c.lineTo(x(px),y(zh));c.stroke();c.beginPath();c.moveTo(x(xl),y(pz));c.lineTo(x(xh),y(pz));c.stroke();c.fillText(px.toFixed(2),x(px),h-25);c.fillText(pz.toFixed(2),30,y(pz)+4);}
  c.fillText('Distance along this board → bucket axis (m)',w/2,h-7);c.textAlign='left';c.fillText('Native ball-center Z (m)',55,15);
  c.strokeStyle='#81978a';c.setLineDash([5,4]);c.beginPath();c.moveTo(x(view.distance),y(zl));c.lineTo(x(view.distance),y(zh));c.stroke();c.fillText(`Bucket center axis (${mm(view.distance)} mm)`,Math.min(x(view.distance)+6,w-190),42);c.setLineDash([]);
  if(view.boardHeight!==null){c.fillStyle='#b88a30';c.beginPath();c.arc(x(0),y(view.boardHeight),4,0,2*Math.PI);c.fill();c.fillText('Configured board center',x(0)+8,y(view.boardHeight)-7);}
  c.strokeStyle=trace.route_success===true?'#128b8c':trace.route_success===false?'#ec7657':'#6e8180';c.lineWidth=2;c.beginPath();view.points.forEach((p,i)=>{if(i)c.lineTo(x(p[0]),y(p[1]));else c.moveTo(x(p[0]),y(p[1]));});c.stroke();
  const bool=v=>v===true?'yes':v===false?'no':'unknown',invalid=trace.invalid===false||trace.invalid===''||trace.invalid===null?'none':trace.invalid===undefined?'unknown':String(trace.invalid);
  const oracle=typeof trace.oracle_agreement==='boolean'?trace.oracle_agreement:trace.scalar_score_audit_passed;
  text(root,'trace-caption',`${trace.record_id??'Unknown record'} · ${trace.phase??trace.role??'unknown phase'} · ${trace.substeps??'?'} substeps · ${view.points.length} actual saved samples · configured forward / lateral center-plane tangent offsets ${xyLabel(trace.impact_offset_m)} mm · board-center distance ${mm(view.distance)} mm · nominal drop-ray-point-to-bucket horizontal distance ${mm(trace.impact_to_bucket_horizontal_m)} mm · ball-center release height ${mm(trace.release_center_height_m)} mm · board XY ${view.board.map(v=>mm(v)).join(' / ')} mm · bucket XY ${view.bucket.map(v=>mm(v)).join(' / ')} mm · whole entry ${bool(trace.whole_ball_entry)} · max retention ${Number.isFinite(trace.max_retention_s)?trace.max_retention_s.toFixed(3)+' s':'unknown'} · invalid ${invalid} · invalid physics ${bool(trace.invalid_physics)} · oracle agreement ${bool(oracle)} · max off-plane ${mm(view.lateral)} mm. ${view.boardHeight===null?'Configured board height unknown; marker omitted.':'Board marker is configured geometry.'} The nominal drop-ray point is on the board center plane; it is not a measured finite-ball contact or top-face surface point. ${impact650AuditReady(raw)?'Completed matching full audit.':'Full audit unverified.'} ${trace.selection_scope??''}`);
}
function controlsRows(groups,finished){return groups.flatMap(g=>(g.substeps?[String(g.substeps)]:STEPS).map(step=>{const s=g.substeps?stats(g):g.by_substeps?.[step]??{};return [g.stage??g.phase??'unknown',`${g.role??'control'} · ${g.group_id??g.id??'unknown'}`,step,fraction(s.native_releases,s.attempts,finished),fraction(s.captures,s.attempts,finished),fraction(s.clean,s.attempts,finished),fraction(s.invalid,s.attempts,finished)];}));}
export function impact650PhaseRows(raw){
 const explicit=raw.by_phase;if(explicit&&Object.keys(explicit).length)return Object.entries(explicit).map(([name,v])=>[name,count(v.attempts),count(v.native_releases),count(v.clean),count(v.captures),count(v.invalid)]);
 const totals={},keys=['attempts','native_releases','clean','captures','invalid'];
 const add=(phase,v)=>{if(typeof phase!=='string')return;const sum=totals[phase]??={};for(const key of keys){if(Number.isInteger(v[key])&&v[key]>=0)sum[key]=(sum[key]??0)+v[key];}};
 for(const g of rows(raw.groups)){if(String(g.role??'').endsWith('_control'))continue;for(const v of Object.values(g.by_substeps??{}))add(g.phase,v);}
 for(const g of rows(raw.controls)){if(g.substeps)add(g.phase,stats(g));}
 const requested=raw.counts?.phase_counts??{},names=Object.keys(requested).length?Object.keys(requested):Object.keys(totals);
 return names.map(name=>{const v=totals[name]??{},known=Number.isInteger(requested[name])?requested[name]:v.attempts,consistent=Number.isInteger(known)&&v.attempts===known;return[name,count(known),...keys.slice(1).map(key=>consistent?count(v[key]):'unknown')];});
}

const xyLabel=v=>finiteVector(v,2)?v.map(n=>mm(n,1)).join(' / '):'unknown';
function settingsRows(groups){return groups.map(g=>{const s=stats(g);return[g.id??'unknown',g.family_id??g.seed_id??g.source_seed_id??'unknown',xyLabel(g.impact_offset_m),mm(g.board_height_m),Number.isFinite(g.tilt_deg)?g.tilt_deg.toFixed(2):'unknown',mm(g.gap_m),mm(g.release_center_height_m),mm(g.distance_m),mm(g.impact_to_bucket_horizontal_m),fraction(s.native_releases,s.attempts),fraction(s.clean,s.attempts),fraction(s.captures,s.attempts),fraction(s.rim_assisted,s.attempts)];});}

export function renderImpact650(root,raw){
  const view=impact650View(raw);root.hidden=!view;if(!view)return;
  const finished=raw.status==='completed'&&raw.phase==='completed',verified=impact650AuditReady(raw);
  text(root,'connection','Read-only snapshot loaded');
  text(root,'status',`${raw.version??'Impact-point study'} · ${raw.status??'pending'} · ${raw.phase??'pending'}${raw.error?' · '+raw.error:''}${raw.reason?' · '+raw.reason:''}${raw.queue?.reason?' · '+raw.queue.reason:''}`);
  text(root,'scope',raw.scope??'Bounded board-surface impact-point exploration at fixed 650 mm board-center distance; nominal matched verification, without noise or new PPO.');
  const s=raw.settings??{};
  text(root,'plan',`${count(s.families)} fixed seed families; forward offsets ${Array.isArray(s.forward_m)?s.forward_m.map(v=>mm(v)).join(' / '):'unknown'} mm, lateral offsets ${Array.isArray(s.lateral_m)?s.lateral_m.map(v=>mm(v)).join(' / '):'unknown'} mm. ${count(s.coarse_requests)} coarse requests; up to ${count(s.verification_sources)} settings × ${count(s.repeats_per_resolution)} repeats × 64 / 128 substeps (${count(s.verification_requests)} requests); ${count(s.control_requests)} controls. Maximum ${count(s.max_requests)} total requests, including rejections (user cap ${count(s.user_request_cap)}); worker limit ${Number.isFinite(s.worker_wall_limit_s)?(s.worker_wall_limit_s/60).toFixed(0)+' minutes':'unknown'}.`);
  text(root,'plan-detail',`No height / tilt / gap refinement: only the prescribed forward / lateral center-plane tangent offsets change within each seed. Before/after controls remain separate from candidates. Fresh qualification is not inherited from source captures. Missing measurements remain unknown; this is finite development search, not held-out policy evaluation.`);
  const c=raw.counts??{};
  table(node(root,'summary'),['All-phase metric','Observed value'],[['Requested shots',count(c.attempts??c.requests)],['Native releases',count(c.native_releases)],['Captures',count(c.captures)],['Clean captures',count(c.clean)],['Rim-contact captures',count(c.rim_assisted)],['Invalid requests',count(c.invalid)],['Geometry rejections',count(c.geometry_rejections)],['Full audit',verified?'Verified':'Unverified / pending']]);
  table(node(root,'phases'),['Phase','Requested','Native','Clean captures','All captures','Invalid'],impact650PhaseRows(raw));
  table(node(root,'controls'),['Stage','Control','Substeps','Native/all','Captures/all','Clean/all','Invalid/all'],controlsRows(view.controls,finished));
  table(node(root,'top'),['Observed setting','Seed family','Surface F / L mm','Board H mm','Tilt °','Nominal plane gap mm','Release H mm','Board-center distance mm','Impact horizontal distance mm','Native/all','Clean/all','Capture/all','Rim/all'],settingsRows(view.top));
  const verification=view.verification.flatMap(g=>STEPS.map(step=>{const v=g.by_substeps?.[step]??{},q=impact650Qualification(raw,g),d=g.sensitivity?.nominal_position_delta_350ms_max_m;return [g.id??'unknown',g.family_id??g.seed_id??g.source_seed_id??'unknown',xyLabel(g.impact_offset_m),`${mm(g.board_height_m)} / ${Number.isFinite(g.tilt_deg)?g.tilt_deg.toFixed(2):'?'} / ${mm(g.gap_m)}`,mm(g.release_center_height_m),mm(g.impact_to_bucket_horizontal_m),step,fraction(v.native_releases,v.attempts,finished),fraction(v.clean,v.attempts,finished),fraction(v.captures,v.attempts,finished),fraction(v.rim_assisted,v.attempts,finished),fraction(v.invalid,v.attempts,finished),Number.isFinite(d)?mm(d,3):'unknown',q.label];}));
  table(node(root,'verification'),['Setting','Seed family','Surface F / L mm','H mm / tilt ° / nominal plane gap mm','Release H mm','Impact horizontal distance mm','Substeps','Native/all','Clean/all','Captures/all','Rim/all','Invalid/all','Δ350 ms mm','Qualification'],verification);
  const cleanQualified=view.verification.filter(g=>impact650Qualification(raw,g).clean===true),captureQualified=view.verification.filter(g=>impact650Qualification(raw,g).capture===true);
  const classified=Number.isInteger(s.verification_sources)&&view.verification.length<=s.verification_sources&&raw.gate?.qualified_settings===cleanQualified.length&&raw.gate?.capture_qualified_settings===captureQualified.length&&view.verification.every(g=>{const q=impact650Qualification(raw,g);return q.capture!==null&&q.clean!==null;});
  text(root,'result',!verified?'Qualification awaits the completed matching full audit.':raw.gate?.controls_pre_passed!==true||raw.gate?.controls_post_passed!==true?'Matched controls did not pass or remain unknown; candidate qualification is not established.':!classified?'The evidence is audited, but candidate qualification is incomplete or inconsistent.':!view.verification.length?'No candidate advanced to matched verification; no configuration qualified in this finite study.':`${captureQualified.length} / ${view.verification.length} settings passed the nominal capture gate; ${cleanQualified.length} / ${view.verification.length} passed the clean gate. Three identical nominal repeats at each resolution are not noise robustness. This finite check does not establish impossibility or continuous workspace coverage.`);
  node(root,'result')?.setAttribute('data-state',verified&&classified?(cleanQualified.length?'passed':captureQualified.length?'capture-only':'failed'):'unknown');
  const select=node(root,'trace-select'),previous=select?.value,traces=rows(raw.trace_examples);
  if(select){const doc=select.ownerDocument;select.replaceChildren();for(const trace of traces){const option=doc.createElement('option');option.value=trace.record_id??'';option.textContent=`${trace.phase??trace.role??'unknown'} · ${trace.group_id??trace.record_id??'?'} · ${trace.substeps??'?'} steps · ${trace.route_quality===2?'clean capture':trace.route_success===true?'capture':trace.route_success===false?'miss':'unknown'}`;select.append(option);}select.disabled=!traces.length;select.value=traces.some(t=>t.record_id===previous)?previous:traces[0]?.record_id??'';select.onchange=()=>drawTrace(root,raw,traces.find(t=>t.record_id===select.value));drawTrace(root,raw,traces.find(t=>t.record_id===select.value));}
  text(root,'audit',`Audit reported ${raw.audit?.status??'pending'} · full checks ${verified?'verified':'unverified'} · ${count(raw.audit?.checks_count)} checks. Contract ${raw.contract_fingerprint??'unknown'}. Physical scene ${raw.physical_scene_fingerprint??'unknown'}; action-bound scene ${raw.scene_fingerprint??raw.scene_fp??'unknown'}. ${raw.audit?.verification_reason??''}`);
}
export function startImpact650(root,{fetchReport=async()=>{const response=await fetch('/api/study',{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.json();},schedule=(fn,ms)=>setTimeout(fn,ms),cancel=id=>clearTimeout(id)}={}){
 let stopped=false,timer=null;
 const tick=async()=>{if(stopped)return;try{const report=await fetchReport();if(!stopped)renderImpact650(root,report?.impact650_study);}catch(error){if(!stopped){root.hidden=false;text(root,'connection',`Snapshot unavailable: ${error?.message??'read failed'}`);text(root,'result','Unverified: refresh failed; existing observations may be stale.');node(root,'result')?.setAttribute('data-state','unknown');}}finally{if(!stopped)timer=schedule(tick,5000);}};
 void tick();return()=>{stopped=true;if(timer!==null)cancel(timer);};
}
if(typeof document!=='undefined'){const mount=()=>{const root=document.getElementById('impact650Study');if(root){const stop=startImpact650(root);globalThis.addEventListener?.('pagehide',stop,{once:true});}};if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();}
