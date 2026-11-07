// Read-only inspection of authenticated recordings; never activates a solver.
export const VERSION = 'sircus-consistent-flight-diagnosis-v1';
export const AUDIT_VERSION = 'sircus-consistent-flight-diagnosis-checks-v1';
export const KEY = 'consistent_flight_diagnosis';
export const SOURCE_REPORT_SHA256 = '62e0f18d2fcaaa4ffb69aae4c583651bce4b54bd984e628c943ccd8dadb39b38';
export const SOURCE_AUDIT_SHA256 = '43a5c6e2dc0e204e3b646b9da7a74ca09e53a420e3869834d12da255e2029952';
export const SOURCE_FINGERPRINT = '5da3a40113e40e580c85ef6b74f501f2005ad684c6e54eae315d092f6020cc70';
export const METHODS = ['bc', 'ppo', 'ensemble', 'consistent'];
const names = {bc:'Imitation', ppo:'PPO', ensemble:'Original ensemble', consistent:'Flight-consistent rank'};
const finite = v => typeof v === 'number' && Number.isFinite(v);
const integer = v => Number.isInteger(v) && v >= 0;
const hash = v => typeof v === 'string' && /^[a-f0-9]{64}$/.test(v);
const rows = v => Array.isArray(v) ? v : [];
const vector = (v,n) => Array.isArray(v) && v.length === n && v.every(finite);
const fmt = (v,n=2) => finite(v) ? v.toFixed(n) : 'unknown';
const mm = v => finite(v) ? fmt(v*1000,1) : 'unknown';
const count = v => integer(v) ? String(v) : 'unknown';
const pct = v => finite(v) ? `${fmt(v*100,1)}%` : 'unknown';
const truth = v => v === true ? 'yes' : v === false ? 'no' : 'unknown';
const at = (root,name) => root.querySelector(`[data-flight-residual="${name}"]`);
function text(root,name,value) { const n=at(root,name); if(n)n.textContent=value; }
function table(root,name,headers,data) {
  const host=at(root,name); if(!host)return;
  const doc=host.ownerDocument, t=doc.createElement('table'), head=doc.createElement('thead'), tr=doc.createElement('tr'), body=doc.createElement('tbody');
  for(const value of headers) { const c=doc.createElement('th'); c.textContent=value; tr.append(c); } head.append(tr);
  for(const row of data) { const line=doc.createElement('tr'); for(const value of row) { const c=doc.createElement('td'); c.textContent=value==null?'unknown':String(value); line.append(c); } body.append(line); }
  t.append(head,body); host.replaceChildren(t);
}
function list(root,name,values) { const host=at(root,name); if(host)host.replaceChildren(...rows(values).map(v=>{const li=host.ownerDocument.createElement('li');li.textContent=typeof v==='string'?v:v?.description||'unknown';return li;})); }
const outcome = o => o?.native_release===false ? 'setup rejected' : o?.native_release!==true ? 'unknown' : o.invalid_physics===true ? 'invalid native' : o.capture===true ? 'capture' : o.capture===false ? o.whole_entry===true ? 'route failed after entry' : 'native flight miss' : 'unknown';
const shot = v => vector(v,5) ? `${mm(v[0])} / ${mm(v[1])} / ${mm(v[2])} / ${fmt(v[3],2)}° / ${mm(v[4])}` : 'unknown';
function validCounts(c) { return c && ['requested','native_releases','captures','clean','rejected','native_misses','invalid'].every(k=>integer(c[k])) && c.native_releases+c.rejected===c.requested && c.captures+c.native_misses+c.invalid===c.native_releases && c.clean<=c.captures; }

export function flightResidualSourceReady(raw) {
  const s=raw?.source, summary=raw?.summary, comparisons=rows(raw?.comparisons), requests=rows(raw?.requests);
  if(raw?.version!==VERSION || s?.verified!==true || s.report_sha256!==SOURCE_REPORT_SHA256 || s.audit_sha256!==SOURCE_AUDIT_SHA256 || s.contract_fingerprint!==SOURCE_FINGERPRINT || !integer(s.original_native_audit_checks) || s.original_native_audit_checks===0 || s.native_requests_added!==0)return false;
  if(comparisons.length!==4 || new Set(comparisons.map(r=>r?.method)).size!==4 || !comparisons.every(r=>METHODS.includes(r?.method)&&validCounts(r?.counts)))return false;
  const c=comparisons.find(r=>r.method==='consistent').counts;
  if(!summary || !['requested','native_releases','captures','native_misses','rejected'].every(k=>summary[k]===c[k]))return false;
  if(requests.length!==c.requested || new Set(requests.map(r=>r?.context_id)).size!==requests.length || !requests.every(r=>typeof r?.context_id==='string'&&typeof r.observed?.native_release==='boolean'&&typeof r.observed?.capture==='boolean'&&(!r.observed.capture||r.observed.native_release)))return false;
  const misses=requests.filter(r=>r.observed.native_release&&!r.observed.capture&&r.observed.invalid_physics!==true);
  return requests.filter(r=>r.observed.capture).length===c.captures && requests.filter(r=>!r.observed.native_release).length===c.rejected && misses.length===c.native_misses
    && summary.no_entry_failures===misses.filter(r=>r.observed.whole_entry===false).length
    && summary.bucket_retention_but_route_failures===misses.filter(r=>r.observed.bucket_capture===true).length
    && summary.all_bank_margins_negative_in_misses===misses.filter(r=>r.candidate_coverage?.all_available_margins_negative===true).length;
}
export function flightResidualReady(raw) {
  const a=raw?.audit;
  return flightResidualSourceReady(raw) && raw.status==='completed' && a?.version===AUDIT_VERSION && a.status==='passed' && a.full_checks_verified===true && hash(a.source_sha256) && integer(a.checks_count) && a.checks_count>0 && a.source_report_sha256===raw.source.report_sha256 && a.source_audit_sha256===raw.source.audit_sha256;
}

export function flightResidualView(example) {
  if(!example || !vector(example.values,5) || !vector(example.bucket_xy_m,2))return null;
  const trace=example.trace, points=trace?.position_m, times=trace?.time_s;
  if(!trace || trace.record_id!==example.record_id || trace.record_sha256!==example.record_sha256 || !hash(example.record_sha256) || !hash(trace.native_cache_sha256))return null;
  if(!Array.isArray(points) || points.length<2 || points.length>241 || !points.every(p=>vector(p,3)) || !Array.isArray(times) || times.length!==points.length || !times.every(finite) || times[0]<0 || times.some((v,i)=>i>0&&v<=times[i-1]))return null;
  const origin=example.values.slice(0,2), delta=example.bucket_xy_m.map((v,i)=>v-origin[i]), distance=Math.hypot(...delta); if(distance<1e-8)return null;
  const axis=delta.map(v=>v/distance), project=p=>[(p[0]-origin[0])*axis[0]+(p[1]-origin[1])*axis[1],-(p[0]-origin[0])*axis[1]+(p[1]-origin[1])*axis[0],p[2]];
  const forecasts=rows(example.forecasts).filter(f=>typeof f?.label==='string'&&typeof f.basis==='string'&&Array.isArray(f.position_m)&&f.position_m.length===points.length&&f.position_m.every(p=>vector(p,3))&&Array.isArray(f.time_s)&&f.time_s.length===times.length&&f.time_s.every((v,i)=>v===times[i])).map(f=>({label:f.label,basis:f.basis,points:f.position_m.map(project)}));
  return {actual:points.map(project),forecasts,distance,times};
}
function plot(root,name,series,vertical) {
  const canvas=at(root,name),ctx=canvas?.getContext?.('2d'); if(!ctx)return;
  const w=canvas.width,h=canvas.height,pad=50;ctx.clearRect(0,0,w,h);ctx.font='12px system-ui';
  const points=series.flatMap(s=>s.points);
  if(!points.length){ctx.fillStyle='#a6bbc8';ctx.fillText('No matching recorded flight is available.',pad,55);return;}
  let x0=Math.min(...points.map(p=>p[0])),x1=Math.max(...points.map(p=>p[0])),y0=Math.min(...points.map(p=>p[1])),y1=Math.max(...points.map(p=>p[1]));
  if(x1-x0<.1)x1=x0+.1;if(y1-y0<.1)y1=y0+.1;const extra=(y1-y0)*.12;y0-=extra;y1+=extra;
  const x=v=>pad+(v-x0)/(x1-x0)*(w-2*pad),y=v=>h-pad-(v-y0)/(y1-y0)*(h-2*pad);
  ctx.strokeStyle='#324654';ctx.lineWidth=1;ctx.setLineDash?.([]);
  for(let i=0;i<=4;i++){const xv=x0+(x1-x0)*i/4,yv=y0+(y1-y0)*i/4;ctx.beginPath();ctx.moveTo(x(xv),pad);ctx.lineTo(x(xv),h-pad);ctx.stroke();ctx.fillStyle='#9eb4c3';ctx.fillText(fmt(xv),x(xv)-15,h-30);ctx.fillText(fmt(yv),5,y(yv)+4);}
  for(const s of series){ctx.strokeStyle=s.color;ctx.lineWidth=s.dashed?2:3;ctx.setLineDash?.(s.dashed?[7,5]:[]);ctx.beginPath();s.points.forEach((p,i)=>i?ctx.lineTo(x(p[0]),y(p[1])):ctx.moveTo(x(p[0]),y(p[1])));ctx.stroke();}
  ctx.setLineDash?.([]);ctx.fillStyle='#afc4d0';ctx.fillText('Board → bucket distance (m)',Math.max(pad,w/2-85),h-8);ctx.fillText(vertical?'Ball-center height (m)':'Lateral displacement (m)',pad,18);
}
function drawCase(root,c) {
  if(!c){text(root,'case-result','No recorded example is available.');for(const n of ['configuration','prediction','references'])table(root,n,[],[]);for(const n of ['selection','candidate-coverage','diagnosis'])text(root,n,'unknown');plot(root,'flight',[],true);plot(root,'lateral',[],false);text(root,'flight-caption','Missing data stays unknown; no flight is invented.');return;}
  const o=c.observed||{},p=c.prediction||{},v=c.values,view=flightResidualView(c);
  text(root,'case-result',`${c.context_id||c.id||'unknown'} · ${outcome(o)} · whole-ball entry ${truth(o.whole_entry)} · retention ${fmt(o.retention_s,3)} s.`);
  table(root,'configuration',['Bucket X / Y mm','Board X / Y mm','Board height mm','Tilt °','Drop gap mm'],[[vector(c.bucket_xy_m,2)?c.bucket_xy_m.map(mm).join(' / '):'unknown',vector(v,5)?v.slice(0,2).map(mm).join(' / '):'unknown',mm(v?.[2]),fmt(v?.[3],3),mm(v?.[4])]]);
  table(root,'prediction',['Quantity','Prediction / nominal geometry','Recorded observation'],[
    ['Capture head score (uncalibrated diagnostic)',pct(p.capture_score),outcome(o)],
    ...['Forward rebound m/s','Lateral rebound m/s','Vertical rebound m/s'].map((name,i)=>[name,fmt(p.rebound_velocity_m_s?.[i],4),fmt(o.rebound_velocity_m_s?.[i],4)]),
    ['Conservative whole-ball rim-band margin mm',mm(p.proxy_aperture_margin_m),'Different from center-at-rim margin'],
    ['Center-at-rim clearance mm','Different from rim-band margin',mm(o.mouth_clearance_m)],
    ['Nominal origin − measured rebound XYZ mm','—',vector(o.nominal_origin_minus_observed_rebound_mm,3)?o.nominal_origin_minus_observed_rebound_mm.map(x=>fmt(x,2)).join(' / '):'unknown'],
    ['First-flight contact cutoff','—',Array.isArray(o.cutoff_contact_channels)?o.cutoff_contact_channels.join(', ')||'none':'unknown']]);
  if(view){const colors=['#d79cec','#dfbb7c','#88aaff'],labels=['purple','gold','blue'];const series=[{points:view.actual,color:'#78d1d5'},...view.forecasts.map((f,i)=>({...f,color:colors[i%colors.length],dashed:true}))];
    const marker=finite(c.bucket_rim_z_m)?[{points:[[view.distance,0],[view.distance,c.bucket_rim_z_m]],color:'#8a9baa'}]:[];
    plot(root,'flight',[...series.map(s=>({...s,points:s.points.map(p=>[p[0],p[2]])})),...marker],true);plot(root,'lateral',series.map(s=>({...s,points:s.points.map(p=>[p[0],p[1]])})),false);
    text(root,'flight-caption',`Solid teal: ${view.actual.length} saved first-flight samples, ${fmt(view.times[0],4)}–${fmt(view.times.at(-1),4)} s. ${view.forecasts.map((f,i)=>`Dashed ${labels[i%labels.length]}: ${f.label}; ${f.basis}.`).join(' ')||'No matching conditional forecast was supplied.'} Native record ${c.record_id}; cache ${c.trace.native_cache_sha256}. These projections do not reconstruct later collisions or retention.`);
  }else{plot(root,'flight',[],true);plot(root,'lateral',[],false);text(root,'flight-caption','No identity-matched finite native trace is available for this request. No flight is invented.');}
  const s=c.selection||{};text(root,'selection',`Candidate ${s.selected_candidate_id||'unknown'} · rank ${count(s.rank)} · geometry checks ${count(s.geometry_checks)}. Geometry reason: ${s.geometry_reason||'none supplied'}. Context reason: ${s.context_reason||'none supplied'}. The capture head is diagnostic only, not a calibrated selector probability.`);
  const bank=c.candidate_coverage||{};text(root,'candidate-coverage',`Frozen proposal bank: ${count(bank.candidates)} candidates; ${count(bank.nonnegative_proxy_candidates)} predicted nonnegative rim-band margins. Best model margin ${mm(bank.best_proxy_margin_m)} mm. Drop gaps span ${mm(bank.gap_min_m)}–${mm(bank.gap_max_m)} mm; anchor ${mm(bank.anchor_gap_m)} mm. This describes model proposals, not measured feasibility or untried-shot success.`);
  text(root,'diagnosis',typeof c.diagnosis==='string'?c.diagnosis:'No diagnostic interpretation supplied.');
  table(root,'references',['Frozen method','Board X / Y / height mm / tilt / gap mm','Observed outcome','Whole-ball entry','Retention s'],rows(c.references).map(r=>[names[r?.method]||r?.method||'unknown',shot(r?.values),outcome(r?.observed),truth(r?.observed?.whole_entry),fmt(r?.observed?.retention_s,3)]));
}

export function renderFlightResidualDiagnosis(root,raw,correction) {
  if(!raw||raw.version!==VERSION){root.hidden=true;return;}root.hidden=false;
  const ready=flightResidualReady(raw),source=flightResidualSourceReady(raw),failed=raw.status==='failed'||raw.audit?.status==='failed',s=raw.summary||{};
  text(root,'status',ready?'Recorded comparison and separate diagnostic checks verified. No new policy or native trial is claimed.':failed?'Diagnostic verification failed. Findings remain unverified.':'Provisional diagnosis: matching source and reconstruction checks are required.');at(root,'status')?.setAttribute('data-state',ready?'passed':failed?'failed':'unknown');text(root,'scope',raw.scope||'Post-hoc recorded-flight inspection; no new training or native evaluation.');
  table(root,'comparison',['Frozen method','Captures / requested','Capture rate','Clean','Setup rejects','Native misses','Invalid'],METHODS.map(m=>{const c=rows(raw.comparisons).find(r=>r?.method===m)?.counts||{};return[names[m],`${count(c.captures)} / ${count(c.requested)}`,integer(c.requested)&&c.requested>0&&integer(c.captures)?pct(c.captures/c.requested):'unknown',count(c.clean),count(c.rejected),count(c.native_misses),count(c.invalid)];}));
  text(root,'comparison-total',`Original completed experiment: ${count(raw.source?.original_total_requests)} total requests including controls; ${count(raw.source?.original_native_releases)} native releases. The four methods below use paired bucket positions.`);
  text(root,'captures',`${count(s.captures)} / ${count(s.requested)}`);text(root,'misses',count(s.native_misses));text(root,'rejects',count(s.rejected));
  text(root,'funnel',`Native failures: ${count(s.no_entry_failures)} without whole-ball entry; ${count(s.bucket_retention_but_route_failures)} with bucket retention but a failed board route. In ${count(s.all_bank_margins_negative_in_misses)} miss contexts, every available candidate margin was negative in the frozen model. Recorded first-flight cutoff contacts: ${s.cutoff_contact_counts?JSON.stringify(s.cutoff_contact_counts):'unknown'}. Setup rejection reasons: ${s.rejection_reason_counts?JSON.stringify(s.rejection_reason_counts):'unknown'}. Entry and retention remain separate from complete board-route success.`);
  const failures=rows(raw.requests).filter(r=>r?.observed?.capture===false);
  table(root,'failures',['Context','Observed outcome','Whole entry','Retention s','Predicted rim-band margin mm','Contact cutoff / rejection'],failures.map(r=>[r.context_id||'unknown',outcome(r.observed),truth(r.observed?.whole_entry),fmt(r.observed?.retention_s,3),mm(r.prediction?.proxy_aperture_margin_m),r.observed?.native_release===false?r.selection?.geometry_reason||r.selection?.context_reason||'unknown':Array.isArray(r.observed?.cutoff_contact_channels)?r.observed.cutoff_contact_channels.join(', ')||'none':'unknown']));
  const cases=rows(raw.cases).filter(c=>c&&typeof c==='object'),select=at(root,'case-select');if(select){const old=select.value;select.replaceChildren(...cases.map((c,i)=>{const n=select.ownerDocument.createElement('option');n.value=String(c.id||c.context_id||i);n.textContent=`${c.context_id||c.id||i} · ${outcome(c.observed)}`;return n;}));select.value=cases.some((c,i)=>String(c.id||c.context_id||i)===old)?old:cases.length?String(cases[0].id||cases[0].context_id||0):'';select.disabled=!cases.length;const selected=()=>cases.find((c,i)=>String(c.id||c.context_id||i)===select.value);select.onchange=()=>drawCase(root,selected());drawCase(root,selected());}
  renderGapCoverage(root,correction);
  table(root,'findings',['Finding','Evidence','Status / limitation'],rows(raw.findings).map(r=>[r?.statement||'unknown',r?.evidence||'unknown',r?.status||'unknown']));list(root,'next-steps',raw.next_steps);
  text(root,'audit',`Original native comparison ${source?'verified':'unverified'}: report ${raw.source?.report_sha256||'unknown'}, audit ${raw.source?.audit_sha256||'unknown'}, ${count(raw.source?.original_native_audit_checks)} native checks. Separate diagnostic reconstruction ${ready?'verified':'unverified'}: ${raw.audit?.source_sha256||'unknown'}, ${count(raw.audit?.checks_count)} checks. Original labels and results are preserved.`);
}
export function gapCoverageReady(raw) {
  const p=raw?.protocol,a=raw?.audit,c=rows(raw?.comparisons);
  return raw?.version==='sircus-gap-coverage-study-v1' && ['completed','verified'].includes(raw.status) && p?.test_layouts===50 && p.maximum_fresh_requests===112 && p.substeps===64 && JSON.stringify(p.methods)===JSON.stringify(['consistent','gap_coverage']) && hash(raw.report_sha256) && a?.passed===true && a.report_sha256===raw.report_sha256 && integer(a.checks_count) && a.checks_count>0 && c.length===2 && new Set(c.map(r=>r?.method)).size===2 && c.every(r=>['consistent','gap_coverage'].includes(r?.method)&&validCounts(r?.counts)&&r.counts.requested===50);
}
export function gapCoveragePaired(raw) {
  if(!gapCoverageReady(raw))return null;
  const positions=rows(raw.test_positions),methods=['consistent','gap_coverage'];
  if(positions.length!==100 || !positions.every(r=>r&&methods.includes(r.method)&&typeof r.context_id==='string'&&r.context_id.length>0&&typeof r.record_id==='string'&&r.record_id.length>0&&vector(r.bucket_xy_m,2)&&vector(r.values,5)&&typeof r.capture==='boolean'&&typeof r.native_release==='boolean'&&typeof r.clean==='boolean'&&(!r.capture||r.native_release)&&(!r.clean||r.capture)) || new Set(positions.map(r=>r.record_id)).size!==100)return null;
  const groups=methods.map(method=>positions.filter(r=>r.method===method));
  if(!groups.every((group,i)=>{const c=raw.comparisons.find(r=>r.method===methods[i]).counts;return group.length===50&&new Set(group.map(r=>r.context_id)).size===50&&group.filter(r=>r.capture).length===c.captures&&group.filter(r=>r.native_release).length===c.native_releases&&group.filter(r=>r.clean).length===c.clean;}))return null;
  const baseline=new Map(groups[0].map(r=>[r.context_id,r])),result={both_capture:0,both_no_capture:0,gained_capture:0,lost_capture:0,changed_actions:0,positions:50};
  for(const expanded of groups[1]){const original=baseline.get(expanded.context_id);if(!original||!expanded.bucket_xy_m.every((v,i)=>v===original.bucket_xy_m[i]))return null;
    result[original.capture?(expanded.capture?'both_capture':'lost_capture'):(expanded.capture?'gained_capture':'both_no_capture')]++;
    if(expanded.values.some((v,i)=>v!==original.values[i]))result.changed_actions++;
  }
  return result;
}
function renderGapCoverage(root,raw) {
  const ready=gapCoverageReady(raw),known=raw?.version==='sircus-gap-coverage-study-v1',failed=known&&(raw.status==='failed'||raw.audit?.passed===false);
  const comparison=method=>known?rows(raw.comparisons).find(r=>r?.method===method)?.counts||{}:{};
  const baseline=comparison('consistent'),expanded=comparison('gap_coverage');
  const interpretation=ready&&baseline.captures===expanded.captures?` No observed capture gain: both methods captured ${count(baseline.captures)} / ${count(baseline.requested)} (${pct(baseline.captures/baseline.requested)}). This single comparison does not establish a winner or generalization.`:'';
  const paired=gapCoveragePaired(raw),pairedText=paired?` Paired positions: ${paired.both_capture} captured by both; ${paired.both_no_capture} without capture by either. Expanded gap proposals gained ${paired.gained_capture} capture(s) and lost ${paired.lost_capture}; ${paired.changed_actions} / ${paired.positions} selected configurations changed.`:'';
  const progress=(value,total)=>`${integer(value)&&value<=total?value:'unknown'} / ${total}`;
  const wait=finite(raw?.queue?.wait_s)&&raw.queue.wait_s>=0?` (${fmt(raw.queue.wait_s,0)} s elapsed)`:'';
  const detail=raw?.status==='preparing'?`preparing ${progress(raw.planned_layouts,50)} layouts`
    :raw?.status==='queued'?`${progress(raw.planned_layouts,50)} layouts prepared; waiting for RTX 4090${wait}`
    :raw?.status==='running'?`running ${progress(raw.counts?.requested,112)} requests`
    :raw?.status||'unknown';
  const state=ready?`Fresh gap-coverage comparison completed and independently verified.${interpretation}${pairedText}`
    :known&&raw.status==='completed'&&raw.audit?.passed===false?'Native comparison completed, but its audit failed; results remain unverified.'
    :known&&raw.status==='failed'?'Gap-coverage run failed; results remain provisional.'
    :failed?'Gap-coverage audit failed; results remain unverified.'
    :known?`Gap-coverage comparison: ${detail}; results are provisional until the matching audit passes.`:'The gap-coverage correction has no published native result yet.';
  text(root,'correction-status',state);at(root,'correction-status')?.setAttribute('data-state',ready?'passed':failed?'failed':'unknown');
  table(root,'correction',['Frozen method','Captures / requested','Capture rate','Clean','Setup rejects','Native misses'],['consistent','gap_coverage'].map(method=>{const c=comparison(method);return[method==='consistent'?'Original flight-consistent rank':'Expanded gap proposals',`${count(c.captures)} / ${count(c.requested)}`,integer(c.requested)&&c.requested>0&&integer(c.captures)?pct(c.captures/c.requested):'unknown',count(c.clean),count(c.rejected),count(c.native_misses)];}));
}
export function startFlightResidualDiagnosis(root,{fetchReport=async()=>{const r=await fetch('/api/study',{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json();},schedule=setTimeout,cancel=clearTimeout}={}) {
  let stopped=false,timer;
  const poll=async()=>{try{const report=await fetchReport();if(stopped)return;renderFlightResidualDiagnosis(root,report?.[KEY],report?.gap_coverage_study);text(root,'connection','Read-only evidence · refresh every 5 seconds.');}catch(error){if(stopped)return;text(root,'connection',`Snapshot unavailable: ${error.message||error}. Previous findings may be stale.`);text(root,'status','Connection unavailable; previous displayed findings are stale and unverified.');at(root,'status')?.setAttribute('data-state','unknown');text(root,'correction-status','Connection unavailable; previous correction results are stale and unverified.');at(root,'correction-status')?.setAttribute('data-state','unknown');}if(!stopped)timer=schedule(poll,5000);};void poll();return()=>{stopped=true;if(timer!==undefined)cancel(timer);};
}
if(typeof document!=='undefined'){const mount=()=>{const root=document.getElementById('flightResidualDiagnosis');if(root){const stop=startFlightResidualDiagnosis(root);globalThis.addEventListener?.('pagehide',stop,{once:true});}};if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();}
