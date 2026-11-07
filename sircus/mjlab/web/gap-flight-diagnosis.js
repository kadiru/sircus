// Read-only inspection of authenticated recordings; never activates a solver.
export const VERSION = 'sircus-gap-flight-diagnosis-v1';
export const AUDIT_VERSION = 'sircus-gap-flight-diagnosis-checks-v1';
export const KEY = 'gap_flight_diagnosis';
export const SOURCE_REPORT_SHA256 = '110b8b7f48f9f0cc153e0dc0c9341ed78bb04d861090abef592bfc98f2f5bf1b';
export const SOURCE_AUDIT_SHA256 = '5c6bb8965ef264f9744c0f72c6da5c5bb4904d724938519eb21f45952d1c925a';
export const SOURCE_MANIFEST_SHA256 = '7af5b7087d3ae24b47a2949e5a5cf2cd6c9a783411387186b0926fd2ae9e091e';
export const SOURCE_FINGERPRINT = 'efaa6a9dfbd5711ce84b000b7fbb6a74a2e63a5df79c930c6410fabcd20f27c4';
export const METHODS = ['consistent', 'gap_coverage'];
const names = {consistent:'Original flight-consistent selector', gap_coverage:'Expanded gap proposals'};
const finite = v => typeof v === 'number' && Number.isFinite(v);
const integer = v => Number.isInteger(v) && v >= 0;
const hash = v => typeof v === 'string' && /^[a-f0-9]{64}$/.test(v);
const rows = v => Array.isArray(v) ? v : [];
const vector = (v,n) => Array.isArray(v) && v.length === n && v.every(finite);
const fmt = (v,n=2) => finite(v) ? v.toFixed(n) : 'unknown';
const mm = v => finite(v) ? fmt(v*1000,1) : 'unknown';
const preciseMm = v => finite(v) ? fmt(v*1000,3) : 'unknown';
const count = v => integer(v) ? String(v) : 'unknown';
const pct = v => finite(v) ? `${fmt(v*100,1)}%` : 'unknown';
const truth = v => v === true ? 'yes' : v === false ? 'no' : 'unknown';
const at = (root,name) => root.querySelector(`[data-gap-flight="${name}"]`);
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

export function gapFlightSourceReady(raw) {
  const s=raw?.source, summary=raw?.summary, comparisons=rows(raw?.comparisons), requests=rows(raw?.requests);
  if(raw?.version!==VERSION || s?.verified!==true || s.report_sha256!==SOURCE_REPORT_SHA256 || s.audit_sha256!==SOURCE_AUDIT_SHA256 || s.manifest_sha256!==SOURCE_MANIFEST_SHA256 || s.contract_fingerprint!==SOURCE_FINGERPRINT || s.original_native_audit_checks!==1441075 || s.original_total_requests!==112 || s.original_native_releases!==96 || s.native_requests_added!==0)return false;
  if(comparisons.length!==2 || new Set(comparisons.map(r=>r?.method)).size!==2 || !comparisons.every(r=>METHODS.includes(r?.method)&&validCounts(r?.counts)&&r.counts.requested===50))return false;
  const c=comparisons.find(r=>r.method==='consistent').counts;
  if(!summary || !['requested','native_releases','captures','native_misses','rejected'].every(k=>summary[k]===c[k]))return false;
  if(requests.length!==c.requested || new Set(requests.map(r=>r?.context_id)).size!==requests.length || new Set(requests.map(r=>r?.record_id)).size!==requests.length || !requests.every(r=>typeof r?.context_id==='string'&&r.context_id.length>0&&typeof r.record_id==='string'&&r.record_id.length>0&&hash(r.record_sha256)&&vector(r.values,5)&&vector(r.bucket_xy_m,2)&&typeof r.observed?.native_release==='boolean'&&typeof r.observed?.capture==='boolean'&&(!r.observed.capture||r.observed.native_release&&r.observed.invalid_physics!==true)))return false;
  const misses=requests.filter(r=>r.observed.native_release&&!r.observed.capture&&r.observed.invalid_physics!==true);
  return requests.filter(r=>r.observed.capture).length===c.captures && requests.filter(r=>!r.observed.native_release).length===c.rejected && misses.length===c.native_misses
    && summary.no_entry_failures===misses.filter(r=>r.observed.whole_entry===false).length
    && summary.bucket_retention_but_route_failures===misses.filter(r=>r.observed.bucket_capture===true).length
    && summary.early_robot_contact_ms_threshold===50
    && summary.early_robot_contact_count===misses.filter(r=>r.observed.early_robot_contact===true).length;
}
export function gapFlightReady(raw) {
  const a=raw?.audit;
  return gapFlightSourceReady(raw) && boundCases(raw) && raw.status==='completed' && a?.version===AUDIT_VERSION && a.status==='passed' && a.full_checks_verified===true && hash(a.source_sha256) && integer(a.checks_count) && a.checks_count>0 && a.source_report_sha256===raw.source.report_sha256 && a.source_audit_sha256===raw.source.audit_sha256;
}
function boundCases(raw) {
  const requests=new Map(rows(raw?.requests).map(r=>[r?.context_id,r]));
  return rows(raw?.cases).every(c=>{const r=requests.get(c?.context_id);return r&&c.record_id===r.record_id&&c.record_sha256===r.record_sha256&&JSON.stringify(c.values)===JSON.stringify(r.values)&&JSON.stringify(c.bucket_xy_m)===JSON.stringify(r.bucket_xy_m)&&['native_release','capture','whole_entry','retention_s'].every(k=>c.observed?.[k]===r.observed?.[k]);});
}

export function gapFlightView(example) {
  if(!example || example.observed?.native_release!==true || !vector(example.values,5) || !vector(example.bucket_xy_m,2))return null;
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
  if(!c){text(root,'case-result','No recorded example is available.');for(const n of ['configuration','prediction','references','setup-proof','endpoint-examples'])table(root,n,[],[]);for(const n of ['selection','candidate-coverage','diagnosis','setup-scope'])text(root,n,'unknown');plot(root,'flight',[],true);plot(root,'lateral',[],false);text(root,'flight-caption','Missing data stays unknown; no flight is invented.');return;}
  const o=c.observed||{},p=c.prediction||{},v=c.values,view=gapFlightView(c);
  text(root,'case-result',`${c.context_id||c.id||'unknown'} · ${outcome(o)} · whole-ball entry ${truth(o.whole_entry)} · retention ${fmt(o.retention_s,3)} s.`);
  table(root,'configuration',['Bucket X / Y mm','Board X / Y mm','Board height mm','Tilt °','Drop gap mm'],[[vector(c.bucket_xy_m,2)?c.bucket_xy_m.map(mm).join(' / '):'unknown',vector(v,5)?v.slice(0,2).map(mm).join(' / '):'unknown',mm(v?.[2]),fmt(v?.[3],3),mm(v?.[4])]]);
  table(root,'prediction',['Quantity','Prediction / nominal geometry','Recorded observation'],[
    ['Capture head score (uncalibrated diagnostic)',pct(p.capture_score),outcome(o)],
    ...['Forward rebound m/s','Lateral rebound m/s','Vertical rebound m/s'].map((name,i)=>[name,fmt(p.rebound_velocity_m_s?.[i],4),fmt(o.rebound_velocity_m_s?.[i],4)]),
    ['Conservative whole-ball rim-band margin mm',mm(p.proxy_aperture_margin_m),'Different from center-at-rim margin'],
    ['Center-at-rim clearance mm','Different from rim-band margin',mm(o.mouth_clearance_m)],
    ['Nominal origin − measured rebound XYZ mm','—',vector(o.nominal_origin_minus_observed_rebound_mm,3)?o.nominal_origin_minus_observed_rebound_mm.map(x=>fmt(x,2)).join(' / '):'unknown'],
    ['First-flight contact cutoff','—',Array.isArray(o.cutoff_contact_channels)?o.cutoff_contact_channels.join(', ')||'none':'unknown'],
    ['Rebound → next contact, ms','—',fmt(o.rebound_to_contact_ms,2)],
    ['Robot contact within 50 ms (descriptive)','—',truth(o.early_robot_contact)],
    ['Contacting robot link','—',typeof o.robot_contact_link==='string'?o.robot_contact_link:'unknown — not recorded'],
    ['Rim-crossing forward / lateral error, mm','—',`${fmt(o.rim_forward_error_mm,2)} / ${fmt(o.rim_lateral_error_mm,2)}`]]);
  if(view){const colors=['#d79cec','#dfbb7c','#88aaff'],labels=['purple','gold','blue'];const series=[{points:view.actual,color:'#78d1d5'},...view.forecasts.map((f,i)=>({...f,color:colors[i%colors.length],dashed:true}))];
    const marker=finite(c.bucket_rim_z_m)?[{points:[[view.distance,0],[view.distance,c.bucket_rim_z_m]],color:'#8a9baa'}]:[];
    plot(root,'flight',[...series.map(s=>({...s,points:s.points.map(p=>[p[0],p[2]])})),...marker],true);plot(root,'lateral',series.map(s=>({...s,points:s.points.map(p=>[p[0],p[1]])})),false);
    text(root,'flight-caption',`Solid teal: ${view.actual.length} saved first-flight samples, ${fmt(view.times[0],4)}–${fmt(view.times.at(-1),4)} s. ${view.forecasts.map((f,i)=>`Dashed ${labels[i%labels.length]}: ${f.label}; ${f.basis}.`).join(' ')||'No matching conditional forecast was supplied.'} Native record ${c.record_id}; cache ${c.trace.native_cache_sha256}. These projections do not reconstruct later collisions or retention.`);
  }else{plot(root,'flight',[],true);plot(root,'lateral',[],false);text(root,'flight-caption','No identity-matched finite native trace is available for this request. No flight is invented.');}
  const s=c.selection||{};text(root,'selection',`Candidate ${s.selected_candidate_id||'unknown'} · rank ${count(s.rank)} · geometry checks ${count(s.geometry_checks)}. Geometry reason: ${s.geometry_reason||'none supplied'}. Context reason: ${s.context_reason||'none supplied'}. The capture head is diagnostic only, not a calibrated selector probability.`);
  const bank=c.candidate_coverage||{};text(root,'candidate-coverage',`Frozen proposal bank: ${count(bank.candidates)} candidates; ${count(bank.nonnegative_proxy_candidates)} predicted nonnegative rim-band margins. Best model margin ${mm(bank.best_proxy_margin_m)} mm. Drop gaps span ${mm(bank.gap_min_m)}–${mm(bank.gap_max_m)} mm; anchor ${mm(bank.anchor_gap_m)} mm. This describes model proposals, not measured feasibility or untried-shot success.`);
  text(root,'diagnosis',typeof c.diagnosis==='string'?c.diagnosis:'No diagnostic interpretation supplied.');
  const proof=c.setup_proof;
  table(root,'setup-proof',['Setup check','Recorded evidence'],proof&&o.native_release===false?[
    ['Rejection reason',proof.reason||'unknown'],
    ['Valid IK branches: dropper / board',vector(proof.valid_ik_branches,2)?proof.valid_ik_branches.map(count).join(' / '):'unknown'],
    ['Endpoint pairs attempted',count(proof.endpoint_pair_attempts)],
    ['Home minimum clearance, mm',preciseMm(proof.home_minimum_clearance_m)],
    ['Home closest pair',Array.isArray(proof.home_closest_pair)?proof.home_closest_pair.join(' / '):typeof proof.home_closest_pair==='string'?proof.home_closest_pair:'unknown'],
    ['Fixed-base blockers',Array.isArray(proof.fixed_base_blockers)?JSON.stringify(proof.fixed_base_blockers):'unknown'],
    ['Connection evidence SHA-256',hash(proof.connection_sha256)?proof.connection_sha256:'unknown'],
    ['Resolution evidence SHA-256',hash(proof.resolution_sha256)?proof.resolution_sha256:'unknown'],
  ]:[]);
  table(root,'endpoint-examples',['Branch: dropper / board','Board / dropper position error, mm','Board normal error, °','Minimum clearance, mm','Closest pair','Independently rejected arms'],proof&&o.native_release===false?rows(proof.endpoint_examples).map(e=>[`${count(e?.branch10)} / ${count(e?.branch14)}`,`${preciseMm(e?.board_position_error_m)} / ${preciseMm(e?.dropper_position_error_m)}`,fmt(e?.board_normal_error_deg,3),preciseMm(e?.minimum_clearance_m),Array.isArray(e?.closest_pair)?e.closest_pair.join(' / '):typeof e?.closest_pair==='string'?e.closest_pair:'unknown',Array.isArray(e?.independent_reject_arms)?e.independent_reject_arms.join(', ')||'none':'unknown']):[]);
  text(root,'setup-scope',proof&&o.native_release===false?`${proof.scope||'Saved endpoint evidence only; no new IK or motion replay.'} ${typeof proof.analytic_hint==='string'?`Analytic hint: ${proof.analytic_hint}`:''}`:'No setup rejection proof applies to this native release.');
  table(root,'references',['Frozen method','Board X / Y / height mm / tilt / gap mm','Observed outcome','Whole-ball entry','Retention s'],rows(c.references).map(r=>[names[r?.method]||r?.method||'unknown',shot(r?.values),outcome(r?.observed),truth(r?.observed?.whole_entry),fmt(r?.observed?.retention_s,3)]));
}

export function renderGapFlightDiagnosis(root,raw,correction) {
  if(!raw||raw.version!==VERSION){root.hidden=true;return;}root.hidden=false;
  const ready=gapFlightReady(raw),source=gapFlightSourceReady(raw),failed=raw.status==='failed'||raw.audit?.status==='failed',s=raw.summary||{};
  text(root,'status',ready?'Recorded comparison and separate diagnostic checks verified. No new policy or native trial is claimed.':failed?'Diagnostic verification failed. Findings remain unverified.':'Provisional diagnosis: matching source and reconstruction checks are required.');at(root,'status')?.setAttribute('data-state',ready?'passed':failed?'failed':'unknown');text(root,'scope',raw.scope||'Post-hoc recorded-flight inspection; no new training or native evaluation.');
  table(root,'comparison',['Frozen method','Captures / requested','Capture rate','Clean','Setup rejects','Native misses','Invalid'],METHODS.map(m=>{const c=rows(raw.comparisons).find(r=>r?.method===m)?.counts||{};return[names[m],`${count(c.captures)} / ${count(c.requested)}`,integer(c.requested)&&c.requested>0&&integer(c.captures)?pct(c.captures/c.requested):'unknown',count(c.clean),count(c.rejected),count(c.native_misses),count(c.invalid)];}));
  text(root,'comparison-total',`Original completed experiment: ${count(raw.source?.original_total_requests)} total requests including controls; ${count(raw.source?.original_native_releases)} native releases. The two methods use paired bucket positions. This diagnosis inspects the original selector.`);
  text(root,'captures',`${count(s.captures)} / ${count(s.requested)}`);text(root,'misses',count(s.native_misses));text(root,'rejects',count(s.rejected));
  text(root,'funnel',`Native failures: ${count(s.no_entry_failures)} without whole-ball entry; ${count(s.bucket_retention_but_route_failures)} with bucket retention but a failed board route. ${count(s.early_robot_contact_count)} misses met robot contact within ${count(s.early_robot_contact_ms_threshold)} ms after rebound. Recorded first-flight cutoff contacts: ${s.cutoff_contact_counts?JSON.stringify(s.cutoff_contact_counts):'unknown'}. Setup rejection reasons: ${s.rejection_reason_counts?JSON.stringify(s.rejection_reason_counts):'unknown'}. Early contact is descriptive, not a new scoring rule; retention alone does not establish complete board-route success.`);
  const failures=rows(raw.requests).filter(r=>r?.observed?.capture===false);
  table(root,'failures',['Context','Observed outcome','Whole entry','Retention s','Predicted rim-band margin mm','Contact cutoff / rejection'],failures.map(r=>[r.context_id||'unknown',outcome(r.observed),truth(r.observed?.whole_entry),fmt(r.observed?.retention_s,3),mm(r.prediction?.proxy_aperture_margin_m),r.observed?.native_release===false?r.selection?.geometry_reason||r.selection?.context_reason||'unknown':Array.isArray(r.observed?.cutoff_contact_channels)?r.observed.cutoff_contact_channels.join(', ')||'none':'unknown']));
  table(root,'all-requests',['Context','Bucket X / Y mm','Board X / Y / height mm / tilt / gap mm','Observed outcome','Whole entry','Retention s','Contact cutoff / rejection'],rows(raw.requests).map(r=>[r?.context_id||'unknown',vector(r?.bucket_xy_m,2)?r.bucket_xy_m.map(mm).join(' / '):'unknown',shot(r?.values),outcome(r?.observed),truth(r?.observed?.whole_entry),fmt(r?.observed?.retention_s,3),r?.observed?.native_release===false?r.selection?.geometry_reason||r.selection?.context_reason||'unknown':Array.isArray(r?.observed?.cutoff_contact_channels)?r.observed.cutoff_contact_channels.join(', ')||'none':'unknown']));
  const featured=rows(raw.example_contexts),priority=c=>{const i=featured.indexOf(c.context_id);return i>=0?i:featured.length+(c.featured===true?0:1);};
  const cases=rows(raw.cases).filter(c=>c&&typeof c==='object').sort((a,b)=>priority(a)-priority(b)),select=at(root,'case-select');if(select){const old=select.value;select.replaceChildren(...cases.map((c,i)=>{const n=select.ownerDocument.createElement('option');n.value=String(c.id||c.context_id||i);n.textContent=`${c.context_id||c.id||i} · ${outcome(c.observed)}${c.featured===true?' · featured':''}`;return n;}));select.value=cases.some((c,i)=>String(c.id||c.context_id||i)===old)?old:cases.length?String(cases[0].id||cases[0].context_id||0):'';select.disabled=!cases.length;const selected=()=>cases.find((c,i)=>String(c.id||c.context_id||i)===select.value);select.onchange=()=>drawCase(root,selected());drawCase(root,selected());}
  renderTargetedCorrection(root,correction);
  table(root,'findings',['Finding','Evidence','Status / limitation'],rows(raw.findings).map(r=>[r?.statement||'unknown',r?.evidence||'unknown',r?.status||'unknown']));list(root,'next-steps',raw.next_steps);
  text(root,'audit',`Original native comparison ${source?'verified':'unverified'}: report ${raw.source?.report_sha256||'unknown'}, audit ${raw.source?.audit_sha256||'unknown'}, manifest ${raw.source?.manifest_sha256||'unknown'}, ${count(raw.source?.original_native_audit_checks)} native checks. Separate diagnostic reconstruction ${ready?'verified':'unverified'}: ${raw.audit?.source_sha256||'unknown'}, ${count(raw.audit?.checks_count)} checks. Original labels and results are preserved.`);
}
// The pending correction is deliberately separate from the immutable historical diagnosis.
export const CORRECTION_KEY = 'gap_flight_correction';
export const CORRECTION_VERSION = 'sircus-gap-flight-correction-v1';
export const CORRECTION_AUDIT_VERSION = 'sircus-gap-flight-correction-checks-v1';
export function targetedCorrectionReady(raw) {
  const p=raw?.protocol,a=raw?.audit,c=rows(raw?.comparisons);
  if(raw?.version!==CORRECTION_VERSION||raw.status!=='passed'||p?.substeps!==64||!integer(p.contexts)||p.contexts===0||!integer(p.repeats)||p.repeats===0||!integer(p.maximum_requests)||p.maximum_requests<2*p.contexts*p.repeats)return false;
  if(!['plan_sha256','report_sha256','manifest_sha256'].every(k=>hash(raw[k])&&a?.[k]===raw[k])||a?.version!==CORRECTION_AUDIT_VERSION||a.passed!==true||a.full_checks_verified!==true||!integer(a.checks_count)||a.checks_count===0)return false;
  return c.length===2&&new Set(c.map(r=>r?.method)).size===2&&c.every(r=>['original','corrected'].includes(r?.method)&&validCounts(r?.counts)&&r.counts.requested===p.contexts*p.repeats);
}
function renderTargetedCorrection(root,raw) {
  const known=raw?.version===CORRECTION_VERSION,ready=targetedCorrectionReady(raw),p=known?raw.protocol||{}:{},failed=known&&(raw.status==='failed'||raw.audit?.passed===false);
  const scope=known&&typeof raw.scope==='string'?raw.scope:'The next correction will be reported separately after its protocol and native verification are available.';
  text(root,'correction-protocol',`${scope}${known?` Protocol: ${count(p.contexts)} contexts × ${count(p.repeats)} repeats per method; at most ${count(p.maximum_requests)} total requests; ${count(p.substeps)} substeps.`:''}`);
  const labels={pending:'pending',queued:'queued for the RTX 4090',running:'running',awaiting_audit:'native comparison finished; awaiting its audit',passed:'reported finished, but verification is incomplete',failed:'failed'};
  let status=known?`Targeted correction: ${labels[raw.status]||'unknown'}. No verified improvement is claimed.`:'No new correction results have been published.';
  if(ready){const before=raw.comparisons.find(r=>r.method==='original').counts,after=raw.comparisons.find(r=>r.method==='corrected').counts;
    status=`Targeted correction independently verified: original ${before.captures} / ${before.requested} captures; corrected ${after.captures} / ${after.requested}. Difference ${after.captures-before.captures>=0?'+':''}${after.captures-before.captures} captures. These development-layout replays do not establish a general winner or fresh held-out performance.`;
  }else if(known&&raw.status==='queued'&&typeof raw.queue?.reason==='string')status+=` Queue reason: ${raw.queue.reason}`;
  text(root,'correction-status',status);at(root,'correction-status')?.setAttribute('data-state',ready?'passed':failed?'failed':'unknown');
  table(root,'correction',['Method','Captures / requested','Clean','Setup rejects','Native misses','Invalid'],ready?['original','corrected'].map(method=>{const c=raw.comparisons.find(r=>r.method===method).counts;return[method==='original'?'Original matched replay':'One targeted correction',`${c.captures} / ${c.requested}`,count(c.clean),count(c.rejected),count(c.native_misses),count(c.invalid)];}):[]);
}
export function startGapFlightDiagnosis(root,{fetchReport=async()=>{const r=await fetch('/api/study',{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json();},schedule=setTimeout,cancel=clearTimeout}={}) {
  let stopped=false,timer;
  const poll=async()=>{try{const report=await fetchReport();if(stopped)return;renderGapFlightDiagnosis(root,report?.[KEY],report?.[CORRECTION_KEY]);text(root,'connection','Read-only evidence · refresh every 5 seconds.');}catch(error){if(stopped)return;text(root,'connection',`Snapshot unavailable: ${error.message||error}. Previous findings may be stale.`);text(root,'status','Connection unavailable; previous displayed findings are stale and unverified.');at(root,'status')?.setAttribute('data-state','unknown');text(root,'correction-status','Connection unavailable; previous correction results are stale and unverified.');at(root,'correction-status')?.setAttribute('data-state','unknown');}if(!stopped)timer=schedule(poll,5000);};void poll();return()=>{stopped=true;if(timer!==undefined)cancel(timer);};
}
if(typeof document!=='undefined'){const mount=()=>{const root=document.getElementById('gapFlightDiagnosis');if(root){const stop=startGapFlightDiagnosis(root);globalThis.addEventListener?.('pagehide',stop,{once:true});}};if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();}
