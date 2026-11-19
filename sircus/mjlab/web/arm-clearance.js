// Saved evidence only. This panel cannot request a shot or change a policy.
export const VERSION='sircus-arm-clearance-study-v1';
export const AUDIT_VERSION='sircus-arm-clearance-audit-v1';
export const KEY='arm_clearance_comparison';
export const METHODS=['consistent','arm_clearance'];
const labels={consistent:'Current selector',arm_clearance:'Arm-clearance selector'};
const fields=['requested','native_releases','captures','clean','rejected','native_misses','invalid'];
const segments=['before_terminal','post_rebound','after_terminal'];
const revealedPanels=new WeakSet();
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const integer=v=>Number.isInteger(v)&&v>=0;
const hash=v=>typeof v==='string'&&/^[a-f0-9]{64}$/.test(v);
const vector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(finite);
const rows=v=>Array.isArray(v)?v:[];
const fmt=(v,n=2)=>finite(v)?v.toFixed(n):'unknown';
const mm=v=>finite(v)?fmt(v*1000,1):'unknown';
const bool=v=>v===true?'yes':v===false?'no':'unknown';
const at=(root,name)=>root.querySelector(`[data-arm-clearance="${name}"]`);
function text(root,name,value){const n=at(root,name);if(n)n.textContent=value;}
function table(root,name,headers,data){
  const host=at(root,name);if(!host)return;const doc=host.ownerDocument,t=doc.createElement('table'),head=doc.createElement('thead'),tr=doc.createElement('tr'),body=doc.createElement('tbody');
  for(const value of headers){const c=doc.createElement('th');c.textContent=value;tr.append(c);}head.append(tr);
  for(const row of data){const line=doc.createElement('tr');for(const value of row){const c=doc.createElement('td');c.textContent=value==null?'unknown':String(value);line.append(c);}body.append(line);}t.append(head,body);host.replaceChildren(t);
}
function validCounts(c){return c&&fields.every(k=>integer(c[k]))&&c.requested===c.native_releases+c.rejected&&c.native_releases===c.captures+c.native_misses+c.invalid&&c.clean<=c.captures;}
const outcome=r=>!r?'unknown':!r.native_release?(r.clearance_rejected?'clearance rejected':'setup rejected'):r.invalid?'invalid native':r.capture?(r.clean?'clean capture':'capture with contact / marginal'):'flight miss';

export function clearanceReady(raw){
  const p=raw?.protocol,a=raw?.audit,comparisons=rows(raw?.comparisons),positions=rows(raw?.test_positions),controls=raw?.control_gates;
  if(raw?.version!==VERSION||raw.status!=='verified'||p?.substeps!==64||p.test_layouts!==50||p.maximum_fresh_requests!==112||p.control_requests!==12||p.new_train_requests!==0||JSON.stringify(p.methods)!==JSON.stringify(METHODS))return false;
  if(a?.version!==AUDIT_VERSION||a.passed!==true||a.full_checks_verified!==true||!integer(a.checks_count)||a.checks_count===0||!hash(a.source_sha256)||!hash(a.auditor_sha256)||!['report_sha256','manifest_sha256','plan_sha256','contract_fingerprint'].every(k=>hash(raw[k])&&a[k]===raw[k]))return false;
  if(!validCounts(raw.counts)||raw.counts.requested!==112||comparisons.length!==2||new Set(comparisons.map(c=>c?.method)).size!==2||!comparisons.every(c=>METHODS.includes(c?.method)&&validCounts(c.counts)&&c.counts.requested===50)||positions.length!==100)return false;
  const totals=comparisons.map(c=>c.counts);
  for(const phase of ['pre','post']){const gate=controls?.[phase],c=gate?.counts;if(gate?.status!=='passed'||!validCounts(c)||!Object.entries({requested:6,native_releases:6,captures:3,rejected:0,native_misses:3,invalid:0}).every(([k,v])=>c[k]===v))return false;totals.push(c);}
  if(!fields.every(k=>totals.reduce((sum,c)=>sum+c[k],0)===raw.counts[k]))return false;
  if(new Set(positions.map(r=>r?.record_id)).size!==100||!positions.every(r=>r&&METHODS.includes(r.method)&&typeof r.context_id==='string'&&r.context_id.length>0&&typeof r.record_id==='string'&&r.record_id.length>0&&vector(r.bucket_xy_m,2)&&vector(r.values,5)&&['capture','native_release','clean','invalid','clearance_rejected'].every(k=>typeof r[k]==='boolean')&&(!r.capture||r.native_release&&!r.invalid)&&(!r.clean||r.capture)&&(!r.clearance_rejected||!r.native_release)))return false;
  for(const method of METHODS){const group=positions.filter(r=>r.method===method),c=comparisons.find(r=>r.method===method).counts;
    if(group.length!==50||new Set(group.map(r=>r.context_id)).size!==50||group.filter(r=>r.native_release).length!==c.native_releases||group.filter(r=>r.capture).length!==c.captures||group.filter(r=>r.clean).length!==c.clean||group.filter(r=>r.native_release&&r.invalid).length!==c.invalid)return false;
  }
  const reference=new Map(positions.filter(r=>r.method===METHODS[0]).map(r=>[r.context_id,r]));
  return positions.filter(r=>r.method===METHODS[1]).every(r=>{const old=reference.get(r.context_id);return old&&JSON.stringify(old.bucket_xy_m)===JSON.stringify(r.bucket_xy_m);});
}

export function clearanceContactsReady(raw){
  if(!clearanceReady(raw))return false;const groups=rows(raw.contact_comparisons);
  if(groups.length!==2||new Set(groups.map(g=>g?.method)).size!==2)return false;
  for(const g of groups){
    const c=raw.comparisons.find(r=>r.method===g?.method)?.counts,items=raw.test_positions.filter(r=>r.method===g?.method);
    if(!c||g.requested!==50||g.native_releases!==c.native_releases||!integer(g.observed_native)||!integer(g.missing_native)||g.observed_native+g.missing_native!==g.native_releases)return false;
    if(!items.every(r=>segments.every(s=>r['robot_contact_'+s]===null||typeof r['robot_contact_'+s]==='boolean')))return false;
    const observed=items.filter(r=>r.native_release&&segments.every(s=>typeof r['robot_contact_'+s]==='boolean'));
    if(observed.length!==g.observed_native||items.some(r=>!r.native_release&&segments.some(s=>r['robot_contact_'+s]!==null)))return false;
    for(const s of segments){const n=g['robot_contact_trials_'+s];if(!integer(n)||n>g.observed_native||n!==observed.filter(r=>r['robot_contact_'+s]).length)return false;}
  }return true;
}

export function pairedResults(raw){
  if(!clearanceReady(raw))return null;const before=new Map(raw.test_positions.filter(r=>r.method==='consistent').map(r=>[r.context_id,r]));
  const pairs=raw.test_positions.filter(r=>r.method==='arm_clearance').map(after=>({before:before.get(after.context_id),after}));
  return {pairs,gain:pairs.filter(p=>!p.before.capture&&p.after.capture).length,loss:pairs.filter(p=>p.before.capture&&!p.after.capture).length,both:pairs.filter(p=>p.before.capture&&p.after.capture).length,neither:pairs.filter(p=>!p.before.capture&&!p.after.capture).length};
}
function drawExample(root,pair){
  if(!pair){text(root,'case-status','No verified paired example is available.');table(root,'case',[],[]);return;}
  text(root,'case-status',`${pair.before.context_id} · bucket X/Y ${pair.before.bucket_xy_m.map(mm).join(' / ')} mm · ${outcome(pair.before)} → ${outcome(pair.after)}.`);
  table(root,'case',['Method','Board X / Y / Z, mm','Tilt °','Drop gap, mm','Predicted minimum arm clearance, mm','Closest predicted arm geometry','Observed outcome','Robot contact before terminal','First post-bounce contact','Rebound to contact, ms','Retention, s'],[pair.before,pair.after].map(r=>[labels[r.method],r.values.slice(0,3).map(mm).join(' / '),fmt(r.values[3],3),mm(r.values[4]),mm(r.predicted_clearance_min_m),typeof r.predicted_clearance_closest_geom==='string'?r.predicted_clearance_closest_geom:'unknown',outcome(r),bool(r.robot_contact_before_terminal),Array.isArray(r.first_post_bounce_contact_channels)?r.first_post_bounce_contact_channels.join(', ')||'none recorded':'unknown',fmt(r.first_contact_delay_ms,3),fmt(r.retention_s,3)]));
}
export function renderArmClearance(root,raw){
  if(!root)return;root.hidden=!raw;if(!raw)return;
  const ready=clearanceReady(raw),contactReady=clearanceContactsReady(raw),p=raw.protocol||{},known=raw.version===VERSION;
  const states={preparing:'preparing the frozen plan',prepared:'prepared; awaiting GPU slot',pending:'pending',queued:'queued for the RTX 4090',running:'running',completed:'finished; awaiting independent verification',awaiting_audit:'finished; awaiting independent verification',audit_failed:'audit failed',failed:'failed'};
  let status=known?`${states[raw.status]||'Verification incomplete'}. No verified improvement is reported.`:'Unknown comparison version. No verified improvement is reported.';
  const paired=pairedResults(raw);
  if(ready){const before=raw.comparisons.find(c=>c.method==='consistent').counts,after=raw.comparisons.find(c=>c.method==='arm_clearance').counts;status=`Verified on the same 50 new bucket positions: current selector ${before.captures}/50 captures; arm-clearance selector ${after.captures}/50. Paired gains ${paired.gain}, losses ${paired.loss}. This finite comparison does not establish all-table success.`;}
  if(raw.error)status+=` ${raw.error}`;
  text(root,'status',status);at(root,'status')?.setAttribute('data-state',ready?'passed':['failed','audit_failed'].includes(raw.status)||raw.audit?.passed===false?'failed':'unknown');
  text(root,'progress',`Planned positions: ${integer(raw.planned_layouts)?raw.planned_layouts:'unknown'} / 50. Recorded requests: ${integer(raw.progress_requests)?raw.progress_requests:'unknown'} / 112 (100 paired attempts plus 12 controls). ${raw.queue?.reason||''}`);
  for(const [method,node] of [['consistent','original'],['arm_clearance','clearance']]){const c=ready?raw.comparisons.find(x=>x.method===method).counts:null;text(root,node,c?`${c.captures} / 50`:'unknown');}
  text(root,'paired',ready?`${paired.gain} / ${paired.loss}`:'unknown');
  table(root,'comparison',['Method','Captures / requested','Clean','Other setup rejects','Clearance abstentions','Native misses','Invalid'],ready?METHODS.map(method=>{const c=raw.comparisons.find(r=>r.method===method).counts;const abstentions=raw.test_positions.filter(r=>r.method===method&&r.clearance_rejected).length;return[labels[method],`${c.captures} / 50`,c.clean,c.rejected-abstentions,abstentions,c.native_misses,c.invalid];}):[]);
  const contacts=contactReady?raw.contact_comparisons:[];
  text(root,'contact-status',contactReady?METHODS.map(m=>{const g=contacts.find(x=>x.method===m),abstentions=raw.test_positions.filter(r=>r.method===m&&r.clearance_rejected).length;return `${labels[m]}: ${g.robot_contact_trials_post_rebound}/${g.observed_native} observed native shots had post-rebound robot contact (${g.robot_contact_trials_post_rebound}/50 requested); ${g.missing_native} native observations unknown, ${50-g.native_releases-abstentions} other setup rejections, ${abstentions} clearance abstentions.`;}).join(' '):'Measured robot-contact counts are unknown until the exact native evidence passes its matching audit.');
  table(root,'contacts',['Method','Native releases','Observed contact records','Unknown native records','Robot contact before terminal','After terminal','Any post-rebound robot contact'],METHODS.map(method=>{const g=contacts.find(x=>x.method===method);return[labels[method],...['native_releases','observed_native','missing_native','robot_contact_trials_before_terminal','robot_contact_trials_after_terminal','robot_contact_trials_post_rebound'].map(k=>g?g[k]:'unknown')];}));
  table(root,'all-positions',['Position','Bucket X / Y, mm','Current selector','Arm clearance','Current / new robot contact'],ready?paired.pairs.map(({before,after})=>[before.context_id,before.bucket_xy_m.map(mm).join(' / '),outcome(before),outcome(after),`${bool(before.robot_contact_before_terminal)} / ${bool(after.robot_contact_before_terminal)}`]):[]);
  const select=at(root,'case-select');if(select){const previous=select.value,pairs=paired?.pairs||[];select.replaceChildren(...pairs.map(pair=>{const option=select.ownerDocument.createElement('option');option.value=pair.before.context_id;option.textContent=`${pair.before.context_id} · ${outcome(pair.before)} → ${outcome(pair.after)}`;return option;}));select.disabled=!pairs.length;select.value=pairs.some(pair=>pair.before.context_id===previous)?previous:(pairs.find(pair=>pair.before.capture!==pair.after.capture)||pairs[0])?.before.context_id||'';const selected=()=>pairs.find(pair=>pair.before.context_id===select.value);select.onchange=()=>drawExample(root,selected());drawExample(root,selected());}
  text(root,'protocol',`${p.test_layouts??'unknown'} new layouts; ${p.substeps??'unknown'} substeps; ${p.candidates??'unknown'} candidates and at most ${p.maximum_geometry_checks??'unknown'} geometry checks per method; ${p.new_train_requests??'unknown'} new training requests. Identical frozen model weights and scoring. Both methods spend one requested shot per position; rejected setups remain in the denominator.`);
  text(root,'rule',typeof raw.rule_description==='string'?raw.rule_description:'The frozen clearance rule is not available in this snapshot.');text(root,'scope',raw.scope||'Missing experiment scope.');
  text(root,'audit',ready?`${raw.audit.checks_count} independent checks. Report ${raw.report_sha256}; plan ${raw.plan_sha256}; manifest ${raw.manifest_sha256}; audit ${raw.audit.source_sha256}. All original study panels and policy routing remain unchanged.`:'No completed, byte-bound independent proof is available. Interim counts are progress only.');
  // Native fragment scrolling can run while this asynchronous panel is hidden.
  // Repair that initial navigation once, without moving the reader on polling.
  if(!revealedPanels.has(root)){revealedPanels.add(root);if(globalThis.location?.hash==='#armClearanceStudy')root.scrollIntoView?.({block:'start'});}
}
export function startArmClearance(root,{fetchReport=async()=>{const r=await fetch('/api/study',{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json();},schedule=setTimeout,cancel=clearTimeout}={}){
  let stopped=false,timer;const poll=async()=>{try{const report=await fetchReport();if(stopped)return;renderArmClearance(root,report?.[KEY]);text(root,'connection','Read-only evidence · refresh every 5 seconds.');}catch(error){if(stopped)return;text(root,'connection',`Snapshot unavailable: ${error.message||error}. Previous results may be stale.`);text(root,'status','Connection unavailable; previous results are stale and unverified.');at(root,'status')?.setAttribute('data-state','unknown');text(root,'contact-status','Connection unavailable; previous contact counts are stale and unverified.');}if(!stopped)timer=schedule(poll,5000);};void poll();return()=>{stopped=true;if(timer!==undefined)cancel(timer);};
}
if(typeof document!=='undefined'){const mount=()=>{const root=document.getElementById('armClearanceStudy');if(root){const stop=startArmClearance(root);globalThis.addEventListener?.('pagehide',stop,{once:true});}};if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();}
