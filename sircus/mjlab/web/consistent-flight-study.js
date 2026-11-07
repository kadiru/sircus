// Read-only snapshot rendering: predictions and measured outcomes stay separate.
export const VERSION='sircus-consistent-flight-study-v1';
export const KEY='consistent_flight_study';
export const METHODS=['bc','ppo','ensemble','consistent'];
const names={bc:'Saved imitation',ppo:'Saved PPO',ensemble:'Original ensemble rank',consistent:'Flight-consistent rank'};
const finite=v=>typeof v==='number'&&Number.isFinite(v),integer=v=>Number.isInteger(v)&&v>=0;
const hash=v=>typeof v==='string'&&/^[a-f0-9]{64}$/.test(v),rows=v=>Array.isArray(v)?v:[];
const vector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(finite),fmt=(v,n=1)=>finite(v)?v.toFixed(n):'unknown';
const mm=v=>finite(v)?fmt(v*1000):'unknown',pct=v=>finite(v)?`${fmt(v*100)}%`:'unknown',count=v=>integer(v)?String(v):'unknown';
const at=(root,name)=>root.querySelector(`[data-consistent-flight="${name}"]`);
function text(root,name,value){const n=at(root,name);if(n)n.textContent=value;}
function table(root,name,headers,data){const host=at(root,name);if(!host)return;const doc=host.ownerDocument,t=doc.createElement('table'),thead=doc.createElement('thead'),head=doc.createElement('tr'),body=doc.createElement('tbody');for(const v of headers){const c=doc.createElement('th');c.textContent=v;head.append(c);}thead.append(head);for(const row of data){const tr=doc.createElement('tr');for(const v of row){const c=doc.createElement('td');c.textContent=v==null?'unknown':String(v);tr.append(c);}body.append(tr);}t.append(thead,body);host.replaceChildren(t);}
function list(root,name,data){const host=at(root,name);if(host)host.replaceChildren(...rows(data).map(v=>{const li=host.ownerDocument.createElement('li');li.textContent=typeof v==='string'?v:v?.description||'unknown';return li;}));}
const observed=c=>c?.observed_native_release===false?'setup rejected':c?.observed_native_release!==true?'unknown':c.observed_capture===true?'capture':c.observed_capture===false?'flight miss':'unknown';

export function consistentFlightReady(raw){
 const p=raw?.protocol,a=raw?.audit,comparison=rows(raw?.comparisons);
 return raw?.version===VERSION&&['completed','verified'].includes(raw.status)&&p?.test_positions===30&&p.maximum_requests===132&&p.substeps===64&&p.new_training===false&&JSON.stringify(p.methods)===JSON.stringify(METHODS)&&hash(raw.report_sha256)&&a?.passed===true&&a.report_sha256===raw.report_sha256&&integer(a.checks_count)&&a.checks_count>0&&comparison.length===4&&new Set(comparison.map(r=>r.method)).size===4&&comparison.every(r=>METHODS.includes(r.method)&&r.requested===30&&['native_releases','captures','clean','rejected','native_misses','invalid'].every(k=>integer(r[k]))&&r.native_releases+r.rejected===r.requested&&r.captures+r.native_misses+r.invalid===r.native_releases&&r.clean<=r.captures);
}

function drawCase(root,c){
 if(!c){text(root,'case-result','No recorded development example is available.');table(root,'case-details',[],[]);text(root,'case-note','No predicted or observed outcome is invented.');return;}
 const v=c.values;
 text(root,'case-result',`${c.id||'unknown'} · original model score ${pct(c.prediction_capture_score)} · original measured outcome: ${observed(c)}.`);
 table(root,'case-details',['Quantity','Recorded request / model prediction'],[
 ['Bucket X / Y (mm)',vector(c.bucket_xy_m,2)?c.bucket_xy_m.map(mm).join(' / '):'unknown'],
 ['Board X / Y (mm)',vector(v,5)?v.slice(0,2).map(mm).join(' / '):'unknown'],
 ['Board height (mm)',mm(v?.[2])],['Board tilt (°)',fmt(v?.[3],3)],['Drop gap (mm)',mm(v?.[4])],
 ['Worst available member: conservative rim-band margin (mm)',mm(c.proxy_aperture_margin_m)],
 ['Flight proxy status',c.proxy_status||'unknown']]);
 text(root,'case-note',`${typeof c.note==='string'?c.note+' ':''}The aperture value is a prediction for this recorded request, not a new native outcome or a calibrated probability. Changed candidate outcomes remain unknown.`);
}

export function renderConsistentFlightStudy(root,raw){
 if(!raw||raw.version!==VERSION){root.hidden=true;return;}root.hidden=false;const ready=consistentFlightReady(raw),p=raw.protocol||{},progress=raw.progress||{},failed=raw.status==='failed'||raw.audit?.passed===false;
 const status=ready?'Fresh fixed64 comparison completed and independently verified. No new weights were trained.':failed?'This study or its verification failed. Counts are provisional; no policy is activated.':raw.status==='queued'?'Fresh comparison queued for the 4090. No new native result is claimed.':raw.status==='running'?'Fresh fixed64 comparison is running. Results remain provisional until the audit completes.':['completed','verified'].includes(raw.status)?'Trials completed; matching independent verification is still pending.':'Development diagnostics / preparation only. Fresh native comparison remains pending.';
 text(root,'status',status);at(root,'status')?.setAttribute('data-state',ready?'passed':failed?'failed':'unknown');text(root,'scope',raw.scope||'Frozen-weight selector comparison; predictions are not measured captures.');
 text(root,'positions',count(p.test_positions));text(root,'progress',`${count(progress.requested)} / ${count(progress.planned_requests??p.maximum_requests)}`);text(root,'training',p.new_training===false?'None':p.new_training===true?'Reported':'unknown');
 const q=raw.queue; text(root,'queue',q?`${q.gpu||'GPU unknown'} · ${q.status||raw.status||'status unknown'}${q.reason?': '+q.reason:''}`:'GPU / queue status not supplied. No availability is inferred.');
 table(root,'comparison',['Frozen method','Requested','Captures','Capture / requested','Clean','Setup rejects','Flight misses','Invalid'],METHODS.map(method=>{const r=rows(raw.comparisons).find(x=>x.method===method)||{};return[names[method],count(r.requested),count(r.captures),integer(r.captures)&&integer(r.requested)&&r.requested>0?pct(r.captures/r.requested):'unknown',count(r.clean),count(r.rejected),count(r.native_misses),count(r.invalid)];}));
 table(root,'development-summary',['Development diagnostic','Value','Scope'],rows(raw.development?.summary).map(r=>[r.metric||'unknown',r.value==null?'unknown':String(r.value),r.scope||'original records / prediction only']));
 const cases=rows(raw.development?.cases),select=at(root,'case-select');if(select){const old=select.value;select.replaceChildren(...cases.map((c,i)=>{const n=select.ownerDocument.createElement('option');n.value=String(c.id||i);n.textContent=`${c.id||i} · ${observed(c)} · original score ${pct(c.prediction_capture_score)}`;return n;}));select.value=cases.some((c,i)=>String(c.id||i)===old)?old:cases.length?String(cases[0].id||0):'';select.disabled=!cases.length;const selected=()=>cases.find((c,i)=>String(c.id||i)===select.value);select.onchange=()=>drawCase(root,selected());drawCase(root,selected());}
 list(root,'findings',raw.findings);list(root,'next-steps',raw.next_steps);text(root,'audit',`${ready?'Verified fresh comparison':'Fresh comparison unverified'} · report ${raw.report_sha256||'unknown'} · audit ${raw.audit?.passed===true?'passed':raw.audit?.passed===false?'failed':'pending'} · ${count(raw.audit?.checks_count)} checks. An audit must bind this exact report before completion is shown as verified.`);
 const prov=at(root,'provenance');if(prov){const s=raw.provenance||{};prov.replaceChildren(...[['Original source report',s.source_report_sha256],['Frozen ensemble checkpoint',s.ensemble_checkpoint_sha256],['Selector',s.selector_sha256],['Frozen fresh plan',s.plan_sha256]].map(([label,value])=>{const n=prov.ownerDocument.createElement('p');n.textContent=`${label}: ${hash(value)?value:'unknown'}`;return n;}));}
}

export function startConsistentFlightStudy(root,{fetchReport=async()=>{const response=await fetch('/api/study',{cache:'no-store'});if(!response.ok)throw Error(`HTTP ${response.status}`);return response.json();},schedule=setTimeout,cancel=clearTimeout}={}){let stopped=false,timer;const poll=async()=>{try{const report=await fetchReport();if(stopped)return;renderConsistentFlightStudy(root,report?.[KEY]);text(root,'connection','Read-only snapshot · refresh every 5 seconds.');}catch(error){if(stopped)return;text(root,'connection',`Snapshot unavailable: ${error.message||error}. Previous results may be stale.`);text(root,'status','Connection unavailable; previous displayed findings are stale and unverified.');at(root,'status')?.setAttribute('data-state','unknown');}if(!stopped)timer=schedule(poll,5000);};void poll();return()=>{stopped=true;if(timer!==undefined)cancel(timer);};}
if(typeof document!=='undefined'){const mount=()=>{const root=document.getElementById('consistentFlightStudy');if(root){const stop=startConsistentFlightStudy(root);globalThis.addEventListener?.('pagehide',stop,{once:true});}};if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();}
