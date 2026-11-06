// Read-only presentation. Model forecasts never become native observations.
export const VERSION='sircus-physics-ensemble-v1';
export const AUDIT_VERSION='sircus-physics-ensemble-audit-v1';
export const KEY='physics_ensemble_study';
const METHODS=['bc','ppo','ensemble'],LABELS={bc:'Saved BC',ppo:'Saved PPO',ensemble:'Physics ensemble'};
const num=v=>typeof v==='number'&&Number.isFinite(v),count=v=>Number.isInteger(v)&&v>=0;
const hash=v=>typeof v==='string'&&/^[a-f0-9]{64}$/.test(v);
const fmt=(v,n=2)=>num(v)?v.toFixed(n):'unknown';
const requested=r=>count(r?.requested)?r.requested:count(r?.attempts)?r.attempts:null;
const pct=v=>num(v)?`${(100*v).toFixed(1)}%`:'unknown';
const interval=(v,percentage=true)=>Array.isArray(v)&&v.length===2&&v.every(num)&&v[0]<=v[1]?`${percentage?pct(v[0]):fmt(v[0])}–${percentage?pct(v[1]):fmt(v[1])}`:'unknown';
const ratio=(v,n)=>count(v)&&count(n)&&n>0&&v<=n?`${v} / ${n}`:'unknown';
const vector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(num);
const targetLabel=v=>({native_execution:'Native execution',board_bounce:'Board bounce | valid native',whole_ball_entry:'Whole-ball entry | valid native',capture:'Capture | valid native',retention_fraction:'Retention fraction | valid native',log_first_flight_opening_distance:'Closest first-flight 3D opening distance',rebound_forward_m_s:'Rebound bucket-forward velocity',rebound_lateral_m_s:'Rebound lateral velocity',rebound_vertical_m_s:'Rebound vertical velocity',requested_capture_uncalibrated_score:'Requested-capture score (uncalibrated)'}[v]||v||'unknown');
const rows=r=>Array.isArray(r)?r:[];
const at=(root,name)=>root.querySelector(`[data-physics-ensemble="${name}"]`);
function text(root,name,value){const node=at(root,name);if(node)node.textContent=value;}
function table(root,name,headers,data){const host=at(root,name);if(!host)return;const doc=host.ownerDocument,t=doc.createElement('table'),head=doc.createElement('thead'),hrow=doc.createElement('tr'),body=doc.createElement('tbody');headers.forEach(s=>{const th=doc.createElement('th');th.textContent=s;hrow.append(th);});head.append(hrow);data.forEach(r=>{const tr=doc.createElement('tr');r.forEach(s=>{const td=doc.createElement('td');td.textContent=s==null?'unknown':String(s);tr.append(td);});body.append(tr);});t.append(head,body);host.replaceChildren(t);}

export function physicsEnsembleOutcome(row){
 const a=row?.actual;if(!a||typeof a.native_release!=='boolean')return'unknown';
 if(a.native_release===false)return'rejected';
 if(a.invalid||a.invalid_physics===true)return'invalid';
 if(typeof a.capture!=='boolean'||typeof a.clean!=='boolean'||a.clean&&!a.capture)return'unknown';
 return a.clean?'clean':a.capture?'capture':'miss';
}

export function physicsEnsembleReady(raw){
 if(!raw||raw.version!==VERSION||raw.status!=='completed'||raw.phase!=='completed')return false;
 const a=raw.audit,p=raw.protocol,g=raw.gate;
 if(a?.version!==AUDIT_VERSION||a.status!=='passed'||a.full_checks_verified!==true||a.bound_files_verified!==true||!count(a.checks_count)||a.checks_count<1||!hash(raw.contract_fingerprint)||a.contract_fingerprint!==raw.contract_fingerprint||!hash(a.source_sha256))return false;
 if(p?.substeps!==64||p.noise!==false||p.validation_layouts!==24||p.test_layouts!==60||p.maximum_fresh_requests!==264||p.no_test_flight_search!==true||p.fixed_model_no_validation_selection!==true||JSON.stringify(p.methods)!==JSON.stringify(METHODS))return false;
 if(g?.status!=='passed'||g.complete!==true||g.controls_passed!==true||g.no_test_flight_search!==true||g.predictions_frozen_before_any_evaluation!==true||raw.controls?.pre?.status!=='passed'||raw.controls?.post?.status!=='passed')return false;
 if(requested(raw.counts)!==264||raw.training?.ensemble_members!==5||raw.training?.fit_device!=='cpu'||raw.training?.fit_steps!==600||!hash(raw.training?.artifact?.sha256))return false;
 const comparisons=rows(raw.test_comparison);if(comparisons.length!==3||new Set(comparisons.map(r=>r.method)).size!==3)return false;
 const positions=rows(raw.test_positions);if(positions.length!==180)return false;
 let ids;
 for(const method of METHODS){
  const cohort=positions.filter(r=>r.method===method),stats=comparisons.find(r=>r.method===method)?.counts;
  if(cohort.length!==60||new Set(cohort.map(r=>r.context_id)).size!==60||cohort.some(r=>!r.context_id||!vector(r.bucket_xy_m,2)||!vector(r.values,5)||physicsEnsembleOutcome(r)==='unknown'))return false;
  const now=new Set(cohort.map(r=>r.context_id));if(ids&&[...now].some(v=>!ids.has(v)))return false;ids=now;
  if(requested(stats)!==60||['native_releases','captures','clean','rejected','native_misses','invalid'].some(k=>!count(stats?.[k])||stats[k]>60)||stats.clean>stats.captures||stats.native_releases+stats.rejected!==60)return false;
  if(stats.native_releases!==cohort.filter(r=>r.actual.native_release).length||stats.captures!==cohort.filter(r=>r.actual.capture).length||stats.clean!==cohort.filter(r=>r.actual.clean).length)return false;
 }
 const buckets=new Map(positions.filter(r=>r.method==='bc').map(r=>[r.context_id,JSON.stringify(r.bucket_xy_m)]));
 return positions.every(r=>buckets.get(r.context_id)===JSON.stringify(r.bucket_xy_m));
}

function plot(root,name,series,{caption=''}={}){
 const canvas=at(root,name),ctx=canvas?.getContext?.('2d');if(!ctx)return;
 const w=canvas.width,h=canvas.height,pad=40;ctx.clearRect(0,0,w,h);
 const points=series.flatMap(s=>s.points);if(!points.length){ctx.fillStyle='#a6bbc8';ctx.fillText(caption||'No observed samples.',pad,40);return;}
 let xmin=Math.min(...points.map(p=>p[0])),xmax=Math.max(...points.map(p=>p[0])),ymin=Math.min(...points.map(p=>p[1])),ymax=Math.max(...points.map(p=>p[1]));
 if(xmax===xmin)xmax=xmin+1;if(ymax===ymin)ymax=ymin+1;
 const x=v=>pad+(v-xmin)/(xmax-xmin)*(w-2*pad),y=v=>h-pad-(v-ymin)/(ymax-ymin)*(h-2*pad);
 ctx.strokeStyle='#324654';ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad,h-pad);ctx.lineTo(w-pad,h-pad);ctx.stroke();
 series.forEach((s,i)=>{if(!s.points.length)return;ctx.strokeStyle=s.color||'#75d3d2';ctx.lineWidth=2;ctx.setLineDash?.(s.dashed?[6,5]:[]);ctx.beginPath();s.points.forEach((p,j)=>j?ctx.lineTo(x(p[0]),y(p[1])):ctx.moveTo(x(p[0]),y(p[1])));ctx.stroke();ctx.fillStyle=s.color||'#75d3d2';ctx.fillText(s.label,pad+(i%3)*250,18+Math.floor(i/3)*16);});ctx.setLineDash?.([]);
 ctx.fillStyle='#9eb4c3';ctx.fillText(fmt(xmin),pad,h-12);ctx.fillText(fmt(xmax),w-pad-38,h-12);ctx.fillText(fmt(ymin),4,h-pad);ctx.fillText(fmt(ymax),4,pad);
}

export function physicsEnsembleTrajectory(raw,trace){
 if(!trace||!vector(trace.values,5)||!vector(trace.bucket_xy_m,2)||!Array.isArray(trace.position_m)||trace.position_m.length<2||!trace.position_m.every(p=>vector(p,3)))return null;
 const origin=trace.values.slice(0,2),delta=trace.bucket_xy_m.map((v,i)=>v-origin[i]),distance=Math.hypot(...delta);if(distance<=1e-6)return null;
 const direction=delta.map(v=>v/distance),project=p=>[(p[0]-origin[0])*direction[0]+(p[1]-origin[1])*direction[1],p[2]];
 let end=trace.position_m.length,cutoff='complete saved flight';
 const fall=trace.position_m.findIndex(p=>p[2]<-.2);if(fall>=0){end=Math.min(end,Math.max(2,fall+1));cutoff='first observed scoring fall below −0.2 m';}
 if(num(trace.capture_time_s)&&num(trace.sample_dt_s)&&trace.sample_dt_s>0){const candidate=Math.max(2,Math.ceil(trace.capture_time_s/trace.sample_dt_s)+2);if(candidate<end){end=candidate;cutoff='capture time plus one saved sample';}}
 const predicted=Array.isArray(trace.predicted_position_m)&&trace.predicted_position_m.length>=2&&trace.predicted_position_m.every(p=>vector(p,3))?trace.predicted_position_m.map(project):[];
 return{points:trace.position_m.slice(0,end).map(project),predicted,distance,shown:end,total:trace.position_m.length,cutoff};
}

function showTrace(root,raw){
 const choices=rows(raw.trace_examples).filter(r=>physicsEnsembleTrajectory(raw,r)),selector=at(root,'trace-select');if(!selector)return;
 const old=selector.value;selector.replaceChildren(...choices.map((r,i)=>{const o=selector.ownerDocument.createElement('option');o.value=String(i);o.textContent=`${r.method||'unknown'} · ${r.context_id||r.id||r.record_id||i}`;return o;}));
 selector.disabled=!choices.length;selector.value=choices[Number(old)]?old:'0';
 const draw=()=>{const trace=choices[Number(selector.value)],view=physicsEnsembleTrajectory(raw,trace);
  if(!view){plot(root,'trace',[],{caption:'No saved native flight; no analytic trajectory is generated.'});text(root,'trace-caption','No actual native trace is available. No model trajectory is invented.');return;}
  const series=[{label:'Actual native flight',points:view.points,color:'#78d1d5'},{label:'Bucket center / reported rim',points:[[view.distance,0],[view.distance,.175]],color:'#dfc17d'}];
  if(view.predicted.length)series.push({label:'Supplied model prediction',points:view.predicted,color:'#d79cec',dashed:true});plot(root,'trace',series);
  text(root,'trace-caption',`${trace.method||'unknown'} · ${trace.context_id||trace.id||trace.record_id} · ${view.shown} / ${view.total} actual saved samples shown; ${view.cutoff}. Projected distance ${fmt(view.distance*1000,1)} mm. ${view.predicted.length?'Dashed line is a model forecast, not a simulator observation.':'This model supplies no dense predicted trajectory; only the recorded flight is drawn.'} Bucket height remains the provisional reported 175 mm.`);
 };selector.onchange=draw;draw();
}

function coverage(root,raw){
 const method=at(root,'method')?.value||'ensemble',cohort=rows(raw.test_positions).filter(r=>r.method===method),canvas=at(root,'coverage'),ctx=canvas?.getContext?.('2d');
 table(root,'positions',['Position','Bucket XY mm','Board XY mm','Board H mm','Tilt °','Gap mm','Uncalibrated capture score','Disagreement','Modeled native likelihood','Conditional capture score','Pred. closest opening mm','Pred. retention fraction','Ranking score','Actual native','Observed outcome'],cohort.map(r=>[r.context_id,vector(r.bucket_xy_m,2)?r.bucket_xy_m.map(v=>fmt(v*1000,1)).join(' / '):'unknown',vector(r.values,5)?r.values.slice(0,2).map(v=>fmt(v*1000,1)).join(' / '):'unknown',fmt(r.values?.[2]*1000,1),fmt(r.values?.[3],1),fmt(r.values?.[4]*1000,1),pct(r.model_prediction?.capture_probability),fmt(r.model_prediction?.uncertainty,3),pct(r.model_prediction?.admission_probability),pct(r.model_prediction?.conditional_capture_probability),fmt(r.model_prediction?.aperture_distance_m*1000,1),fmt(r.model_prediction?.retention_fraction,3),fmt(r.model_prediction?.ranking_score,3),typeof r.actual?.native_release==='boolean'?(r.actual.native_release?'yes':'no'):'unknown',physicsEnsembleOutcome(r)]));
 const points=cohort.filter(r=>vector(r.bucket_xy_m,2)),colors={clean:'#86d4a6',capture:'#6dc7d6',miss:'#dfb977',rejected:'#e78f88',invalid:'#d19fda',unknown:'#7e929f'};
 if(ctx){ctx.clearRect(0,0,canvas.width,canvas.height);if(points.length){let xmin=Math.min(...points.map(r=>r.bucket_xy_m[0])),xmax=Math.max(...points.map(r=>r.bucket_xy_m[0])),ymin=Math.min(...points.map(r=>r.bucket_xy_m[1])),ymax=Math.max(...points.map(r=>r.bucket_xy_m[1]));if(xmin===xmax)xmax++;if(ymin===ymax)ymax++;
  points.forEach(r=>{const x=40+(r.bucket_xy_m[0]-xmin)/(xmax-xmin)*(canvas.width-80),y=canvas.height-40-(r.bucket_xy_m[1]-ymin)/(ymax-ymin)*(canvas.height-80);ctx.fillStyle=colors[physicsEnsembleOutcome(r)];ctx.beginPath();ctx.arc(x,y,5,0,2*Math.PI);ctx.fill();});
 }else{ctx.fillStyle='#a6bbc8';ctx.fillText('No observed test positions yet.',30,40);}}
 const counts=Object.fromEntries(Object.keys(colors).map(o=>[o,cohort.filter(r=>physicsEnsembleOutcome(r)===o).length]));text(root,'coverage-caption',`${LABELS[method]}: ${cohort.length} reported positions · ${counts.clean} clean, ${counts.capture} assisted/other captures, ${counts.miss} misses, ${counts.rejected} rejected, ${counts.invalid} invalid, ${counts.unknown} unknown. Dots are observed outcomes at evaluated coordinates, not an interpolated success map.`);
}

export function renderPhysicsEnsemble(root,raw){
 if(!raw||raw.version!==VERSION){root.hidden=true;return;}root.hidden=false;
 const ready=physicsEnsembleReady(raw),waiting=raw.status==='waiting_for_4090'||raw.phase==='waiting_for_4090',fitted=raw.training?.status==='completed'||raw.fit_status==='model_fit_completed';
 text(root,'status',`${raw.status||'unknown'} · ${raw.phase||'unknown'}${waiting?' · CPU model fit and queued 4090 native evaluation are separate':''}${raw.reason?' · '+raw.reason:''}${raw.error?' · '+raw.error:''}`);text(root,'scope',raw.scope||'State-based learned physics; privileged geometry and held endpoints.');
 const blockers=rows(raw.queue?.reason).filter(v=>Number.isInteger(v)&&v>0||v==='shared-lock');
 text(root,'queue',waiting?`Native evaluation is queued${num(raw.queue?.wait_s)?' · observed wait '+fmt(raw.queue.wait_s,0)+' s':''}${blockers.length?' · existing owner '+blockers.map(v=>v==='shared-lock'?'shared lock':'PID '+v).join(', '):''}. It waits for availability without interrupting existing workloads. Snapshot and queue status refresh every 5 seconds.`:'');
 const d=raw.dataset||{},t=raw.training||{},p=raw.protocol||{};
 table(root,'progress',['Stage','Observed / planned','Status'],[
  ['Existing parent TRAIN requests',`${count(d.train_requests)?d.train_requests:'unknown'} / 1024`,'No new TRAIN collection'],
  ['CPU physics model',`${count(t.ensemble_members)?t.ensemble_members:'unknown'} members · ${count(t.fit_steps)?t.fit_steps:'unknown'} fit steps`,fitted?'model_fit_completed':t.status||'pending'],
  ['Native validation',`${requested(raw.phase_counts?.validation)??'unknown'} / 72`,waiting?'waiting_for_4090':'Fixed model; no VAL selection'],
  ['Native TEST',`${requested(raw.phase_counts?.test)??'unknown'} / 180`,waiting?'waiting_for_4090':raw.phase||'pending'],
  ['Matched controls',`${requested(raw.phase_counts?.controls_pre)!==null&&requested(raw.phase_counts?.controls_post)!==null?requested(raw.phase_counts.controls_pre)+requested(raw.phase_counts.controls_post):'unknown'} / 12`,`${raw.controls?.pre?.status||'pending'} / ${raw.controls?.post?.status||'pending'}`],
  ['All fresh native requests',`${requested(raw.counts)??'unknown'} / ${p.maximum_fresh_requests??264}`,waiting?'No new native outcomes claimed':raw.status||'pending']]);
 text(root,'dataset',`TRAIN labels: ${count(d.train_labels)?d.train_labels:d.train_labels&&typeof d.train_labels==='object'?'per-target support below':'unknown'} · native TRAIN releases: ${count(d.train_native_releases)?d.train_native_releases:'unknown'} · TRAIN layouts: ${count(d.train_layouts)?d.train_layouts:'unknown'}. Fresh split sizes: 24 VAL and 60 TEST. ${d.labels_scope||'Existing parent TRAIN labels only; prior VAL/TEST and diagnosis layouts are excluded from new holdouts.'}`);
 text(root,'features',`Model inputs: ${rows(t.input_features).join(', ')||'unknown'}. Targets: ${rows(t.target_features).join(', ')||'unknown'}.`);
 const support=d.train_labels&&typeof d.train_labels==='object'?Object.entries(d.train_labels).filter(([,v])=>count(v)):rows(d.target_support).map((v,i)=>[t.target_features?.[i]||`target ${i}`,v]).filter(([,v])=>count(v));
 table(root,'target-support',['TRAIN target','Observed labels'],support);
 text(root,'artifact',`Model ${t.artifact?.sha256||'not frozen yet'} · fit device ${t.fit_device||'unknown'}. ${t.selection_basis||'Fixed model before any native evaluation; validation outcomes do not select or tune it.'}`);
 const curves=rows(t.curves),colors=['#76cfd1','#d6a0dc','#dfc17d','#8bc39a','#e39787'],series=Array.from({length:5},(_,i)=>({label:`Member ${i+1} · full TRAIN loss`,color:colors[i],points:curves.filter(r=>r.member===i&&num(r.step)&&num(r.full_train_loss)).map(r=>[r.step,r.full_train_loss])}));
 plot(root,'loss',series);text(root,'loss-caption',`${curves.length} reported member/step loss records. Each line is one fitted member evaluated on the same full TRAIN set; bootstrap losses remain separate measurements. Loss is not a native capture rate; missing epochs/metrics are not filled in.`);
 table(root,'member-losses',['Member','Last reported step','Bootstrap TRAIN loss','Full TRAIN loss'],Array.from({length:5},(_,i)=>{const last=curves.filter(r=>r.member===i&&num(r.step)).sort((a,b)=>a.step-b.step).at(-1);return[i+1,last?.step??'unknown',fmt(last?.train_loss,4),fmt(last?.full_train_loss,4)];}));
 table(root,'metrics',['Split','Target','Observed support','MAE','RMSE','Brier','Accuracy'],rows(raw.prediction_metrics).map(r=>[r.split||'unknown',targetLabel(r.target),count(r.support)?r.support:'unknown',`${fmt(r.mae,4)}${r.unit?' '+r.unit:''}`,fmt(r.rmse,4),fmt(r.brier,4),pct(r.accuracy)]));
 table(root,'comparison',['Method','Requested','Native','Captures/all','Clean/all','Rejects/all','Flight misses/all','Invalid/all','Capture 95% interval'],rows(raw.test_comparison).map(r=>{const c=r.counts||{},n=requested(c);return[LABELS[r.method]||r.method,n??'unknown',count(c.native_releases)?c.native_releases:'unknown',ratio(c.captures,n),ratio(c.clean,n),ratio(c.rejected,n),ratio(c.native_misses,n),ratio(c.invalid,n),`${interval(r.capture_ci95)}${r.ci95_method?' ('+r.ci95_method+')':''}`];}));
 table(root,'paired',['Comparison a − b','a only','b only','Both capture','Neither','Difference','Paired 95% interval'],rows(raw.paired_differences).map(r=>[`${LABELS[r.a]||r.a} − ${LABELS[r.b]||r.b}`,r.a_only,r.b_only,r.shared,r.neither,pct(r.delta_rate),`${interval(r.ci95)}${r.interval_method?' ('+r.interval_method+')':''}`]));
 let message=ready?'Completed audited matched TEST results. No model prediction is counted as a simulator success.':waiting?`${fitted?'CPU model fit completed. ':''}Waiting for the 4090; no native evaluation result or method winner is claimed.`:raw.status==='failed'?`Study failed or stopped: ${raw.error||raw.reason||'inspect evidence'}. Results remain unqualified.`:raw.audit?.status==='failed'?'Independent audit failed; results remain unverified.':'Reported results are provisional until all264 requests, matched controls, frozen predictions and the full byte-bound independent audit are verified.';
 text(root,'result',message);at(root,'result')?.setAttribute('data-state',ready?'passed':raw.status==='failed'||raw.audit?.status==='failed'?'failed':'unknown');
 const method=at(root,'method');if(method){if(!METHODS.includes(method.value))method.value='ensemble';method.onchange=()=>coverage(root,raw);}coverage(root,raw);showTrace(root,raw);
 text(root,'audit',`Independent ${AUDIT_VERSION}: ${ready?'full completed proof verified':raw.audit?.status||'pending'} · ${count(raw.audit?.checks_count)?raw.audit.checks_count:'unknown'} checks. Exact report/manifest bytes: ${raw.audit?.bound_files_verified===true?'verified':'unverified'}. ${raw.audit?.verification_reason||''}`);
}

export function startPhysicsEnsemble(root,{fetchReport=async()=>{const r=await fetch('/api/study',{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json();},schedule=setTimeout,cancel=clearTimeout}={}){
 let stopped=false,timer;
 const poll=async()=>{try{const report=await fetchReport();if(stopped)return;renderPhysicsEnsemble(root,report?.[KEY]);text(root,'connection','Read-only snapshot · refresh every 5 seconds.');}catch(e){if(stopped)return;text(root,'connection',`Snapshot unavailable: ${e.message||e}. Previous observations may be stale.`);text(root,'result','Connection unavailable; previous displayed data is stale and unverified.');at(root,'result')?.setAttribute('data-state','unknown');}if(!stopped)timer=schedule(poll,5000);};poll();return()=>{stopped=true;if(timer!==undefined)cancel(timer);};
}
