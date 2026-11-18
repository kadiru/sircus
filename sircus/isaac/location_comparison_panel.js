/* Separate read-only publication. No execution or motion commands. */
export const SCHEMA='sircus-location-comparison-publication-v1';
export const MODELS=['parent','original_usable','broader'];
const COPIES=MODELS.slice(1);
const LABELS={parent:'Original seed 22',original_usable:'Continued · 31 original locations',broader:'Continued · 8,960 broader locations'};
const COLORS={parent:'#77858c',original_usable:'#416fa0',broader:'#0c8178'};
const PHASES={restore:'Restore model',checkpoint:'Save checkpoint',update:'PPO update',train:'Exploratory TRAIN',monitor:'Frozen monitoring',test_broad:'70-position TEST',test_local:'25-position nearby TEST'};
const METRICS={captures:'Captures',clean:'Clean captures',rejected:'Rejected requests'};
const n=x=>typeof x==='number'&&Number.isFinite(x)&&x>=0?x:null;
const c=x=>Number.isInteger(x)&&x>=0?x:null;
const s=x=>typeof x==='string'?x:'';
const hash=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
const integer=x=>c(x)===null?'unknown':x.toLocaleString('en-US');
const decimal=(x,d=1)=>n(x)===null?'unknown':x.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const seconds=x=>n(x)===null?'unknown':`${decimal(x,2)} s`;
const memory=x=>n(x)===null?'unknown':`${decimal(x)} MiB`;
const rate=(value,total)=>c(value)===null||c(total)===null||value>total||!total?'unavailable':`${value} / ${total} · ${decimal(100*value/total)}%`;
const el=(tag,text)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=String(text);return node;};
const clear=id=>{const node=document.getElementById(id);node.replaceChildren();return node;};
const note=(node,text,cls='note')=>{const p=el('p',text);p.className=cls;node.appendChild(p);return p;};
const pending=(node,text='Pending independent audit.')=>note(node,text,'pending');
function table(headers,rows){const t=el('table'),head=el('thead'),tr=el('tr');for(const h of headers)tr.appendChild(el('th',h));head.appendChild(tr);t.appendChild(head);const body=el('tbody');for(const row of rows){const r=el('tr');for(const value of row)r.appendChild(el('td',value));body.appendChild(r);}t.appendChild(body);return t;}

export function validMetrics(row,expected=null){
  return row&&['requests','captures','clean','rejected','misses'].every(k=>c(row[k])!==null)
    &&row.clean<=row.captures&&row.captures+row.rejected+row.misses===row.requests
    &&(expected===null||row.requests===expected);
}

export function recoveryVerified(report){
  const audit=report?.audit,failure=report?.costs?.shutdown_failure;
  return report?.schema===SCHEMA&&report.synthetic===false&&report.model_promoted===false
    &&report.status==='data_verified_shutdown_incomplete'&&audit?.status===report.status
    &&audit.complete===true&&audit.synthetic===false&&audit.original_audit_status==='not_passed'
    &&audit.native_shutdown_status==='failed'&&audit.native_clean_shutdown===false
    &&['manifest_fingerprint','comparison_fingerprint','audit_sha256','audit_exit_sha256','native_exit_sha256',
      'native_exit_fingerprint','audit_exit_fingerprint','trace_sha256','recovery_request_fingerprint',
      'inventory_sha256','retirement_proof_sha256'].every(k=>hash(audit[k]))
    &&['session_id','recovery_session_id'].every(k=>typeof audit[k]==='string'&&/^[a-f0-9]{32}$/.test(audit[k]))
    &&audit.session_id!==audit.recovery_session_id&&failure?.status==='failed'
    &&failure.original_audit_status==='not_passed'&&failure.clean_shutdown===false
    &&failure.reason==='independent watchdog wall deadline reached'&&failure.shared_phase_limit_s===3600
    &&s(failure.scope).length>0&&['native_watchdog_wall_s','last_progress_shared_phase_s','after_last_progress_wall_s'].every(k=>n(failure[k])!==null)
    &&failure.after_last_progress_wall_s<=failure.native_watchdog_wall_s
    &&failure.native_watchdog_wall_s===report.costs.native_watchdog_wall_s;
}

export function resultsAudited(report){
  const canonical=report?.schema===SCHEMA&&report.synthetic===false&&report.status==='completed_audited'
    &&report.audit?.status==='passed'&&report.audit.complete===true&&report.audit.synthetic===false
    &&report.model_promoted===false;
  if(!canonical&&!recoveryVerified(report))return false;
  return COPIES.every(key=>validMetrics(report.train?.[key],8960))
    &&MODELS.every(key=>validMetrics(report.final_tests?.[key]?.test_broad,70)
      &&validMetrics(report.final_tests?.[key]?.test_local,25));
}

export function statusView(report){
  if(report?.schema!==SCHEMA||report.synthetic!==false)return {state:'failed',badge:'Unavailable',title:'Native comparison evidence unavailable',detail:'A compatible native report has not been supplied. Synthetic fixtures are not performance results.'};
  if(resultsAudited(report)&&recoveryVerified(report))return {state:'failed',badge:'Data verified · shutdown failed',title:'Data verified; simulator shutdown failed',detail:'Both copies and their saved results passed a separate CPU recovery audit. The original final audit remains not passed: the simulator did not close within its shutdown limit. The failure and its cost are preserved. No model has been promoted.'};
  if(resultsAudited(report))return {state:'passed',badge:'Audit passed',title:'The equal-budget comparison is complete',detail:'Both copies completed 8,960 new TRAIN requests. All three final models were independently checked on the same fresh test positions. No model has been promoted.'};
  if(['completed_audited','data_verified_shutdown_incomplete'].includes(report.status))return {state:'failed',badge:'Evidence inconsistent',title:'Final results cannot be verified',detail:'The audit, preserved shutdown failure or required counts are missing or inconsistent. Outcome comparisons remain hidden.'};
  if(['stopped_no_retry','incomplete_no_retry','failed'].includes(report.status))return {state:'failed',badge:'Stopped · incomplete',title:'The attempt stopped; evidence is preserved',detail:s(report.reason)||s(report.error)||'The required comparison did not finish. Partial counts do not establish final performance, and no automatic retry is implied.'};
  if(report.status==='queued')return {state:'queued',badge:'Waiting',title:'Waiting for the 4090',detail:s(report.reason)||'Another workload owns the GPU. No training result is available for this attempt.'};
  if(report.status==='prepared')return {state:'prepared',badge:'Prepared',title:'The comparison is prepared',detail:'Both copies start from the saved seed-22 model. No new outcome is available yet.'};
  if(report.status==='running')return {state:'running',badge:'Running · unaudited',title:'The comparison is running',detail:'Recorded request counts show progress. Learning curves and held-out results appear only after the independent audit passes.'};
  if(['native_complete','awaiting_audit','audit_pending'].includes(report.status))return {state:'awaiting_audit',badge:'Audit pending',title:'Final performance is awaiting verification',detail:'Native execution and independent verification are separate. The report does not yet establish audited model performance.'};
  return {state:'failed',badge:'Unknown',title:'Current execution status is unknown',detail:'An unrecognized status cannot establish completion.'};
}

export function progressFor(report,model){
  const row=report?.progress?.[model];
  const cap=model==='assessment'?600:8960;
  if(!row||c(row.requested)===null||c(row.completed)===null||row.completed>row.requested||row.requested>cap)return null;
  return {requested:row.requested,completed:row.completed,unresolved:row.requested-row.completed,cap};
}

export function curveSeries(report,kind,metric='captures'){
  if(!resultsAudited(report)||!Object.hasOwn(METRICS,metric)||!['train_curves','monitor_curves'].includes(kind))return [];
  return COPIES.flatMap(model=>{
    const rows=report[kind]?.[model];if(!Array.isArray(rows)||rows.length!==(kind==='train_curves'?256:5))return [];
    let previous=-1;
    const valid=rows.every((row,index)=>{
      const accepted=validMetrics(row,35)&&c(row.train_requests)!==null
        &&row.train_requests>previous&&row.train_requests===(kind==='train_curves'?(index+1)*35:index*2240)
        &&(kind!=='monitor_curves'||row.source_model===(index===0?'parent':model));
      previous=row.train_requests;return accepted;
    });
    if(!valid)return [];
    return [{model,label:LABELS[model],color:COLORS[model],points:rows.map(row=>({x:row.train_requests,y:row[metric]/row.requests,row}))}];
  });
}

export function heldoutRows(report,phase){
  const denominator=phase==='test_broad'?70:25;
  if(!resultsAudited(report))return MODELS.map(key=>[LABELS[key],...Array(4).fill('pending audit')]);
  return MODELS.map(key=>{const row=report.final_tests[key][phase];return [LABELS[key],rate(row.captures,denominator),rate(row.clean,denominator),rate(row.rejected,denominator),integer(row.misses)];});
}

export function computeRows(report){
  if(!resultsAudited(report))return [];
  const result=[];
  for(const model of MODELS)for(const [phase,label] of Object.entries(PHASES)){
    const row=report.costs?.per_model_phase?.[model]?.[phase];if(!row||typeof row!=='object')continue;
    const operation=phase==='restore'?row.restore_wall_s:phase==='checkpoint'?row.checkpoint_wall_s:phase==='update'?row.update_wall_s:row.episode_wall_s;
    result.push([LABELS[model],label,seconds(operation),seconds(row.episode_process_cpu_s),seconds(row.proposal_wall_s),seconds(row.geometry_wall_s)]);
  }
  return result;
}

export function memoryRows(report){
  if(!resultsAudited(report))return [];
  const samples=report.costs?.memory_by_model_phase;
  if(!Array.isArray(samples))return [];
  return samples.filter(row=>MODELS.includes(row.model)&&Object.hasOwn(PHASES,row.phase)&&c(row.samples)!==null&&row.samples>0).map(row=>[
    LABELS[row.model],PHASES[row.phase],integer(row.samples),memory(row.lifetime_peak_rss_mib),memory(row.sampled_gpu_used_mib)]);
}

export function videoEntries(report){
  if(!resultsAudited(report)||!Array.isArray(report.videos))return [];
  const candidates=report.videos.filter(video=>typeof video.id==='string'&&MODELS.includes(video.model)
    &&['train','test_broad','test_local'].includes(video.phase)&&c(video.frames)!==null
    &&c(video.active_slots)!==null&&c(video.parked_slots)!==null&&video.active_slots+video.parked_slots===35
    &&hash(video.packet_sha256)&&typeof video.label==='string');
  const keys=candidates.map(v=>`${v.model}:${v.phase}:${v.phase==='train'?v.train_requests_before:0}`);
  const expected=[...COPIES.flatMap(m=>[0,2240,4480,6720].map(x=>`${m}:train:${x}`)),'parent:test_broad:0','parent:test_local:0'];
  if(candidates.length!==10||new Set(candidates.map(v=>v.id)).size!==10||new Set(keys).size!==10
    ||expected.some(k=>!keys.includes(k))||candidates.some(v=>v.frames<=0||v.active_slots!==(v.phase==='train'?35:v.phase==='test_broad'?24:9)))return [];
  return candidates.map(video=>{
      const m=video.media;
      const safe=m&&m.verified===true&&hash(m.sha256)&&n(m.duration_s)!==null&&m.duration_s>0
        &&typeof m.url==='string'&&/^\/media\/[a-zA-Z0-9_-][a-zA-Z0-9_.-]*\.mp4$/.test(m.url);
      const chapters=safe&&Array.isArray(video.chapters)?video.chapters.filter(ch=>n(ch.start_s)!==null&&ch.start_s<m.duration_s&&typeof ch.label==='string'):[];
      return {...video,media:safe?m:null,chapters};
    });
}

function plot(node,series,metric,title){
  if(!series.length){pending(node,'No audited curve is available.');return;}
  const NS='http://www.w3.org/2000/svg';
  const svg=(tag,attrs={})=>{const element=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))element.setAttribute(k,String(v));return element;};
  const root=svg('svg',{viewBox:'0 0 530 275',role:'img','aria-label':`${title}: ${METRICS[metric]} per requested position, including rejections`});
  const left=45,right=515,top=18,bottom=225;
  const x=v=>left+(right-left)*v/8960,y=v=>bottom-(bottom-top)*v;
  for(const level of [0,.25,.5,.75,1]){root.appendChild(svg('line',{x1:left,x2:right,y1:y(level),y2:y(level),stroke:'#dce4e1','stroke-width':1}));const text=svg('text',{x:left-8,y:y(level)+4,'text-anchor':'end',fill:'#607377','font-size':11});text.textContent=`${level*100}%`;root.appendChild(text);}
  for(const value of [0,2240,4480,6720,8960]){const text=svg('text',{x:x(value),y:245,'text-anchor':value===0?'start':value===8960?'end':'middle',fill:'#607377','font-size':11});text.textContent=integer(value);root.appendChild(text);}
  for(const line of series){root.appendChild(svg('polyline',{points:line.points.map(p=>`${x(p.x)},${y(p.y)}`).join(' '),fill:'none',stroke:line.color,'stroke-width':2.3}));if(line.points.length<=6)for(const point of line.points)root.appendChild(svg('circle',{cx:x(point.x),cy:y(point.y),r:3.4,fill:line.color}));}
  const label=svg('text',{x:280,y:269,'text-anchor':'middle',fill:'#607377','font-size':11});label.textContent='Additional TRAIN requests (rejections included)';root.appendChild(label);node.appendChild(root);
}

function renderModels(report){
  const grid=clear('modelCards');
  const assessment=progressFor(report,'assessment');document.getElementById('assessmentProgress').textContent=assessment?`Frozen monitoring and final assessment: ${integer(assessment.completed)} / 600 completed requests · ${integer(assessment.requested)} charged · ${integer(assessment.unresolved)} unresolved.`:'Frozen monitoring and final assessment: current request count unknown.';
  for(const model of MODELS){const card=el('div');card.className=`card model-card ${model}`;card.appendChild(el('h3',LABELS[model]));
    if(model==='parent'){note(card,'Saved starting model · no additional learning','small muted');note(card,'Frozen control','value');note(card,'Shares the same geometry checks and final test positions.','small muted');}
    else{const progress=progressFor(report,model);note(card,model==='original_usable'?'31 valid original locations, repeated':'8,960 distinct valid bucket locations','small muted');note(card,`${integer(progress?.completed)} / 8,960`,'value');note(card,'Completed additional TRAIN requests','small muted');if(progress){const bar=el('progress');bar.max=8960;bar.value=progress.completed;bar.setAttribute('aria-label',`${LABELS[model]} completed TRAIN requests`);card.appendChild(bar);note(card,`${integer(progress.requested)} charged · ${integer(progress.unresolved)} unresolved`,'small muted');}else note(card,'No current request count supplied.','small muted');}
    grid.appendChild(card);
  }
}

function renderCurves(report,metric){
  const train=curveSeries(report,'train_curves',metric),monitor=curveSeries(report,'monitor_curves',metric);
  document.getElementById('curveStatus').textContent=resultsAudited(report)?'Audited outcomes. TRAIN curves describe exploration; frozen monitoring describes direct predictions on the same 35 monitoring positions.':'Outcome curves remain hidden until the complete independent audit passes. Request progress is shown above.';
  const legend=clear('curveLegend');for(const model of COPIES){const item=el('span'),swatch=el('span');swatch.className='swatch';swatch.setAttribute('style',`background:${COLORS[model]}`);item.appendChild(swatch);item.appendChild(el('span',LABELS[model]));legend.appendChild(item);}
  plot(clear('trainChart'),train,metric,'Exploratory TRAIN · 35-request batches');plot(clear('monitorChart'),monitor,metric,'Frozen 35-position monitoring');
  const totals=clear('trainTotals');if(resultsAudited(report)){totals.appendChild(table(['Exploratory TRAIN totals','Captures','Clean captures','Rejected','Released misses'],COPIES.map(model=>{const row=report.train[model];return [LABELS[model],rate(row.captures,row.requests),rate(row.clean,row.requests),rate(row.rejected,row.requests),integer(row.misses)];})));}
  const values=clear('curveTable');if(!resultsAudited(report)){pending(values);return;}
  const rows=[];for(const [kind,series] of [['TRAIN · 35-request batch',train],['Frozen monitoring',monitor]])for(const line of series)for(const point of line.points)rows.push([kind,line.label,integer(point.x),rate(point.row[metric],point.row.requests)]);
  if(rows.length)values.appendChild(table(['Measurement','Model','New TRAIN requests',METRICS[metric]+' / requests'],rows));else pending(values,'Audited curve data was not included in this publication.');
}

function renderHeldout(report){
  const passed=resultsAudited(report);
  document.getElementById('heldoutStatus').textContent=passed?'Audited frozen direct inference. Compare each continued model with the original parent on exactly the same locations.':'Final held-out outcomes remain pending a complete independent audit. Native completion alone does not unlock this comparison.';
  for(const [id,phase] of [['broadResults','test_broad'],['localResults','test_local']])clear(id).appendChild(table(['Model','Captures','Clean captures','Rejected','Released misses'],heldoutRows(report,phase)));
  const reasons=clear('rejections');if(!passed){pending(reasons);return;}
  const rows=[];for(const model of MODELS)for(const [phase,label] of [['test_broad','70 broad'],['test_local','25 nearby']]){const values=report.final_tests[model][phase].rejection_reasons;if(!values||typeof values!=='object')continue;for(const [reason,count] of Object.entries(values))if(c(count)!==null)rows.push([LABELS[model],label,reason,integer(count)]);}
  if(rows.length)reasons.appendChild(table(['Model','Test positions','Recorded reason','Requests'],rows));else pending(reasons,'No rejection-reason breakdown supplied.');
}

function renderCompute(report){
  const phase=clear('phaseWall'),operations=clear('computeTable'),mem=clear('memoryTable');
  document.getElementById('computeScope').textContent=(s(report?.costs?.scope)||'Only measured scopes are shown. Geometry time is inside rollout time, not additional elapsed time. CPU and wall time are different measurements. Missing measurements stay unknown.')+' '+(s(report?.costs?.memory_scope)||'RSS is a lifetime high-water mark; GPU values are whole-device samples, not model memory or phase peaks.');
  if(!resultsAudited(report)){for(const node of [phase,operations,mem])pending(node);return;}
  const wall=report.costs?.observed_phase_wall_s??{};
  phase.appendChild(table(['Measured scope · overlapping scopes must not be added','Observed elapsed wall'],[[LABELS.original_usable,seconds(wall.original_usable)],[LABELS.broader,seconds(wall.broader)],[recoveryVerified(report)?'Shared phase recorded before shutdown · stale':'Shared bounded phase',seconds(wall.shared)],['Entire native process watchdog',seconds(report.costs?.native_watchdog_wall_s)],[recoveryVerified(report)?'Separate CPU recovery audit watchdog':'Separate independent audit watchdog',seconds(report.costs?.audit_watchdog_wall_s)]]));
  if(recoveryVerified(report)){
    const failure=report.costs.shutdown_failure,audit=report.costs.recovery_audit??{};
    note(phase,'Simulator shutdown failed; the original final audit remains not passed.','scope');
    phase.appendChild(table(['Preserved shutdown-failure scope · overlaps totals above','Measured value'],[
      ['Original native watchdog total, including failed shutdown',seconds(failure.native_watchdog_wall_s)],
      ['Shared phase at last progress update · stale, not final',seconds(failure.last_progress_shared_phase_s)],
      ['After last progress update, including shutdown wait',seconds(failure.after_last_progress_wall_s)],
      ['Original shared-phase limit',seconds(failure.shared_phase_limit_s)]]));
    note(phase,failure.scope,'small muted');
    phase.appendChild(table(['Separate CPU recovery audit','Measured value'],[
      ['Watchdog wall',seconds(audit.wall_s)],['Exact audit child user CPU',seconds(audit.cpu_user_s)],
      ['Exact audit child system CPU',seconds(audit.cpu_system_s)],
      ['Exact audit child lifetime peak RSS',memory(n(audit.peak_rss_bytes)===null?null:audit.peak_rss_bytes/1048576)]]));
    note(phase,'Audit CPU and RSS cover the exact audit child, including reaped descendants. These are separate from training cost.','small muted');
  }
  const rows=computeRows(report);if(rows.length)operations.appendChild(table(['Model','Operation / phase','Operation wall','Rollout CPU only','Proposal wall','Geometry wall · nested'],rows));else pending(operations,'No phase-operation measurements supplied.');
  const memoryTable=memoryRows(report);if(memoryTable.length)mem.appendChild(table(['Model','Phase','Samples','Max recorded process lifetime RSS','Max sampled whole-device GPU usage'],memoryTable));else pending(mem,'Memory measurements were not supplied.');
}

function renderVideos(report){
  const node=clear('videoList');
  if(!resultsAudited(report)){pending(node,'Audited trial recordings are not available yet.');return;}
  note(node,'Schedule: eight TRAIN recordings (four per continued model), plus two parent-only held-out subsets: 24 broad positions and 9 nearby positions. Final-model TEST videos were not scheduled.','small muted');
  const entries=videoEntries(report);if(!entries.length){pending(node,'No verified recording manifest supplied.');return;}
  const grid=el('div');grid.className='video-grid';node.appendChild(grid);
  for(const item of entries){const card=el('article');card.className='video-card';card.appendChild(el('h3',item.label));note(card,`${LABELS[item.model]} · ${PHASES[item.phase]} · ${integer(item.active_slots)} active / ${integer(item.parked_slots)} parked slots`,'small muted');if(item.phase==='train')note(card,`${integer(item.train_requests_before)} additional TRAIN requests completed before this recording.`,'small muted');note(card,`${integer(item.frames)} verified source frames`,'small muted');
    if(item.media){const video=el('video');video.controls=true;video.preload='metadata';video.playsInline=true;video.setAttribute('aria-label',item.label);video.src=item.media.url;card.appendChild(video);const chapters=el('div');chapters.className='chapters';for(const chapter of item.chapters){const button=el('button',`${decimal(chapter.start_s,1)} s · ${chapter.label}`);button.type='button';button.addEventListener('click',()=>{video.currentTime=chapter.start_s;});chapters.appendChild(button);}card.appendChild(chapters);note(card,'Chapters seek within this recording; playback is controlled above.','small muted');}
    else pending(card,'Source frames verified; playable video has not been published.');grid.appendChild(card);
  }
}

function renderEvidence(report){const node=clear('evidenceDetails');note(node,resultsAudited(report)?(recoveryVerified(report)?'Separate saved-data recovery audit: passed. Original final audit: not passed. Simulator shutdown: failed. Required scientific checks and comparison counts verified.':'Independent audit: passed; required comparison counts verified.'):'No complete independently audited comparison is established by this publication.');note(node,'No model is automatically promoted. Both trained copies start from the same saved seed-22 weights and optimizer state. This is neural-network continuation with continuous geometry checks, without a shot-settings library.');if(resultsAudited(report)){for(const model of MODELS){const value=report.checkpoints?.[model];const identity=value?.sha256??value?.checkpoint_sha256;if(hash(identity))note(node,`${LABELS[model]} checkpoint SHA256: ${identity}`,'hash');}for(const key of ['comparison_fingerprint','manifest_fingerprint','audit_sha256','trace_sha256','audit_exit_sha256','native_exit_sha256','inventory_sha256','retirement_proof_sha256'])if(hash(report.audit?.[key]))note(node,`${key.replaceAll('_',' ')}: ${report.audit[key]}`,'hash');}note(node,'The historical six-run sweep/PPO benchmark is separate. The simulator uses known bucket coordinates; these results do not validate camera perception, physical robot execution or unrestricted arm motion.','scope');}

let current=null;
export function render(report){current=report;const view=statusView(report);document.getElementById('statusCard').className='card status '+view.state;document.getElementById('statusBadge').textContent=view.badge;document.getElementById('statusTitle').textContent=view.title;document.getElementById('statusDetail').textContent=view.detail;const time=s(report?.generated_at);document.getElementById('freshness').textContent=time&&Number.isFinite(Date.parse(time))?`Publication generated: ${new Date(time).toLocaleString()} · a dated report, not proof of a live process`:'Publication time unavailable.';renderModels(report);const metric=document.getElementById('curveMetric').value||'captures';renderCurves(report,metric);renderHeldout(report);renderCompute(report);renderVideos(report);renderEvidence(report);}
export async function loadReport(fetcher=globalThis.fetch){const response=await fetcher('/api/location-comparison',{cache:'no-store'});if(!response.ok)throw new Error('Report HTTP '+response.status);const report=await response.json();render(report);return report;}
let loading=false;
export async function refresh(fetcher=globalThis.fetch){if(loading)return;loading=true;try{await loadReport(fetcher);}catch(error){render(null);document.getElementById('statusTitle').textContent='Could not read current comparison evidence';document.getElementById('statusDetail').textContent=String(error.message||error);}finally{loading=false;}}
if(typeof document!=='undefined'){document.getElementById('refresh')?.addEventListener('click',()=>refresh());document.getElementById('curveMetric')?.addEventListener('change',()=>renderCurves(current,document.getElementById('curveMetric').value));refresh();setInterval(()=>refresh(),15000);}
