/* Read-only rendering. A complete native run is not an independently audited result. */
const SCHEMA='sircus-candidate-native-comparison-report-v1';
const METHOD_KEYS=['parent','bank'];
const LABELS={parent:'Original seed 22',bank:'Candidate-bank selector'};
const COLORS={parent:'#4b6c8b',bank:'#087e78'};
const STATES={prepared:'Prepared',queued:'Queued',running:'Running',native_complete:'Native complete · audit pending',audit_pending:'Audit pending',passed:'Audit passed',failed:'Stopped'};
const number=x=>typeof x==='number'&&Number.isFinite(x)?x:null;
const count=x=>Number.isInteger(x)&&x>=0?x:null;
const txt=x=>typeof x==='string'?x:'';
const field=(x,...keys)=>keys.map(k=>x?.[k]).find(v=>v!==undefined);
const el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=String(text);return n;};
const clear=id=>{const n=document.getElementById(id);n.replaceChildren();return n;};
const cell=(row,value,tag='td')=>row.appendChild(el(tag,value));
const fixed=(x,d=2)=>number(x)===null?'unknown':x.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const integer=x=>count(x)===null?'unknown':x.toLocaleString('en-US');
const seconds=x=>number(x)===null||x<0?'unknown':`${fixed(x)} s`;
const memory=x=>number(x)===null||x<0?'unknown':`${fixed(x,1)} MiB`;

// Optional additive publication. The historical 70-request report stays at the
// root; this panel never treats its earlier audit as a continuation audit.
const CONTINUATION_SCHEMA='sircus-candidate-continuation-published-v1';
const CONTINUATION_METHODS=['parent','candidate_bank','continued'];
const CONTINUATION_LABELS={parent:'Original seed 22',candidate_bank:'Frozen bank selector',continued:'Continued seed 22'};
const SPLITS=[['train','TRAIN',35],['validation','Fresh validation',35],['test','Fresh TEST',70]];
const PHASE_LABELS={initialization:'Initialization',imitation:'Imitation',train:'Native TRAIN',assessment:'Frozen assessment',audit:'Independent audit'};

export function continuationAudited(value){
  if(value?.schema!==CONTINUATION_SCHEMA||value.status!=='completed_audited'||value.audit?.status!=='passed'||value.audit.all_components_passed!==true||value.audit.synthetic!==false||value.model_promoted!==false)return false;
  const counters=value.counters??value.progress??{};
  if(count(counters.train_requests)===null||counters.train_requests>1750||count(counters.train_releases)===null||counters.train_releases>1400||counters.train_releases>counters.train_requests||(counters.train_requests!==1750&&counters.train_releases!==1400)||counters.assessment_requests!==420)return false;
  let releases=0;
  for(const method of CONTINUATION_METHODS)for(const [split,,requested] of SPLITS){
    const row=value.comparison?.methods?.[method]?.[split];
    if(!row||row.requested!==requested||['released','rejected','captures','clean_captures'].some(key=>count(row[key])===null)||row.released+row.rejected!==requested||row.clean_captures>row.captures||row.captures>row.released)return false;
    releases+=row.released;
  }
  return counters.assessment_releases===releases;
}

export function continuationStatus(value){
  if(value?.schema!==CONTINUATION_SCHEMA)return {state:'failed',badge:'Unavailable',title:'Continuation evidence unavailable',detail:'A compatible continuation publication has not been supplied.'};
  if(continuationAudited(value))return {state:'passed',badge:'Audit passed',title:'Bounded continuation completed',detail:'Learning, native TRAIN, startup and the final 420-request assessment passed the independent audit. The model has not been promoted.'};
  if(value.status==='completed_audited')return {state:'failed',badge:'Evidence incomplete',title:'Continuation results are not verified',detail:'The supplied audit or final counts are incomplete or inconsistent. Performance results remain hidden.'};
  if(['failed','incomplete_no_retry'].includes(value.status))return {state:'failed',badge:'Stopped',title:'Continuation stopped; evidence preserved',detail:txt(value.error)||'The bounded application did not complete its required audits. There is no automatic retry or model promotion.'};
  if(['native_complete','audit_pending'].includes(value.status))return {state:'audit_pending',badge:'Audit pending',title:'Native execution finished; audit pending',detail:'Native completion does not establish a passed audit. Final capture, rejection and cost comparisons remain pending.'};
  if(value.status==='running')return {state:'running',badge:'Running',title:'Bounded seed-22 continuation in progress',detail:'The existing policy is being continued under fixed limits. Final assessment outcomes remain hidden until all independent audits pass.'};
  return {state:'failed',badge:'Unavailable',title:'Continuation status unavailable',detail:'An unrecognized execution state cannot establish completion.'};
}

export function continuationRows(value){
  const audited=continuationAudited(value),unverified=['failed','incomplete_no_retry'].includes(value?.status)?'not audited':'pending audit';
  return SPLITS.flatMap(([split,label,requested])=>CONTINUATION_METHODS.map(method=>{
    const row=value?.comparison?.methods?.[method]?.[split];
    return [label,CONTINUATION_LABELS[method],...(!audited?Array(4).fill(unverified):[`${row.captures} / ${requested} · ${fixed(100*row.captures/requested,1)}%`,`${row.clean_captures} / ${requested}`,`${row.rejected} / ${requested}`,`${row.released} / ${requested}`])];
  }));
}

export function continuationComputeRows(value){
  const unverified=['failed','incomplete_no_retry'].includes(value?.status)?'not audited':'pending audit';
  return Object.entries(PHASE_LABELS).map(([phase,label])=>{
    const row=value?.resources?.phases?.[phase]??{},device=row.device??{};
    return [label,...(!continuationAudited(value)?Array(7).fill(unverified):[seconds(row.wall_s),seconds(row.aggregate_cpu_s),memory(row.lifetime_peak_rss_mib),memory(row.largest_child_lifetime_peak_rss_mib),memory(device.sampled_whole_device_occupied_mib),memory(device.torch_peak_allocated_mib),memory(device.torch_peak_reserved_mib)])];
  });
}

// Failed producer receipts are useful cost evidence, but never a substitute
// for the separate final-comparison or independently audited compute gates.
export function continuationStoppedResources(value){
  const stopped=value?.stopped_resources,record=x=>x!==null&&typeof x==='object'&&!Array.isArray(x)
    &&[Object.prototype,null].includes(Object.getPrototypeOf(x));
  if(value?.schema!==CONTINUATION_SCHEMA||!['failed','incomplete_no_retry'].includes(value.status)
    ||!record(stopped)||typeof stopped.producer_receipt_sha256!=='string'||!/^[a-f0-9]{64}$/i.test(stopped.producer_receipt_sha256)
    ||typeof stopped.scope!=='string'||!stopped.scope.trim()||!record(stopped.phases)||!Object.keys(stopped.phases).length)return null;
  if(Object.entries(stopped.phases).some(([phase,row])=>!Object.hasOwn(PHASE_LABELS,phase)||!record(row)
    ||(row.device!==undefined&&row.device!==null&&!record(row.device))))return null;
  const phases=Object.entries(PHASE_LABELS).filter(([phase])=>phase!=='audit'||Object.hasOwn(stopped.phases,phase));
  return {producer_receipt_sha256:stopped.producer_receipt_sha256,scope:stopped.scope,
    rows:phases.map(([phase,label])=>{
      const row=stopped.phases[phase]??{},device=row.device??{};
      return [phase==='assessment'?'Partial frozen assessment':label,seconds(row.wall_s),seconds(row.aggregate_cpu_s),
        memory(row.lifetime_peak_rss_mib),memory(row.largest_child_lifetime_peak_rss_mib),
        memory(device.sampled_whole_device_occupied_mib),memory(device.torch_peak_allocated_mib),memory(device.torch_peak_reserved_mib)];
    })};
}

export function continuationTrainMonitoring(value){
  const monitoring=value?.train_monitoring,totals=monitoring?.totals;
  const valid=row=>row&&['terminal','captures','clean_captures','released','rejected'].every(key=>count(row[key])!==null)
    &&row.terminal<=1750&&row.released+row.rejected===row.terminal&&row.clean_captures<=row.captures&&row.captures<=row.released;
  if(value?.schema!==CONTINUATION_SCHEMA||monitoring?.audited!==false||!valid(totals))return null;
  if(totals.requested!==null&&(count(totals.requested)===null||totals.requested>1750||totals.requested<totals.terminal))return null;
  if(totals.unresolved!==null&&(count(totals.unresolved)===null||totals.requested===null||totals.unresolved!==totals.requested-totals.terminal))return null;
  let curve=[];
  if(Array.isArray(monitoring.curve)&&monitoring.curve.length<=1751){
    let previous=null;
    const consistent=monitoring.curve.every(row=>{
      const accepted=valid(row)&&row.terminal>0&&row.terminal<=totals.terminal
        &&['captures','clean_captures','released','rejected'].every(key=>row[key]<=totals[key])
        &&(!previous||(row.terminal>previous.terminal&&['captures','clean_captures','released','rejected'].every(key=>row[key]>=previous[key])));
      previous=row;return accepted;
    });
    if(consistent)curve=monitoring.curve;
  }
  const prefix=valid(monitoring.latest_terminal_prefix)&&monitoring.latest_terminal_prefix.terminal<=totals.terminal?monitoring.latest_terminal_prefix:null;
  return {totals,curve,prefix};
}

export function methodData(report,key){
  const raw=Array.isArray(report?.methods)?report.methods.find(x=>x.method===key):report?.methods?.[key];
  const value=raw??{};
  return {...value,requests:count(value.requests),planned_requests:count(value.planned_requests)??35,
    captures:count(field(value,'captures','captured')),clean:count(value.clean),
    rejected:count(value.rejected),released:count(field(value,'released','release')),compute:value.compute??{}};
}

export function resultsAudited(report){
  if(report?.schema!==SCHEMA||report.status!=='passed'||report.audit?.status!=='passed')return false;
  return METHOD_KEYS.every(key=>{const m=methodData(report,key);return m.requests===35&&m.planned_requests===35&&m.captures!==null&&m.clean!==null&&m.rejected!==null&&m.released!==null&&m.clean<=m.captures&&m.captures<=m.released&&m.released+m.rejected===m.requests;});
}

export function statusView(report){
  if(report?.schema!==SCHEMA)return {badge:'Unavailable',title:'Comparison report unavailable',detail:'A compatible bound report has not been supplied.',state:'failed'};
  const state=STATES[report.status]?report.status:'failed';
  if(resultsAudited(report))return {badge:'Audit passed',title:'Both frozen methods completed',detail:'70 requested TRAIN shots completed. Native outcomes and source bindings passed the independent audit.',state:'passed'};
  if(state==='passed')return {badge:'Evidence incomplete',title:'Final results are not verified',detail:'The report does not contain complete, consistent audited counts. Results remain hidden.',state:'failed'};
  if(state==='queued')return {badge:STATES[state],title:'Waiting for the 4090',detail:txt(report.queue?.reason)||'Another workload owns the GPU. It will not be interrupted for this comparison.',state};
  if(state==='running')return {badge:STATES[state],title:'Native comparison in progress',detail:'The weights stay frozen. Final capture and rejection results remain pending the independent audit.',state};
  if(state==='native_complete'||state==='audit_pending')return {badge:STATES[state],title:'Native execution finished; audit pending',detail:'Native completion is recorded separately from audited results. Capture and cost comparisons remain hidden until the audit passes.',state};
  if(state==='failed')return {badge:STATES[state],title:'Comparison needs attention',detail:txt(report.error)||txt(report.audit?.error)||'Execution or verification did not pass. No completed performance claim is shown.',state};
  return {badge:STATES.prepared,title:'Comparison prepared',detail:'Two frozen methods will each request the same 35 TRAIN positions. No native result is available yet.',state:'prepared'};
}

export function methodRows(report){
  return METHOD_KEYS.map(key=>{const m=methodData(report,key);if(!resultsAudited(report))return [LABELS[key],'pending audit','pending audit','pending audit','pending audit'];return [LABELS[key],`${m.captures} / ${m.requests} · ${fixed(100*m.captures/m.requests,1)}%`,`${m.clean} / ${m.requests}`,`${m.rejected} / ${m.requests}`,`${m.released} / ${m.requests}`];});
}

export function computeRows(report){
  const audited=resultsAudited(report);
  return METHOD_KEYS.map(key=>{const c=methodData(report,key).compute;return [LABELS[key],...(!audited?Array(5).fill('pending audit'):[seconds(field(c,'wall_s','native_wall_s')),seconds(c.setup_wall_s),seconds(c.score_wall_s),seconds(field(c,'cpu_s','process_cpu_s')),seconds(field(c,'selection_s','selection_wall_s'))])];});
}

export function memoryRows(report){
  const audited=resultsAudited(report);
  return METHOD_KEYS.map(key=>{const c=methodData(report,key).compute;return [LABELS[key],...(!audited?Array(4).fill('pending audit'):[memory(c.peak_rss_mib),memory(c.sampled_device_occupied_mib),memory(c.cuda_peak_allocated_mib),memory(c.cuda_peak_reserved_mib)])];});
}

export function pairedText(report){
  if(!resultsAudited(report))return 'Paired gains and losses are pending the independent audit.';
  const p=report.paired??{};
  const gains=count(field(p,'gains','capture_gains')),losses=count(field(p,'losses','capture_losses'));
  const changed=count(field(p,'changed_actions','actions_changed'));
  return `Capture gains: ${integer(gains)}. Capture losses: ${integer(losses)}. Changed shot choices: ${integer(changed)} / 35.`;
}

function table(headers,rows){const result=el('table');const head=el('thead'),tr=el('tr');for(const h of headers)cell(tr,h,'th');head.appendChild(tr);result.appendChild(head);const body=el('tbody');for(const row of rows){const tr=el('tr');for(const value of row)cell(tr,value);body.appendChild(tr);}result.appendChild(body);return result;}
function line(parent,label,value){const p=el('p');p.appendChild(el('strong',label+' '));p.appendChild(el('span',value));parent.appendChild(p);}
function pending(parent,text='Pending audited native results.'){parent.appendChild(Object.assign(el('p',text),{className:'pending'}));}

function renderChart(report){
  const node=clear('rateChart');
  if(!resultsAudited(report)){pending(node);return;}
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 620 205');svg.setAttribute('role','img');svg.setAttribute('aria-label','Capture and clean-capture rates across all 35 requested positions per method');
  const add=(name,attrs,text)=>{const n=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=String(text);svg.appendChild(n);return n;};
  const metrics=[['Capture','captures'],['Clean capture','clean']];
  for(let group=0;group<metrics.length;group++){
    const [label,metric]=metrics[group],top=group*96;
    add('text',{x:0,y:top+20,fill:'#5e7075','font-size':13},label);
    for(let j=0;j<METHOD_KEYS.length;j++){
      const key=METHOD_KEYS[j],m=methodData(report,key),rate=m[metric]/m.requests,y=top+32+j*24;
      add('rect',{x:0,y,width:500,height:14,rx:3,fill:'#edf1ef'});
      add('rect',{x:0,y,width:500*rate,height:14,rx:3,fill:COLORS[key]});
      add('text',{x:510,y:y+12,fill:COLORS[key],'font-size':13},`${m[metric]}/35 · ${fixed(rate*100,1)}%`);
    }
  }
  node.appendChild(svg);
}

function renderMetrics(report){const node=clear('metrics'),audited=resultsAudited(report);for(const [title,key] of [['Captures / all requests','captures'],['Clean captures / all requests','clean'],['Rejected requests / all requests','rejected']]){const card=Object.assign(el('div'),{className:'card metric'});card.appendChild(Object.assign(el('div',title),{className:'metric-label'}));for(const method of METHOD_KEYS){const m=methodData(report,method);card.appendChild(Object.assign(el('div',audited?`${m[key]} / 35`:'Pending'),{className:'pair-value '+method}));card.appendChild(Object.assign(el('p',LABELS[method]),{className:'muted'}));}node.appendChild(card);}}

const actionRows=action=>[['Board X',action?.board_x_m,1000,'mm'],['Board Y',action?.board_y_m,1000,'mm'],['Board height',action?.height_m,1000,'mm'],['Drop gap',action?.gap_m,1000,'mm'],['Tilt',action?.tilt_deg,1,'°']].map(([name,v,scale,unit])=>[name,number(v)===null?'unknown':`${fixed(v*scale,2)} ${unit}`]);
function outcome(value){if(value?.rejection||value?.invalid)return 'Rejected: '+txt(value.rejection||value.invalid);if(value?.capture===true||value?.captured===true||value?.route_success===true)return value?.clean===true||value?.route_quality===2?'Captured · clean':'Captured';if(value?.capture===false||value?.captured===false||value?.route_success===false)return 'Released · no capture';return 'Unknown native outcome';}

export function renderCase(report,caseId){
  const node=clear('caseDetail');
  if(!resultsAudited(report)){pending(node,'Paired native outcomes remain pending audit.');return;}
  const rows=Array.isArray(report.cases)?report.cases:[];
  const selected=rows.find(c=>c.cell_id===caseId)??rows[0];if(!selected){pending(node,'No per-position evidence supplied.');return;}
  const xy=selected.bucket_xy_m;if(Array.isArray(xy)&&xy.length===2&&xy.every(x=>number(x)!==null))node.appendChild(Object.assign(el('p',`Bucket: X ${fixed(xy[0]*1000,1)} mm · Y ${fixed(xy[1]*1000,1)} mm. ${selected.action_changed===true?'Shot settings changed.':selected.action_changed===false?'Shot settings unchanged.':'Action change not supplied.'}`),{className:'muted'}));
  const grid=Object.assign(el('div'),{className:'case-grid'});for(const key of METHOD_KEYS){const value=selected[key]??{};const box=Object.assign(el('div'),{className:'case-box'});box.appendChild(Object.assign(el('h3',LABELS[key]),{className:key}));box.appendChild(el('p',outcome(value)));box.appendChild(table(['Requested setting','Value'],actionRows(value.action)));if(value.reason||value.selection_reason)box.appendChild(Object.assign(el('p',txt(value.reason||value.selection_reason)),{className:'muted'}));grid.appendChild(box);}node.appendChild(grid);
}

function renderCases(report){const node=clear('caseList');if(!resultsAudited(report)){pending(node,'Position outcomes appear after the independent audit.');renderCase(report);return;}const rows=Array.isArray(report.cases)?report.cases:[];if(!rows.length){pending(node,'No per-position evidence supplied.');renderCase(report);return;}const label=el('label','Choose a paired TRAIN position');label.setAttribute('for','caseSelect');node.appendChild(label);const select=el('select');select.id='caseSelect';for(const row of rows){const option=el('option',`${row.cell_id} · ${outcome(row.parent)} → ${outcome(row.bank)}`);option.value=row.cell_id;select.appendChild(option);}select.addEventListener('change',()=>renderCase(report,select.value));node.appendChild(select);renderCase(report,rows[0].cell_id);}

function renderSources(report){const node=clear('sourceIdentity');line(node,'Audit:',txt(report.audit?.status)||'pending');if(report.audit?.sha256)node.appendChild(Object.assign(el('p','Audit SHA256: '+txt(report.audit.sha256)),{className:'hash'}));const source=report.source_binding??{};for(const [label,key] of [['Parent checkpoint SHA256','checkpoint_sha256'],['Candidate bank SHA256','candidate_bank_sha256'],['Frozen comparison plan SHA256','plan_sha256']])if(source[key])node.appendChild(Object.assign(el('p',label+': '+txt(source[key])),{className:'hash'}));const scopes=clear('scopeList');const lines=Array.isArray(report.scope)?report.scope:[txt(report.scope)].filter(Boolean);for(const text of lines)scopes.appendChild(el('li',String(text)));if(!lines.length)scopes.appendChild(el('li','Known TRAIN positions; frozen model weights; no optimizer updates or held-out generalization claim.'));}

function renderTrainMonitoring(value){
  const node=clear('continuationTrainMonitoring'),monitoring=continuationTrainMonitoring(value);
  node.className=monitoring?'case-box':'case-box hidden';
  if(!monitoring)return;
  const {totals,curve,prefix}=monitoring,rate=n=>totals.terminal?`${integer(n)} / ${integer(totals.terminal)} · ${fixed(100*n/totals.terminal,1)}%`:`${integer(n)} / 0 · rate unavailable`;
  node.appendChild(el('h3','Exploratory TRAIN monitoring'));
  node.appendChild(Object.assign(el('p','Unaudited exploration on TRAIN positions. Frozen held-out performance remains gated by all independent audits.'),{className:'scope'}));
  if(['failed','incomplete_no_retry'].includes(value.status))node.appendChild(el('p','Stopped attempt: these are preserved TRAIN journal counts. The pilot did not complete.'));
  const wrapper=Object.assign(el('div'),{className:'table-wrap'});
  wrapper.appendChild(table(['Charged requests','Completed TRAIN requests','Unresolved charged requests','Captures / completed','Clean captures / completed','Released','Rejected'],[[integer(totals.requested),integer(totals.terminal),integer(totals.unresolved),rate(totals.captures),rate(totals.clean_captures),integer(totals.released),integer(totals.rejected)]]));
  node.appendChild(wrapper);
  node.appendChild(Object.assign(el('p','Rates divide by completed TRAIN requests, including rejections. Unresolved charged requests remain separate until a terminal outcome is recorded.'),{className:'compute-note'}));
  if(prefix)line(node,'Latest recorded batch outcomes:',`${integer(prefix.terminal)} requests · ${integer(prefix.captures)} captures · ${integer(prefix.clean_captures)} clean · ${integer(prefix.released)} released · ${integer(prefix.rejected)} rejected.`);
  if(!curve.length){pending(node,totals.terminal?'Cumulative TRAIN curve unavailable.':'No completed TRAIN requests yet; capture rates are unavailable.');return;}
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
  svg.setAttribute('viewBox','0 0 620 240');svg.setAttribute('role','img');svg.setAttribute('aria-label','Unaudited cumulative TRAIN capture and clean-capture rates across completed TRAIN requests, including rejections');
  const add=(name,attrs,text)=>{const n=document.createElementNS(ns,name);for(const [key,v] of Object.entries(attrs))n.setAttribute(key,String(v));if(text!==undefined)n.textContent=String(text);svg.appendChild(n);return n;};
  const left=50,right=595,top=22,bottom=172,maxX=totals.terminal;
  for(const percentage of [0,50,100]){const y=bottom-(bottom-top)*percentage/100;add('line',{x1:left,x2:right,y1:y,y2:y,stroke:'#dce4e1'});add('text',{x:left-8,y:y+4,'text-anchor':'end',fill:'#5e7075','font-size':12},percentage+'%');}
  add('line',{x1:left,x2:left,y1:top,y2:bottom,stroke:'#90a19c'});
  add('text',{x:left,y:bottom+19,fill:'#5e7075','font-size':12},'0');
  add('text',{x:right,y:bottom+19,'text-anchor':'end',fill:'#5e7075','font-size':12},integer(maxX));
  add('text',{x:(left+right)/2,y:225,'text-anchor':'middle',fill:'#5e7075','font-size':13},'Completed TRAIN requests (including rejections)');
  for(const [key,label,color,dash] of [['captures','Capture','#087e78',''],['clean_captures','Clean capture','#946e29','5 3']]){
    const points=curve.map(row=>`${left+(right-left)*row.terminal/maxX},${bottom-(bottom-top)*row[key]/row.terminal}`).join(' ');
    add('polyline',{points,fill:'none',stroke:color,'stroke-width':2.5,'stroke-dasharray':dash});
    const last=curve[curve.length-1];add('circle',{cx:left+(right-left)*last.terminal/maxX,cy:bottom-(bottom-top)*last[key]/last.terminal,r:3,fill:color});
    const x=key==='captures'?left:left+150;add('text',{x,y:12,fill:color,'font-size':12},label+(dash?' (dashed)':''));
  }
  const chart=Object.assign(el('div'),{className:'chart'});chart.appendChild(svg);node.appendChild(chart);
}

function renderStoppedResources(value){
  const node=clear('continuationStoppedResources'),resources=continuationStoppedResources(value);
  node.className=resources?'case-box':'case-box hidden';
  if(!resources)return;
  node.appendChild(el('h3','Stopped-run resource measurements'));
  node.appendChild(Object.assign(el('p','Producer receipts for an incomplete attempt; full assessment and audit did not finish; not independently remeasured.'),{className:'scope'}));
  const wrapper=Object.assign(el('div'),{className:'table-wrap'});
  wrapper.appendChild(table(['Phase','Wall','Aggregate CPU','Lifetime RSS high-water','Largest child RSS high-water','Sampled whole-device occupancy','Torch CUDA allocated peak','Torch CUDA reserved peak'],resources.rows));
  node.appendChild(wrapper);
  node.appendChild(Object.assign(el('p',resources.scope),{className:'compute-note'}));
  node.appendChild(Object.assign(el('p','Producer receipt SHA256: '+resources.producer_receipt_sha256),{className:'hash'}));
}

export function renderContinuation(value){
  const section=document.getElementById('continuation'),nav=document.getElementById('continuationNav');
  if(!section||!nav)return;
  section.className=value===undefined?'card hidden':'card';nav.className=value===undefined?'hidden':'';
  for(const id of ['continuationProgress','continuationResults','continuationCompute','continuationEvidence','continuationPaired'])clear(id);
  renderTrainMonitoring(value);
  renderStoppedResources(value);
  if(value===undefined)return;
  const view=continuationStatus(value),card=document.getElementById('continuationStatus');card.className='status '+view.state;
  document.getElementById('continuationBadge').textContent=view.badge;document.getElementById('continuationTitle').textContent=view.title;document.getElementById('continuationDetail').textContent=view.detail;
  const progress=clear('continuationProgress'),counters=value?.counters??value?.progress??{},phase=value?.phase??value?.progress?.phase;
  line(progress,'Current phase:',PHASE_LABELS[phase]??'unknown');
  line(progress,'Charged native TRAIN requests:',`${integer(counters.train_requests)} / 1,750 maximum`);
  line(progress,'Native TRAIN releases:',`${integer(counters.train_releases)} / 1,400 maximum`);
  line(progress,'Charged frozen-assessment requests:',`${integer(counters.assessment_requests)} / 420 planned`);
  line(progress,'Frozen-assessment releases:',`${integer(counters.assessment_releases)} / 420 maximum`);
  if(value?.learning){
    const learning=value.learning,committed=learning.committed_update_counters??learning;
    const description=row=>`Imitation steps ${integer(row.imitation_steps)} / 240 maximum · PPO updates ${integer(row.ppo_updates)} / 13 maximum · optimizer steps ${integer(row.optimizer_steps)}.`;
    line(progress,'Recorded/committed update receipts:',description(committed));
    if(learning.committed_update_counters)line(progress,'Latest reported counters:',description(learning));
    if(learning.unreceipted_optimizer_work_possible===true)progress.appendChild(el('p','Additional in-flight optimizer work is unknown. No final checkpoint is committed in these receipts.'));
  }
  progress.appendChild(Object.assign(el('p','Charged requests include reservations and rejections; they are not a count of completed releases. TRAIN stops at the first request or release cap. Assessment uses 35 TRAIN + 35 fresh validation + 70 fresh TEST positions per method. One application attempt, no automatic retry; 3-hour application limit and 6 GiB storage floor.'),{className:'compute-note'}));
  const results=clear('continuationResults');results.appendChild(table(['Split','Method','Captures / requests','Clean / requests','Rejected / requests','Released / requests'],continuationRows(value)));
  const paired=clear('continuationPaired');
  if(continuationAudited(value)){
    for(const control of ['candidate_bank','parent']){
      const pair=value.comparison.test_pairs?.['continued_minus_'+control];
      if(pair)line(paired,`Fresh TEST vs ${CONTINUATION_LABELS[control]}:`,`Capture gains ${integer(pair.captured?.gains)}, losses ${integer(pair.captured?.losses)}; clean-capture gains ${integer(pair.clean?.gains)}, losses ${integer(pair.clean?.losses)}.`);
    }
    paired.appendChild(Object.assign(el('p','Paired comparisons use the same 70 fresh TEST positions. Position-wise bootstrap intervals describe this grid and one continuation, not uncertainty across independent training seeds. Historical TRAIN references remain separate from these newly measured controls.'),{className:'compute-note'}));
  }else pending(paired,['failed','incomplete_no_retry'].includes(value?.status)?'No audited final paired comparison is available for this stopped attempt.':'Final paired outcomes remain pending a complete independent audit.');
  const compute=clear('continuationCompute');compute.appendChild(table(['Phase','Wall','Aggregate CPU','Lifetime RSS high-water','Largest child RSS high-water','Sampled device occupancy','Torch CUDA allocated peak','Torch CUDA reserved peak'],continuationComputeRows(value)));
  compute.appendChild(Object.assign(el('p',txt(value?.resources?.scope)||'Phase timings are separate. CPU includes the application and waited/reaped children; audit-child cost overlaps the application audit phase. RSS is lifetime high-water, not isolated phase memory. Device occupancy includes PhysX/renderer; Torch peaks are not model-only memory. Missing measurements stay unknown.'),{className:'compute-note'}));
  const evidence=clear('continuationEvidence');line(evidence,'Continuation audit:',txt(value?.audit?.status)||'not supplied');
  const finalSha=value?.final_checkpoint_sha256??(value?.final_checkpoint?.status==='saved_audit_pending'?value.final_checkpoint.sha256:undefined);
  if(finalSha&&value?.learning?.unreceipted_optimizer_work_possible!==true)evidence.appendChild(Object.assign(el('p','Saved final checkpoint SHA256: '+txt(finalSha)),{className:'hash'}));
  evidence.appendChild(el('p','No automatic model promotion. This is extra-budget continuation of one seed; it is not a new equal-cost three-seed benchmark. The original frozen benchmark and the historical 70-request comparison remain separate.'));
  evidence.appendChild(el('p','No trial videos are recorded by this continuation protocol. Physics traces support the audits; existing benchmark videos remain in the original frozen interface.'));
}

export function render(report){
  const view=statusView(report),card=document.getElementById('statusCard');card.className='card status '+view.state;document.getElementById('statusBadge').textContent=view.badge;document.getElementById('statusTitle').textContent=view.title;document.getElementById('statusDetail').textContent=view.detail;
  const completed=METHOD_KEYS.reduce((sum,key)=>sum+(methodData(report,key).requests??0),0);document.getElementById('progress').textContent=`Native requests recorded: ${completed} / 70 · Optimizer updates: 0`;
  renderMetrics(report);renderChart(report);const pair=clear('paired');pair.appendChild(el('p',pairedText(report)));pair.appendChild(table(['Method','Captures','Clean','Rejected','Released'],methodRows(report)));
  const compute=clear('computeTable');compute.appendChild(table(['Method','Native wall','Setup + settling','Score phase','Process CPU','Selection'],computeRows(report)));compute.appendChild(el('h3','Memory measurements'));compute.appendChild(table(['Method','Peak process RSS','Sampled device occupancy','Torch CUDA allocated peak','Torch CUDA reserved peak'],memoryRows(report)));compute.appendChild(Object.assign(el('p','Device occupancy is a whole-GPU sample, not a peak or model-only memory. Torch allocations cover Torch tensors and omit most simulator memory.'),{className:'compute-note'}));
  const shared=clear('sharedCompute');if(resultsAudited(report)){const prep=report.preprocessing??report.preparation??{};const init=field(report.compute??{},'initialization_shared_wall_s')??report.initialization_shared_wall_s;line(shared,'Shared native initialization:',seconds(init));line(shared,'Inherited candidate-bank geometry preparation:',seconds(field(prep,'candidate_bank_wall_s','geometry_wall_s')));if(number(prep.candidate_bank_audit_wall_s)!==null)line(shared,'Inherited bank preparation + audit:',seconds(prep.candidate_bank_audit_wall_s));}
  document.getElementById('computeScope').textContent=txt(report.resource_scope)||'Unknown measurements stay unknown. Shared initialization and inherited bank preparation are separate from per-method execution; do not add overlapping intervals. Device occupancy is not model memory.';
  renderCases(report);renderSources(report);renderContinuation(report?.continuation);
}

export async function loadReport(fetcher=globalThis.fetch){const response=await fetcher('/report.json',{cache:'no-store'});if(!response.ok)throw new Error('Report HTTP '+response.status);const report=await response.json();render(report);return report;}
let loading=false;
async function refresh(){if(loading)return;loading=true;try{await loadReport();}catch(error){document.getElementById('statusCard').className='card status failed';document.getElementById('statusBadge').textContent='Unavailable';document.getElementById('statusTitle').textContent='Could not read comparison evidence';document.getElementById('statusDetail').textContent=String(error.message||error);for(const id of ['metrics','rateChart','paired','computeTable','sharedCompute','caseList','caseDetail','sourceIdentity','scopeList','computeScope'])clear(id);renderContinuation(undefined);document.getElementById('progress').textContent='No current result claim is shown.';}finally{loading=false;}}
if(typeof document!=='undefined'){document.getElementById('refresh')?.addEventListener('click',refresh);refresh();setInterval(refresh,10000);}
