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
  return METHOD_KEYS.map(key=>{const c=methodData(report,key).compute;return [LABELS[key],...(!audited?Array(7).fill('pending audit'):[seconds(field(c,'wall_s','native_wall_s')),seconds(field(c,'cpu_s','process_cpu_s')),memory(c.peak_rss_mib),memory(c.cuda_peak_allocated_mib),memory(c.cuda_peak_reserved_mib),seconds(field(c,'geometry_s','preflight_wall_s')),seconds(field(c,'selection_s','selection_wall_s'))])];});
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

export function render(report){
  const view=statusView(report),card=document.getElementById('statusCard');card.className='card status '+view.state;document.getElementById('statusBadge').textContent=view.badge;document.getElementById('statusTitle').textContent=view.title;document.getElementById('statusDetail').textContent=view.detail;
  const completed=METHOD_KEYS.reduce((sum,key)=>sum+(methodData(report,key).requests??0),0);document.getElementById('progress').textContent=`Native requests recorded: ${completed} / 70 · Optimizer updates: 0`;
  renderMetrics(report);renderChart(report);const pair=clear('paired');pair.appendChild(el('p',pairedText(report)));pair.appendChild(table(['Method','Captures','Clean','Rejected','Released'],methodRows(report)));
  clear('computeTable').appendChild(table(['Method','Native wall','Process CPU','Peak RSS','CUDA allocated peak','CUDA reserved peak','Geometry checks','Selection'],computeRows(report)));
  const shared=clear('sharedCompute');if(resultsAudited(report)){const prep=report.preprocessing??report.preparation??{};const init=field(report.compute??{},'initialization_shared_wall_s')??report.initialization_shared_wall_s;line(shared,'Shared native initialization:',seconds(init));line(shared,'Inherited candidate-bank geometry preparation:',seconds(field(prep,'candidate_bank_wall_s','geometry_wall_s')));if(number(prep.candidate_bank_audit_wall_s)!==null)line(shared,'Inherited bank preparation + audit:',seconds(prep.candidate_bank_audit_wall_s));}
  document.getElementById('computeScope').textContent=txt(report.resource_scope)||'Unknown measurements stay unknown. Shared initialization and inherited bank preparation are separate from per-method execution; do not add overlapping intervals. Device occupancy is not model memory.';
  renderCases(report);renderSources(report);
}

export async function loadReport(fetcher=globalThis.fetch){const response=await fetcher('/report.json',{cache:'no-store'});if(!response.ok)throw new Error('Report HTTP '+response.status);const report=await response.json();render(report);return report;}
let loading=false;
async function refresh(){if(loading)return;loading=true;try{await loadReport();}catch(error){document.getElementById('statusCard').className='card status failed';document.getElementById('statusBadge').textContent='Unavailable';document.getElementById('statusTitle').textContent='Could not read comparison evidence';document.getElementById('statusDetail').textContent=String(error.message||error);for(const id of ['metrics','rateChart','paired','computeTable','sharedCompute','caseList','caseDetail','sourceIdentity','scopeList','computeScope'])clear(id);document.getElementById('progress').textContent='No current result claim is shown.';}finally{loading=false;}}
if(typeof document!=='undefined'){document.getElementById('refresh')?.addEventListener('click',refresh);refresh();setInterval(refresh,10000);}
