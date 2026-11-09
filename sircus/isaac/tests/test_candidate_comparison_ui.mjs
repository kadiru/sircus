import fs from 'node:fs';
import assert from 'node:assert/strict';
import test from 'node:test';

const source=fs.readFileSync(new URL('../candidate_comparison_panel.js',import.meta.url),'utf8');
const html=fs.readFileSync(new URL('../candidate_comparison.html',import.meta.url),'utf8');
const panel=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const fixture=()=>({schema:'sircus-candidate-native-comparison-report-v1',status:'passed',audit:{status:'passed',sha256:'a'.repeat(64)},source_binding:{checkpoint_sha256:'b'.repeat(64)},methods:{
  parent:{requests:35,planned_requests:35,captures:13,clean:4,rejected:7,released:28,compute:{wall_s:102.5,setup_wall_s:95,score_wall_s:7.5,cpu_s:97.2,peak_rss_mib:7448.7,sampled_device_occupied_mib:6189.4,cuda_peak_allocated_mib:.3,cuda_peak_reserved_mib:2,selection_s:0}},
  bank:{requests:35,planned_requests:35,captures:16,clean:5,rejected:2,released:33,compute:{wall_s:108,setup_wall_s:100,score_wall_s:8,cpu_s:105,peak_rss_mib:7455,sampled_device_occupied_mib:6200,cuda_peak_allocated_mib:.4,cuda_peak_reserved_mib:2,selection_s:7.265}}},
  paired:{gains:4,losses:1,changed_actions:7},compute:{initialization_shared_wall_s:55},preprocessing:{candidate_bank_wall_s:45.7,candidate_bank_audit_wall_s:117},
  cases:[{cell_id:'train-000',bucket_xy_m:[.2,-.1],action_changed:true,parent:{capture:false,clean:false,rejection:'endpoint_ik_unresolved_Arm14',action:{board_x_m:.4,board_y_m:-.2,height_m:.35,gap_m:.24,tilt_deg:29}},bank:{capture:true,clean:true,rejection:null,action:{board_x_m:.41,board_y_m:-.2,height_m:.35,gap_m:.24,tilt_deg:29}}}],
  scope:['Synthetic test fixture only. No measured performance.']});

class Element{
  constructor(tag){this.tag=tag;this.children=[];this.attributes={};this.textContent='';this.listeners={};}
  appendChild(node){this.children.push(node);return node;}
  replaceChildren(...nodes){this.children=nodes;this.textContent='';}
  setAttribute(key,value){this.attributes[key]=value;}
  addEventListener(key,callback){this.listeners[key]=callback;}
}
function dom(){const nodes=new Map();globalThis.document={createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Element(id));return nodes.get(id);}};return nodes;}
const text=node=>node.textContent+node.children.map(text).join(' ');

test('only complete internally consistent independently audited results are final',()=>{
  assert.equal(panel.resultsAudited(fixture()),true);
  for(const status of ['prepared','queued','running','native_complete','audit_pending','failed']){const report=fixture();report.status=status;assert.equal(panel.resultsAudited(report),false);assert.equal(panel.methodRows(report)[0][1],'pending audit');}
  for(const mutate of [r=>r.audit.status='pending',r=>r.methods.bank.requests=34,r=>r.methods.bank.clean=17,r=>r.methods.parent.released=29,r=>r.schema='legacy']){const report=fixture();mutate(report);assert.equal(panel.resultsAudited(report),false);}
});

test('queued and native-complete states distinguish execution from passed audit',()=>{
  assert.match(panel.statusView({...fixture(),status:'queued',queue:{reason:'A different project owns the GPU.'}}).detail,/different project/);
  assert.match(panel.statusView({...fixture(),status:'native_complete'}).title,/audit pending/);
  assert.match(panel.statusView({...fixture(),audit:{status:'pending'}}).title,/not verified/);
  assert.match(panel.statusView({schema:'unknown'}).title,/unavailable/);
});

test('requested position denominators survive rejection accounting',()=>{
  const rows=panel.methodRows(fixture());assert.match(rows[0][1],/13 \/ 35 · 37.1%/);assert.equal(rows[0][3],'7 / 35');assert.equal(rows[1][4],'33 / 35');
  assert.match(panel.pairedText(fixture()),/gains: 4.*losses: 1.*choices: 7 \/ 35/);
});

test('phase-separated resource values keep missing measurements unknown',()=>{
  const report=fixture();delete report.methods.bank.compute.peak_rss_mib;report.methods.bank.compute.cpu_s=-1;
  const rows=panel.computeRows(report);assert.deepEqual(rows[0],['Original seed 22','102.50 s','95.00 s','7.50 s','97.20 s','0.00 s']);assert.equal(rows[1][4],'unknown');assert.equal(rows[1][5],'7.27 s');
  const memory=panel.memoryRows(report);assert.equal(memory[0][1],'7,448.7 MiB');assert.equal(memory[0][2],'6,189.4 MiB');assert.equal(memory[0][3],'0.3 MiB');assert.equal(memory[1][1],'unknown');
  assert.equal(panel.computeRows({...report,status:'audit_pending'})[0][1],'pending audit');
  assert.equal(panel.memoryRows({...report,status:'audit_pending'})[0][2],'pending audit');
});

test('DOM renders audited results and exact paired settings safely',()=>{
  const nodes=dom();try{const report=fixture();report.cases[0].parent.rejection='<script>never executable</script>';panel.render(report);
    assert.match(nodes.get('statusTitle').textContent,/completed/);assert.match(text(nodes.get('metrics')),/13 \/ 35.*16 \/ 35/);assert.equal(nodes.get('rateChart').children[0].tag,'svg');
    assert.match(text(nodes.get('caseDetail')),/<script>never executable<\/script>/);assert.match(text(nodes.get('caseDetail')),/Captured · clean/);assert.match(text(nodes.get('caseDetail')),/350.00 mm/);
    assert.match(text(nodes.get('sharedCompute')),/55.00 s.*45.70 s.*117.00 s/);assert.match(text(nodes.get('sourceIdentity')),/Audit:\s+passed/);
    assert.match(text(nodes.get('computeTable')),/Setup \+ settling.*Score phase/);assert.match(text(nodes.get('computeTable')),/Sampled device occupancy.*Torch CUDA allocated peak/);assert.match(text(nodes.get('computeTable')),/6,189.4 MiB.*0.3 MiB/);assert.match(text(nodes.get('computeTable')),/omit most simulator memory/);
  }finally{delete globalThis.document;}
});

test('queued render hides previously visible final metrics and cases',()=>{
  const nodes=dom();try{panel.render(fixture());const report=fixture();report.status='queued';report.methods={parent:{requests:0,planned_requests:35},bank:{requests:0,planned_requests:35}};panel.render(report);
    assert.match(nodes.get('statusTitle').textContent,/Waiting/);assert.match(nodes.get('progress').textContent,/0 \/ 70/);assert.doesNotMatch(text(nodes.get('metrics')),/13 \/ 35/);assert.match(text(nodes.get('rateChart')),/Pending/);assert.match(text(nodes.get('caseDetail')),/pending audit/);
  }finally{delete globalThis.document;}
});

test('report fetch is GET-only and compatible page exposes all result sections',async()=>{
  const nodes=dom();try{let called;await panel.loadReport(async(url,options)=>{called={url,options};return {ok:true,json:async()=>fixture()};});assert.equal(called.url,'/report.json');assert.equal(called.options.cache,'no-store');assert.equal(called.options.method,undefined);
    await assert.rejects(panel.loadReport(async()=>({ok:false,status:404})),/Report HTTP 404/);
  }finally{delete globalThis.document;}
  assert.doesNotMatch(source,/innerHTML|WebSocket|XMLHttpRequest|method:\s*['"]POST/);
  for(const id of ['statusCard','metrics','rateChart','paired','computeTable','sharedCompute','caseList','caseDetail','sourceIdentity'])assert.ok(html.includes(`id="${id}"`));
  assert.match(html,/TRAIN positions/);assert.match(html,/No learning or held-out evaluation/);assert.match(html,/127.0.0.1:18797/);
});

// Fabricated values below exercise publication behavior, not measured pilot outcomes.
function continuationFixture(){
  const methods={};
  for(const method of ['parent','candidate_bank','continued']){
    methods[method]={};for(const [split,requested] of [['train',35],['validation',35],['test',70]]){
      methods[method][split]={requested,released:requested-2,rejected:2,captures:12,clean_captures:5};
    }
  }
  return {schema:'sircus-candidate-continuation-published-v1',status:'completed_audited',phase:'audit',
    audit:{status:'passed',all_components_passed:true,synthetic:false},model_promoted:false,
    counters:{train_requests:1450,train_releases:1400,assessment_requests:420,assessment_releases:402},
    comparison:{methods,test_pairs:{continued_minus_candidate_bank:{captured:{gains:3,losses:2},clean:{gains:1,losses:0}}}},
    resources:{phases:{initialization:{wall_s:20,aggregate_cpu_s:30,lifetime_peak_rss_mib:6000,largest_child_lifetime_peak_rss_mib:0,device:null},train:{wall_s:120,aggregate_cpu_s:400,lifetime_peak_rss_mib:6500,largest_child_lifetime_peak_rss_mib:0,device:{sampled_whole_device_occupied_mib:6200,torch_peak_allocated_mib:.3,torch_peak_reserved_mib:2}}}},
    final_checkpoint_sha256:'c'.repeat(64)};
}

test('continuation requires its own passed nonsynthetic audit and exact three-method final grid',()=>{
  assert.equal(panel.continuationAudited(continuationFixture()),true);
  for(const mutate of [c=>c.schema='legacy',c=>c.status='running',c=>c.audit.status='not_run',c=>c.audit.all_components_passed=false,c=>c.audit.synthetic=true,c=>c.model_promoted=true,c=>c.counters.assessment_requests=419,c=>c.counters.assessment_releases=401,c=>c.counters.train_requests=1399,c=>c.counters.train_releases=1300,c=>c.counters.train_requests=1751,c=>c.comparison.methods.continued.test.requested=69,c=>c.comparison.methods.parent.validation.released=34,c=>c.comparison.methods.candidate_bank.train.clean_captures=13,c=>delete c.comparison.methods.continued]){
    const continuation=continuationFixture();mutate(continuation);assert.equal(panel.continuationAudited(continuation),false);assert.equal(panel.continuationRows(continuation)[0][2],'pending audit');
  }
  const requestCap=continuationFixture();requestCap.counters.train_requests=1750;requestCap.counters.train_releases=1350;assert.equal(panel.continuationAudited(requestCap),true);
});

test('native completion and failed zero-activity attempt never inherit historical passed audit',()=>{
  const nodes=dom();try{
    const report=fixture();report.continuation={schema:'sircus-candidate-continuation-published-v1',status:'failed',phase:'initialization',error:'Linux pidfd support required <script>text only</script>',audit:{status:'not_run',synthetic:false},model_promoted:false,progress:{train_requests:0,train_releases:0,assessment_requests:0,assessment_releases:0}};
    panel.render(report);
    assert.match(nodes.get('statusTitle').textContent,/Both frozen methods completed/);
    assert.match(nodes.get('continuationTitle').textContent,/stopped/);assert.match(nodes.get('continuationDetail').textContent,/Linux pidfd.*<script>text only<\/script>/);
    assert.match(text(nodes.get('continuationProgress')),/0 \/ 1,750 maximum.*0 \/ 1,400 maximum.*0 \/ 420 planned/);
    assert.match(text(nodes.get('continuationEvidence')),/not_run.*No automatic model promotion.*No trial videos/);
    assert.doesNotMatch(text(nodes.get('continuationEvidence')),/Saved final checkpoint SHA256/);
    assert.match(text(nodes.get('continuationResults')),/not audited/);assert.match(text(nodes.get('continuationCompute')),/not audited/);assert.match(text(nodes.get('continuationPaired')),/stopped attempt/);
    report.continuation.status='native_complete';panel.render(report);assert.match(nodes.get('continuationTitle').textContent,/audit pending/);assert.match(text(nodes.get('continuationResults')),/pending audit/);
  }finally{delete globalThis.document;}
});

test('running continuation shows charged limits and learning receipts without final results',()=>{
  const nodes=dom();try{
    const continuation=continuationFixture();continuation.status='running';continuation.phase='train';continuation.learning={imitation_steps:240,ppo_updates:2,optimizer_steps:248};
    panel.renderContinuation(continuation);
    assert.match(nodes.get('continuationTitle').textContent,/in progress/);assert.match(text(nodes.get('continuationProgress')),/Native TRAIN.*Imitation steps 240 \/ 240 maximum.*PPO updates 2 \/ 13 maximum/);
    assert.match(text(nodes.get('continuationProgress')),/include reservations and rejections/);assert.match(text(nodes.get('continuationResults')),/pending audit/);assert.doesNotMatch(text(nodes.get('continuationResults')),/12 \/ 70/);
  }finally{delete globalThis.document;}
});

test('audited continuation displays all three split denominators and scoped memory',()=>{
  const nodes=dom();try{
    panel.renderContinuation(continuationFixture());
    assert.match(nodes.get('continuationTitle').textContent,/completed/);assert.equal(panel.continuationRows(continuationFixture()).length,9);
    assert.match(text(nodes.get('continuationResults')),/TRAIN.*Original seed 22.*12 \/ 35.*Frozen bank selector.*Continued seed 22.*Fresh validation.*Fresh TEST.*12 \/ 70/);
    assert.match(text(nodes.get('continuationPaired')),/Capture gains 3, losses 2.*not uncertainty across independent training seeds/);
    const compute=text(nodes.get('continuationCompute'));assert.match(compute,/20.00 s.*30.00 s.*6,000.0 MiB/);assert.match(compute,/120.00 s.*400.00 s.*6,500.0 MiB.*6,200.0 MiB.*0.3 MiB.*2.0 MiB/);assert.match(compute,/unknown/);assert.match(compute,/not isolated phase memory/);
    assert.match(text(nodes.get('continuationEvidence')),/Saved final checkpoint SHA256: c{64}/);
  }finally{delete globalThis.document;}
});

test('removing optional continuation hides its panel and clears old final claims',()=>{
  const nodes=dom();try{
    panel.render({...fixture(),continuation:continuationFixture()});assert.equal(nodes.get('continuation').className,'card');
    panel.render(fixture());assert.equal(nodes.get('continuation').className,'card hidden');assert.equal(nodes.get('continuationNav').className,'hidden');assert.equal(text(nodes.get('continuationResults')),'');assert.equal(text(nodes.get('continuationEvidence')),'');
    assert.match(text(nodes.get('metrics')),/13 \/ 35.*16 \/ 35/);
  }finally{delete globalThis.document;}
  for(const id of ['continuation','continuationNav','continuationStatus','continuationProgress','continuationResults','continuationCompute','continuationEvidence'])assert.ok(html.includes(`id="${id}"`));
});

function trainMonitoringFixture(){
  return {audited:false,totals:{requested:40,terminal:35,unresolved:5,captures:10,clean_captures:4,released:30,rejected:5},
    latest_terminal_prefix:{terminal:25,captures:8,clean_captures:3,released:23,rejected:2,event_seq:12,event_fingerprint:'d'.repeat(64)},
    curve:[{terminal:10,captures:2,clean_captures:1,released:7,rejected:3},{terminal:35,captures:10,clean_captures:4,released:30,rejected:5}]};
}
const descendants=node=>[node,...node.children.flatMap(descendants)];

test('exploratory TRAIN counts and curve use completed requests including rejections',()=>{
  const nodes=dom();try{
    const continuation=continuationFixture();continuation.status='running';continuation.phase='train';continuation.train_monitoring=trainMonitoringFixture();
    panel.renderContinuation(continuation);
    const node=nodes.get('continuationTrainMonitoring'),content=text(node);
    assert.equal(node.className,'case-box');assert.match(content,/Unaudited exploration.*Frozen held-out performance remains gated/);
    assert.match(content,/Charged requests.*Completed TRAIN requests.*Unresolved charged requests/);
    assert.match(content,/40 35 5 10 \/ 35 · 28.6% 4 \/ 35 · 11.4% 30 5/);
    assert.match(content,/including rejections.*Unresolved charged requests remain separate/);
    assert.match(content,/Latest recorded batch outcomes:\s+25 requests · 8 captures · 3 clean · 23 released · 2 rejected/);
    const svg=descendants(node).find(child=>child.tag==='svg');assert.ok(svg);assert.match(svg.attributes['aria-label'],/Unaudited cumulative TRAIN.*including rejections/);
    const curves=svg.children.filter(child=>child.tag==='polyline');assert.equal(curves.length,2);
    for(const curve of curves){const points=curve.attributes.points.split(' ').map(point=>point.split(',').map(Number));assert.equal(points.length,2);assert.equal(points.at(-1)[0],595);assert.ok(points.flat().every(Number.isFinite));}
    assert.match(content,/Completed TRAIN requests \(including rejections\)/);
    assert.match(text(nodes.get('continuationResults')),/pending audit/);assert.doesNotMatch(text(nodes.get('continuationResults')),/12 \/ 70/);
  }finally{delete globalThis.document;}
});

test('missing optional TRAIN monitoring hides the panel and clears stale exploration',()=>{
  const nodes=dom();try{
    const continuation=continuationFixture();continuation.status='running';continuation.train_monitoring=trainMonitoringFixture();
    panel.render({...fixture(),continuation});assert.equal(nodes.get('continuationTrainMonitoring').className,'case-box');
    delete continuation.train_monitoring;panel.render({...fixture(),continuation});
    assert.equal(nodes.get('continuationTrainMonitoring').className,'case-box hidden');assert.equal(text(nodes.get('continuationTrainMonitoring')),'');
    assert.match(text(nodes.get('metrics')),/13 \/ 35.*16 \/ 35/);
    continuation.train_monitoring=trainMonitoringFixture();panel.renderContinuation(continuation);panel.renderContinuation(undefined);
    assert.equal(text(nodes.get('continuationTrainMonitoring')),'');assert.equal(nodes.get('continuationTrainMonitoring').className,'case-box hidden');
  }finally{delete globalThis.document;}
  assert.match(html,/id="continuationTrainMonitoring" class="case-box hidden"/);
});

test('stopped imitation shows committed versus reported receipts and unknown in-flight work',()=>{
  const nodes=dom();try{
    const continuation=continuationFixture();continuation.status='failed';continuation.phase='imitation';
    continuation.learning={imitation_steps:63,ppo_updates:0,optimizer_steps:63,
      committed_update_counters:{imitation_steps:62,ppo_updates:0,optimizer_steps:62},unreceipted_optimizer_work_possible:true};
    continuation.train_monitoring={audited:false,totals:{requested:0,terminal:0,unresolved:0,captures:0,clean_captures:0,released:0,rejected:0},latest_terminal_prefix:null,curve:[]};
    panel.renderContinuation(continuation);
    assert.match(text(nodes.get('continuationProgress')),/Recorded\/committed update receipts:\s+Imitation steps 62.*Latest reported counters:\s+Imitation steps 63/);
    assert.match(text(nodes.get('continuationProgress')),/Additional in-flight optimizer work is unknown\. No final checkpoint is committed in these receipts/);
    assert.doesNotMatch(text(nodes.get('continuationEvidence')),/Saved final checkpoint SHA256/);
    assert.match(nodes.get('continuationTitle').textContent,/stopped/);assert.match(text(nodes.get('continuationResults')),/not audited/);
    const monitor=nodes.get('continuationTrainMonitoring');assert.match(text(monitor),/Stopped attempt.*pilot did not complete/);
    assert.match(text(monitor),/0 \/ 0 · rate unavailable/);assert.match(text(monitor),/No completed TRAIN requests yet/);
    assert.equal(descendants(monitor).filter(child=>child.tag==='svg').length,0);
  }finally{delete globalThis.document;}
});

test('unknown charged counts stay unknown and optional curve can be absent',()=>{
  const nodes=dom();try{
    const continuation=continuationFixture();continuation.status='running';const monitoring=trainMonitoringFixture();
    monitoring.totals.requested=null;monitoring.totals.unresolved=null;delete monitoring.curve;continuation.train_monitoring=monitoring;
    panel.renderContinuation(continuation);const monitor=nodes.get('continuationTrainMonitoring');
    assert.match(text(monitor),/unknown 35 unknown 10 \/ 35/);assert.match(text(monitor),/Cumulative TRAIN curve unavailable/);
    assert.equal(descendants(monitor).filter(child=>child.tag==='svg').length,0);
  }finally{delete globalThis.document;}
});

test('inconsistent TRAIN totals or claims cannot produce exploration metrics',()=>{
  for(const change of [m=>m.audited=true,m=>m.totals.requested=34,m=>m.totals.unresolved=6,m=>m.totals.rejected=6,m=>m.totals.captures=31,m=>m.totals.clean_captures=11,m=>m.totals.terminal=NaN]){
    const continuation=continuationFixture();continuation.train_monitoring=trainMonitoringFixture();change(continuation.train_monitoring);
    assert.equal(panel.continuationTrainMonitoring(continuation),null);
  }
  const continuation=continuationFixture();continuation.train_monitoring=trainMonitoringFixture();
  continuation.train_monitoring.curve.push({...continuation.train_monitoring.curve[1]});
  assert.deepEqual(panel.continuationTrainMonitoring(continuation).curve,[]);
});

test('saved checkpoint identity supports the journal helper without implying an audit',()=>{
  const nodes=dom();try{
    const continuation=continuationFixture();continuation.status='audit_pending';delete continuation.final_checkpoint_sha256;
    continuation.final_checkpoint={status:'saved_audit_pending',sha256:'e'.repeat(64)};
    panel.renderContinuation(continuation);assert.match(text(nodes.get('continuationEvidence')),/Saved final checkpoint SHA256: e{64}/);
    assert.match(text(nodes.get('continuationResults')),/pending audit/);
    continuation.final_checkpoint.status='not_confirmed';panel.renderContinuation(continuation);
    assert.doesNotMatch(text(nodes.get('continuationEvidence')),/Saved final checkpoint SHA256/);
  }finally{delete globalThis.document;}
});

function stoppedResourceFixture(){
  const continuation=continuationFixture();continuation.status='incomplete_no_retry';continuation.phase='assessment';
  continuation.stopped_resources={producer_receipt_sha256:'f'.repeat(64),
    phases:{initialization:{wall_s:20,aggregate_cpu_s:30,lifetime_peak_rss_mib:6000,largest_child_lifetime_peak_rss_mib:0,device:null},
      imitation:{wall_s:40,aggregate_cpu_s:45,lifetime_peak_rss_mib:6100,largest_child_lifetime_peak_rss_mib:0},
      train:{wall_s:120,aggregate_cpu_s:400,lifetime_peak_rss_mib:6500,largest_child_lifetime_peak_rss_mib:12,device:{sampled_whole_device_occupied_mib:6200,torch_peak_allocated_mib:.3,torch_peak_reserved_mib:2}},
      assessment:{wall_s:8,aggregate_cpu_s:7,lifetime_peak_rss_mib:6500,largest_child_lifetime_peak_rss_mib:12,device:null}},
    scope:'Non-overlapping producer phase wall and aggregate CPU. RSS is lifetime high-water, not a summed phase peak. Whole-device occupancy includes renderer/PhysX; Torch peaks are not model-only memory. Audit cost was not measured.'};
  return continuation;
}

test('stopped producer costs show four scoped phases without unlocking final results or audited compute',()=>{
  const nodes=dom();try{
    const continuation=stoppedResourceFixture();panel.render({...fixture(),continuation});
    const node=nodes.get('continuationStoppedResources'),content=text(node);
    assert.equal(node.className,'case-box');assert.match(content,/Stopped-run resource measurements/);
    assert.match(content,/Producer receipts for an incomplete attempt; full assessment and audit did not finish; not independently remeasured/);
    assert.match(content,/Initialization.*20.00 s.*30.00 s.*6,000.0 MiB.*0.0 MiB/);
    assert.match(content,/Imitation.*40.00 s.*45.00 s.*Native TRAIN.*120.00 s.*400.00 s/);
    assert.match(content,/6,500.0 MiB.*12.0 MiB.*6,200.0 MiB.*0.3 MiB.*2.0 MiB/);
    assert.match(content,/Partial frozen assessment.*8.00 s.*7.00 s.*unknown/);
    assert.match(content,/not a summed phase peak.*not model-only memory.*Audit cost was not measured/);
    assert.match(content,/Producer receipt SHA256: f{64}/);
    assert.equal(panel.continuationStoppedResources(continuation).rows.length,4);
    assert.equal(panel.continuationAudited(continuation),false);
    assert.equal(panel.continuationRows(continuation).length,9);
    assert.ok(panel.continuationRows(continuation).every(row=>row.slice(2).every(value=>value==='not audited')));
    assert.ok(panel.continuationComputeRows(continuation).every(row=>row.slice(1).every(value=>value==='not audited')));
    assert.match(text(nodes.get('continuationResults')),/not audited/);assert.match(text(nodes.get('continuationCompute')),/not audited/);
    assert.doesNotMatch(text(nodes.get('continuationResults')),/12 \/ 70/);
    assert.match(text(nodes.get('metrics')),/13 \/ 35.*16 \/ 35/);
  }finally{delete globalThis.document;}
  assert.match(html,/id="continuationStoppedResources" class="case-box hidden"/);
});

test('stopped resource measurements reject negative nonfinite and nonnumeric values without inventing zero',()=>{
  const continuation=stoppedResourceFixture(),row=continuation.stopped_resources.phases.train;
  Object.assign(row,{wall_s:-1,aggregate_cpu_s:NaN,lifetime_peak_rss_mib:Infinity,largest_child_lifetime_peak_rss_mib:'12',
    device:{sampled_whole_device_occupied_mib:-2,torch_peak_allocated_mib:true,torch_peak_reserved_mib:null}});
  assert.deepEqual(panel.continuationStoppedResources(continuation).rows[2],['Native TRAIN',...Array(7).fill('unknown')]);
  delete continuation.stopped_resources.phases.imitation;
  assert.deepEqual(panel.continuationStoppedResources(continuation).rows[1],['Imitation',...Array(7).fill('unknown')]);
  continuation.stopped_resources.phases.audit={wall_s:0,aggregate_cpu_s:0};
  const audit=panel.continuationStoppedResources(continuation).rows[4];
  assert.deepEqual(audit,['Independent audit','0.00 s','0.00 s',...Array(5).fill('unknown')]);
});

test('absent running completed or malformed stopped-resource payload hides and clears previous costs',()=>{
  const mutations=[c=>delete c.stopped_resources,c=>c.status='running',c=>c.status='completed_audited',
    c=>c.schema='legacy',c=>c.stopped_resources.producer_receipt_sha256='not a hash',c=>c.stopped_resources.scope=null,
    c=>c.stopped_resources.phases=[],c=>c.stopped_resources.phases={},c=>c.stopped_resources.phases.train=null,
    c=>c.stopped_resources.phases.train.device=[],c=>c.stopped_resources.phases.unrecognized={wall_s:1}];
  const nodes=dom();try{
    for(const mutate of mutations){
      panel.renderContinuation(stoppedResourceFixture());assert.equal(nodes.get('continuationStoppedResources').className,'case-box');
      const continuation=stoppedResourceFixture();mutate(continuation);panel.renderContinuation(continuation);
      assert.equal(panel.continuationStoppedResources(continuation),null);
      assert.equal(nodes.get('continuationStoppedResources').className,'case-box hidden');
      assert.equal(text(nodes.get('continuationStoppedResources')),'');
    }
    panel.renderContinuation(stoppedResourceFixture());panel.renderContinuation(undefined);
    assert.equal(nodes.get('continuationStoppedResources').className,'case-box hidden');assert.equal(text(nodes.get('continuationStoppedResources')),'');
  }finally{delete globalThis.document;}
});
