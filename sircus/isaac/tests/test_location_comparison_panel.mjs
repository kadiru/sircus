import fs from 'node:fs';
import assert from 'node:assert/strict';
import test from 'node:test';

const source=fs.readFileSync(new URL('../location_comparison_panel.js',import.meta.url),'utf8');
const html=fs.readFileSync(new URL('../location_comparison.html',import.meta.url),'utf8');
const css=fs.readFileSync(new URL('../location_comparison.css',import.meta.url),'utf8');
const panel=await import('data:text/javascript;base64,'+Buffer.from(source+'\n//# sourceURL=location_comparison_panel.js').toString('base64'));
const metrics=(requests,captures=10,clean=3,rejected=2)=>({requests,captures,clean,rejected,misses:requests-captures-rejected,rejection_reasons:{setup_not_quiet:rejected}});
function fixture(){
  // Fabricated publication data only. None of these counts are measurements.
  const report={schema:panel.SCHEMA,status:'completed_audited',synthetic:false,model_promoted:false,
    audit:{status:'passed',complete:true,synthetic:false,manifest_fingerprint:'1'.repeat(64),comparison_fingerprint:'2'.repeat(64),audit_sha256:'3'.repeat(64),trace_sha256:'4'.repeat(64)},generated_at:'2026-11-17T18:00:00Z',
    train:{},progress:{assessment:{requested:600,completed:600}},train_curves:{},monitor_curves:{},final_tests:{},checkpoints:{},
    costs:{per_model_phase:{},observed_phase_wall_s:{original_usable:200,broader:210,shared:40},native_watchdog_wall_s:460,audit_watchdog_wall_s:15,memory_samples:[],memory_by_model_phase:[]},videos:[]};
  for(const model of panel.MODELS){report.final_tests[model]={test_broad:metrics(70),test_local:metrics(25)};report.checkpoints[model]={sha256:'a'.repeat(64)};}
  for(const model of panel.MODELS.slice(1)){
    report.train[model]=metrics(8960,2560,768,512);report.progress[model]={requested:8960,completed:8960};
    report.train_curves[model]=Array.from({length:256},(_,i)=>({...metrics(35,10+(i%3),3,2),train_requests:(i+1)*35}));
    report.monitor_curves[model]=Array.from({length:5},(_,i)=>({...metrics(35,10+i,3,2),train_requests:i*2240,source_model:i===0?'parent':model}));
    report.costs.per_model_phase[model]={train:{episode_wall_s:100,episode_process_cpu_s:80,proposal_wall_s:4,geometry_wall_s:30},update:{update_wall_s:2}};
    report.costs.memory_samples.push({model,phase:'train',lifetime_peak_rss_mib:6000,sampled_gpu_used_mib:6500});
    report.costs.memory_by_model_phase.push({model,phase:'train',samples:512,lifetime_peak_rss_mib:6000,sampled_gpu_used_mib:6500});
    for(const train_requests_before of [0,2240,4480,6720])report.videos.push({id:`${model}-${train_requests_before}`,model,phase:'train',train_requests_before,active_slots:35,parked_slots:0,label:`${model} TRAIN ${train_requests_before}`,packet_sha256:'b'.repeat(64),frames:120,media:null});
  }
  for(const [phase,active_slots] of [['test_broad',24],['test_local',9]])report.videos.push({id:phase,model:'parent',phase,train_requests_before:0,active_slots,parked_slots:35-active_slots,label:`Original parent ${phase} subset`,packet_sha256:'c'.repeat(64),frames:90,media:null});
  return report;
}

function recoveryFixture(){
  const r=fixture();r.status='data_verified_shutdown_incomplete';
  Object.assign(r.audit,{status:r.status,original_audit_status:'not_passed',native_shutdown_status:'failed',native_clean_shutdown:false,session_id:'f'.repeat(32),recovery_session_id:'e'.repeat(32)});
  for(const key of ['audit_exit_sha256','native_exit_sha256','native_exit_fingerprint','audit_exit_fingerprint','recovery_request_fingerprint','inventory_sha256','retirement_proof_sha256'])r.audit[key]='a'.repeat(64);
  r.costs.native_watchdog_wall_s=1000;
  r.costs.shutdown_failure={status:'failed',original_audit_status:'not_passed',clean_shutdown:false,reason:'independent watchdog wall deadline reached',native_watchdog_wall_s:1000,last_progress_shared_phase_s:100,after_last_progress_wall_s:400,shared_phase_limit_s:3600,scope:'Shared time is stale; tail includes shutdown wait and is not isolated compute.'};
  r.costs.recovery_audit={wall_s:2,cpu_user_s:1,cpu_system_s:.1,peak_rss_bytes:1048576,scope:'wait4_exact_audit_child_including_reaped_descendants'};
  return r;
}

test('separate recovery visibly retains shutdown failure and exposes measured curves/costs',()=>{
  const report=recoveryFixture();assert.equal(panel.resultsAudited(report),true);
  assert.equal(panel.statusView(report).state,'failed');
  const nodes=dom();try{panel.render(report);
    assert.equal(nodes.get('statusTitle').textContent,'Data verified; simulator shutdown failed');
    assert.match(text(nodes.get('evidenceDetails')),/Original final audit: not passed.*Simulator shutdown: failed/);
    assert.match(text(nodes.get('phaseWall')),/Shared phase recorded before shutdown · stale/);
    assert.match(text(nodes.get('phaseWall')),/After last progress update, including shutdown wait 400.00 s/);
    assert.match(text(nodes.get('phaseWall')),/Exact audit child user CPU 1.00 s/);
    assert.match(text(nodes.get('phaseWall')),/Exact audit child lifetime peak RSS 1.0 MiB/);
    assert.match(text(nodes.get('phaseWall')),/Original shared-phase limit 3,600.00 s/);
    assert.match(text(nodes.get('phaseWall')),/separate from training cost/);
    assert.equal(descendants(nodes.get('trainChart')).filter(n=>n.tag==='polyline').length,2);
    assert.match(text(nodes.get('broadResults')),/10 \/ 70/);
    assert.equal(descendants(nodes.get('videoList')).filter(n=>n.tag==='article').length,10);
    assert.match(text(nodes.get('videoList')),/Final-model TEST videos were not scheduled/);
  }finally{delete globalThis.document;}
});

test('malformed recovery identity, failure scope or costs hide all outcomes',()=>{
  for(const mutate of [r=>r.audit.status='passed',r=>r.audit.original_audit_status='passed',r=>r.audit.native_clean_shutdown=true,
      r=>delete r.audit.audit_exit_sha256,r=>delete r.audit.native_exit_sha256,r=>r.audit.inventory_sha256='bad',
      r=>r.audit.recovery_session_id=r.audit.session_id,r=>r.audit.native_shutdown_status='passed',r=>r.model_promoted=true,
      r=>delete r.costs.shutdown_failure,r=>r.costs.shutdown_failure.after_last_progress_wall_s=1001,
      r=>r.costs.shutdown_failure.native_watchdog_wall_s=999,r=>r.costs.shutdown_failure.shared_phase_limit_s=9999]){
    const r=recoveryFixture();mutate(r);assert.equal(panel.resultsAudited(r),false);assert.deepEqual(panel.videoEntries(r),[]);
    assert.deepEqual(panel.curveSeries(r,'train_curves'),[]);assert.equal(panel.statusView(r).badge,'Evidence inconsistent');
  }
});

class Element{constructor(tag){this.tag=tag;this.children=[];this.attributes={};this.listeners={};this.textContent='';this.value='';}appendChild(node){this.children.push(node);return node;}replaceChildren(...nodes){this.children=nodes;this.textContent='';}setAttribute(key,value){this.attributes[key]=value;}addEventListener(key,callback){this.listeners[key]=callback;}}
function dom(){const nodes=new Map();globalThis.document={createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Element(id));return nodes.get(id);}};return nodes;}
const text=node=>node.textContent+node.children.map(text).join(' ');
const descendants=node=>[node,...node.children.flatMap(descendants)];

test('requires complete independent audit and exact equal budgets and both distinct test denominators',()=>{
  assert.equal(panel.resultsAudited(fixture()),true);
  for(const change of [r=>r.schema='legacy',r=>r.synthetic=true,r=>r.audit.complete=false,r=>r.audit.status='pending',r=>r.audit.synthetic=true,r=>delete r.model_promoted,r=>r.model_promoted=true,r=>r.train.broader.requests=8959,r=>r.final_tests.parent.test_broad.requests=69,r=>r.final_tests.broader.test_local.requests=24,r=>r.final_tests.parent.test_broad.clean=11,r=>r.final_tests.parent.test_broad.misses=0]){
    const report=fixture();change(report);assert.equal(panel.resultsAudited(report),false);assert.equal(panel.heldoutRows(report,'test_broad')[0][1],'pending audit');assert.deepEqual(panel.curveSeries(report,'train_curves'),[]);assert.deepEqual(panel.videoEntries(report),[]);
  }
});

test('queued running native-complete incomplete and missing states make no final result claim',()=>{
  for(const state of ['prepared','queued','running','native_complete','awaiting_audit','stopped_no_retry']){const report=fixture();report.status=state;assert.equal(panel.resultsAudited(report),false);assert.notEqual(panel.statusView(report).state,'passed');}
  assert.match(panel.statusView({...fixture(),status:'queued'}).title,/Waiting for the 4090/);
  assert.match(panel.statusView({...fixture(),status:'stopped_no_retry'}).title,/stopped/);
  assert.match(panel.statusView({...fixture(),status:'native_complete'}).badge,/Audit pending/);
  assert.match(panel.statusView(null).title,/unavailable/);
});

test('progress distinguishes charged completed and unresolved requests without invented zeros',()=>{
  const report=fixture();report.status='running';report.progress.original_usable={requested:70,completed:35};
  assert.deepEqual(panel.progressFor(report,'original_usable'),{requested:70,completed:35,unresolved:35,cap:8960});
  for(const row of [{requested:10,completed:11},{requested:8961,completed:20},{requested:null,completed:0},{requested:NaN,completed:0}]){report.progress.original_usable=row;assert.equal(panel.progressFor(report,'original_usable'),null);}
});

test('full curves are numerically finite and use requests including rejected setups',()=>{
  const report=fixture(),train=panel.curveSeries(report,'train_curves'),monitor=panel.curveSeries(report,'monitor_curves','clean');
  assert.equal(train.length,2);assert.equal(train[0].points.length,256);assert.equal(train[0].points.at(-1).x,8960);assert.equal(train[0].points[0].y,10/35);
  assert.equal(monitor[0].points.length,5);assert.deepEqual(monitor[0].points.map(x=>x.x),[0,2240,4480,6720,8960]);
  assert.ok(train[0].points[3].y<train[0].points[2].y,'batch outcomes may decrease');
  for(const change of [r=>r.train_curves.original_usable.pop(),r=>r.train_curves.original_usable[0].train_requests=0,r=>r.train_curves.original_usable[0].captures=NaN,r=>r.train_curves.original_usable[1].requests=70]){const bad=fixture();change(bad);assert.equal(panel.curveSeries(bad,'train_curves').length,1);}
  report.monitor_curves.broader[0].source_model='broader';assert.equal(panel.curveSeries(report,'monitor_curves').length,1);
});

test('broad and nearby results stay separate and preserve all-request denominators',()=>{
  assert.match(panel.heldoutRows(fixture(),'test_broad')[0][1],/10 \/ 70 · 14.3%/);
  assert.match(panel.heldoutRows(fixture(),'test_local')[0][1],/10 \/ 25 · 40.0%/);
  assert.match(panel.heldoutRows(fixture(),'test_broad')[0][3],/2 \/ 70/);
});

test('recorded compute scopes and unknown memory never turn into fabricated zero or sums',()=>{
  const report=fixture();const rows=panel.computeRows(report);
  assert.deepEqual(rows[0],['Continued · 31 original locations','PPO update','2.00 s','unknown','unknown','unknown']);
  assert.deepEqual(rows[1].slice(2),['100.00 s','80.00 s','4.00 s','30.00 s']);
  report.costs.per_model_phase.original_usable.train.episode_process_cpu_s=-1;assert.equal(panel.computeRows(report)[1][3],'unknown');
  report.costs.memory_by_model_phase[0].sampled_gpu_used_mib=null;assert.deepEqual(panel.memoryRows(report)[0].slice(2),['512','6,000.0 MiB','unknown']);
  report.costs.memory_samples=Array.from({length:1078},()=>({model:'parent',phase:'test_broad',lifetime_peak_rss_mib:200,sampled_gpu_used_mib:999}));assert.equal(panel.memoryRows(report).length,2,'grouped maxima, not 1,078 raw sample rows');
});

test('DOM renders distinct models, full curves, held-out scopes and recording limitations',()=>{
  const nodes=dom();try{panel.render(fixture());assert.match(nodes.get('statusTitle').textContent,/complete/);assert.match(text(nodes.get('modelCards')),/Frozen control.*31 original locations.*8,960 \/ 8,960.*8,960 broader locations/);
    assert.match(text(nodes.get('broadResults')),/10 \/ 70/);assert.match(text(nodes.get('localResults')),/10 \/ 25/);
    assert.equal(descendants(nodes.get('trainChart')).filter(n=>n.tag==='polyline').length,2);
    for(const line of descendants(nodes.get('trainChart')).filter(n=>n.tag==='polyline')){const points=line.attributes.points.split(' ');assert.equal(points.length,256);assert.ok(points.every(p=>p.split(',').every(x=>Number.isFinite(Number(x)))));}
    assert.match(text(nodes.get('videoList')),/eight TRAIN.*parent-only.*24 broad.*9 nearby.*Final-model TEST videos were not scheduled/);
    assert.equal(descendants(nodes.get('videoList')).filter(n=>n.tag==='article').length,10);assert.equal(descendants(nodes.get('videoList')).filter(n=>n.tag==='video').length,0);
    assert.match(text(nodes.get('evidenceDetails')),/No model is automatically promoted.*without a shot-settings library/);
    assert.match(text(nodes.get('evidenceDetails')),new RegExp('audit sha256: '+'3'.repeat(64)));
    assert.match(text(nodes.get('phaseWall')),/Entire native process watchdog 460.00 s.*Separate independent audit watchdog 15.00 s/);
    assert.match(text(nodes.get('assessmentProgress')),/600 \/ 600 completed requests/);
    assert.match(text(nodes.get('trainTotals')),/2,560|2560/);
  }finally{delete globalThis.document;}
});

test('waiting or failed reports clear all previously visible outcomes and recordings',()=>{
  const nodes=dom();try{panel.render(fixture());const report=fixture();report.status='running';report.progress.original_usable={requested:70,completed:35};panel.render(report);
    assert.match(text(nodes.get('modelCards')),/35 \/ 8,960.*70 charged · 35 unresolved/);assert.match(text(nodes.get('broadResults')),/pending audit/);assert.doesNotMatch(text(nodes.get('broadResults')),/10 \/ 70/);
    assert.equal(descendants(nodes.get('trainChart')).filter(n=>n.tag==='svg').length,0);assert.equal(descendants(nodes.get('videoList')).filter(n=>n.tag==='article').length,0);assert.match(text(nodes.get('evidenceDetails')),/No complete independently audited/);
    panel.render(null);assert.match(text(nodes.get('modelCards')),/unknown \/ 8,960/);assert.doesNotMatch(text(nodes.get('evidenceDetails')),/checkpoint SHA256/);
  }finally{delete globalThis.document;}
});

test('videos require the exact prescheduled ten cohorts and cannot imply all final models were filmed',()=>{
  assert.equal(panel.videoEntries(fixture()).length,10);
  for(const change of [r=>r.videos.pop(),r=>r.videos[0].model='parent',r=>r.videos[1].train_requests_before=0,r=>r.videos[8].active_slots=35,r=>r.videos[8].model='broader',r=>r.videos[0].frames=0]){const report=fixture();change(report);assert.deepEqual(panel.videoEntries(report),[]);}
});

test('only verified local media plays and chapter buttons seek without autoplay',()=>{
  const report=fixture();report.videos[0].media={url:'/media/original-00000.mp4',sha256:'d'.repeat(64),verified:true,duration_s:4};report.videos[0].chapters=[{start_s:0,label:'TRAIN · before 0 requests'},{start_s:2,label:'Native release'},{start_s:99,label:'bad'}];
  const nodes=dom();try{panel.render(report);const video=descendants(nodes.get('videoList')).find(n=>n.tag==='video');assert.ok(video);assert.equal(video.src,'/media/original-00000.mp4');assert.equal(video.autoplay,undefined);assert.equal(video.preload,'metadata');const button=descendants(nodes.get('videoList')).find(n=>n.tag==='button'&&n.textContent.includes('Native release'));button.listeners.click();assert.equal(video.currentTime,2);assert.doesNotMatch(text(nodes.get('videoList')),/99.0 s/);
  }finally{delete globalThis.document;}
  for(const url of ['https://elsewhere.invalid/v.mp4','javascript:alert(1)','/media/../escape.mp4','//elsewhere/v.mp4','/media/clip.mp4?x=1']){report.videos[0].media.url=url;assert.equal(panel.videoEntries(report)[0].media,null);}
  report.videos[0].media.url='/media/ok.mp4';report.videos[0].media.verified=false;assert.equal(panel.videoEntries(report)[0].media,null);
});

test('report strings remain plain text and endpoint is GET-only',async()=>{
  const nodes=dom();try{const report=fixture();report.videos[0].label='<img src=x onerror=bad>';panel.render(report);assert.match(text(nodes.get('videoList')),/<img src=x onerror=bad>/);let call;await panel.loadReport(async(url,options)=>{call={url,options};return {ok:true,json:async()=>report};});assert.equal(call.url,'/api/location-comparison');assert.deepEqual(call.options,{cache:'no-store'});
    await panel.refresh(async()=>({ok:false,status:404}));assert.match(nodes.get('statusTitle').textContent,/Could not read/);assert.doesNotMatch(text(nodes.get('broadResults')),/10 \/ 70/);
  }finally{delete globalThis.document;}
  assert.doesNotMatch(source,/innerHTML|WebSocket|XMLHttpRequest|method:\s*['"](?:POST|PUT|DELETE)|\.play\(/);
});

test('static page exposes all read-only sections with responsive charts and historical link',()=>{
  for(const id of ['overview','learning','heldout','compute','videos','evidence','modelCards','trainChart','monitorChart','broadResults','localResults','phaseWall','videoList'])assert.ok(html.includes(`id="${id}"`));
  assert.match(html,/31 original locations/);assert.match(html,/70 fresh, unseen/);assert.match(html,/25 nearby positions/);assert.match(html,/127.0.0.1:18797/);assert.match(html,/No training or motion controls/);assert.match(css,/@media\(max-width:840px\)/);
});
