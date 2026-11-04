import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const source=readFileSync(new URL('../web/replay650.js',import.meta.url),'utf8');
const ui=await import('data:text/javascript;charset=utf-8,'+encodeURIComponent(source));
function fixture(){
  const group=(id,role,distance,captures)=>({id,role,distance_m:distance,board_xy_m:[0,0],bucket_xy_m:[distance,0],board_height_m:.38,tilt_deg:28,gap_m:.3868,release_center_height_m:.8043,
    by_substeps:Object.fromEntries(['64','128'].map(s=>[s,{attempts:3,native_releases:3,captures,invalid:0,clean:0,whole_entries:captures,max_retention_s:captures?1.2:0}])),
    sensitivity:{paired_attempts:3,capture_agreement:3,nominal_position_delta_350ms_max_m:.002}});
  return {version:ui.VERSION,status:'completed',phase:'completed',contract_fingerprint:'replay',audit:{version:ui.AUDIT_VERSION,status:'passed',contract_fingerprint:'replay',checks_count:1,full_checks_verified:true,bound_files_verified:true,source_sha256:'a'.repeat(64)},
    settings:{candidate:{board_height_m:.38,tilt_deg:28,gap_m:.3868,release_center_height_m:.8043,distance_m:.65},
      controls:{positive_control:{id:'positive-control',role:'positive_control'},negative_control:{id:'negative-control',role:'negative_control'}},
      gaps_m:[.3768,.3808,.3848,.3868,.3888],substeps:[64,128]},
    counts:{attempts:18,native_releases:18,captures:12,invalid:0,clean:0},
    gate:{status:'passed',positive_control:true,negative_control:true,candidate:true,physics_valid:true,oracle_agreement:true,max_drift_m:.002,window_opened:true},
    groups:[group('positive-control','positive_control',.45,3),group('negative-control','negative_control',.75,0),group('candidate-650','candidate',.65,3)],
    trace_examples:[{record_id:'actual-candidate',group_id:'candidate-650',role:'candidate',distance_m:.65,board_xy_m:[0,0],bucket_xy_m:[.65,0],board_height_m:.38,substeps:64,
      route_success:true,whole_ball_entry:true,max_retention_s:1.2,invalid:'',invalid_physics:false,scalar_score_audit_passed:true,position_m:[[0,0,.8043],[.65,0,.1]],selection_scope:'Declared saved native nominal'}]};
}
function dom(){
  const nodes=new Map(),drawCalls=[],draw=new Proxy({fillText:(...v)=>drawCalls.push(v)},{get:(o,k)=>o[k]??(()=>{})});
  let doc;const element=tag=>({tagName:tag,ownerDocument:doc,children:[],attrs:{},hidden:false,textContent:'',value:'',width:820,height:280,
    append(...v){this.children.push(...v);},replaceChildren(...v){this.children=v;},setAttribute(k,v){this.attrs[k]=v;},getContext(){return draw;}});
  doc={createElement:element};const root=element('section');root.querySelector=selector=>{const key=selector.match(/="([^"]+)"/)[1];if(!nodes.has(key))nodes.set(key,element(key==='trace'?'canvas':'div'));return nodes.get(key);};
  const get=name=>root.querySelector(`[data-replay650="${name}"]`),rows=name=>get(name).children[0]?.children[1]?.children.map(r=>r.children.map(c=>c.textContent))??[];
  return {root,get,rows,drawCalls};
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));

test('initial gate requires matching full audit, exact controls, explicit booleans and drift',()=>{
  let raw=fixture();assert.equal(ui.replay650Gate(raw).qualified,true);
  for(const corrupt of [r=>r.status='running',r=>r.phase='window-128',r=>r.version='historical',r=>r.audit.contract_fingerprint='old',r=>r.audit.full_checks_verified='true',
    r=>delete r.audit.version,r=>r.audit.version='old-audit',r=>r.audit.bound_files_verified=false,r=>r.audit.bound_files_verified='true',
    r=>r.audit.checks_count=0,r=>r.audit.source_sha256='bad',r=>delete r.gate.physics_valid,r=>delete r.gate.window_opened,r=>r.groups[0].by_substeps['64'].captures=2]){
    raw=fixture();corrupt(raw);assert.equal(ui.replay650Gate(raw).qualified,null);
  }
  raw=fixture();raw.gate={...raw.gate,status:'failed',candidate:false,window_opened:false};assert.equal(ui.replay650Gate(raw).qualified,false);
  raw=fixture();raw.gate={...raw.gate,status:'failed',max_drift_m:.005001,window_opened:false};assert.equal(ui.replay650Gate(raw).qualified,false);
  raw=fixture();raw.gate.max_drift_m=.005;assert.equal(ui.replay650Gate(raw).qualified,true);
});
test('actual native projection follows this record/group rather than the candidate or historical bucket',()=>{
  const raw=fixture(),trace={...raw.trace_examples[0],group_id:'positive-control',distance_m:.45,bucket_xy_m:[.45,0],position_m:[[0,0,.8],[.45,0,.1]]};
  const view=ui.replay650Trajectory(raw,trace);assert.equal(view.distance,.45);assert.equal(view.points[1][0],.45);
  assert.equal(ui.replay650Trajectory(raw,{...trace,bucket_xy_m:[.65,0]}),null);
  assert.equal(ui.replay650Trajectory(raw,{...trace,distance_m:.65}),null);
  assert.equal(ui.replay650Trajectory(raw,{...trace,position_m:[[0,0,NaN],[0,0,0]]}),null);
  assert.equal(ui.replay650Trajectory(raw,{position_m:[[0,0,.8],[.45,0,.1]]}),null);
});
test('additive assets use inline placeholders and contain no solver, physics or policy requests',()=>{
  const fragment=readFileSync(new URL('../web/replay650.html',import.meta.url),'utf8'),css=readFileSync(new URL('../web/replay650.css',import.meta.url),'utf8');
  assert.ok(fragment.includes('id="calibrated650Replay"'));assert.ok(fragment.includes('<!--REPLAY650_STYLE-->'));assert.ok(fragment.includes('<!--REPLAY650_SCRIPT-->'));
  assert.ok(fragment.includes('Historical lower-board 0/208 and shallow-candidate 0/4'));assert.ok(fragment.includes('No fresh perturbation test'));
  assert.ok(source.includes("fetch('/api/study'"));for(const forbidden of ['/api/solve','/api/control','startTraining','activateCheckpoint','</script'])assert.ok(!source.includes(forbidden));
  assert.ok(css.includes('min-width:0;max-width:100%'));assert.ok(css.includes('overflow-x:auto'));
});
test('render keeps controls/candidate/window separate and exposes entry/retention/invalid observations',()=>{
  const {root,get,rows,drawCalls}=dom(),raw=fixture(),before=JSON.stringify(raw);ui.renderReplay650(root,raw);
  assert.equal(root.hidden,false);assert.ok(get('candidate').textContent.includes('gap 386.800 mm'));assert.equal(rows('summary')[0][1],'18');
  assert.equal(rows('controls').length,4);assert.equal(rows('candidate-results').length,2);assert.equal(rows('candidate-results')[0][3],'3 / 3');
  assert.equal(get('gate').attrs['data-state'],'passed');assert.ok(get('window-status').textContent.includes('three nominal repeats'));
  assert.ok(drawCalls.some(v=>v[0]==='Bucket center axis (650.0 mm)'));assert.ok(get('trace-caption').textContent.includes('whole entry yes'));
  assert.ok(get('trace-caption').textContent.includes('retention 1.200 s'));assert.ok(get('trace-caption').textContent.includes('invalid physics no'));assert.ok(get('trace-caption').textContent.includes('oracle agreement yes'));
  assert.equal(JSON.stringify(raw),before);
  raw.groups.push({id:'window-gap-3768',role:'window',gap_m:.3768,by_substeps:{'64':{attempts:0,native_releases:0,captures:0,invalid:0,whole_entries:0,clean:0,max_retention_s:0}}});
  raw.status='running';raw.phase='window-64';ui.renderReplay650(root,raw);assert.equal(get('gate').attrs['data-state'],'unknown');assert.equal(rows('window')[0][7],'pending');
  assert.equal(rows('window')[1][3],'unknown');assert.ok(get('window-status').textContent.includes('unverified'));
  raw.status='completed';raw.phase='completed';raw.gate={...raw.gate,status:'failed',candidate:false,window_opened:false};ui.renderReplay650(root,raw);
  assert.equal(get('gate').attrs['data-state'],'failed');assert.ok(get('window-status').textContent.includes('not opened'));
  raw.groups.push(...raw.settings.gaps_m.slice(1).map((gap,i)=>({id:`window-extra-${i}`,role:'window',gap_m:gap,by_substeps:{}})));ui.renderReplay650(root,raw);
  assert.equal(rows('window').length,10);
  assert.ok(rows('window').every(row=>row.slice(2).every(v=>v==='not run (gate closed)')));
  assert.ok(rows('sensitivity').slice(-5).every(row=>row.slice(1).every(v=>v==='not run (gate closed)')));
  assert.notEqual(rows('sensitivity')[0][1],'not run (gate closed)');
  raw.trace_examples[0].invalid=null;ui.renderReplay650(root,raw);assert.ok(get('trace-caption').textContent.includes(' · invalid none ·'));
  delete raw.trace_examples[0].invalid;ui.renderReplay650(root,raw);assert.ok(get('trace-caption').textContent.includes(' · invalid unknown ·'));
  raw.audit.bound_files_verified=false;ui.renderReplay650(root,raw);assert.equal(rows('window')[0][3],'pending');assert.notEqual(rows('sensitivity').at(-1)[1],'not run (gate closed)');
  ui.renderReplay650(root,null);assert.equal(root.hidden,true);
});
test('reader polls only the study snapshot every five seconds and marks unavailable refreshes unverified',async()=>{
  const {root,get,rows}=dom(),raw=fixture(),old={captures:0},report={calibrated650_replay:raw,shallow_evaluation:old},before=JSON.stringify(report),scheduled=[],cancelled=[];let calls=0;
  const stop=ui.startReplay650(root,{fetchReport:async()=>{if(++calls===1)return report;throw new Error('offline');},schedule:(fn,ms)=>{scheduled.push({fn,ms});return scheduled.length;},cancel:id=>cancelled.push(id)});
  await flush();assert.equal(calls,1);assert.equal(scheduled[0].ms,5000);const existing=rows('candidate-results');
  await scheduled[0].fn();assert.equal(calls,2);assert.ok(get('connection').textContent.includes('offline'));assert.equal(get('gate').attrs['data-state'],'unknown');assert.deepEqual(rows('candidate-results'),existing);
  assert.equal(JSON.stringify(report),before);stop();assert.deepEqual(cancelled,[2]);await scheduled[1].fn();assert.equal(calls,2);
});
test('default fetch is GET-only and disposal suppresses in-flight rendering and later polling',async()=>{
  const {root}=dom(),previous=globalThis.fetch,requests=[],scheduled=[];
  try{globalThis.fetch=async(...args)=>{requests.push(args);return {ok:true,json:async()=>({calibrated650_replay:fixture()})};};
    const stop=ui.startReplay650(root,{schedule:(fn,ms)=>{scheduled.push({fn,ms});return 1;},cancel:()=>{}});await flush();stop();
    assert.equal(requests.length,1);assert.equal(requests[0][0],'/api/study');assert.deepEqual(requests[0][1],{cache:'no-store'});
  }finally{globalThis.fetch=previous;}
  let resolve;const d=dom(),pending=new Promise(r=>resolve=r),later=[];
  const stop=ui.startReplay650(d.root,{fetchReport:()=>pending,schedule:(fn,ms)=>later.push({fn,ms})});stop();resolve({calibrated650_replay:fixture()});await flush();
  assert.equal(later.length,0);assert.equal(d.rows('controls').length,0);
});
