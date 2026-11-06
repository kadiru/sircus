// Synthetic read-only fixtures; no training, simulator or native audit is run.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../web/policy64.js',import.meta.url),'utf8');
const ui=await import('data:text/javascript;charset=utf-8,'+encodeURIComponent(source));
const stat=(n,c,clean=0)=>({attempts:n,native_releases:n,captures:c,clean,invalid:0,mean_reward:.5});
function fixture(){const artifact={path:'models/selected.npz',sha256:'b'.repeat(64),weight_sha256:'c'.repeat(64),samples:512};return{
 version:ui.VERSION,status:'completed',phase:'completed',scope:'Fixed64 privileged geometry; no flight search at TEST.',contract_fingerprint:'current',protocol:{train_requests:1024,updates:32,validation_layouts:12,test_layouts:50,maximum_fresh_requests:1284,milestones:[0,256,512,1024],substeps:64,noise:false},
 bootstrap_summary:{requests:6144,native_releases:3000,captures:1000,clean:12,layouts:35,successful_layouts:20,teacher_count:35},train_requests:1024,updates:32,
 curves:[0,256,512,1024].map((samples,i)=>({samples,validation:stat(12,[3,5,7,6][i],i),artifact})),selected:artifact,
 test_comparison:['prior','nearest','bc','ppo'].map((method,i)=>({method,counts:stat(50,[10,20,24,25][i],i===3?10:0),samples:method==='ppo'?512:0,artifact_sha256:method==='ppo'?'b'.repeat(64):'a'.repeat(64)})),
 test_positions:Array.from({length:50},(_,i)=>({context_id:`test-${i}`,bucket_xy_m:[.2+(i%10)*.05,-.6+Math.floor(i/10)*.05],method:'ppo',values:[0,0,.38,28,.4],board_bucket_distance_m:.65,capture:i<25,clean:i<10,native_release:true,invalid:null})),
 control_gates:{pre:{status:'passed'},post:{status:'passed'}},gate:{status:'passed',complete:true,substeps:64},audit:{version:ui.AUDIT_VERSION,status:'passed',contract_fingerprint:'current',checks_count:1,full_checks_verified:true,bound_files_verified:true,source_sha256:'d'.repeat(64)},
 trace_examples:[{id:'test-ppo-0',method:'ppo',context_id:'test-0',substeps:64,values:[0,0,.38,28,.4],bucket_xy_m:[.65,0],board_center_m:[.0001,0,.3802],board_normal:[0,0,1],position_m:[[0,0,.8175],[.65,0,.1]],capture:true,clean:false,whole_ball_entry:true,max_retention_s:1,invalid:null,invalid_physics:false,scalar_score_audit_passed:true}]};}
function dom(){const nodes=new Map(),drawCalls=[],draw=new Proxy({fillText:(...v)=>drawCalls.push(v)},{get:(o,k)=>o[k]??(()=>{})});let doc;const element=tag=>({tagName:tag,ownerDocument:doc,children:[],attrs:{},hidden:false,textContent:'',value:'',width:820,height:280,append(...v){this.children.push(...v);},replaceChildren(...v){this.children=v;},setAttribute(k,v){this.attrs[k]=v;},getContext(){return draw;}});doc={createElement:element};const root=element('section');root.querySelector=selector=>{const key=selector.match(/="([^"]+)"/)[1];if(!nodes.has(key))nodes.set(key,element(key==='trace'||key==='curves'||key==='coverage'?'canvas':'div'));return nodes.get(key);};const get=name=>root.querySelector(`[data-policy64="${name}"]`),rows=name=>get(name).children[0]?.children[1]?.children.map(r=>r.children.map(c=>c.textContent))??[];return{root,get,rows,drawCalls};}
const flush=()=>new Promise(r=>setImmediate(r));
test('final comparison requires exact full audit, controls, four 50-shot cohorts and selected artifact',()=>{
 assert.equal(ui.policy64ComparisonReady(fixture()),true);
 for(const corrupt of [r=>r.status='running',r=>r.phase='test',r=>r.audit.version='old',r=>r.audit.bound_files_verified='true',r=>r.audit.contract_fingerprint='other',r=>r.audit.checks_count=0,r=>r.audit.full_checks_verified=false,r=>r.gate.status='pending',r=>r.protocol.substeps=128,r=>delete r.gate.complete,r=>r.control_gates.post.status='failed',r=>r.test_comparison.pop(),r=>r.test_comparison[1].method='prior',r=>r.test_comparison[3].counts.attempts=49,r=>r.test_comparison[3].artifact_sha256='a'.repeat(64),r=>r.selected.sha256='bad',r=>r.test_positions.pop(),r=>r.test_positions[1].context_id='test-0',r=>delete r.test_positions[0].native_release]){const raw=fixture();corrupt(raw);assert.equal(ui.policy64ComparisonReady(raw),false);}
});
test('render separates bootstrap and fresh costs, validation curves, matched test and actual coverage',()=>{
 const raw=fixture(),before=JSON.stringify(raw),d=dom();ui.renderPolicy64(d.root,raw);assert.equal(d.root.hidden,false);
 assert.equal(d.rows('progress')[0][1],'6144');assert.equal(d.rows('progress')[1][1],'1024 / 1024');assert.equal(d.rows('progress')[2][1],'48');assert.equal(d.rows('progress')[3][1],'200');
 assert.equal(d.rows('curve-table').length,4);assert.equal(d.rows('curve-table')[2][3],'7 / 12');assert.equal(d.rows('comparison').length,4);assert.equal(d.rows('comparison')[3][3],'25 / 50');assert.equal(d.rows('comparison')[3][4],'10 / 50');assert.equal(d.rows('comparison')[3][5],'unknown');
 assert.equal(d.rows('coverage-table').length,50);assert.equal(d.rows('coverage-table')[0][3],'clean');assert.equal(d.rows('coverage-table')[15][3],'capture');assert.equal(d.rows('coverage-table')[40][3],'miss');
 assert.ok(d.get('result').textContent.includes('warm-start ablation'));assert.ok(d.get('result').textContent.includes('25 / 50'));assert.ok(d.get('result').textContent.includes('not a 128-substep'));assert.equal(d.get('result').attrs['data-state'],'passed');
 assert.ok(d.get('trace-caption').textContent.includes('2 actual saved samples'));assert.ok(d.get('trace-caption').textContent.includes('configured board XY'));assert.ok(d.get('trace-caption').textContent.includes('observed board center 0.1 / 0.0 / 380.2 mm'));assert.equal(JSON.stringify(raw),before);
});
test('live curves stay provisional, failures are explicit and zero captures are not a software qualification failure',()=>{
 const d=dom(),raw=fixture();raw.status='running';ui.renderPolicy64(d.root,raw);assert.ok(d.get('result').textContent.includes('provisional'));assert.equal(d.get('result').attrs['data-state'],'unknown');
 raw.status='failed';raw.error='Own finite worker stopped';ui.renderPolicy64(d.root,raw);assert.ok(d.get('status').textContent.includes('Own finite worker stopped'));assert.ok(d.get('result').textContent.includes('failed'));assert.equal(d.get('result').attrs['data-state'],'failed');
 raw.status='completed';raw.audit.status='failed';ui.renderPolicy64(d.root,raw);assert.ok(d.get('result').textContent.includes('audit failed'));
 raw.audit.status='passed';raw.test_comparison[3].counts.captures=0;raw.test_comparison[3].counts.clean=0;raw.test_positions.forEach(r=>{r.capture=false;r.clean=false;});ui.renderPolicy64(d.root,raw);assert.ok(d.get('result').textContent.includes('0 / 50'));assert.ok(d.get('result').textContent.includes('Completed audited'));assert.equal(d.get('result').attrs['data-state'],'passed');
});
test('coverage distinguishes rejected/invalid/missing observations and does not invent a continuous map',()=>{
 const raw=fixture();Object.assign(raw.test_positions[0],{native_release:false,capture:false,clean:false,invalid:'strict_endpoint_rejected'});raw.test_positions[1].invalid='invalid_physics';delete raw.test_positions[2].native_release;
 assert.deepEqual(ui.policy64Coverage(raw).slice(0,3).map(r=>r.outcome),['rejected','invalid','unknown']);
 const d=dom();ui.renderPolicy64(d.root,raw);assert.ok(d.get('coverage-caption').textContent.includes('1 rejected'));assert.ok(d.get('coverage-caption').textContent.includes('1 unknown'));assert.equal(d.get('result').attrs['data-state'],'unknown');
 raw.curves=[{samples:256,validation:{attempts:0}}];const point=ui.policy64CurveRows(raw)[0];assert.equal(point.captureRate,null);assert.equal(point.cleanRate,null);ui.renderPolicy64(d.root,raw);assert.equal(d.rows('curve-table')[0][3],'unknown');
});
test('trajectory binds supplied bucket and configured action without analytic flight generation',()=>{
 const raw=fixture(),trace=raw.trace_examples[0],view=ui.policy64Trajectory(raw,trace);assert.equal(view.distance,.65);assert.equal(view.boardHeight,.38);assert.equal(view.points[1][0],.65);
 assert.equal(ui.policy64Trajectory(raw,{...trace,distance_m:.7}),null);assert.equal(ui.policy64Trajectory(raw,{...trace,position_m:[[0,0,NaN],[1,0,0]]}),null);assert.equal(ui.policy64Trajectory(raw,{id:'missing'}),null);
 const d=dom();raw.trace_examples=[];ui.renderPolicy64(d.root,raw);assert.ok(d.get('trace-caption').textContent.includes('no analytic trajectory'));assert.equal(d.get('trace-select').disabled,true);
});
test('fragment retains 64-only, privileged geometry, historical128 and unequal-exposure boundaries',()=>{
 const html=readFileSync(new URL('../web/policy64.html',import.meta.url),'utf8'),css=readFileSync(new URL('../web/policy64.css',import.meta.url),'utf8');
 for(const value of ['id="policy64Study"','<!--POLICY64_STYLE-->','<!--POLICY64_SCRIPT-->','historical 128-substep capture failures','warm-start ablation','6,144 historical TRAIN','not an equal-training-budget contest','test inference must not search simulated ball flights','Untested positions remain unknown'])assert.ok(html.includes(value),value);
 assert.ok(css.includes('overflow-x:auto'));for(const forbidden of ['/api/solve','/api/control','/api/predict','startTraining','activateCheckpoint','</script'])assert.ok(!source.includes(forbidden));
});
test('poller GETs every 5 seconds, leaves old report keys untouched and marks stale reads',async()=>{
 const d=dom(),scheduled=[],cancelled=[],report={policy64_study:fixture(),impact650_study:{original:true}},before=JSON.stringify(report);let calls=0;
 const stop=ui.startPolicy64(d.root,{fetchReport:async()=>{if(++calls===1)return report;throw Error('offline');},schedule:(fn,ms)=>{scheduled.push({fn,ms});return scheduled.length;},cancel:id=>cancelled.push(id)});await flush();assert.equal(scheduled[0].ms,5000);const old=d.rows('comparison');await scheduled[0].fn();assert.equal(d.get('result').attrs['data-state'],'unknown');assert.ok(d.get('connection').textContent.includes('offline'));assert.deepEqual(d.rows('comparison'),old);stop();assert.deepEqual(cancelled,[2]);assert.equal(JSON.stringify(report),before);
 let resolve;const later=[],pending=new Promise(r=>resolve=r),e=dom();const dispose=ui.startPolicy64(e.root,{fetchReport:()=>pending,schedule:(fn,ms)=>later.push({fn,ms})});dispose();resolve(report);await flush();assert.equal(later.length,0);assert.equal(e.rows('progress').length,0);
 const prior=globalThis.fetch,requests=[];try{globalThis.fetch=async(...args)=>{requests.push(args);return{ok:true,json:async()=>report};};const end=ui.startPolicy64(dom().root,{schedule:()=>1,cancel:()=>{}});await flush();end();assert.deepEqual(requests[0],['/api/study',{cache:'no-store'}]);}finally{globalThis.fetch=prior;}
});
