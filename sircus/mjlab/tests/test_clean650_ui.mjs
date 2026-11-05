import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../web/clean650.js',import.meta.url),'utf8');
const ui=await import('data:text/javascript;charset=utf-8,'+encodeURIComponent(source));
const stat=(n=3,c=3,clean=c)=>({attempts:n,native_releases:n,captures:c,clean,rim_assisted:c-clean,invalid:0,whole_entries:c,max_retention_s:c?1:0});
function fixture(){const geometry={id:'selected',group_id:'selected',phase:'verification',board_xy_m:[0,0],bucket_xy_m:[.65,0],board_height_m:.32,tilt_deg:24,gap_m:.46,release_center_height_m:.8175,distance_m:.65};
 const row={...geometry,by_substeps:{64:stat(),128:stat()},qualification:{qualified:true,clean_both:true,physics_valid:true,oracle_agreement:true},sensitivity:{paired_attempts:3,capture_agreement:3,nominal_position_delta_350ms_max_m:.002}};
 return {version:ui.VERSION,status:'completed',phase:'completed',contract_fingerprint:'current',scope:'No noise, PPO or physical validation.',settings:{max_requests:496,grid:{heights_m:[.22,.26],tilts_deg:[16,20],gaps_m:[.2,.24]}},
  counts:{attempts:18,native_releases:18,captures:15,clean:12,rim_assisted:3,invalid:0,phase_counts:{verification:6,coarse:12}},
  gate:{status:'passed',controls_pre_passed:true,controls_post_passed:true,qualified_settings:1,more_ppo_started:false},
  audit:{version:ui.AUDIT_VERSION,status:'passed',contract_fingerprint:'current',checks_count:1,full_checks_verified:true,bound_files_verified:true,source_sha256:'a'.repeat(64)},
  controls:[{phase:'controls_pre',group_id:'positive',role:'positive_control',substeps:64,statistics:stat()},{phase:'controls_pre',group_id:'negative',role:'negative_control',substeps:128,statistics:stat(3,0)}],
  groups:[row,{id:'coarse',phase:'coarse',by_substeps:{64:stat(12,9,6)}}],top_settings:[{...geometry,phase:'coarse',statistics:stat(1,1),minimum_mouth_clearance_mm:12}],verification:[row],
  trace_examples:[{...geometry,record_id:'native-128',substeps:128,route_success:true,route_quality:2,whole_ball_entry:true,max_retention_s:1,invalid:null,invalid_physics:false,scalar_score_audit_passed:true,position_m:[[0,0,.8175],[.65,0,.1]],selection_scope:'Actual first nominal'}]};}
function dom(){const nodes=new Map(),drawCalls=[],draw=new Proxy({fillText:(...v)=>drawCalls.push(v)},{get:(o,k)=>o[k]??(()=>{})});let doc;
 const element=tag=>({tagName:tag,ownerDocument:doc,children:[],attrs:{},hidden:false,textContent:'',value:'',width:820,height:280,append(...v){this.children.push(...v);},replaceChildren(...v){this.children=v;},setAttribute(k,v){this.attrs[k]=v;},getContext(){return draw;}});doc={createElement:element};const root=element('section');root.querySelector=selector=>{const key=selector.match(/="([^"]+)"/)[1];if(!nodes.has(key))nodes.set(key,element(key==='trace'?'canvas':'div'));return nodes.get(key);};
 const get=name=>root.querySelector(`[data-clean650="${name}"]`),rows=name=>get(name).children[0]?.children[1]?.children.map(r=>r.children.map(c=>c.textContent))??[];return {root,get,rows,drawCalls};}
const flush=()=>new Promise(r=>setImmediate(r));
test('nominal clean qualification requires exact complete audit, controls, observed clean counts and paired drift',()=>{
 let raw=fixture();assert.equal(ui.clean650Qualification(raw,raw.verification[0]).passed,true);
 for(const corrupt of [r=>r.status='running',r=>r.audit.version='old',r=>r.audit.bound_files_verified='true',r=>r.audit.contract_fingerprint='other',r=>r.audit.checks_count=0,r=>r.audit.full_checks_verified=false,
  r=>delete r.gate.controls_post_passed,r=>r.gate.controls_pre_passed=false,r=>r.verification[0].by_substeps[128].clean=2,r=>delete r.verification[0].qualification.physics_valid,r=>r.verification[0].sensitivity.nominal_position_delta_350ms_max_m=.005001]){
  raw=fixture();corrupt(raw);assert.equal(ui.clean650Qualification(raw,raw.verification[0]).passed,null);}
 raw=fixture();raw.verification[0].qualification={qualified:false,status:'failed'};assert.equal(ui.clean650Qualification(raw,raw.verification[0]).passed,false);
});
test('render distinguishes all-phase/search/controls/verification and missing versus observed counts',()=>{
 const d=dom(),raw=fixture(),before=JSON.stringify(raw);ui.renderClean650(d.root,raw);
 assert.equal(d.root.hidden,false);assert.equal(d.rows('summary')[0][1],'18');assert.equal(d.rows('summary')[4][1],'3');assert.equal(d.rows('controls').length,2);assert.equal(d.rows('controls')[0][2],'64');
 assert.equal(d.rows('top')[0][2],'320.0');assert.equal(d.rows('verification').length,2);assert.equal(d.rows('verification')[1][4],'3 / 3');assert.equal(d.get('result').attrs['data-state'],'passed');assert.ok(d.get('result').textContent.includes('identical nominal repeats'));
 assert.ok(d.get('trace-caption').textContent.includes('invalid none'));assert.ok(d.get('trace-caption').textContent.includes('241')===false);assert.equal(JSON.stringify(raw),before);
 delete raw.counts.rim_assisted;ui.renderClean650(d.root,raw);assert.equal(d.rows('summary')[4][1],'unknown');
 raw.verification[0].qualification={qualified:false,status:'failed'};ui.renderClean650(d.root,raw);assert.ok(d.get('result').textContent.includes('No verified setting'));
 raw.gate.status='blocked_controls';ui.renderClean650(d.root,raw);assert.ok(d.get('result').textContent.includes('controls did not pass'));
 raw.status='running';ui.renderClean650(d.root,raw);assert.equal(d.get('result').attrs['data-state'],'unknown');assert.ok(d.get('result').textContent.includes('awaits'));
 ui.renderClean650(d.root,null);assert.equal(d.root.hidden,true);
});
test('phase totals remain unknown if group counts cannot explain the request ledger',()=>{
 const raw=fixture();assert.deepEqual(ui.clean650PhaseRows(raw)[0],['verification','6','6','6','6','0']);raw.counts.phase_counts.coarse=13;
 assert.deepEqual(ui.clean650PhaseRows(raw)[1],['coarse','13','unknown','unknown','unknown','unknown']);
});
test('trajectory projection binds actual matching group coordinates and never creates a flight',()=>{
 const raw=fixture(),trace=raw.trace_examples[0];assert.equal(ui.clean650Trajectory(raw,trace).points[1][0],.65);
 assert.equal(ui.clean650Trajectory(raw,{...trace,bucket_xy_m:[.75,0]}),null);assert.equal(ui.clean650Trajectory(raw,{...trace,distance_m:.75}),null);assert.equal(ui.clean650Trajectory(raw,{...trace,position_m:[[0,0,NaN],[1,0,0]]}),null);
 const d=dom();delete trace.invalid;ui.renderClean650(d.root,raw);assert.ok(d.get('trace-caption').textContent.includes('invalid unknown'));assert.ok(d.drawCalls.some(v=>v[0]==='Bucket center axis (650.0 mm)'));
});
test('inline fragment and reader are additive read-only assets preserving historical separation',()=>{
 const html=readFileSync(new URL('../web/clean650.html',import.meta.url),'utf8'),css=readFileSync(new URL('../web/clean650.css',import.meta.url),'utf8');
 assert.ok(html.includes('id="clean650Study"'));assert.ok(html.includes('<!--CLEAN650_STYLE-->'));assert.ok(html.includes('<!--CLEAN650_SCRIPT-->'));assert.ok(html.includes('Noise testing and further PPO are deferred'));assert.ok(html.includes('original 3/3 at 64 and 0/3 at 128'));
 for(const forbidden of ['/api/solve','/api/control','startTraining','activateCheckpoint','</script'])assert.ok(!source.includes(forbidden));assert.ok(css.includes('overflow-x:auto'));
});
test('polling is GET only, five seconds, safe on refresh errors and disposable with pending reads',async()=>{
 const d=dom(),scheduled=[],cancelled=[];let calls=0;const report={clean650_study:fixture(),calibrated650_replay:{original:true}},before=JSON.stringify(report);
 const stop=ui.startClean650(d.root,{fetchReport:async()=>{if(++calls===1)return report;throw Error('offline');},schedule:(fn,ms)=>{scheduled.push({fn,ms});return scheduled.length;},cancel:id=>cancelled.push(id)});await flush();assert.equal(scheduled[0].ms,5000);const old=d.rows('verification');await scheduled[0].fn();assert.equal(d.get('result').attrs['data-state'],'unknown');assert.ok(d.get('connection').textContent.includes('offline'));assert.deepEqual(d.rows('verification'),old);stop();assert.deepEqual(cancelled,[2]);assert.equal(JSON.stringify(report),before);
 let resolve;const later=[],pending=new Promise(r=>resolve=r),e=dom();const dispose=ui.startClean650(e.root,{fetchReport:()=>pending,schedule:(fn,ms)=>later.push({fn,ms})});dispose();resolve(report);await flush();assert.equal(later.length,0);assert.equal(e.rows('summary').length,0);
 const previous=globalThis.fetch,requests=[];try{globalThis.fetch=async(...args)=>{requests.push(args);return {ok:true,json:async()=>report};};const end=ui.startClean650(dom().root,{schedule:()=>1,cancel:()=>{}});await flush();end();assert.deepEqual(requests[0],['/api/study',{cache:'no-store'}]);}finally{globalThis.fetch=previous;}
});
