// Synthetic CPU-only interface fixtures; these are not native shot results.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../web/tilt650.js',import.meta.url),'utf8');
const ui=await import('data:text/javascript;charset=utf-8,'+encodeURIComponent(source));
const stat=(n=3,c=3,clean=0)=>({attempts:n,native_releases:n,captures:c,clean,rim_assisted:c-clean,invalid:0,whole_entries:c,max_retention_s:c?1:0});
function fixture(){
 const geometry={id:'tilt30-00',group_id:'tilt30-00',source_id:'coarse-64-xy-grid-0043-r0',phase:'verification',translation_m:[-.01,.06],board_xy_m:[-.01,.06],bucket_xy_m:[.64,.06],board_height_m:.38,tilt_deg:30,gap_m:.41,release_center_height_m:.8275,distance_m:.65,parent_route_quality:1};
 const row={...geometry,by_substeps:{64:stat(),128:stat()},qualification:{status:'capture_only',qualified:false,clean_qualified:false,capture_qualified:true,clean_both:false,capture_both:true,parent_source_capture:true,physics_valid:true,oracle_agreement:true},sensitivity:{paired_attempts:3,capture_agreement:3,nominal_position_delta_350ms_max_m:.002}};
 return {version:ui.VERSION,status:'completed',phase:'completed',contract_fingerprint:'current',scope:'No noise, PPO or physical validation.',settings:{source_count:1,source_ids:[geometry.source_id],repeats_per_resolution:3,substeps:[64,128],control_requests:24,verification_requests:30,max_requests:54},
  counts:{attempts:12,native_releases:12,captures:9,clean:3,rim_assisted:6,invalid:0,phase_counts:{verification:6,controls_pre:6}},
  gate:{status:'capture_only',controls_pre_passed:true,controls_post_passed:true,qualified_settings:0,capture_qualified_settings:1,more_ppo_started:false},
  audit:{version:ui.AUDIT_VERSION,status:'passed',contract_fingerprint:'current',checks_count:1,full_checks_verified:true,bound_files_verified:true,source_sha256:'a'.repeat(64)},
  controls:[{phase:'controls_pre',group_id:'positive',role:'positive_control',substeps:64,statistics:stat(3,3,3)},{phase:'controls_pre',group_id:'negative',role:'negative_control',substeps:128,statistics:stat(3,0,0)}],
  groups:[row],top_settings:[],verification:[row],trace_examples:[{...geometry,record_id:'native-128',substeps:128,route_success:true,route_quality:1,whole_ball_entry:true,max_retention_s:1,invalid:null,invalid_physics:false,scalar_score_audit_passed:true,position_m:[[-.01,.06,.8275],[.64,.06,.1]],selection_scope:'Actual first nominal'}]};
}
function dom(){const nodes=new Map(),drawCalls=[],draw=new Proxy({fillText:(...v)=>drawCalls.push(v)},{get:(o,k)=>o[k]??(()=>{})});let doc;
 const element=tag=>({tagName:tag,ownerDocument:doc,children:[],attrs:{},hidden:false,textContent:'',value:'',width:820,height:280,append(...v){this.children.push(...v);},replaceChildren(...v){this.children=v;},setAttribute(k,v){this.attrs[k]=v;},getContext(){return draw;}});doc={createElement:element};const root=element('section');root.querySelector=selector=>{const key=selector.match(/="([^"]+)"/)[1];if(!nodes.has(key))nodes.set(key,element(key==='trace'?'canvas':'div'));return nodes.get(key);};
 const get=name=>root.querySelector(`[data-tilt650="${name}"]`),rows=name=>get(name).children[0]?.children[1]?.children.map(r=>r.children.map(c=>c.textContent))??[];return {root,get,rows,drawCalls};}
const flush=()=>new Promise(r=>setImmediate(r));
test('capture-only qualification stays distinct from clean, independent of historical parent rim classification',()=>{
 let raw=fixture(),g=raw.verification[0];assert.deepEqual(ui.tilt650Qualification(raw,g),{capture:true,clean:false,label:'Capture gate passed; clean failed'});
 g.qualification={...g.qualification,qualified:true,clean_qualified:true,clean_both:true,status:'qualified'};g.by_substeps={64:stat(3,3,3),128:stat(3,3,3)};
 assert.equal(ui.tilt650Qualification(raw,g).clean,true);assert.equal(g.parent_route_quality,1);
 for(const corrupt of [r=>r.status='running',r=>r.phase='verify128',r=>r.audit.version='sircus-xy650-independent-audit-v2',r=>r.audit.bound_files_verified='true',r=>r.audit.contract_fingerprint='other',r=>r.audit.checks_count=0,r=>r.audit.full_checks_verified=false,r=>delete r.gate.controls_post_passed,r=>r.gate.controls_pre_passed=false,r=>r.verification[0].by_substeps[128].captures=2,r=>delete r.verification[0].qualification.physics_valid,r=>delete r.verification[0].qualification.parent_source_capture,r=>r.verification[0].sensitivity.nominal_position_delta_350ms_max_m=.005001]){
  raw=fixture();corrupt(raw);assert.equal(ui.tilt650Qualification(raw,raw.verification[0]).capture,null);
 }
 raw=fixture();raw.verification[0].qualification={status:'failed',capture_qualified:false,clean_qualified:false};assert.equal(ui.tilt650Qualification(raw,raw.verification[0]).capture,false);
});
test('render keeps all-phase control totals separate and shows fixed setting geometry and two gates',()=>{
 const d=dom(),raw=fixture(),before=JSON.stringify(raw);ui.renderTilt650(d.root,raw);
 assert.equal(d.root.hidden,false);assert.equal(d.rows('summary')[0][1],'12');assert.equal(d.rows('summary')[3][1],'3');assert.equal(d.rows('summary')[4][1],'6');assert.equal(d.rows('controls').length,2);
 assert.equal(d.rows('settings')[0][1],'coarse-64-xy-grid-0043-r0');assert.equal(d.rows('settings')[0][2],'-10.0 / 60.0');assert.equal(d.rows('settings')[0][4],'380.0');assert.equal(d.rows('settings')[0][5],'30.00');assert.equal(d.rows('settings')[0][6],'410.0');assert.equal(d.rows('settings')[0][9],'rim-assisted capture');
 assert.equal(d.rows('verification').length,2);assert.equal(d.rows('verification')[1][5],'128');assert.equal(d.rows('verification')[1][7],'0 / 3');assert.equal(d.rows('verification')[1][8],'3 / 3');assert.equal(d.get('result').attrs['data-state'],'capture-only');assert.ok(d.get('result').textContent.includes('1 / 1 settings passed the nominal capture gate; 0 / 1 passed the clean gate'));
 assert.ok(d.get('plan').textContent.includes('Maximum 54'));assert.ok(d.get('result').textContent.includes('not noise robustness'));assert.ok(d.get('trace-caption').textContent.includes('invalid none'));assert.equal(JSON.stringify(raw),before);
 delete raw.counts.rim_assisted;ui.renderTilt650(d.root,raw);assert.equal(d.rows('summary')[4][1],'unknown');
 raw.verification[0].qualification={capture_qualified:false,clean_qualified:false,status:'failed'};ui.renderTilt650(d.root,raw);assert.equal(d.get('result').attrs['data-state'],'failed');
 raw.gate.controls_post_passed=false;ui.renderTilt650(d.root,raw);assert.ok(d.get('result').textContent.includes('controls did not pass'));
 raw.status='running';ui.renderTilt650(d.root,raw);assert.equal(d.get('result').attrs['data-state'],'unknown');assert.ok(d.get('result').textContent.includes('awaits'));
 ui.renderTilt650(d.root,null);assert.equal(d.root.hidden,true);
});
test('missing candidate data and inconsistent claims stay unverified, never turn pending into a failed result',()=>{
 const d=dom(),raw=fixture();delete raw.verification[0].qualification.capture_qualified;ui.renderTilt650(d.root,raw);assert.equal(d.get('result').attrs['data-state'],'unknown');assert.ok(d.get('result').textContent.includes('incomplete'));
 raw.verification[0].qualification.capture_qualified=true;raw.verification[0].by_substeps[128]=stat(3,0,0);ui.renderTilt650(d.root,raw);assert.equal(d.get('result').attrs['data-state'],'unknown');
 assert.deepEqual(ui.tilt650PhaseRows(fixture())[0],['verification','6','6','0','6','0']);raw.counts.phase_counts.verification=7;assert.deepEqual(ui.tilt650PhaseRows(raw)[0],['verification','7','unknown','unknown','unknown','unknown']);
});
test('five fixed candidates produce five geometry rows and ten matched resolution rows',()=>{
 const raw=fixture(),d=dom(),seed=raw.verification[0];raw.settings.source_count=5;
 raw.verification=Array.from({length:5},(_,i)=>({...structuredClone(seed),id:`tilt30-0${i}`,source_id:`parent-${i}`}));
 raw.settings.source_ids=raw.verification.map(g=>g.source_id);ui.renderTilt650(d.root,raw);
 assert.equal(d.rows('settings').length,5);assert.equal(d.rows('verification').length,10);
 assert.ok(d.get('result').textContent.includes('5 / 5 settings passed the nominal capture gate; 0 / 5 passed the clean gate'));
});
test('actual saved flight uses own geometry and rejects mismatched or fabricated positions',()=>{
 const raw=fixture(),trace=raw.trace_examples[0];assert.ok(Math.abs(ui.tilt650Trajectory(raw,trace).points[1][0]-.65)<1e-12);
 assert.equal(ui.tilt650Trajectory(raw,{...trace,bucket_xy_m:[.75,0]}),null);assert.equal(ui.tilt650Trajectory(raw,{...trace,distance_m:.75}),null);assert.equal(ui.tilt650Trajectory(raw,{...trace,position_m:[[0,0,NaN],[1,0,0]]}),null);
 const d=dom();delete trace.invalid;ui.renderTilt650(d.root,raw);assert.ok(d.get('trace-caption').textContent.includes('invalid unknown'));assert.ok(d.drawCalls.some(v=>v[0]==='Bucket center axis (650.0 mm)'));
 trace.whole_ball_entry=false;trace.route_success=false;trace.max_retention_s=0;ui.renderTilt650(d.root,raw);assert.ok(d.get('trace-caption').textContent.includes('whole entry no'));assert.ok(d.get('trace-caption').textContent.includes('max retention 0.000 s'));
});
test('fragment is additive, explicitly bounded, no action routes or original panel mutations',()=>{
 const html=readFileSync(new URL('../web/tilt650.html',import.meta.url),'utf8'),css=readFileSync(new URL('../web/tilt650.css',import.meta.url),'utf8');
 assert.ok(html.includes('id="tilt650Study"'));assert.ok(html.includes('<!--TILT650_STYLE-->'));assert.ok(html.includes('<!--TILT650_SCRIPT-->'));assert.ok(html.includes('54-request diagnostic'));assert.ok(html.includes('stop blind grid expansion'));assert.ok(html.includes('Earlier XY650'));assert.ok(html.includes('rim-assisted captures still count'));assert.ok(css.includes('overflow-x:auto'));
 for(const forbidden of ['/api/solve','/api/control','startTraining','activateCheckpoint','</script'])assert.ok(!source.includes(forbidden));
});
test('GET polling is five seconds, stale on errors, disposable and preserves every other report key',async()=>{
 const d=dom(),scheduled=[],cancelled=[];let calls=0;const report={tilt650_study:fixture(),xy650_study:{original:true},clean650_study:{original:true}},before=JSON.stringify(report);
 const stop=ui.startTilt650(d.root,{fetchReport:async()=>{if(++calls===1)return report;throw Error('offline');},schedule:(fn,ms)=>{scheduled.push({fn,ms});return scheduled.length;},cancel:id=>cancelled.push(id)});await flush();assert.equal(scheduled[0].ms,5000);const old=d.rows('verification');await scheduled[0].fn();assert.equal(d.get('result').attrs['data-state'],'unknown');assert.ok(d.get('connection').textContent.includes('offline'));assert.deepEqual(d.rows('verification'),old);stop();assert.deepEqual(cancelled,[2]);assert.equal(JSON.stringify(report),before);
 let resolve;const later=[],pending=new Promise(r=>resolve=r),e=dom();const dispose=ui.startTilt650(e.root,{fetchReport:()=>pending,schedule:(fn,ms)=>later.push({fn,ms})});dispose();resolve(report);await flush();assert.equal(later.length,0);assert.equal(e.rows('summary').length,0);
 const previous=globalThis.fetch,requests=[];try{globalThis.fetch=async(...args)=>{requests.push(args);return {ok:true,json:async()=>report};};const end=ui.startTilt650(dom().root,{schedule:()=>1,cancel:()=>{}});await flush();end();assert.deepEqual(requests[0],['/api/study',{cache:'no-store'}]);}finally{globalThis.fetch=previous;}
});
