import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../web/clearance-diagnosis.js',import.meta.url),'utf8');
const ui=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const H='a'.repeat(64),J='b'.repeat(64);
function count(rows){const native=rows.filter(r=>r.native_release).length,capture=rows.filter(r=>r.capture).length;return{requested:rows.length,native_releases:native,captures:capture,clean:rows.filter(r=>r.clean).length,rejected:rows.length-native,native_misses:native-capture,invalid:0};}
function fixture(){
 const diagnosis={status:'verified',report_sha256:H,source_audit_sha256:J,audit:{passed:true,full_checks_verified:true,checks_count:5,source_sha256:'c'.repeat(64),report_sha256:H,source_audit_sha256:J},finger_summary:'Conservative predicted bounds compared with observed outcomes.',horizon_summary:'Predicted and actual timing are separate.',finger_cases:[{context_id:'case04',old_lower_bound_m:-.01,corrected_lower_bound_m:.004,observed_first_flight_sample_clearance_m:.017,closest_geom:'Arm10/finger',native_capture:true,native_robot_contact:false,diagnosis:'Corner-distance lower bound is conservative.'}],horizon_cases:[{context_id:'case01',guard_end_min_s:.321,guard_end_max_s:.327,first_contact_delay_s:.335,first_contact_channels:['robot'],robot_contact_delay_s:.335,diagnosis:'Robot contact came after the predicted horizon.'}]};
 const methods=['original','corrected'],rows=[];
 for(const method of methods)for(let i=0;i<6;i++)for(let repeat=0;repeat<3;repeat++){
  const native=method==='original'?(i===1||i===5):i!==0,capture=method==='corrected'&&i>=2&&i<=4;
  rows.push({record_id:`${method}-${i}-${repeat}`,context_id:`case${i}`,repeat,method,bucket_xy_m:[.5+i*.05,-.2],values:[.6,-.1,.3,30,.1],native_release:native,capture,clean:capture&&i===3,invalid:false,clearance_rejected:!native,robot_contact_before_terminal:native?!capture:null,robot_contact_after_terminal:native?false:null});
 }
 const comparisons=methods.map(method=>({method,label:method==='original'?'Frozen clearance gate':'Exact-distance candidate',counts:count(rows.filter(r=>r.method===method))})),control={requested:6,native_releases:6,captures:3,clean:3,rejected:0,native_misses:3,invalid:0},totals={};for(const key of Object.keys(control))totals[key]=comparisons.reduce((n,c)=>n+c.counts[key],0)+2*control[key];
 const pins={report_sha256:H,manifest_sha256:J,plan_sha256:'c'.repeat(64),contract_fingerprint:'d'.repeat(64)};
 const correction={status:'verified',description:'Replace only the geometric distance bound.',protocol:{methods,substeps:64,new_training_requests:0,requests_per_method:18,maximum_requests:48,evaluation_scope:'development_replay'},counts:totals,comparisons,rows,...pins,audit:{passed:true,full_checks_verified:true,checks_count:100,source_sha256:'e'.repeat(64),...pins},control_gates:{pre:{status:'passed',counts:control},post:{status:'passed',counts:control}},progress_requests:48};
 return{version:ui.VERSION,diagnosis,correction,recommendation:'This development replay does not activate the candidate.',scope:'Held endpoints; unchanged fixed64 scoring.'};
}
function dom(){const nodes=new Map();let doc;const element=tag=>({tagName:tag,ownerDocument:doc,children:[],attrs:{},hidden:false,textContent:'',append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;},setAttribute(k,v){this.attrs[k]=v;}});doc={createElement:element};const root=element('section');root.querySelector=selector=>{const key=selector.match(/="([^"]+)"/)[1];if(!nodes.has(key))nodes.set(key,element('div'));return nodes.get(key);};const get=name=>root.querySelector(`[data-clearance-diagnosis="${name}"]`);return{root,get,rows:name=>get(name).children[0]?.children[1]?.children.map(r=>r.children.map(c=>c.textContent))??[]};}

test('diagnosis must be independently bound while unknown observations remain unknown',()=>{
 const raw=fixture(),d=dom();assert.equal(ui.diagnosisReady(raw.diagnosis),true);ui.renderClearanceDiagnosis(d.root,raw);assert.equal(d.rows('fingers')[0][1],'-10.00');assert.equal(d.rows('fingers')[0][3],'17.00');assert.equal(d.rows('fingers')[0][5],'yes');assert.equal(d.rows('fingers')[0][6],'no');assert.equal(d.rows('horizons')[0][1],'321.00 – 327.00');
 for(const change of [r=>r.status='diagnosed',r=>r.audit.passed=false,r=>r.audit.full_checks_verified=false,r=>r.audit.report_sha256=J,r=>r.source_audit_sha256='bad',r=>r.horizon_cases[0].robot_contact_delay_s=NaN,r=>r.finger_cases[0].native_robot_contact='false']){const bad=fixture().diagnosis;change(bad);assert.equal(ui.diagnosisReady(bad),false);}
 raw.diagnosis.finger_cases[0].corrected_lower_bound_m=null;raw.diagnosis.finger_cases[0].native_robot_contact=null;raw.diagnosis.horizon_cases[0].robot_contact_delay_s=null;assert.equal(ui.diagnosisReady(raw.diagnosis),true);ui.renderClearanceDiagnosis(d.root,raw);assert.equal(d.rows('fingers')[0][2],'unknown');assert.equal(d.rows('fingers')[0][6],'unknown');assert.equal(d.rows('horizons')[0][4],'unknown');
});

test('native correction requires exact paired accounting and both controls',()=>{
 assert.equal(ui.correctionReady(fixture().correction),true);
 for(const change of [r=>r.status='completed',r=>r.audit.full_checks_verified=false,r=>r.audit.plan_sha256=H,r=>r.protocol.substeps=128,r=>r.protocol.new_training_requests=1,r=>r.protocol.evaluation_scope='held_out',r=>r.protocol.requests_per_method=17,r=>r.control_gates.post.status='failed',r=>r.control_gates.pre.counts.clean=2,r=>r.counts.requested=47,r=>r.comparisons[0].counts.rejected--,r=>r.rows.pop(),r=>r.rows[1].record_id=r.rows[0].record_id,r=>r.rows[1].repeat=r.rows[0].repeat,r=>r.rows[18].bucket_xy_m[0]=.9,r=>r.rows[0].robot_contact_before_terminal=false,r=>r.rows[18].capture=true,r=>r.rows[0].values[2]=Infinity]){const bad=fixture().correction;change(bad);assert.equal(ui.correctionReady(bad),false);}
});

test('rejected requests remain in denominator and contacts use executed observations',()=>{
 const raw=fixture(),before=JSON.stringify(raw),d=dom();ui.renderClearanceDiagnosis(d.root,raw);
 assert.equal(d.rows('comparison')[0][1],'0/18');assert.equal(d.rows('comparison')[0][3],'0');assert.equal(d.rows('comparison')[0][4],'12');assert.equal(d.rows('comparison')[0][5],'6/6');assert.equal(d.rows('comparison')[1][1],'9/18');assert.equal(d.rows('comparison')[1][5],'6/15');assert.equal(d.rows('trials').length,36);assert.ok(d.get('paired').textContent.includes('gains 9; losses 0'));assert.ok(d.get('paired').textContent.includes('do not measure generalization'));assert.equal(before,JSON.stringify(raw));
 raw.correction.rows.find(r=>r.native_release).robot_contact_before_terminal=null;assert.equal(ui.correctionReady(raw.correction),true);ui.renderClearanceDiagnosis(d.root,raw);assert.equal(d.rows('comparison')[0][5],'5/5');assert.equal(d.rows('comparison')[0][6],'1');
});

test('pending correction does not hide verified diagnosis or publish interim successes',()=>{
 for(const status of ['preparing','queued','running','completed','awaiting_audit','failed','audit_failed']){const raw=fixture(),d=dom();raw.correction.status=status;ui.renderClearanceDiagnosis(d.root,raw);assert.equal(d.rows('fingers').length,1);assert.equal(d.rows('comparison').length,0);assert.equal(d.rows('trials').length,0);assert.ok(d.get('correction-status').textContent.includes('not established'));assert.ok(d.get('recommendation').textContent.includes('existing selector'));}
 const raw=fixture(),d=dom();raw.version='unknown';ui.renderClearanceDiagnosis(d.root,raw);assert.equal(d.rows('fingers').length,0);assert.equal(d.rows('comparison').length,0);ui.renderClearanceDiagnosis(d.root,null);assert.equal(d.root.hidden,true);
});

test('initial exact fragment navigation scrolls once only',()=>{
 const prior=Object.getOwnPropertyDescriptor(globalThis,'location'),loc={hash:'#clearanceDiagnosis'};Object.defineProperty(globalThis,'location',{value:loc,configurable:true});
 try{const d=dom(),calls=[];d.root.scrollIntoView=o=>calls.push(o);ui.renderClearanceDiagnosis(d.root,null);assert.equal(calls.length,0);ui.renderClearanceDiagnosis(d.root,fixture());ui.renderClearanceDiagnosis(d.root,fixture());assert.deepEqual(calls,[{block:'start'}]);const other=dom();other.root.scrollIntoView=o=>calls.push(o);loc.hash='#armClearanceStudy';ui.renderClearanceDiagnosis(other.root,fixture());loc.hash='#clearanceDiagnosis';ui.renderClearanceDiagnosis(other.root,fixture());assert.equal(calls.length,1);}finally{if(prior)Object.defineProperty(globalThis,'location',prior);else delete globalThis.location;}
});

test('UI stays read only, responsive and explicit about scope',()=>{
 const html=readFileSync(new URL('../web/clearance-diagnosis.html',import.meta.url),'utf8'),css=readFileSync(new URL('../web/clearance-diagnosis.css',import.meta.url),'utf8');for(const text of ['id="clearanceDiagnosis"','<!--CLEARANCE_DIAGNOSIS_STYLE-->','<!--CLEARANCE_DIAGNOSIS_SCRIPT-->','now development cases','relative to rebound','No new model training'])assert.ok(html.includes(text),text);assert.ok(css.includes('overflow-x:auto'));assert.ok(css.includes('grid-column:1/-1'));for(const value of ['/api/solve','/api/control','startTraining','activatePolicy','innerHTML','</script'])assert.ok(!source.includes(value),value);
});

test('polling reports stale evidence and stops cleanly',async()=>{
 const d=dom(),scheduled=[],cancelled=[];let reads=0;const stop=ui.startClearanceDiagnosis(d.root,{fetchReport:async()=>{if(++reads===1)return{[ui.KEY]:fixture()};throw Error('offline');},schedule:(fn,ms)=>{scheduled.push({fn,ms});return scheduled.length;},cancel:id=>cancelled.push(id)});await new Promise(r=>setImmediate(r));assert.equal(scheduled[0].ms,5000);await scheduled[0].fn();assert.ok(d.get('status').textContent.includes('stale'));assert.ok(d.get('correction-status').textContent.includes('stale'));stop();assert.deepEqual(cancelled,[2]);
});
