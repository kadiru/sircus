// Synthetic presentation fixtures only. No model training, native physics or GPU.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../web/ensemble-flight-diagnosis.js',import.meta.url),'utf8');
const ui=await import('data:text/javascript;charset=utf-8,'+encodeURIComponent(source));
function example(id='test-29'){
 const time_s=[0,.1,.2,.3],position_m=[[0,0,.3],[.15,.01,.35],[.4,.01,.2],[.6,.02,.1]],record_id='record-'+id,record_sha256='d'.repeat(64);
 return{id,context_id:id,record_id,record_sha256,bucket_xy_m:[.65,0],values:[0,0,.3,28,.35],board_bucket_distance_m:.65,bucket_rim_z_m:.183,prediction:{capture_score:.9965,disagreement:.0013,aperture_distance_m:.0084,rebound_velocity_m_s:[1.8,.01,1.7]},observed:{native_release:true,capture:false,whole_entry:false,retention_s:0,rebound_velocity_m_s:[1.6,.02,1.6],aperture_distance_m:.1,mouth_clearance_m:-.04,rim_crossing:{time_s:.25,radial_distance_m:.09},pose_gate:'passed',cutoff_contact_channels:{board:true,rim:false}},diagnosis:'Recorded flight misses; the model score is overconfident, not a success measurement.',trace:{time_s,position_m,record_id,record_sha256,native_cache_sha256:'e'.repeat(64)},forecasts:[{label:'Model-velocity ballistic extrapolation',basis:'observed_rebound_origin_model_velocity_gravity_no_contacts',time_s:[...time_s],position_m:position_m.map(p=>[p[0]+.01,p[1],p[2]+.02])}],references:[{method:'bc',values:[0,0,.31,27,.36],prediction:{capture_score:.95},observed:{native_release:true,capture:true,whole_entry:true,retention_s:1.01}}]};
}
function fixture(){const cases=[example(),example('test-49')];return{version:ui.VERSION,status:'completed',scope:'Post-hoc original64 native observations; no new training.',source:{verified:true,report_sha256:ui.SOURCE_REPORT_SHA256,audit_sha256:ui.SOURCE_AUDIT_SHA256,contract_fingerprint:ui.SOURCE_FINGERPRINT},audit:{version:ui.AUDIT_VERSION,status:'passed',full_checks_verified:true,source_sha256:'f'.repeat(64),checks_count:12},summary:{requested:60,native_releases:49,captures:24,rejected:11,native_misses:25,high_score_threshold:.9,high_score_requests:40,high_score_captures:20,high_score_native_misses:19},calibration_bins:[{low:.9,high:1,requested:40,captures:20,mean_score:.98}],cases,misses:Array.from({length:25},(_,i)=>example('miss-'+i)),findings:[{statement:'Predicted and actual rebound differ',evidence:'Observed forward speed is lower',status:'descriptive'}],next_steps:['Validate aperture ranking before training.']};}
function dom(){const nodes=new Map(),drawCalls=[],ctx=new Proxy({fillText:(...args)=>drawCalls.push(args)},{get:(obj,key)=>obj[key]??(()=>{})});let doc;const element=tag=>({tagName:tag,ownerDocument:doc,children:[],attrs:{},hidden:false,textContent:'',value:'',width:760,height:330,append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;},setAttribute(key,value){this.attrs[key]=value;},getContext(){return ctx;}});doc={createElement:element};const root=element('section');root.querySelector=selector=>{const key=selector.match(/="([^"]+)"/)[1];if(!nodes.has(key))nodes.set(key,element(['flight','lateral'].includes(key)?'canvas':'div'));return nodes.get(key);};const get=name=>root.querySelector(`[data-ensemble-flight="${name}"]`),tableRows=name=>get(name).children[0]?.children[1]?.children.map(row=>row.children.map(cell=>cell.textContent))??[];return{root,get,tableRows,drawCalls};}
const flush=()=>new Promise(resolve=>setImmediate(resolve));

test('verified diagnosis requires original source hashes and a separate strictly passed proof',()=>{
 assert.equal(ui.ensembleFlightReady(fixture()),true);
 for(const mutate of [r=>r.version='old',r=>r.status='pending',r=>r.source.verified='true',r=>r.source.report_sha256='a'.repeat(64),r=>r.source.audit_sha256='b'.repeat(64),r=>r.source.contract_fingerprint='c'.repeat(64),r=>r.audit.status='failed',r=>r.audit.version='old',r=>r.audit.full_checks_verified='true',r=>r.audit.source_sha256='bad',r=>r.audit.checks_count=0,r=>r.summary.captures=25,r=>r.misses.pop(),r=>r.misses[0].observed.capture=true,r=>r.misses[0].context_id=r.misses[1].context_id]){const raw=fixture();mutate(raw);assert.equal(ui.ensembleFlightReady(raw),false);}
});

test('recorded projection requires matched record/cache identities, finite first-flight samples and times',()=>{
 const e=example(),view=ui.ensembleFlightView(e);assert.equal(view.distance,.65);assert.deepEqual(view.actual[2],[.4,.01,.2]);assert.equal(view.forecasts.length,1);
 for(const mutate of [e=>e.values=null,e=>e.bucket_xy_m=[0,0],e=>e.trace.record_id='other',e=>e.trace.record_sha256='x'.repeat(64),e=>e.trace.native_cache_sha256='bad',e=>e.trace.position_m[0][0]=NaN,e=>e.trace.time_s[2]=.1,e=>e.trace.time_s.pop(),e=>e.trace.position_m=Array.from({length:242},()=>[0,0,1])]){const bad=example();mutate(bad);assert.equal(ui.ensembleFlightView(bad),null);}
 const mismatched=example();mismatched.forecasts[0].time_s[1]=.11;assert.equal(ui.ensembleFlightView(mismatched).forecasts.length,0);
 const rotated=example();rotated.bucket_xy_m=[0,.65];const v=ui.ensembleFlightView(rotated);assert.deepEqual(v.actual[2],[.01,-.4,.2]);
});

test('case renderer preserves exact poses, distinguishes measured quantities from predictions and never mutates input',()=>{
 const raw=fixture(),before=JSON.stringify(raw),d=dom();ui.renderEnsembleFlightDiagnosis(d.root,raw);
 assert.equal(d.root.hidden,false);assert.equal(d.get('status').attrs['data-state'],'passed');assert.equal(d.tableRows('summary')[0][4],'25');assert.equal(d.tableRows('calibration')[0][4],'50.0%');assert.equal(d.tableRows('configuration')[0][2],'300.0');assert.equal(d.tableRows('configuration')[0][4],'350.0');
 assert.equal(d.tableRows('prediction')[0][0],'99.7%');assert.equal(d.tableRows('errors')[0][3],'0.2000');assert.equal(d.tableRows('errors')[3][3],'-91.6');assert.equal(d.tableRows('references')[0][3],'capture');assert.equal(d.tableRows('misses').length,25);assert.equal(d.get('next-steps').children[0].textContent,'Validate aperture ranking before training.');
 assert.ok(d.get('case-result').textContent.includes('native flight miss'));assert.ok(d.get('flight-caption').textContent.includes('actual saved first-flight'));assert.ok(d.get('flight-caption').textContent.includes('ballistic extrapolation'));assert.ok(d.get('flight-caption').textContent.includes('not a collision reconstruction'));assert.ok(d.get('crossing').textContent.includes('-40.0 mm'));assert.ok(d.get('crossing').textContent.includes('not proof of whole-ball'));assert.ok(d.get('training-support').textContent.includes('do not prove'));assert.equal(JSON.stringify(raw),before);
 d.get('case-select').value='test-49';d.get('case-select').onchange();assert.ok(d.get('case-result').textContent.startsWith('test-49'));ui.renderEnsembleFlightDiagnosis(d.root,raw);assert.equal(d.get('case-select').value,'test-49');
});

test('missing traces forecasts and outcome measurements stay unknown rather than fabricated',()=>{
 const raw=fixture(),d=dom();delete raw.cases[0].trace;delete raw.cases[0].observed.whole_entry;delete raw.cases[0].observed.rim_crossing;delete raw.cases[0].observed.mouth_clearance_m;ui.renderEnsembleFlightDiagnosis(d.root,raw);
 assert.ok(d.get('flight-caption').textContent.includes('No finite, identity-matched'));assert.ok(d.get('case-result').textContent.includes('whole-ball entry unknown'));assert.ok(d.get('crossing').textContent.includes('none / unavailable'));assert.ok(d.get('crossing').textContent.includes('(necessary only) unknown mm'));
 const noForecast=fixture();noForecast.cases[0].forecasts=[];ui.renderEnsembleFlightDiagnosis(d.root,noForecast);assert.ok(d.get('flight-caption').textContent.includes('none is invented'));
 raw.cases=[];ui.renderEnsembleFlightDiagnosis(d.root,raw);assert.ok(d.get('case-result').textContent.includes('No diagnostic'));assert.equal(d.get('case-select').disabled,true);
 raw.audit.status='failed';ui.renderEnsembleFlightDiagnosis(d.root,raw);assert.equal(d.get('status').attrs['data-state'],'failed');ui.renderEnsembleFlightDiagnosis(d.root,null);assert.equal(d.root.hidden,true);
});

test('HTML and source retain post-hoc scopes and provide read-only accessible plots and tables',()=>{
 const html=readFileSync(new URL('../web/ensemble-flight-diagnosis.html',import.meta.url),'utf8'),css=readFileSync(new URL('../web/ensemble-flight-diagnosis.css',import.meta.url),'utf8');
 for(const phrase of ['id="ensembleFlightDiagnosis"','<!--ENSEMBLE_FLIGHT_DIAGNOSIS_STYLE-->','<!--ENSEMBLE_FLIGHT_DIAGNOSIS_SCRIPT-->','uncalibrated prediction','now diagnostic development data','ballistic extrapolations','not learned dense trajectories','No new native counterfactual','All native ensemble misses','175 mm bucket height remains provisional','Fixed 64 substeps'])assert.ok(html.includes(phrase),phrase);
 assert.ok(css.includes('overflow-x:auto'));for(const bad of ['/api/solve','/api/control','/api/predict','activatePolicy','startTraining','</script'])assert.ok(!source.includes(bad),bad);
});

test('poller only reads snapshots, exposes stale data and stops without mutating other report keys',async()=>{
 const d=dom(),report={ensemble_flight_diagnosis:fixture(),physics_ensemble_study:{preserved:true}},before=JSON.stringify(report),scheduled=[],cancelled=[];let reads=0;
 const stop=ui.startEnsembleFlightDiagnosis(d.root,{fetchReport:async()=>{if(++reads===1)return report;throw Error('offline');},schedule:(fn,ms)=>{scheduled.push({fn,ms});return scheduled.length;},cancel:id=>cancelled.push(id)});await flush();assert.equal(scheduled[0].ms,5000);const previous=d.tableRows('summary');await scheduled[0].fn();assert.equal(d.get('status').attrs['data-state'],'unknown');assert.ok(d.get('connection').textContent.includes('stale'));assert.deepEqual(d.tableRows('summary'),previous);stop();assert.deepEqual(cancelled,[2]);assert.equal(JSON.stringify(report),before);
 let resolve;const later=[],pending=new Promise(r=>resolve=r),other=dom();const halt=ui.startEnsembleFlightDiagnosis(other.root,{fetchReport:()=>pending,schedule:(fn,ms)=>later.push({fn,ms})});halt();resolve(report);await flush();assert.equal(later.length,0);
 const original=globalThis.fetch,calls=[];try{globalThis.fetch=async(...args)=>{calls.push(args);return{ok:true,json:async()=>report};};const end=ui.startEnsembleFlightDiagnosis(dom().root,{schedule:()=>1,cancel:()=>{}});await flush();end();assert.deepEqual(calls,[['/api/study',{cache:'no-store'}]]);}finally{globalThis.fetch=original;}
});
