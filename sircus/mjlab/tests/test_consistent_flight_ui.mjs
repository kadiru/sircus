// Synthetic UI fixtures only; no fitting, native shots, hardware or GPU.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../web/consistent-flight-study.js',import.meta.url),'utf8');
const ui=await import('data:text/javascript;charset=utf-8,'+encodeURIComponent(source));
const h='a'.repeat(64);
function example(id='original-test-29'){return{id,values:[.59,-.34,.2681,31.52,.2616],bucket_xy_m:[.25,-.14],prediction_capture_score:.9965,proxy_aperture_margin_m:-.02,proxy_status:'direct aperture miss',observed_capture:false,observed_native_release:true,note:'Existing recorded shot; not a newly simulated alternative.'};}
function fixture(){return{version:ui.VERSION,status:'completed',scope:'Fixed64 frozen-weight comparison.',protocol:{test_positions:30,maximum_requests:132,substeps:64,methods:[...ui.METHODS],new_training:false},progress:{requested:132,planned_requests:132,native_releases:101},development:{summary:[{metric:'Existing high-score failure',value:'1 original example',scope:'post-hoc development'}],cases:[example(),example('original-test-36')]},comparisons:ui.METHODS.map((method,i)=>({method,requested:30,native_releases:25,captures:15+i,clean:3,rejected:5,native_misses:10-i,invalid:0})),report_sha256:h,audit:{passed:true,report_sha256:h,checks_count:100},queue:{gpu:'RTX 4090',status:'completed'},findings:['No model fitting in this experiment.'],next_steps:['Inspect fresh capture and setup results.'],provenance:{source_report_sha256:'b'.repeat(64),ensemble_checkpoint_sha256:'c'.repeat(64),selector_sha256:'d'.repeat(64),plan_sha256:'e'.repeat(64)}};}
function dom(){const nodes=new Map();let doc;const element=tag=>({tagName:tag,ownerDocument:doc,children:[],attrs:{},hidden:false,textContent:'',value:'',append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;},setAttribute(key,value){this.attrs[key]=value;}});doc={createElement:element};const root=element('section');root.querySelector=selector=>{const key=selector.match(/="([^"]+)"/)[1];if(!nodes.has(key))nodes.set(key,element('div'));return nodes.get(key);};const get=name=>root.querySelector(`[data-consistent-flight="${name}"]`),tableRows=name=>get(name).children[0]?.children[1]?.children.map(row=>row.children.map(cell=>cell.textContent))??[];return{root,get,tableRows};}
const flush=()=>new Promise(resolve=>setImmediate(resolve));

test('fresh verification requires exact bound proof, fixed protocol and coherent complete denominators',()=>{
 assert.equal(ui.consistentFlightReady(fixture()),true);
 for(const mutate of [r=>r.version='old',r=>r.status='running',r=>r.protocol.test_positions=60,r=>r.protocol.maximum_requests=264,r=>r.protocol.substeps=128,r=>r.protocol.new_training='false',r=>r.protocol.methods.reverse(),r=>r.report_sha256='bad',r=>r.audit.passed='true',r=>r.audit.report_sha256='f'.repeat(64),r=>r.audit.checks_count=0,r=>r.comparisons.pop(),r=>r.comparisons[1].method='bc',r=>r.comparisons[0].requested=29,r=>r.comparisons[0].captures=26,r=>r.comparisons[0].clean=16,r=>r.comparisons[0].native_misses=-1,r=>r.comparisons[0].invalid=1]){const raw=fixture();mutate(raw);assert.equal(ui.consistentFlightReady(raw),false);}
 const statusOnly=fixture();statusOnly.status='verified';delete statusOnly.audit;assert.equal(ui.consistentFlightReady(statusOnly),false);
});

test('queued and completed-unverified snapshots never invent a measured result or claim verification',()=>{
 const raw=fixture(),d=dom();raw.status='queued';raw.comparisons=[];delete raw.audit;raw.progress={planned_requests:132};raw.queue={status:'waiting',gpu:'RTX 4090',reason:'Foreign workload remains active.'};ui.renderConsistentFlightStudy(d.root,raw);
 assert.equal(d.root.hidden,false);assert.equal(d.get('status').attrs['data-state'],'unknown');assert.ok(d.get('status').textContent.includes('queued'));assert.ok(d.get('queue').textContent.includes('Foreign workload'));assert.equal(d.get('progress').textContent,'unknown / 132');assert.equal(d.get('training').textContent,'None');assert.equal(d.tableRows('comparison').length,4);assert.ok(d.tableRows('comparison').every(r=>r.slice(1).every(v=>v==='unknown')));
 raw.status='completed';ui.renderConsistentFlightStudy(d.root,raw);assert.ok(d.get('status').textContent.includes('verification is still pending'));assert.ok(d.get('audit').textContent.startsWith('Fresh comparison unverified'));
 raw.status='failed';ui.renderConsistentFlightStudy(d.root,raw);assert.equal(d.get('status').attrs['data-state'],'failed');ui.renderConsistentFlightStudy(d.root,null);assert.equal(d.root.hidden,true);
});

test('fresh measured counts and original development predictions remain distinct, input and selection persist',()=>{
 const raw=fixture(),before=JSON.stringify(raw),d=dom();ui.renderConsistentFlightStudy(d.root,raw);
 assert.equal(d.get('status').attrs['data-state'],'passed');assert.equal(d.get('positions').textContent,'30');assert.equal(d.get('progress').textContent,'132 / 132');assert.deepEqual(d.tableRows('comparison')[0],['Saved imitation','30','15','50.0%','3','5','10','0']);assert.ok(d.get('case-result').textContent.includes('original model score 99.7%'));assert.ok(d.get('case-result').textContent.includes('original measured outcome: flight miss'));assert.equal(d.tableRows('case-details')[2][1],'268.1');assert.equal(d.tableRows('case-details')[5][1],'-20.0');assert.ok(d.get('case-note').textContent.includes('not a new native outcome'));assert.ok(d.get('case-note').textContent.includes('remain unknown'));assert.equal(d.get('findings').children[0].textContent,raw.findings[0]);assert.equal(JSON.stringify(raw),before);
 d.get('case-select').value='original-test-36';d.get('case-select').onchange();assert.ok(d.get('case-result').textContent.startsWith('original-test-36'));ui.renderConsistentFlightStudy(d.root,raw);assert.equal(d.get('case-select').value,'original-test-36');
});

test('missing native labels, measurements and examples stay unknown',()=>{
 const raw=fixture(),d=dom();delete raw.development.cases[0].observed_capture;delete raw.development.cases[0].proxy_aperture_margin_m;ui.renderConsistentFlightStudy(d.root,raw);assert.ok(d.get('case-result').textContent.endsWith('outcome: unknown.'));assert.equal(d.tableRows('case-details')[5][1],'unknown');
 raw.development.cases=[];ui.renderConsistentFlightStudy(d.root,raw);assert.equal(d.get('case-select').disabled,true);assert.ok(d.get('case-result').textContent.includes('No recorded'));assert.equal(d.tableRows('case-details').length,0);
});

test('fragment is responsive, accessible and has no action endpoint or implicit solver activation',()=>{
 const html=readFileSync(new URL('../web/consistent-flight-study.html',import.meta.url),'utf8'),css=readFileSync(new URL('../web/consistent-flight-study.css',import.meta.url),'utf8');
 for(const text of ['id="consistentFlightStudy"','<!--CONSISTENT_FLIGHT_STUDY_STYLE-->','<!--CONSISTENT_FLIGHT_STUDY_SCRIPT-->','now development examples','unknown outcome until it is simulated','rim-assisted capture','one continuous second inside','does not start training','175 mm bucket height remains provisional','worst available member','closest 3D opening distance','ball center crosses the rim','None predicts later retention'])assert.ok(html.includes(text),text);
 assert.match(css,/\.consistent-flight-panel\{[^}]*min-width:0;[^}]*grid-column:1\/-1;[^}]*width:100%;[^}]*max-width:100%;[^}]*box-sizing:border-box;[^}]*overflow-wrap:anywhere;/);assert.match(css,/\.consistent-flight-scroll\{[^}]*min-width:0;[^}]*max-width:100%;[^}]*overflow-x:auto/);assert.ok(html.includes('aria-label="Choose a consistent flight development example"'));
 for(const text of ['/api/solve','/api/control','/api/predict','startTraining','activatePolicy','</script','innerHTML'])assert.ok(!source.includes(text),text);
});

test('read-only polling labels stale results and stops cleanly',async()=>{
 const d=dom(),report={consistent_flight_study:fixture(),oldStudy:{untouched:true}},before=JSON.stringify(report),scheduled=[],cancelled=[];let calls=0;
 const stop=ui.startConsistentFlightStudy(d.root,{fetchReport:async()=>{if(++calls===1)return report;throw Error('offline');},schedule:(fn,ms)=>{scheduled.push({fn,ms});return scheduled.length;},cancel:id=>cancelled.push(id)});await flush();assert.equal(scheduled[0].ms,5000);const previous=d.tableRows('comparison');await scheduled[0].fn();assert.equal(d.get('status').attrs['data-state'],'unknown');assert.ok(d.get('connection').textContent.includes('stale'));assert.deepEqual(d.tableRows('comparison'),previous);stop();assert.deepEqual(cancelled,[2]);assert.equal(JSON.stringify(report),before);
 let resolve;const pending=new Promise(r=>resolve=r),later=[];const halt=ui.startConsistentFlightStudy(dom().root,{fetchReport:()=>pending,schedule:(fn,ms)=>later.push({fn,ms})});halt();resolve(report);await flush();assert.equal(later.length,0);
 const original=globalThis.fetch,reads=[];try{globalThis.fetch=async(...args)=>{reads.push(args);return{ok:true,json:async()=>report};};const end=ui.startConsistentFlightStudy(dom().root,{schedule:()=>1,cancel:()=>{}});await flush();end();assert.deepEqual(reads,[['/api/study',{cache:'no-store'}]]);}finally{globalThis.fetch=original;}
});
