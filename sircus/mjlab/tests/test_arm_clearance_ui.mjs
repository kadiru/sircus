import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../web/arm-clearance.js',import.meta.url),'utf8');
const ui=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const H='a'.repeat(64);
function counts(group){const captures=group.filter(r=>r.capture).length,native=group.filter(r=>r.native_release).length;return{requested:group.length,native_releases:native,captures,clean:group.filter(r=>r.clean).length,rejected:group.length-native,native_misses:native-captures,invalid:0};}
function fixture(){
 const pins={report_sha256:H,plan_sha256:'b'.repeat(64),manifest_sha256:'c'.repeat(64),contract_fingerprint:'d'.repeat(64)},test_positions=[];
 for(const method of ui.METHODS)for(let i=0;i<50;i++){
  const changed=method==='arm_clearance',native=i<(changed?40:42),capture=changed?(i<23||i>=30&&i<33):i<24,contact=native&&!capture&&(changed?i>=25&&i<29:i<33);
  test_positions.push({method,context_id:'clearance-test-'+i,record_id:method+'-'+i,bucket_xy_m:[.2+i*.01,-.3],values:[.6,-.1,.3,28,.15],native_release:native,capture,clean:capture&&i<12,invalid:false,clearance_rejected:changed&&i>=40&&i<42,robot_contact_before_terminal:native?contact:null,robot_contact_post_rebound:native?contact:null,robot_contact_after_terminal:native?false:null,retention_s:capture?1:0,predicted_clearance_min_m:changed?.003:null,first_post_bounce_contact_channels:native?[contact?'robot':'bucket']:null,first_contact_delay_ms:native?180:null});
 }
 const comparisons=ui.METHODS.map(method=>({method,counts:counts(test_positions.filter(r=>r.method===method))}));
 const control={requested:6,native_releases:6,captures:3,clean:3,rejected:0,native_misses:3,invalid:0};
 const totals={};for(const key of Object.keys(control))totals[key]=comparisons.reduce((n,r)=>n+r.counts[key],0)+2*control[key];
 const contact_comparisons=ui.METHODS.map(method=>{const items=test_positions.filter(r=>r.method===method&&r.native_release);const g={method,requested:50,native_releases:items.length,observed_native:items.length,missing_native:0};for(const s of ['before_terminal','post_rebound','after_terminal'])g['robot_contact_trials_'+s]=items.filter(r=>r['robot_contact_'+s]).length;return g;});
 return{version:ui.VERSION,status:'verified',protocol:{substeps:64,test_layouts:50,maximum_fresh_requests:112,control_requests:12,new_train_requests:0,methods:ui.METHODS,candidates:16,maximum_geometry_checks:4},...pins,audit:{version:ui.AUDIT_VERSION,passed:true,full_checks_verified:true,checks_count:120,source_sha256:'e'.repeat(64),auditor_sha256:'f'.repeat(64),...pins},counts:totals,comparisons,test_positions,contact_comparisons,control_gates:{pre:{status:'passed',counts:control},post:{status:'passed',counts:control}},planned_layouts:50,progress_requests:112,rule_description:'Fixed five-head predicted clearance; no post-outcome selection.',scope:'Held endpoints, fixed64, one requested shot per method and new bucket position.'};
}
function dom(){
 const nodes=new Map();let doc;const element=tag=>({tagName:tag,ownerDocument:doc,children:[],attrs:{},hidden:false,textContent:'',value:'',append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;},setAttribute(k,v){this.attrs[k]=v;}});doc={createElement:element};
 const root=element('section');root.querySelector=selector=>{const key=selector.match(/="([^"]+)"/)[1];if(!nodes.has(key))nodes.set(key,element('div'));return nodes.get(key);};const get=name=>root.querySelector(`[data-arm-clearance="${name}"]`),tableRows=name=>get(name).children[0]?.children[1]?.children.map(r=>r.children.map(c=>c.textContent))??[];return{root,get,tableRows};
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));

test('verified results bind the full paired112-request denominator and native proof',()=>{
 assert.equal(ui.clearanceReady(fixture()),true);
 for(const mutate of [r=>r.version='old',r=>r.status='completed',r=>r.protocol.substeps=128,r=>r.protocol.test_layouts=49,r=>r.protocol.new_train_requests=1,r=>r.protocol.methods=['arm_clearance','consistent'],r=>r.audit.passed='true',r=>r.audit.full_checks_verified=false,r=>r.audit.checks_count=0,r=>r.audit.auditor_sha256='bad',r=>r.audit.plan_sha256='f'.repeat(64),r=>r.audit.source_sha256=null,r=>r.control_gates.pre.status='failed',r=>r.control_gates.post.counts.native_misses=2,r=>r.counts.requested=111,r=>r.comparisons[1].counts.captures++,r=>r.test_positions.pop(),r=>r.test_positions[1].record_id=r.test_positions[0].record_id,r=>r.test_positions[1].context_id=r.test_positions[0].context_id,r=>r.test_positions[50].bucket_xy_m[0]=.99,r=>r.test_positions[0].values[1]=NaN,r=>r.test_positions[0].capture='yes',r=>r.test_positions[0].invalid=true,r=>r.test_positions[42].clean=true]){const r=fixture();mutate(r);assert.equal(ui.clearanceReady(r),false);}
});

test('paired gains and losses and all50 including rejects remain visible',()=>{
 const r=fixture(),before=JSON.stringify(r),d=dom();const p=ui.pairedResults(r);assert.deepEqual([p.gain,p.loss,p.both,p.neither],[3,1,23,23]);ui.renderArmClearance(d.root,r);
 assert.equal(d.get('status').attrs['data-state'],'passed');assert.equal(d.get('original').textContent,'24 / 50');assert.equal(d.get('clearance').textContent,'26 / 50');assert.equal(d.get('paired').textContent,'3 / 1');assert.ok(d.get('status').textContent.includes('Paired gains 3, losses 1'));assert.equal(d.tableRows('all-positions').length,50);assert.equal(d.tableRows('comparison')[1][3],'8');assert.equal(d.tableRows('comparison')[1][4],'2');assert.equal(d.get('case-select').children.length,50);assert.ok(d.get('case-status').textContent.includes('clean capture')||d.get('case-status').textContent.includes('capture'));assert.ok(d.get('protocol').textContent.includes('16 candidates'));assert.equal(JSON.stringify(r),before);
 d.get('case-select').value='clearance-test-40';d.get('case-select').onchange();assert.ok(d.get('case-status').textContent.includes('clearance rejected'));ui.renderArmClearance(d.root,r);assert.equal(d.get('case-select').value,'clearance-test-40');
});

test('contact counts need exact native observations and keep missing distinct from zero',()=>{
 const r=fixture(),d=dom();assert.equal(ui.clearanceContactsReady(r),true);ui.renderArmClearance(d.root,r);assert.ok(d.get('contact-status').textContent.includes('9/42'));assert.ok(d.get('contact-status').textContent.includes('4/40'));assert.equal(d.tableRows('case')[0][4],'unknown');
 for(const mutate of [v=>v.contact_comparisons.pop(),v=>v.contact_comparisons[1].method='consistent',v=>v.contact_comparisons[0].observed_native--,v=>v.contact_comparisons[0].robot_contact_trials_post_rebound++,v=>v.test_positions[0].robot_contact_before_terminal='false',v=>v.test_positions[42].robot_contact_before_terminal=false]){const bad=fixture();mutate(bad);assert.equal(ui.clearanceContactsReady(bad),false);ui.renderArmClearance(d.root,bad);assert.ok(d.get('contact-status').textContent.includes('unknown'));assert.ok(d.tableRows('contacts').every(row=>row.slice(1).every(v=>v==='unknown')));}
 for(const s of ['before_terminal','post_rebound','after_terminal'])r.test_positions[0]['robot_contact_'+s]=null;r.contact_comparisons[0].observed_native--;r.contact_comparisons[0].missing_native++;assert.equal(ui.clearanceContactsReady(r),true);ui.renderArmClearance(d.root,r);assert.ok(d.get('contact-status').textContent.includes('1 native observations unknown'));
});

test('pending queue and failed audit cannot expose successes or a winner',()=>{
 for(const status of ['preparing','prepared','pending','queued','running','completed','awaiting_audit','failed','audit_failed']){const r=fixture(),d=dom();r.status=status;r.queue={reason:'Existing GPU workload left untouched.'};ui.renderArmClearance(d.root,r);assert.equal(d.get('original').textContent,'unknown');assert.equal(d.tableRows('comparison').length,0);assert.equal(d.tableRows('all-positions').length,0);assert.ok(d.get('status').textContent.includes('No verified improvement'));assert.equal(d.get('case-select').disabled,true);}
 const d=dom();ui.renderArmClearance(d.root,null);assert.equal(d.root.hidden,true);
});

test('fragment stays responsive accessible and unable to control the simulator',()=>{
 const html=readFileSync(new URL('../web/arm-clearance.html',import.meta.url),'utf8'),css=readFileSync(new URL('../web/arm-clearance.css',import.meta.url),'utf8');
 for(const text of ['id="armClearanceStudy"','<!--ARM_CLEARANCE_STYLE-->','<!--ARM_CLEARANCE_SCRIPT-->','All 50 paired','whole-ball entry','Setup rejections','predicted path','Simulation only'])assert.ok(html.includes(text),text);
 assert.match(css,/\.arm-clearance-panel\{[^}]*min-width:0;[^}]*grid-column:1\/-1;[^}]*width:100%;[^}]*max-width:100%;/);assert.ok(css.includes('overflow-x:auto'));
 for(const value of ['/api/solve','/api/control','/api/predict','activatePolicy','startTraining','innerHTML','</script'])assert.ok(!source.includes(value),value);
});

test('polling marks stale evidence and cancels after disposal without mutating other data',async()=>{
 const d=dom(),report={[ui.KEY]:fixture(),historical:{unchanged:true}},before=JSON.stringify(report),scheduled=[],cancelled=[];let reads=0;
 const stop=ui.startArmClearance(d.root,{fetchReport:async()=>{if(++reads===1)return report;throw Error('offline');},schedule:(fn,ms)=>{scheduled.push({fn,ms});return scheduled.length;},cancel:id=>cancelled.push(id)});await flush();assert.equal(scheduled[0].ms,5000);await scheduled[0].fn();assert.equal(d.get('status').attrs['data-state'],'unknown');assert.ok(d.get('contact-status').textContent.includes('stale'));stop();assert.deepEqual(cancelled,[2]);assert.equal(JSON.stringify(report),before);
 let resolve;const pending=new Promise(r=>resolve=r),later=[];const halt=ui.startArmClearance(dom().root,{fetchReport:()=>pending,schedule:(fn,ms)=>later.push({fn,ms})});halt();resolve(report);await flush();assert.equal(later.length,0);
 const original=globalThis.fetch,calls=[];try{globalThis.fetch=async(...args)=>{calls.push(args);return{ok:true,json:async()=>report};};const end=ui.startArmClearance(dom().root,{schedule:()=>1,cancel:()=>{}});await flush();end();assert.deepEqual(calls,[['/api/study',{cache:'no-store'}]]);}finally{globalThis.fetch=original;}
});
