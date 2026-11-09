import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../web/clearance-horizon.js',import.meta.url),'utf8');
const ui=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const H='a'.repeat(64),J='b'.repeat(64);
const FIELDS=['requested','native_releases','captures','clean','rejected','native_misses','invalid'];
function count(rows){const native=rows.filter(r=>r.native_release).length,capture=rows.filter(r=>r.capture).length;return{requested:rows.length,native_releases:native,captures:capture,clean:rows.filter(r=>r.clean).length,rejected:rows.length-native,native_misses:native-capture,invalid:0};}
function fixture(){
 const methods=['short_window','long_window'],rows=[];
 for(const method of methods)for(let repeat=0;repeat<3;repeat++){const native=method==='short_window';rows.push({record_id:`${method}-${repeat}`,context_id:'arm-clearance-test-01',repeat,method,bucket_xy_m:[.5,-.2],values:[.6,-.1,.3,30,.1],native_release:native,capture:false,clean:false,invalid:false,clearance_rejected:!native,robot_contact_before_terminal:native?true:null,robot_contact_after_terminal:native?false:null,table_contact_before_terminal:native?false:null,table_contact_after_terminal:native?true:null,bucket_retained:native?false:null});}
 const comparisons=methods.map(method=>({method,label:method==='short_window'?'Shorter window':'Longer window',counts:count(rows.filter(r=>r.method===method))})),control={requested:6,native_releases:6,captures:3,clean:3,rejected:0,native_misses:3,invalid:0},totals={};for(const key of FIELDS)totals[key]=comparisons.reduce((n,c)=>n+c.counts[key],0)+2*control[key];
 const pins={report_sha256:H,manifest_sha256:J,plan_sha256:'c'.repeat(64),contract_fingerprint:'d'.repeat(64)};
 return{version:ui.VERSION,status:'verified',description:'Change only the predicted end time.',protocol:{methods,substeps:64,new_training_requests:0,requests_per_method:3,maximum_requests:18,evaluation_scope:'development_replay',context_id:'arm-clearance-test-01'},counts:totals,comparisons,rows,...pins,audit:{passed:true,full_checks_verified:true,checks_count:100,source_sha256:'e'.repeat(64),...pins},control_gates:{pre:{status:'passed',counts:control},post:{status:'passed',counts:control}},progress_requests:18,windows:methods.map((method,i)=>({method,passed:i===0,horizon_min_s:i?.42:.32,horizon_max_s:i?.43:.33,minimum_clearance_lower_bound_m:i?-.003:.005,required_clearance_m:.002})),recommendation:'Withholding a shot is not an improved capture. Keep the selector unchanged.'};
}
function dom(){const nodes=new Map();let doc;const element=tag=>({tagName:tag,ownerDocument:doc,children:[],attrs:{},hidden:false,textContent:'',append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;},setAttribute(k,v){this.attrs[k]=v;}});doc={createElement:element};const root=element('section');root.querySelector=selector=>{const key=selector.match(/="([^"]+)"/)[1];if(!nodes.has(key))nodes.set(key,element('div'));return nodes.get(key);};const get=name=>root.querySelector(`[data-clearance-horizon="${name}"]`);return{root,get,rows:name=>get(name).children[0]?.children[1]?.children.map(r=>r.children.map(c=>c.textContent))??[]};}

test('only complete proof with all paired requests and controls authorizes outcomes',()=>{
 assert.equal(ui.horizonReady(fixture()),true);
 for(const change of [r=>r.status='completed',r=>r.version='unknown',r=>r.audit.full_checks_verified=false,r=>r.audit.plan_sha256=H,r=>r.protocol.substeps=128,r=>r.protocol.new_training_requests=1,r=>r.protocol.evaluation_scope='held_out',r=>r.protocol.requests_per_method=2,r=>r.protocol.context_id='case02',r=>r.control_gates.post.status='failed',r=>r.control_gates.pre.counts.clean=2,r=>r.counts.requested=17,r=>r.comparisons[0].counts.rejected++,r=>r.rows.pop(),r=>r.rows[1].record_id=r.rows[0].record_id,r=>r.rows[1].repeat=r.rows[0].repeat,r=>r.rows[3].bucket_xy_m[0]=.9,r=>r.rows[3].values[4]=.2,r=>r.rows[3].robot_contact_before_terminal=false,r=>r.rows[3].capture=true,r=>r.rows[0].values[2]=Infinity,r=>r.windows[1].horizon_max_s=2,r=>r.windows[1].horizon_min_s=-.01,r=>r.windows[1].required_clearance_m=0,r=>r.windows[1].passed='false']){const bad=fixture();change(bad);assert.equal(ui.horizonReady(bad),false);}
});

test('abstention is not zero observed contacts or a new capture; route and retention stay distinct',()=>{
 const raw=fixture(),before=JSON.stringify(raw),d=dom();ui.renderClearanceHorizon(d.root,raw);
 assert.equal(d.rows('comparison')[0][1],'0/3');assert.equal(d.rows('comparison')[0][5],'3/3');assert.equal(d.rows('comparison')[1][4],'3');assert.equal(d.rows('comparison')[1][5],'unknown (no release observations)');assert.equal(d.rows('contacts')[3][2],'unknown (not released)');assert.equal(d.rows('contacts')[3][3],'unknown');assert.equal(d.rows('contacts')[0][5],'no / yes');assert.equal(d.rows('trials').length,6);assert.equal(d.rows('windows')[0][1],'320.00 – 330.00');assert.equal(d.rows('windows')[1][2],'-3.00');assert.ok(d.get('paired').textContent.includes('do not measure generalization'));assert.equal(before,JSON.stringify(raw));
 raw.rows[0].bucket_retained=true;ui.renderClearanceHorizon(d.root,raw);assert.equal(d.rows('contacts')[0][2],'no');assert.equal(d.rows('contacts')[0][3],'yes');assert.equal(d.rows('comparison')[0][1],'0/3');
});

test('pending or failed results never publish interim improvements',()=>{
 for(const status of ['pending','preparing','queued','running','completed','awaiting_audit','failed','audit_failed']){const raw=fixture(),d=dom();raw.status=status;ui.renderClearanceHorizon(d.root,raw);for(const key of ['comparison','trials','contacts','windows','controls'])assert.equal(d.rows(key).length,0);assert.ok(d.get('status').textContent.includes('not established'));assert.ok(d.get('recommendation').textContent.includes('existing selector'));}
 const d=dom();ui.renderClearanceHorizon(d.root,null);assert.equal(d.root.hidden,true);
});

test('initial matching fragment scrolls once only',()=>{
 const prior=Object.getOwnPropertyDescriptor(globalThis,'location'),loc={hash:'#clearanceHorizon'};Object.defineProperty(globalThis,'location',{value:loc,configurable:true});
 try{const d=dom(),calls=[];d.root.scrollIntoView=o=>calls.push(o);ui.renderClearanceHorizon(d.root,null);assert.equal(calls.length,0);ui.renderClearanceHorizon(d.root,fixture());ui.renderClearanceHorizon(d.root,fixture());assert.deepEqual(calls,[{block:'start'}]);const other=dom();other.root.scrollIntoView=o=>calls.push(o);loc.hash='#clearanceDiagnosis';ui.renderClearanceHorizon(other.root,fixture());loc.hash='#clearanceHorizon';ui.renderClearanceHorizon(other.root,fixture());assert.equal(calls.length,1);}finally{if(prior)Object.defineProperty(globalThis,'location',prior);else delete globalThis.location;}
});

test('scope is read-only and explains case, horizon and unchanged scoring',()=>{
 const html=readFileSync(new URL('../web/clearance-horizon.html',import.meta.url),'utf8'),css=readFileSync(new URL('../web/clearance-horizon.css',import.meta.url),'utf8');for(const value of ['id="clearanceHorizon"','<!--CLEARANCE_HORIZON_STYLE-->','<!--CLEARANCE_HORIZON_SCRIPT-->','numbered bucket layout','look for a robot collision','board → table → bucket','not relabeled as strict success','one second'])assert.ok(html.includes(value),value);assert.ok(css.includes('overflow-x:auto'));assert.ok(css.includes('grid-column:1/-1'));for(const value of ['/api/solve','/api/control','startTraining','activatePolicy','innerHTML','</script'])assert.ok(!source.includes(value),value);
});

test('historical table-assisted example stays separately labelled and matches sanitized evidence',()=>{
 const html=readFileSync(new URL('../web/clearance-horizon.html',import.meta.url),'utf8');
 const evidence=JSON.parse(readFileSync(new URL('../../../baselines/mjlab-clearance-horizon-20261108/table-assisted-example.json',import.meta.url),'utf8'));
 assert.equal(evidence.status,'verified_historical_example');assert.equal(evidence.shot.phase,'train');assert.equal(evidence.shot.method,'ppo');assert.equal(evidence.shot.substeps,64);
 for(const value of [evidence.events.board_impact.start_s,evidence.events.table_bounce.start_s,evidence.events.first_bucket_contact_s,evidence.events.whole_ball_entry_s,evidence.events.capture_s])assert.ok(html.includes(value.toFixed(3)+' s'));
 assert.equal(evidence.original_score.success,true);assert.equal(evidence.original_score.route_success,false);assert.equal(evidence.events.robot_floor_or_unattributed_contacts_before_capture,false);
 for(const value of ['Separate historical evidence — not part of the 18-slot horizon test','one verified historical rollout','not evidence that the final policy learned','bucket retained: yes; strict route success: no','MJLAB_TABLE_ASSISTED_EXAMPLE_20261108.md'])assert.ok(html.includes(value),value);
});

test('polling marks failed reads stale and stops without simulator commands',async()=>{
 const d=dom(),scheduled=[],cancelled=[];let reads=0;const stop=ui.startClearanceHorizon(d.root,{fetchReport:async()=>{if(++reads===1)return{[ui.KEY]:fixture()};throw Error('offline');},schedule:(fn,ms)=>{scheduled.push({fn,ms});return scheduled.length;},cancel:id=>cancelled.push(id)});await new Promise(r=>setImmediate(r));assert.equal(scheduled[0].ms,5000);await scheduled[0].fn();assert.ok(d.get('status').textContent.includes('stale'));stop();assert.deepEqual(cancelled,[2]);
});
