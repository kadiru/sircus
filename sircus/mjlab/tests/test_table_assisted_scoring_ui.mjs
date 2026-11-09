import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../web/table-assisted-scoring.js',import.meta.url),'utf8');
const ui=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const H='a'.repeat(64),J='b'.repeat(64);
function fixture(){
 const counts={unique_requests:9,scorable_native:6,retained_rejections:2,unscorable:1,proof_errors:0,invalid_native:1,old_captures:2,new_captures:4,new_table_assisted_captures:2,unchanged_misses:1};
 const pins={summary_sha256:H,manifest_sha256:J,rule_sha256:'c'.repeat(64)};
 return{version:ui.VERSION,status:'verified',new_trials:0,new_training_requests:0,totals:{...counts,metadata_rows:10,duplicate_rows:1},groups:[{source_study:'saved-study',method:'ppo',phase:'train',substeps:64,counts,old_reward_sum:8,new_reward_sum:15,reward_count:6}],examples:[{record_id:'gained',source_study:'saved-study',method:'ppo',phase:'train',old_capture:false,new_capture:true,old_reward:.1,new_reward:3.6,native_archive_sha256:H,native_row:0,board_s:.137,table_contact_s:.1,first_entry_s:.5,capture_s:2.1}],old_rule_version:'old-v2',new_rule_version:'table-v3',...pins,audit:{passed:true,full_checks_verified:true,checks_count:30,proof_sha256:'d'.repeat(64),...pins}};
}
function dom(){const nodes=new Map();let doc;const element=tag=>({tagName:tag,ownerDocument:doc,children:[],attrs:{},textContent:'',append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;},setAttribute(k,v){this.attrs[k]=v;}});doc={createElement:element};const root=element('section');root.querySelector=s=>{const key=s.match(/="([^"]+)"/)[1];if(!nodes.has(key))nodes.set(key,element('div'));return nodes.get(key);};const get=name=>root.querySelector(`[data-table-scoring="${name}"]`);return{root,get,rows:name=>get(name).children[0]?.children[1]?.children.map(r=>r.children.map(c=>c.textContent))??[]};}

test('verified same-shot counts preserve invalid, rejection and excluded denominators',()=>{
 const raw=fixture(),d=dom();assert.equal(ui.scoringReady(raw),true);ui.renderTableAssistedScoring(d.root,raw);assert.deepEqual(d.rows('overview')[0],['6','2','4','2','1']);assert.equal(d.rows('groups')[0][3],'64');assert.equal(d.rows('groups')[0][9],'2');assert.equal(d.rows('groups')[0][10],'1');assert.ok(d.get('denominator').textContent.includes('2 rejected/withheld'));assert.ok(d.get('recommendation').textContent.includes('trajectories and model weights are unchanged'));
});
test('partial proof, broken accounting or relabeled examples never publish improvement',()=>{
 for(const change of [r=>r.status='running',r=>r.audit.full_checks_verified=false,r=>r.audit.summary_sha256=J,r=>r.new_trials=1,r=>r.new_training_requests=1,r=>r.totals.invalid_native=0,r=>r.totals.proof_errors=1,r=>r.totals.metadata_rows=11,r=>r.groups[0].substeps='64',r=>r.groups[0].reward_count=5,r=>r.examples[0].old_capture=true,r=>r.examples[0].native_row=-1,r=>r.examples[0].capture_s=NaN]){const raw=fixture();change(raw);assert.equal(ui.scoringReady(raw),false);const d=dom();ui.renderTableAssistedScoring(d.root,raw);assert.equal(d.rows('overview').length,0);assert.equal(d.rows('groups').length,0);assert.equal(d.rows('examples').length,0);}
});
test('table presence is not a fabricated causal bounce and first entry is not retention start',()=>{
 const raw=fixture(),d=dom();assert.equal(ui.scoringReady(raw),true);ui.renderTableAssistedScoring(d.root,raw);assert.equal(d.rows('examples')[0][2],'0.137 / 0.100');assert.equal(d.rows('examples')[0][3],'0.500 / 2.100');assert.equal(d.rows('examples')[0][5],'3.500');raw.examples[0].table_contact_s=null;assert.equal(ui.scoringReady(raw),true);ui.renderTableAssistedScoring(d.root,raw);assert.equal(d.rows('examples')[0][2],'0.137 / unknown');
});
test('groups cannot pool identical labels across timestep without separate unique identity',()=>{
 const raw=fixture();raw.groups.push({...raw.groups[0]});assert.equal(ui.scoringReady(raw),false);raw.groups[1].substeps=128;assert.equal(ui.scoringReady(raw),false); // doubled denominator still fails
});
test('zero native trajectories show only verified inventory, not a successful comparison',()=>{
 const raw=fixture(),d=dom();const counts={unique_requests:3,scorable_native:0,retained_rejections:0,unscorable:3,proof_errors:0,invalid_native:0,old_captures:0,new_captures:0,new_table_assisted_captures:0,unchanged_misses:0};
 raw.totals={...counts,metadata_rows:3,duplicate_rows:0};raw.groups[0]={...raw.groups[0],counts,old_reward_sum:0,new_reward_sum:0,reward_count:0};raw.examples=[];
 assert.equal(ui.evidenceReady(raw),true);assert.equal(ui.scoringReady(raw),false);ui.renderTableAssistedScoring(d.root,raw);
 assert.equal(d.rows('overview').length,0);assert.equal(d.rows('examples').length,0);assert.equal(d.rows('coverage').length,1);assert.equal(d.rows('coverage')[0][3],'0');assert.equal(d.rows('coverage')[0][6],'3');assert.ok(d.get('status').textContent.includes('zero scorable native'));assert.ok(d.get('recommendation').textContent.includes('No saved native trajectories qualified'));assert.ok(!d.get('recommendation').textContent.includes('versioned rescoring baseline'));
});
test('six examples stay visible while the remaining sample is expandable',()=>{
 const raw=fixture(),d=dom();raw.totals={...raw.totals,unique_requests:15,scorable_native:12,new_captures:10,new_table_assisted_captures:8,metadata_rows:16};raw.groups[0].counts={...raw.totals};raw.groups[0].reward_count=12;const example=raw.examples[0];raw.examples=Array.from({length:8},(_,i)=>({...example,record_id:'gain-'+i,native_row:i}));
 assert.equal(ui.scoringReady(raw),true);ui.renderTableAssistedScoring(d.root,raw);assert.equal(d.rows('examples').length,6);assert.equal(d.rows('examples-more').length,2);assert.equal(d.get('more-examples').hidden,false);assert.ok(d.get('example-count').textContent.includes('6 of 8 illustrative'));assert.ok(d.get('more-examples-label').textContent.includes('remaining 2'));
 ui.renderTableAssistedScoring(d.root,fixture());assert.equal(d.get('more-examples').hidden,true);
});
test('embedded mount fails closed on broken JSON and sends no network or motion calls',()=>{
 const d=dom(),doc={getElementById:id=>id==='tableAssistedScoring'?d.root:id===ui.SNAPSHOT_ID?{textContent:'not json'}:null};ui.mountTableAssistedScoring(doc);assert.ok(d.get('status').textContent.includes('failed'));doc.getElementById=id=>id==='tableAssistedScoring'?d.root:id===ui.SNAPSHOT_ID?{textContent:JSON.stringify(fixture())}:null;ui.mountTableAssistedScoring(doc);assert.equal(d.rows('overview').length,1);
 for(const value of ['fetch(','XMLHttpRequest','/api/','innerHTML','</script','activatePolicy','startTraining'])assert.ok(!source.includes(value),value);
});
test('panel explains retrospective scope and responsive display',()=>{
 const html=readFileSync(new URL('../web/table-assisted-scoring.html',import.meta.url),'utf8'),css=readFileSync(new URL('../web/table-assisted-scoring.css',import.meta.url),'utf8');for(const value of ['id="tableAssistedScoring"','Historical results keep their original definitions','not proof of a causal bounce','rejected or withheld','No simulation shots','read-only'])assert.ok(html.toLowerCase().includes(value.toLowerCase()),value);assert.ok(css.includes('overflow-x:auto'));assert.ok(css.includes('grid-column:1/-1'));
});
test('exhaustive groups are collapsed by default and coverage stays outside details',()=>{
 const html=readFileSync(new URL('../web/table-assisted-scoring.html',import.meta.url),'utf8');assert.match(html,/<details><summary>Expand study \/ method \/ phase \/ timestep breakdown<\/summary>/);assert.ok(!html.includes('<details open'));const group=html.indexOf('data-table-scoring="groups"');const close=html.indexOf('</details>',group);const coverage=html.indexOf('data-table-scoring="coverage"');assert.ok(close<coverage);assert.ok(html.indexOf('data-table-scoring="overview"')<group);
});
