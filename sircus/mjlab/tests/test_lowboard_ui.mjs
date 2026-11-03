import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
const section=name=>source.slice(source.indexOf(`// BEGIN READ-ONLY ${name}`),source.indexOf(`// END READ-ONLY ${name}`));
const code=section('LOW BOARD STUDY'),make=extra=>{const c=vm.createContext({finiteVector:(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite),...extra});vm.runInContext(section('WINDOW STUDY')+section('HEIGHT TILT STUDY')+section('DISTANCE STUDY')+code,c);return c;};
function fixture(){
  const r={id:'low-recipe',context_id:'low-context',distance_id:'corner-700',ray_id:'corner',distance_m:.7,bucket_xy_m:[.7,0],board_xy_m:[0,0],
    board_height_m:.22,tilt_deg:28,gap_m:.5375,release_center_height_m:.795,nominal_by_substeps:{},fresh_by_substeps:{},by_substeps:{},
    sensitivity:{paired_engineering_gate:true,all_resolution_repeat_gate:true,paired_attempts:33,capture_agreement_64_128:33,nominal_position_delta_350ms_max_m:.002}};
  for(const step of ['32','64','128']){r.nominal_by_substeps[step]={attempts:3,captures:3,native_releases:3};r.fresh_by_substeps[step]={attempts:30,captures:30,native_releases:30};r.by_substeps[step]={attempts:33,captures:33,native_releases:33,clean:0};}
  return {version:'sircus-low-board-far-development-v1',status:'completed',phase:'completed',contract_fingerprint:'lowboard',max_requests:1968,
    audit:{status:'passed',contract_fingerprint:'lowboard',checks_count:2,full_checks_verified:true,source_sha256:'a'.repeat(64)},
    settings:{board_xy_m:[0,0],grid:{board_heights_m:[.22,.24],tilts_deg:[28,32],release_headroom_m:[.005,.015,.03,.05,.08]},
      reach_limit:{ball_center_height_m:.8,board_height_plus_gap_m:.7625,scope:'unchanged cheap spherical prior only; native IK/clearance separate',formula:'ceiling minus radius minus board H minus headroom'}},
    counts:{requests:12,native_releases:8,captures:3},distances:[{id:'corner-700',ray_id:'corner',distance_m:.7,board_xy_m:[0,0],bucket_xy_m:[.7,0]}],
    contexts:[{id:'low-context',distance_id:'corner-700',stage:'development',board_xy_m:[0,0],bucket_xy_m:[.7,0],board_height_m:.22,tilt_deg:28,board_admitted:true,
      geometry_requests:5,eligible:3,search_attempts:5,search_native_releases:3,search_captures:1},
      {id:'control-context',distance_id:'control-only',stage:'control',board_height_m:.4,tilt_deg:32,geometry_requests:3,eligible:3,search_attempts:3,search_native_releases:3,search_captures:3}],
    selected_recipes:[r],trace_examples:[{record_id:'native-low',context_id:'low-context',distance_id:'corner-700',board_xy_m:[0,0],bucket_xy_m:[.7,0],distance_m:.7,
      substeps:128,route_success:false,position_m:[[0,0,.795],[.7,0,.1]],selection_scope:'Actual saved illustrative miss; not aggregate performance'}]};
}

test('low-board study remains optional and keeps controls out of development/search summaries',()=>{
  const c=make();assert.equal(c.lowBoardStudyView(null),null);const s=fixture(),view=c.lowBoardStudyView(s),summary=c.distanceSearchSummary(view,view.distances[0]);
  assert.equal(view.development.length,1);assert.equal(summary.requests,5);assert.equal(summary.native,3);assert.equal(summary.captures,1);
  const old={distance_evaluation:{captures:2},active_policy:'untouched'},before=JSON.stringify(old);c.lowBoardStudyView(s);assert.equal(JSON.stringify(old),before);
});
test('own version and complete matching full-audit proof are mandatory for qualification',()=>{
  const c=make();let s=fixture();assert.equal(c.lowBoardAuditReady(s),true);assert.equal(c.lowBoardRepeatCheck(s,s.selected_recipes[0]).qualified,true);
  for(const corrupt of [s=>s.version='sircus-bucket-distance-development-v1',s=>s.status='running',s=>s.phase='fresh-128',s=>s.contract_fingerprint='',
    s=>s.audit.contract_fingerprint='parent',s=>s.audit.full_checks_verified=false,s=>s.audit.full_checks_verified='yes',s=>s.audit.checks_count=0,
    s=>s.audit.checks_count=true,s=>delete s.audit.source_sha256,s=>s.audit.source_sha256='bad']){
    s=fixture();corrupt(s);assert.equal(c.lowBoardAuditReady(s),false);assert.equal(c.lowBoardRepeatCheck(s,s.selected_recipes[0]).qualified,null);
  }
});
test('all-resolution count threshold and fine-pair gate remain distinct with complete native denominators',()=>{
  const c=make(),s=fixture(),r=s.selected_recipes[0];r.fresh_by_substeps['32'].captures=27;r.by_substeps['32'].captures=30;
  let check=c.lowBoardRepeatCheck(s,r);assert.equal(check.qualified,true);assert.equal(check.perfect,false);
  r.fresh_by_substeps['32'].captures=26;r.by_substeps['32'].captures=29;r.sensitivity.all_resolution_repeat_gate=false;
  check=c.lowBoardRepeatCheck(s,r);assert.equal(check.qualified,false);assert.equal(r.sensitivity.paired_engineering_gate,true);
  r.sensitivity.all_resolution_repeat_gate=true;assert.equal(c.lowBoardRepeatCheck(s,r).qualified,null);
  delete r.by_substeps['128'].native_releases;assert.equal(c.lowBoardRepeatCheck(s,r).verified,false);
});
test('new low-board panel is read-only, above its parent and explicitly labels the cheap prior',()=>{
  for(const forbidden of ['fetch(', '/api/solve','startTraining','activateCheckpoint','syncSolveButtons'])assert.ok(!code.includes(forbidden));
  assert.ok(source.includes("fetch('/api/solve'"));assert.ok(source.includes('showHeightTiltStudy(); showDistanceStudy(); showLowBoardStudy();'));
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8'),css=readFileSync(new URL('../web/style.css',import.meta.url),'utf8');
  assert.ok(html.indexOf('id="lowBoardStudy"')<html.indexOf('id="distanceStudy"'));
  for(const id of ['lowBoardStudy','lowBoardSummary','lowBoardCeiling','lowBoardTargets','lowBoardMaps','lowBoardRecipes','lowBoardRepeats','lowBoardTrace','lowBoardAudit'])assert.ok(html.includes(`id="${id}"`));
  for(const text of ['cheap spherical reach prior','Table-assisted entries remain failures','No new RL training','reported 175 mm','parent distance study'])assert.ok(html.includes(text));
  assert.match(css,/\.lowboard-section\{[^}]*min-width:0;max-width:100%/);assert.ok(css.includes('.lowboard-section .table-wrap{max-width:100%}'));
});
test('actual partial and completed data render exact settings and native flight references without parent mutations',()=>{
  const nodes=new Map(),tables=new Map(),labels=[],draw=new Proxy({fillText:(s)=>labels.push(s)},{get:(o,k)=>o[k]??(()=>{})});
  const element=()=>({children:[],value:'0',width:820,height:260,hidden:false,textContent:'',append(...x){this.children.push(...x);},replaceChildren(...x){this.children=x;},setAttribute(){},getContext(){return draw;}});
  const node=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
  const previous={captures:2,contract_fingerprint:'parent'};
  const c=make({$:node,document:{createElement:element},report:{distance_evaluation:previous},addMetrics:(r,t)=>t.metrics=r,expansionTable:(t,h,r)=>tables.set(t,{headers:h,rows:r})});
  c.showLowBoardStudy();assert.equal(node('lowBoardStudy').hidden,true);
  const s=fixture();c.report.lowboard_evaluation=s;const before=JSON.stringify(s.trace_examples);c.showLowBoardStudy();
  const target=tables.get(node('lowBoardTargets')).rows[0];assert.equal(target[3],'3 / 5');assert.equal(target[4],'3 / 5');assert.equal(target[5],'1 / 5');
  const recipe=tables.get(node('lowBoardRecipes')).rows[0];assert.equal(recipe[1],'220.0 / 28.00°');assert.equal(recipe[2],'537.5 / 795.0');assert.equal(recipe[3],'5.0');assert.equal(recipe[4],'Perfect 33/33 each');
  assert.equal(tables.get(node('lowBoardRepeats')).rows[0][3],'30 / 30');assert.equal(tables.get(node('lowBoardRepeats')).rows[0][5],'33 / 33');
  assert.equal(tables.get(node('lowBoardSensitivity')).rows[0][4],'Passed');assert.equal(node('lowBoardMaps').children.length,1);
  assert.ok(node('lowBoardCeiling').textContent.includes('cheap spherical prior'));assert.ok(node('lowBoardCeiling').textContent.includes('not a certified native reach limit'));
  assert.ok(labels.includes('Bucket center axis (700.0 mm)'));assert.ok(labels.includes('Configured board center'));assert.ok(node('lowBoardTraceCaption').textContent.includes('not a native observation'));
  assert.equal(JSON.stringify(s.trace_examples),before);assert.equal(c.report.distance_evaluation,previous);
  const selected=s.selected_recipes;s.selected_recipes=[];s.counts.captures=24;c.showLowBoardStudy();
  assert.equal(node('lowBoardConclusion').textContent,'No captured development setting was selected; repeatability is not established.');assert.equal(s.counts.captures,24);
  s.selected_recipes=selected;
  s.status='waiting_for_4090';s.phase='device_queue';s.queue={reason:'shared GPU lock',wait_s:120};c.showLowBoardStudy();assert.ok(node('lowBoardStatus').textContent.includes('2 min queued'));assert.ok(node('lowBoardConclusion').textContent.includes('awaits'));assert.equal(tables.get(node('lowBoardSensitivity')).rows[0][4],'Unverified');
  s.status='failed';s.error='preflight stopped';s.trace_examples=[];c.showLowBoardStudy();assert.ok(node('lowBoardStatus').textContent.includes('preflight stopped'));assert.equal(node('lowBoardTraceSelect').disabled,true);assert.ok(node('lowBoardTraceCaption').textContent.includes('no ideal trajectory'));
});
