import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source=readFileSync(new URL('../web/app.js',import.meta.url),'utf8'),section=n=>source.slice(source.indexOf(`// BEGIN READ-ONLY ${n}`),source.indexOf(`// END READ-ONLY ${n}`));
const code=section('SHALLOW BOARD STUDY'),make=extra=>{const c=vm.createContext({finiteVector:(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite),...extra});vm.runInContext(['WINDOW STUDY','HEIGHT TILT STUDY','DISTANCE STUDY','LOW BOARD STUDY','SHALLOW BOARD STUDY'].map(section).join('\n'),c);return c;};
function fixture(){
  const r={id:'recipe',context_id:'discovery',distance_id:'corner-700',distance_m:.7,bucket_xy_m:[.7,0],board_xy_m:[0,0],board_height_m:.38,tilt_deg:28,gap_m:.3868,release_center_height_m:.8043,
    nominal_by_substeps:{},fresh_by_substeps:{},by_substeps:{},sensitivity:{paired_engineering_gate:true,all_resolution_repeat_gate:true,paired_attempts:33,capture_agreement_64_128:33,nominal_position_delta_350ms_max_m:.002}};
  for(const step of ['32','64','128']){r.nominal_by_substeps[step]={attempts:3,captures:3,native_releases:3};r.fresh_by_substeps[step]={attempts:30,captures:30,native_releases:30};r.by_substeps[step]={attempts:33,captures:33,native_releases:33,clean:0};}
  return {version:'sircus-shallow-board-candidate-v1',status:'completed',phase:'completed',contract_fingerprint:'exact-shallow',max_requests:424,
    audit:{status:'passed',contract_fingerprint:'exact-shallow',full_checks_verified:true,checks_count:1,source_sha256:'a'.repeat(64)},
    settings:{board_xy_m:[0,0],candidate:{board_height_m:.38,tilt_deg:28,gap_m:.3868,release_center_height_m:.8043,ball_radius_m:.0375,release_headroom_m:.0150243380486807,
      input_precision:'Exact 386.8 mm user input',gap_definition:'Vertical ball-bottom clearance to board center'}},
    counts:{requests:25,native_releases:25,captures:25,clean:0,phase_counts:{controls:24,coarse:1}},
    distances:[{id:'corner-700',distance_m:.7,bucket_xy_m:[.7,0],board_xy_m:[0,0]}],
    contexts:[{id:'discovery',stage:'shallow',distance_id:'corner-700',bucket_xy_m:[.7,0],board_xy_m:[0,0],board_height_m:.38,tilt_deg:28,
      board_admitted:true,geometry_requests:1,eligible:1,search_attempts:1,search_native_releases:1,search_captures:1,clean:0},
      {id:'control',stage:'control',distance_id:'control-only',search_attempts:3,search_native_releases:3,search_captures:3,clean:0}],
    selected_recipes:[r],trace_examples:[{record_id:'actual-shot',context_id:'discovery',distance_id:'corner-700',distance_m:.7,bucket_xy_m:[.7,0],board_xy_m:[0,0],substeps:64,route_success:true,
      position_m:[[0,0,.8043],[.7,0,.1]],selection_scope:'Saved native illustrative success; not aggregate performance'}]};
}

test('shallow view keeps baseline controls out of discovery and leaves missing quality unknown',()=>{
  const c=make(),s=fixture(),v=c.shallowStudyView(s);assert.equal(c.shallowStudyView(null),null);assert.equal(v.development.length,1);
  const counts=c.shallowSearchCounts(v.development);assert.equal(counts.requests,1);assert.equal(counts.native,1);assert.equal(counts.captures,1);assert.equal(counts.clean,0);assert.equal(counts.assisted,1);
  delete s.contexts[0].clean;assert.equal(c.shallowSearchCounts(c.shallowStudyView(s).development).assisted,null);
});
test('qualification requires the exact candidate version and completed full-audit proof',()=>{
  const c=make();let s=fixture();assert.equal(c.shallowRepeatCheck(s,s.selected_recipes[0]).qualified,true);
  for(const corrupt of [s=>s.version='sircus-low-board-far-development-v1',s=>s.audit.contract_fingerprint='old',s=>s.audit.full_checks_verified=false,
    s=>s.audit.full_checks_verified='true',s=>s.audit.checks_count=0,s=>s.audit.checks_count=true,s=>s.audit.source_sha256='short',s=>s.status='running',s=>s.phase='nominal-128']){
    s=fixture();corrupt(s);assert.equal(c.shallowAuditReady(s),false);assert.equal(c.shallowRepeatCheck(s,s.selected_recipes[0]).qualified,null);
  }
  s=fixture();const r=s.selected_recipes[0];r.fresh_by_substeps['32'].captures=26;r.by_substeps['32'].captures=29;r.sensitivity.all_resolution_repeat_gate=false;
  assert.equal(c.shallowRepeatCheck(s,r).qualified,false);assert.equal(r.sensitivity.paired_engineering_gate,true);
  delete r.by_substeps['128'].native_releases;assert.equal(c.shallowRepeatCheck(s,r).verified,false);
});
test('new section is engineering-only, above earlier panels, with untouched solve handlers',()=>{
  for(const forbidden of ['fetch(', '/api/solve','startTraining','activateCheckpoint','syncSolveButtons'])assert.ok(!code.includes(forbidden));
  assert.ok(source.includes("fetch('/api/solve'"));assert.ok(source.includes('showDistanceStudy(); showLowBoardStudy(); showShallowStudy();'));
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8'),css=readFileSync(new URL('../web/style.css',import.meta.url),'utf8');
  assert.ok(html.indexOf('id="shallowBoardStudy"')<html.indexOf('id="lowBoardStudy"'));
  for(const id of ['shallowBoardStudy','shallowCandidate','shallowSummary','shallowTargets','shallowRecipes','shallowRepeats','shallowTrace','shallowAudit'])assert.ok(html.includes(`id="${id}"`));
  for(const text of ['not new RL training','assisted/marginal does not by itself prove a rim contact','Table-assisted entries remain failures','frozen learned selector on separate bucket layouts'])assert.ok(html.includes(text));
  assert.match(css,/\.shallow-section\{[^}]*min-width:0;max-width:100%/);
});
test('partial and completed rendering show exact rounded candidate, separate phases and actual own-bucket trace',()=>{
  const nodes=new Map(),tables=new Map(),labels=[],draw=new Proxy({fillText:s=>labels.push(s)},{get:(o,k)=>o[k]??(()=>{})});
  const element=()=>({children:[],value:'0',width:820,height:260,textContent:'',hidden:false,append(...x){this.children.push(...x);},replaceChildren(...x){this.children=x;},setAttribute(){},getContext(){return draw;}}),node=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
  const parent={captures:24,contract_fingerprint:'previous'};
  const c=make({$:node,document:{createElement:element},report:{lowboard_evaluation:parent},addMetrics:(r,t)=>t.metrics=r,expansionTable:(t,h,r)=>tables.set(t,{h,r})});
  c.showShallowStudy();assert.equal(node('shallowBoardStudy').hidden,true);
  const s=fixture();c.report.shallow_evaluation=s;const traceBefore=JSON.stringify(s.trace_examples);c.showShallowStudy();
  assert.ok(node('shallowCandidate').textContent.includes('gap 386.800 mm'));assert.ok(node('shallowCandidate').textContent.includes('ball-center H 804.300 mm'));assert.ok(node('shallowCandidate').textContent.includes('Exact 386.8 mm user input'));
  let metrics=new Map(node('shallowSummary').metrics);assert.equal(metrics.get('Discovery strict captures'),1);assert.equal(metrics.get('Baseline control requests'),24);assert.equal(metrics.get('All-phase captures (includes controls/repeats)'),25);
  const target=tables.get(node('shallowTargets')).r[0];assert.equal(target[3],'1 / 1');assert.equal(target[4],'1 / 1');assert.equal(target[5],'0 / 1');assert.equal(target[6],'1 / 1');
  const repeat=tables.get(node('shallowRepeats')).r[0];assert.equal(repeat[3],'30 / 30');assert.equal(repeat[5],'33 / 33');assert.equal(repeat[6],'0 / 33');assert.equal(repeat[7],'33 / 33');
  assert.equal(tables.get(node('shallowSensitivity')).r[0][4],'Passed');assert.ok(labels.includes('Bucket center axis (700.0 mm)'));assert.ok(labels.includes('Configured board center'));
  assert.ok(node('shallowTraceCaption').textContent.includes('not a native observation'));assert.equal(JSON.stringify(s.trace_examples),traceBefore);assert.equal(c.report.lowboard_evaluation,parent);
  s.selected_recipes=[];s.contexts[0].search_captures=0;s.counts.captures=24;c.showShallowStudy();metrics=new Map(node('shallowSummary').metrics);
  assert.equal(metrics.get('Discovery strict captures'),0);assert.equal(metrics.get('All-phase captures (includes controls/repeats)'),24);assert.equal(node('shallowConclusion').textContent,'No captured discovery setting was selected; repeatability is not established.');
  s.status='waiting_for_4090';s.phase='device_queue';s.queue={reason:'shared lock',wait_s:60};c.showShallowStudy();assert.ok(node('shallowStatus').textContent.includes('1 min queued'));assert.ok(node('shallowConclusion').textContent.includes('await'));
  s.status='failed';s.error='preflight stopped';s.trace_examples=[];c.showShallowStudy();assert.ok(node('shallowStatus').textContent.includes('preflight stopped'));assert.equal(node('shallowTraceSelect').disabled,true);assert.ok(node('shallowTraceCaption').textContent.includes('no ideal flight'));
});
