import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
const code=source.slice(source.indexOf('// BEGIN READ-ONLY HEIGHT TILT STUDY'),source.indexOf('// END READ-ONLY HEIGHT TILT STUDY'));
const windowCode=source.slice(source.indexOf('// BEGIN READ-ONLY WINDOW STUDY'),source.indexOf('// END READ-ONLY WINDOW STUDY'));
const make=extra=>{const c=vm.createContext({finiteVector:(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite),...extra});vm.runInContext(windowCode+'\n'+code,c);return c;};
function referenceFixture(){
  const load=name=>JSON.parse(readFileSync(new URL(`../../../baselines/mjlab-window-sac-20261102/${name}`,import.meta.url),'utf8'));
  const previous=load('native-results.json');previous.audit=load('audit-window.json');previous.window_confirmations=load('confirmation/report.json');previous.window_confirmations.audit=load('confirmation/audit-window-confirmation.json');
  const recipes=previous.window_confirmations.recipes,context=previous.test_contexts.find(r=>r.methods?.adaptive_surrogate?.total?.captures>0),model=context.methods.adaptive_surrogate;
  const anchors=[...recipes.filter(r=>r.nominal.captures===r.nominal.attempts&&r.fresh.captures===r.fresh.attempts).map(r=>({id:r.id,values:[...previous.settings.board_xy_m,r.board_height_m,r.tilt_deg,r.gap_m]})),
    {id:'surrogate-reference',values:[...previous.settings.board_xy_m,context.board_height_m,context.tilt_deg,model.gap_m]}];
  return {study:{settings:{board_xy_m:previous.settings.board_xy_m,bucket_xy_m:previous.settings.bucket_xy_m},anchors},previous};
}
function repeatFixture(fresh32=20){
  const study={version:'sircus-local-height-tilt-sensitivity-v1',status:'completed',contract_fingerprint:'matched-contract',audit:{status:'passed',contract_fingerprint:'matched-contract'}};
  const recipe={id:'recipe',board_height_m:.4,tilt_deg:32,gap_m:.3,nominal_by_substeps:{},fresh_by_substeps:{},by_substeps:{},sensitivity:{paired_engineering_gate:true}};
  for(const step of ['32','64','128']){
    recipe.nominal_by_substeps[step]={attempts:3,captures:3,native_releases:3};
    recipe.fresh_by_substeps[step]={attempts:20,captures:step==='32'?fresh32:20,native_releases:20};
    recipe.by_substeps[step]={attempts:23,captures:3+recipe.fresh_by_substeps[step].captures,native_releases:23};
  }
  return {study,recipe};
}

test('neighborhood evidence stays optional and does not conflate IK with a capture',()=>{
  const c=make();assert.equal(c.neighborhoodView(null),null);
  let cell=c.neighborhoodCell({board_admitted:false,search_attempts:17,search_captures:0});
  assert.equal(cell.state,'unresolved');assert.equal(cell.fraction,'0/17');assert.equal(cell.released,null);
  cell=c.neighborhoodCell({board_admitted:true,geometry_requests:17,eligible:0,search_attempts:17,search_captures:0});
  assert.equal(cell.state,'pair_rejected');
  cell=c.neighborhoodCell({board_admitted:true,eligible:10,search_attempts:17,search_captures:0});assert.equal(cell.state,'endpoint');
  cell=c.neighborhoodCell({board_admitted:true,search_attempts:17,search_captures:0,search_native_releases:10});assert.equal(cell.state,'miss');
  cell=c.neighborhoodCell({board_admitted:true,search_attempts:17,search_captures:5,search_native_releases:10});assert.equal(cell.state,'capture');assert.equal(cell.fraction,'5/17');
  assert.equal(c.neighborhoodCell({}).fraction,'—');assert.equal(c.neighborhoodCell({}).state,'pending');
  assert.equal(c.neighborhoodCell({board_admitted:null,search_attempts:0,search_captures:0,search_native_releases:0}).fraction,'pending');
});
test('queued family data stays unknown rather than implying failed IK or missed shots',()=>{
  const c=make(),rows=Array.from({length:25},()=>({board_admitted:null,search_attempts:0,search_captures:0}));
  let value=c.neighborhoodFamilySummary(rows);assert.equal(value.resolved,0);assert.equal(value.unresolved,0);assert.equal(value.unknown,25);assert.equal(value.requests,0);
  rows[0]={board_admitted:true,search_attempts:17,search_captures:5};rows[1]={board_admitted:false,search_attempts:17,search_captures:0};
  value=c.neighborhoodFamilySummary(rows);assert.equal(value.resolved,1);assert.equal(value.unresolved,1);assert.equal(value.unknown,23);assert.equal(value.requests,34);assert.equal(value.captures,5);
  delete rows[2].search_attempts;assert.equal(c.neighborhoodFamilySummary(rows).requests,null);
});
test('narrow layouts allow grid tracks to shrink while tables remain scrollable',()=>{
  const css=readFileSync(new URL('../web/style.css',import.meta.url),'utf8');
  assert.match(css,/\.workspace\{min-width:0\}/);
  assert.match(css,/\.neighborhood-section\{min-width:0;max-width:100%;overflow-wrap:anywhere\}/);
  assert.match(css,/\.neighborhood-map\{min-width:0\}/);
  assert.match(css,/\.table-wrap\{overflow-x:auto/);
});
test('prior reference table reads the three audited source shots without changing counts',()=>{
  const c=make(),{study,previous}=referenceFixture(),before=JSON.stringify(previous),rows=c.neighborhoodReferenceShots(study,previous);
  assert.equal(rows.length,3);assert.ok(rows.every(r=>r.verified&&r.substeps===64));
  assert.deepEqual(Array.from(rows,r=>r.source),['Confirm','Confirm','Test']);
  for(const row of rows){assert.equal(row.nominal.captures,3);assert.equal(row.fresh.captures,20);assert.equal(row.total.native_releases,23);}
  assert.equal(JSON.stringify(previous),before);
  const recipe=previous.window_confirmations.recipes.find(r=>r.id===rows[0].source_id);
  recipe.nominal={attempts:2,captures:1,native_releases:2};recipe.fresh={attempts:4,captures:3,native_releases:4};recipe.total={attempts:6,captures:4,native_releases:6};
  const changed=c.neighborhoodReferenceShots(study,previous)[0];assert.equal(changed.nominal.captures,1);assert.equal(changed.fresh.captures,3);assert.equal(changed.total.native_releases,6);
});
test('missing or mismatched reference audits and geometry stay unknown',()=>{
  const c=make();let data=referenceFixture();
  assert.ok(c.neighborhoodReferenceShots(data.study,undefined).every(r=>!r.verified));
  for(const corrupt of [d=>d.previous.audit.contract_fingerprint='other',d=>d.previous.audit.status='failed',d=>d.previous.status='running',d=>d.previous.version='unknown',d=>d.study.settings.bucket_xy_m=[0,0]]){
    data=referenceFixture();corrupt(data);assert.ok(c.neighborhoodReferenceShots(data.study,data.previous).every(r=>!r.verified));
  }
  data=referenceFixture();data.previous.window_confirmations.audit.status='failed';let rows=c.neighborhoodReferenceShots(data.study,data.previous);assert.equal(rows[0].verified,false);assert.equal(rows[1].verified,false);assert.equal(rows[2].verified,true);
  data=referenceFixture();data.previous.window_confirmations.parent_contract_fingerprint='other';rows=c.neighborhoodReferenceShots(data.study,data.previous);assert.equal(rows[0].verified,false);assert.equal(rows[2].verified,true);
  data=referenceFixture();data.study.anchors[0].values[0]+=.1;assert.equal(c.neighborhoodReferenceShots(data.study,data.previous)[0].verified,false);
});
test('prior matches must be unique and complete; missing native counts are never inferred',()=>{
  const c=make();let data=referenceFixture();const id=data.study.anchors[0].id;
  data.previous.window_confirmations.recipes=data.previous.window_confirmations.recipes.filter(r=>r.id!==id);
  assert.equal(c.neighborhoodReferenceShots(data.study,data.previous)[0].verified,false);
  data=referenceFixture();data.previous.window_confirmations.recipes.push(data.previous.window_confirmations.recipes.find(r=>r.id===id));
  assert.equal(c.neighborhoodReferenceShots(data.study,data.previous)[0].verified,false);
  data=referenceFixture();let recipe=data.previous.window_confirmations.recipes.find(r=>r.id===id);delete recipe.fresh;
  assert.equal(c.neighborhoodReferenceShots(data.study,data.previous)[0].verified,false);
  data=referenceFixture();recipe=data.previous.window_confirmations.recipes.find(r=>r.id===id);delete recipe.total.native_releases;
  const row=c.neighborhoodReferenceShots(data.study,data.previous)[0];assert.equal(row.verified,true);assert.equal(c.windowStatistics(row.total).released,null);
  recipe.total.native_releases=0;assert.equal(c.neighborhoodReferenceShots(data.study,data.previous)[0].verified,false);
});
test('timestep agreement does not invent a denominator or assume convergence',()=>{
  const c=make();assert.equal(c.neighborhoodAgreement({agreements:22,attempts:23}),'22 / 23');
  assert.equal(c.neighborhoodAgreement(22),'22 matched');assert.equal(c.neighborhoodAgreement(22,23),'22 / 23');assert.equal(c.neighborhoodAgreement(.95),'unreported');
  assert.equal(c.neighborhoodAgreement({agreements:24,attempts:23}),'unreported');
});
test('three-resolution capture check distinguishes perfect, threshold and fine-pair qualification',()=>{
  const c=make();let {study,recipe}=repeatFixture();const before=JSON.stringify(recipe);
  let check=c.neighborhoodRepeatCheck(study,recipe);assert.equal(check.perfect,true);assert.equal(check.qualified,true);assert.equal(JSON.stringify(recipe),before);
  for(const captures of [18,19]){
    ({study,recipe}=repeatFixture(captures));check=c.neighborhoodRepeatCheck(study,recipe);assert.equal(check.perfect,false);assert.equal(check.thresholds,true);assert.equal(check.qualified,true);
  }
  ({study,recipe}=repeatFixture(17));check=c.neighborhoodRepeatCheck(study,recipe);assert.equal(check.verified,true);assert.equal(check.thresholds,false);assert.equal(check.qualified,false);
  ({study,recipe}=repeatFixture());recipe.sensitivity.paired_engineering_gate=false;check=c.neighborhoodRepeatCheck(study,recipe);assert.equal(check.perfect,true);assert.equal(check.thresholds,true);assert.equal(check.qualified,false);
  delete recipe.sensitivity.paired_engineering_gate;assert.equal(c.neighborhoodRepeatCheck(study,recipe).qualified,null);
});
test('three-resolution qualification stays unverified without complete counts and a matching completed audit',()=>{
  const c=make();
  for(const corrupt of [d=>delete d.study.audit,d=>d.study.audit.status='pending',d=>d.study.audit.contract_fingerprint='other',d=>d.study.contract_fingerprint='',d=>d.study.status='running',d=>d.study.version='other',
    d=>delete d.recipe.fresh_by_substeps['32'],d=>delete d.recipe.by_substeps['128'].native_releases,d=>d.recipe.by_substeps['64'].captures=22,d=>d.recipe.nominal_by_substeps['32'].captures=4,d=>d.recipe.fresh_by_substeps['32'].native_releases=19]){
    const data=repeatFixture();corrupt(data);const check=c.neighborhoodRepeatCheck(data.study,data.recipe);assert.equal(check.verified,false);assert.equal(check.qualified,null);assert.equal(check.label,'Unverified');
  }
  const {study,recipe}=repeatFixture();recipe.nominal_by_substeps['32'].captures=2;recipe.by_substeps['32'].captures=22;
  const check=c.neighborhoodRepeatCheck(study,recipe);assert.equal(check.verified,true);assert.equal(check.thresholds,false);
});
test('readonly panel has no solve, training, network or policy mutation hooks',()=>{
  for(const text of ['fetch(', '/api/solve','syncSolveButtons','activateCheckpoint','startTraining'])assert.ok(!code.includes(text));
  assert.ok(source.includes("fetch('/api/solve'"));assert.ok(source.includes('showCorrectedGrid(); showWindowStudy(); showHeightTiltStudy();'));
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
  for(const id of ['heightTiltStudy','heightTiltReferences','heightTiltReferenceScope','heightTiltMaps','heightTiltConclusion','heightTiltRecipes','heightTiltContexts','heightTiltSensitivity','heightTiltRepeats','heightTiltTraceSelect','heightTiltTrace'])assert.ok(html.includes(`id="${id}"`));
  for(const text of ['No RL training','No RL training or new held-out evaluation','reported 175 mm','does not activate a policy','not a certified continuous range'])assert.ok(html.includes(text));
  assert.ok(html.includes('64↔128 substeps only'));assert.ok(html.includes('does not assess 32-substep positional convergence'));
  assert.ok(html.includes('Additional descriptive check, defined after the run'));
});
test('render preserves requested, native and per-timestep repeat denominators',()=>{
  const nodes=new Map(),tables=new Map();
  const draw=new Proxy({},{get:(o,k)=>o[k]??(()=>{})});
  const element=()=>({children:[],value:'0',width:260,height:235,hidden:false,textContent:'',append(...x){this.children.push(...x);},replaceChildren(...x){this.children=x;},setAttribute(){},getContext(){return draw;}});
  const node=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
  const c=make({$:node,document:{createElement:element},report:{},addMetrics:(rows,target)=>{target.metrics=rows;},expansionTable:(target,headers,rows)=>{tables.set(target,{headers,rows});}});
  c.showHeightTiltStudy();assert.equal(node('heightTiltStudy').hidden,true);
  c.report.neighborhood_evaluation={status:'waiting_for_4090',phase:'device_queue',contexts:[{id:'a',source_family:'anchor-a',board_height_m:.4,tilt_deg:32,board_admitted:null,geometry_requests:0,eligible:0,search_attempts:0,search_native_releases:0,search_captures:0}]};
  c.showHeightTiltStudy();assert.ok(node('heightTiltMaps').children[0].children[2].textContent.includes('1 unknown'));assert.ok(node('heightTiltMaps').children[0].children[2].textContent.includes('search pending'));
  const earlier=referenceFixture();Object.assign(c.report.neighborhood_evaluation,{anchors:earlier.study.anchors,settings:earlier.study.settings});c.report.window_study_evaluation=earlier.previous;c.showHeightTiltStudy();
  assert.equal(tables.get(node('heightTiltReferences')).rows[0][3],'3 / 3');assert.equal(tables.get(node('heightTiltReferences')).rows[0][4],'20 / 20');assert.ok(node('heightTiltReferenceScope').textContent.includes('32/128-substep results are pending'));assert.equal(c.report.neighborhood_evaluation.contexts[0].search_attempts,0);
  c.report.neighborhood_evaluation.status='failed';c.showHeightTiltStudy();assert.ok(node('heightTiltReferenceScope').textContent.includes('No new 32/128-substep results are available'));
  c.report.neighborhood_evaluation={status:'running',phase:'search',contexts:[{id:'a',source_family:'anchor-a',board_height_m:.4,tilt_deg:32,board_admitted:true,geometry_requests:17,eligible:10,search_attempts:17,search_native_releases:10,search_captures:7,best_gap_m:.3}],
    settings:{board_xy_m:[.5,-.3],bucket_xy_m:[.1,-.6]},counts:{requests:17,native_releases:10},selected_recipes:[{id:'recipe',context_id:'a',board_height_m:.4,tilt_deg:32,gap_m:.3,release_center_height_m:.7375,distance_m:.5,
      by_substeps:{'32':{attempts:23,captures:13,native_releases:20,clean:3},'64':{attempts:23,captures:21,native_releases:22,clean:5},'128':{attempts:23,captures:20}},
      nominal_by_substeps:{'64':{attempts:3,captures:3}},fresh_by_substeps:{'64':{attempts:20,captures:18}},sensitivity:{capture_agreement_64_128:{agreements:22,attempts:23},nominal_position_delta_350ms_max_m:.004,paired_engineering_gate:false}}],
    trace_examples:[{record_id:'actual',context_id:'a',substeps:128,route_success:false,position_m:[[.5,-.3,.7],[.2,-.6,.2]]}]};
  c.showHeightTiltStudy();assert.equal(node('heightTiltStudy').hidden,false);assert.equal(node('heightTiltMaps').children.length,1);
  const searched=tables.get(node('heightTiltContexts')).rows[0];assert.equal(searched[3],'10 / 17');assert.equal(searched[4],'10 / 17');assert.equal(searched[5],'7 / 17');assert.equal(searched[6],'300.0');
  const sensitivity=tables.get(node('heightTiltSensitivity')).rows[0];assert.deepEqual(Array.from(sensitivity.slice(1,4)),['13 / 23','21 / 23','20 / 23']);assert.equal(sensitivity[4],'22 / 23');assert.equal(sensitivity[5],'4.0');assert.equal(sensitivity[6],'Unverified');
  assert.equal(tables.get(node('heightTiltSensitivity')).headers[6],'64↔128 gate');assert.ok(node('heightTiltConclusion').textContent.includes('qualification is unverified'));
  const repeats=tables.get(node('heightTiltRepeats')).rows;assert.equal(repeats[1][2],'3 / 3');assert.equal(repeats[1][3],'18 / 20');assert.equal(repeats[1][5],'22 / 23');assert.equal(repeats[2][5],'unreported');
  assert.ok(node('heightTiltTraceCaption').textContent.includes('2 saved samples'));assert.equal(node('heightTiltTraceSelect').disabled,false);
  c.report.neighborhood_evaluation.trace_examples=[];c.showHeightTiltStudy();assert.equal(node('heightTiltTraceSelect').disabled,true);assert.ok(node('heightTiltTraceCaption').textContent.includes('no predicted or ideal-flight path'));
  const {study,recipe}=repeatFixture(),threshold=repeatFixture(18).recipe;threshold.id='threshold';
  Object.assign(c.report.neighborhood_evaluation,study,{selected_recipes:[recipe,threshold]});c.showHeightTiltStudy();
  assert.ok(node('heightTiltConclusion').textContent.includes('1 / 2 selected recipes captured 23/23'));
  assert.ok(node('heightTiltConclusion').textContent.includes('2 / 2 meet 3/3 nominal'));
  assert.equal(tables.get(node('heightTiltRecipes')).rows[0][4],'Perfect 23/23 each');assert.equal(tables.get(node('heightTiltRecipes')).rows[1][4],'Meets repeat threshold');
  assert.equal(tables.get(node('heightTiltSensitivity')).rows[0][6],'Passed');
  c.report.neighborhood_evaluation.audit.contract_fingerprint='other';c.showHeightTiltStudy();assert.equal(tables.get(node('heightTiltSensitivity')).rows[0][6],'Unverified');assert.equal(tables.get(node('heightTiltRecipes')).rows[0][4],'Unverified');
});
