import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
const section=name=>source.slice(source.indexOf(`// BEGIN READ-ONLY ${name}`),source.indexOf(`// END READ-ONLY ${name}`));
const code=section('DISTANCE STUDY'),make=extra=>{const c=vm.createContext({finiteVector:(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite),...extra});vm.runInContext(section('WINDOW STUDY')+section('HEIGHT TILT STUDY')+code,c);return c;};
function fixture(){
  const far=[.8*Math.cos(-Math.PI/12),.8*Math.sin(-Math.PI/12)];
  const study={version:'sircus-bucket-distance-development-v1',status:'completed',phase:'completed',contract_fingerprint:'same',audit:{status:'passed',contract_fingerprint:'same'},
    settings:{board_xy_m:[0,0],bucket_xy_m:[99,99],table_bounds_xy_m:[[-.5,.6],[-1,1]],bucket_outer_radius_m:.1,table_bucket_margin_m:.005,rays:[{id:'original',label:'Original ray'},{id:'inward',label:'Inward corner ray'}]},
    distances:[{id:'near',ray_id:'original',distance_m:.35,bucket_xy_m:[.35,0]},{id:'far',ray_id:'inward',distance_m:.8,bucket_xy_m:far}],
    contexts:[{id:'ctx',distance_id:'near',board_admitted:true,geometry_requests:17,eligible:10,search_attempts:17,search_native_releases:10,search_captures:7},
      {id:'far-ctx',distance_id:'far',board_admitted:null,geometry_requests:0,eligible:0,search_attempts:0,search_native_releases:0,search_captures:0}],
    selected_recipes:[],trace_examples:[{record_id:'actual-near',distance_id:'near',substeps:64,route_success:true,position_m:[[0,0,.8],[.35,0,.1]]},
      {record_id:'actual-far',distance_id:'far',substeps:128,route_success:false,position_m:[[0,0,.8],[...far,.1]]}]};
  const recipe={id:'recipe',distance_id:'near',ray_id:'original',bucket_xy_m:[.35,0],distance_m:.35,board_height_m:.4,tilt_deg:32,gap_m:.3,release_center_height_m:.7375,
    nominal_by_substeps:{},fresh_by_substeps:{},by_substeps:{},sensitivity:{paired_engineering_gate:true,all_resolution_repeat_gate:true,capture_agreement_64_128:33,paired_attempts:33,nominal_position_delta_350ms_max_m:.002}};
  for(const s of ['32','64','128']){recipe.nominal_by_substeps[s]={attempts:3,captures:3,native_releases:3};recipe.fresh_by_substeps[s]={attempts:30,captures:30,native_releases:30};recipe.by_substeps[s]={attempts:33,captures:33,native_releases:33,clean:0};}
  study.selected_recipes=[recipe];return study;
}

test('distance panel is optional and separates ray identities and search denominators',()=>{
  const c=make();assert.equal(c.distanceStudyView(null),null);const study=fixture(),v=c.distanceStudyView(study);
  assert.deepEqual(Array.from(v.rays),['original','inward']);const s=c.distanceSearchSummary(v,v.distances[0]);
  assert.equal(s.total,1);assert.equal(s.resolved,1);assert.equal(s.geometry,17);assert.equal(s.eligible,10);assert.equal(s.requests,17);assert.equal(s.native,10);assert.equal(s.captures,7);
  const queued=c.distanceSearchSummary(v,v.distances[1]);assert.equal(queued.unknown,1);assert.equal(queued.native,0);
  delete study.contexts[0].search_native_releases;assert.equal(c.distanceSearchSummary(c.distanceStudyView(study),v.distances[0]).native,null);
});
test('outer footprint uses table bounds and radius, with missing dimensions left unknown',()=>{
  const c=make(),study=fixture();let foot=c.distanceFootprint(study,study.distances[0]);assert.equal(foot.known,true);assert.equal(foot.onTable,true);assert.ok(Math.abs(foot.margin-.15)<1e-12);
  assert.equal(foot.admitted,true);foot=c.distanceFootprint(study,{bucket_xy_m:[.497,0]});assert.equal(foot.onTable,true);assert.equal(foot.admitted,false);
  assert.equal(c.distanceFootprint(study,study.distances[1]).onTable,false);delete study.settings.bucket_outer_radius_m;assert.equal(c.distanceFootprint(study,study.distances[0]).known,false);
  study.settings.bucket_outer_top_diameter_m=.2;assert.equal(c.distanceFootprint(study,study.distances[0]).onTable,true);
  delete study.settings.table_bounds_xy_m;assert.equal(c.distanceFootprint(study,study.distances[0]).onTable,null);
});
test('saved native trace projection follows its own bucket and rejects ambiguous association',()=>{
  const c=make(),study=fixture();const near=c.distanceTrajectoryView(study,study.trace_examples[0]),far=c.distanceTrajectoryView(study,study.trace_examples[1]);
  assert.equal(near.distance,.35);assert.ok(Math.abs(far.distance-.8)<1e-12);assert.ok(Math.abs(far.points[1][0]-.8)<1e-12);assert.ok(far.lateral<1e-12);
  assert.notDeepEqual(Array.from(near.bucket),Array.from(far.bucket));
  assert.equal(c.distanceTrajectoryView(study,{position_m:[[0,0,.7],[.3,0,.1]]}),null);
  assert.equal(c.distanceTrajectoryView(study,{...study.trace_examples[1],bucket_xy_m:[.35,0]}),null);
  assert.equal(c.distanceTrajectoryView(study,{...study.trace_examples[1],distance_m:.35}),null);
  assert.equal(c.distanceTrajectoryView(study,{...study.trace_examples[1],position_m:[[0,0,NaN],[0,0,0]]}),null);
});
test('flight references use this far bucket and only linked configured board geometry',()=>{
  const calls=[],nodes=new Map();let dash=[];
  const draw=new Proxy({setLineDash:value=>{dash=Array.from(value);},moveTo:(...args)=>calls.push({method:'moveTo',args,dash:Array.from(dash)}),fillText:(...args)=>calls.push({method:'fillText',args})},{get:(o,k)=>o[k]??(()=>{})});
  const node=id=>{if(!nodes.has(id))nodes.set(id,{width:820,height:260,textContent:'',getContext:()=>draw});return nodes.get(id);};
  const c=make({$:node}),study=fixture(),trace=study.trace_examples[1];trace.context_id='far-ctx';
  Object.assign(study.contexts[1],{board_height_m:.45,bucket_xy_m:study.distances[1].bucket_xy_m,board_xy_m:[0,0]});
  const original=JSON.stringify(trace.position_m);c.drawDistanceTrace(study,1);
  const labels=calls.filter(r=>r.method==='fillText').map(r=>r.args[0]);
  assert.ok(labels.includes('Bucket center axis (800.0 mm)'));assert.ok(labels.includes('Tabletop reference (z = 0)'));assert.ok(labels.includes('Configured board center'));
  const bucketX=55+(.8+.1)/(.95+.1)*(820-80);assert.ok(calls.some(r=>r.method==='moveTo'&&r.dash.length>0&&Math.abs(r.args[0]-bucketX)<1e-9));
  assert.equal(JSON.stringify(trace.position_m),original);assert.ok(node('distanceTraceCaption').textContent.includes('configured geometry, not a native observation'));
  calls.length=0;delete study.contexts[1].board_height_m;study.settings.board_height_m=99;c.drawDistanceTrace(study,1);
  assert.ok(!calls.some(r=>r.method==='fillText'&&r.args[0]==='Configured board center'));assert.ok(node('distanceTraceCaption').textContent.includes('height unknown; marker omitted'));
  study.contexts[1].board_height_m=.45;study.contexts[1].distance_id='near';calls.length=0;c.drawDistanceTrace(study,1);
  assert.ok(!calls.some(r=>r.method==='fillText'&&r.args[0]==='Configured board center'));
});
test('distance repeat qualification requires its own matching completed audit and all three resolutions',()=>{
  const c=make();let study=fixture(),r=study.selected_recipes[0],check=c.distanceRepeatCheck(study,r);assert.equal(check.perfect,true);assert.equal(check.qualified,true);
  for(const corrupt of [d=>d.audit.status='pending',d=>d.audit.contract_fingerprint='old',d=>d.contract_fingerprint='',d=>d.status='running',d=>d.version='unknown',d=>delete d.selected_recipes[0].fresh_by_substeps['32']]){
    study=fixture();corrupt(study);assert.equal(c.distanceRepeatCheck(study,study.selected_recipes[0]).qualified,null);
  }
  study=fixture();r=study.selected_recipes[0];r.fresh_by_substeps['32'].captures=26;r.by_substeps['32'].captures=29;r.sensitivity.all_resolution_repeat_gate=false;check=c.distanceRepeatCheck(study,r);assert.equal(check.thresholds,false);assert.equal(check.qualified,false);
  r.sensitivity.all_resolution_repeat_gate=true;assert.equal(c.distanceRepeatCheck(study,r).qualified,null);
  r.fresh_by_substeps['32'].captures=27;r.by_substeps['32'].captures=30;assert.equal(c.distanceRepeatCheck(study,r).qualified,true);
});
test('new distance evidence is read-only, responsive and preserves previous handlers and panels',()=>{
  for(const forbidden of ['fetch(', '/api/solve','startTraining','activateCheckpoint','syncSolveButtons'])assert.ok(!code.includes(forbidden));
  assert.ok(source.includes('showCorrectedGrid(); showWindowStudy(); showHeightTiltStudy(); showDistanceStudy();'));
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8'),css=readFileSync(new URL('../web/style.css',import.meta.url),'utf8');
  for(const id of ['distanceStudy','distanceStatus','distanceRays','distanceRecipes','distanceLayout','distanceRepeats','distanceTrace'])assert.ok(html.includes(`id="${id}"`));
  for(const text of ['No new RL training','full-arm trajectory validation','reported height 175 mm','no 32-substep positional convergence claim'])assert.ok(html.includes(text));
  assert.match(css,/\.distance-section\{[^}]*min-width:0;max-width:100%/);assert.ok(css.includes('.distance-section .table-wrap{max-width:100%}'));
});
test('DOM rendering exposes queued, failed and completed evidence with exact repeat counts',()=>{
  const nodes=new Map(),tables=new Map(),draw=new Proxy({},{get:(o,k)=>o[k]??(()=>{})});
  const element=()=>({children:[],value:'0',width:820,height:340,hidden:false,textContent:'',append(...x){this.children.push(...x);},replaceChildren(...x){this.children=x;},setAttribute(){},getContext(){return draw;}});
  const node=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
  const c=make({$:node,document:{createElement:element},report:{},addMetrics:(rows,target)=>target.metrics=rows,expansionTable:(target,headers,rows)=>tables.set(target,{headers,rows})});
  c.showDistanceStudy();assert.equal(node('distanceStudy').hidden,true);
  const study=fixture();c.report.distance_evaluation=study;c.showDistanceStudy();assert.equal(node('distanceStudy').hidden,false);assert.equal(node('distanceRays').children.length,2);
  const table=node('distanceRays').children[0].children[1],near=tables.get(table).rows[0];assert.equal(near[4],'10 / 17');assert.equal(near[5],'10 / 17');assert.equal(near[6],'7 / 17');
  assert.equal(tables.get(node('distanceRepeats')).rows.length,3);assert.equal(tables.get(node('distanceRepeats')).rows[0][3],'30 / 30');assert.equal(tables.get(node('distanceRepeats')).rows[0][5],'33 / 33');
  assert.equal(tables.get(node('distanceRecipes')).rows[0][4],'Perfect 33/33 each');assert.ok(node('distanceConclusion').textContent.includes('1 / 1 selected recipes'));
  assert.equal(tables.get(node('distanceSensitivity')).rows[0][4],'Passed');
  node('distanceTraceSelect').value='1';node('distanceTraceSelect').onchange();assert.ok(node('distanceTraceCaption').textContent.includes('800.0 mm'));assert.ok(node('distanceTraceCaption').textContent.includes('actual-far'));
  study.status='waiting_for_4090';study.phase='device_queue';study.queue={reason:'GPU lock held',wait_s:120};c.showDistanceStudy();assert.ok(node('distanceStatus').textContent.includes('2 min queued'));assert.ok(node('distanceConclusion').textContent.includes('awaits'));assert.equal(tables.get(node('distanceSensitivity')).rows[0][3],'Unverified');
  study.status='failed';study.error='native preflight stopped';study.trace_examples=[];c.showDistanceStudy();assert.ok(node('distanceStatus').textContent.includes('native preflight stopped'));assert.equal(node('distanceTraceSelect').disabled,true);assert.ok(node('distanceTraceCaption').textContent.includes('no ideal flight'));
});
