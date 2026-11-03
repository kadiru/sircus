import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
const code=source.slice(source.indexOf('// BEGIN READ-ONLY HEIGHT TILT STUDY'),source.indexOf('// END READ-ONLY HEIGHT TILT STUDY'));
const windowCode=source.slice(source.indexOf('// BEGIN READ-ONLY WINDOW STUDY'),source.indexOf('// END READ-ONLY WINDOW STUDY'));
const make=extra=>{const c=vm.createContext({finiteVector:(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite),...extra});vm.runInContext(windowCode+'\n'+code,c);return c;};

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
test('timestep agreement does not invent a denominator or assume convergence',()=>{
  const c=make();assert.equal(c.neighborhoodAgreement({agreements:22,attempts:23}),'22 / 23');
  assert.equal(c.neighborhoodAgreement(22),'22 matched');assert.equal(c.neighborhoodAgreement(22,23),'22 / 23');assert.equal(c.neighborhoodAgreement(.95),'unreported');
  assert.equal(c.neighborhoodAgreement({agreements:24,attempts:23}),'unreported');
});
test('readonly panel has no solve, training, network or policy mutation hooks',()=>{
  for(const text of ['fetch(', '/api/solve','syncSolveButtons','activateCheckpoint','startTraining'])assert.ok(!code.includes(text));
  assert.ok(source.includes("fetch('/api/solve'"));assert.ok(source.includes('showCorrectedGrid(); showWindowStudy(); showHeightTiltStudy();'));
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
  for(const id of ['heightTiltStudy','heightTiltMaps','heightTiltRecipes','heightTiltContexts','heightTiltSensitivity','heightTiltRepeats','heightTiltTraceSelect','heightTiltTrace'])assert.ok(html.includes(`id="${id}"`));
  for(const text of ['No RL training','No RL training or new held-out evaluation','reported 175 mm','does not activate a policy','not a certified continuous range'])assert.ok(html.includes(text));
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
  c.report.neighborhood_evaluation={status:'running',phase:'search',contexts:[{id:'a',source_family:'anchor-a',board_height_m:.4,tilt_deg:32,board_admitted:true,geometry_requests:17,eligible:10,search_attempts:17,search_native_releases:10,search_captures:7,best_gap_m:.3}],
    settings:{board_xy_m:[.5,-.3],bucket_xy_m:[.1,-.6]},counts:{requests:17,native_releases:10},selected_recipes:[{id:'recipe',context_id:'a',board_height_m:.4,tilt_deg:32,gap_m:.3,release_center_height_m:.7375,distance_m:.5,
      by_substeps:{'32':{attempts:23,captures:13,native_releases:20,clean:3},'64':{attempts:23,captures:21,native_releases:22,clean:5},'128':{attempts:23,captures:20}},
      nominal_by_substeps:{'64':{attempts:3,captures:3}},fresh_by_substeps:{'64':{attempts:20,captures:18}},sensitivity:{capture_agreement_64_128:{agreements:22,attempts:23},nominal_position_delta_350ms_max_m:.004,paired_engineering_gate:false}}],
    trace_examples:[{record_id:'actual',context_id:'a',substeps:128,route_success:false,position_m:[[.5,-.3,.7],[.2,-.6,.2]]}]};
  c.showHeightTiltStudy();assert.equal(node('heightTiltStudy').hidden,false);assert.equal(node('heightTiltMaps').children.length,1);
  const searched=tables.get(node('heightTiltContexts')).rows[0];assert.equal(searched[3],'10 / 17');assert.equal(searched[4],'10 / 17');assert.equal(searched[5],'7 / 17');assert.equal(searched[6],'300.0');
  const sensitivity=tables.get(node('heightTiltSensitivity')).rows[0];assert.deepEqual(Array.from(sensitivity.slice(1,4)),['13 / 23','21 / 23','20 / 23']);assert.equal(sensitivity[4],'22 / 23');assert.equal(sensitivity[5],'4.0');assert.equal(sensitivity[6],'Failed');
  const repeats=tables.get(node('heightTiltRepeats')).rows;assert.equal(repeats[1][2],'3 / 3');assert.equal(repeats[1][3],'18 / 20');assert.equal(repeats[1][5],'22 / 23');assert.equal(repeats[2][5],'unreported');
  assert.ok(node('heightTiltTraceCaption').textContent.includes('2 saved samples'));assert.equal(node('heightTiltTraceSelect').disabled,false);
  c.report.neighborhood_evaluation.trace_examples=[];c.showHeightTiltStudy();assert.equal(node('heightTiltTraceSelect').disabled,true);assert.ok(node('heightTiltTraceCaption').textContent.includes('no predicted or ideal-flight path'));
});
