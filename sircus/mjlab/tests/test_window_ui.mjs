import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
const code=source.slice(source.indexOf('// BEGIN READ-ONLY WINDOW STUDY'),source.indexOf('// END READ-ONLY WINDOW STUDY'));
const context=vm.createContext({finiteVector:(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite)});
vm.runInContext(code,context);

test('window view remains optional and native counts remain unknown unless observed',()=>{
  assert.equal(context.windowStudyView(null),null);
  const stats=context.windowStatistics({attempts:20,captures:13,invalid:2});
  assert.equal(stats.requests,20);assert.equal(stats.released,null);assert.equal(stats.fraction,'13 / 20');
  assert.equal(context.windowStatistics({attempts:20,captures:30}).rate,null);
  assert.equal(context.windowStatistics({attempts:0,captures:0}).rate,null);
});
test('context splits are explicitly disjoint height/tilt identities, never new bucket positions',()=>{
  const groups={train:[{id:'train',board_height_m:.3,tilt_deg:24}],validation:[{id:'val',board_height_m:.35,tilt_deg:24}],test:[{id:'test',board_height_m:.4,tilt_deg:24}]};
  assert.equal(context.windowStudyView({settings:{contexts:groups}}).disjoint,true);
  groups.test[0].board_height_m=.3;
  assert.equal(context.windowStudyView({settings:{contexts:groups}}).disjoint,false);
});
test('native trajectory view projects the actual stored points and exposes lateral deviation',()=>{
  const study={settings:{board_xy_m:[0,0],bucket_xy_m:[0,-.6]}};
  const trace={position_m:[[0,0,.7],[.02,-.2,.3],[.01,-.6,.1]]};
  const view=context.windowTrajectoryView(study,trace);
  assert.equal(view.distance,.6);assert.equal(view.points.length,3);
  assert.equal(view.points[1][0],.2);assert.equal(view.points[1][1],.3);assert.equal(view.lateral,.02);
  assert.equal(context.windowTrajectoryView(study,{position_m:[[0,0,NaN],[0,0,0]]}),null);
});
test('read-only report section cannot solve, mutate a model or alter existing inference controls',()=>{
  assert.ok(!code.includes('fetch('));assert.ok(!code.includes('/api/solve'));assert.ok(!code.includes('syncSolveButtons'));
  assert.ok(source.includes("fetch('/api/solve'"));assert.ok(source.includes('showCorrectedGrid(); showWindowStudy();'));
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
  for(const id of ['windowStudy','windowLearningChart','windowComparison','windowTestContexts','windowWindows','windowTraceSelect','windowTrace'])assert.ok(html.includes(`id="${id}"`));
  for(const text of ['uncalibrated','no outcome search','not an ideal ballistic','does not activate a checkpoint','not new bucket positions'])assert.ok(html.includes(text));
});
test('partial and completed reports render without inferred results or changed denominators',()=>{
  const nodes=new Map(),tables=new Map();
  const drawing=new Proxy({}, {get:(target,key)=>target[key]??(()=>{}),set:(target,key,value)=>(target[key]=value,true)});
  const element=()=>({value:'',width:820,height:330,getContext:()=>drawing,replaceChildren(...children){this.children=children;}});
  context.$=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
  context.document={createElement:element};context.addMetrics=()=>{};
  context.expansionTable=(target,headings,rows)=>tables.set(target,rows);
  context.report={window_study_evaluation:{status:'running',phase:'seed'}};
  assert.doesNotThrow(()=>context.showWindowStudy());
  assert.equal(nodes.get('windowStudy').hidden,false);assert.equal(nodes.get('windowTraceSelect').disabled,true);
  context.report.window_study_evaluation={status:'completed',settings:{board_xy_m:[0,0],bucket_xy_m:[0,-.6]},
    test:{sac:{attempts:20,captures:13,native_releases:18}},trace_examples:[{context_id:'test',gap_m:.3,route_success:true,position_m:[[0,0,.7],[0,-.6,.1]]}]};
  assert.doesNotThrow(()=>context.showWindowStudy());
  const row=tables.get(nodes.get('windowComparison'))[0];
  assert.equal(row[1],'13 / 20');assert.equal(row[2],'18 / 20');
  assert.equal(nodes.get('windowTraceSelect').disabled,false);
  assert.ok(nodes.get('windowTraceCaption').textContent.includes('2 saved samples'));
});
