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
test('prepared-board coverage distinguishes resolved, unresolved and unknown evidence',()=>{
  const make=(id,h)=>({id,board_height_m:h,tilt_deg:24});
  const groups={train:[make('a',.2),make('b',.3)],validation:[make('c',.4)],test:[make('d',.5)]};
  const study={version:'sircus-fixed-bucket-gap-study-v1',settings:{version:'sircus-fixed-bucket-gap-study-v1',contexts:groups},
    prepared_contexts:[{...groups.train[0],reason:null},{...groups.train[1],reason:'board_ik_unresolved'},
      {...groups.validation[0],prepared:true,board_admitted:true},{...groups.test[0]}]};
  let value=context.windowBoardCoverage(study);
  assert.equal(value.all.resolved,2);assert.equal(value.all.unresolved,1);assert.equal(value.all.unknown,1);
  assert.equal(value.train.resolved,1);assert.equal(value.validation.resolved,1);assert.equal(value.test.unknown,1);
  study.version='unrecognized';value=context.windowBoardCoverage(study);
  assert.equal(value.all.resolved,1);assert.equal(value.all.unknown,3);
  study.prepared_contexts.push({...groups.validation[0],board_admitted:true});
  assert.equal(context.windowBoardCoverage(study).validation.unknown,1);
});
test('board coverage never admits missing, mismatched or explicitly unprepared rows',()=>{
  const row={id:'a',board_height_m:.2,tilt_deg:24};
  const study={settings:{contexts:{train:[row]}},prepared_contexts:[{...row,prepared:false,board_admitted:true}]};
  assert.equal(context.windowBoardCoverage(study).all.unknown,1);
  study.prepared_contexts=[{...row,board_height_m:.3,board_admitted:true}];
  assert.equal(context.windowBoardCoverage(study).all.unknown,1);
  study.prepared_contexts={train:[{...row,prepared:true,board_admitted:false}]};
  assert.equal(context.windowBoardCoverage(study).all.unresolved,1);
});
test('separate recipe repeats require the exact parent fingerprint',()=>{
  assert.equal(context.windowConfirmationView({}),null);
  const supplement={parent_contract_fingerprint:'a',status:'completed',recipes:[{id:'train-recipe'}]};
  assert.equal(context.windowConfirmationView({contract_fingerprint:'a',window_confirmations:supplement}).recipes.length,1);
  const bad=context.windowConfirmationView({contract_fingerprint:'b',window_confirmations:supplement});
  assert.equal(bad.status,'failed');assert.equal(bad.recipes.length,0);
  assert.equal(context.windowConfirmationView({window_confirmations:{recipes:[]}}).status,'failed');
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
  assert.equal(nodes.get('windowConfirmations').hidden,true);
  context.report.window_study_evaluation.contract_fingerprint='a';
  context.report.window_study_evaluation.window_confirmations={parent_contract_fingerprint:'a',status:'completed',recipes:[{
    id:'recipe',context_id:'train',nominal:{attempts:3,captures:3},fresh:{attempts:20,captures:10},total:{attempts:23,captures:13,native_releases:20,clean:9}}]};
  context.showWindowStudy();const repeat=tables.get(nodes.get('windowConfirmationTable'))[0];
  assert.equal(nodes.get('windowConfirmations').hidden,false);
  assert.equal(repeat[6],'13 / 23');assert.equal(repeat[7],'20 / 23');assert.equal(repeat[8],'9 / 23');
  assert.equal(tables.get(nodes.get('windowComparison'))[0][1],'13 / 20');
});
