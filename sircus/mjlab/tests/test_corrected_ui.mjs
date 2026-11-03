import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {correctedGridProgress,correctedGridRows,correctedTrainingReadiness,studyReady} from '../web/state.mjs';

test('corrected study remains optional and never changes active inference readiness',()=>{
  assert.equal(correctedGridProgress({}),null);assert.deepEqual(correctedGridRows({}),[]);
  const report={status:'completed',frozen_evaluation:{captures:1},corrected_grid_evaluation:{status:'waiting',cells:[],counts:{geometry_requests:0}}};
  assert.equal(studyReady(report),true);
  const view=correctedGridProgress(report);
  assert.equal(view.geometryRequests,0);assert.equal(view.nativeAttempts,null);assert.equal(view.qualified,null);
});
test('historical queue metadata does not claim an active native run is waiting',()=>{
  const value={status:'running',phase:'confirmation',queue:{status:'waiting_for_4090'}};
  assert.equal(correctedGridProgress({corrected_grid_evaluation:value}).queue,null);
  value.status='waiting';value.phase='waiting_for_4090';
  assert.equal(correctedGridProgress({corrected_grid_evaluation:value}).queue.status,'waiting_for_4090');
});
test('pending, endpoint, native capture and qualified motion remain distinct',()=>{
  const cells=[
    {id:'pending',bucket_xy_m:[.1,-.65]},
    {id:'endpoint',bucket_xy_m:[.2,-.65],geometry:{eligible:3},search:{attempts:0,captures:0}},
    {id:'capture-only',bucket_xy_m:[.3,-.65],search:{attempts:3,captures:3},confirmations:[]},
    {id:'qualified',bucket_xy_m:[.4,-.65],search:{attempts:3,captures:3},motion:{connected:true},confirmations:[{qualified:true}]},
    {id:'miss',bucket_xy_m:[.5,-.65],search:{attempts:3,captures:0}},
  ];
  const rows=correctedGridRows({corrected_grid_evaluation:{cells}});
  assert.deepEqual(rows.map(r=>r.state),['pending','endpoint','captured','qualified','missed']);
  assert.equal(rows[2].qualified,false);assert.equal(rows[2].motion,false);assert.equal(rows[3].motion,true);
});
test('new grid has no request handler and preserves the original solve endpoints',()=>{
  const js=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
  const section=js.slice(js.indexOf('function showCorrectedGrid()'),js.indexOf('function showContactStudy()'));
  assert.ok(!section.includes('fetch('));assert.ok(!section.includes('/api/solve'));
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
  for(const id of ['correctedGrid','correctedGridMap','correctedGridStatus','correctedGridCells']) assert.ok(html.includes(`id="${id}"`));
  assert.ok(html.includes('does not measure continuous state-space coverage'));
  assert.ok(js.includes("fetch('/api/solve'"));
});
test('training assessment needs completed matched audits and never launches a learner',()=>{
  const value={status:'completed',counts:{qualified_positions:12,qualified_settings:17,positions_with_multiple_qualified:5},
    cells:Array.from({length:35},(_,i)=>({id:String(i),bucket_xy_m:[.1+i*.01,0]})),verification:{status:'passed'},
    audits:{native:{status:'passed'},supplemental:{status:'pending'}}};
  const report={corrected_grid_evaluation:value};
  assert.equal(correctedTrainingReadiness(report).title,'Study finished; audit clearance required');
  value.audits.supplemental.status='passed';
  const view=correctedTrainingReadiness(report);
  assert.equal(view.title,'Audited evidence ready for a limited pilot');
  assert.equal(view.readyToLaunch,false);assert.ok(view.text.includes('17 settings at 12/35'));assert.ok(view.text.includes('5 positions'));
  value.status='running';assert.equal(correctedTrainingReadiness(report).title,'Training decision pending');
  value.status='completed';value.counts.qualified_positions=0;
  assert.equal(correctedTrainingReadiness(report).title,'More feasible settings needed');
});
