import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { sweepRlSummary, sweepRlFinalRows, sweepRlAdmissionRows, sweepRlHistory, sweepRlResources, sweepRlCurves, sweepRlCsv, solveButtonState, historicalPlannerIsPreV2 } from '../web/state.mjs';

const hash = 'a'.repeat(64);
const counts = { attempts: 72, captures: 60, native_releases: 70, invalid: 2, clean: 55 };
const fixture = extra => ({ benchmark_read_only: true, sweep_rl_benchmark: {
  version: 'sircus-calibrated-sweep-rl-benchmark-v2', status: 'completed', test_disclosure: 'verified',
  audit: { status: 'passed', synthetic_fixture: false }, contract_fingerprint: hash,
  protocol: { methods: [{ id: 'sweep', label: 'Guided sweep' }, { id: 'ppo', label: 'Residual PPO' }], seeds: [42], sample_budget: 1024 },
  runs: [{ method: 'ppo', seed: 42, status: 'completed', train_requests: 1024, selected_samples: 256, checkpoint_sha256: hash }],
  final_test: [{ method: 'ppo', seed: 42, counts, frozen_artifact_sha256: hash, query_s: .01,
    context_resolution_timing: { wall_s: 12, geometry_s: 9, path_s: 3 }, pre_prediction_plus_policy_s: 12.01 }], ...extra } });

test('v2 TEST requires completed and byte-verified native audit; v1 behavior stays separate', () => {
  const r = fixture();
  assert.equal(sweepRlSummary(r).calibrated, true); assert.equal(sweepRlFinalRows(r).length, 1);
  for (const extra of [{status:'running'}, {test_disclosure:'hidden_until_completed_and_audited'},
    {audit:{status:'pending',synthetic_fixture:false}}, {audit:{status:'passed',synthetic_fixture:true}}, {audit:{status:'passed'}}]) {
    const r = fixture(extra); assert.deepEqual(sweepRlFinalRows(r), []);
    assert.ok(!sweepRlCsv(r).includes('"test"'));
  }
  const old = fixture({version:'sircus-sweep-rl-benchmark-v1',audit:undefined,test_disclosure:undefined});
  assert.equal(sweepRlSummary(old).calibrated,false);assert.equal(sweepRlFinalRows(old).length,1);
});

test('pre-policy geometry time is distinct from prediction/preparation and absent costs stay unknown', () => {
  const row=sweepRlFinalRows(fixture())[0];
  assert.equal(row.queryS,.01);assert.equal(row.contextResolutionS,12);assert.equal(row.contextGeometryS,9);
  assert.equal(row.contextPathS,3);assert.equal(row.prePredictionPlusPolicyS,12.01);
  assert.equal(sweepRlResources({path_s:0}).path_s,0);assert.equal(sweepRlResources({}).path_s,null);
  const r=fixture({curves:[{method:'ppo',seed:42,samples:32,validation:counts,resources:{path_s:4}}]});
  assert.equal(sweepRlCurves(r)[1].points[0].pathS,4);
  const csv=sweepRlCsv(r), lines=csv.trimEnd().split('\n');
  assert.ok(csv.includes('test_prepolicy_geometry_wall_s'));assert.ok(csv.includes('sampled_path_wall_s'));
  for (const line of lines) assert.equal(line.split(',').length,lines[0].split(',').length);
});

const worker = extra => ({method:'ppo',seed:42,admission:{requested:32,native_releases:30,rejected:2,
  fixed_base_rejected:1,anchor_unresolved:3,invalid_physics:1,reasons:{fixed_base_clearance_rejected:1,finite_ik_unresolved:1},...extra}});
test('TRAIN admission keeps native numerical failures separate from setup rejections', () => {
  const r=fixture({runs:[worker()]});const before=JSON.stringify(r),row=sweepRlAdmissionRows(r)[0];
  assert.equal(row.requested,32);assert.equal(row.native,30);assert.equal(row.rejected,2);assert.equal(row.invalidPhysics,1);
  assert.equal(row.reasons.reduce((sum,[,n])=>sum+n,0),2);assert.equal(JSON.stringify(r),before);
  assert.deepEqual(sweepRlAdmissionRows(fixture({runs:[worker({rejected:3})]})),[]);
  assert.deepEqual(sweepRlAdmissionRows(fixture({runs:[worker({reasons:{}})]})),[]);
});

test('admission drops undeclared, overbudget and conflicting counters instead of double counting', () => {
  const w=worker();assert.equal(sweepRlAdmissionRows(fixture({runs:[w,structuredClone(w)]})).length,1);
  assert.deepEqual(sweepRlAdmissionRows(fixture({runs:[w,worker({invalid_physics:2})]})),[]);
  assert.deepEqual(sweepRlAdmissionRows(fixture({runs:[{...w,seed:43}]})),[]);
  assert.deepEqual(sweepRlAdmissionRows(fixture({runs:[worker({requested:2048,native_releases:2046})]})),[]);
});

test('historical v1 curves cannot be pooled into calibrated v2 results', () => {
  const r=fixture({curves:[]});r.historical_sweep_rl_benchmark={version:'sircus-sweep-rl-benchmark-v1',historical:true,curves:[{captures:99}]};
  assert.equal(sweepRlHistory(r).version,'sircus-sweep-rl-benchmark-v1');
  assert.ok(sweepRlCurves(r).every(c=>c.points.length===0));
  r.historical_sweep_rl_benchmark.historical=false;assert.equal(sweepRlHistory(r),null);
});

test('read-only v2 view disables solves even when an inherited planner is ready', () => {
  const state=solveButtonState({...fixture(),planner:{ready:true}},false,'search');
  assert.equal(state.disabled,true);assert.match(state.mainLabel,/Read-only/);
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
  const js=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
  for (const id of ['benchmarkRule','benchmarkMetricNote','benchmarkResourceNote','benchmarkQueryNote','benchmarkAdmissionTable','benchmarkHistoryTable']) {
    assert.equal(html.split(`id="${id}"`).length-1,1);assert.ok(js.includes(id));
  }
  assert.ok(js.includes('TEST results remain hidden until the full benchmark completes'));
  assert.ok(html.includes('different scoring/setup rules and different evaluation coordinates'));
});

test('calibrated read-only view labels inherited farm/grid as historical without changing their evidence', () => {
  const report=fixture();report.planner_grid_study={qualified:17,blocked:6};
  const original=JSON.stringify(report);
  assert.equal(historicalPlannerIsPreV2(report),true);
  assert.equal(historicalPlannerIsPreV2({...report,benchmark_read_only:false}),false);
  assert.equal(historicalPlannerIsPreV2(fixture({version:'sircus-sweep-rl-benchmark-v1'})),false);
  assert.equal(historicalPlannerIsPreV2({}),false);
  assert.equal(JSON.stringify(report),original);
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
  const js=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
  assert.ok(html.includes('Capture rate as samples accumulate'));
  assert.ok(!html.includes('Strict capture rate as samples accumulate'));
  assert.ok(html.indexOf('id="plannerHistoricalNotice"')<html.indexOf('id="plannerFarm"'));
  assert.ok(html.indexOf('id="plannerHistoricalNotice"')<html.indexOf('id="plannerGridStudy"'));
  assert.ok(html.includes('earlier scoring and setup rules'));
  assert.ok(html.includes('do not report calibrated-v2 admission or capture performance'));
  assert.ok(js.includes("$('plannerHistoricalNotice').hidden = !historicalPlanner"));
  assert.ok(js.includes('Historical farm targets · pre-v2'));
  assert.ok(js.includes('Historical shot settings and repeats · pre-v2'));
});
