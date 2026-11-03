import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { sweepRlStudy, sweepRlSummary, sweepRlPoints, sweepRlCurves, sweepRlFinalRows, sweepRlResources, sweepRlCsv } from '../web/state.mjs';

const sha = 'a'.repeat(64);
const counts = (captures = 12, attempts = 36) => ({ captures, attempts, native_releases: attempts - 2, invalid: 2, clean: Math.min(10, captures), mean_reward: .4 });
const resources = extra => ({ wall_s: 20, cpu_s: 8, geometry_s: 5, gpu_s: 10, peak_rss_mb: 120, peak_vram_mb: 700, torch_allocated_mb: 30, torch_reserved_mb: 40, setup_s: 2, evaluation_wall_s: 4, ...extra });
const point = extra => ({ method: 'ppo', seed: 42, samples: 256, validation: counts(), resources: resources(), artifact_sha256: sha, ...extra });
const fixture = extra => ({ sweep_rl_benchmark: { version: 'sircus-sweep-rl-benchmark-v1', status: 'running', phase: 'training', contract_fingerprint: sha,
  protocol: { methods: [{ id: 'sweep', label: 'Guided sweep' }, { id: 'ppo', label: 'Residual PPO' }], seeds: [42, 43, 44], milestones: [0, 256, 512, 1024], sample_budget: 1024,
    prior: { recipes: 17, latest_study_native_releases: 4234, full_historical_cost_known: false, scope: 'Shared prior; earlier acquisition cost unknown' } }, curves: [], runs: [], final_test: [], ...extra } });

test('missing benchmark and pending measurements stay distinct from observed zero', () => {
  assert.equal(sweepRlStudy({}), null); assert.equal(sweepRlSummary({}), null);
  const report = fixture(), summary = sweepRlSummary(report);
  assert.equal(summary.budget, 1024); assert.equal(summary.seedCount, 3); assert.equal(summary.totalRuns, 6);
  assert.equal(summary.curvePoints, 0); assert.equal(summary.finalRows, 0);
  assert.ok(sweepRlCurves(report).every(row => row.points.length === 0));
  const observed = fixture({ curves: [point({ samples: 0, validation: counts(0), resources: resources({ wall_s: 0 }) })] });
  const value = sweepRlCurves(observed)[1].points[0];
  assert.equal(value.samples, 0); assert.equal(value.rate, 0); assert.equal(value.wallS, 0);
});

test('curves retain all validation requests, pool seed counts and expose seed range', () => {
  const report = fixture({ curves: [point(), point({ seed: 43, validation: counts(24), resources: resources({ wall_s: 30, peak_vram_mb: 800 }) })] });
  const value = sweepRlCurves(report).find(row => row.id === 'ppo').points[0];
  assert.equal(value.captures, 36); assert.equal(value.attempts, 72); assert.equal(value.rate, .5);
  assert.equal(value.nativeReleases, 68); assert.equal(value.invalid, 4); assert.equal(value.seeds, 2);
  assert.equal(value.minRate, 1 / 3); assert.equal(value.maxRate, 2 / 3);
  assert.equal(value.wallS, 25); assert.equal(value.peakVramMb, 800);
});

test('unknown resource data stays unknown in points and aggregate metrics', () => {
  const report = fixture({ curves: [point(), point({ seed: 43, resources: { wall_s: NaN, peak_vram_mb: null } })] });
  const row = sweepRlPoints(report)[1];
  assert.equal(row.resources.wall_s, null); assert.equal(row.resources.cpu_s, null);
  const value = sweepRlCurves(report)[1].points[0];
  assert.equal(value.wallS, null); assert.equal(value.cpuS, null); assert.equal(value.peakVramMb, null);
  for (const value of [-1, '0', false, NaN, Infinity]) assert.equal(sweepRlResources({ gpu_s: value }).gpu_s, null);
});

test('invalid counts and undeclared methods are never converted to zero-capture evidence', () => {
  const bad = [point({ method: 'other' }), point({ validation: counts(40, 36) }), point({ seed: false }), point({ samples: 1.2 }), point({ validation: { captures: false, attempts: 36 } })];
  assert.deepEqual(sweepRlPoints(fixture({ curves: bad })), []);
  const row = sweepRlPoints(fixture({ curves: [point({ validation: { captures: 2, attempts: 36, invalid: 40, native_releases: '34', clean: 3 } })] }))[0];
  assert.equal(row.counts.invalid, null); assert.equal(row.counts.nativeReleases, null); assert.equal(row.counts.clean, null);
});

test('duplicate checkpoints do not double denominators; conflicting records fail closed', () => {
  const report = fixture({ curves: [point(), point()] });
  assert.equal(sweepRlPoints(report).length, 1); assert.equal(sweepRlCurves(report)[1].points[0].attempts, 36);
  report.sweep_rl_benchmark.curves.push(point({ validation: counts(13) }));
  assert.deepEqual(sweepRlPoints(report), []);
});

test('training counters and partial final tests never enter validation learning curves', () => {
  const final = { method: 'ppo', seed: 42, counts: counts(35), frozen_artifact_sha256: sha, query_s: .001 };
  const report = fixture({ curves: [point()], runs: [{ method: 'ppo', seed: 42, status: 'completed', train_requests: 1024, train_native_releases: 1000, captures: 1000 }], final_test: [final] });
  assert.equal(sweepRlCurves(report)[1].points[0].captures, 12);
  assert.equal(sweepRlSummary(report).completedRuns, 1); assert.deepEqual(sweepRlFinalRows(report), []);
  report.sweep_rl_benchmark.status = 'completed';
  assert.equal(sweepRlFinalRows(report)[0].counts.captures, 35);
  assert.equal(sweepRlFinalRows(report)[0].artifactSha, sha);
  assert.equal(sweepRlCurves(report)[1].points[0].captures, 12);
});

test('final results require exact frozen artifact hashes and valid counts', () => {
  const final = { method: 'sweep', seed: 42, counts: counts(), frozen_artifact_sha256: sha };
  const report = fixture({ status: 'completed', final_test: [final] });
  assert.equal(sweepRlFinalRows(report).length, 1);
  for (const value of [null, 'pending', 'a'.repeat(63), '/private/checkpoint.pt']) {
    final.frozen_artifact_sha256 = value; assert.deepEqual(sweepRlFinalRows(report), []);
  }
});

test('selected checkpoint budget is exact-hash bound and separate from spent training requests', () => {
  const final = { method: 'ppo', seed: 42, counts: counts(), frozen_artifact_sha256: sha };
  const run = { method: 'ppo', seed: 42, status: 'completed', checkpoint_sha256: sha, selected_samples: 256, train_requests: 1024 };
  const report = fixture({ status: 'completed', runs: [run], final_test: [final] });
  const row = sweepRlFinalRows(report)[0];
  assert.equal(row.selectedSamples, 256); assert.equal(row.trainingRequests, 1024);
  assert.ok(sweepRlCsv(report).includes('"test","1024","256"'));
  run.checkpoint_sha256 = 'b'.repeat(64);
  assert.equal(sweepRlFinalRows(report)[0].selectedSamples, null);
  assert.equal(sweepRlFinalRows(report)[0].trainingRequests, null);
});

test('shared historical cost is explicit and not added to new training sample budgets', () => {
  const summary = sweepRlSummary(fixture({ curves: [point()] }));
  assert.equal(summary.priorRecipes, 17); assert.equal(summary.priorReleases, 4234);
  assert.equal(summary.priorFullCostKnown, false); assert.equal(summary.budget, 1024);
  assert.equal(sweepRlPoints(fixture({ curves: [point()] }))[0].samples, 256);
});

test('CSV preserves validation/test scope, exact counters, unknowns and artifact bindings', () => {
  const report = fixture({ status: 'completed', curves: [point({ resources: { wall_s: 0 } })], final_test: [{ method: 'sweep', seed: 42, counts: counts(), frozen_artifact_sha256: sha, query_s: .000123456 }] });
  const csv = sweepRlCsv(report), rows = csv.trimEnd().split('\n');
  assert.equal(rows.length, 3); assert.ok(rows[1].includes('"validation"')); assert.ok(rows[2].includes('"test"'));
  assert.ok(csv.includes('0.000123456')); assert.ok(csv.includes(sha));
  assert.ok(!csv.includes('undefined') && !csv.includes('NaN') && !csv.includes('Unknown'));
  for (const row of rows) assert.equal(row.split(',').length, rows[0].split(',').length);
});

test('benchmark panel is additive and cannot start training or move the existing solver route', () => {
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  assert.ok(html.indexOf('id="sweepRlBenchmark"') < html.indexOf('id="plannerGridStudy"'));
  assert.ok(html.includes('Final held-out test · frozen artifacts'));
  assert.ok(html.includes('Validation is excluded from the training budget'));
  assert.ok(html.includes('Process VRAM is sampled'));
  const show = js.slice(js.indexOf('function showSweepRlBenchmark'), js.indexOf("$('play').onclick"));
  assert.ok(!show.includes('/api/solve') && !show.includes('POST') && !show.includes('applyResult('));
  assert.ok(js.includes('showSweepRlBenchmark();')); assert.ok(js.includes('sweep_rl_error'));
});
