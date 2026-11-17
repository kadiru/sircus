import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { neuralShotSummary } from '../web/interactive-physics.mjs';

function fixture() {
  return { neural_shot_study: { status: 'completed', phase: 'completed', audit_status: 'passed',
    split_positions: { train: 160, validation: 40, test: 200 },
    progress: { train: { prepared_positions: 160, completed_releases: 3200, planned_releases: 3200 },
      validation: { prepared_positions: 40, completed_releases: 120, planned_releases: 120 },
      test: { prepared_positions: 200, completed_releases: 587, planned_releases: 587 } },
    comparison: { methods: {
      current_planner: { positions: 200, admitted: 195, released: 195, captures: 160, unresolved: 5, robot_contacts: 17 },
      clearance_expansion: { positions: 200, admitted: 198, released: 198, captures: 166, unresolved: 2, robot_contacts: 14 },
      neural_selector: { positions: 200, admitted: 194, released: 194, captures: 165, unresolved: 6, robot_contacts: 11 } },
      latency: { current_planner: { median_s: 32, p90_s: 90 }, clearance_expansion: { median_s: 38, p90_s: 210 },
        neural_selector: { median_s: 17, p90_s: 40 } } }, gpu_lease_s: 3600, gpu_cap_s: 25200 } };
}

test('all planners retain the full new-position denominator, including declined setups', () => {
  const view = neuralShotSummary(fixture());
  assert.equal(view.complete, true);
  assert.equal(view.total, 200);
  assert.deepEqual(view.methods.map(row => [row.positions, row.captures, row.unresolved]),
    [[200, 160, 5], [200, 166, 2], [200, 165, 6]]);
  assert.equal(view.methods[2].median_s, 17);
  assert.equal(view.methods[2].p90_s, 40);
  assert.equal(view.splits[0].released, 3200);
});

test('an unfinished or unaudited experiment cannot publish a final comparison', () => {
  for (const [status, audit] of [['running', 'passed'], ['completed', 'pending'], ['completed', 'failed'], ['failed', 'passed'], ['completed', null]]) {
    const report = fixture(); report.neural_shot_study.status = status; report.neural_shot_study.audit_status = audit;
    assert.equal(neuralShotSummary(report).complete, false);
  }
});

test('missing progress and absent results stay unknown rather than becoming completed zero-valued evidence', () => {
  assert.equal(neuralShotSummary({}), null);
  assert.equal(neuralShotSummary({ neural_shot_study: [] }), null);
  const view = neuralShotSummary({ neural_shot_study: { status: 'running', phase: 'preparing',
    split_positions: { train: 160, validation: 40, test: 200 }, progress: { train: { prepared_positions: 7 } } } });
  assert.equal(view.complete, false);
  assert.equal(view.splits[0].prepared, 7);
  assert.equal(view.splits[0].released, null);
  assert.equal(view.splits[1].prepared, null);
  assert.equal(view.methods[0].captures, null);
});

test('impossible per-method accounting prevents publication even with a passed audit field', () => {
  for (const [field, value] of [['captures', 201], ['captures', -1], ['captures', '160'], ['released', 196],
    ['unresolved', 0], ['positions', 199], ['robot_contacts', 196], ['admitted', null]]) {
    const report = fixture(); report.neural_shot_study.comparison.methods.current_planner[field] = value;
    assert.equal(neuralShotSummary(report).complete, false, `${field}=${value}`);
  }
});

test('progress never presents more prepared positions than the split or a target below actual releases', () => {
  const report = fixture(); report.neural_shot_study.progress.train = { prepared_positions: 161, completed_releases: 20, planned_releases: 10 };
  const view = neuralShotSummary(report);
  assert.equal(view.splits[0].prepared, null);
  assert.equal(view.splits[0].released, 20);
  assert.equal(view.splits[0].planned, null);
});

test('invalid duration metrics remain unknown and are never rendered as measured zero time', () => {
  const report = fixture(); report.neural_shot_study.comparison.latency.current_planner = { median_s: -1, p90_s: Infinity };
  report.neural_shot_study.comparison.latency.neural_selector = { median_s: 40, p90_s: 17 };
  const view = neuralShotSummary(report);
  assert.equal(view.methods[0].median_s, null);
  assert.equal(view.methods[0].p90_s, null);
  assert.equal(view.methods[2].p90_s, null);
});

test('study display is read-only, refreshes with status, and cannot plan or execute a drop', () => {
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  assert.ok(html.indexOf('id="neuralShotStudy"') < html.indexOf('id="historicalStudies"'));
  const body = js.split('function showNeuralShotStudy() {')[1].split('function showClearanceExpansion()')[0];
  assert.match(body, /if \(view\.complete\)/);
  assert.match(body, /neuralShotSummary\(report\)/);
  assert.doesNotMatch(body, /fetch\(|\.plan\(|\.drop\(|POST|requestSolve|innerHTML/);
  const refresh = js.split('async function loadStatus() {')[1].split('function showSavedPhysicsReplays')[0];
  assert.match(refresh, /showNeuralShotStudy\(\)/);
  assert.match(html, /Inputs currently come from simulator geometry, not camera perception/);
  assert.match(html, /This study does not change the live Plan shot button/);
});
