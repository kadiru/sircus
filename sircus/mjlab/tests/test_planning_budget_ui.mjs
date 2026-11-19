import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { planningBudgetSummary } from '../web/interactive-physics.mjs';

function fixture() {
  return { planning_budget_study: { status: 'completed', phase: 'completed', audit_status: 'passed',
    budgets: { planning_s: 30, positions: 100 },
    progress: { prepared: 200, planned: 200, completed_releases: 187, planned_releases: 187 },
    comparison: { methods: {
      current_planner: { attempts: 100, releases: 92, captures: 76, setup_rate: .92, capture_rate: .76,
        deadline_reached: 32, no_candidate: 8, robot_contacts: 6 },
      neural_selector: { attempts: 100, releases: 95, captures: 79, setup_rate: .95, capture_rate: .79,
        deadline_reached: 22, no_candidate: 5, robot_contacts: 7 } },
      latency: { current_planner: { median_s: 26, p90_s: 30, max_s: 30 },
        neural_selector: { median_s: 19, p90_s: 30, max_s: 30 } },
      paired: { gained: 8, lost: 5, both: 71, neither: 16 } } } };
}

test('full target denominator includes no-shot positions while completed releases stay separate', () => {
  const view = planningBudgetSummary(fixture());
  assert.equal(view.complete, true);
  assert.equal(view.total, 100);
  assert.equal(view.budget, 30);
  assert.equal(view.prepared, 200);
  assert.equal(view.releases, 187);
  assert.deepEqual(view.methods.map(row => [row.attempts, row.releases, row.captures, row.no_candidate]),
    [[100, 92, 76, 8], [100, 95, 79, 5]]);
  assert.equal(view.paired.gained, 8);
});

test('deadline reached is not treated as a failed release or automatically subtracted from captures', () => {
  const report = fixture();
  report.planning_budget_study.comparison.methods.neural_selector.deadline_reached = 100;
  const view = planningBudgetSummary(report);
  assert.equal(view.complete, true);
  assert.equal(view.methods[1].captures, 79);
  assert.equal(view.methods[1].no_candidate, 5);
});

test('partial results and absent or failed audits never publish final scores', () => {
  for (const [status, audit] of [['running', 'passed'], ['completed', 'pending'], ['completed', 'failed'], ['failed', 'passed'], ['completed', undefined]]) {
    const report = fixture(); report.planning_budget_study.status = status; report.planning_budget_study.audit_status = audit;
    assert.equal(planningBudgetSummary(report).complete, false);
  }
});

test('missing evidence remains unknown instead of inventing zero counts', () => {
  assert.equal(planningBudgetSummary({}), null);
  assert.equal(planningBudgetSummary({ planning_budget_study: [] }), null);
  const view = planningBudgetSummary({ planning_budget_study: { status: 'running', phase: 'preparing',
    budgets: { planning_s: 30, positions: 100 }, progress: { prepared: 3, planned: 200 } } });
  assert.equal(view.prepared, 3);
  assert.equal(view.releases, null);
  assert.equal(view.planned_releases, null);
  assert.equal(view.methods[0].captures, null);
  assert.equal(view.complete, false);
});

test('impossible method accounting or mismatched rates blocks the final comparison', () => {
  for (const [key, value] of [['attempts', 99], ['captures', 93], ['no_candidate', 0], ['deadline_reached', 101],
    ['robot_contacts', 93], ['releases', -1], ['releases', '92'], ['capture_rate', .9], ['setup_rate', 1]]) {
    const report = fixture(); report.planning_budget_study.comparison.methods.current_planner[key] = value;
    assert.equal(planningBudgetSummary(report).complete, false, `${key}=${value}`);
  }
});

test('inconsistent paired gains and losses cannot be presented as measured improvement', () => {
  const report = fixture(); report.planning_budget_study.comparison.paired.gained = 9;
  const view = planningBudgetSummary(report);
  assert.equal(view.complete, false);
  assert.equal(view.paired, null);
});

test('impossible progress totals are hidden without discarding actual completed releases', () => {
  const report = fixture(); report.planning_budget_study.progress = {
    prepared: 201, planned: 200, completed_releases: 50, planned_releases: 40 };
  const view = planningBudgetSummary(report);
  assert.equal(view.prepared, null);
  assert.equal(view.releases, 50);
  assert.equal(view.planned_releases, null);
  report.planning_budget_study.progress.planned = 100;
  assert.equal(planningBudgetSummary(report).planned, null);
});

test('negative and inconsistent timing measurements render as unknown', () => {
  const report = fixture(); report.planning_budget_study.comparison.latency.current_planner = { median_s: -1, p90_s: Infinity, max_s: -1 };
  report.planning_budget_study.comparison.latency.neural_selector = { median_s: 29, p90_s: 20, max_s: 10 };
  const view = planningBudgetSummary(report);
  assert.equal(view.methods[0].median_s, null);
  assert.equal(view.methods[0].p90_s, null);
  assert.equal(view.methods[0].max_s, null);
  assert.equal(view.methods[1].p90_s, null);
  assert.equal(view.methods[1].max_s, null);
});

test('startup and worker lifecycle timings are optional and separate from the decision deadline', () => {
  const report = fixture();
  let view = planningBudgetSummary(report);
  assert.equal(view.complete, true);
  assert.equal(view.methods[0].startup_median_s, null);
  assert.equal(view.methods[0].end_to_end_median_s, null);
  report.planning_budget_study.comparison.timing = {
    current_planner: { startup: { median_s: 6 }, end_to_end: { median_s: 34 } },
    neural_selector: { startup: { median_s: -1 }, end_to_end: { median_s: Infinity } },
  };
  view = planningBudgetSummary(report);
  assert.equal(view.complete, true);
  assert.equal(view.budget, 30);
  assert.equal(view.methods[0].median_s, 26);
  assert.equal(view.methods[0].startup_median_s, 6);
  assert.equal(view.methods[0].end_to_end_median_s, 34);
  assert.equal(view.methods[1].startup_median_s, null);
  assert.equal(view.methods[1].end_to_end_median_s, null);
});

test('comparison UI is additive and read-only, refreshes on status, and waits for audited results', () => {
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  assert.ok(html.indexOf('id="planningBudgetStudy"') < html.indexOf('id="neuralShotStudy"'));
  const body = js.split('function showPlanningBudgetStudy() {')[1].split('function showNeuralShotStudy()')[0];
  assert.match(body, /if \(view\.complete\)/);
  assert.match(body, /planningBudgetSummary\(report\)/);
  assert.match(body, /Median decision time/);
  assert.match(body, /Worker lifecycle runs from startup through worker shutdown/);
  assert.match(body, /is not total UI request latency/);
  assert.doesNotMatch(body, /fetch\(|\.plan\(|\.drop\(|POST|requestSolve|innerHTML/);
  const refresh = js.split('async function loadStatus() {')[1].split('function showSavedPhysicsReplays')[0];
  assert.match(refresh, /showPlanningBudgetStudy\(\)/);
  assert.match(html, /A planning timeout is not a measured flight miss/);
  assert.match(html, /This earlier study did not change the live planner at the time/);
});
