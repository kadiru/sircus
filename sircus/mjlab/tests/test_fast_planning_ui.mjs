import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { fastPlanningSummary } from '../web/interactive-physics.mjs';

const ids = ['cold_30', 'warm_30', 'warm_early'];
const stages = ['menu', 'neural_order', 'arm_setup_checks', 'physics_ranking', 'evidence_publication', 'other'];
function fixture() {
  const metrics = [[40, 32, 20, 15, 5, 0], [42, 36, 18, 14, 4, 0], [41, 35, 6, 3, 3, 30]];
  const methods = Object.fromEntries(ids.map((id, i) => {
    const [releases, captures, deadline, withShot, withoutShot, early] = metrics[i];
    return [id, { attempts: 50, releases, captures, no_candidate: 50 - releases,
      capture_rate: captures / 50, setup_rate: releases / 50, flight_misses: releases - captures,
      deadline_reached: deadline, deadline_with_candidate: withShot, deadline_no_candidate: withoutShot,
      early_return: early, robot_contacts: 3 }];
  }));
  const stats = { mean_s: 24, median_s: 28, p90_s: 30, max_s: 30, aggregate_s: 1200 };
  return { fast_planning_study: { status: 'completed', phase: 'completed', audit_status: 'passed',
    budgets: { positions: 50, planning_s: 30 },
    progress: { prepared: 150, planned: 150, completed_releases: 123, planned_releases: 123 },
    comparison: { methods, latency: Object.fromEntries(ids.map(id => [id, { ...stats }])),
      timing: Object.fromEntries(ids.map(id => [id, { request: { ...stats, mean_s: 26, median_s: 30, p90_s: 32, max_s: 33 },
        startup: { mean_s: 1.2, median_s: 1, p90_s: 2, max_s: 3 } }])),
      profile: Object.fromEntries(ids.map(id => [id, { stages: Object.fromEntries(stages.map((stage, i) =>
        [stage, { aggregate_s: (i + 1) * 10, mean_s: (i + 1) / 5, fraction: (i + 1) / 21 }])) }])),
      workers: { cold_30: { cold_starts: 50, reused_requests: 0, restarts: 0 },
        warm_30: { cold_starts: 12, reused_requests: 38, restarts: 11 },
        warm_early: { cold_starts: 6, reused_requests: 44, restarts: 5 } },
      paired: { warm_vs_cold: { both: 30, gained: 6, lost: 2, neither: 12 },
        early_vs_warm: { both: 33, gained: 2, lost: 3, neither: 12 },
        early_vs_cold: { both: 30, gained: 5, lost: 2, neither: 13 } } } } };
}

const app = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
const renderSource = `function showFastPlanningStudy() {${app.split('function showFastPlanningStudy() {')[1].split('function showPlanningBudgetStudy()')[0]}`;
class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.textContent = ''; this.hidden = false; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  get text() { return this.textContent + this.children.map(node => node.text).join(' '); }
}
function render(report) {
  const nodes = new Map();
  const $ = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  runInNewContext(`${renderSource}\nshowFastPlanningStudy();`, { report, fastPlanningSummary, $,
    document: { createElement: tag => new Element(tag) } });
  return $;
}

test('audited three-method comparison retains missing setups in the capture denominator', () => {
  const view = fastPlanningSummary(fixture());
  assert.equal(view.complete, true);
  assert.equal(view.accounting_valid, true);
  assert.equal(view.prepared, 150);
  assert.equal(view.releases, 123);
  assert.deepEqual(view.methods.map(row => [row.id, row.captures, row.releases, row.no_candidate]),
    [['cold_30', 32, 40, 10], ['warm_30', 36, 42, 8], ['warm_early', 35, 41, 9]]);
  assert.ok(view.paired.every(pair => pair.valid));
});

test('a deadline with a checked shot is not automatically counted as a missed drop', () => {
  const view = fastPlanningSummary(fixture());
  assert.equal(view.methods[0].deadline_reached, 20);
  assert.equal(view.methods[0].deadline_with_candidate, 15);
  assert.equal(view.methods[0].deadline_no_candidate, 5);
  assert.equal(view.methods[0].captures, 32);
  assert.equal(view.methods[2].early_return, 30);
});

test('unfinished, failed or unverified reports do not render final captures or profiling', () => {
  for (const [status, audit] of [['running', 'passed'], ['failed', 'passed'], ['completed', 'pending'], ['completed', 'failed'], ['completed', undefined]]) {
    const report = fixture(); Object.assign(report.fast_planning_study, { status, audit_status: audit });
    const view = fastPlanningSummary(report), $ = render(report);
    assert.equal(view.complete, false);
    assert.equal($('fastPlanningTable').children.length, 0);
    assert.equal($('fastPlanningProfiles').children.length, 0);
    assert.equal($('fastPlanningProfileDetails').hidden, true);
    assert.doesNotMatch($('fastPlanningResultNote').text, /winner|faster method/i);
  }
});

test('native batch phases identify the method being tested without publishing partial scores', () => {
  for (const [phase, description] of [['test_cold_30', 'fresh workers'], ['test_warm_30', 'reused workers'],
    ['test_warm_early', 'earlier-return rule']]) {
    const report = fixture(); Object.assign(report.fast_planning_study, { status: 'running', audit_status: 'pending', phase });
    const $ = render(report);
    assert.ok($('fastPlanningStatus').text.includes(description));
    assert.equal($('fastPlanningTable').children.length, 0);
  }
});

test('impossible counts, mismatched rates, missing methods and contradictory deadlines fail closed', () => {
  for (const [key, value] of [['attempts', 49], ['captures', 41], ['no_candidate', 0], ['deadline_reached', 21],
    ['deadline_with_candidate', 41], ['deadline_no_candidate', 11], ['flight_misses', 9],
    ['robot_contacts', 41], ['early_return', 26], ['releases', '40'], ['capture_rate', .99], ['setup_rate', 1]]) {
    const report = fixture(); report.fast_planning_study.comparison.methods.cold_30[key] = value;
    assert.equal(fastPlanningSummary(report).complete, false, `${key}=${value}`);
  }
  const report = fixture(); delete report.fast_planning_study.comparison.methods.warm_30;
  assert.equal(fastPlanningSummary(report).complete, false);
});

test('final progress must account for all 150 plans and every actual release', () => {
  for (const progress of [{ prepared: 149 }, { planned: 50 }, { completed_releases: 122 }, { planned_releases: 124 }]) {
    const report = fixture(); Object.assign(report.fast_planning_study.progress, progress);
    const view = fastPlanningSummary(report), $ = render(report);
    assert.equal(view.complete, false);
    assert.match($('fastPlanningStatus').text, /totals disagree/);
    assert.equal($('fastPlanningTable').children.length, 0);
  }
  const report = fixture(); report.fast_planning_study.progress = { prepared: 151, planned: 150, completed_releases: 170, planned_releases: 160 };
  const view = fastPlanningSummary(report);
  assert.equal(view.prepared, null); assert.equal(view.releases, null); assert.equal(view.planned_releases, null);
});

test('unknown counts and optional profiles remain unknown, with no invented zero timings', () => {
  assert.equal(fastPlanningSummary({}), null);
  assert.equal(fastPlanningSummary({ fast_planning_study: [] }), null);
  const pending = fastPlanningSummary({ fast_planning_study: { status: 'running', budgets: { positions: 50, planning_s: 30 },
    progress: { prepared: 3, planned: 150 } } });
  assert.equal(pending.releases, null); assert.equal(pending.methods[0].captures, null);
  const report = fixture(); delete report.fast_planning_study.comparison.profile;
  delete report.fast_planning_study.comparison.timing; delete report.fast_planning_study.comparison.workers;
  const view = fastPlanningSummary(report);
  assert.equal(view.complete, true);
  assert.equal(view.methods[0].request.mean_s, null);
  assert.equal(view.methods[0].startup.median_s, null);
  assert.equal(view.methods[0].workers, null);
  assert.ok(view.methods.every(row => row.stages.every(stage => stage.aggregate_s === null && stage.fraction === null)));
  assert.match(render(report)('fastPlanningProfiles').text, /—/);
});

test('invalid optional times and profiler fractions are hidden independently of capture counts', () => {
  const report = fixture(), comparison = report.fast_planning_study.comparison;
  comparison.latency.cold_30 = { mean_s: 32, median_s: 29, p90_s: 20, max_s: 10 };
  comparison.timing.warm_30 = { request: null, startup: { median_s: -1 } };
  comparison.profile.cold_30.stages.menu = { aggregate_s: -4, mean_s: Infinity, fraction: 2 };
  comparison.workers.warm_30.reused_requests = 50;
  const view = fastPlanningSummary(report);
  assert.equal(view.complete, true);
  assert.equal(view.methods[0].decision.mean_s, null);
  assert.equal(view.methods[0].decision.p90_s, null);
  assert.equal(view.methods[0].decision.max_s, null);
  assert.equal(view.methods[1].request.median_s, null);
  assert.equal(view.methods[1].workers, null);
  assert.equal(view.methods[0].stages[0].aggregate_s, null);
  assert.equal(view.methods[0].stages[0].fraction, null);
});

test('matched gains are shown only when their counts agree with both methods', () => {
  const report = fixture(); report.fast_planning_study.comparison.paired.early_vs_warm.gained = 9;
  const view = fastPlanningSummary(report), $ = render(report);
  assert.equal(view.complete, true);
  assert.equal(view.paired.find(row => row.id === 'early_vs_warm').valid, false);
  assert.equal($('fastPlanningPaired').children.length, 2);
  assert.doesNotMatch($('fastPlanningPaired').text, /9 gained/);
});

test('renderer shows a compact four-column comparison with measured request and decision times', () => {
  const $ = render(fixture());
  assert.equal($('fastPlanningStudy').hidden, false);
  assert.match($('fastPlanningStatus').text, /independent audit passed/);
  assert.match($('fastPlanningTable').text, /32 \/ 50 \(64%\)/);
  assert.match($('fastPlanningTable').text, /Median measured planning request/);
  assert.match($('fastPlanningTable').text, /90th-percentile decision time/);
  const table = $('fastPlanningTable').children[0];
  assert.equal(table.children[0].children[0].children.length, 4);
  assert.ok(table.children[1].children.every(row => row.children.length === 4 && row.children[0].scope === 'row'));
  assert.match($('fastPlanningProfiles').text, /Checking arm poses and paths/);
  assert.match($('fastPlanningProfiles').text, /Requests reusing a worker/);
});

test('panel is additive and read-only; no default switch, plan request or synthetic result is introduced', () => {
  assert.ok(html.indexOf('id="fastPlanningStudy"') < html.indexOf('id="planningBudgetStudy"'));
  assert.match(html, /Predicted quality is not a measured capture/);
  assert.match(html, /at least 5 mm of predicted margin/);
  assert.match(html, /not total UI latency/);
  assert.doesNotMatch(renderSource, /fetch\(|\.plan\(|\.drop\(|POST|requestSolve|innerHTML/);
  const refresh = app.split('async function loadStatus() {')[1].split('function showSavedPhysicsReplays')[0];
  assert.match(refresh, /showFastPlanningStudy\(\)/);
  assert.match(app, /fastPlanningSummary = module\.fastPlanningSummary/);
});
