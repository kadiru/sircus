import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { stalledReferenceSummary, physicsPlanners, InteractivePhysics } from '../web/interactive-physics.mjs';

const app = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
const source = `function showStalledReference() {${app.split('function showStalledReference() {')[1].split('function showDirectPathFirst()')[0]}`;
class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.textContent = ''; this.hidden = false; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  get text() { return this.textContent + this.children.map(node => node.text).join(' '); }
}
function render(report) {
  const nodes = new Map(), $ = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  runInNewContext(`${source}\nshowStalledReference();`, { report, stalledReferenceSummary, $, document: { createElement: tag => new Element(tag) } });
  return $;
}
function fixture() {
  const timing = { mean_s: 24, median_s: 22, p90_s: 31, max_s: 34 };
  const outcomes = [['direct', 'direct'], ['table', 'table'], ['setup_rejected', 'direct'], ['flight_miss', 'table'], ['direct', 'flight_miss']];
  return { stalled_reference_study: { version: 'sircus-stalled-reference-study-v1', status: 'completed', phase: 'completed', audit_status: 'passed',
    budgets: { positions: 5, planning_s: 30 }, progress: { prepared: 10, planned: 10, completed_releases: 9, planned_releases: 9 },
    comparison: { methods: {
      baseline: { positions: 5, admitted: 4, captures: 3, flight_misses: 1, no_candidate: 1 },
      hybrid: { positions: 5, admitted: 5, captures: 4, flight_misses: 1, no_candidate: 0 },
    }, latency: { baseline: timing, hybrid: { ...timing, mean_s: 13 } }, paired: { both: 2, gained: 2, lost: 1, neither: 0 },
    positions: outcomes.map(([baseline, hybrid], index) => ({ id: `case-${index}`, bucket_xy_m: [0.1 + index / 10, -.2854], outcomes: { baseline, hybrid } })) },
    diagnostic: { bucket_xy_m: [.971201, -.285388], methods: { baseline: { checked_shot: false, planning_s: 31.3, outcome: 'setup_rejected' }, hybrid: { checked_shot: true, planning_s: 9.8, outcome: 'direct' } } },
    next: 'Keep the current planner until this comparison is reviewed.' } };
}

test('audited stalled-reference fallback comparison keeps no-setup positions in denominator and shows matched timing', () => {
  const report = fixture(), view = stalledReferenceSummary(report), $ = render(report);
  assert.equal(view.complete, true); assert.equal(view.total, 5);
  assert.match($('stalledReferenceTable').text, /3 \/ 5/);
  assert.match($('stalledReferenceTable').text, /4 \/ 5/);
  assert.match($('stalledReferenceTable').text, /Mean planning time 24.0 s 13.0 s/);
  assert.match($('stalledReferencePair').text, /gained 2 captures and lost 1/);
  assert.match($('stalledReferencePositions').text, /100.0, -285.4 Direct capture Direct capture/);
  assert.match($('stalledReferencePositions').text, /No checked shot/);
  assert.equal($('stalledReferencePositionDetails').hidden, false);
});

test('incomplete, unaudited, inconsistent and duplicate-position results cannot expose final scores', () => {
  for (const mutate of [
    s => { s.status = 'running'; }, s => { s.audit_status = 'pending'; },
    s => { s.progress.prepared = 9; }, s => { s.progress.completed_releases = 8; },
    s => { s.comparison.methods.baseline.captures = 6; }, s => { s.comparison.methods.hybrid.no_candidate = 1; },
    s => { s.comparison.paired.gained = 3; }, s => { s.comparison.positions.pop(); },
    s => { s.comparison.positions[1].id = s.comparison.positions[0].id; },
    s => { s.comparison.positions[1].bucket_xy_m = s.comparison.positions[0].bucket_xy_m; },
    s => { s.comparison.positions[0].outcomes.baseline = 'flight_miss'; },
    s => { s.comparison.positions[0].bucket_xy_m[0] = Infinity; },
    s => { s.comparison.positions[2].outcomes.hybrid = 'flight_miss'; s.comparison.positions[4].outcomes.hybrid = 'direct'; },
  ]) {
    const report = fixture(); mutate(report.stalled_reference_study);
    assert.equal(stalledReferenceSummary(report).complete, false);
    const $ = render(report);
    assert.equal($('stalledReferenceTable').children.length, 0);
    assert.equal($('stalledReferencePositions').children.length, 0);
    assert.equal($('stalledReferencePositionDetails').hidden, true);
  }
});

test('progress does not present predictions as captures; missing timings remain unknown', () => {
  const report = fixture(), section = report.stalled_reference_study;
  section.status = 'running'; section.phase = 'preparing'; section.audit_status = 'pending';
  section.progress = { prepared: 2, planned: 10, completed_releases: 0, planned_releases: null };
  delete section.comparison;
  const $ = render(report);
  assert.match($('stalledReferenceStatus').text, /Checking both planners/);
  assert.match($('stalledReferenceProgress').text, /Plans completed: 2 \/ 10 · Actual drops completed: 0 \/ —/);
  assert.equal($('stalledReferenceTable').children.length, 0);
  const final = fixture(); delete final.stalled_reference_study.comparison.latency.baseline;
  assert.equal(stalledReferenceSummary(final).methods[0].timing.mean_s, null);
  assert.match(render(final)('stalledReferenceTable').text, /Mean planning time — 13.0 s/);
});

test('panel preserves simulation scope and remains optional on historical displays', () => {
  assert.equal(stalledReferenceSummary({}), null);
  assert.equal(stalledReferenceSummary({ stalled_reference_study: { version: 'unknown' } }), null);
  assert.equal(render({})('stalledReference').hidden, true);
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const section = html.split('<section id="stalledReference"')[1].split('</section>')[0];
  assert.match(section, /same 5 mm clearance requirement/);
  assert.match(section, /64 substeps with the arms held at planned poses/);
  assert.match(section, /one continuous second fully inside/);
  assert.match(section, /switches to checking a direct home-to-candidate path first/);
  assert.match(app, /stalledReferenceSummary = module.stalledReferenceSummary/);
});


test('the old timeout has separate planning and native outcomes, never added to fresh counts', () => {
  const report = fixture(), $ = render(report);
  assert.match($('stalledReferenceDiagnostic').text, /Checked shot available No Yes/);
  assert.match($('stalledReferenceDiagnostic').text, /Planning time 31.3 s 9.8 s/);
  assert.match($('stalledReferenceDiagnostic').text, /Drop outcome No checked shot Direct capture/);
  assert.match($('stalledReferenceDiagnosticNote').text, /971.2, -285.4/);
  assert.match($('stalledReferenceDiagnosticNote').text, /separate from the 5 fresh positions/);
  report.stalled_reference_study.diagnostic.methods.hybrid.outcome = null;
  assert.match(render(report)('stalledReferenceDiagnostic').text, /Drop outcome No checked shot Not run/);
  report.stalled_reference_study.audit_status = 'pending';
  assert.equal(render(report)('stalledReferenceDiagnostic').children.length, 0);
});

test('malformed diagnostic does not hide the valid fresh comparison or imply a successful replay', () => {
  for (const mutate of [
    d => { d.bucket_xy_m[0] = Infinity; },
    d => { d.methods.hybrid.planning_s = -1; },
    d => { d.methods.hybrid.outcome = 'predicted_capture'; },
    d => { d.methods.baseline.outcome = 'direct'; },
    d => { d.methods.hybrid.checked_shot = 'true'; },
  ]) {
    const report = fixture(); mutate(report.stalled_reference_study.diagnostic);
    assert.equal(stalledReferenceSummary(report).complete, true);
    assert.equal(stalledReferenceSummary(report).diagnostic, null);
    assert.equal(render(report)('stalledReferenceDiagnostic').children.length, 0);
  }
});

test('rollout evidence requires capture non-regression, bounded mean time and native recovery', () => {
  const good = fixture();
  assert.equal(stalledReferenceSummary(good).promotion_eligible, true);
  assert.match(render(good)('stalledReferenceDecision').text, /meets the predeclared/);
  for (const mutate of [
    s => { s.audit_status = 'pending'; },
    s => { s.diagnostic.methods.hybrid.outcome = 'flight_miss'; },
    s => { s.comparison.latency.hybrid.mean_s = 27; },
    s => { delete s.comparison.latency.baseline; },
  ]) {
    const report=fixture(); mutate(report.stalled_reference_study);
    assert.equal(stalledReferenceSummary(report).promotion_eligible, false);
  }
});

test('hybrid becomes selectable only through server capabilities and preserves old fallbacks', async () => {
  assert.equal(physicsPlanners({ default_planner: 'hybrid' }).defaultPlanner, 'classic');
  const capabilities = { enabled: true, planners: [{ id: 'hybrid' }, { id: 'fast' }, { id: 'classic' }], default_planner: 'hybrid' };
  const view = physicsPlanners(capabilities);
  assert.deepEqual(view.choices.map(row => row.id), ['hybrid', 'fast', 'classic']);
  assert.equal(view.defaultPlanner, 'hybrid');
  const calls = [];
  const client = new InteractivePhysics({ fetcher: async (url, options = {}) => {
    calls.push({ url, options });
    return { ok: true, json: async () => url === '/api/interactive-physics' ? capabilities
      : url.endsWith('/plan') ? { id: 'new', planner: 'hybrid' }
      : { id: 'new', planner: 'hybrid', state: 'no_setup', bucket_xy_m: [.6, -.4] } };
  }, schedule: () => 0, cancel: () => {} });
  await client.initialize();
  await client.plan([.6, -.4]);
  await new Promise(resolve => setImmediate(resolve));
  const posted=calls.filter(row => row.options.method === 'POST');
  assert.equal(posted.length, 1);
  assert.deepEqual(JSON.parse(posted[0].options.body), { bucket_xy_m: [.6, -.4], planner: 'hybrid' });
  assert.equal(client.selectPlanner('fast'), true);
  assert.equal(client.selectPlanner('classic'), true);
  assert.equal(calls.filter(row => row.options.method === 'POST').length, 1);
});
