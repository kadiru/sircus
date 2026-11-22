import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { directPathFirstSummary } from '../web/interactive-physics.mjs';

const app = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
const source = `function showDirectPathFirst() {${app.split('function showDirectPathFirst() {')[1].split('function showArmCheckSpeed()')[0]}`;
class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.textContent = ''; this.hidden = false; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  get text() { return this.textContent + this.children.map(node => node.text).join(' '); }
}
function render(report) {
  const nodes = new Map(), $ = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  runInNewContext(`${source}\nshowDirectPathFirst();`, { report, directPathFirstSummary, $, document: { createElement: tag => new Element(tag) } });
  return $;
}
function fixture() {
  const timing = { mean_s: 24, median_s: 22, p90_s: 31, max_s: 34 };
  const outcomes = [['direct', 'direct'], ['table', 'table'], ['setup_rejected', 'direct'], ['flight_miss', 'table'], ['direct', 'flight_miss']];
  return { direct_path_first_study: { version: 'sircus-direct-path-first-study-v1', status: 'completed', phase: 'completed', audit_status: 'passed',
    budgets: { positions: 5, planning_s: 30 }, progress: { prepared: 10, planned: 10, completed_releases: 9, planned_releases: 9 },
    comparison: { methods: {
      baseline: { positions: 5, admitted: 4, captures: 3, flight_misses: 1, no_candidate: 1 },
      direct_first: { positions: 5, admitted: 5, captures: 4, flight_misses: 1, no_candidate: 0 },
    }, latency: { baseline: timing, direct_first: { ...timing, mean_s: 13 } }, paired: { both: 2, gained: 2, lost: 1, neither: 0 },
    positions: outcomes.map(([baseline, direct_first], index) => ({ id: `case-${index}`, bucket_xy_m: [0.1 + index / 10, -.2854], outcomes: { baseline, direct_first } })) },
    diagnostic: { bucket_xy_m: [.971201, -.285388], methods: { baseline: { checked_shot: false, planning_s: 31.3, outcome: 'setup_rejected' }, direct_first: { checked_shot: true, planning_s: 9.8, outcome: 'direct' } } },
    next: 'Keep the current planner until this comparison is reviewed.' } };
}

test('audited candidate-first comparison keeps no-setup positions in denominator and shows matched timing', () => {
  const report = fixture(), view = directPathFirstSummary(report), $ = render(report);
  assert.equal(view.complete, true); assert.equal(view.total, 5);
  assert.match($('directPathFirstTable').text, /3 \/ 5/);
  assert.match($('directPathFirstTable').text, /4 \/ 5/);
  assert.match($('directPathFirstTable').text, /Mean planning time 24.0 s 13.0 s/);
  assert.match($('directPathFirstPair').text, /gained 2 captures and lost 1/);
  assert.match($('directPathFirstPositions').text, /100.0, -285.4 Direct capture Direct capture/);
  assert.match($('directPathFirstPositions').text, /No checked shot/);
  assert.equal($('directPathFirstPositionDetails').hidden, false);
});

test('incomplete, unaudited, inconsistent and duplicate-position results cannot expose final scores', () => {
  for (const mutate of [
    s => { s.status = 'running'; }, s => { s.audit_status = 'pending'; },
    s => { s.progress.prepared = 9; }, s => { s.progress.completed_releases = 8; },
    s => { s.comparison.methods.baseline.captures = 6; }, s => { s.comparison.methods.direct_first.no_candidate = 1; },
    s => { s.comparison.paired.gained = 3; }, s => { s.comparison.positions.pop(); },
    s => { s.comparison.positions[1].id = s.comparison.positions[0].id; },
    s => { s.comparison.positions[1].bucket_xy_m = s.comparison.positions[0].bucket_xy_m; },
    s => { s.comparison.positions[0].outcomes.baseline = 'flight_miss'; },
    s => { s.comparison.positions[0].bucket_xy_m[0] = Infinity; },
    s => { s.comparison.positions[2].outcomes.direct_first = 'flight_miss'; s.comparison.positions[4].outcomes.direct_first = 'direct'; },
  ]) {
    const report = fixture(); mutate(report.direct_path_first_study);
    assert.equal(directPathFirstSummary(report).complete, false);
    const $ = render(report);
    assert.equal($('directPathFirstTable').children.length, 0);
    assert.equal($('directPathFirstPositions').children.length, 0);
    assert.equal($('directPathFirstPositionDetails').hidden, true);
  }
});

test('progress does not present predictions as captures; missing timings remain unknown', () => {
  const report = fixture(), section = report.direct_path_first_study;
  section.status = 'running'; section.phase = 'preparing'; section.audit_status = 'pending';
  section.progress = { prepared: 2, planned: 10, completed_releases: 0, planned_releases: null };
  delete section.comparison;
  const $ = render(report);
  assert.match($('directPathFirstStatus').text, /Checking both planners/);
  assert.match($('directPathFirstProgress').text, /Plans completed: 2 \/ 10 · Actual drops completed: 0 \/ —/);
  assert.equal($('directPathFirstTable').children.length, 0);
  const final = fixture(); delete final.direct_path_first_study.comparison.latency.baseline;
  assert.equal(directPathFirstSummary(final).methods[0].timing.mean_s, null);
  assert.match(render(final)('directPathFirstTable').text, /Mean planning time — 13.0 s/);
});

test('panel preserves simulation scope and remains optional on historical displays', () => {
  assert.equal(directPathFirstSummary({}), null);
  assert.equal(directPathFirstSummary({ direct_path_first_study: { version: 'unknown' } }), null);
  assert.equal(render({})('directPathFirst').hidden, true);
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const section = html.split('<section id="directPathFirst"')[1].split('</section>')[0];
  assert.match(section, /same 5 mm clearance requirement/);
  assert.match(section, /64 substeps with the arms held at planned poses/);
  assert.match(section, /one continuous second fully inside/);
  assert.match(section, /checks a direct home-to-candidate path first/);
  assert.match(app, /directPathFirstSummary = module.directPathFirstSummary/);
});


test('the old timeout has separate planning and native outcomes, never added to fresh counts', () => {
  const report = fixture(), $ = render(report);
  assert.match($('directPathFirstDiagnostic').text, /Checked shot available No Yes/);
  assert.match($('directPathFirstDiagnostic').text, /Planning time 31.3 s 9.8 s/);
  assert.match($('directPathFirstDiagnostic').text, /Drop outcome No checked shot Direct capture/);
  assert.match($('directPathFirstDiagnosticNote').text, /971.2, -285.4/);
  assert.match($('directPathFirstDiagnosticNote').text, /separate from the 5 fresh positions/);
  report.direct_path_first_study.diagnostic.methods.direct_first.outcome = null;
  assert.match(render(report)('directPathFirstDiagnostic').text, /Drop outcome No checked shot Not run/);
  report.direct_path_first_study.audit_status = 'pending';
  assert.equal(render(report)('directPathFirstDiagnostic').children.length, 0);
});

test('malformed diagnostic does not hide the valid fresh comparison or imply a successful replay', () => {
  for (const mutate of [
    d => { d.bucket_xy_m[0] = Infinity; },
    d => { d.methods.direct_first.planning_s = -1; },
    d => { d.methods.direct_first.outcome = 'predicted_capture'; },
    d => { d.methods.baseline.outcome = 'direct'; },
    d => { d.methods.direct_first.checked_shot = 'true'; },
  ]) {
    const report = fixture(); mutate(report.direct_path_first_study.diagnostic);
    assert.equal(directPathFirstSummary(report).complete, true);
    assert.equal(directPathFirstSummary(report).diagnostic, null);
    assert.equal(render(report)('directPathFirstDiagnostic').children.length, 0);
  }
});
