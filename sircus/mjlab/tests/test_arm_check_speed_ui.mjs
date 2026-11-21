import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { armCheckSpeedSummary } from '../web/interactive-physics.mjs';

const app = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
const source = `function showArmCheckSpeed() {${app.split('function showArmCheckSpeed() {')[1].split('function showInteractiveFastAcceptance()')[0]}`;
class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.textContent = ''; this.hidden = false; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  get text() { return this.textContent + this.children.map(node => node.text).join(' '); }
}
function render(report) {
  const nodes = new Map(), $ = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  runInNewContext(`${source}\nshowArmCheckSpeed();`, { report, armCheckSpeedSummary, $, document: { createElement: tag => new Element(tag) } });
  return $;
}
function fixture() {
  const timing = { mean_s: 24, median_s: 22, p90_s: 31, max_s: 34 };
  const outcomes = [['direct', 'direct'], ['table', 'table'], ['setup_rejected', 'direct'], ['flight_miss', 'table'], ['direct', 'flight_miss']];
  return { arm_check_speed_study: { version: 'sircus-arm-check-speed-study-v1', status: 'completed', phase: 'completed', audit_status: 'passed',
    budgets: { positions: 5, planning_s: 30 }, progress: { prepared: 10, planned: 10, completed_releases: 9, planned_releases: 9 },
    comparison: { methods: {
      baseline: { positions: 5, admitted: 4, captures: 3, flight_misses: 1, no_candidate: 1 },
      optimized: { positions: 5, admitted: 5, captures: 4, flight_misses: 1, no_candidate: 0 },
    }, latency: { baseline: timing, optimized: { ...timing, mean_s: 13 } }, paired: { both: 2, gained: 2, lost: 1, neither: 0 },
    positions: outcomes.map(([baseline, optimized], index) => ({ id: `case-${index}`, bucket_xy_m: [0.1 + index / 10, -.2854], outcomes: { baseline, optimized } })) },
    next: 'Keep the current planner until this comparison is reviewed.' } };
}

test('audited arm-check comparison keeps no-setup positions in denominator and shows matched timing', () => {
  const report = fixture(), view = armCheckSpeedSummary(report), $ = render(report);
  assert.equal(view.complete, true); assert.equal(view.total, 5);
  assert.match($('armCheckSpeedTable').text, /3 \/ 5/);
  assert.match($('armCheckSpeedTable').text, /4 \/ 5/);
  assert.match($('armCheckSpeedTable').text, /Mean planning time 24.0 s 13.0 s/);
  assert.match($('armCheckSpeedPair').text, /gained 2 captures and lost 1/);
  assert.match($('armCheckSpeedPositions').text, /100.0, -285.4 Direct capture Direct capture/);
  assert.match($('armCheckSpeedPositions').text, /No checked shot/);
  assert.equal($('armCheckSpeedPositionDetails').hidden, false);
});

test('incomplete, unaudited, inconsistent and duplicate-position results cannot expose final scores', () => {
  for (const mutate of [
    s => { s.status = 'running'; }, s => { s.audit_status = 'pending'; },
    s => { s.progress.prepared = 9; }, s => { s.progress.completed_releases = 8; },
    s => { s.comparison.methods.baseline.captures = 6; }, s => { s.comparison.methods.optimized.no_candidate = 1; },
    s => { s.comparison.paired.gained = 3; }, s => { s.comparison.positions.pop(); },
    s => { s.comparison.positions[1].id = s.comparison.positions[0].id; },
    s => { s.comparison.positions[1].bucket_xy_m = s.comparison.positions[0].bucket_xy_m; },
    s => { s.comparison.positions[0].outcomes.baseline = 'flight_miss'; },
    s => { s.comparison.positions[0].bucket_xy_m[0] = Infinity; },
    s => { s.comparison.positions[2].outcomes.optimized = 'flight_miss'; s.comparison.positions[4].outcomes.optimized = 'direct'; },
  ]) {
    const report = fixture(); mutate(report.arm_check_speed_study);
    assert.equal(armCheckSpeedSummary(report).complete, false);
    const $ = render(report);
    assert.equal($('armCheckSpeedTable').children.length, 0);
    assert.equal($('armCheckSpeedPositions').children.length, 0);
    assert.equal($('armCheckSpeedPositionDetails').hidden, true);
  }
});

test('progress does not present predictions as captures; missing timings remain unknown', () => {
  const report = fixture(), section = report.arm_check_speed_study;
  section.status = 'running'; section.phase = 'preparing'; section.audit_status = 'pending';
  section.progress = { prepared: 2, planned: 10, completed_releases: 0, planned_releases: null };
  delete section.comparison;
  const $ = render(report);
  assert.match($('armCheckSpeedStatus').text, /Checking both planners/);
  assert.match($('armCheckSpeedProgress').text, /Plans completed: 2 \/ 10 · Actual drops completed: 0 \/ —/);
  assert.equal($('armCheckSpeedTable').children.length, 0);
  const final = fixture(); delete final.arm_check_speed_study.comparison.latency.baseline;
  assert.equal(armCheckSpeedSummary(final).methods[0].timing.mean_s, null);
  assert.match(render(final)('armCheckSpeedTable').text, /Mean planning time — 13.0 s/);
});

test('panel preserves simulation scope and remains optional on historical displays', () => {
  assert.equal(armCheckSpeedSummary({}), null);
  assert.equal(armCheckSpeedSummary({ arm_check_speed_study: { version: 'unknown' } }), null);
  assert.equal(render({})('armCheckSpeed').hidden, true);
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const section = html.split('<section id="armCheckSpeed"')[1].split('</section>')[0];
  assert.match(section, /same 5 mm clearance requirement/);
  assert.match(section, /64 substeps with the arms held at planned poses/);
  assert.match(section, /one continuous second fully inside/);
  assert.match(section, /stops a rejected path at its first failed segment/);
  assert.match(app, /armCheckSpeedSummary = module.armCheckSpeedSummary/);
});
