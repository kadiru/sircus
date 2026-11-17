import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { InteractivePhysics, physicsPlanners, physicsProgress, interactiveFastAcceptanceSummary } from '../web/interactive-physics.mjs';

const capabilities = { enabled: true, planners: [{ id: 'fast' }, { id: 'classic' }], default_planner: 'fast' };
const xy = [.6, -.4];
const preview = { status: 'feasible', planner: 'fast', bucket_xy_m: xy, configuration: { board_center_m: [.6, 0, .4], board_quaternion_wxyz: [1, 0, 0, 0], release_center_m: [.6, 0, .7], q10: [0, 1], q14: [1, 0], link_positions10: [[0, 0, 0], [.6, 0, .7]], link_positions14: [[1.3, 0, 0], [.6, 0, .4]] } };
const response = (data, status = 200) => ({ ok: status < 400, json: async () => data });
const flush = () => new Promise(resolve => setImmediate(resolve));
function harness(caps = capabilities) {
  const requests = [], timers = [];
  let handler = url => response(url === '/api/interactive-physics' ? caps : { id: 'job', state: 'queued' });
  const client = new InteractivePhysics({ fetcher: async (url, options) => { requests.push({ url, options }); return handler(url, options); }, schedule: fn => { timers.push(fn); return fn; }, cancel: fn => { timers.splice(timers.indexOf(fn), 1); } });
  return { client, requests, timers, set handler(value) { handler = value; } };
}

test('fast is selected only when advertised; old API keeps its exact request contract', async () => {
  assert.equal(physicsPlanners({ default_planner: 'fast' }).defaultPlanner, 'classic');
  assert.equal(physicsPlanners({ planners: [{ id: 'classic' }], default_planner: 'fast' }).defaultPlanner, 'classic');
  assert.deepEqual(physicsPlanners({ planners: [{ id: 'unknown' }] }).choices.map(row => row.id), ['classic']);
  for (const [caps, expected] of [[capabilities, { bucket_xy_m: xy, planner: 'fast' }], [{ enabled: true }, { bucket_xy_m: xy }]]) {
    const h = harness(caps); await h.client.initialize();
    h.handler = url => response(url.endsWith('/plan') ? { id: 'job' } : { id: 'job', bucket_xy_m: xy, state: 'no_setup' });
    await h.client.plan(xy); await flush();
    assert.deepEqual(JSON.parse(h.requests.find(row => row.options.method === 'POST').options.body), expected);
    assert.equal(h.client.canDrop, false);
  }
});

test('switching planner clears a ready preview; the original fallback needs an explicit new Plan click', async () => {
  const h = harness(); await h.client.initialize();
  h.handler = url => response(url.endsWith('/plan') ? { id: 'job', planner: 'fast' } : { id: 'job', planner: 'fast', state: 'ready', bucket_xy_m: xy, preview });
  await h.client.plan(xy); await flush(); assert.equal(h.client.canDrop, true);
  const calls = h.requests.length;
  assert.equal(h.client.selectPlanner('classic'), true);
  assert.equal(h.client.snapshot.state, 'stale'); assert.equal(h.client.snapshot.preview, undefined);
  assert.equal(h.client.canDrop, false); assert.equal(h.requests.length, calls);
  h.handler = url => response(url.endsWith('/plan') ? { id: 'fallback', planner: 'classic' } : { id: 'fallback', planner: 'classic', state: 'no_setup', bucket_xy_m: xy });
  await h.client.plan(xy); await flush();
  assert.equal(JSON.parse(h.requests.filter(row => row.options.method === 'POST').at(-1).options.body).planner, 'classic');
  assert.equal(h.client.planner, 'classic'); assert.equal(h.client.busy, false);
});

test('pending plan and drop lock planner selection; stale old plan cannot re-enable Drop', async () => {
  const h = harness(); await h.client.initialize(); let resolve;
  h.handler = () => new Promise(done => { resolve = done; });
  const plan = h.client.plan(xy);
  assert.equal(h.client.selectPlanner('classic'), false); assert.equal(h.client.planner, 'fast');
  h.client.invalidate('Scene replaced'); h.client.selectPlanner('classic');
  resolve(response({ id: 'old', planner: 'fast' })); await plan;
  assert.equal(h.client.snapshot.state, 'stale'); assert.equal(h.client.canDrop, false);
  h.client.selectPlanner('fast');
  h.handler = url => response(url.endsWith('/plan') ? { id: 'job', planner: 'fast' } : { id: 'job', planner: 'fast', state: 'ready', bucket_xy_m: xy, preview });
  await h.client.plan(xy); await flush();
  h.handler = () => new Promise(done => { resolve = done; });
  const drop = h.client.drop(); assert.equal(h.client.selectPlanner('classic'), false);
  resolve(response({ error: 'GPU unavailable' }, 503)); await drop;
  assert.equal(h.client.busy, false); assert.equal(h.client.canDrop, false); assert.equal(h.client.planner, 'fast');
});

test('failed fast requests never trigger a hidden fallback; contradictory planner identity is rejected', async () => {
  for (const wrong of ['job', 'preview']) {
    const h = harness(); await h.client.initialize();
    h.handler = url => response(url.endsWith('/plan') ? { id: 'job', planner: 'fast' } : {
      id: 'job', planner: wrong === 'job' ? 'classic' : 'fast', state: 'ready', bucket_xy_m: xy,
      preview: { ...preview, planner: wrong === 'preview' ? 'classic' : 'fast' },
    });
    await h.client.plan(xy); await flush();
    assert.equal(h.client.snapshot.state, 'failed'); assert.equal(h.client.canDrop, false);
    assert.equal(h.client.planner, 'fast'); assert.equal(h.requests.filter(row => row.options.method === 'POST').length, 1);
    assert.equal(h.timers.length, 0);
  }
  assert.equal(physicsProgress({ planning_elapsed_s: 31.5, progress: { elapsed_s: 29.9, proposed: 10 } }), '10 candidates · Elapsed 31 s');
});

function fixture() {
  const timing = { mean_s: 23, median_s: 22, p90_s: 31, max_s: 34 };
  return { interactive_fast_acceptance: { version: 'sircus-interactive-fast-acceptance-v1', status: 'completed', audit_status: 'passed',
    progress: { completed: 20, planned: 20 },
    comparison: {
      fast: { positions: 10, releases: 9, captures: 7, no_setup: 1, failed: 0, worker_reused: 9, planning: timing, backend_request: timing },
      classic: { positions: 10, releases: 8, captures: 6, no_setup: 1, failed: 1, worker_reused: 0, planning: { ...timing, mean_s: 45 } },
    }, paired: { both: 5, gained: 2, lost: 1, neither: 2 } } };
}
const app = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
const source = `function showInteractiveFastAcceptance() {${app.split('function showInteractiveFastAcceptance() {')[1].split('function showFastPlanningStudy()')[0]}`;
class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.textContent = ''; this.hidden = false; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  get text() { return this.textContent + this.children.map(node => node.text).join(' '); }
}
function render(report) {
  const nodes = new Map(), $ = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  runInNewContext(`${source}\nshowInteractiveFastAcceptance();`, { report, interactiveFastAcceptanceSummary, $, document: { createElement: tag => new Element(tag) } });
  return $;
}

test('live acceptance retains all placements in capture denominator and separates request vs server timing', () => {
  const report = fixture(), view = interactiveFastAcceptanceSummary(report), $ = render(report);
  assert.equal(view.complete, true); assert.equal(view.total, 10);
  assert.match($('interactiveFastAcceptanceTable').text, /7 \/ 10/);
  assert.match($('interactiveFastAcceptanceTable').text, /Mean wait for plan/);
  assert.match($('interactiveFastAcceptanceTable').text, /Mean server planning time/);
  assert.match($('interactiveFastAcceptancePair').text, /2 gained captures, 1 lost/);
});

test('unfinished, unaudited and inconsistent live results hide scores; absent optional timing stays unknown', () => {
  for (const mutate of [
    section => { section.status = 'running'; }, section => { section.audit_status = 'pending'; },
    section => { section.progress.completed = 19; }, section => { section.comparison.fast.captures = 11; },
    section => { section.paired.gained = 3; }, section => { delete section.comparison.classic; },
  ]) {
    const report = fixture(); mutate(report.interactive_fast_acceptance);
    assert.equal(interactiveFastAcceptanceSummary(report).complete, false);
    assert.equal(render(report)('interactiveFastAcceptanceTable').children.length, 0);
  }
  assert.equal(interactiveFastAcceptanceSummary({}), null);
  assert.equal(interactiveFastAcceptanceSummary({ interactive_fast_acceptance: { version: 'unknown' } }), null);
  assert.equal(interactiveFastAcceptanceSummary(fixture()).methods[1].backend_request.mean_s, null);
});

test('busy UI locks bucket dragging, inputs, planner, clear-preview and recorded replays', () => {
  assert.match(app, /pointerdown', event => \{\s+if \(pending \|\| physics\?\.busy\) return/);
  assert.match(app, /if \(!dragging \|\| pending \|\| physics\?\.busy\) return/);
  const sync = app.split('function syncSolveButtons() {')[1].split('function setPending')[0];
  assert.match(sync, /\['bucketX', 'bucketY', 'mode', 'searchBudget'\].*disabled = blocked/);
  assert.match(sync, /\$\('physicsPlanner'\).disabled = blocked/);
  assert.match(sync, /\$\('physicsClear'\).disabled = pending \|\| physics\?\.busy === true/);
  assert.match(app, /clearResult\('Planner changed — plan again', true\)/);
});


test('planner selection precedes Plan and Drop while older modes keep it hidden', () => {
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  assert.ok(html.indexOf('id="physicsPlanner"') < html.indexOf('id="solve"'));
  assert.ok(html.indexOf('id="solve"') < html.indexOf('id="physicsDrop"'));
  assert.match(html, /id="physicsPlannerControls"[^>]*hidden/);
  assert.match(app, /\$\('physicsPlannerControls'\).hidden = !physicsSelected\(\)/);
  assert.doesNotMatch(html, /live Plan shot button is unchanged|This study does not change the live Plan shot button/);
});


test('pending acceptance explains the active placement wait without exposing final scores', () => {
  const report = fixture(), section = report.interactive_fast_acceptance;
  section.status = 'running'; section.audit_status = 'pending';
  section.next = 'Plan and drop each frozen placement once per method.';
  section.positions = [
    { case: 'case-00', planner: 'classic', state: 'completed', message: 'Old completed message' },
    { case: 'case-01', planner: 'fast', state: 'ready', message: 'Checked shot ready; waiting for the 4090 before requesting its drop.' },
    { case: 'case-02', planner: 'classic', state: 'pending' },
  ];
  const view = interactiveFastAcceptanceSummary(report), $ = render(report);
  assert.equal(view.complete, false);
  assert.equal(view.next, 'case-01 · Faster planner: Checked shot ready; waiting for the 4090 before requesting its drop.');
  assert.match($('interactiveFastAcceptanceNext').text, /waiting for the 4090/);
  assert.equal($('interactiveFastAcceptanceTable').children.length, 0);
  section.positions[1].message = '';
  assert.equal(interactiveFastAcceptanceSummary(report).next, section.next);
  section.positions[1].message = 'Stale wait'; section.status = 'completed'; section.audit_status = 'passed';
  assert.equal(interactiveFastAcceptanceSummary(report).next, section.next);
});
