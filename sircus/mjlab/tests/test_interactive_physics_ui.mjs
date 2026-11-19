import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { InteractivePhysics, hasNativePreview, physicsViewerResult, physicsProgress, savedPhysicsRecordings, clearanceExpansionSummary } from '../web/interactive-physics.mjs';
const xy = [.6, -.4];
const preview = () => ({ status: 'feasible', bucket_xy_m: xy, configuration: { board_center_m: [.6, 0, .4], board_normal: [0, 0, 1], board_quaternion_wxyz: [1, 0, 0, 0], release_center_m: [.6, 0, .7], q10: [0, 1], q14: [1, 0], link_positions10: [[0, 0, 0], [.6, 0, .7]], link_positions14: [[1.3, 0, 0], [.6, 0, .4]] }, prediction: { predicted_route: 'table' }, provenance: { native_cache_sha256: 'original' } });
const response = (data, status = 200) => ({ ok: status >= 200 && status < 300, status, json: async () => data });
const flush = () => new Promise(resolve => setImmediate(resolve));
function harness() {
  const requests = [], changes = [], timers = [];
  let handler = async url => url === '/api/interactive-physics' ? response({ enabled: true, bucket_bounds: { x: [0, 1.4], y: [-.8, .3] } }) : response({ id: 'job', state: 'queued' });
  const client = new InteractivePhysics({ fetcher: (url, options) => { requests.push({ url, options }); return handler(url, options); }, onChange: v => changes.push(v), schedule: f => { timers.push(f); return f; }, cancel: f => { const i = timers.indexOf(f); if (i >= 0) timers.splice(i, 1); } });
  return { client, requests, changes, timers, set handler(v) { handler = v; } };
}
async function ready(h) {
  await h.client.initialize();
  h.handler = async url => response(url.endsWith('/plan') ? { id: 'job', state: 'queued' } : { id: 'job', state: 'ready', bucket_xy_m: xy, preview: preview() });
  await h.client.plan(xy); await flush();
  assert.equal(h.client.canDrop, true);
}
test('API discovery alone never plans or drops; older servers leave historical UI available', async () => {
  const h = harness(); await h.client.initialize();
  assert.equal(h.client.available, true); assert.equal(h.client.canDrop, false);
  assert.equal(h.requests.length, 1); assert.equal(h.requests[0].options.method, undefined);
  const missing = harness(); missing.handler = async () => response({}, 404);
  assert.equal(await missing.client.initialize(), false); assert.equal(missing.changes.length, 0);
});
test('plan submits only exact SI bucket coordinates and requires both native FK poses', async () => {
  const h = harness(); await ready(h);
  const post = h.requests.find(r => r.url.endsWith('/plan'));
  assert.deepEqual(JSON.parse(post.options.body), { bucket_xy_m: xy });
  const missing = preview(); delete missing.configuration.link_positions14;
  assert.equal(hasNativePreview(missing), false);
  assert.equal(hasNativePreview({ ...preview(), status: 'unresolved' }), false);
  assert.equal(hasNativePreview(preview()), true);
  assert.equal(h.requests.some(r => r.url.endsWith('/drop')), false);
});
test('double Plan clicks submit once while the first POST is pending', async () => {
  const h = harness(); await h.client.initialize(); let resolve;
  h.handler = () => new Promise(r => { resolve = r; });
  const first = h.client.plan(xy); assert.equal(await h.client.plan(xy), false);
  assert.equal(h.requests.filter(r => r.options.method === 'POST').length, 1);
  h.client.invalidate(); resolve(response({ id: 'old', state: 'queued' })); await first;
  assert.equal(h.client.canDrop, false);
});
test('scene edits while planning discard late ready responses and never enable Drop', async () => {
  const h = harness(); await h.client.initialize(); let resolvePoll;
  h.handler = async url => url.endsWith('/plan') ? response({ id: 'old', state: 'queued' }) : new Promise(r => { resolvePoll = r; });
  await h.client.plan(xy); h.client.invalidate('Bucket edited');
  resolvePoll(response({ id: 'old', state: 'ready', bucket_xy_m: xy, preview: preview() })); await flush();
  assert.equal(h.client.snapshot.state, 'stale'); assert.equal(h.client.canDrop, false);
  assert.equal(h.timers.length, 0);
});
test('ready plan invalidation blocks drop after target, grid, path, or mode replacement', async () => {
  for (const reason of ['Bucket edited', 'Recorded table selected', 'Path preview selected', 'Method changed']) {
    const h = harness(); await ready(h); h.client.invalidate(reason);
    assert.equal(await h.client.drop(), false); assert.equal(h.requests.some(r => r.url.endsWith('/drop')), false);
  }
});
test('one explicit drop sends empty body once and measured completion cannot auto-repeat', async () => {
  const h = harness(); await ready(h); let acceptDrop;
  h.handler = async url => url.endsWith('/drop') ? new Promise(r => { acceptDrop = r; }) : response({ id: 'job', state: 'completed', bucket_xy_m: xy, result: { ...preview(), status: 'completed', capture: false, outcome: 'flight_miss', trajectory: [{ time: 0, ball: [.6, 0, .7] }, { time: 1, ball: [.8, -.5, .1] }], provenance: { substeps: 64 } } });
  const first = h.client.drop(); assert.equal(await h.client.drop(), false);
  const request = h.requests.find(r => r.url.endsWith('/drop')); assert.equal(request.options.body, '{}');
  acceptDrop(response({ id: 'job', state: 'running' }, 202)); await first; await flush();
  assert.equal(h.client.snapshot.state, 'completed'); assert.equal(h.client.snapshot.result.capture, false);
  assert.equal(await h.client.drop(), false); assert.equal(h.requests.filter(r => r.url.endsWith('/drop')).length, 1);
});
test('edited scene suppresses a late completed drop instead of replaying wrong target', async () => {
  const h = harness(); await ready(h); let resolveDrop;
  h.handler = () => new Promise(r => { resolveDrop = r; });
  const pending = h.client.drop(); h.client.invalidate('Scene replaced'); resolveDrop(response({ id: 'job', state: 'running' }, 202));
  await pending; assert.equal(h.client.snapshot.state, 'stale'); assert.equal(h.client.canDrop, false);
});
test('mismatched job or bucket, missing FK, expired plan and HTTP failures cannot enable drop', async () => {
  const cases = [
    { id: 'wrong', state: 'ready', bucket_xy_m: xy, preview: preview() },
    { id: 'job', state: 'ready', bucket_xy_m: [.7, -.4], preview: preview() },
    { id: 'job', state: 'ready', bucket_xy_m: xy, preview: { ...preview(), configuration: null } },
    { id: 'job', state: 'expired', bucket_xy_m: xy },
  ];
  for (const job of cases) {
    const h = harness(); await h.client.initialize();
    h.handler = async url => response(url.endsWith('/plan') ? { id: 'job', state: 'queued' } : job);
    await h.client.plan(xy); await flush(); assert.equal(h.client.canDrop, false);
  }
  const h = harness(); await ready(h); h.handler = async () => response({ error: 'Unavailable' }, 503);
  assert.equal(await h.client.drop(), false); assert.equal(h.client.snapshot.state, 'failed');
});
test('renderer adapter retains measured trajectory/provenance and never invents preview flight', () => {
  const p = preview(), mapped = physicsViewerResult({ ...p, trajectory: [{ time: 0, ball: [0, 0, 1] }] }, 'abc');
  assert.deepEqual(mapped.configuration.board_center, p.configuration.board_center_m);
  assert.deepEqual(mapped.configuration.release_center, p.configuration.release_center_m);
  assert.deepEqual(mapped.configuration.link_positions14, p.configuration.link_positions14);
  assert.deepEqual(mapped.trajectory, []); assert.equal(mapped.success, false);
  assert.equal(mapped.provenance.native_cache_sha256, 'original');
  const result = { ...p, status: 'completed', outcome: 'table', capture: true, trajectory: [{ time: 0, ball: [.6, 0, .7] }] };
  assert.deepEqual(physicsViewerResult(result, 'abc').trajectory, result.trajectory);
});
test('viewer wiring invalidates on raw edits and historical loads before response; default physics requires capability', () => {
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  assert.match(js, /if \(!preservePhysics && physics\?\.available\) physics\.invalidate/);
  assert.match(js, /addEventListener\('input', \(\) => \{ if \(physics\?\.available\) \{[\s\S]*?clearResult\('Bucket edited/);
  assert.match(js, /clearResult\('Loading recorded shot'\);\s+setPending\(true\)/);
  assert.match(js, /clearResult\('Sampled FK route/);
  assert.match(js, /if \(physics\.available\)\s*\{[\s\S]*?\$\('mode'\)\.value = 'physics'/);
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  assert.match(html, /id="physicsOption" value="physics" hidden disabled/);
  assert.match(html, /id="physicsDrop"[^>]*disabled>Run one drop/);
});

test('progress exposes only human candidate/time counts, never internal phases or runtime details', () => {
  assert.equal(physicsProgress({ progress: { proposed: 12, wait_s: 2.9, elapsed_s: 4.2, status: 'checking_base', reason: 'busy pid 12345', message: 'Planning…' } }), '12 candidates · Waiting 2 s · Elapsed 4 s');
  assert.equal(physicsProgress({ progress: { proposed: NaN, wait_s: -1, arbitrary: 'internal' } }), '');
});

test('historical server does not require the new controller asset', () => {
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  assert.doesNotMatch(js, /import .* from '\/interactive-physics.mjs'/);
  assert.match(js, /await import\('\/interactive-physics.mjs'\)/);
  assert.ok(js.indexOf("if (!response.ok || (await response.json()).enabled !== true) return") < js.indexOf("await import('/interactive-physics.mjs')"));
});

test('plan preserves exact grid and drag target coordinates despite rounded input display', () => {
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  assert.match(js, /const xy = finiteVector\(\[number\('bucketX'\), number\('bucketY'\)\], 2\) \? bucketXY.slice\(\)/);
});

function recording() {
  return { id: 'case-04', label: 'Northeast miss', explanation: 'Missed the rim.', markers: [{ time: .4, label: 'Board contact' }], result: {
    ...preview(), status: 'completed', capture: false, outcome: 'flight_miss', scalar_score_audit_passed: true,
    provenance: { substeps: 64, native_release_id: 'native-04', native_cache_sha256: 'a'.repeat(64) },
    trajectory: [{ time: 0, ball: [.6, 0, .7] }, { time: .5, ball: [.8, -.5, .1] }],
  } };
}
const recordedReport = rows => ({ interactive_recordings: { version: 'sircus-interactive-recordings-v1', recordings: rows } });
test('saved replays retain original missed outcome, trajectory, exact coordinates and provenance', () => {
  const source = recording(), [row] = savedPhysicsRecordings(recordedReport([source]));
  assert.equal(row.result, source.result);
  assert.equal(row.result.capture, false);
  assert.deepEqual(row.markers, [{ time: .4, label: 'Board contact' }]);
  const rendered = physicsViewerResult(row.result, row.id);
  assert.equal(rendered.trajectory, source.result.trajectory);
  assert.equal(rendered.provenance, source.result.provenance);
  assert.equal(rendered.success, false);
});
test('saved replays reject unaudited, malformed and unsupported native recordings', () => {
  const corruptions = [
    row => { row.result.status = 'feasible'; },
    row => { row.result.bucket_xy_m = [NaN, 0]; },
    row => { row.result.configuration.link_positions14 = []; },
    row => { row.result.provenance.substeps = 128; },
    row => { delete row.result.provenance.native_release_id; },
    row => { row.result.provenance.native_cache_sha256 = 'unknown'; },
    row => { row.result.scalar_score_audit_passed = false; },
    row => { row.result.trajectory = []; },
    row => { row.result.trajectory[1].time = 0; },
    row => { row.result.trajectory[1].ball = [0, Infinity, 1]; },
  ];
  for (const corrupt of corruptions) { const row = recording(); corrupt(row); assert.deepEqual(savedPhysicsRecordings(recordedReport([row])), []); }
  assert.deepEqual(savedPhysicsRecordings({}), []);
  assert.deepEqual(savedPhysicsRecordings({ interactive_recordings: { version: 'unknown', recordings: [recording()] } }), []);
});
test('saved replay markers stay within the recorded interval and duplicate IDs cannot select another record', () => {
  const row = recording(); row.markers.push({ time: 9, label: 'Outside' }, { time: NaN, label: 'Invalid' }, { time: .2, label: 42 });
  const rows = savedPhysicsRecordings(recordedReport([row, recording()]));
  assert.equal(rows.length, 1); assert.deepEqual(rows[0].markers, [{ time: .4, label: 'Board contact' }]);
});
test('saved replay blocks busy runs, clears the live plan, uses recorded poses and cannot submit a drop', () => {
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  const body = js.split('function replaySavedPhysics(row) {')[1].split("$('savedReplayLoad').onclick")[0];
  assert.match(body, /if \(pending \|\| physics\?\.busy/);
  assert.match(body, /clearResult\('Loading saved native shot'\)/);
  assert.match(body, /setBucket\(row.result.bucket_xy_m.slice\(\)\)/);
  assert.match(body, /applyResult\(physicsViewerResult\(row.result, row.id\)/);
  assert.match(body, /\$\('speed'\).value = '0.25'/);
  assert.doesNotMatch(body, /fetch\(|\.plan\(|\.drop\(|POST|requestSolve/);
});

function clearanceReport() {
  return { clearance_expansion_study: { status: 'completed', phase: 'completed', total_positions: 50,
    prepared_positions: 50, completed_releases: 96, planned_releases: 96, audit_status: 'passed',
    comparison: {
      methods: {
        current_planner: { positions: 50, admitted: 47, released: 47, captures: 37, unresolved: 3, flight_misses: 10, robot_contacts: 6 },
        clearance_expansion: { positions: 50, admitted: 49, released: 49, captures: 41, unresolved: 1, flight_misses: 8, robot_contacts: 4 },
      },
      paired: { gained: 6, lost: 2, both: 35, neither: 7, changed_positions: 11 },
      latency: { current_planner: { median_s: 18, max_s: 29 }, clearance_expansion: { median_s: 22, max_s: 91 } },
    } } };
}
test('clearance comparison retains all requested positions in capture denominator and preserves route-independent counts', () => {
  const view = clearanceExpansionSummary(clearanceReport());
  assert.equal(view.total, 50); assert.equal(view.complete, true); assert.equal(view.audit, 'passed');
  assert.equal(view.methods[0].captures, 37); assert.equal(view.methods[1].unresolved, 1);
  assert.equal(view.methods[0].positions, 50); assert.equal(view.methods[0].released, 47);
  assert.equal(view.methods[1].median_s, 22); assert.equal(view.paired.gained, 6); assert.equal(view.paired.lost, 2);
});
test('clearance progress never turns missing counts or incomplete runs into zero results or completion', () => {
  assert.equal(clearanceExpansionSummary({}), null);
  const missing = clearanceExpansionSummary({ clearance_expansion_study: { status: 'running', prepared_positions: 4, total_positions: 50 } });
  assert.equal(missing.complete, false); assert.equal(missing.prepared, 4); assert.equal(missing.completed, null);
  assert.equal(missing.methods[0].captures, null); assert.equal(missing.paired, null);
  const pending = clearanceReport(); pending.clearance_expansion_study.status = 'running';
  assert.equal(clearanceExpansionSummary(pending).complete, false);
});
test('clearance result rejects impossible totals and inconsistent paired outcomes', () => {
  for (const invalid of [null, -1, NaN, Infinity, 51, '41']) {
    const report = clearanceReport(); report.clearance_expansion_study.comparison.methods.clearance_expansion.captures = invalid;
    assert.equal(clearanceExpansionSummary(report).complete, false);
  }
  const badUnresolved = clearanceReport(); badUnresolved.clearance_expansion_study.comparison.methods.current_planner.unresolved = 0;
  assert.equal(clearanceExpansionSummary(badUnresolved).complete, false);
  const badPair = clearanceReport(); badPair.clearance_expansion_study.comparison.paired.both = 36;
  assert.equal(clearanceExpansionSummary(badPair).paired, null);
});
test('clearance display stays outside historical panels and only reads saved study measurements', () => {
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  assert.ok(html.indexOf('id="clearanceExpansionStudy"') < html.indexOf('id="historicalStudies"'));
  const body = js.split('function showClearanceExpansion() {')[1].split('function replaySavedPhysics')[0];
  assert.match(body, /clearanceExpansionSummary\(report\)/);
  assert.match(body, /view\.complete/);
  assert.doesNotMatch(body, /fetch\(|\.plan\(|\.drop\(|POST|requestSolve/);
});
