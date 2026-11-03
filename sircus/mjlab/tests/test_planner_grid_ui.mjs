import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { plannerGridRows, plannerGridSummary, plannerGridCsv, plannerGridResultPath } from '../web/state.mjs';

const cells = () => Array.from({ length: 35 }, (_, i) => ({
  id: `dev-grid-${Math.floor(i / 7)}-${i % 7}`,
  bucket_xy_m: [.1 + i % 7 * .19, -.65 + Math.floor(i / 7) * .225],
  status: 'queued', stage: 'initial', nominal: null, fresh: null,
}));
const fixture = rows => ({ planner_grid_study: { version: 'sircus-planner-grid-study-v1', status: 'running', cells: rows } });

test('all 35 bucket positions remain visible while pending repeats stay unknown', () => {
  const report = fixture(cells()), rows = plannerGridRows(report), summary = plannerGridSummary(report);
  assert.equal(rows.length, 35);
  assert.equal(new Set(rows.map(row => JSON.stringify(row.xy))).size, 35);
  assert.deepEqual(rows.map(row => row.label).slice(-2), ['5·6', '5·7']);
  assert.ok(rows.every(row => row.nominal === null && row.fresh === null && row.selected === null));
  assert.equal(summary.finished, 0);
  assert.equal(summary.qualified, 0);
  assert.equal(summary.nominal, null);
  assert.equal(summary.fresh, null);
  assert.equal(summary.frozenFresh, null);
});

test('saved evidence and newly repeated selected shots keep separate denominators', () => {
  const all = cells();
  all[0] = { ...all[0], status: 'saved_verified', selected: { source: 'saved_reference' },
    saved_nominal: { captures: 3, attempts: 3 }, saved_fresh: { captures: 19, attempts: 20 } };
  all[1] = { ...all[1], status: 'verified_fresh', nominal: { captures: 3, attempts: 3 }, fresh: { captures: 18, attempts: 20 },
    selected: { source: 'fresh_study', board_center_m: [.61, -.41, .26], release_ball_center_m: [.61, -.41, .49], tilt_deg: 37.2, gap_m: .1925, distance_m: .24 } };
  all[2] = { ...all[2], status: 'unresolved', nominal: { captures: 2, attempts: 3 }, fresh: { captures: 0, attempts: 20 } };
  all[3] = { ...all[3], status: 'blocked' };
  const report = fixture(all), rows = plannerGridRows(report), summary = plannerGridSummary(report);
  assert.deepEqual(rows[0].savedFresh, { captures: 19, attempts: 20 });
  assert.equal(rows[0].fresh, null);
  assert.equal(rows[0].selected.source, 'Saved reference');
  assert.equal(rows[1].selected.source, 'Fresh study');
  assert.deepEqual(rows[1].selected.board, [.61, -.41, .26]);
  assert.equal(summary.finished, 4);
  assert.equal(summary.qualified, 2);
  assert.equal(summary.freshQualified, 1);
  assert.equal(summary.savedQualified, 1);
  assert.equal(summary.unresolved, 2);
  assert.deepEqual(summary.nominal, { captures: 5, attempts: 6, reported: 2 });
  assert.deepEqual(summary.fresh, { captures: 18, attempts: 40, reported: 2 });
});

test('invalid or unavailable counts never become observed zeros or percentages', () => {
  const values = [null, {}, { captures: 2, attempts: 1 }, { captures: 1, attempts: 2.5 }, { captures: -1, attempts: 20 }, { captures: false, attempts: 3 }];
  for (const value of values) {
    const row = plannerGridRows(fixture([{ id: 'bad', nominal: value, fresh: value, search: { native_requests: NaN }, quality: { clean: '0' } }]))[0];
    assert.equal(row.nominal, null);
    assert.equal(row.fresh, null);
    assert.equal(row.search.nativeRequests, null);
    assert.equal(row.quality.clean, null);
  }
  const row = plannerGridRows(fixture([{ nominal: { captures: 0, attempts: 20 }, fresh: { captures: 0, attempts: 0 } }]))[0];
  assert.deepEqual(row.nominal, { captures: 0, attempts: 20 });
  assert.deepEqual(row.fresh, { captures: 0, attempts: 0 });
});

test('recorded replay paths remain a narrow same-origin read-only endpoint', () => {
  assert.equal(plannerGridResultPath('/api/planner-grid/result/dev-grid-00-01'), '/api/planner-grid/result/dev-grid-00-01');
  for (const value of [null, '//host/api/planner-grid/result/a', 'https://host/api/planner-grid/result/a', '/api/solve', '/api/planner-grid/result/../a', '/api/planner-grid/result/%2E', '/api/planner-grid/result/a?x=y']) {
    assert.equal(plannerGridResultPath(value), null);
  }
});

test('verified envelope blocker notes replace only display reasons, not signed native data', () => {
  const native = { version: 'sircus-planner-grid-study-v1', cells: [{ id: 'blocked', status: 'blocked', reason: 'Finite search stage ended' }] };
  const original = JSON.stringify(native), note = 'Prescribed starting pose is blocked; another staging strategy needs validation.';
  const report = { planner_grid_study: native, planner_grid_blockers: { blocked: note } };
  assert.equal(plannerGridRows(report)[0].reason, note);
  assert.equal(JSON.stringify(native), original);
  assert.equal(plannerGridRows({ ...report, planner_grid_blockers: { blocked: null } })[0].reason, 'Finite search stage ended');
  assert.equal(plannerGridRows({ ...report, planner_grid_blockers: {} })[0].reason, 'Finite search stage ended');
  assert.ok(plannerGridCsv(report).includes(note));
});

test('frozen validation is independent of qualification and saved repeats', () => {
  const row = { id: 'checked', status: 'saved_verified', saved_nominal: { captures: 3, attempts: 3 }, saved_fresh: { captures: 20, attempts: 20 },
    frozen_validation: { status: 'completed', nominal: { captures: 1, attempts: 3 }, fresh: { captures: 2, attempts: 30 }, configuration_sha256: 'a'.repeat(64),
      quality: { clean: 1, assisted: 1, cohort: 'fresh', source: 'frozen_validation' } } };
  const report = fixture([row]), normalized = plannerGridRows(report)[0], summary = plannerGridSummary(report);
  assert.deepEqual(normalized.savedFresh, { captures: 20, attempts: 20 });
  assert.equal(normalized.fresh, null);
  assert.deepEqual(normalized.frozen.fresh, { captures: 2, attempts: 30 });
  assert.equal(normalized.frozen.configurationSha, 'a'.repeat(64));
  assert.equal(normalized.frozen.quality.cohort, 'fresh');
  assert.equal(normalized.frozen.quality.source, 'frozen_validation');
  assert.equal(summary.fresh, null);
  assert.deepEqual(summary.frozenFresh, { captures: 2, attempts: 30, reported: 1 });
  assert.deepEqual(summary.frozenNominal, { captures: 1, attempts: 3, reported: 1 });
});

test('CSV exports exact settings with explicit study versus saved cohorts and every unresolved cell', () => {
  const all = cells();
  all[0] = { ...all[0], id: '=unsafe', reason: 'still unresolved', selected: { source: 'fresh_study', board_center_m: [.6193376779556274, -.4343374967575073, .2566870152950287],
    release_ball_center_m: [.6193377971368164, -.43433738363365904, .48903120111277915], tilt_deg: 37.21897657594422, gap_m: .19484428737039663, distance_m: .243 },
    fresh: { captures: 0, attempts: 20 }, saved_fresh: { captures: 20, attempts: 20 } };
  const text = plannerGridCsv(fixture(all)), lines = text.trimEnd().split('\n');
  assert.equal(lines.length, 36);
  assert.match(lines[0], /study_fresh_captures/);
  assert.match(lines[0], /saved_fresh_captures/);
  assert.match(lines[0], /search_captures/);
  assert.match(lines[0], /search_cpu_candidates/);
  assert.match(lines[0], /frozen_quality_cohort/);
  assert.ok(lines[1].includes('0.48903120111277915'));
  assert.ok(lines[1].includes('0.19484428737039663'));
  assert.ok(lines[1].includes("'=unsafe"));
  assert.ok(!text.includes('undefined') && !text.includes('NaN'));
  assert.equal(lines.filter(line => line.includes('queued')).length, 35);
});

test('study UI is outside historical panels and loading a record cannot launch native work', () => {
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  assert.ok(html.indexOf('id="plannerGridStudy"') < html.indexOf('id="historicalStudies"'));
  for (const label of ['Qualification nominal', 'Qualification perturbed', 'Frozen validation', 'Clean / assisted', 'Board XYZ mm', 'Ball release Z mm']) assert.ok(js.includes(label));
  assert.ok(html.includes('saved-reference counts are labeled separately'));
  assert.ok(html.includes('Ball release Z refers to the ball center'));
  const replayFunction = js.slice(js.indexOf('async function viewPlannerGridResult'), js.indexOf('function showPlannerGridStudy'));
  assert.ok(replayFunction.includes('fetch(row.resultUrl'));
  assert.ok(!replayFunction.includes('/api/solve') && !replayFunction.includes('POST'));
  assert.ok(replayFunction.includes('applyResult(result, { recordedPlayback: true })'));
  assert.ok(js.includes('showPlannerGridStudy();'));
  assert.ok(js.includes("typeof report.planner_grid_error === 'string'"));
  assert.ok(js.includes('Study evidence unavailable'));
});
