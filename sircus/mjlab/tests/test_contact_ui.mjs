import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { contactStudy, contactGroups, contactConfigurations, configurationsCsv, routePreview } from '../web/state.mjs';
import { studyReady } from '../web/state.mjs';

test('contact report is optional and does not enable or disable policy controls', () => {
  assert.equal(contactStudy({}), null);
  assert.deepEqual(contactGroups({}), []);
  const report = { status: 'completed', frozen_evaluation: { captures: 1 },
    contact_repeatability_evaluation: { status: 'failed', no_training: true } };
  assert.equal(studyReady(report), true);
  report.status = 'running'; assert.equal(studyReady(report), false);
});

test('unknown denominator stays unknown and explicit zero stays zero', () => {
  const rows = contactGroups({ contact_repeatability_evaluation: { groups: [
    { id: 'a', attempts: 20, captures: 13, invalid: 0 }, { id: 'b', captures: 0 },
  ] } });
  assert.equal(rows[0].attempts, 20); assert.equal(rows[0].captures, 13);
  assert.equal(rows[0].invalid, 0); assert.equal(rows[1].attempts, null);
  assert.equal(rows[1].captures, 0);
});

test('CSV preserves exact joint values and metre-to-mm heights with quoted provenance', () => {
  const report = { contact_repeatability_evaluation: { configurations: [
    { id: 'source,"21"', condition: 'old-final', board_xyz_m: [.47, -.278, .397], bucket_xy_m: [.163, -.637],
      tilt_deg: 36.15479, drop_gap_m: .303, release_center_m: [.47, -.278, .7375],
      q10: [1,2,3,4,5,6], q14: [-1,-2,-3,-4,-5,-6] },
    { id: 'corrupt', board_xyz_m: [NaN,0,0] },
  ] } };
  assert.equal(contactConfigurations(report).length, 1);
  const csv = configurationsCsv(report);
  assert.ok(csv.includes('"source,""21"""'));
  assert.ok(csv.includes('"737.5"')); assert.ok(csv.includes('"q14_6_rad"'));
  assert.ok(csv.includes('"-6"')); assert.equal(csv.split('\r\n').length, 3);
});

test('only accepted routes with finite complete monotonic samples can preview', () => {
  const sample = time => ({ time, board_center: [0,0,.4], board_quaternion_wxyz: [1,0,0,0],
    link_positions10: [[0,0,0],[0,0,1]], link_positions14: [[1,0,0],[1,0,.4]] });
  const route = { accepted: true, frames: [sample(0),sample(1)] };
  assert.equal(routePreview(route).length, 2);
  assert.deepEqual(routePreview({ ...route, accepted: false }), []);
  assert.deepEqual(routePreview({ ...route, frames: [sample(0),sample(0)] }), []);
  assert.deepEqual(routePreview({ ...route, frames: [sample(0),{...sample(1),board_center:[0,NaN,0]}] }), []);
});

test('contact panel is read-only, downloads evidence, and cannot actuate or alter solve routing', () => {
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  for (const id of ['contactRepeatability','contactGroups','contactConfigurations','contactTransit','contactDownload'])
    assert.ok(html.includes(`id="${id}"`));
  const section = js.slice(js.indexOf('function showContactStudy()'), js.indexOf('function showStableIk()'));
  assert.ok(section.includes('configurationsCsv(report)'));
  assert.ok(section.includes('if (pending) return;'));
  assert.ok(!section.includes('fetch(')); assert.ok(!section.includes('/api/solve'));
  assert.ok(html.includes('not physical calibration or policy improvement'));
});
