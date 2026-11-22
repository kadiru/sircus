import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { hybridMissCorrectionSummary } from '../web/interactive-physics.mjs';

const app = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
const source = `function showHybridMissCorrection() {${app.split('function showHybridMissCorrection() {')[1].split('function showStalledReference()')[0]}`;
const hash = 'a'.repeat(64);
const assets = { 'case03-original': { url: 'data:video/mp4;base64,AAAA', trace_sha256: hash }, 'case03-corrected': { url: 'data:video/mp4;base64,BBBB', trace_sha256: hash } };
class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.textContent = ''; this.hidden = false; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(key, value) { this[key] = value; }
  get text() { return this.textContent + this.children.map(node => node.text).join(' '); }
  get videos() { return [...(this.tag === 'video' ? [this] : []), ...this.children.flatMap(node => node.videos)]; }
}
function render(report, videos = assets) {
  const nodes = new Map(), $ = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  $('hybridMissVideoAssets').textContent = JSON.stringify(videos);
  runInNewContext(`${source}\nshowHybridMissCorrection();`, { report, hybridMissCorrectionSummary, $, document: { createElement: tag => new Element(tag) } });
  return $;
}
function fixture() {
  const original = { requested: 3, completed: 3, released: 3, captures: 0, direct: 0, table: 0, robot_contacts: 3, outcomes: ['flight_miss', 'flight_miss', 'flight_miss'] };
  const corrected = { requested: 3, completed: 3, released: 3, captures: 3, direct: 3, table: 0, robot_contacts: 0, outcomes: ['direct', 'direct', 'direct'] };
  const settings = { board_xy_m: [.6, -.1], board_height_m: .38, tilt_deg: 28, drop_gap_m: .3868, q10: [0, 1, 0, 0, 0, 1], q14: [0, 0, 1, 0, 1, 0] };
  return { hybrid_miss_correction: { version: 'sircus-hybrid-miss-correction-v1', status: 'completed', phase: 'completed', audit_status: 'passed', physics_substeps: 64,
    correction: { parameter: 'joint_configuration', label: 'Change the board-arm joint configuration', description: 'Keep the board pose, release point and bucket unchanged.' },
    progress: { completed: 36, planned: 36 }, controls: { status: 'passed', completed: 18, planned: 18 },
    cases: [[.9484147228, -.0350750276], [1.1950241752, .1605393061], [1.2260114072, .1834252171]].map((xy, i) => ({
      id: `case${i}`, bucket_xy_m: xy,
      diagnosis: { summary: 'The recorded rebound hit the board arm.', evidence_sha256: hash, events: [{ label: 'board contact', time_s: .2 }, { label: 'board-arm contact', time_s: .4 }] },
      modification: 'Same shot geometry; alternative board-arm elbow bend.',
      settings: { original: structuredClone(settings), corrected: { ...structuredClone(settings), q14: [1, 0, 1, 0, 1, 0] } },
      results: { original: structuredClone(original), corrected: structuredClone(corrected) },
      videos: i === 0 ? { original: { asset_id: 'case03-original', label: 'Original native64 miss', trace_sha256: hash }, corrected: { asset_id: 'case03-corrected', trace_sha256: hash } } : {},
    })), next: 'Keep successful historical shots; no planner promotion.' } };
}

test('audited paired repeats keep setup declines in denominators and show exact settings and native videos', () => {
  const report = fixture(), case1 = report.hybrid_miss_correction.cases[1].results.corrected;
  Object.assign(case1, { released: 0, captures: 0, direct: 0, robot_contacts: 0, outcomes: ['setup_rejected', 'setup_rejected', 'setup_rejected'] });
  const view = hybridMissCorrectionSummary(report, assets), $ = render(report);
  assert.equal(view.complete, true);
  assert.equal(view.cases[1].results.corrected.rejected, 3);
  assert.match($('hybridMissResults').text, /948.4, -35.1 0 \/ 3 3 \/ 3 0 \/ 0 3 \/ 0/);
  assert.match($('hybridMissResults').text, /1195.0, 160.5 0 \/ 3 0 \/ 3 0 \/ 3/);
  assert.match($('hybridMissCases').text, /0.200 s: board contact → 0.400 s: board-arm contact/);
  assert.match($('hybridMissCases').text, /Drop gap \(mm\) 386.8 386.8/);
  assert.match($('hybridMissCases').text, /Board-arm joints \(rad\) 0.00000/);
  assert.equal($('hybridMissCases').videos.length, 2);
  assert.equal($('hybridMissCases').videos[0].preload, 'none');
  assert.equal($('hybridMissCases').videos[0].controls, true);
});

test('waiting for GPU shows saved diagnosis and original movie without correction captures', () => {
  const report = fixture(), section = report.hybrid_miss_correction;
  section.status = 'running'; section.phase = 'waiting_for_4090'; section.audit_status = 'pending';
  section.progress = { completed: 0, planned: 36 }; section.controls = { status: 'not_run', completed: 0, planned: 18 };
  section.cases.forEach(row => { delete row.results; delete row.settings.corrected; });
  const $ = render(report);
  assert.match($('hybridMissStatus').text, /Waiting for the shared 4090/);
  assert.equal($('hybridMissResults').children.length, 0);
  assert.equal($('hybridMissCases').videos.length, 1);
  assert.match($('hybridMissCases').text, /Original native64 miss/);
  assert.match($('hybridMissCases').text, /Corrected footage awaits an executed, verified drop/);
  assert.match($('hybridMissCases').text, /Not prepared/);
});

test('incomplete audit, controls, non64 physics or inconsistent outcomes withhold final counts', () => {
  for (const mutate of [
    s => { s.status = 'running'; }, s => { s.audit_status = 'pending'; }, s => { s.physics_substeps = 128; },
    s => { s.controls.status = 'failed'; }, s => { s.controls.completed = 17; },
    s => { s.cases[1].id = s.cases[0].id; }, s => { s.cases[1].bucket_xy_m = s.cases[0].bucket_xy_m; },
    s => { s.cases[1].bucket_xy_m[0] = Infinity; }, s => { s.cases.pop(); },
    s => { s.cases[0].results.corrected.captures = 4; }, s => { s.cases[0].results.corrected.requested = 2; },
    s => { s.cases[0].results.corrected.outcomes[0] = 'predicted_capture'; },
    s => { s.cases[0].results.corrected.outcomes[0] = 'flight_miss'; },
    s => { s.cases[0].results.original.robot_contacts = 4; },
  ]) {
    const report = fixture(); mutate(report.hybrid_miss_correction);
    assert.equal(hybridMissCorrectionSummary(report, assets).complete, false);
    const $ = render(report);
    assert.equal($('hybridMissResults').children.length, 0);
    assert.ok($('hybridMissCases').videos.length <= 1);
  }
});

test('video sources require matching saved-trace identity and bounded inline MP4 data', () => {
  const report = fixture();
  for (const value of [
    { url: 'https://unrelated.example/video.mp4', trace_sha256: hash },
    { url: 'javascript:alert(1)', trace_sha256: hash },
    { url: 'data:text/html;base64,AAAA', trace_sha256: hash },
    { url: 'data:video/mp4;base64,AAAA', trace_sha256: 'b'.repeat(64) },
    { url: 'data:video/mp4;base64,' + 'A'.repeat(14 * 1024 * 1024), trace_sha256: hash },
  ]) {
    assert.equal(hybridMissCorrectionSummary(report, { 'case03-original': value }).cases[0].videos.original, null);
  }
  assert.equal(hybridMissCorrectionSummary(report, assets).cases[0].videos.original.trace_sha256, hash);
  report.hybrid_miss_correction.cases[0].diagnosis.evidence_sha256 = 'unknown';
  assert.equal(hybridMissCorrectionSummary(report, assets).cases[0].diagnosis, null);
  assert.match(render(report)('hybridMissCases').text, /diagnosis is not yet verified/);
});

test('robot-assisted outcomes stay separate from direct and table capture counts', () => {
  const report = fixture(), row = report.hybrid_miss_correction.cases[0].results.corrected;
  Object.assign(row, { captures: 2, direct: 1, table: 1, robot_contacts: 1, outcomes: ['direct', 'table', 'arm_assisted_capture'] });
  assert.equal(hybridMissCorrectionSummary(report).complete, true);
  assert.match(render(report)('hybridMissResults').text, /0 \/ 3 2 \/ 3/);
});

test('historical pages omit optional panel and documentation distinguishes local retest from generalization', () => {
  assert.equal(hybridMissCorrectionSummary({}), null);
  assert.equal(hybridMissCorrectionSummary({ hybrid_miss_correction: { version: 'unknown' } }), null);
  assert.equal(render({})('hybridMissCorrection').hidden, true);
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const section = html.split('<section id="hybridMissCorrection"')[1].split('</section>')[0];
  assert.match(section, /local correction test, not a new unseen-position score/);
  assert.match(section, /64 substeps with both arms held/);
  assert.match(section, /one continuous second fully inside/);
  assert.match(section, /not generated illustrations or new drops/);
  assert.match(section, /id="hybridMissVideoAssets" type="application\/json">\{\}/);
});
