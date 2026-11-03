import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Quaternion, Vector3 } from '../../simulation/web/vendor/three.module.js';
import { friendlyError, buildRequest, studyReady, coverageRows, methodLabel, coverageSummary, comparisonRows, learningPoints, inStudiedRegion, BOARD_BODY_SIZE, bankComparisonRows, policyRouteLabel, poseInputValues, bankCoverageRows, bankCoverageStatus, formatMetric, solveButtonState } from '../web/state.mjs';
import { plannerReady, plannerOnly, searchReady, modeReady, plannerFarmRows } from '../web/state.mjs';

test('only an explicit standalone flag hides historical panels without removing controls', () => {
  const combined = { planner: { ready: true, version: 'sircus-corrected-sweep-planner-v1' },
    frozen_evaluation: { attempts: 10 }, learning_curve: [{ update: 1 }] };
  assert.equal(plannerOnly({ ...combined, planner_only: true }), true);
  for (const report of [null, {}, combined, { ...combined, planner_only: false }, { planner_only: 'true' }]) {
    assert.equal(plannerOnly(report), false);
  }
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  assert.ok(html.indexOf('id="plannerFarmGrid"') < html.indexOf('id="historicalStudies"'));
  for (const id of ['coverage', 'comparisonTable', 'captureChart', 'rewardChart']) {
    assert.ok(html.indexOf(`id="${id}"`) > html.indexOf('id="historicalStudies"'));
  }
  for (const id of ['bucketX', 'bucketY', 'boardLock', 'dropLock', 'mode', 'solve', 'searchBudget']) {
    assert.ok(html.indexOf(`id="${id}"`) > html.indexOf('<aside>'));
  }
  assert.ok(js.includes("$('historicalStudies').hidden = standalone"));
  assert.ok(js.includes("'Sircus · Sweep shot planner'"));
  assert.ok(js.includes("for (const id of ['solve', 'solveTop']) $(id).onclick = requestSolve"));
  assert.ok(js.includes('if (!standalone) showCoverage()'));
});

test('a separately ready sweep planner never enables the learned policy', () => {
  const report = { status: 'running', planner: { ready: true, version: 'sircus-corrected-sweep-planner-v1' } };
  assert.equal(plannerReady(report), true);
  assert.equal(searchReady(report), true);
  assert.equal(modeReady(report, 'search'), true);
  assert.equal(modeReady(report, 'policy'), false);
  assert.equal(solveButtonState(report, false, 'search').disabled, false);
  assert.equal(solveButtonState(report, false).disabled, true);
  assert.equal(solveButtonState(report, true, 'search').disabled, true);
  assert.equal(modeReady(report, 'physical'), false);
});
test('invalid present planner capability fails closed while legacy studies retain search', () => {
  const legacy = { status: 'completed', frozen_evaluation: { attempts: 1 } };
  assert.equal(searchReady(legacy), true);
  for (const planner of [null, {}, { ready: 'true', version: 'sircus-corrected-sweep-planner-v1' }, { ready: true, version: 'legacy' }]) {
    assert.equal(searchReady({ ...legacy, planner }), false);
  }
});
test('search budgets stay explicit, bounded and absent from policy requests', () => {
  const args = [[.65, -.4], false, [], false, [], 'id'];
  assert.equal(buildRequest('search', ...args, 64).search_budget, 64);
  assert.equal(Object.hasOwn(buildRequest('policy', ...args), 'search_budget'), false);
  for (const budget of [true, 15, 129, 32.5, NaN]) assert.throws(() => buildRequest('search', ...args, budget));
  assert.throws(() => buildRequest('policy', ...args, 64));
});
test('farm selector preserves exact coordinates and does not fabricate cached success', () => {
  const rows = plannerFarmRows({ planner: { farm_cells: [
    { id: 'first', bucket_xy_m: [.1, -.65], status: 'qualified', cache_available: true },
    { id: 'second', bucket_xy_m: [.3, -.65], status: 'unresolved' },
    { id: 'bad', bucket_xy_m: [NaN, 0], cache_available: true },
  ] } });
  assert.deepEqual(rows.map(r => [r.id, r.xy, r.cached]), [['first', [.1, -.65], true], ['second', [.3, -.65], false]]);
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  assert.ok(html.includes('id="plannerFarmGrid"'));
  assert.ok(html.includes('single-cell view'));
  assert.ok(js.includes('Sweep planner · corrected native MuJoCo · endpoint-held'));
  assert.ok(js.includes('Saved verified shot · recorded native evidence'));
  assert.ok(js.includes("modeReady(report, $('mode').value)"));
  assert.match(methodLabel('sweep_planner'), /corrected native MuJoCo/);
});

test('manual solve gates until a frozen evaluation is complete', () => {
  assert.equal(studyReady({ status: 'running', frozen_evaluation: { trials: 5 } }), false);
  assert.equal(studyReady({ status: 'completed', frozen_evaluation: {} }), false);
  assert.equal(studyReady({ status: 'completed', frozen_evaluation: { trials: 5 } }), true);
});
test('position locks are explicit ball-center and board-center values in metres', () => {
  const board = [.65, 0, .4], release = [.65, 0, .7];
  const request = buildRequest('policy', [.65, -.4], true, board, false, release, 'a');
  assert.deepEqual(request.constraints, { board_position: board, dropper_position: null });
  assert.equal(request.mode, 'policy');
  assert.throws(() => buildRequest('robot', [0, 0], false, [], false, [], 'a'));
  assert.throws(() => buildRequest('policy', [NaN, 0], false, [], false, [], 'a'));
});
test('search and saved settings stay visibly distinct from policy inference', () => {
  assert.match(methodLabel('search'), /not policy inference/);
  assert.match(methodLabel('dataset_lookup'), /not policy inference/);
  assert.equal(methodLabel('frozen_policy'), 'Frozen learned policy');
});
test('coverage keeps unresolved and heldout layouts', () => {
  assert.deepEqual(coverageRows({ coverage: { cells: [
    { id: 'train', bucket_xy_m: [.5, -.3], successes: 2, attempts: 8 },
    { id: 'test', xy: [.7, -.6], split: 'test', captures: 0, trials: 3 },
  ] } }).map(r => [r.id, r.captures, r.trials]), [['train', 2, 8], ['test', 0, 3]]);
});
test('UI has explicit simulation and provisional geometry labels', () => {
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  for (const label of ['SIMULATION ONLY', 'interior depth remains provisional', 'Frozen learned policy', 'Search / refinement', 'Ball-center XYZ']) assert.ok(html.includes(label));
  assert.ok(!html.includes('sfhack10'));
});

test('coverage confirmation is counted separately from explored captures', () => {
  const report = { coverage: { cells: [
    { bucket_xy_m: [.1, -.65], captures: 2, trials: 9, confirmation: { attempts: 3, captures: 3 } },
    { bucket_xy_m: [.5, -.3], captures: 1, trials: 9, confirmation: { attempts: 3, captures: 2 } },
    { bucket_xy_m: [.7, 0], captures: 0, trials: 9 },
  ] } };
  assert.deepEqual(coverageSummary(report), { total: 3, resolved: 2, unresolved: 1, confirmed: 1, confirmationTrials: 6 });
});
test('actual frozen comparison schema preserves all checkpoints and lock conditions', () => {
  const comparison = Object.fromEntries(['initial', 'warm', 'selected', 'board_locked', 'dropper_locked'].map((key, i) => [key, { attempts: 100, captures: i, invalid: 9 - i, mean_reward: i / 10, layouts: 50 }]));
  const rows = comparisonRows({ frozen_evaluation: { comparison } });
  assert.equal(rows.length, 5);
  assert.equal(rows[2].label, 'Selected frozen policy');
  assert.equal(rows[4].captures, 4);
  assert.equal(rows[4].invalid, 5);
  assert.equal(rows[4].meanReward, .4);
  assert.equal(rows[4].attempts, 100);
});
test('learning curves keep training reward separate from validation reward', () => {
  const points = learningPoints({ learning_curve: [
    { update: 1, attempts: 64, captures: 3, mean_reward: .4 },
    { update: 2, attempts: 64, captures: 4, mean_reward: .5, validation: { mean_reward: .1 } },
  ] });
  assert.deepEqual(points[0], { update: 1, captures: 3, attempts: 64, reward: .4, validationReward: null });
  assert.equal(points[1].validationReward, .1);
  assert.equal(points[1].reward, .5);
});
test('studied region and privileged endpoint scope stay explicit', () => {
  assert.equal(inStudiedRegion([.1, -.65]), true);
  assert.equal(inStudiedRegion([1.24, .25]), true);
  assert.equal(inStudiedRegion([1.25, .25]), false);
  assert.equal(inStudiedRegion([.1, -.66]), false);
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  for (const label of ['100–1240 mm', '−650–250 mm', 'endpoints initialized', 'privileged', 'Frozen', 'comparisonTable', 'captureChart', 'rewardChart']) {
    assert.ok(html.toLowerCase().includes(label.toLowerCase()), label);
  }
});

test('native board quaternion applies landscape mount once to URDF body dimensions', () => {
  const q = new Quaternion(0, 0, Math.sin(Math.PI / 4), Math.cos(Math.PI / 4));
  const corners = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    corners.push(new Vector3(sx * BOARD_BODY_SIZE[0] / 2, sy * BOARD_BODY_SIZE[1] / 2, sz * BOARD_BODY_SIZE[2] / 2).applyQuaternion(q));
  }
  const extent = axis => Math.max(...corners.map(v => v[axis])) - Math.min(...corners.map(v => v[axis]));
  assert.ok(Math.abs(extent('x') - .3) < 1e-12, 'native Rz90 yields 300mm world-X extent');
  assert.ok(Math.abs(extent('y') - .2) < 1e-12, 'native Rz90 yields 200mm world-Y extent');
  assert.ok(Math.abs(extent('z') - .005) < 1e-12);
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  assert.ok(js.includes('box(BOARD_BODY_SIZE'));
  assert.ok(js.includes('board.quaternion.set(x, y, z, w).normalize()'));
});

test('relative-bank evaluation is separate from the original continuous split', () => {
  const report = { status: 'completed', frozen_evaluation: { comparison: { selected: { attempts: 100, captures: 43, invalid: 1, mean_reward: .8 } } },
    bank_evaluation: { status: 'running', comparison: {
      continuous_v2: { attempts: 50, captures: 20, invalid: 2, mean_reward: .7 },
      bank_uniform: { attempts: 50, captures: 25, invalid: 0, mean_reward: 1.2 },
      bank_selected: { attempts: 50, captures: 30, invalid: 1, mean_reward: 1.4 },
    } } };
  assert.equal(studyReady(report), false, 'do not unlock before the new study finishes');
  report.bank_evaluation.status = 'completed';
  assert.equal(studyReady(report), true);
  const bank = bankComparisonRows(report);
  assert.equal(bank.length, 3);
  assert.equal(bank[0].label, 'Continuous v2 · fresh split');
  assert.equal(bank[2].captures, 30);
  assert.equal(comparisonRows(report)[0].captures, 43, 'original evidence is not replaced');
});
test('free versus constrained model routing and geometry prior stay explicit', () => {
  const report = { bank_evaluation: { status: 'completed' } };
  assert.match(policyRouteLabel(report, false, false), /relative-bank selector with a geometry prior/);
  assert.match(policyRouteLabel(report, true, false), /continuous v2 policy/);
  assert.match(policyRouteLabel(report, false, true), /continuous v2 policy/);
  assert.match(methodLabel('bank_policy'), /geometry prior/);
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  assert.ok(html.includes('fresh held-out split'));
  assert.ok(html.includes('Uniform bank proposals'));
  assert.ok(html.includes('bankComparison'));
});

test('actual solved coordinates prefill editable pose inputs at 0.1mm', () => {
  assert.deepEqual(poseInputValues([.650123, -.00056, .400067]), ['650.1', '-0.6', '400.1']);
  assert.equal(poseInputValues([NaN, 0, 0]), null);
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  assert.ok(js.includes("!$(prefix + 'Lock').checked && values"), 'locked inputs are not overwritten');
  assert.ok(js.includes("['attempts', 'Native attempts']"));
});
test('fresh test map distinguishes policy misses from no eligible candidates', () => {
  const report = { bank_evaluation: { coverage: [
    { bucket_xy_m: [.2, -.3], captures: 1, trials: 2, eligible_candidates: 8 },
    { bucket_xy_m: [.4, -.5], captures: 0, trials: 2, eligible_candidates: 8 },
    { bucket_xy_m: [.8, -.2], captures: 0, trials: 2, eligible_candidates: 0 },
  ] } };
  assert.deepEqual(bankCoverageRows(report).map(bankCoverageStatus), ['captured', 'missed', 'unresolved']);
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  assert.ok(html.includes('id="coverage"'), 'original training coverage preserved');
  assert.ok(html.includes('id="bankCoverage"'), 'fresh map is separate');
  assert.ok(html.includes('fresh test subset is separate'));
});

test('visible canvas solve and original solve share gating, pending and handler', () => {
  const ready = { status: 'completed', frozen_evaluation: { attempts: 2 } };
  assert.deepEqual(solveButtonState(ready, false), { disabled: false, mainLabel: 'Find drop-bounce posture ', topLabel: 'Find posture ↗' });
  assert.equal(solveButtonState(ready, true).disabled, true);
  assert.equal(solveButtonState(ready, true).topLabel, 'Solving…');
  assert.equal(solveButtonState({}, false).disabled, true);
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  assert.ok(html.includes('id="solveTop"'));
  assert.ok(html.includes('id="solve"'));
  assert.ok(js.includes("for (const id of ['solve', 'solveTop']) $(id).onclick = requestSolve"));
  assert.ok(js.includes("for (const id of ['solve', 'solveTop']) $(id).disabled = state.disabled"));
});
test('operator metrics use compact units and readable capture quality', () => {
  assert.equal(formatMetric('tilt_deg', 41.28168398674323), '41.3°');
  assert.equal(formatMetric('retention_s', .9999999999997), '1.00 s');
  assert.equal(formatMetric('board_bounces', 1), '1');
  assert.equal(formatMetric('quality', 2), 'Clean capture');
  assert.equal(formatMetric('quality', 1), 'Assisted capture');
  assert.equal(formatMetric('quality', 0), 'No capture');
});

test("busy solver errors hide the Python traceback",()=>{assert.equal(friendlyError("RuntimeError: Native failed: Traceback\n file path\nRuntimeError: 4090 is busy"),"4090 is busy");});
