import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { expansionTargetRows, expansionProgress, expandedComparisonRows, studyReady, bankComparisonRows, comparisonRows } from '../web/state.mjs';

test('expansion targets keep exploratory results separate from perturbed confirmations', () => {
  const report = { expansion_evaluation: { status: 'search_running', search: { cells: [
    { id: 'old-test-02', bucket_xy_m: [.23, -.41], search: { captures: 2, attempts: 256 },
      promotion: { source_trial_id: 'expand-shot-10', development_phase: 'wide', confirmation: { captures: 8, attempts: 10 } } },
    { id: 'old-test-11', bucket_xy_m: [.15, .2], search: { captures: 0, attempts: 256 }, unresolved_reason: 'Still unresolved' },
    { id: 'invalid', bucket_xy_m: [NaN, .2] },
  ] } } };
  const rows = expansionTargetRows(report);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].attempts, 256);
  assert.equal(rows[0].captures, 2);
  assert.equal(rows[0].confirmationCaptures, 8);
  assert.equal(rows[0].confirmationAttempts, 10);
  assert.equal(rows[0].sourceId, 'expand-shot-10');
  assert.equal(rows[1].promoted, false);
  assert.equal(rows[1].unresolvedReason, 'Still unresolved');
  const status = expansionProgress(report);
  assert.equal(status.promoted, 1);
  assert.equal(status.targetCount, 2);
  assert.match(status.activePolicy, /Previous bank remains active/);
});

test('missing expansion is optional and in-flight study does not reroute existing policy', () => {
  assert.equal(expansionProgress({}), null);
  assert.deepEqual(expansionTargetRows({}), []);
  assert.deepEqual(expandedComparisonRows({}), []);
  const original = { status: 'completed', frozen_evaluation: { comparison: { selected: { attempts: 100, captures: 43 } } },
    bank_evaluation: { status: 'completed', comparison: { bank_selected: { attempts: 100, captures: 60 } } },
    expansion_evaluation: { status: 'training_running', active_policy: 'Unverified value', training: { status: 'running',
      phase: 'waiting-for-4090', gpu_queue: { status: 'waiting_for_4090', wait_s: 125, reason: [123] },
      frozen_evaluation: { comparison: { expanded_learned: { captures: 999 } } },
      geometry_progress: { group: 'expanded-train', completed: 12, total: 175 },
      study_plan: { train_layouts: 175, validation_layouts: 30, test_layouts: 100, test_repeats: 3, ppo_updates: 32 } } } };
  assert.equal(studyReady(original), true);
  assert.equal(comparisonRows(original)[0].captures, 43);
  assert.equal(bankComparisonRows(original)[0].captures, 60);
  assert.deepEqual(expandedComparisonRows(original), [], 'partial results are not a completed frozen comparison');
  const progress = expansionProgress(original);
  assert.equal(progress.geometryProposals, null, 'unreported join effort is unknown, not zero');
  assert.equal(progress.activePolicy, 'Previous bank remains active; expanded model not deployed');
  assert.equal(progress.queued, true);
  assert.equal(progress.waitSeconds, 125);
  assert.equal(progress.testLayouts, 100);
  assert.equal(progress.testRepeats, 3);
  assert.equal(progress.geometry.completed, 12);
});

test('new frozen comparison retains all requests, eligibility and baseline methods', () => {
  const keys = ['old_uniform', 'old_learned', 'expanded_uniform', 'expanded_learned', 'old_uniform_expanded_bounds'];
  const comparison = Object.fromEntries(keys.map((key, i) => [key, { attempts: 300, captures: i * 30, clean: i * 20, invalid: i,
    eligible_attempts: 240, conditional_capture_rate: i / 8, positions_with_capture: i * 10,
    unresolved_positions: 20, layouts: 100, repeats: 3 }]));
  const rows = expandedComparisonRows({ expansion_evaluation: { training: { status: 'completed', frozen_evaluation: { comparison } } } });
  assert.equal(rows.length, 5);
  assert.equal(rows[3].attempts, 300);
  assert.equal(rows[3].captures, 90);
  assert.equal(rows[3].clean, 60);
  assert.equal(rows[3].eligibleAttempts, 240);
  assert.equal(rows[3].conditionalCaptureRate, 3 / 8);
  assert.equal(rows[3].unresolvedPositions, 20);
  assert.equal(rows[3].layouts, 100);
  assert.equal(rows[3].repeats, 3);
  assert.match(rows[4].label, /wider bounds/);
  comparison.old_uniform_new_bounds = comparison.old_uniform_expanded_bounds;
  delete comparison.old_uniform_expanded_bounds;
  assert.equal(expandedComparisonRows({ expansion_evaluation: { training: { status: 'completed', frozen_evaluation: { comparison } } } }).length, 5);
});

test('continuation and joined effort counts do not hide or double-count prior throws', () => {
  const report = { expansion_evaluation: { search: {
    native_search_attempts: 0, confirmation_attempts: 710,
    prior_native_search_attempts: 6144, prior_confirmation_attempts: 290,
  } } };
  assert.equal(expansionProgress(report).nativeSearchAttempts, 6144);
  assert.equal(expansionProgress(report).confirmationAttempts, 1000);
  report.expansion_evaluation.search.effort = {
    native_search_attempts_total: 15360, native_confirmation_attempts_total: 1100,
  };
  assert.equal(expansionProgress(report).nativeSearchAttempts, 15360);
  assert.equal(expansionProgress(report).confirmationAttempts, 1100);
});

test('expansion panel is separate, labels development honestly and does not change solve handlers', () => {
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  for (const id of ['expandedStudy', 'expansionTargets', 'expansionComparison', 'expansionCaptureChart', 'expansionRewardChart']) assert.ok(html.includes(`id="${id}"`));
  for (const text of ['Previous bank remains active', 'formerly unresolved OLD test positions', 'Conditional capture rate', 'evaluation never optimizes per scene']) assert.ok(html.includes(text));
  for (const id of ['comparisonTable', 'bankComparison', 'captureChart', 'rewardChart']) assert.ok(html.includes(`id="${id}"`));
  assert.ok(js.includes("for (const id of ['solve', 'solveTop']) $(id).onclick = requestSolve"));
  assert.ok(js.includes('showComparison(); showBankComparison(); showExpansion();'));
});
