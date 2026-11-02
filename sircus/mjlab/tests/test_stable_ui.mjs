import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stableIkProgress, stableIkTargets, stableIkAuditRows, studyReady, expandedComparisonRows } from '../web/state.mjs';

test('IK study remains optional and cannot change readiness or previous comparisons', () => {
  const report = { status: 'completed', frozen_evaluation: { captures: 1 },
    expansion_evaluation: { training: { status: 'completed', frozen_evaluation: { comparison: {
      expanded_learned: { captures: 188, attempts: 300, layouts: 100, repeats: 3 },
    } } } }, stable_ik_evaluation: { status: 'running', phase: 'waiting-for-4090', no_training: true,
      waiting_for_4090: { status: 'waiting_for_4090', wait_s: 45 },
      native_search_attempts: 0, confirmation_attempts: 0, cells: [] } };
  assert.equal(stableIkProgress({}), null);
  assert.deepEqual(stableIkTargets({}), []);
  assert.deepEqual(stableIkAuditRows({}), []);
  assert.equal(studyReady(report), true);
  assert.equal(expandedComparisonRows(report)[0].captures, 188);
  assert.equal(stableIkProgress(report).queued, true);
  assert.equal(stableIkProgress(report).nativeSearchAttempts, 0);
  assert.equal(stableIkProgress(report).geometryProposals, null, 'unknown is not zero');
  assert.equal(stableIkProgress(report).invariance, null, 'unknown is not passed');
});

test('source search, assisted entries and final confirmations stay distinct', () => {
  const report = { stable_ik_evaluation: { status: 'completed', no_training: true, cells: [
    { id: 'old-test-08', bucket_xy_m: [1.2, -.15], search: { attempts: 100, captures: 2, clean: 1 },
      observed_assisted_entries: 31, promotion: { source_trial_id: 'stable-shot-12',
        confirmation: { captures: 9, attempts: 10 }, final_confirmation: { captures: 17, attempts: 20 } } },
    { id: 'old-test-09', bucket_xy_m: [1.1, -.6], search: { attempts: 100, captures: 0 },
      observed_assisted_entries: 40, unresolved_reason: 'No direct airborne capture' },
    { id: 'bad', bucket_xy_m: [NaN, 0] },
  ] } };
  const rows = stableIkTargets(report);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].captures, 2);
  assert.equal(rows[0].assisted, 31);
  assert.equal(rows[0].firstConfirmation.captures, 9);
  assert.equal(rows[0].finalConfirmation.captures, 17);
  assert.equal(rows[1].promoted, false);
  assert.equal(stableIkProgress(report).qualified, 1);
  assert.equal(stableIkProgress(report).targets, 2);
});

test('order audit reports endpoint masks separately from known branch preservation', () => {
  const report = { stable_ik_evaluation: { solver_audit: {
    comparison: [
      { method: 'old', queries: 6, accepted: 6, order_mask_changes: 4, batch_mask_changes: 0, max_q_delta_rad: 9.38 },
      { method: 'stable_seeded', queries: 6, accepted: 6, order_mask_changes: 0, batch_mask_changes: 0, max_q_delta_rad: 0, preserved_known_branches: 6 },
    ], order_batch_invariance: { passed: true }, known_branch_preservation: { passed: true },
  } } };
  const rows = stableIkAuditRows(report);
  assert.equal(rows[0].orderChanges, 4);
  assert.equal(rows[0].preserved, null);
  assert.equal(rows[1].preserved, 6);
  assert.equal(stableIkProgress(report).invariance, true);
  assert.equal(stableIkProgress(report).preservation, true);
});

test('failed final group remains visible with matching screening source and no promotion', () => {
  const report = { stable_ik_evaluation: { status: 'completed', cells: [{
    id: 'old-test-21',bucket_xy_m:[.16,-.63],promotion:null,
    best_screening:{source_trial_id:'stable-shot-77',captures:8,attempts:10,invalid:1,status:'passed'},
    final_confirmation:{source_trial_id:'stable-shot-77',captures:13,attempts:20,invalid:3,status:'failed'},
    confirmation_summary_scope:'Final-tested source and its own screening group',
  }] } };
  const row=stableIkTargets(report)[0];
  assert.equal(row.promoted,false);
  assert.equal(row.firstConfirmation.captures,8);
  assert.equal(row.finalConfirmation.captures,13);
  assert.equal(row.finalConfirmation.attempts,20);
  assert.equal(row.finalConfirmation.status,'failed');
  assert.equal(row.confirmationSourceId,'stable-shot-77');
  assert.equal(stableIkProgress(report).qualified,0);
});

test('new section is read-only and preserves all existing inference handlers', () => {
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const js = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
  for (const id of ['ikStability', 'stableIkSummary', 'stableIkAudit', 'stableIkTargets']) assert.ok(html.includes(`id="${id}"`));
  for (const text of ['does not train or deploy', '16/20 independent final confirmations', 'Assisted entries', 'not held-out policy performance']) assert.ok(html.includes(text));
  assert.ok(js.includes("for (const id of ['solve', 'solveTop']) $(id).onclick = requestSolve"));
  assert.ok(js.includes('showComparison(); showBankComparison(); showExpansion(); showStableIk();'));
  const stable = js.slice(js.indexOf('function showStableIk()'), js.indexOf('function showComparison()'));
  assert.ok(!/fetch\(|requestSolve\(|onclick|\.click\(/.test(stable));
});
