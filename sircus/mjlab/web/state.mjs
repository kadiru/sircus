/** Browser-independent request and report helpers. Coordinates are metres. */
// URDF impact-board BODY axes. The native quaternion already includes the
// landscape mount's Rz(90°); swapping X/Y again would rotate the footprint twice.
export const BOARD_BODY_SIZE = Object.freeze([.2, .3, .005]);
export function studyReady(report) {
  return report?.status === 'completed' && !!report.frozen_evaluation &&
    typeof report.frozen_evaluation === 'object' && !Array.isArray(report.frozen_evaluation) &&
    Object.keys(report.frozen_evaluation).length > 0 &&
    (!report.bank_evaluation || report.bank_evaluation.status === 'completed');
}
export function finiteVector(values, length) {
  return Array.isArray(values) && values.length === length && values.every(x => typeof x === 'number' && Number.isFinite(x));
}
export function friendlyError(value) {
  const lines = String(value || 'Simulation request failed').split('\n').map(x => x.trim()).filter(Boolean);
  return (lines.at(-1) || 'Simulation request failed').replace(/^(RuntimeError|ValueError|OSError):\s*/, '');
}
export function buildRequest(mode, bucket, boardLock, board, dropLock, drop, requestId) {
  if (!['policy', 'search'].includes(mode)) throw Error('Unknown inference mode');
  if (!finiteVector(bucket, 2)) throw Error('Enter a valid bucket position');
  if (boardLock && !finiteVector(board, 3)) throw Error('Enter a valid board center');
  if (dropLock && !finiteVector(drop, 3)) throw Error('Enter a valid release center');
  return { mode, bucket_xy: bucket, constraints: {
    board_position: boardLock ? board : null,
    dropper_position: dropLock ? drop : null,
  }, request_id: requestId };
}
export function methodLabel(method) {
  if (['bank_policy', 'relative_bank_policy', 'bank_selected'].includes(method)) return 'Frozen relative-bank policy · geometry prior';
  if (['frozen_policy', 'ppo', 'scene_conditioned_ppo', 'policy'].includes(method)) return 'Frozen learned policy';
  if (['search', 'refinement', 'search_refinement'].includes(method)) return 'Per-scene search · not policy inference';
  if (['catalogue', 'catalog_lookup', 'cached', 'dataset_lookup'].includes(method)) return 'Saved setting lookup · not policy inference';
  return method || 'Solver method not reported';
}
export function coverageRows(report) {
  const values = report?.coverage?.cells ?? report?.coverage?.grid ?? report?.coverage_grid ?? report?.layouts ?? [];
  if (!Array.isArray(values)) return [];
  return values.map((row, i) => ({ ...row, id: row.id ?? row.layout_id ?? i,
    xy: row.bucket_xy ?? row.bucket_xy_m ?? row.xy,
    captures: row.captures ?? row.successes ?? 0,
    trials: row.trials ?? row.attempts ?? 0,
  })).filter(row => finiteVector(row.xy, 2));
}
export function coverageSummary(report) {
  const rows = coverageRows(report);
  const resolved = rows.filter(r => r.captures > 0 || r.success === true).length;
  const confirmed = rows.filter(r => Number(r.confirmation?.attempts) > 0 && Number(r.confirmation?.captures) === Number(r.confirmation?.attempts)).length;
  const confirmationTrials = rows.reduce((sum, r) => sum + (Number(r.confirmation?.attempts) || 0), 0);
  return { total: rows.length, resolved, unresolved: rows.length - resolved, confirmed, confirmationTrials };
}
export function comparisonRows(report) {
  const comparison = report?.frozen_evaluation?.comparison ?? {};
  const labels = { initial: 'Initial / untrained', warm: 'Imitation warm start', selected: 'Selected frozen policy', board_locked: 'Selected · board XYZ locked', dropper_locked: 'Selected · release XYZ locked' };
  return Object.entries(labels).filter(([key]) => comparison[key]).map(([key, label]) => {
    const row = comparison[key];
    return { key, label, attempts: Number(row.attempts) || 0, captures: Number(row.captures) || 0, invalid: Number(row.invalid) || 0,
      meanReward: Number.isFinite(row.mean_reward) ? row.mean_reward : null, layouts: Number(row.layouts) || 0, constraintSource: row.constraint_source || '' };
  });
}
export function learningPoints(report) {
  return (Array.isArray(report?.learning_curve) ? report.learning_curve : []).filter(r => Number.isFinite(r.update) && r.update >= 0).map(r => ({
    update: r.update, captures: Number(r.captures) || 0, attempts: Number(r.attempts) || 0,
    reward: Number.isFinite(r.mean_reward) ? r.mean_reward : null,
    validationReward: Number.isFinite(r.validation?.mean_reward) ? r.validation.mean_reward : null,
  }));
}
export function inStudiedRegion(xy) {
  return finiteVector(xy, 2) && xy[0] >= .1 && xy[0] <= 1.24 && xy[1] >= -.65 && xy[1] <= .25;
}

export function bankComparisonRows(report) {
  const comparison = report?.bank_evaluation?.comparison ?? {};
  const labels = { continuous_v2: 'Continuous v2 · fresh split', bank_uniform: 'Uniform relative-bank prior', bank_selected: 'Learned bank selector · frozen' };
  return Object.entries(labels).filter(([key]) => comparison[key]).map(([key, label]) => {
    const row = comparison[key];
    return { key, label, attempts: Number(row.attempts) || 0, captures: Number(row.captures) || 0,
      invalid: Number(row.invalid) || 0, meanReward: Number.isFinite(row.mean_reward) ? row.mean_reward : null };
  });
}
export function policyRouteLabel(report, boardLocked, dropperLocked) {
  const bankReady = report?.bank_evaluation?.status === 'completed';
  if (bankReady && !boardLocked && !dropperLocked) return 'Free positions: frozen relative-bank selector with a geometry prior, then one simulation. No per-query rollout search.';
  if (bankReady) return 'Locked positions: continuous v2 policy, then one simulation. The relative-bank selector is used only with both position locks off.';
  return 'One learned prediction, then an independent simulation. No optimization at inference.';
}

export function poseInputValues(position) {
  return finiteVector(position, 3) ? position.map(v => (v * 1000).toFixed(1)) : null;
}
export function bankCoverageRows(report) {
  const rows = report?.bank_evaluation?.coverage;
  if (!Array.isArray(rows)) return [];
  return rows.map(row => ({ ...row, xy: row.bucket_xy_m ?? row.bucket_xy,
    captures: Number(row.captures) || 0, trials: Number(row.trials) || 0,
    eligible: Number.isFinite(row.eligible_candidates) ? row.eligible_candidates : null,
  })).filter(row => finiteVector(row.xy, 2));
}
export function bankCoverageStatus(row) {
  if (row.captures > 0) return 'captured';
  return row.eligible > 0 ? 'missed' : 'unresolved';
}

export function formatMetric(key, value) {
  if (key === 'quality') return ({ 0: 'No capture', 1: 'Assisted capture', 2: 'Clean capture' })[value] ?? 'Unknown';
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (key === 'tilt_deg') return `${value.toFixed(1)}°`;
    if (key === 'retention_s') return `${value.toFixed(2)} s`;
  }
  return String(value);
}
export function solveButtonState(report, pending) {
  return { disabled: pending || !studyReady(report),
    mainLabel: pending ? 'Solving and simulating… ' : 'Find drop-bounce posture ',
    topLabel: pending ? 'Solving…' : 'Find posture ↗' };
}

/** Read-only study display: these fields never change the active solver route. */
export function expansionTargetRows(report) {
  const search = report?.expansion_evaluation?.search;
  const rows = search?.cells?.length ? search.cells : (search?.development_layouts ?? []);
  if (!Array.isArray(rows)) return [];
  return rows.map(row => ({
    id: String(row.id ?? 'Unidentified target'), xy: row.bucket_xy_m ?? row.bucket_xy,
    captures: Number(row.search?.captures) || 0, attempts: Number(row.search?.attempts) || 0,
    retainedSources: Number(row.retained_source_captures) || 0,
    promoted: !!row.promotion, phase: row.promotion?.development_phase ?? '',
    sourceId: row.promotion?.source_trial_id ?? '',
    confirmationCaptures: Number(row.promotion?.confirmation?.captures) || 0,
    confirmationAttempts: Number(row.promotion?.confirmation?.attempts) || 0,
    unresolvedReason: row.unresolved_reason || 'No independently confirmed template yet',
  })).filter(row => finiteVector(row.xy, 2));
}
export function expansionProgress(report) {
  const value = report?.expansion_evaluation;
  if (!value || typeof value !== 'object') return null;
  const search = value.search, training = value.training, plan = training?.study_plan ?? {};
  const targets = expansionTargetRows(report);
  const queue = training?.gpu_queue ?? search?.waiting_for_4090;
  return {
    status: String(value.status ?? 'not_started').replaceAll('_', ' '),
    searchPhase: search?.phase ?? 'Not started', trainingPhase: training?.phase ?? 'Waiting for a qualified expanded bank',
    activePolicy: 'Previous bank remains active; expanded model not deployed',
    targetCount: targets.length, promoted: targets.filter(row => row.promoted).length,
    nativeSearchAttempts: Number(search?.effort?.native_search_attempts_total) ||
      (Number(search?.native_search_attempts) || 0) + (Number(search?.prior_native_search_attempts) || 0),
    confirmationAttempts: Number(search?.effort?.native_confirmation_attempts_total) ||
      (Number(search?.confirmation_attempts) || 0) + (Number(search?.prior_confirmation_attempts) || 0),
    geometryProposals: Number.isFinite(search?.geometry_proposals) ? search.geometry_proposals : null,
    trainAttempts: Number(training?.train_episodes) || 0,
    nativeTrainReleases: Number(training?.native_train_releases) || 0,
    updates: Number(training?.updates) || 0, updateBudget: Number(plan.ppo_updates) || null,
    trainLayouts: Number(plan.train_layouts) || null, validationLayouts: Number(plan.validation_layouts) || null,
    testLayouts: Number(plan.test_layouts) || null, testRepeats: Number(plan.test_repeats) || null,
    eligibleContexts: training?.train_sampling?.eligible_contexts ?? null,
    excludedContexts: training?.train_sampling?.excluded_contexts ?? null,
    geometry: training?.geometry_progress ?? null,
    queued: queue?.status === 'waiting_for_4090', waitSeconds: Number(queue?.wait_s) || 0,
    training,
  };
}
export function expandedComparisonRows(report) {
  const study = report?.expansion_evaluation?.training;
  if (study?.status !== 'completed') return [];
  const comparison = study.frozen_evaluation?.comparison ?? {};
  const labels = { old_uniform: 'Old bank · uniform', old_learned: 'Old bank · learned',
    expanded_uniform: 'Expanded bank · uniform', expanded_learned: 'Expanded bank · learned',
    old_uniform_expanded_bounds: 'Old bank · uniform / wider bounds',
    old_uniform_new_bounds: 'Old bank · uniform / wider bounds' };
  return Object.entries(labels).filter(([key]) => comparison[key]).map(([key, label]) => {
    const row = comparison[key];
    return { key, label, attempts: Number(row.attempts) || 0, captures: Number(row.captures) || 0, clean: Number(row.clean) || 0,
      invalid: Number(row.invalid) || 0, layouts: Number(row.layouts) || 0, repeats: Number(row.repeats) || 0,
      eligibleAttempts: Number(row.eligible_attempts) || 0,
      conditionalCaptureRate: Number.isFinite(row.conditional_capture_rate) ? row.conditional_capture_rate : null,
      positionsWithCapture: Number(row.positions_with_capture) || 0,
      unresolvedPositions: Number(row.unresolved_positions) || 0,
      meanReward: Number.isFinite(row.mean_reward) ? row.mean_reward : null };
  });
}
