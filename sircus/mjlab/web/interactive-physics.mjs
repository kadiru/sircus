/** Two-step simulation workflow. No robot, policy, or playback side effects. */
const vector = (v, n) => Array.isArray(v) && v.length === n && v.every(Number.isFinite);
const sameXY = (a, b) => vector(a, 2) && vector(b, 2) && a.every((v, i) => Math.abs(v - b[i]) < 1e-9);
const states = new Set(['queued', 'planning', 'ready', 'no_setup', 'running', 'completed', 'failed', 'expired']);
const terminal = new Set(['ready', 'no_setup', 'completed', 'failed', 'expired']);
export function hasNativePreview(preview) {
  const c = preview?.configuration;
  return preview?.status === 'feasible' && vector(c?.board_center_m, 3) && vector(c?.release_center_m, 3)
    && vector(c?.board_quaternion_wxyz, 4) && c.board_quaternion_wxyz.some(v => v !== 0)
    && ['link_positions10', 'link_positions14'].every(key => Array.isArray(c[key]) && c[key].length >= 2 && c[key].every(p => vector(p, 3)))
    && ['q10', 'q14'].every(key => Array.isArray(c[key]) && c[key].length > 0 && c[key].every(Number.isFinite));
}
export function physicsViewerResult(value, id) {
  const c = value?.configuration || {};
  return { ...value, request_id: id, method: 'interactive_physics', success: value?.capture === true,
    configuration: { ...c, board_center: c.board_center_m, release_center: c.release_center_m },
    metrics: { tilt_deg: c.tilt_deg, retention_s: value?.retention_s },
    trajectory: value?.status === 'completed' && Array.isArray(value.trajectory) ? value.trajectory : [] };
}
export function physicsProgress(snapshot) {
  const p = snapshot?.progress || {}, parts = [];
  if (Number.isInteger(p.proposed) && p.proposed >= 0) parts.push(`${p.proposed} candidates`);
  if (Number.isFinite(p.wait_s) && p.wait_s >= 0) parts.push(`Waiting ${Math.floor(p.wait_s)} s`);
  const elapsed = snapshot?.planning_elapsed_s ?? p.elapsed_s;
  if (Number.isFinite(elapsed) && elapsed >= 0) parts.push(`Elapsed ${Math.floor(elapsed)} s`);
  return parts.join(' · ');
}
/** Read saved native results without planning, simulation, or reconstructing flight. */
export function savedPhysicsRecordings(study) {
  const section = study?.interactive_recordings;
  if (section?.version !== 'sircus-interactive-recordings-v1' || !Array.isArray(section.recordings)) return [];
  const ids = new Set();
  return section.recordings.filter(row => {
    const r = row?.result, trace = r?.trajectory;
    if (typeof row?.id !== 'string' || !row.id || ids.has(row.id) || typeof row.label !== 'string'
        || r?.status !== 'completed' || !vector(r.bucket_xy_m, 2)
        || !hasNativePreview({ ...r, status: 'feasible' }) || typeof r.capture !== 'boolean'
        || r.provenance?.substeps !== 64 || typeof r.provenance?.native_release_id !== 'string'
        || !r.provenance.native_release_id || !/^[a-f0-9]{64}$/i.test(r.provenance?.native_cache_sha256 || '')
        || r.scalar_score_audit_passed !== true || !Array.isArray(trace) || trace.length < 2
        || !trace.every((p, i) => Number.isFinite(p.time) && p.time >= 0 && vector(p.ball, 3)
          && (i === 0 || p.time > trace[i - 1].time))) return false;
    ids.add(row.id); return true;
  }).map(row => ({
    ...row, explanation: typeof row.explanation === 'string' ? row.explanation : '',
    markers: (Array.isArray(row.markers) ? row.markers : []).filter(marker =>
      Number.isFinite(marker?.time) && typeof marker.label === 'string'
      && marker.time >= row.result.trajectory[0].time && marker.time <= row.result.trajectory.at(-1).time),
  }));
}
/** Display only reported measurements; incomplete or inconsistent totals stay pending. */
export function clearanceExpansionSummary(study) {
  const section = study?.clearance_expansion_study;
  if (!section || typeof section !== 'object' || Array.isArray(section)) return null;
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const seconds = value => Number.isFinite(value) && value >= 0 ? value : null;
  const total = count(section.total_positions);
  const labels = { current_planner: 'Current planner', clearance_expansion: 'Wider clearance search' };
  const methods = Object.entries(labels).map(([id, label]) => {
    const raw = section.comparison?.methods?.[id] || {};
    const values = Object.fromEntries(['positions', 'admitted', 'released', 'captures', 'unresolved', 'flight_misses', 'robot_contacts'].map(key => [key, count(raw[key])]));
    const complete = total > 0 && Object.values(values).every(value => value !== null)
      && values.positions === total && values.captures <= values.released && values.released <= values.admitted
      && values.admitted <= total && values.unresolved === total - values.admitted
      && values.flight_misses <= values.released - values.captures && values.robot_contacts <= values.released;
    const timing = section.comparison?.latency?.[id] || {};
    return { id, label, ...values, complete, median_s: seconds(timing.median_s), max_s: seconds(timing.max_s) };
  });
  const rawPair = section.comparison?.paired || {};
  const paired = Object.fromEntries(['gained', 'lost', 'both', 'neither', 'changed_positions'].map(key => [key, count(rawPair[key])]));
  const pairComplete = ['gained', 'lost', 'both', 'neither'].every(key => paired[key] !== null)
    && paired.gained + paired.lost + paired.both + paired.neither === total
    && paired.both + paired.lost === methods[0].captures && paired.both + paired.gained === methods[1].captures;
  const status = typeof section.status === 'string' ? section.status : 'unknown';
  return {
    status, phase: typeof section.phase === 'string' ? section.phase : '', total,
    prepared: count(section.prepared_positions), completed: count(section.completed_releases), planned: count(section.planned_releases),
    methods, paired: pairComplete ? paired : null,
    complete: status === 'completed' && methods.every(row => row.complete),
    audit: typeof section.audit_status === 'string' ? section.audit_status : null,
    scope: typeof section.scope === 'string' ? section.scope : '',
    next: typeof section.next === 'string' ? section.next : '',
    source: typeof section.source === 'string' ? section.source : '',
  };
}
/** Neural-study progress is evidence-only; final scores require the completed audit. */
export function neuralShotSummary(study) {
  const section = study?.neural_shot_study;
  if (!section || typeof section !== 'object' || Array.isArray(section)) return null;
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const seconds = value => Number.isFinite(value) && value >= 0 ? value : null;
  const splits = Object.entries({ train: 'Training', validation: 'Validation', test: 'Final test' }).map(([id, label]) => {
    const progress = section.progress?.[id] || {};
    const positions = count(section.split_positions?.[id]);
    const prepared = count(progress.prepared_positions);
    const released = count(progress.completed_releases), planned = count(progress.planned_releases);
    return { id, label, positions, prepared: prepared !== null && (positions === null || prepared <= positions) ? prepared : null,
      released, planned: planned !== null && (released === null || planned >= released) ? planned : null,
      baseline_prepared: count(progress.baseline_prepared_positions), neural_prepared: count(progress.neural_prepared_positions) };
  });
  const total = splits.find(row => row.id === 'test').positions;
  const labels = { current_planner: 'Current planner', clearance_expansion: 'Wider search', neural_selector: 'Learned selector' };
  const methods = Object.entries(labels).map(([id, label]) => {
    const raw = section.comparison?.methods?.[id] || {};
    const values = Object.fromEntries(['positions', 'admitted', 'released', 'captures', 'unresolved', 'robot_contacts'].map(key => [key, count(raw[key])]));
    const timing = section.comparison?.latency?.[id] || {};
    const median = seconds(timing.median_s), p90 = seconds(timing.p90_s);
    return { id, label, ...values, median_s: median, p90_s: p90 !== null && (median === null || p90 >= median) ? p90 : null,
      valid: total > 0 && Object.values(values).every(value => value !== null) && values.positions === total
        && values.captures <= values.released && values.released <= values.admitted && values.admitted <= total
        && values.unresolved === total - values.admitted && values.robot_contacts <= values.released };
  });
  const status = typeof section.status === 'string' ? section.status : 'unknown';
  const audit = typeof section.audit_status === 'string' ? section.audit_status : null;
  const complete = status === 'completed' && audit === 'passed' && methods.every(row => row.valid);
  return { status, phase: typeof section.phase === 'string' ? section.phase : '', audit, splits, methods, total, complete,
    next: typeof section.next === 'string' ? section.next : '',
    error: typeof section.error === 'string' ? section.error : '',
    gpu_seconds: seconds(section.gpu_lease_s), gpu_budget_seconds: seconds(section.gpu_cap_s) };
}
/** Planning deadlines are separate from native flight outcomes; audited totals only. */
export function planningBudgetSummary(study) {
  const section = study?.planning_budget_study;
  if (!section || typeof section !== 'object' || Array.isArray(section)) return null;
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const seconds = value => Number.isFinite(value) && value >= 0 ? value : null;
  const total = count(section.budgets?.positions), budget = seconds(section.budgets?.planning_s);
  const progress = section.progress || {};
  const prepared = count(progress.prepared), planned = count(progress.planned);
  const releases = count(progress.completed_releases), plannedReleases = count(progress.planned_releases);
  const labels = { current_planner: 'Current planner', neural_selector: 'Learned selector' };
  const methods = Object.entries(labels).map(([id, label]) => {
    const raw = section.comparison?.methods?.[id] || {};
    const values = Object.fromEntries(['attempts', 'releases', 'captures', 'deadline_reached', 'no_candidate', 'robot_contacts'].map(key => [key, count(raw[key])]));
    const timing = section.comparison?.latency?.[id] || {};
    const median = seconds(timing.median_s), p90 = seconds(timing.p90_s), maximum = seconds(timing.max_s);
    const lifecycle = section.comparison?.timing?.[id] || {};
    const matchingRate = (value, numerator) => Number.isFinite(value) && total > 0 && numerator !== null
      && Math.abs(value - numerator / total) < 1e-9;
    return { id, label, ...values, median_s: median,
      startup_median_s: seconds(lifecycle.startup?.median_s),
      end_to_end_median_s: seconds(lifecycle.end_to_end?.median_s),
      p90_s: p90 !== null && (median === null || p90 >= median) ? p90 : null,
      max_s: maximum !== null && (p90 === null || maximum >= p90) && (median === null || maximum >= median) ? maximum : null,
      valid: total > 0 && Object.values(values).every(value => value !== null)
        && values.attempts === total && values.captures <= values.releases && values.releases <= total
        && values.releases + values.no_candidate === total && values.deadline_reached <= total
        && values.robot_contacts <= values.releases
        && matchingRate(raw.capture_rate, values.captures) && matchingRate(raw.setup_rate, values.releases) };
  });
  const rawPair = section.comparison?.paired || {};
  const paired = Object.fromEntries(['gained', 'lost', 'both', 'neither'].map(key => [key, count(rawPair[key])]));
  const pairValid = total > 0 && Object.values(paired).every(value => value !== null)
    && paired.gained + paired.lost + paired.both + paired.neither === total
    && paired.both + paired.lost === methods[0].captures && paired.both + paired.gained === methods[1].captures;
  const status = typeof section.status === 'string' ? section.status : 'unknown';
  const audit = typeof section.audit_status === 'string' ? section.audit_status : null;
  return { status, phase: typeof section.phase === 'string' ? section.phase : '', audit, total, budget,
    prepared: prepared !== null && (planned === null || prepared <= planned) ? prepared : null,
    planned: planned !== null && (total === null || planned === total * methods.length) ? planned : null,
    releases, planned_releases: plannedReleases !== null && (releases === null || plannedReleases >= releases)
      && (total === null || plannedReleases <= total * methods.length) ? plannedReleases : null,
    methods, paired: pairValid ? paired : null,
    complete: status === 'completed' && audit === 'passed' && budget > 0 && methods.every(row => row.valid) && pairValid,
    next: typeof section.next === 'string' ? section.next : '', error: typeof section.error === 'string' ? section.error : '' };
}
/** Read-only worker comparison. Predictions and partial runs are never capture evidence. */
export function fastPlanningSummary(study) {
  const section = study?.fast_planning_study;
  if (!section || typeof section !== 'object' || Array.isArray(section)) return null;
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const seconds = value => Number.isFinite(value) && value >= 0 ? value : null;
  const stats = (raw = {}) => {
    raw = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
    const mean = seconds(raw.mean_s), median = seconds(raw.median_s), p90 = seconds(raw.p90_s), maximum = seconds(raw.max_s);
    return { mean_s: mean !== null && (maximum === null || mean <= maximum) ? mean : null, median_s: median,
      p90_s: p90 !== null && (median === null || p90 >= median) ? p90 : null,
      max_s: maximum !== null && (median === null || maximum >= median) && (p90 === null || maximum >= p90) ? maximum : null,
      aggregate_s: seconds(raw.aggregate_s) };
  };
  const total = count(section.budgets?.positions), budget = seconds(section.budgets?.planning_s);
  const comparison = section.comparison || {};
  const labels = { cold_30: 'Fresh worker · 30s', warm_30: 'Reused worker · 30s', warm_early: 'Reused worker · early return' };
  const stageLabels = { menu: 'Proposing shots', neural_order: 'Neural ranking', arm_setup_checks: 'Checking arm poses and paths',
    physics_ranking: 'Predicting ball flight', evidence_publication: 'Recording checked choices', other: 'Other planner work' };
  const methods = Object.entries(labels).map(([id, label]) => {
    const raw = comparison.methods?.[id] || {};
    const values = Object.fromEntries(['attempts', 'releases', 'captures', 'no_candidate', 'deadline_reached',
      'deadline_with_candidate', 'deadline_no_candidate', 'flight_misses', 'robot_contacts', 'early_return'].map(key => [key, count(raw[key])]));
    const rateMatches = (value, numerator) => Number.isFinite(value) && total > 0 && numerator !== null
      && Math.abs(value - numerator / total) < 1e-9;
    const profile = comparison.profile?.[id] || {};
    const stages = Object.entries(stageLabels).map(([key, name]) => {
      const stage = profile.stages?.[key] || {};
      return { id: key, label: name, aggregate_s: seconds(stage.aggregate_s), mean_s: seconds(stage.mean_s),
        fraction: Number.isFinite(stage.fraction) && stage.fraction >= 0 && stage.fraction <= 1 ? stage.fraction : null };
    });
    const workers = Object.fromEntries(['cold_starts', 'reused_requests', 'restarts'].map(key => [key, count(comparison.workers?.[id]?.[key])]));
    // Missing optional profiler measurements remain unknown; they cannot fabricate a faster result.
    const workersValid = Object.values(workers).every(value => value !== null)
      && workers.cold_starts + workers.reused_requests === total && workers.restarts <= workers.cold_starts;
    return { id, label, ...values, decision: stats(comparison.latency?.[id]),
      request: stats(comparison.timing?.[id]?.request), startup: stats(comparison.timing?.[id]?.startup),
      stages, workers: workersValid ? workers : null,
      valid: total > 0 && Object.values(values).every(value => value !== null)
        && values.attempts === total && values.captures <= values.releases && values.releases <= total
        && values.releases + values.no_candidate === total
        && values.deadline_with_candidate + values.deadline_no_candidate === values.deadline_reached
        && values.deadline_with_candidate <= values.releases && values.deadline_no_candidate <= values.no_candidate
        && values.flight_misses <= values.releases - values.captures && values.robot_contacts <= values.releases
        && values.early_return <= values.releases && values.early_return + values.deadline_with_candidate <= values.releases
        && rateMatches(raw.capture_rate, values.captures) && rateMatches(raw.setup_rate, values.releases) };
  });
  const paired = Object.entries({ warm_vs_cold: [0, 1], early_vs_warm: [1, 2], early_vs_cold: [0, 2] }).map(([id, [base, alternative]]) => {
    const raw = comparison.paired?.[id] || {};
    const values = Object.fromEntries(['both', 'gained', 'lost', 'neither'].map(key => [key, count(raw[key])]));
    const valid = total > 0 && Object.values(values).every(value => value !== null)
      && Object.values(values).reduce((sum, value) => sum + value, 0) === total
      && values.both + values.lost === methods[base].captures && values.both + values.gained === methods[alternative].captures;
    return { id, base: methods[base].label, alternative: methods[alternative].label, ...(valid ? values : {}), valid };
  });
  const progress = section.progress || {};
  const rawPlanned = count(progress.planned), rawPrepared = count(progress.prepared);
  const rawReleases = count(progress.completed_releases), rawPlannedReleases = count(progress.planned_releases);
  const maximum = total === null ? null : total * methods.length;
  const planned = rawPlanned !== null && (maximum === null || rawPlanned === maximum) ? rawPlanned : null;
  const prepared = rawPrepared !== null && (maximum === null || rawPrepared <= maximum)
    && (planned === null || rawPrepared <= planned) ? rawPrepared : null;
  const releases = rawReleases !== null && (maximum === null || rawReleases <= maximum) ? rawReleases : null;
  const plannedReleases = rawPlannedReleases !== null && (maximum === null || rawPlannedReleases <= maximum)
    && (releases === null || rawPlannedReleases >= releases) ? rawPlannedReleases : null;
  const status = typeof section.status === 'string' ? section.status : 'unknown';
  const audit = typeof section.audit_status === 'string' ? section.audit_status : null;
  const accountingValid = methods.every(row => row.valid) && prepared === maximum && planned === maximum
    && releases === methods.reduce((sum, row) => sum + row.releases, 0) && plannedReleases === releases;
  return { status, phase: typeof section.phase === 'string' ? section.phase : '', audit, total, budget,
    prepared, planned, releases, planned_releases: plannedReleases, methods, paired, accounting_valid: accountingValid,
    complete: status === 'completed' && audit === 'passed' && budget > 0 && accountingValid,
    next: typeof section.next === 'string' ? section.next : '', error: typeof section.error === 'string' ? section.error : '' };
}
/** Live API acceptance is separate from the earlier offline planning comparison. */
export function interactiveFastAcceptanceSummary(study) {
  const section = study?.interactive_fast_acceptance;
  if (!section || section.version !== 'sircus-interactive-fast-acceptance-v1') return null;
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const seconds = value => Number.isFinite(value) && value >= 0 ? value : null;
  const methods = Object.entries({ fast: 'Faster planner', classic: 'Original planner' }).map(([id, label]) => {
    const raw = section.comparison?.[id] || {};
    const values = Object.fromEntries(['positions', 'captures', 'releases', 'no_setup', 'failed', 'worker_reused'].map(key => [key, count(raw[key])]));
    return { id, label, ...values, planning_samples: count(raw.planning?.samples),
      planning: Object.fromEntries(['mean_s', 'median_s', 'p90_s', 'max_s'].map(key => [key, seconds(raw.planning?.[key])])),
      backend_request: Object.fromEntries(['mean_s', 'median_s', 'p90_s', 'max_s'].map(key => [key, seconds(raw.backend_request?.[key])])),
      valid: values.positions > 0 && ['captures', 'releases', 'no_setup', 'failed'].every(key => values[key] !== null)
        && values.captures <= values.releases && values.releases + values.no_setup + values.failed === values.positions };
  });
  const paired = Object.fromEntries(['both', 'gained', 'lost', 'neither'].map(key => [key, count(section.paired?.[key])]));
  const total = methods[0].positions;
  const completed = count(section.progress?.completed), planned = count(section.progress?.planned);
  const accountingValid = methods.every(row => row.valid && row.positions === total)
    && planned === total * 2 && completed === planned
    && Object.values(paired).every(value => value !== null)
    && paired.both + paired.gained + paired.lost + paired.neither === total
    && paired.both + paired.gained === methods[0].captures && paired.both + paired.lost === methods[1].captures;
  const complete = section.status === 'completed' && section.audit_status === 'passed' && accountingValid;
  const active = !complete && Array.isArray(section.positions) ? section.positions.find(row =>
    ['plan_intent', 'queued', 'planning', 'ready', 'drop_intent', 'running'].includes(row?.state)
    && methods.some(method => method.id === row.planner)
    && (typeof row.case === 'string' || Number.isInteger(row.case))
    && typeof row.message === 'string' && row.message.trim()) : null;
  const next = active ? `${active.case} · ${methods.find(method => method.id === active.planner).label}: ${active.message.trim()}`
    : typeof section.next === 'string' ? section.next : '';
  return { status: section.status, audit: section.audit_status, methods, paired, total, completed, planned,
    accounting_valid: accountingValid, complete,
    gpu_wait_expired: section.status === 'failed' && typeof section.error === 'string'
      && section.error.includes('4090 remains occupied; saved ready plan can resume without replanning'),
    scope: typeof section.scope === 'string' ? section.scope : '', next,
    error: typeof section.error === 'string' ? section.error : '' };
}
/** Compare equivalent arm checkers; only complete, audited accounting exposes scores. */
export function armCheckSpeedSummary(study) {
  const section = study?.arm_check_speed_study;
  if (section?.version !== 'sircus-arm-check-speed-study-v1') return null;
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const seconds = value => Number.isFinite(value) && value >= 0 ? value : null;
  const total = count(section.budgets?.positions), budget = seconds(section.budgets?.planning_s);
  const comparison = section.comparison || {}, maximum = total === null ? null : total * 2;
  const methods = Object.entries({ baseline: 'Current arm checks', optimized: 'Faster arm checks' }).map(([id, label]) => {
    const raw = comparison.methods?.[id] || {}, stats = comparison.latency?.[id] || {};
    const values = Object.fromEntries(['positions', 'admitted', 'captures', 'flight_misses', 'no_candidate'].map(key => [key, count(raw[key])]));
    const timing = Object.fromEntries(['mean_s', 'median_s', 'p90_s', 'max_s'].map(key => [key, seconds(stats[key])]));
    const validTiming = Object.values(timing).every(value => value !== null) && timing.mean_s <= timing.max_s
      && timing.median_s <= timing.p90_s && timing.p90_s <= timing.max_s;
    return { id, label, ...values, timing: validTiming ? timing : Object.fromEntries(Object.keys(timing).map(key => [key, null])),
      valid: total > 0 && Object.values(values).every(value => value !== null) && values.positions === total
        && values.admitted + values.no_candidate === total && values.captures <= values.admitted
        && values.flight_misses <= values.admitted - values.captures };
  });
  const paired = Object.fromEntries(['both', 'gained', 'lost', 'neither'].map(key => [key, count(comparison.paired?.[key])]));
  const pairValid = Object.values(paired).every(value => value !== null)
    && paired.both + paired.gained + paired.lost + paired.neither === total
    && paired.both + paired.lost === methods[0].captures && paired.both + paired.gained === methods[1].captures;
  const progress = section.progress || {};
  const bounded = value => { const n = count(value); return n !== null && (maximum === null || n <= maximum) ? n : null; };
  const prepared = bounded(progress.prepared), planned = bounded(progress.planned), releases = bounded(progress.completed_releases), plannedReleases = bounded(progress.planned_releases);
  const positions = Array.isArray(comparison.positions) ? comparison.positions : [];
  const captures = new Set(['direct', 'table', 'unclassified_capture']);
  const knownOutcomes = new Set([...captures, 'setup_rejected', 'flight_miss', 'invalid_physics', 'arm_assisted_capture']);
  const positionRowsValid = positions.length === total && new Set(positions.map(row => row?.id)).size === total
    && new Set(positions.map(row => JSON.stringify(row?.bucket_xy_m))).size === total
    && positions.every(row => typeof row?.id === 'string' && row.id && vector(row.bucket_xy_m, 2)
      && methods.every(method => knownOutcomes.has(row.outcomes?.[method.id])));
  const observedPair = { both: 0, gained: 0, lost: 0, neither: 0 };
  if (positionRowsValid) for (const row of positions) {
    const before = captures.has(row.outcomes.baseline), after = captures.has(row.outcomes.optimized);
    observedPair[before && after ? 'both' : before ? 'lost' : after ? 'gained' : 'neither']++;
  }
  const positionsValid = positionRowsValid && Object.keys(observedPair).every(key => observedPair[key] === paired[key])
    && methods.every(method => positions.filter(row => captures.has(row.outcomes[method.id])).length === method.captures
      && positions.filter(row => row.outcomes[method.id] === 'setup_rejected').length === method.no_candidate
      && positions.filter(row => row.outcomes[method.id] === 'flight_miss').length === method.flight_misses);
  const accountingValid = total > 0 && methods.every(method => method.valid) && pairValid
    && prepared === maximum && planned === maximum && plannedReleases === releases
    && releases === methods.reduce((sum, method) => sum + method.admitted, 0) && positionsValid;
  return { total, budget, methods, paired, prepared, planned, releases, planned_releases: plannedReleases,
    positions: positionsValid ? positions : [], status: section.status, phase: section.phase, audit: section.audit_status,
    accounting_valid: accountingValid, complete: section.status === 'completed' && section.audit_status === 'passed' && budget > 0 && accountingValid,
    next: typeof section.next === 'string' ? section.next : '', error: typeof section.error === 'string' ? section.error : '' };
}
/** Compare direct-path-first planning against the same optimized arm checks; only complete, audited accounting exposes scores. */
export function directPathFirstSummary(study) {
  const section = study?.direct_path_first_study;
  if (section?.version !== 'sircus-direct-path-first-study-v1') return null;
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const seconds = value => Number.isFinite(value) && value >= 0 ? value : null;
  const total = count(section.budgets?.positions), budget = seconds(section.budgets?.planning_s);
  const comparison = section.comparison || {}, maximum = total === null ? null : total * 2;
  const methods = Object.entries({ baseline: 'Reference path first', direct_first: 'Candidate path first' }).map(([id, label]) => {
    const raw = comparison.methods?.[id] || {}, stats = comparison.latency?.[id] || {};
    const values = Object.fromEntries(['positions', 'admitted', 'captures', 'flight_misses', 'no_candidate'].map(key => [key, count(raw[key])]));
    const timing = Object.fromEntries(['mean_s', 'median_s', 'p90_s', 'max_s'].map(key => [key, seconds(stats[key])]));
    const validTiming = Object.values(timing).every(value => value !== null) && timing.mean_s <= timing.max_s
      && timing.median_s <= timing.p90_s && timing.p90_s <= timing.max_s;
    return { id, label, ...values, timing: validTiming ? timing : Object.fromEntries(Object.keys(timing).map(key => [key, null])),
      valid: total > 0 && Object.values(values).every(value => value !== null) && values.positions === total
        && values.admitted + values.no_candidate === total && values.captures <= values.admitted
        && values.flight_misses <= values.admitted - values.captures };
  });
  const paired = Object.fromEntries(['both', 'gained', 'lost', 'neither'].map(key => [key, count(comparison.paired?.[key])]));
  const pairValid = Object.values(paired).every(value => value !== null)
    && paired.both + paired.gained + paired.lost + paired.neither === total
    && paired.both + paired.lost === methods[0].captures && paired.both + paired.gained === methods[1].captures;
  const progress = section.progress || {};
  const bounded = value => { const n = count(value); return n !== null && (maximum === null || n <= maximum) ? n : null; };
  const prepared = bounded(progress.prepared), planned = bounded(progress.planned), releases = bounded(progress.completed_releases), plannedReleases = bounded(progress.planned_releases);
  const positions = Array.isArray(comparison.positions) ? comparison.positions : [];
  const captures = new Set(['direct', 'table', 'unclassified_capture']);
  const knownOutcomes = new Set([...captures, 'setup_rejected', 'flight_miss', 'invalid_physics', 'arm_assisted_capture']);
  const positionRowsValid = positions.length === total && new Set(positions.map(row => row?.id)).size === total
    && new Set(positions.map(row => JSON.stringify(row?.bucket_xy_m))).size === total
    && positions.every(row => typeof row?.id === 'string' && row.id && vector(row.bucket_xy_m, 2)
      && methods.every(method => knownOutcomes.has(row.outcomes?.[method.id])));
  const observedPair = { both: 0, gained: 0, lost: 0, neither: 0 };
  if (positionRowsValid) for (const row of positions) {
    const before = captures.has(row.outcomes.baseline), after = captures.has(row.outcomes.direct_first);
    observedPair[before && after ? 'both' : before ? 'lost' : after ? 'gained' : 'neither']++;
  }
  const positionsValid = positionRowsValid && Object.keys(observedPair).every(key => observedPair[key] === paired[key])
    && methods.every(method => positions.filter(row => captures.has(row.outcomes[method.id])).length === method.captures
      && positions.filter(row => row.outcomes[method.id] === 'setup_rejected').length === method.no_candidate
      && positions.filter(row => row.outcomes[method.id] === 'flight_miss').length === method.flight_misses);
  const accountingValid = total > 0 && methods.every(method => method.valid) && pairValid
    && prepared === maximum && planned === maximum && plannedReleases === releases
    && releases === methods.reduce((sum, method) => sum + method.admitted, 0) && positionsValid;
  const diagnostic = section.diagnostic;
  const diagnosticMethods = methods.map(method => {
    const row = diagnostic?.methods?.[method.id];
    const outcome = row?.outcome === null || knownOutcomes.has(row?.outcome) ? row.outcome : undefined;
    return { id: method.id, label: method.label, checked_shot: row?.checked_shot,
      planning_s: seconds(row?.planning_s), outcome };
  });
  const diagnosticValid = vector(diagnostic?.bucket_xy_m, 2) && diagnosticMethods.every(row =>
    typeof row.checked_shot === 'boolean' && row.planning_s !== null && row.outcome !== undefined
    && (row.checked_shot ? row.outcome !== 'setup_rejected' : row.outcome === null || row.outcome === 'setup_rejected'));
  return { diagnostic: diagnosticValid ? { bucket_xy_m: diagnostic.bucket_xy_m, methods: diagnosticMethods } : null, total, budget, methods, paired, prepared, planned, releases, planned_releases: plannedReleases,
    positions: positionsValid ? positions : [], status: section.status, phase: section.phase, audit: section.audit_status,
    accounting_valid: accountingValid, complete: section.status === 'completed' && section.audit_status === 'passed' && budget > 0 && accountingValid,
    next: typeof section.next === 'string' ? section.next : '', error: typeof section.error === 'string' ? section.error : '' };
}
/** Capability-gated choice: older servers keep their exact original request shape. */
export function physicsPlanners(capabilities) {
  const labels = { fast: 'Faster planner · ~30 s decision budget', classic: 'Original planner · explicit fallback' };
  const ids = Array.isArray(capabilities?.planners)
    ? [...new Set(capabilities.planners.map(row => row?.id).filter(id => Object.hasOwn(labels, id)))] : [];
  const choices = (ids.length ? ids : ['classic']).map(id => ({ id, label: labels[id] }));
  const defaultPlanner = ids.includes(capabilities?.default_planner) ? capabilities.default_planner
    : choices.some(row => row.id === 'classic') ? 'classic' : choices[0].id;
  return { choices, defaultPlanner, advertised: ids.length > 0 };
}
export class InteractivePhysics {
  constructor({
    fetcher = (...args) => globalThis.fetch(...args),
    onChange = () => {},
    schedule = (callback, delay) => globalThis.setTimeout(callback, delay),
    cancel = timer => globalThis.clearTimeout(timer),
  } = {}) {
    this.fetcher = fetcher; this.onChange = onChange; this.schedule = schedule; this.cancel = cancel;
    this.epoch = 0; this.timer = null; this.capabilities = null; this.planner = 'classic';
    this.snapshot = { state: 'unavailable', message: 'Physics planner unavailable.' };
  }
  get plannerOptions() { return physicsPlanners(this.capabilities); }
  selectPlanner(id) {
    if (!this.available || this.busy || !this.plannerOptions.choices.some(row => row.id === id)) return false;
    if (id !== this.planner) { this.planner = id; this.invalidate('Planner changed. Plan a new shot.'); }
    return true;
  }
  get available() { return this.capabilities?.enabled === true; }
  get busy() { return ['submitting', 'submitting_drop', 'queued', 'planning', 'running'].includes(this.snapshot.state); }
  get canDrop() { return this.available && this.snapshot.state === 'ready' && hasNativePreview(this.snapshot.preview); }
  publish(value) { this.snapshot = value; this.onChange(value); }
  async initialize() {
    try {
      const response = await this.fetcher('/api/interactive-physics', { cache: 'no-store' });
      if (!response.ok) return false;
      const data = await response.json();
      if (data.enabled !== true) return false;
      this.capabilities = data;
      this.planner = this.plannerOptions.defaultPlanner;
      this.publish({ state: 'idle', message: 'Place the bucket, then plan a shot.' });
      return true;
    } catch { return false; } // An older viewer remains usable without this optional API.
  }
  invalidate(message = 'Scene changed. Plan a new shot.') {
    ++this.epoch;
    if (this.timer !== null) this.cancel(this.timer);
    this.timer = null;
    this.publish({ state: this.available ? 'stale' : 'unavailable', message });
  }
  async json(url, options = {}) {
    const response = await this.fetcher(url, { cache: 'no-store', ...options });
    const data = await response.json();
    if (!response.ok) throw Error(data.error || data.message || `HTTP ${response.status}`);
    return data;
  }
  async plan(xy) {
    if (!this.available || this.busy) return false;
    if (!vector(xy, 2)) { this.publish({ state: 'failed', message: 'Enter a valid bucket X and Y.' }); return false; }
    const bounds = this.capabilities.bucket_bounds;
    if (bounds && ['x', 'y'].some((axis, i) => vector(bounds[axis], 2) && (xy[i] < bounds[axis][0] || xy[i] > bounds[axis][1]))) {
      this.publish({ state: 'failed', message: 'Bucket is outside the physics planner bounds.' }); return false;
    }
    const epoch = ++this.epoch, target = xy.slice(), planner = this.planner;
    const payload = { bucket_xy_m: target, ...(this.plannerOptions.advertised ? { planner } : {}) };
    if (this.timer !== null) this.cancel(this.timer);
    this.timer = null;
    this.publish({ state: 'submitting', bucket_xy_m: target, planner, message: 'Submitting shot plan…' });
    try {
      const job = await this.json('/api/interactive-physics/plan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (epoch !== this.epoch) return false;
      if (typeof job.id !== 'string' || !job.id) throw Error('Planner did not return a job ID.');
      if (job.planner && job.planner !== planner) throw Error('Planner response belongs to another selected planner.');
      this.publish({ ...job, state: 'queued', bucket_xy_m: target, planner, message: 'Planning geometry and both arm poses…' });
      void this.poll(epoch, job.id, target);
      return true;
    } catch (error) { if (epoch === this.epoch) this.publish({ state: 'failed', message: error.message }); return false; }
  }
  async drop() {
    if (!this.canDrop) return false;
    const before = this.snapshot, epoch = this.epoch;
    // Synchronous state change prevents a second click from submitting another drop.
    this.publish({ ...before, state: 'submitting_drop', message: 'Starting one native simulated drop…' });
    try {
      const job = await this.json(`/api/interactive-physics/${encodeURIComponent(before.id)}/drop`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      if (epoch !== this.epoch) return false;
      if (job.id !== before.id) throw Error('Drop response belongs to another plan.');
      this.publish({ ...before, ...job, state: 'running', message: 'Running one native simulated drop…' });
      void this.poll(epoch, before.id, before.bucket_xy_m);
      return true;
    } catch (error) { if (epoch === this.epoch) this.publish({ ...before, state: 'failed', message: error.message }); return false; }
  }
  async poll(epoch, id, xy) {
    try {
      const job = await this.json(`/api/interactive-physics/${encodeURIComponent(id)}`);
      if (epoch !== this.epoch) return;
      if (job.id !== id || !sameXY(job.bucket_xy_m, xy)) throw Error('Planner response does not match this target.');
      if (!states.has(job.state)) throw Error('Unknown planner state.');
      if (job.planner && job.planner !== this.planner) throw Error('Planner response belongs to another selected planner.');
      if (job.preview?.planner && job.preview.planner !== this.planner) throw Error('Preview belongs to another selected planner.');
      if (job.state === 'ready' && (!hasNativePreview(job.preview) || !sameXY(job.preview.bucket_xy_m, xy))) throw Error('Plan has no complete native arm-pose preview.');
      if (job.state === 'completed' && (!job.result || !sameXY(job.result.bucket_xy_m, xy))) throw Error('Drop result does not match this target.');
      this.publish(job);
      if (!terminal.has(job.state)) this.timer = this.schedule(() => { this.timer = null; void this.poll(epoch, id, xy); }, 700);
    } catch (error) { if (epoch === this.epoch) this.publish({ state: 'failed', id, bucket_xy_m: xy, message: error.message }); }
  }
}
