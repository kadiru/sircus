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
export function stalledReferenceSummary(study) {
  const section = study?.stalled_reference_study;
  if (section?.version !== 'sircus-stalled-reference-study-v1') return null;
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const seconds = value => Number.isFinite(value) && value >= 0 ? value : null;
  const total = count(section.budgets?.positions), budget = seconds(section.budgets?.planning_s);
  const comparison = section.comparison || {}, maximum = total === null ? null : total * 2;
  const methods = Object.entries({ baseline: 'Current arm checks', hybrid: 'Fallback after stalled reference' }).map(([id, label]) => {
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
    const before = captures.has(row.outcomes.baseline), after = captures.has(row.outcomes.hybrid);
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
  const finalEvidence = section.status === 'completed' && section.audit_status === 'passed' && budget > 0 && accountingValid;
  const promotionEligible = finalEvidence && diagnosticValid && captures.has(diagnosticMethods[1].outcome)
    && methods[1].captures >= methods[0].captures && methods[0].timing.mean_s !== null
    && methods[1].timing.mean_s !== null && methods[1].timing.mean_s <= methods[0].timing.mean_s * 1.10;
  return { promotion_eligible: promotionEligible, reference_path_s: seconds(section.budgets?.reference_path_s), diagnostic: diagnosticValid ? { bucket_xy_m: diagnostic.bucket_xy_m, methods: diagnosticMethods } : null, total, budget, methods, paired, prepared, planned, releases, planned_releases: plannedReleases,
    positions: positionsValid ? positions : [], status: section.status, phase: section.phase, audit: section.audit_status,
    accounting_valid: accountingValid, complete: section.status === 'completed' && section.audit_status === 'passed' && budget > 0 && accountingValid,
    next: typeof section.next === 'string' ? section.next : '', error: typeof section.error === 'string' ? section.error : '' };
}
/** Capability-gated choice: older servers keep their exact original request shape. */
export function boardFallbackSummary(study) {
  const raw = study?.board_fallback_study;
  if (raw?.version !== 'sircus-board-fallback-study-v1') return null;
  const c = raw.comparison || {};
  const translated = { ...raw, version: 'sircus-stalled-reference-study-v1', comparison: { ...c,
    methods: { baseline: c.methods?.baseline, hybrid: c.methods?.fallback },
    latency: { baseline: c.latency?.baseline, hybrid: c.latency?.fallback },
    positions: Array.isArray(c.positions) ? c.positions.map(row => ({ ...row,
      outcomes: { baseline: row.outcomes?.baseline, hybrid: row.outcomes?.fallback } })) : [],
  } };
  const view = stalledReferenceSummary({ stalled_reference_study: translated });
  view.methods[0].label = 'Current planner'; view.methods[1].label = 'Board-pose fallback';
  view.complete = view.complete && raw.physics_substeps === 64 && view.total === 50
    && ['controls_pre', 'controls_post'].every(key => raw.controls?.[key]?.status === 'passed');
  view.complete = Boolean(view.complete);
  view.promotion_eligible = false;
  view.controls_passed = ['controls_pre', 'controls_post'].every(key => raw.controls?.[key]?.status === 'passed');
  const n = value => Number.isInteger(value) && value >= 0 ? value : null;
  view.fallback = Object.fromEntries(['activated', 'changed_choice'].map(key => [key,
    n(raw.fallback?.[key]) !== null && raw.fallback[key] <= view.total ? raw.fallback[key] : null]));
  const d = raw.heading_diagnostic;
  const result = value => value && ['released', 'captures', 'arm_assisted'].every(key => n(value[key]) !== null)
    && value.captures + value.arm_assisted <= value.released;
  view.diagnostic = d?.status === 'completed' && d.audit_status === 'passed' && d.physics_substeps === 64
    && vector(d.bucket_xy_m, 2) && ['original', 'search', 'repeat'].every(key => result(d[key]))
    && n(d.confirmed_settings) !== null && d.confirmed_settings <= d.search.captures
    && d.confirmed_settings * 3 <= d.repeat.captures
    && d.search.captures * 3 === d.repeat.released ? d : null;
  return view;
}

/** Separate guarded experiment; historical fallback evidence and live choices stay unchanged. */
export function guardedBoardFallbackSummary(study) {
  const raw = study?.guarded_board_fallback_study;
  if (raw?.version !== 'sircus-guarded-board-fallback-study-v1') return null;
  const comparison = raw.comparison || {};
  const translated = { ...raw, version: 'sircus-stalled-reference-study-v1', diagnostic: undefined,
    comparison: { ...comparison,
      methods: { baseline: comparison.methods?.baseline, hybrid: comparison.methods?.guarded },
      latency: { baseline: comparison.latency?.baseline, hybrid: comparison.latency?.guarded },
      positions: Array.isArray(comparison.positions) ? comparison.positions.map(row => ({ ...row,
        outcomes: { baseline: row?.outcomes?.baseline, hybrid: row?.outcomes?.guarded } })) : [],
    } };
  const view = stalledReferenceSummary({ stalled_reference_study: translated });
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const methods = [{ id: 'baseline', label: 'Current planner' }, { id: 'guarded', label: 'Guarded board fallback' }];
  view.methods = view.methods.map((method, index) => ({ ...method, ...methods[index] }));
  view.controls_passed = ['controls_pre', 'controls_post'].every(key => {
    const gate = raw.controls?.[key];
    return gate?.status === 'passed' && gate.direct_captures === 3
      && gate.table_assisted_captures === 3 && gate.negative_captures === 0;
  });
  view.fallback = Object.fromEntries(['activated', 'changed_choice'].map(key => [key, count(raw.fallback?.[key])]));
  const fallbackValid = view.fallback.activated !== null && view.fallback.changed_choice !== null
    && view.fallback.changed_choice <= view.fallback.activated && view.fallback.activated <= 20;
  const captures = new Set(['direct', 'table', 'unclassified_capture']);
  const outcomes = new Set([...captures, 'flight_miss', 'invalid_physics', 'arm_assisted_capture']);
  const diagnostic = raw.diagnostic;
  const diagnosticMethods = methods.map(method => {
    const row = diagnostic?.methods?.[method.id] || {};
    const labels = Array.isArray(row.outcomes) ? row.outcomes : [];
    const requested = count(row.requested), released = count(row.released), captured = count(row.captures);
    return { ...method, requested, released, captures: captured,
      direct: labels.filter(label => label === 'direct').length,
      table: labels.filter(label => label === 'table').length,
      arm_assisted: labels.filter(label => label === 'arm_assisted_capture').length,
      valid: requested === 3 && [0, 3].includes(released) && captured !== null && captured <= released
        && Array.isArray(row.outcomes) && labels.length === released && labels.every(label => outcomes.has(label))
        && labels.filter(label => captures.has(label)).length === captured
        && (row.checked_shot === undefined || row.checked_shot === (released === 3)) };
  });
  const diagnosticValid = vector(diagnostic?.bucket_xy_m, 2) && diagnosticMethods.every(method => method.valid)
    && !view.positions.some(row => row.bucket_xy_m.every((value, index) => value === diagnostic.bucket_xy_m[index]));
  const timingValid = view.methods.every(method => Object.values(method.timing).every(value => value !== null));
  view.complete = Boolean(view.complete && raw.phase === 'completed' && raw.physics_substeps === 64
    && view.total === 20 && view.budget === 30 && view.controls_passed && fallbackValid && diagnosticValid && timingValid);
  view.promotion_eligible = false;
  view.positions = view.complete ? comparison.positions : [];
  view.diagnostic = view.complete ? { bucket_xy_m: diagnostic.bucket_xy_m, methods: diagnosticMethods } : null;
  return view;
}

/** Selected development targets and repeated frozen commands, never a fresh-position policy score. */
export function clearShotRangeSummary(study) {
  const raw=study?.clear_shot_range;
  if(raw?.version!=='sircus-clear-shot-range-v1')return null;
  const count=value=>Number.isInteger(value)&&value>=0?value:null;
  const text=value=>typeof value==='string'?value:'';
  const hash=value=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
  const same=(a,b)=>Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<1e-8;
  const source=raw.source_summary||{},case_=raw.cases?.[0]||{};
  const sourceValid=source.id==='parallel-board-miss-09-clearance-0016-b0'&&vector(source.values,5)
    &&vector(source.bucket_xy_m,2)&&sameXY(source.bucket_xy_m,case_.bucket_xy_m)
    &&Number.isFinite(source.board_heading_offset_deg)&&same(source.board_twist_deg,135)
    &&vector(source.selected_q10,6)&&vector(source.selected_q14,6)&&source.prior_no_robot_contact===true;
  const allowed=new Set(['direct','table','flight_miss','invalid_physics','arm_assisted_capture']);
  const result=group=>{
    const row=Object.fromEntries(['requested','completed','released','captures','direct','table','robot_contacts'].map(key=>[key,count(group?.[key])]));
    const outcomes=Array.isArray(group?.outcomes)?group.outcomes:[];
    row.arm_assisted=outcomes.filter(value=>value==='arm_assisted_capture').length;
    row.valid=Object.values(row).every(value=>value!==null)&&row.requested===row.completed&&row.released===row.completed
      &&row.captures===row.direct+row.table&&row.captures<=row.released&&row.robot_contacts<=row.released&&row.arm_assisted<=row.robot_contacts
      &&outcomes.length===row.completed&&outcomes.every(value=>allowed.has(value))
      &&row.direct===outcomes.filter(value=>value==='direct').length&&row.table===outcomes.filter(value=>value==='table').length;
    return row;
  };
  const results=Object.fromEntries(['original','search','repeat'].map(key=>[key,result(case_.results?.[key])]));
  const grid=Array.isArray(case_.grid)?case_.grid:[];
  const gaps=[0,.02,.04,.06,.08,.10,.13,.16,.20,.24],tilts=[0,-3,3,-6,6];
  const coordinateKeys=new Set();
  const gridValid=grid.length===49&&grid.every(row=>{
    const key=JSON.stringify([row.gap_delta_m,row.tilt_delta_deg]);
    const unique=!coordinateKeys.has(key);coordinateKeys.add(key);
    const admitted=row.setup_status==='admitted',unreleased=['rejected','not_checked'].includes(row.setup_status);
    return unique&&Boolean(text(row.proposal_id))&&gaps.some(value=>same(value,row.gap_delta_m))&&tilts.includes(row.tilt_delta_deg)
      &&!(same(row.gap_delta_m,0)&&row.tilt_delta_deg===0)&&same(row.tilt_deg,source.values?.[3]+row.tilt_delta_deg)
      &&same(row.drop_gap_m,source.values?.[4]+row.gap_delta_m)
      &&(admitted ? Boolean(text(row.candidate_id))&&allowed.has(row.initial_outcome)&&typeof row.robot_contact==='boolean'
        &&row.initial_capture===['direct','table'].includes(row.initial_outcome)
        &&count(row.repeat_completed)!==null&&count(row.repeat_captures)!==null&&row.repeat_captures<=row.repeat_completed
        &&row.repeat_completed===(row.initial_capture?3:0)&&row.confirmed===(row.repeat_completed===3&&row.repeat_captures===3)
        :unreleased&&row.candidate_id===null&&row.initial_outcome===null&&row.initial_capture===null&&row.robot_contact===null
          &&row.repeat_completed===0&&row.repeat_captures===0&&row.confirmed===false);
  });
  const repeated=Array.isArray(case_.repeated_settings)?case_.repeated_settings:[];
  const confirmed=repeated.filter(row=>row.repeat_completed===3&&row.repeat_captures===3);
  const repeatValid=repeated.length===results.search.captures&&repeated.every(row=>{
    const request=row.request||{},match=grid.find(item=>item.candidate_id===row.candidate_id);
    return Boolean(match?.initial_capture)&&request.id===row.candidate_id&&request.context_id===case_.id
      &&sameXY(request.bucket_xy_m,case_.bucket_xy_m)&&vector(request.values,5)
      &&same(request.values[0],source.values?.[0])&&same(request.values[1],source.values?.[1])&&same(request.values[2],source.values?.[2])
      &&same(request.values[3],match.tilt_deg)&&same(request.values[4],match.drop_gap_m)
      &&same(request.board_heading_offset_deg,source.board_heading_offset_deg)&&same(request.board_twist_deg,source.board_twist_deg)
      &&vector(request.selected_q10,6)&&vector(request.selected_q14,6)&&row.repeat_completed===match.repeat_completed&&row.repeat_captures===match.repeat_captures;
  })&&new Set(repeated.map(row=>row.candidate_id)).size===repeated.length
    &&JSON.stringify(case_.confirmed_settings)===JSON.stringify(confirmed);
  const admitted=grid.filter(row=>row.setup_status==='admitted');
  const accounting=Array.isArray(raw.cases)&&raw.cases.length===1&&case_.id==='parallel-board-miss-09'&&sourceValid&&gridValid&&repeatValid
    &&Object.values(results).every(row=>row.valid)&&results.original.requested===3&&results.search.requested===admitted.length
    &&case_.selected===admitted.length&&case_.proposed===49&&new Set(admitted.map(row=>row.candidate_id)).size===admitted.length
    &&results.search.captures===admitted.filter(row=>row.initial_capture).length
    &&results.search.robot_contacts===admitted.filter(row=>row.robot_contact).length
    &&results.search.direct===admitted.filter(row=>row.initial_outcome==='direct').length&&results.search.table===admitted.filter(row=>row.initial_outcome==='table').length
    &&results.repeat.released===3*results.search.captures&&results.repeat.captures===grid.reduce((sum,row)=>sum+row.repeat_captures,0)
    &&raw.initial_requests===3+admitted.length&&raw.confirmation_requests===results.repeat.released
    &&raw.progress?.completed===raw.progress?.planned&&raw.progress.completed===raw.initial_requests+raw.confirmation_requests
    &&raw.native_releases===raw.progress.completed+27&&raw.native_releases<=226;
  const controls=['controls_pre','controls_post'].every(key=>raw.controls?.[key]?.status==='passed'&&raw.controls[key].direct_captures===3
    &&raw.controls[key].table_assisted_captures===3&&raw.controls[key].negative_captures===0);
  const preserved=raw.preserved_shots||{},provenance=raw.published_provenance||{};
  const proof=['report_sha256','audit_sha256','evidence_manifest_sha256','plan_sha256','contract_sha256','source_bank_sha256'].every(key=>hash(provenance[key]))
    &&provenance.evidence_manifest_sha256===raw.evidence_manifest_sha256;
  const complete=Boolean(raw.status==='completed'&&raw.phase==='completed'&&raw.audit_status==='passed'&&raw.scope==='development'
    &&raw.physics_substeps===64&&controls&&preserved.requested===9&&preserved.completed===9&&preserved.captures===9&&preserved.passed===true&&proof&&accounting);
  return {complete,accounting_valid:accounting,promotion_eligible:false,status:raw.status,phase:raw.phase,audit:raw.audit_status,
    completed:count(raw.progress?.completed),planned:count(raw.progress?.planned),source:complete?source:null,results:complete?results:null,
    grid:complete?grid:[],confirmed:complete?confirmed:[],native_releases:complete?raw.native_releases:null,next:text(raw.next),error:text(raw.error)};
}

export function remainingLinkClearanceSummary(study) {
  const raw = study?.remaining_link_clearance;
  if (raw?.version !== 'sircus-link-clearance-study-v1') return null;
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const text = value => typeof value === 'string' ? value : '';
  const hash = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
  const expected = ['parallel-board-miss-09', 'parallel-board-miss-19'];
  const allowed = new Set(['direct', 'table', 'flight_miss', 'invalid_physics', 'arm_assisted_capture']);
  const result = source => {
    const row = Object.fromEntries(['requested', 'completed', 'released', 'captures', 'direct', 'table', 'robot_contacts'].map(key => [key, count(source?.[key])]));
    const outcomes = Array.isArray(source?.outcomes) ? source.outcomes : [];
    row.arm_assisted = outcomes.filter(value => value === 'arm_assisted_capture').length;
    row.valid = Object.values(row).every(value => value !== null) && row.completed === row.requested
      && row.released === row.completed && row.robot_contacts <= row.released && row.arm_assisted <= row.robot_contacts
      && row.captures === row.direct + row.table && row.captures <= row.released
      && Array.isArray(source?.outcomes) && outcomes.length === row.completed && outcomes.every(value => allowed.has(value))
      && row.direct === outcomes.filter(value => value === 'direct').length && row.table === outcomes.filter(value => value === 'table').length;
    return row;
  };
  const cases = (Array.isArray(raw.cases) ? raw.cases : []).map(source => {
    const preparation = Object.fromEntries(['proposed', 'checked', 'admitted', 'selected'].map(key => [key, count(source?.[key])]));
    const results = Object.fromEntries(['original', 'search', 'repeat'].map(key => [key, result(source?.results?.[key])]));
    const supplied = Array.isArray(source?.repeated_settings) ? source.repeated_settings : [];
    const repeated = supplied.map(row => {
      const request = row?.request || {}, captures = count(row?.repeat_captures), completed = count(row?.repeat_completed);
      return {candidate_id:text(row?.candidate_id), values:request.values, heading:request.board_heading_offset_deg,
        twist:request.board_twist_deg, q10:request.selected_q10, q14:request.selected_q14, captures, completed,
        valid:Boolean(text(row?.candidate_id)) && row.candidate_id === request.id && request.context_id === source.id
          && captures !== null && captures <= 3 && completed === 3 && vector(request.values, 5) && request.values[4] > 0
          && sameXY(request.bucket_xy_m, source.bucket_xy_m) && vector(request.selected_q10, 6) && vector(request.selected_q14, 6)
          && Number.isFinite(request.board_heading_offset_deg) && Number.isFinite(request.board_twist_deg)};
    });
    const confirmed = repeated.filter(row => row.captures === 3);
    const confirmationsValid = Array.isArray(source?.repeated_settings) && repeated.length === results.search.captures
      && repeated.every(row => row.valid) && new Set(repeated.map(row => row.candidate_id)).size === repeated.length
      && results.repeat.requested === 3 * repeated.length && results.repeat.captures === repeated.reduce((sum,row) => sum + row.captures, 0)
      && Array.isArray(source?.confirmed_settings) && JSON.stringify(source.confirmed_settings) === JSON.stringify(supplied.filter(row => row.repeat_captures === 3 && row.repeat_completed === 3));
    return {id:source?.id,bucket_xy_m:source?.bucket_xy_m,preparation,results,confirmed,
      valid:expected.includes(source?.id) && vector(source.bucket_xy_m,2) && Object.values(preparation).every(value => value !== null)
        && preparation.selected <= preparation.admitted && preparation.selected <= preparation.proposed && preparation.selected <= 32
        && Object.values(results).every(row => row.valid) && results.original.requested === 3
        && results.search.requested === preparation.selected && confirmationsValid};
  });
  const diagnosis = (Array.isArray(raw.diagnosis_rows) ? raw.diagnosis_rows : []).map(row => {
    const counts = row?.first_contact_link_counts || {}, times = row?.first_contact_time_s || {};
    return {id:row?.id,bucket_xy_m:row?.bucket_xy_m,trials:count(row?.trials),robot_contacts:count(row?.robot_contacts),
      links:Object.entries(counts).map(([name,total]) => ({name,total})),time_min:times.min,time_max:times.max,
      valid:expected.includes(row?.id) && vector(row?.bucket_xy_m,2) && row.trials === 20 && row.robot_contacts === 20
        && row.method === 'native_force_cpu_reconstruction' && typeof counts === 'object' && !Array.isArray(counts)
        && Object.entries(counts).every(([name,total]) => Boolean(text(name)) && count(total) !== null)
        && Object.values(counts).reduce((sum,total) => sum + total,0) === row.robot_contacts
        && Number.isFinite(times.min) && times.min >= 0 && Number.isFinite(times.max) && times.max >= times.min};
  });
  const progress = raw.progress || {}, completed=count(progress.completed), planned=count(progress.planned);
  const initial=count(raw.initial_requests), confirmations=count(raw.confirmation_requests), releases=count(raw.native_releases);
  const preserved=raw.preserved_shots || {};
  const preservationValid=preserved.requested === 9 && preserved.completed === 9 && preserved.captures === 9 && preserved.passed === true;
  const controlsPassed=['controls_pre','controls_post'].every(key => raw.controls?.[key]?.status === 'passed'
    && raw.controls[key].direct_captures === 3 && raw.controls[key].table_assisted_captures === 3 && raw.controls[key].negative_captures === 0);
  const provenance=raw.published_provenance || {};
  const provenanceValid=['report_sha256','audit_sha256','evidence_manifest_sha256','plan_sha256','source_bank_sha256','contract_sha256','diagnosis_sha256'].every(key => hash(provenance[key]))
    && provenance.evidence_manifest_sha256 === raw.evidence_manifest_sha256;
  const accounting=cases.length === 2 && cases.every(row => row.valid) && expected.every((id,i) => cases[i].id === id)
    && new Set(cases.map(row => JSON.stringify(row.bucket_xy_m))).size === 2
    && initial === cases.reduce((sum,row) => sum + row.results.original.released + row.results.search.released,0)
    && confirmations === cases.reduce((sum,row) => sum + row.results.repeat.released,0)
    && completed === planned && completed === initial + confirmations && releases === completed + 27 && releases <= 289
    && diagnosis.length === 2 && diagnosis.every((row,i) => row.valid && row.id === cases[i].id && sameXY(row.bucket_xy_m,cases[i].bucket_xy_m));
  const complete=Boolean(raw.status === 'completed' && raw.phase === 'completed' && raw.audit_status === 'passed'
    && raw.scope === 'development' && raw.physics_substeps === 64 && controlsPassed && preservationValid && provenanceValid && accounting);
  return {complete,accounting_valid:accounting,promotion_eligible:false,status:raw.status,phase:raw.phase,audit:raw.audit_status,
    completed,planned,native_releases:complete?releases:null,cases:complete?cases:[],diagnosis:complete?diagnosis:[],
    next:text(raw.next),error:text(raw.error)};
}

export function nearbyShotTransferSummary(study) {
  const raw = study?.nearby_shot_transfer;
  if (raw?.version !== 'sircus-nearby-transfer-study-v1') return null;
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const number = value => Number.isFinite(value) && value >= 0 ? value : null;
  const text = value => typeof value === 'string' ? value : '';
  const hash = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
  const same = (a, b) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= 1e-8 * Math.max(1, Math.abs(a), Math.abs(b));
  const expected = ['03', '09', '13', '15', '19'].map(id => `parallel-board-miss-${id}`);
  const allowed = new Set(['direct', 'table', 'flight_miss', 'invalid_physics', 'arm_assisted_capture']);
  const result = source => {
    const row = Object.fromEntries(['requested', 'completed', 'released', 'captures', 'direct', 'table', 'robot_contacts'].map(key => [key, count(source?.[key])]));
    const outcomes = Array.isArray(source?.outcomes) ? source.outcomes : [];
    row.arm_assisted = outcomes.filter(value => value === 'arm_assisted_capture').length;
    row.valid = Object.values(row).every(value => value !== null) && row.completed === row.requested
      && row.released === row.completed && row.robot_contacts <= row.released
      && row.captures === row.direct + row.table && row.captures <= row.released
      && Array.isArray(source?.outcomes) && outcomes.length === row.completed && outcomes.every(value => allowed.has(value))
      && row.direct === outcomes.filter(value => value === 'direct').length && row.table === outcomes.filter(value => value === 'table').length;
    return row;
  };
  const bank = raw.source_bank_summary || {};
  const studyNames = { recovery: 'Board-position recovery', parallel: 'Parallel-search control', guarded: 'Guarded-search comparison' };
  const sources = (Array.isArray(bank.sources) ? bank.sources : []).map((source, index) => ({
    id: text(source?.id), label: `S${String(index + 1).padStart(2, '0')}`, bucket_xy_m: source?.bucket_xy_m,
    study: studyNames[source?.study_key] || '', study_key: source?.study_key,
    strength: source?.strength === 'confirmed_3_of_3' ? '3 / 3 prior repeats' : 'One prior capture · not repeated',
    report_sha256: source?.report_sha256, audit_sha256: source?.audit_sha256,
    valid: Boolean(text(source?.id)) && vector(source?.bucket_xy_m, 2) && Object.hasOwn(studyNames, source?.study_key)
      && hash(source?.report_sha256) && hash(source?.audit_sha256) && vector(source?.values, 5)
      && Number.isFinite(source?.board_heading_offset_deg) && Number.isFinite(source?.board_twist_deg)
      && (source?.strength === 'confirmed_3_of_3' ? source.repeat_captures === 3 && source.repeat_completed === 3
        : source?.strength === 'single_capture' && source.repeat_captures === 1 && source.repeat_completed === 1),
  }));
  const bankValid = bank.settings === sources.length && bank.confirmed_settings === 8 && bank.single_capture_settings === 15
    && sources.length === 23 && sources.filter(row => row.study_key === 'recovery').length === 7
    && sources.filter(row => row.study_key === 'parallel').length === 1 && sources.filter(row => row.study_key === 'guarded').length === 15
    && bank.sources.filter(row => row.strength === 'confirmed_3_of_3').length === bank.confirmed_settings
    && bank.sources.filter(row => row.strength === 'single_capture').length === bank.single_capture_settings;
  const bySource = new Map(sources.map(row => [row.id, row]));
  const cases = (Array.isArray(raw.cases) ? raw.cases : []).map(source => {
    const preparation = Object.fromEntries(['proposed', 'checked', 'admitted', 'selected'].map(key => [key, count(source?.[key])]));
    const results = Object.fromEntries(['original', 'search', 'repeat'].map(key => [key, result(source?.results?.[key])]));
    const repeatRows = Array.isArray(source?.repeated_settings) ? source.repeated_settings : [];
    const repeated = repeatRows.map(row => {
      const request = row?.request || {}, seed = bySource.get(request.source_bank_id);
      const distance = vector(source?.bucket_xy_m, 2) && seed ? Math.hypot(...source.bucket_xy_m.map((value, i) => value - seed.bucket_xy_m[i])) : null;
      const captures = count(row?.repeat_captures), completed = count(row?.repeat_completed);
      return { candidate_id: text(row?.candidate_id), source_id: text(request.source_bank_id), source_label: seed?.label || '', captures, completed,
        values: request.values, heading: request.board_heading_offset_deg, twist: request.board_twist_deg,
        valid: Boolean(text(row?.candidate_id)) && row.candidate_id === request.id && request.context_id === source.id && Boolean(seed?.valid)
          && same(request.source_distance_m, distance) && captures !== null && captures <= 3 && completed === 3
          && vector(request.values, 5) && request.values[4] > 0 && sameXY(request.bucket_xy_m, source.bucket_xy_m)
          && vector(request.selected_q10, 6) && vector(request.selected_q14, 6)
          && Number.isFinite(request.board_heading_offset_deg) && Number.isFinite(request.board_twist_deg) };
    });
    const confirmed = repeated.filter(row => row.captures === 3);
    const suppliedConfirmed = Array.isArray(source?.confirmed_settings) ? source.confirmed_settings : [];
    const expectedConfirmed = repeatRows.filter(row => row?.repeat_captures === 3 && row?.repeat_completed === 3);
    const confirmationsValid = Array.isArray(source?.repeated_settings) && repeated.length === results.search.captures
      && repeated.every(row => row.valid) && new Set(repeated.map(row => row.candidate_id)).size === repeated.length
      && results.repeat.requested === 3 * repeated.length && results.repeat.captures === repeated.reduce((sum, row) => sum + row.captures, 0)
      && Array.isArray(source?.confirmed_settings) && JSON.stringify(suppliedConfirmed) === JSON.stringify(expectedConfirmed);
    const distances = vector(source?.bucket_xy_m, 2) ? sources.map(seed => ({ seed, distance: Math.hypot(...source.bucket_xy_m.map((value, i) => value - seed.bucket_xy_m[i])) })) : [];
    distances.sort((a, b) => a.distance - b.distance || a.seed.id.localeCompare(b.seed.id));
    const nearest = distances[0];
    const sourceCounts = source?.source_counts || {};
    const selectedSources = Object.keys(sourceCounts).filter(key => count(sourceCounts[key]) > 0);
    const sourceCountsValid = typeof sourceCounts === 'object' && !Array.isArray(sourceCounts)
      && Object.entries(sourceCounts).every(([id, value]) => bySource.has(id) && count(value) !== null)
      && Object.values(sourceCounts).reduce((sum, value) => sum + value, 0) === preparation.selected
      && repeated.every(row => sourceCounts[row.source_id] > 0);
    return { id: source?.id, bucket_xy_m: source?.bucket_xy_m, preparation, results, confirmed,
      nearest_source_id: nearest?.seed.id || '', nearest_source_label: nearest?.seed.label || '', nearest_seed_distance_m: number(source?.nearest_seed_distance_m),
      selected_sources: selectedSources, selected_source_labels: selectedSources.map(id => bySource.get(id)?.label || id),
      valid: expected.includes(source?.id) && vector(source.bucket_xy_m, 2)
        && Object.values(preparation).every(value => value !== null)
        && preparation.selected <= preparation.admitted && preparation.selected <= preparation.proposed && preparation.selected <= 20
        && Object.values(results).every(row => row.valid) && results.original.requested === 3 && results.search.requested === preparation.selected
        && confirmationsValid && sourceCountsValid && Boolean(nearest) && same(source.nearest_seed_distance_m, nearest.distance) };
  });
  const progress = raw.progress || {}, completed = count(progress.completed), planned = count(progress.planned);
  const nativeReleases = count(raw.native_releases), initial = count(raw.initial_requests), confirmations = count(raw.confirmation_requests);
  const controlsPassed = ['controls_pre', 'controls_post'].every(key => raw.controls?.[key]?.status === 'passed'
    && raw.controls[key].direct_captures === 3 && raw.controls[key].table_assisted_captures === 3 && raw.controls[key].negative_captures === 0);
  const provenance = raw.published_provenance || {};
  const provenanceValid = ['report_sha256', 'audit_sha256', 'evidence_manifest_sha256', 'plan_sha256', 'source_bank_sha256', 'contract_sha256'].every(key => hash(provenance[key]))
    && provenance.evidence_manifest_sha256 === raw.evidence_manifest_sha256;
  const accounting = cases.length === 5 && cases.every(row => row.valid) && expected.every((id, index) => cases[index].id === id)
    && new Set(cases.map(row => JSON.stringify(row.bucket_xy_m))).size === 5 && bankValid && sources.every(row => row.valid)
    && new Set(sources.map(row => row.id)).size === sources.length
    && initial === cases.reduce((sum, row) => sum + row.results.original.released + row.results.search.released, 0)
    && confirmations === cases.reduce((sum, row) => sum + row.results.repeat.released, 0)
    && completed === planned && completed === initial + confirmations && nativeReleases === completed + 18 && nativeReleases <= 433;
  const complete = Boolean(raw.status === 'completed' && raw.phase === 'completed' && raw.audit_status === 'passed'
    && raw.scope === 'development' && raw.physics_substeps === 64 && controlsPassed && provenanceValid && accounting);
  return { complete, accounting_valid: accounting, promotion_eligible: false, status: raw.status, phase: raw.phase, audit: raw.audit_status,
    completed, planned, native_releases: complete ? nativeReleases : null, cases: complete ? cases : [], sources: complete ? sources : [],
    next: text(raw.next), error: text(raw.error) };
}

export function parallelBoardSummary(study) {
  const raw = study?.parallel_board_study;
  if (raw?.version !== 'sircus-parallel-board-study-v1') return null;
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const seconds = value => Number.isFinite(value) && value >= 0 ? value : null;
  const near = (a, b) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= 1e-7 * Math.max(1, Math.abs(a), Math.abs(b));
  const methodNames = { baseline: 'Current planner', parallel: 'Parallel board search' };
  const captures = new Set(['direct', 'table', 'unclassified_capture']);
  const outcomes = new Set([...captures, 'flight_miss', 'invalid_physics', 'arm_assisted_capture']);
  const source = raw.comparison || {}, positions = Array.isArray(source.positions) ? source.positions : [];
  const rows = positions.map(row => {
    const methods = Object.entries(methodNames).map(([id, label]) => {
      const value = row?.methods?.[id] || {}, labels = Array.isArray(value.outcomes) ? value.outcomes : [];
      const requested = count(value.requested), released = count(value.released), captured = count(value.captures);
      const wall = seconds(value.request_wall_s), decision = seconds(value.decision_latency_s);
      return { id, label, requested, released, captures: captured, checked_shot: value.checked_shot,
        outcomes: labels, request_wall_s: wall, decision_latency_s: decision,
        direct: labels.filter(v => v === 'direct').length, table: labels.filter(v => v === 'table').length,
        arm_assisted: labels.filter(v => v === 'arm_assisted_capture').length,
        valid: requested === 3 && typeof value.checked_shot === 'boolean'
          && released === (value.checked_shot ? 3 : 0) && Array.isArray(value.outcomes)
          && labels.length === released && labels.every(v => outcomes.has(v))
          && captured !== null && captured === labels.filter(v => captures.has(v)).length
          && wall !== null && decision !== null && decision <= 30 && decision <= wall + 1e-7 };
    });
    const lane = row?.parallel_lane || {};
    const checked = count(lane.checked_candidates), admitted = count(lane.admitted_candidates);
    const laneValid = typeof lane.triggered === 'boolean' && typeof lane.changed_choice === 'boolean'
      && checked !== null && admitted !== null && admitted <= checked
      && (!lane.changed_choice || (lane.triggered && admitted > 0 && methods[1].checked_shot))
      && (lane.triggered || (checked === 0 && admitted === 0 && !lane.changed_choice));
    return { id: row?.id, source_id: row?.source_id, bucket_xy_m: row?.bucket_xy_m, role: row?.role, methods,
      lane: { triggered: lane.triggered, changed_choice: lane.changed_choice, checked_candidates: checked, admitted_candidates: admitted },
      valid: typeof row?.id === 'string' && Boolean(row.id) && typeof row?.source_id === 'string' && Boolean(row.source_id)
        && vector(row.bucket_xy_m, 2) && ['previous_miss', 'known_regression'].includes(row.role)
        && methods.every(method => method.valid) && laneValid };
  });
  const positionsValid = rows.length === 6 && rows.every(row => row.valid)
    && new Set(rows.map(row => row.id)).size === 6 && new Set(rows.map(row => row.source_id)).size === 6
    && new Set(rows.map(row => JSON.stringify(row.bucket_xy_m))).size === 6
    && rows.filter(row => row.role === 'previous_miss').length === 5
    && rows.filter(row => row.role === 'known_regression').length === 1;
  const methods = Object.entries(methodNames).map(([id, label], index) => {
    const value = source.methods?.[id] || {};
    const totals = Object.fromEntries(['positions', 'requested', 'checked_positions', 'releases', 'captures', 'reliable_positions']
      .map(key => [key, count(value[key])]));
    const observed = rows.map(row => row.methods[index]);
    const timing = Object.fromEntries(['mean_s', 'median_s', 'p90_s', 'max_s', 'aggregate_s'].map(key => [key, seconds(source.latency?.[id]?.[key])]));
    const times = observed.map(row => row.request_wall_s).sort((a, b) => a - b);
    const aggregate = times.reduce((a, b) => a + b, 0);
    const timingValid = observed.length === 6 && observed.every(row => row.request_wall_s !== null)
      && source.latency?.[id]?.positions === 6 && Object.values(timing).every(value => value !== null)
      && near(timing.aggregate_s, aggregate) && near(timing.mean_s, aggregate / 6)
      && near(timing.median_s, (times[2] + times[3]) / 2) && near(timing.p90_s, times[5]) && near(timing.max_s, times[5]);
    return { id, label, ...totals, timing, direct: observed.reduce((n, row) => n + row.direct, 0),
      table: observed.reduce((n, row) => n + row.table, 0), arm_assisted: observed.reduce((n, row) => n + row.arm_assisted, 0),
      valid: totals.positions === 6 && totals.requested === 18 && Object.values(totals).every(n => n !== null)
        && totals.checked_positions === observed.filter(row => row.checked_shot).length
        && totals.releases === observed.reduce((n, row) => n + row.released, 0)
        && totals.captures === observed.reduce((n, row) => n + row.captures, 0)
        && totals.reliable_positions === observed.filter(row => row.captures === 3).length && timingValid };
  });
  const progress = raw.progress || {};
  const bounded = (value, limit) => { const n = count(value); return n !== null && n <= limit ? n : null; };
  const prepared = bounded(progress.prepared, 12), planned = bounded(progress.planned, 12);
  const released = bounded(progress.completed_releases, 36), plannedReleases = bounded(progress.planned_releases, 36);
  const lane = { activated: count(raw.parallel?.activated), changed_choice: count(raw.parallel?.changed_choice) };
  const laneValid = lane.activated !== null && lane.changed_choice !== null
    && lane.activated === rows.filter(row => row.lane.triggered).length
    && lane.changed_choice === rows.filter(row => row.lane.changed_choice).length;
  const controlsPassed = ['controls_pre', 'controls_post'].every(key => raw.controls?.[key]?.status === 'passed'
    && raw.controls[key].direct_captures === 3 && raw.controls[key].table_assisted_captures === 3 && raw.controls[key].negative_captures === 0);
  const provenance = raw.published_provenance || {};
  const hashesValid = ['report_sha256', 'audit_sha256', 'evidence_manifest_sha256']
    .every(key => /^[a-f0-9]{64}$/.test(provenance[key] || ''))
    && provenance.evidence_manifest_sha256 === raw.evidence_manifest_sha256;
  const accountingValid = positionsValid && methods.every(method => method.valid) && laneValid
    && prepared === 12 && planned === 12 && released === plannedReleases
    && released === methods.reduce((sum, method) => sum + method.releases, 0);
  const complete = Boolean(raw.status === 'completed' && raw.phase === 'completed' && raw.audit_status === 'passed'
    && raw.scope === 'development' && raw.physics_substeps === 64 && raw.budgets?.positions === 6
    && raw.budgets?.repeats === 3 && raw.budgets?.planning_s === 30 && controlsPassed && hashesValid && accountingValid);
  return { complete, promotion_eligible: false, accounting_valid: accountingValid, status: raw.status, phase: raw.phase,
    prepared, planned, released, planned_releases: plannedReleases, methods: complete ? methods : [],
    positions: complete ? rows : [], parallel: complete ? lane : null,
    next: typeof raw.next === 'string' ? raw.next : '', error: typeof raw.error === 'string' ? raw.error : '' };
}

export function physicsPlanners(capabilities) {
  const labels = { hybrid: 'Checked planner · stalled-search fallback', fast: 'Faster planner · ~30 s decision budget', classic: 'Original planner · explicit fallback' };
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

/** Audited local development comparison; never promotes a planner or starts a drop. */
export function hybridMissCorrectionSummary(study, videoAssets = {}) {
  const section = study?.hybrid_miss_correction;
  if (section?.version !== 'sircus-hybrid-miss-correction-v1') return null;
  const number = value => Number.isFinite(value) ? value : null;
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const text = value => typeof value === 'string' ? value : '';
  const hash = value => /^[a-f0-9]{64}$/i.test(value || '');
  const methods = ['original', 'corrected'];
  const outcomes = new Set(['direct', 'table', 'flight_miss', 'setup_rejected', 'invalid_physics', 'arm_assisted_capture']);
  const settings = raw => raw && typeof raw === 'object' ? {
    board_xy_m: vector(raw.board_xy_m, 2) ? raw.board_xy_m : null,
    board_height_m: number(raw.board_height_m), tilt_deg: number(raw.tilt_deg), drop_gap_m: number(raw.drop_gap_m),
    q10: vector(raw.q10, 6) ? raw.q10 : null, q14: vector(raw.q14, 6) ? raw.q14 : null,
    description: text(raw.description),
  } : null;
  const result = raw => {
    if (!raw || typeof raw !== 'object') return null;
    const row = Object.fromEntries(['requested', 'completed', 'released', 'captures', 'direct', 'table', 'robot_contacts'].map(key => [key, count(raw[key])]));
    const recorded = Array.isArray(raw.outcomes) ? raw.outcomes : [];
    row.valid = Object.values(row).every(value => value !== null) && row.requested > 0
      && row.completed === row.requested && row.released <= row.completed && row.robot_contacts <= row.released
      && row.captures === row.direct + row.table && row.captures <= row.released
      && recorded.length === row.completed && recorded.every(value => outcomes.has(value))
      && recorded.filter(value => value === 'direct').length === row.direct
      && recorded.filter(value => value === 'table').length === row.table
      && recorded.filter(value => value !== 'setup_rejected').length === row.released;
    row.rejected = row.completed === null || row.released === null ? null : row.completed - row.released;
    row.outcomes = recorded.filter(value => outcomes.has(value));
    return row;
  };
  const video = raw => {
    if (!raw || typeof raw.asset_id !== 'string' || !hash(raw.trace_sha256)) return null;
    const asset = videoAssets?.[raw.asset_id];
    if (!asset || asset.trace_sha256 !== raw.trace_sha256 || typeof asset.url !== 'string'
        || asset.url.length > 14 * 1024 * 1024 || !/^data:video\/mp4;base64,[A-Za-z0-9+/]+={0,2}$/.test(asset.url)) return null;
    return { asset_id: raw.asset_id, url: asset.url, trace_sha256: raw.trace_sha256, label: text(raw.label) };
  };
  const seen = new Set(), xy = new Set();
  let unique = true;
  const cases = (Array.isArray(section.cases) ? section.cases : []).flatMap(raw => {
    if (typeof raw?.id !== 'string' || !raw.id || !vector(raw.bucket_xy_m, 2)) { unique = false; return []; }
    const coord = JSON.stringify(raw.bucket_xy_m);
    if (seen.has(raw.id) || xy.has(coord)) unique = false;
    seen.add(raw.id); xy.add(coord);
    const d = raw.diagnosis;
    const diagnosis = hash(d?.evidence_sha256) ? {
      summary: text(d.summary), evidence_sha256: d.evidence_sha256,
      events: (Array.isArray(d.events) ? d.events : []).filter(e => Number.isFinite(e?.time_s) && e.time_s >= 0 && typeof e.label === 'string'),
    } : null;
    return [{ id: raw.id, bucket_xy_m: raw.bucket_xy_m, diagnosis,
      settings: Object.fromEntries(methods.map(key => [key, settings(raw.settings?.[key])])),
      results: Object.fromEntries(methods.map(key => [key, result(raw.results?.[key])])),
      videos: Object.fromEntries(methods.map(key => [key, video(raw.videos?.[key])])),
      modification: text(raw.modification),
    }];
  });
  const controls = { status: text(section.controls?.status), completed: count(section.controls?.completed), planned: count(section.controls?.planned) };
  const accounting = unique && cases.length === 3 && cases.every(row => methods.every(key => row.results[key]?.valid)
    && row.results.original.requested === row.results.corrected.requested);
  const complete = section.status === 'completed' && section.audit_status === 'passed' && section.physics_substeps === 64
    && controls.status === 'passed' && controls.planned > 0 && controls.completed === controls.planned && accounting;
  return { complete, accounting, status: text(section.status), phase: text(section.phase), audit: text(section.audit_status),
    completed: count(section.progress?.completed), planned: count(section.progress?.planned), controls,
    correction: { parameter: text(section.correction?.parameter), label: text(section.correction?.label), description: text(section.correction?.description) },
    cases, next: text(section.next), error: text(section.error), substeps: number(section.physics_substeps),
  };
}

/** Development search: preparation, first tries and fresh repeats stay separate. */
export function boardPoseRecoverySummary(study, videoAssets = {}) {
  const section = study?.board_pose_recovery;
  if (section?.version !== 'sircus-board-pose-recovery-v1') return null;
  const number = value => Number.isFinite(value) ? value : null;
  const count = value => Number.isInteger(value) && value >= 0 ? value : null;
  const text = value => typeof value === 'string' ? value : '';
  const hash = value => /^[a-f0-9]{64}$/i.test(value || '');
  const methods = ['original', 'search', 'repeat'];
  const allowed = new Set(['direct', 'table', 'flight_miss', 'setup_rejected', 'invalid_physics', 'arm_assisted_capture']);
  const result = raw => {
    if (!raw) return null;
    const row = Object.fromEntries(['requested', 'completed', 'released', 'captures', 'direct', 'table', 'arm_assisted', 'robot_contacts'].map(k => [k, count(raw[k])]));
    const outcomes = Array.isArray(raw.outcomes) ? raw.outcomes : [];
    row.valid = Object.values(row).every(x => x !== null) && row.completed === row.requested && row.released <= row.completed
      && row.robot_contacts <= row.released && row.captures === row.direct + row.table && row.captures <= row.released
      && outcomes.length === row.completed && outcomes.every(x => allowed.has(x))
      && outcomes.filter(x => x === 'direct').length === row.direct && outcomes.filter(x => x === 'table').length === row.table
      && outcomes.filter(x => x === 'arm_assisted_capture').length === row.arm_assisted
      && outcomes.filter(x => x !== 'setup_rejected').length === row.released;
    return row;
  };
  const settings = raw => raw ? { board_xy_m: vector(raw.board_xy_m, 2) ? raw.board_xy_m : null,
    board_height_m: number(raw.board_height_m), tilt_deg: number(raw.tilt_deg), drop_gap_m: number(raw.drop_gap_m),
    heading_offset_deg: number(raw.heading_offset_deg), twist_deg: number(raw.twist_deg),
    q10: vector(raw.q10, 6) ? raw.q10 : null, q14: vector(raw.q14, 6) ? raw.q14 : null } : null;
  const video = raw => {
    const asset = videoAssets?.[raw?.asset_id];
    return hash(raw?.trace_sha256) && asset?.trace_sha256 === raw.trace_sha256 && typeof asset.url === 'string'
      && asset.url.length <= 14 * 1024 * 1024 && /^data:video\/mp4;base64,[A-Za-z0-9+/]+={0,2}$/.test(asset.url)
      ? { url: asset.url, trace_sha256: raw.trace_sha256, label: text(raw.label) } : null;
  };
  const seen = new Set(), coordinates = new Set(); let valid = true;
  const cases = (Array.isArray(section.cases) ? section.cases : []).flatMap(raw => {
    if (!text(raw?.id) || !vector(raw.bucket_xy_m, 2)) { valid = false; return []; }
    const xy = JSON.stringify(raw.bucket_xy_m);
    if (seen.has(raw.id) || coordinates.has(xy)) valid = false;
    seen.add(raw.id); coordinates.add(xy);
    const preparation = Object.fromEntries(['proposed', 'checked', 'admitted', 'selected'].map(k => [k, count(raw.preparation?.[k])]));
    if (Object.values(preparation).some(x => x === null) || preparation.selected > preparation.admitted || preparation.selected > preparation.proposed) valid = false;
    const confirmed = (Array.isArray(raw.confirmed_settings) ? raw.confirmed_settings : []).map(r => ({
      candidate_id: text(r.candidate_id), captures: count(r.repeat_captures), completed: count(r.repeat_completed), settings: settings(r.settings),
    }));
    if (confirmed.some(r => !r.candidate_id || r.completed !== 3 || r.captures !== 3)
        || new Set(confirmed.map(r => r.candidate_id)).size !== confirmed.length) valid = false;
    return [{ id: raw.id, bucket_xy_m: raw.bucket_xy_m, preparation, confirmed,
      results: Object.fromEntries(methods.map(k => [k, result(raw.results?.[k])])),
      settings: Object.fromEntries(['original', 'selected'].map(k => [k, settings(raw.settings?.[k])])),
      videos: Object.fromEntries(['original', 'selected', 'closest'].map(k => [k, video(raw.videos?.[k])])),
    }];
  });
  const controls = { status: text(section.controls?.status), completed: count(section.controls?.completed), planned: count(section.controls?.planned) };
  const accounting = valid && cases.length === 3 && cases.every(r => methods.every(k => r.results[k]?.valid)
    && r.results.original.requested === 3 && r.results.search.requested === r.preparation.selected
    && r.confirmed.length * 3 <= r.results.repeat.captures);
  const complete = section.status === 'completed' && section.audit_status === 'passed' && section.physics_substeps === 64
    && controls.status === 'passed' && controls.completed === 18 && controls.planned === 18 && accounting;
  return { complete, accounting, cases, controls, status: text(section.status), phase: text(section.phase), audit: text(section.audit_status),
    completed: count(section.progress?.completed), planned: count(section.progress?.planned), substeps: number(section.physics_substeps),
    next: text(section.next), error: text(section.error) };
}


// The existing server serves this module; no new endpoint or restart is needed.
export const { boardFirstValidCounts, boardFirstRescoreReady, boardFirstComparisonReady, renderBoardFirst } = (() => {
// Read-only view. Audited historical relabeling is separate from new test drops.
const VERSION = 'sircus-board-first-display-v1';
const integer = n => Number.isInteger(n) && n >= 0;
const hash = h => typeof h === 'string' && /^[a-f0-9]{64}$/.test(h);
function validCounts(c) {
  return !!c && ['scorable_native','old_captures','new_captures','gained','lost','unchanged_captures','unchanged_misses','invalid_native'].every(k => integer(c[k]))
    && c.old_captures === c.unchanged_captures + c.lost
    && c.new_captures === c.unchanged_captures + c.gained
    && c.scorable_native === c.unchanged_captures + c.gained + c.lost + c.unchanged_misses + c.invalid_native;
}
const audited = x => x?.status === 'completed' && x.audit_status === 'passed'
  && hash(x.provenance?.report_sha256) && hash(x.provenance?.audit_sha256);
function rescoreReady(x) {
  if (!audited(x) || !validCounts(x.counts) || !Array.isArray(x.positions)) return false;
  const keys = new Set();
  for (const p of x.positions) {
    if (!Array.isArray(p.bucket_xy_m) || p.bucket_xy_m.length !== 2 || !p.bucket_xy_m.every(Number.isFinite) || !validCounts(p.counts)) return false;
    const key = JSON.stringify(p.bucket_xy_m); if (keys.has(key)) return false; keys.add(key);
  }
  return ['scorable_native','old_captures','new_captures','gained','lost','unchanged_captures','unchanged_misses','invalid_native'].every(k => x.positions.reduce((a,p)=>a+p.counts[k],0) === x.counts[k]);
}
function comparisonReady(x) {
  return audited(x) && x.physics_substeps === 64 && x.positions === 50 && x.frozen_before_test === true
    && Array.isArray(x.methods) && x.methods.length === 2 && new Set(x.methods.map(m=>m.id)).size === 2
    && x.methods.every(m => hash(m.checkpoint_sha256) && m.attempts === 50
      && ['captures','setup_rejections','released_misses','invalid_physics'].every(k=>integer(m[k]))
      && m.captures + m.setup_rejections + m.released_misses + m.invalid_physics === 50);
}
function table(root, id, headings, rows) {
  const node = root.querySelector(`[data-board-first="${id}"]`); node.replaceChildren();
  if (!rows.length) return;
  const doc = root.ownerDocument, tab = doc.createElement('table'), head = doc.createElement('thead'), body = doc.createElement('tbody');
  const line = (values, tag) => { const tr=doc.createElement('tr'); for(const value of values){const td=doc.createElement(tag);td.textContent=String(value);tr.append(td);}return tr; };
  head.append(line(headings,'th')); for(const row of rows)body.append(line(row,'td'));
  tab.append(head,body);node.append(tab);
}
function renderBoardFirst(root, raw) {
  root.hidden = raw?.version !== VERSION; if(root.hidden)return;
  const text = (id,value) => {root.querySelector(`[data-board-first="${id}"]`).textContent=value;};
  const r=raw.rescore, c=raw.comparison, ready=rescoreReady(r), tested=comparisonReady(c);
  text('status', ready ? 'Saved-shot rescoring complete · original labels preserved' : 'Checking saved trajectories · final counts await the independent audit');
  text('scope', raw.scope ?? 'Simulation only. Historical relabeling creates no new drops.');
  table(root,'counts',['Saved drops scored','Previously accepted','Accepted under board-first rule','Newly accepted','No longer accepted'],ready?[[r.counts.scorable_native,r.counts.old_captures,r.counts.new_captures,r.counts.gained,r.counts.lost]]:[]);
  text('changes',ready?`${r.counts.gained} saved captures are newly accepted; ${r.counts.lost} are not confirmed under the board-first evidence rule. ${r.counts.invalid_native} invalid-physics traces remain excluded. Contact presence does not prove that the contact caused a bounce.`:'No new capture total is claimed yet.');
  const changed=ready?r.positions.filter(p=>p.counts.gained||p.counts.lost).sort((a,b)=>b.counts.gained-a.counts.gained||b.counts.lost-a.counts.lost):[];
  table(root,'positions',['Bucket X, Y (mm)','Saved drops','Old captures','New captures','Gained / lost'],changed.map(p=>[p.bucket_xy_m.map(v=>(1000*v).toFixed(1)).join(', '),p.counts.scorable_native,p.counts.old_captures,p.counts.new_captures,`${p.counts.gained} / ${p.counts.lost}`]));
  text('training',raw.training_text??'The same small network ranks proposed shots. It is retrained on corrected TRAIN labels, selected using validation data, then frozen before new test drops.');
  text('test-status',tested?'Fresh 50-position comparison complete · both checkpoints frozen':`Fresh 50-position comparison: ${c?.phase??'pending'}. Results appear after the independent audit.`);
  table(root,'comparison',['Frozen selector','Captures / 50','Setup refused','Released misses','Invalid physics','Mean decision time (s)'],tested?c.methods.map(m=>[m.label,`${m.captures} / 50`,m.setup_rejections,m.released_misses,m.invalid_physics,Number.isFinite(m.mean_planning_s)?m.mean_planning_s.toFixed(2):'—']):[]);
  text('conclusion',tested?c.conclusion:'Re-scoring alone is not evidence that a newly trained model generalizes better.');
  text('proof',ready?`Rescore report SHA-256: ${r.provenance.report_sha256}. Independent audit: ${r.provenance.audit_sha256}.`:'');
}

return { boardFirstValidCounts:validCounts, boardFirstRescoreReady:rescoreReady, boardFirstComparisonReady:comparisonReady, renderBoardFirst };
})();

export const { expandedSelectorReady, renderExpandedSelector } = (() => {
const VERSION = 'sircus-expanded-selector-display-v1';
const integer = n => Number.isInteger(n) && n >= 0;
const hash = h => typeof h === 'string' && /^[a-f0-9]{64}$/.test(h);
const outcomes = new Set(['capture','setup refused','flight miss','invalid physics']);
function ready(x) {
  if (x?.version !== VERSION || x.status !== 'completed' || x.audit_status !== 'passed'
    || x.positions !== 50 || x.physics_substeps !== 64 || x.frozen_before_test !== true
    || !hash(x.provenance?.report_sha256) || !hash(x.provenance?.audit_sha256)
    || !Array.isArray(x.methods) || x.methods.length !== 2
    || x.methods.map(m=>m.id).sort().join(',') !== 'current,expanded') return false;
  if (!x.methods.every(m => m.attempts === 50 && hash(m.checkpoint_sha256)
    && ['captures','setup_rejections','released_misses','invalid_physics'].every(k=>integer(m[k]))
    && m.captures+m.setup_rejections+m.released_misses+m.invalid_physics === 50
    && Number.isFinite(m.mean_decision_s) && m.mean_decision_s >= 0
    && Number.isFinite(m.mean_request_s) && m.mean_request_s >= m.mean_decision_s)) return false;
  if (!Array.isArray(x.library_rows) || x.library_rows.length !== 2
    || !x.library_rows.every(r=>typeof r.label==='string' && ['positions','settings','capturing_settings'].every(k=>integer(r[k])) && r.capturing_settings<=r.settings)
    || !Array.isArray(x.results) || x.results.length !== 50) return false;
  const positions = new Set();
  for (const row of x.results) {
    if (!Array.isArray(row.bucket_xy_m) || row.bucket_xy_m.length !== 2 || !row.bucket_xy_m.every(Number.isFinite)
      || !outcomes.has(row.current) || !outcomes.has(row.expanded)) return false;
    const key = JSON.stringify(row.bucket_xy_m); if (positions.has(key)) return false; positions.add(key);
  }
  return x.methods.every(m => [['captures','capture'],['setup_rejections','setup refused'],['released_misses','flight miss'],['invalid_physics','invalid physics']]
    .every(([key,value])=>x.results.filter(r=>r[m.id]===value).length === m[key]));
}
function table(root, id, headings, rows) {
  const node=root.querySelector(`[data-expanded="${id}"]`); node.replaceChildren();
  if (!rows.length) return;
  const doc=root.ownerDocument, tab=doc.createElement('table'), head=doc.createElement('thead'), body=doc.createElement('tbody');
  const line=(values,tag)=>{const tr=doc.createElement('tr');for(const value of values){const td=doc.createElement(tag);td.textContent=String(value);tr.append(td);}return tr;};
  head.append(line(headings,'th'));for(const row of rows)body.append(line(row,'td'));tab.append(head,body);node.append(tab);
}
function render(root, raw) {
  root.hidden=raw?.version !== VERSION;if(root.hidden)return;
  const text=(id,value)=>{root.querySelector(`[data-expanded="${id}"]`).textContent=value;};
  const complete=ready(raw);
  text('status',complete?'Expanded-library comparison complete · independently checked':`Expanded-library comparison: ${raw.phase??'preparing'}. Final capture counts await the evidence audit.`);
  text('data',raw.training_text??'Preparing distinct shot settings and position-separated training and validation data.');
  table(root,'library',['Data split','Bucket positions','Distinct settings','Settings with captures'],complete?raw.library_rows.map(r=>[r.label,r.positions,r.settings,r.capturing_settings]):[]);
  const ordered=complete?['current','expanded'].map(id=>raw.methods.find(m=>m.id===id)):[];
  table(root,'comparison',['Result','Current planner','Expanded selector'],complete?[
    ['Captured and retained',...ordered.map(m=>`${m.captures} / 50`)],
    ['No checked setup',...ordered.map(m=>m.setup_rejections)],
    ['Released but missed',...ordered.map(m=>m.released_misses)],
    ['Invalid physics',...ordered.map(m=>m.invalid_physics)],
    ['Mean decision time (s)',...ordered.map(m=>m.mean_decision_s.toFixed(2))],
    ['Mean full request time (s)',...ordered.map(m=>m.mean_request_s.toFixed(2))],
  ]:[]);
  table(root,'positions',['Bucket X, Y (mm)','Current planner','Expanded selector'],complete?raw.results.map(r=>[r.bucket_xy_m.map(v=>(1000*v).toFixed(1)).join(', '),r.current,r.expanded]):[]);
  text('conclusion',complete?raw.conclusion:'No improvement is claimed before all test attempts and controls are checked.');
  text('timing',complete?raw.timing_text:'');
  text('proof',complete?`Report SHA-256: ${raw.provenance.report_sha256}. Independent audit: ${raw.provenance.audit_sha256}. Checkpoint and library identities are bound before test drops.`:'');
}
return {expandedSelectorReady:ready,renderExpandedSelector:render};
})();

// A read-only view of four audited saved flights. It never calls the planner.
export const { expandedSelectorReplaysReady, expandedSelectorReplayMovie, renderExpandedSelectorReplays } = (() => {
  const VERSION = 'sircus-expanded-selector-replays-display-v1';
  const METHODS = ['current', 'expanded'];
  const CASES = ['04', '10', '19', '47'].map(n => `expanded-selector-test-${n}`);
  const hash = h => typeof h === 'string' && /^[a-f0-9]{64}$/.test(h);
  const vector = (v,n) => Array.isArray(v) && v.length === n && v.every(Number.isFinite);
  const cache = new WeakMap();
  function ready(x) {
    if (x?.version !== VERSION || x.status !== 'completed' || x.audit_status !== 'passed'
      || x.physics_substeps !== 64 || !['report_sha256','audit_sha256','diagnosis_sha256'].every(k=>hash(x.provenance?.[k]))
      || x.parent_summary?.positions !== 50 || x.parent_summary.current_captures !== 44 || x.parent_summary.expanded_captures !== 44
      || !Array.isArray(x.cases) || x.cases.length !== 4 || x.cases.map(c=>c.id).join() !== CASES.join()) return false;
    return x.cases.every((c,i) => {
      const lost = i === 0 || i === 2;
      if (c.change !== (lost?'expanded_lost':'expanded_gained') || !vector(c.bucket_xy_m,2)
        || !Array.isArray(c.findings) || !c.findings.every(t=>typeof t==='string')
        || !Array.isArray(c.hypotheses) || !c.hypotheses.every(t=>typeof t==='string')) return false;
      return METHODS.every(method => {
        const m=c.methods?.[method], s=m?.settings;
        return m?.outcome?.capture === (method==='current'?lost:!lost)
          && typeof m.id==='string' && typeof m.summary==='string' && Number.isFinite(m.outcome.max_retention_s) && m.outcome.max_retention_s>=0
          && vector(s?.board_center_m,3) && vector(s?.command_board_center_m,3) && vector(s?.release_ball_center_m,3)
          && ['board_height_m','tilt_deg','drop_gap_m','heading_offset_deg','twist_deg'].every(k=>Number.isFinite(s[k]))
          && s.drop_gap_m>0 && vector(s.q10_rad,6) && vector(s.q14_rad,6)
          && hash(m.trace?.sha256) && Number.isInteger(m.trace.column) && m.trace.column>=0;
      });
    });
  }
  function movie(c,assets) {
    const v=c.video, a=v && assets?.[v.asset_id];
    if (!a || !hash(v.video_sha256) || !hash(v.receipt_sha256)
      || a.video_sha256!==v.video_sha256 || a.receipt_sha256!==v.receipt_sha256
      || a.case_id!==c.id || a.native_steps_added!==0 || a.held_arm_poses!==true
      || !METHODS.every(m=>a.methods?.[m]?.record_id===c.methods[m].id
        && a.methods[m].trace_sha256===c.methods[m].trace.sha256 && a.methods[m].column===c.methods[m].trace.column)
      || typeof a.url!=='string' || a.url.length>80*1024*1024
      || !/^data:video\/mp4;base64,[A-Za-z0-9+/]+={0,2}$/.test(a.url)) return null;
    return a;
  }
  function render(root,raw,assets={}) {
    root.hidden=raw?.version!==VERSION;if(root.hidden)return;
    const signature=JSON.stringify([raw,Object.entries(assets).map(([k,v])=>[k,v?.video_sha256,v?.receipt_sha256,v?.case_id,v?.url?.length])]);
    if(cache.get(root)===signature)return;cache.set(root,signature);
    const doc=root.ownerDocument, find=k=>root.querySelector(`[data-expanded-replay="${k}"]`);
    const complete=ready(raw), cases=find('cases');cases.replaceChildren();
    find('status').textContent=complete?'Four paired saved-flight replays · evidence checked':'Saved-flight comparison awaits verified evidence. No final outcomes are shown yet.';
    find('summary').textContent=complete?'Both methods captured 44 of 50 new bucket positions. The expanded selector gained two positions and lost two. The current interactive planner remains unchanged.':'';
    find('proof').textContent=complete?`Parent report: ${raw.provenance.report_sha256}. Parent audit: ${raw.provenance.audit_sha256}. Diagnosis: ${raw.provenance.diagnosis_sha256}.`:'';
    if(!complete)return;
    const el=(parent,tag,text,cls)=>{const n=doc.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;parent.append(n);return n;};
    const mm=v=>(v*1000).toFixed(1), deg=v=>v.toFixed(2), vec=v=>v.map(mm).join(', ');
    const table=(parent,rows)=>{
      const wrap=el(parent,'div',undefined,'table-wrap'),t=el(wrap,'table'),head=el(t,'thead'),h=el(head,'tr'),body=el(t,'tbody');
      ['Setting','Current planner','Expanded selector'].forEach(x=>{el(h,'th',x).scope='col';});
      for(const row of rows){const tr=el(body,'tr');row.forEach((x,i)=>{const cell=el(tr,i?'td':'th',x);if(!i)cell.scope='row';});}
    };
    for(const c of raw.cases){
      const card=el(cases,'article',undefined,'hybrid-miss-case');card.id=`expanded-selector-replay-${c.id.slice(-2)}`;
      el(card,'h3',`Case ${c.id.slice(-2)} · bucket (${vec(c.bucket_xy_m)}) mm · expanded selector ${c.change==='expanded_lost'?'lost':'gained'} this capture`);
      const list=el(card,'ul',undefined,'diagnosis-list');for(const m of METHODS)el(list,'li',`${m==='current'?'Current planner':'Expanded selector'}: ${c.methods[m].summary}`);
      const settings=(key,format)=>METHODS.map(m=>format(c.methods[m].settings[key]));
      table(card,[
        ['Recorded outcome',...METHODS.map(m=>c.methods[m].outcome.capture?`Capture · ${c.methods[m].outcome.route_class}`:'Miss')],
        ['Measured board center X, Y, Z (mm)',...settings('board_center_m',vec)],
        ['Board tilt from horizontal (°)',...settings('tilt_deg',deg)],
        ['Drop gap (mm)',...settings('drop_gap_m',mm)],
        ['Ball release center X, Y, Z (mm)',...settings('release_ball_center_m',vec)],
        ['Longest continuous whole-ball retention (s)',...METHODS.map(m=>c.methods[m].outcome.max_retention_s.toFixed(3))],
      ]);
      const detail=el(card,'details');el(detail,'summary','Orientation, arm joints and source identities');
      table(detail,[
        ['Commanded board center X, Y, Z (mm)',...settings('command_board_center_m',vec)],
        ['Commanded board center height (mm)',...settings('board_height_m',mm)],
        ['Heading offset / board twist (°)',...METHODS.map(m=>`${deg(c.methods[m].settings.heading_offset_deg)} / ${deg(c.methods[m].settings.twist_deg)}`)],
        ['Dropper arm joints (rad)',...settings('q10_rad',v=>v.map(n=>n.toFixed(5)).join(', '))],
        ['Board arm joints (rad)',...settings('q14_rad',v=>v.map(n=>n.toFixed(5)).join(', '))],
        ['Saved setting used as a starting candidate',...METHODS.map(m=>c.methods[m].selection?.source_id??'Not recorded')],
        ['Deduplicated source key',...METHODS.map(m=>c.methods[m].selection?.source_setting_key??'No expanded-library key')],
        ['Source bucket X, Y (mm)',...METHODS.map(m=>vector(c.methods[m].selection?.source_bucket_xy_m,2)?vec(c.methods[m].selection.source_bucket_xy_m):'Not recorded')],
        ['Source-to-target bucket distance (mm)',...METHODS.map(m=>Number.isFinite(c.methods[m].selection?.source_distance_m)?mm(c.methods[m].selection.source_distance_m):'Not recorded')],
        ['Intended route / predicted flight route',...METHODS.map(m=>`${c.methods[m].selection?.intended_route??'Not recorded'} / ${c.methods[m].selection?.prediction?.predicted_route??'Not recorded'}`)],
        ['Learned capture estimate (not a guarantee)',...METHODS.map(m=>Number.isFinite(c.methods[m].selection?.probabilities?.capture_probability)?`${(100*c.methods[m].selection.probabilities.capture_probability).toFixed(1)}%`:'Not recorded')],
        ['How this estimate was used',...METHODS.map(m=>c.methods[m].selection?.probability_use==='used_for_neural_proposal_order'?'Used to rank proposed shots':c.methods[m].selection?.probability_use==='posthoc_score_of_unranked_joint_alternative'?'Computed after the test for explanation; did not choose this arm alternative':'Not recorded')],
        ['Recorded shot ID',...METHODS.map(m=>c.methods[m].id)],
        ['Saved trajectory SHA-256',...METHODS.map(m=>c.methods[m].trace.sha256)],
      ]);
      const measured=el(card,'details');el(measured,'summary','Measured contact timing and flight details');
      const eventList=el(measured,'ul',undefined,'diagnosis-list');for(const fact of c.findings)el(eventList,'li',fact);
      if(c.hypotheses.length){el(measured,'p','Possible next changes — not yet tested','fine');const l=el(measured,'ul',undefined,'diagnosis-list');for(const h of c.hypotheses)el(l,'li',h);}
      const clip=movie(c,assets),figure=el(card,'figure');
      if(clip){
        el(figure,'figcaption','Current planner (left) · expanded selector (right). Recorded samples replay at ¼ speed; a finished side holds its final recorded frame.');
        const player=el(figure,'video');player.controls=true;player.preload='none';player.playsInline=true;player.src=clip.url;
        player.setAttribute('aria-label',`Paired saved flights at bucket ${vec(c.bucket_xy_m)} mm`);
        const controls=el(figure,'div',undefined,'replay-controls');
        const replay=el(controls,'button','Replay from start','quiet');replay.type='button';
        const label=el(controls,'label','Playback speed '),speed=el(label,'select');speed.setAttribute('aria-label','Playback speed relative to simulation');
        for(const [value,text]of [['1','¼× simulation'],['2','½× simulation'],['4','1× simulation']]){const o=el(speed,'option',text);o.value=value;}
        const note=el(figure,'p','','fine');
        replay.onclick=()=>{player.currentTime=0;Promise.resolve(player.play()).catch(()=>{note.textContent='Use the video Play control to start playback.';});};
        speed.onchange=()=>{player.playbackRate=Number(speed.value);};
        el(figure,'p',`Video SHA-256: ${c.video.video_sha256}`,'fine evidence-id');
      }else el(figure,'p','The paired movie will appear after its file and both source traces are verified.','fine');
    }
  }
  return {expandedSelectorReplaysReady:ready,expandedSelectorReplayMovie:movie,renderExpandedSelectorReplays:render};
})();

// Read-only native64 diagnostic; this module never selects or executes a shot.
export const {bucketApproachReady, renderBucketApproach} = (() => {
  const VERSION='sircus-bucket-approach-display-v1';
  const cache=new WeakMap();
  const CASES=['04','10','19'].map(n=>`expanded-selector-test-${n}`);
  const METHODS=['original','corrected'];
  const hash=v=>typeof v==='string' && /^[a-f0-9]{64}$/.test(v);
  const integer=v=>Number.isInteger(v) && v>=0;
  const vector=(v,n)=>Array.isArray(v) && v.length===n && v.every(Number.isFinite);
  const settings=s=>s===null || (vector(s?.values,5) && s.values[4]>0 && vector(s.q10_rad,6) && vector(s.q14_rad,6)
    && vector(s.release_ball_center_m,3) && Number.isFinite(s.heading_offset_deg) && Number.isFinite(s.twist_deg));
  function counts(r){return r && r.requested===3 && r.completed===3
    && ['released','captures','flight_misses','setup_rejections','invalid_physics','robot_contacts','bucket_contacts'].every(k=>integer(r[k]))
    && r.released+r.setup_rejections===3 && r.captures+r.flight_misses+r.invalid_physics===r.released
    && r.robot_contacts<=r.released && r.bucket_contacts<=r.released;}
  function forecast(f){return f && typeof f.summary==='string' && typeof f.limits==='string'
    && ['first_collision_s','minimum_margin_m','opening_margin_m'].every(k=>f[k]===null || Number.isFinite(f[k]))
    && (f.first_collision_s===null || f.first_collision_s>=0)
    && (f.collision===null || typeof f.collision==='boolean');}
  function ready(x){
    if(x?.version!==VERSION || x.status!=='completed' || x.audit_status!=='passed' || x.scope!=='development'
      || x.physics_substeps!==64 || x.frozen_before_native!==true
      || !['report_sha256','audit_sha256','preparation_sha256'].every(k=>hash(x.provenance?.[k]))
      || !integer(x.checks_count) || x.checks_count<1
      || !Array.isArray(x.cases) || x.cases.map(c=>c.id).join()!==CASES.join()
      || !Array.isArray(x.diagnostic_controls) || x.diagnostic_controls.length!==1
      || !Array.isArray(x.preservation) || x.preservation.length!==4
      || new Set(x.preservation.map(p=>p.id)).size!==4
      || !integer(x.native_releases) || !['controls_pre','controls_post'].every(p=>x.controls?.[p]?.status==='passed' && x.controls[p].requested===9))return false;
    for(const c of x.cases){
      if(!vector(c.bucket_xy_m,2) || !['current','expanded'].includes(c.original_method)
        || typeof c.selection_comparison?.changed!=='boolean' || typeof c.selection_comparison.statement!=='string'
        || typeof c.selection_comparison.without_check_id!=='string' || !vector(c.selection_comparison.without_check_values,5)
        || !integer(c.selection_comparison.checked_proposals) || !integer(c.selection_comparison.admitted_candidates)
        || (c.selection_comparison.with_check_id!==null && typeof c.selection_comparison.with_check_id!=='string')
        || (c.selection_comparison.with_check_values!==null && !vector(c.selection_comparison.with_check_values,5)))return false;
      for(const key of METHODS){const m=c.methods?.[key];if(!counts(m)||!settings(m.settings)||!forecast(m.forecast)
        || (m.settings===null)!==(m.setup_rejections===3))return false;}
    }
    if(![...x.preservation,...x.diagnostic_controls].every(p=>vector(p.bucket_xy_m,2)&&counts(p)&&settings(p.settings)&&p.settings!==null&&forecast(p.forecast)))return false;
    return x.native_releases===18+[...x.preservation,...x.diagnostic_controls].reduce((n,p)=>n+p.released,0)
      +x.cases.reduce((n,c)=>n+METHODS.reduce((s,k)=>s+c.methods[k].released,0),0);
  }
  function render(root,raw){
    root.hidden=raw?.version!==VERSION;if(root.hidden)return;
    const signature=JSON.stringify(raw);if(cache.get(root)===signature)return;cache.set(root,signature);
    const doc=root.ownerDocument,find=k=>root.querySelector(`[data-bucket-approach="${k}"]`),complete=ready(raw);
    for(const k of ['cases','preservation','diagnostic'])find(k).replaceChildren();
    find('status').textContent=complete?'Bucket-approach correction complete · native drops independently checked':'Bucket-approach test awaits a completed independent audit. Final capture counts are not shown yet.';
    for(const k of ['summary','controls','conclusion','proof','limits'])find(k).textContent='';
    if(!complete)return;
    const el=(p,tag,text,cls)=>{const n=doc.createElement(tag);if(text!==undefined)n.textContent=String(text);if(cls)n.className=cls;p.append(n);return n;};
    const table=(parent,headings,rows)=>{const t=el(parent,'table'),h=el(el(t,'thead'),'tr'),b=el(t,'tbody');headings.forEach(v=>{el(h,'th',v).scope='col';});for(const values of rows){const tr=el(b,'tr');values.forEach((v,i)=>{const n=el(tr,i?'td':'th',v);if(!i)n.scope='row';});}};
    const mm=n=>(1000*n).toFixed(1),vec=v=>v.map(mm).join(', '),number=(n,format)=>n===null?'Not predicted':format(n);
    const sum=k=>raw.cases.reduce((n,c)=>n+c.methods[k].captures,0);
    find('summary').textContent=`Three previously missed placements, three repeat slots each: unchanged shots ${sum('original')}/9 captures; approach-checked selection ${sum('corrected')}/9. Setup refusals remain unsuccessful attempts. Choices were frozen before these drops.`;
    for(const c of raw.cases){
      const card=el(find('cases'),'article',undefined,'hybrid-miss-case');card.id=`bucket-approach-${c.id.slice(-2)}`;
      el(card,'h3',`Case ${c.id.slice(-2)} · bucket (${vec(c.bucket_xy_m)}) mm`);
      el(card,'p',`Original shot came from the ${c.original_method==='current'?'current planner':'expanded selector'} in the earlier comparison.`,'fine');
      const comparison=c.selection_comparison;
      el(card,'p',`${comparison.statement} (${comparison.admitted_candidates} admitted candidates from ${comparison.checked_proposals} setup checks.)`,'scope-note');
      const pair=METHODS.map(k=>c.methods[k]);
      const wrap=el(card,'div',undefined,'table-wrap');
      table(wrap,['Result / setting','Unchanged missed shot','Approach-checked choice'],[
        ['Captured and retained',...pair.map(m=>`${m.captures} / 3`)],
        ['Released but missed',...pair.map(m=>m.flight_misses)],
        ['No checked setup / invalid physics',...pair.map(m=>`${m.setup_rejections} / ${m.invalid_physics}`)],
        ['Shots with robot / pre-entry bucket contact',...pair.map(m=>`${m.robot_contacts} / ${m.bucket_contacts}`)],
        ['Commanded board center X, Y, Z (mm)',...pair.map(m=>m.settings?vec(m.settings.values.slice(0,3)):'No selection')],
        ['Board tilt from horizontal (°)',...pair.map(m=>m.settings?m.settings.values[3].toFixed(2):'—')],
        ['Drop gap (mm)',...pair.map(m=>m.settings?mm(m.settings.values[4]):'—')],
        ['Ball release center X, Y, Z (mm)',...pair.map(m=>m.settings?vec(m.settings.release_ball_center_m):'—')],
        ['Forecast approach',...pair.map(m=>m.forecast.summary)],
        ['Predicted first wall contact (s)',...pair.map(m=>number(m.forecast.first_collision_s,n=>n.toFixed(4)))],
        ['Predicted minimum wall margin (mm)',...pair.map(m=>number(m.forecast.minimum_margin_m,mm))],
        ['Predicted opening margin (mm)',...pair.map(m=>number(m.forecast.opening_margin_m,mm))],
      ]);
      const rankDetail=el(card,'details');el(rankDetail,'summary','What would ranking without the new check choose?');
      el(rankDetail,'p',`Without full-path checking: ${comparison.without_check_id}. With checking: ${comparison.with_check_id??'no selection'}.`,'fine');
      const recipe=v=>v?`board (${vec(v.slice(0,3))}) mm; tilt ${v[3].toFixed(2)}°; gap ${mm(v[4])} mm`:'No selection';
      el(rankDetail,'p',`Without: ${recipe(comparison.without_check_values)}. With: ${recipe(comparison.with_check_values)}.`,'fine');
      el(rankDetail,'p','This compares ranking on exactly the same admitted candidates. Only the selected correction and unchanged original were dropped; a different counterfactual choice was not tested. More candidate search can itself improve the result.','fine');
      const detail=el(card,'details');el(detail,'summary','Arm joints, orientation and forecast limits');
      table(el(detail,'div',undefined,'table-wrap'),['Setting','Unchanged missed shot','Approach-checked choice'],[
        ['Heading offset / board twist (°)',...pair.map(m=>m.settings?`${m.settings.heading_offset_deg.toFixed(2)} / ${m.settings.twist_deg.toFixed(2)}`:'—')],
        ['Dropper joints (rad)',...pair.map(m=>m.settings?m.settings.q10_rad.map(n=>n.toFixed(5)).join(', '):'—')],
        ['Bouncer joints (rad)',...pair.map(m=>m.settings?m.settings.q14_rad.map(n=>n.toFixed(5)).join(', '):'—')],
        ['Forecast limitation',...pair.map(m=>m.forecast.limits)],
      ]);
    }
    table(find('diagnostic'),['Unchanged shot','Bucket X, Y (mm)','Captures','Robot contact repeats'],raw.diagnostic_controls.map(p=>[p.id,vec(p.bucket_xy_m),`${p.captures} / 3`,`${p.robot_contacts} / 3`]));
    table(find('preservation'),['Saved successful shot','Bucket X, Y (mm)','Captures','Forecast approach'],raw.preservation.map(p=>[p.id,vec(p.bucket_xy_m),`${p.captures} / 3`,p.forecast.summary]));
    const preserved=raw.preservation.reduce((n,p)=>n+p.captures,0);
    const warnings=raw.preservation.filter(p=>p.captures>0 && p.forecast.collision===true).length;
    find('controls').textContent=`${warnings?`${warnings} previously successful setting(s) captured despite a nominal bucket-collision warning. `:''}Unchanged successful shots: ${preserved}/12 captures. All 18 positive/negative control checks passed. ${raw.native_releases} native releases; ${raw.checks_count.toLocaleString()} independent evidence checks.`;
    find('conclusion').textContent=raw.conclusion;
    find('limits').textContent=raw.limits;
    find('proof').textContent=`Report: ${raw.provenance.report_sha256}. Independent audit: ${raw.provenance.audit_sha256}. Frozen preparation: ${raw.provenance.preparation_sha256}.`;
  }
  return {bucketApproachReady:ready,renderBucketApproach:render};
})();

// Additive result display: no planner API, execution controls or automatic promotion.
export const {approachComparisonReady, renderApproachComparison} = (() => {
  const VERSION='sircus-approach-comparison-display-v1', METHODS=['current','approach'];
  const hash=h=>typeof h==='string' && /^[a-f0-9]{64}$/.test(h);
  const integer=n=>Number.isInteger(n) && n>=0;
  const nonnegative=n=>Number.isFinite(n) && n>=0;
  const vector=v=>Array.isArray(v) && v.length===2 && v.every(Number.isFinite);
  const outcomes=['capture','flight miss','setup refused','invalid physics'];
  const cache=new WeakMap();
  function ready(x){
    if(x?.version!==VERSION || x.status!=='completed' || x.audit_status!=='passed' || x.positions!==50
      || x.physics_substeps!==64 || x.frozen_before_test!==true || x.attempts_per_position!==1
      || !nonnegative(x.planning_budget_s) || x.planning_budget_s<=0 || !integer(x.checks_count) || x.checks_count<1
      || !['report_sha256','audit_sha256','contract_sha256','manifest_sha256'].every(k=>hash(x.provenance?.[k]))
      || !Array.isArray(x.methods) || x.methods.length!==2 || x.methods.map(m=>m.id).sort().join()!==[...METHODS].sort().join()
      || !Array.isArray(x.results) || x.results.length!==50 || !integer(x.native_releases)
      || !nonnegative(x.gpu_wait_s) || !nonnegative(x.gpu_lease_s) || x.gpu_lease_s<=0 || x.controls_passed!==18
      || typeof x.conclusion!=='string' || typeof x.protocol!=='string')return false;
    for(const m of x.methods){
      if(m.attempts!==50 || !hash(m.checkpoint_sha256) || !['captures','releases','setup_rejections','released_misses','invalid_physics'].every(k=>integer(m[k]))
        || m.releases+m.setup_rejections!==50 || m.captures+m.released_misses+m.invalid_physics!==m.releases
        || !['mean','median','p90'].every(k=>nonnegative(m.decision_s?.[k])&&nonnegative(m.request_s?.[k])&&m.request_s[k]>=m.decision_s[k])
        || m.decision_s.p90<m.decision_s.median || m.request_s.p90<m.request_s.median)return false;
    }
    const ids=new Set(),coordinates=new Set(),paired={both:0,current_only:0,approach_only:0,neither:0};
    for(const r of x.results){
      if(typeof r.id!=='string' || ids.has(r.id) || !vector(r.bucket_xy_m) || coordinates.has(JSON.stringify(r.bucket_xy_m)))return false;
      ids.add(r.id);coordinates.add(JSON.stringify(r.bucket_xy_m));
      for(const method of METHODS){const a=r[method];if(!outcomes.includes(a?.outcome)||!nonnegative(a.decision_s)||!nonnegative(a.request_s)||a.request_s<a.decision_s)return false;}
      const old=r.current.outcome==='capture',changed=r.approach.outcome==='capture';paired[old&&changed?'both':old?'current_only':changed?'approach_only':'neither']++;
    }
    if(!Object.keys(paired).every(k=>x.paired?.[k]===paired[k]))return false;
    if(!x.methods.every(m=>[['captures','capture'],['released_misses','flight miss'],['setup_rejections','setup refused'],['invalid_physics','invalid physics']].every(([k,o])=>x.results.filter(r=>r[m.id].outcome===o).length===m[k])))return false;
    const stats=values=>{const sorted=[...values].sort((a,b)=>a-b);return {mean:values.reduce((a,b)=>a+b,0)/values.length,median:(sorted[24]+sorted[25])/2,p90:sorted[Math.ceil(.9*sorted.length)-1]};};
    for(const m of x.methods){for(const field of ['decision_s','request_s']){const expected=stats(x.results.map(r=>r[m.id][field]));if(!['mean','median','p90'].every(k=>Math.abs(m[field][k]-expected[k])<=1e-8))return false;}}
    return x.native_releases===18+x.methods.reduce((n,m)=>n+m.releases,0);
  }
  function render(root,raw){
    root.hidden=raw?.version!==VERSION;if(root.hidden)return;
    const signature=JSON.stringify(raw);if(cache.get(root)===signature)return;cache.set(root,signature);
    const doc=root.ownerDocument,find=k=>root.querySelector(`[data-approach-comparison="${k}"]`),complete=ready(raw);
    for(const k of ['outcomes','times','positions'])find(k).replaceChildren();
    for(const k of ['status','progress','paired','timing-note','gpu','conclusion','protocol','proof'])find(k).textContent='';
    const timestamp=Number.isFinite(raw.observed_at_unix)?new Date(raw.observed_at_unix*1000).toLocaleString():null;
    if(!complete){
      find('status').textContent=raw.status==='failed'?'Comparison stopped before an audited result.':raw.queue_expired===true?'GPU queue wait expired before simulation started; a retry is needed.':raw.phase==='waiting_for_4090'?'Last reported: waiting for the RTX 4090.':'Comparison in progress — final capture rates await the independent audit.';
      find('progress').textContent=raw.progress_text??'No final capture counts are published yet.';
      find('gpu').textContent=raw.phase==='waiting_for_4090'&&nonnegative(raw.gpu_wait_s)?`Cumulative GPU queue wait at the recorded update: ${(raw.gpu_wait_s/60).toFixed(1)} min. Existing workloads are not interrupted.`:'';
      find('proof').textContent=timestamp?`Status snapshot recorded ${timestamp}. This is the last reported state, not a live process check.`:'';
      return;
    }
    const element=(parent,tag,value)=>{const node=doc.createElement(tag);if(value!==undefined)node.textContent=String(value);parent.append(node);return node;};
    const table=(slot,headings,rows)=>{const tab=element(find(slot),'table'),head=element(element(tab,'thead'),'tr'),body=element(tab,'tbody');headings.forEach(t=>{element(head,'th',t).scope='col';});for(const row of rows){const tr=element(body,'tr');row.forEach((value,i)=>{const cell=element(tr,i?'td':'th',value);if(!i)cell.scope='row';});}};
    const ordered=METHODS.map(id=>raw.methods.find(m=>m.id===id)),fmt=n=>n.toFixed(2);
    find('status').textContent='50-position comparison complete · native outcomes independently checked';
    find('progress').textContent='Both planners were frozen before test drops. Each had one attempt at the same 50 new positions; setup refusals remain in the denominator.';
    table('outcomes',['Result','Current planner','Expanded search + approach check'],[
      ['Captured and retained',...ordered.map(m=>`${m.captures} / 50 (${(2*m.captures).toFixed(0)}%)`)],
      ['Released balls',...ordered.map(m=>m.releases)],['No checked setup',...ordered.map(m=>m.setup_rejections)],
      ['Released but missed',...ordered.map(m=>m.released_misses)],['Invalid physics',...ordered.map(m=>m.invalid_physics)],
    ]);
    find('paired').textContent=`Both captured ${raw.paired.both} positions. New method gained ${raw.paired.approach_only}, lost ${raw.paired.current_only}; neither captured ${raw.paired.neither}.`;
    table('times',['Planning time (seconds)','Current planner','Expanded search + approach check'],[
      ...[['mean','Mean'],['median','Median'],['p90','90th percentile']].map(([key,label])=>[`${label} decision`,...ordered.map(m=>fmt(m.decision_s[key]))]),
      ...[['mean','Mean'],['median','Median'],['p90','90th percentile']].map(([key,label])=>[`${label} full request`,...ordered.map(m=>fmt(m.request_s[key]))]),
    ]);
    find('timing-note').textContent=`Both use the same ${raw.planning_budget_s}-second decision deadline. Full request time includes worker initialization, protocol and shutdown overhead. Times cover all 50 requests per planner, including setup refusals; the nearest-rank 90th percentile is the 45th of 50 ordered times.`;
    find('gpu').textContent=`GPU queue wait: ${(raw.gpu_wait_s/60).toFixed(1)} min, separate from planning. Native execution lease: ${(raw.gpu_lease_s/60).toFixed(1)} min. ${raw.native_releases} native releases including all 18 passed controls; ${raw.checks_count.toLocaleString()} evidence checks.`;
    find('conclusion').textContent=raw.conclusion;
    find('protocol').textContent=raw.protocol;
    table('positions',['Bucket X, Y (mm)','Current outcome','New outcome','Decision seconds: current / new','Full request seconds: current / new'],raw.results.map(r=>[r.bucket_xy_m.map(n=>(n*1000).toFixed(1)).join(', '),r.current.outcome,r.approach.outcome,`${fmt(r.current.decision_s)} / ${fmt(r.approach.decision_s)}`,`${fmt(r.current.request_s)} / ${fmt(r.approach.request_s)}`]));
    find('proof').textContent=`Report ${raw.provenance.report_sha256}. Independent audit ${raw.provenance.audit_sha256}. Frozen contract ${raw.provenance.contract_sha256}. Evidence manifest ${raw.provenance.manifest_sha256}.`;
  }
  return {approachComparisonReady:ready,renderApproachComparison:render};
})();
