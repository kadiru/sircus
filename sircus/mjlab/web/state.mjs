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
export function plannerReady(report) {
  return report?.planner?.ready === true && report.planner.version === 'sircus-corrected-sweep-planner-v1';
}
export function plannerOnly(report) {
  return report?.planner_only === true;
}
export function searchReady(report) {
  return report && Object.hasOwn(report, 'planner') ? plannerReady(report) : studyReady(report);
}
export function modeReady(report, mode = 'policy') {
  return mode === 'search' ? searchReady(report) : mode === 'policy' && studyReady(report);
}
export function finiteVector(values, length) {
  return Array.isArray(values) && values.length === length && values.every(x => typeof x === 'number' && Number.isFinite(x));
}
export function friendlyError(value) {
  const lines = String(value || 'Simulation request failed').split('\n').map(x => x.trim()).filter(Boolean);
  return (lines.at(-1) || 'Simulation request failed').replace(/^(RuntimeError|ValueError|OSError):\s*/, '');
}
export function buildRequest(mode, bucket, boardLock, board, dropLock, drop, requestId, searchBudget) {
  if (!['policy', 'search'].includes(mode)) throw Error('Unknown inference mode');
  if (!finiteVector(bucket, 2)) throw Error('Enter a valid bucket position');
  if (boardLock && !finiteVector(board, 3)) throw Error('Enter a valid board center');
  if (dropLock && !finiteVector(drop, 3)) throw Error('Enter a valid release center');
  if (searchBudget !== undefined && (mode !== 'search' || !Number.isInteger(searchBudget) || searchBudget < 16 || searchBudget > 128)) throw Error('Search budget must be an integer from 16 to 128');
  const request = { mode, bucket_xy: bucket, constraints: {
    board_position: boardLock ? board : null,
    dropper_position: dropLock ? drop : null,
  }, request_id: requestId };
  if (searchBudget !== undefined) request.search_budget = searchBudget;
  return request;
}
export function methodLabel(method) {
  if (['bank_policy', 'relative_bank_policy', 'bank_selected'].includes(method)) return 'Frozen relative-bank policy · geometry prior';
  if (['frozen_policy', 'ppo', 'scene_conditioned_ppo', 'policy'].includes(method)) return 'Frozen learned policy';
  if (['search', 'refinement', 'search_refinement'].includes(method)) return 'Per-scene search · not policy inference';
  if (['sweep_planner', 'corrected_sweep'].includes(method)) return 'Sweep planner · corrected native MuJoCo · endpoint-held';
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
export function solveButtonState(report, pending, mode = 'policy') {
  return { disabled: pending || !modeReady(report, mode),
    mainLabel: pending ? 'Solving and simulating… ' : 'Find drop-bounce posture ',
    topLabel: pending ? 'Solving…' : 'Find posture ↗' };
}

export function plannerFarmRows(report) {
  const rows = report?.planner?.farm_cells;
  if (!Array.isArray(rows)) return [];
  return rows.map((row, i) => ({
    id: String(row.id ?? `farm-${i + 1}`),
    xy: row.bucket_xy_m,
    status: String(row.status ?? 'unresolved'),
    cached: row.cache_available === true || row.cacheavailable === true,
    index: i,
  })).filter(row => finiteVector(row.xy, 2));
}

function observedCount(value) {
  return Number.isInteger(value) && value >= 0 ? value : null;
}
function observedRepeats(value) {
  const captures = observedCount(value?.captures), attempts = observedCount(value?.attempts);
  return captures !== null && attempts !== null && captures <= attempts ? { captures, attempts } : null;
}
function observedQuality(value) {
  return { clean: observedCount(value?.clean), assisted: observedCount(value?.assisted),
    cohort: ['nominal', 'fresh'].includes(value?.cohort) ? value.cohort : null,
    source: ['fresh_study', 'saved_reference', 'frozen_validation'].includes(value?.source) ? value.source : null };
}
function observedFrozen(value) {
  if (!value || typeof value !== 'object') return null;
  return { status: typeof value.status === 'string' ? value.status : 'unknown',
    nominal: observedRepeats(value.nominal), fresh: observedRepeats(value.fresh), quality: observedQuality(value.quality),
    configurationSha: typeof value.configuration_sha256 === 'string' && /^[a-f0-9]{64}$/.test(value.configuration_sha256) ? value.configuration_sha256 : null };
}
export function plannerGridResultPath(value) {
  return typeof value === 'string' && /^\/api\/planner-grid\/result\/[a-zA-Z0-9_-]+$/.test(value) ? value : null;
}
/** New study repeats remain separate from any saved-reference confirmation. */
export function plannerGridRows(report) {
  const rows = report?.planner_grid_study?.cells;
  if (!Array.isArray(rows)) return [];
  const inherited = report.planner_grid_study.protocol?.inherited_baseline === true;
  const newSelectedOnly = report.planner_grid_study.protocol?.frozen_scope === 'new_selected_only';
  const parentRows = report.planner_grid_parent?.status === 'completed' && Array.isArray(report.planner_grid_parent.cells) ? report.planner_grid_parent.cells : [];
  const baselineComplete = rows.length > 0 && rows.every(row => Array.isArray(row.attempts) && row.attempts.some(attempt => attempt.stage === 'baseline'));
  return rows.map((row, index) => {
    const selected = row.selected && typeof row.selected === 'object' ? row.selected : null;
    const frozen = newSelectedOnly && selected?.source === 'saved_reference' ? null : observedFrozen(row.frozen_validation);
    const configurationSha = typeof selected?.configuration_sha256 === 'string' && /^[a-f0-9]{64}$/.test(selected.configuration_sha256) ? selected.configuration_sha256 : null;
    const parent = parentRows.find(old => old.id === row.id && finiteVector(old.bucket_xy_m, 2) && finiteVector(row.bucket_xy_m, 2)
      && old.bucket_xy_m.every((value, i) => value === row.bucket_xy_m[i]));
    const previousFrozen = selected?.source === 'saved_reference' && configurationSha !== null && parent?.selected?.configuration_sha256 === configurationSha
      && parent?.frozen_validation?.configuration_sha256 === configurationSha ? observedFrozen(parent.frozen_validation) : null;
    const blocker = report?.planner_grid_blockers?.[row.id];
    const stage = typeof row.stage === 'string' ? row.stage : '';
    const displayStage = row.status === 'searching'
      ? `${inherited ? 'diversified' : baselineComplete ? 'expanded' : 'baseline'} search running${stage ? ` · last completed ${stage.replaceAll('_', ' ')}` : ''}`
      : stage.replaceAll('_', ' ');
    return {
      id: String(row.id ?? `table-${index + 1}`), index,
      label: `${Math.floor(index / 7) + 1}·${index % 7 + 1}`,
      xy: finiteVector(row.bucket_xy_m, 2) ? row.bucket_xy_m.slice() : null,
      status: typeof row.status === 'string' ? row.status : 'unknown',
      stage, displayStage,
      reason: typeof blocker === 'string' ? blocker : typeof row.reason === 'string' ? row.reason : '',
      selected: selected ? {
        board: finiteVector(selected.board_center_m, 3) ? selected.board_center_m.slice() : null,
        release: finiteVector(selected.release_ball_center_m, 3) ? selected.release_ball_center_m.slice() : null,
        tilt: Number.isFinite(selected.tilt_deg) ? selected.tilt_deg : null,
        gap: Number.isFinite(selected.gap_m) ? selected.gap_m : null,
        distance: Number.isFinite(selected.distance_m) ? selected.distance_m : null,
        configurationSha,
        source: selected.source === 'fresh_study' ? 'Fresh study' : selected.source === 'saved_reference' ? 'Saved reference' : 'Source unknown',
      } : null,
      nominal: observedRepeats(row.nominal), fresh: observedRepeats(row.fresh),
      savedNominal: observedRepeats(row.saved_nominal), savedFresh: observedRepeats(row.saved_fresh),
      quality: observedQuality(row.quality),
      frozen, previousFrozen, frozenExpected: !newSelectedOnly || selected?.source !== 'saved_reference',
      search: { nativeRequests: observedCount(row.search?.native_requests), rejected: observedCount(row.search?.rejected),
        cpuCandidates: observedCount(row.search?.cpu_candidates), captures: observedCount(row.search?.captures) },
      resultUrl: plannerGridResultPath(row.result_url ?? selected?.result_url),
    };
  });
}
export function plannerGridSummary(report) {
  const study = report?.planner_grid_study;
  if (!study || typeof study !== 'object' || !Array.isArray(study.cells)) return null;
  const rows = plannerGridRows(report), complete = new Set(['qualified', 'verified_fresh', 'saved_verified', 'unresolved', 'blocked', 'failed', 'completed']);
  const inherited = study.protocol?.inherited_baseline === true;
  const newSelectedOnly = study.protocol?.frozen_scope === 'new_selected_only';
  const targeted = Array.isArray(study.protocol?.targeted_cell_ids) ? new Set(study.protocol.targeted_cell_ids.filter(id => rows.some(row => row.id === id))) : null;
  const hasAttemptLedger = study.cells.some(row => Array.isArray(row.attempts));
  const baselines = study.cells.map(row => (Array.isArray(row.attempts) ? row.attempts : []).filter(attempt => attempt.stage === 'baseline').at(-1));
  const baselineDone = baselines.filter(Boolean).length;
  const phase = hasAttemptLedger ? {
    baselineDone: inherited && report.planner_grid_parent?.status === 'completed' && report.planner_grid_parent?.cells?.length === rows.length ? rows.length : baselineDone,
    ...(inherited ? { inherited: true } : {}),
    expandedDone: study.cells.filter(row => (!targeted || targeted.has(row.id)) && Array.isArray(row.attempts) && row.attempts.some(attempt => attempt.stage === 'expanded')).length,
    expandedPlanned: inherited && targeted ? targeted.size : baselineDone === rows.length ? baselines.filter(attempt => attempt?.success === false).length : null,
    frozenCompleted: rows.filter(row => row.frozenExpected && row.frozen?.status === 'completed').length,
    selectedTotal: rows.filter(row => newSelectedOnly ? row.selected?.source === 'Fresh study' : ['Fresh study', 'Saved reference'].includes(row.selected?.source)).length,
  } : null;
  const aggregate = get => {
    const available = rows.map(get).filter(value => value != null);
    if (!available.length) return null;
    return available.reduce((sum, value) => ({ captures: sum.captures + value.captures, attempts: sum.attempts + value.attempts, reported: available.length }), { captures: 0, attempts: 0, reported: available.length });
  };
  return {
    status: typeof study.status === 'string' ? study.status : 'unknown',
    phase,
    total: rows.length, finished: rows.filter(row => complete.has(row.status)).length,
    qualified: rows.filter(row => ['Fresh study', 'Saved reference'].includes(row.selected?.source)).length,
    freshQualified: rows.filter(row => row.selected?.source === 'Fresh study').length,
    savedQualified: rows.filter(row => row.selected?.source === 'Saved reference').length,
    unresolved: rows.filter(row => ['unresolved', 'blocked', 'failed'].includes(row.status)).length,
    nominal: aggregate(row => row.nominal), fresh: aggregate(row => row.fresh),
    frozenNominal: aggregate(row => row.frozenExpected ? row.frozen?.nominal : null), frozenFresh: aggregate(row => row.frozenExpected ? row.frozen?.fresh : null),
    updated: typeof study.updated_at === 'string' ? study.updated_at : null,
    scope: typeof study.scope === 'string' ? study.scope : '',
  };
}
export function plannerGridProgress(summary) {
  const phase = summary?.phase;
  if (!phase) return summary ? `${summary.finished}/${summary.total} cells finished` : 'Waiting for study progress';
  return `${phase.inherited ? 'Archived baseline' : 'Baseline'} ${phase.baselineDone}/${summary.total} · ${phase.inherited ? 'diversified search' : 'expanded'} ${phase.expandedDone}/${phase.expandedPlanned ?? 'pending'}${['validating', 'completed'].includes(summary.status) ? ` · Frozen repeats ${phase.frozenCompleted}/${phase.selectedTotal} selected` : ''}`;
}
export function plannerGridCsv(report) {
  const csvValue = value => {
    if (value === null || value === undefined) return '';
    const text = typeof value === 'string' && /^[=+\-@]/.test(value) ? `'${value}` : String(value);
    return `"${text.replaceAll('"', '""')}"`;
  };
  const header = ['table', 'id', 'bucket_x_m', 'bucket_y_m', 'status', 'stage', 'source', 'selected_configuration_sha256', 'board_x_m', 'board_y_m', 'board_center_z_m', 'board_tilt_deg', 'drop_gap_m', 'release_ball_center_x_m', 'release_ball_center_y_m', 'release_ball_center_z_m', 'board_to_bucket_distance_m', 'study_nominal_captures', 'study_nominal_attempts', 'study_fresh_captures', 'study_fresh_attempts', 'saved_nominal_captures', 'saved_nominal_attempts', 'saved_fresh_captures', 'saved_fresh_attempts', 'frozen_validation_status', 'frozen_nominal_captures', 'frozen_nominal_attempts', 'frozen_fresh_captures', 'frozen_fresh_attempts', 'frozen_configuration_sha256', 'frozen_quality_clean', 'frozen_quality_assisted', 'frozen_quality_cohort', 'previous_frozen_status', 'previous_frozen_nominal_captures', 'previous_frozen_nominal_attempts', 'previous_frozen_fresh_captures', 'previous_frozen_fresh_attempts', 'previous_frozen_configuration_sha256', 'quality_clean', 'quality_assisted', 'quality_cohort', 'quality_source', 'search_captures', 'search_native_requests', 'search_geometry_rejected', 'search_cpu_candidates', 'reason'];
  const rows = plannerGridRows(report).map(row => [row.label, row.id, ...(row.xy ?? [null, null]), row.status, row.stage, row.selected?.source, row.selected?.configurationSha,
    ...(row.selected?.board ?? [null, null, null]), row.selected?.tilt, row.selected?.gap, ...(row.selected?.release ?? [null, null, null]), row.selected?.distance,
    row.nominal?.captures, row.nominal?.attempts, row.fresh?.captures, row.fresh?.attempts,
    row.savedNominal?.captures, row.savedNominal?.attempts, row.savedFresh?.captures, row.savedFresh?.attempts,
    row.frozen?.status, row.frozen?.nominal?.captures, row.frozen?.nominal?.attempts, row.frozen?.fresh?.captures, row.frozen?.fresh?.attempts, row.frozen?.configurationSha,
    row.frozen?.quality?.clean, row.frozen?.quality?.assisted, row.frozen?.quality?.cohort,
    row.previousFrozen?.status, row.previousFrozen?.nominal?.captures, row.previousFrozen?.nominal?.attempts, row.previousFrozen?.fresh?.captures, row.previousFrozen?.fresh?.attempts, row.previousFrozen?.configurationSha,
    row.quality.clean, row.quality.assisted, row.quality.cohort, row.quality.source,
    row.search.captures, row.search.nativeRequests, row.search.rejected, row.search.cpuCandidates, row.reason]);
  return [header, ...rows].map(row => row.map(csvValue).join(',')).join('\n') + '\n';
}

// Read-only matched-budget benchmark. Missing observations never become zeros.
const benchNumber = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
const benchInteger = value => Number.isSafeInteger(value) && value >= 0 ? value : null;
const benchHash = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value) ? value : null;
export function sweepRlStudy(report) {
  const value = report?.sweep_rl_benchmark;
  return value && typeof value === 'object' && !Array.isArray(value) ? value : null;
}
function benchCounts(value) {
  if (!value || typeof value !== 'object') return null;
  const captures = benchInteger(value.captures), attempts = benchInteger(value.attempts);
  if (captures === null || attempts === null || captures > attempts) return null;
  const bounded = key => {
    const count = benchInteger(value[key]);
    return count !== null && count <= attempts ? count : null;
  };
  const clean = bounded('clean');
  return { captures, attempts, nativeReleases: bounded('native_releases'), invalid: bounded('invalid'),
    clean: clean !== null && clean <= captures ? clean : null,
    meanReward: typeof value.mean_reward === 'number' && Number.isFinite(value.mean_reward) ? value.mean_reward : null };
}
export function sweepRlResources(value) {
  const raw = value && typeof value === 'object' ? value : {};
  return Object.fromEntries(['wall_s', 'cpu_s', 'geometry_s', 'gpu_s', 'peak_rss_mb', 'peak_vram_mb',
    'torch_allocated_mb', 'torch_reserved_mb', 'setup_s', 'evaluation_wall_s', 'gpu_reservation_s'].map(key => [key, benchNumber(raw[key])]));
}
function benchMethods(study) {
  const values = study?.protocol?.methods;
  return Array.isArray(values) ? values.filter(row => row && typeof row.id === 'string' && typeof row.label === 'string') : [];
}
export function sweepRlPoints(report) {
  const study = sweepRlStudy(report), methods = benchMethods(study), labels = new Map(methods.map(row => [row.id, row.label]));
  if (!Array.isArray(study?.curves)) return [];
  const points = new Map(), conflicts = new Set();
  for (const raw of study.curves) {
    if (!raw || !labels.has(raw.method) || benchInteger(raw.seed) === null || benchInteger(raw.samples) === null) continue;
    const counts = benchCounts(raw.validation);
    if (!counts) continue;
    const row = { method: raw.method, label: labels.get(raw.method), seed: raw.seed, samples: raw.samples,
      split: 'validation', counts, resources: sweepRlResources(raw.resources), artifactSha: benchHash(raw.artifact_sha256) };
    const key = JSON.stringify([row.method, row.seed, row.samples]);
    if (conflicts.has(key)) continue;
    if (points.has(key) && JSON.stringify(points.get(key)) !== JSON.stringify(row)) { points.delete(key); conflicts.add(key); }
    else points.set(key, row);
  }
  return [...points.values()].sort((a, b) => a.samples - b.samples || a.method.localeCompare(b.method) || a.seed - b.seed);
}
export function sweepRlCurves(report) {
  const methods = benchMethods(sweepRlStudy(report)), points = sweepRlPoints(report);
  return methods.map(method => {
    const byBudget = new Map();
    for (const row of points.filter(row => row.method === method.id)) {
      const rows = byBudget.get(row.samples) ?? []; rows.push(row); byBudget.set(row.samples, rows);
    }
    const values = [...byBudget.entries()].sort((a, b) => a[0] - b[0]).map(([samples, rows]) => {
      const captures = rows.reduce((sum, row) => sum + row.counts.captures, 0), attempts = rows.reduce((sum, row) => sum + row.counts.attempts, 0);
      const rates = rows.filter(row => row.counts.attempts > 0).map(row => row.counts.captures / row.counts.attempts);
      const mean = key => rows.every(row => row.resources[key] !== null) ? rows.reduce((sum, row) => sum + row.resources[key], 0) / rows.length : null;
      const maximum = key => rows.every(row => row.resources[key] !== null) ? Math.max(...rows.map(row => row.resources[key])) : null;
      const count = key => rows.every(row => row.counts[key] !== null) ? rows.reduce((sum, row) => sum + row.counts[key], 0) : null;
      return { samples, captures, attempts, rate: attempts ? captures / attempts : null, seeds: rows.length,
        minRate: rates.length ? Math.min(...rates) : null, maxRate: rates.length ? Math.max(...rates) : null,
        nativeReleases: count('nativeReleases'), invalid: count('invalid'),
        wallS: mean('wall_s'), cpuS: mean('cpu_s'), geometryS: mean('geometry_s'), gpuS: mean('gpu_s'), gpuReservationS: mean('gpu_reservation_s'),
        peakRssMb: maximum('peak_rss_mb'), peakVramMb: maximum('peak_vram_mb'),
        torchAllocatedMb: maximum('torch_allocated_mb'), torchReservedMb: maximum('torch_reserved_mb') };
    });
    return { id: method.id, label: method.label, points: values };
  });
}
export function sweepRlFinalRows(report) {
  const study = sweepRlStudy(report), labels = new Map(benchMethods(study).map(row => [row.id, row.label]));
  if (study?.status !== 'completed' || !Array.isArray(study.final_test)) return [];
  return study.final_test.flatMap(raw => {
    const counts = benchCounts(raw?.counts), artifactSha = benchHash(raw?.frozen_artifact_sha256);
    if (!raw || !labels.has(raw.method) || benchInteger(raw.seed) === null || !counts || !artifactSha) return [];
    const matching = (Array.isArray(study.runs) ? study.runs : []).filter(run => run.method === raw.method && run.seed === raw.seed
      && run.status === 'completed' && benchHash(run.checkpoint_sha256) === artifactSha);
    const unique = key => { const values = new Set(matching.map(run => benchInteger(run[key])).filter(value => value !== null)); return values.size === 1 ? [...values][0] : null; };
    return [{ method: raw.method, label: labels.get(raw.method), seed: raw.seed, counts, artifactSha,
      selectedSamples: unique('selected_samples'), trainingRequests: unique('train_requests'), queryS: benchNumber(raw.query_s) }];
  });
}
export function sweepRlProgress(report) {
  const study = sweepRlStudy(report);
  if (!study) return null;
  const methods = new Map(benchMethods(study).map(method => [method.id, method.label]));
  const seeds = new Set((Array.isArray(study.protocol?.seeds) ? study.protocol.seeds : []).filter(seed => benchInteger(seed) !== null));
  const budget = benchInteger(study.protocol?.sample_budget), declared = methods.size * seeds.size;
  const declaredRuns = declared > 0 ? declared : null;
  const declaredRequests = declaredRuns !== null && budget !== null && budget > 0 && Number.isSafeInteger(declaredRuns * budget) ? declaredRuns * budget : null;
  const workers = new Map(), conflicts = new Set();
  for (const raw of Array.isArray(study.runs) ? study.runs : []) {
    if (!raw || !methods.has(raw.method) || !seeds.has(raw.seed)) continue;
    const key = JSON.stringify([raw.method, raw.seed]);
    if (conflicts.has(key)) continue;
    const requested = benchInteger(raw.train_requests), native = benchInteger(raw.train_native_releases);
    const requests = budget !== null && budget > 0 && requested !== null && requested <= budget ? requested : null;
    const row = { method: raw.method, label: methods.get(raw.method), seed: raw.seed,
      status: typeof raw.status === 'string' ? raw.status : 'unknown', requests,
      nativeReleases: requests !== null && native !== null && native <= requests ? native : null };
    if (workers.has(key) && JSON.stringify(workers.get(key)) !== JSON.stringify(row)) { workers.delete(key); conflicts.add(key); }
    else workers.set(key, row);
  }
  const rows = [...workers.values()], requestRows = rows.filter(row => row.requests !== null), nativeRows = rows.filter(row => row.nativeReleases !== null);
  return { declaredRuns, declaredRequests, budget,
    requests: requestRows.length ? requestRows.reduce((total, row) => total + row.requests, 0) : null,
    nativeReleases: nativeRows.length ? nativeRows.reduce((total, row) => total + row.nativeReleases, 0) : null,
    requestCounterRuns: requestRows.length, nativeCounterRuns: nativeRows.length,
    conflictingRuns: conflicts.size, reportedRunning: rows.filter(row => row.status === 'running') };
}
export function sweepRlCompletedWorkers(report) {
  const study = sweepRlStudy(report), methods = new Map(benchMethods(study).map(method => [method.id, method.label]));
  const seeds = new Set((Array.isArray(study?.protocol?.seeds) ? study.protocol.seeds : []).filter(seed => benchInteger(seed) !== null));
  const budget = benchInteger(study?.protocol?.sample_budget);
  if (budget === null || budget === 0) return [];
  const workers = new Map(), conflicts = new Set();
  for (const raw of Array.isArray(study?.runs) ? study.runs : []) {
    if (!raw || !methods.has(raw.method) || !seeds.has(raw.seed)) continue;
    const key = JSON.stringify([raw.method, raw.seed]);
    if (conflicts.has(key)) continue;
    const requested = benchInteger(raw.train_requests), native = benchInteger(raw.train_native_releases);
    const requests = requested !== null && requested <= budget ? requested : null;
    const row = { method: raw.method, label: methods.get(raw.method), seed: raw.seed, status: raw.status,
      requests, nativeReleases: requests !== null && native !== null && native <= requests ? native : null,
      resources: sweepRlResources(raw.resources) };
    if (workers.has(key) && JSON.stringify(workers.get(key)) !== JSON.stringify(row)) { workers.delete(key); conflicts.add(key); }
    else workers.set(key, row);
  }
  return [...workers.values()].filter(row => row.status === 'completed').sort((a, b) => a.method.localeCompare(b.method) || a.seed - b.seed);
}
export function sweepRlSummary(report) {
  const study = sweepRlStudy(report);
  if (!study) return null;
  const protocol = study.protocol ?? {}, seeds = Array.isArray(protocol.seeds) ? protocol.seeds.filter(v => benchInteger(v) !== null) : [];
  const runs = Array.isArray(study.runs) ? study.runs.filter(row => row && typeof row.method === 'string') : [];
  const prior = protocol.prior && typeof protocol.prior === 'object' ? protocol.prior : {};
  return { status: typeof study.status === 'string' ? study.status : 'unknown', phase: typeof study.phase === 'string' ? study.phase : '',
    seedCount: seeds.length, seeds, methodCount: benchMethods(study).length, budget: benchInteger(protocol.sample_budget),
    milestones: Array.isArray(protocol.milestones) ? protocol.milestones.filter(v => benchInteger(v) !== null) : [],
    completedRuns: runs.filter(row => row.status === 'completed').length, totalRuns: seeds.length * benchMethods(study).length,
    curvePoints: sweepRlPoints(report).length, finalRows: sweepRlFinalRows(report).length,
    scope: typeof protocol.scope === 'string' ? protocol.scope : typeof study.scope === 'string' ? study.scope : '',
    priorRecipes: benchInteger(prior.recipes), priorReleases: benchInteger(prior.latest_study_native_releases),
    priorFullCostKnown: prior.full_historical_cost_known === true,
    priorScope: typeof prior.scope === 'string' ? prior.scope : '',
    fingerprint: benchHash(study.contract_fingerprint), auditStatus: typeof study.audit?.status === 'string' ? study.audit.status : null };
}
export function sweepRlStatusLabel(summary) {
  if (!summary) return 'Benchmark measurements unavailable';
  if (summary.status === 'queued' && summary.curvePoints === 0) return 'Awaiting first completed measurements';
  return `${String(summary.status).replaceAll('_', ' ')}${summary.phase ? ` · ${summary.phase.replaceAll('_', ' ')}` : ''}`;
}
export function sweepRlCsv(report) {
  const study = sweepRlStudy(report);
  const escape = value => {
    if (value === null || value === undefined) return '';
    const text = typeof value === 'string' && /^[=+\-@]/.test(value) ? `'${value}` : String(value);
    return `"${text.replaceAll('"', '""')}"`;
  };
  const columns = ['method', 'seed', 'split', 'training_requests', 'selected_training_requests', 'captures', 'attempts', 'native_releases', 'invalid', 'clean',
    'capture_rate', 'wall_s_excluding_evaluation', 'cpu_s', 'geometry_s', 'gpu_phase_s', 'peak_rss_mib', 'sampled_peak_vram_mib',
    'torch_allocated_mib', 'torch_reserved_mib', 'setup_s', 'evaluation_wall_s', 'gpu_reservation_s_including_evaluation', 'query_s', 'artifact_sha256', 'contract_fingerprint'];
  const values = sweepRlPoints(report).map(row => [row.method, row.seed, row.split, row.samples, null, row.counts.captures,
    row.counts.attempts, row.counts.nativeReleases, row.counts.invalid, row.counts.clean,
    row.counts.attempts ? row.counts.captures / row.counts.attempts : null,
    ...['wall_s', 'cpu_s', 'geometry_s', 'gpu_s', 'peak_rss_mb', 'peak_vram_mb', 'torch_allocated_mb', 'torch_reserved_mb', 'setup_s', 'evaluation_wall_s', 'gpu_reservation_s'].map(key => row.resources[key]),
    null, row.artifactSha, study?.contract_fingerprint]);
  for (const row of sweepRlFinalRows(report)) values.push([row.method, row.seed, 'test', row.trainingRequests, row.selectedSamples, row.counts.captures,
    row.counts.attempts, row.counts.nativeReleases, row.counts.invalid, row.counts.clean,
    row.counts.attempts ? row.counts.captures / row.counts.attempts : null,
    ...Array(11).fill(null), row.queryS, row.artifactSha, study?.contract_fingerprint]);
  return [columns, ...values].map(row => row.map(escape).join(',')).join('\n') + '\n';
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

/** Versioned stability diagnostics remain read-only and cannot activate a policy. */
export function stableIkProgress(report) {
  const value = report?.stable_ik_evaluation;
  if (!value || typeof value !== 'object') return null;
  const finite = x => typeof x === 'number' && Number.isFinite(x) ? x : null;
  const queue = value.waiting_for_4090 ?? value.gpu_queue;
  return {
    status: String(value.status ?? 'not_started').replaceAll('_', ' '),
    phase: String(value.phase ?? 'Waiting for diagnostic evidence'),
    nativeSearchAttempts: finite(value.native_search_attempts),
    confirmationAttempts: finite(value.confirmation_attempts),
    finalConfirmationAttempts: finite(value.final_confirmation_attempts),
    geometryProposals: finite(value.geometry_proposals),
    qualified: stableIkTargets(report).filter(row => row.promoted).length,
    targets: stableIkTargets(report).length,
    queued: queue?.status === 'waiting_for_4090',
    waitSeconds: finite(queue?.wait_s),
    invariance: value.solver_audit?.order_batch_invariance?.passed ?? null,
    preservation: value.solver_audit?.known_branch_preservation?.passed ?? null,
    noTraining: value.no_training === true,
  };
}
export function stableIkTargets(report) {
  const value = report?.stable_ik_evaluation;
  const rows = value?.cells?.length ? value.cells : value?.development_layouts;
  if (!Array.isArray(rows)) return [];
  const count = x => typeof x === 'number' && Number.isFinite(x) ? x : null;
  return rows.map(row => ({
    id: String(row.id ?? 'Unknown target'), xy: row.bucket_xy_m ?? row.bucket_xy,
    attempts: count(row.search?.attempts), captures: count(row.search?.captures), clean: count(row.search?.clean),
    assisted: count(row.observed_assisted_entries ?? row.search?.assisted),
    firstConfirmation: row.best_screening ?? row.promotion?.first_confirmation ?? row.promotion?.confirmation ?? row.first_confirmation ?? null,
    finalConfirmation: row.final_confirmation ?? row.promotion?.final_confirmation ?? null,
    confirmationSourceId: row.best_screening?.source_trial_id ?? row.final_confirmation?.source_trial_id ?? row.promotion?.source_trial_id ?? '',
    confirmationScope: row.confirmation_summary_scope ?? '',
    multipleFinalGroups: row.multiple_final_groups === true,
    promoted: !!row.promotion,
    sourceId: row.promotion?.source_trial_id ?? '',
    reason: row.unresolved_reason || 'No independently qualified setting yet',
  })).filter(row => finiteVector(row.xy, 2));
}
export function stableIkAuditRows(report) {
  const rows = report?.stable_ik_evaluation?.solver_audit?.comparison;
  if (!Array.isArray(rows)) return [];
  const finite = x => typeof x === 'number' && Number.isFinite(x) ? x : null;
  const labels = { old: 'Original · 4 starts / 90 steps', stable_matched: 'Stable · matched 4 / 90', stable_free: 'Stable · 8 starts / 120 steps', stable_seeded: 'Stable · saved branch seed' };
  return rows.map(row => ({
    label: labels[row.method] ?? String(row.method ?? 'Unidentified solver'),
    queries: finite(row.queries), accepted: finite(row.accepted),
    orderChanges: finite(row.order_mask_changes), batchChanges: finite(row.batch_mask_changes),
    maxDelta: finite(row.max_q_delta_rad), preserved: finite(row.preserved_known_branches),
  }));
}
export function stableIkControlRows(report) {
  const values = report?.stable_ik_evaluation?.control_statistics;
  if (!values || typeof values !== 'object') return [];
  const labels = { saved: 'Saved source joints', legacy_fresh: 'Original fresh IK',
    stable_matched: 'Stable · matched 4 / 90', stable_free: 'Stable · 8 / 120', stable_seeded: 'Stable · saved branch seed' };
  const count = x => typeof x === 'number' && Number.isFinite(x) ? x : null;
  return Object.entries(labels).filter(([key]) => values[key]).map(([key, label]) => ({
    label, attempts: count(values[key].attempts), captures: count(values[key].captures),
    clean: count(values[key].clean), invalid: count(values[key].invalid),
  }));
}

// Read-only contact/transit evidence. This module never builds a solve request.
const contactFinite = value => typeof value === 'number' && Number.isFinite(value);
const contactCount = value => Number.isInteger(value) && value >= 0 ? value : null;

export function contactStudy(report) {
  const study = report?.contact_repeatability_evaluation;
  return study && typeof study === 'object' && !Array.isArray(study) ? study : null;
}

export function contactGroups(report) {
  return (contactStudy(report)?.groups ?? []).filter(row => row && typeof row.id === 'string').map(row => ({
    ...row, attempts: contactCount(row.attempts), captures: contactCount(row.captures),
    clean: contactCount(row.clean), invalid: contactCount(row.invalid),
  }));
}

export function contactConfigurations(report) {
  return (contactStudy(report)?.configurations ?? []).filter(row => row &&
    typeof row.id === 'string' && finiteVector(row.board_xyz_m, 3) && finiteVector(row.bucket_xy_m, 2) &&
    contactFinite(row.tilt_deg) && contactFinite(row.drop_gap_m) && finiteVector(row.release_center_m, 3));
}

export function transitRoutes(report) {
  return (contactStudy(report)?.transit?.routes ?? []).filter(row => row && typeof row.id === 'string');
}

export function routePreview(route) {
  if (route?.accepted !== true || !Array.isArray(route.frames) || route.frames.length < 2) return [];
  const frames = route.frames;
  if (!frames.every(frame => frame && contactFinite(frame.time) && finiteVector(frame.board_center, 3) &&
      finiteVector(frame.board_quaternion_wxyz, 4) && Array.isArray(frame.link_positions10) &&
      frame.link_positions10.length >= 2 && frame.link_positions10.every(p => finiteVector(p, 3)) &&
      Array.isArray(frame.link_positions14) && frame.link_positions14.length >= 2 &&
      frame.link_positions14.every(p => finiteVector(p, 3)))) return [];
  if (frames.some((frame, i) => i && frame.time <= frames[i - 1].time)) return [];
  return frames;
}

export function configurationsCsv(report) {
  const cell = value => `"${String(value ?? '').replaceAll('"', '""')}"`;
  const fields = ['id', 'condition', 'board_x_mm', 'board_y_mm', 'board_z_mm', 'tilt_deg',
    'drop_gap_mm', 'release_ball_center_z_mm', 'bucket_x_mm', 'bucket_y_mm', 'horizontal_distance_mm',
    ...Array.from({length: 6}, (_, i) => `q10_${i + 1}_rad`), ...Array.from({length: 6}, (_, i) => `q14_${i + 1}_rad`)];
  const rows = contactConfigurations(report).map(row => [row.id, row.condition,
    ...row.board_xyz_m.map(x => x * 1000), row.tilt_deg, row.drop_gap_m * 1000,
    row.release_center_m[2] * 1000, ...row.bucket_xy_m.map(x => x * 1000),
    Math.hypot(row.board_xyz_m[0] - row.bucket_xy_m[0], row.board_xyz_m[1] - row.bucket_xy_m[1]) * 1000,
    ...(finiteVector(row.q10, 6) ? row.q10 : Array(6).fill('')), ...(finiteVector(row.q14, 6) ? row.q14 : Array(6).fill(''))]);
  return [fields, ...rows].map(row => row.map(cell).join(',')).join('\r\n') + '\r\n';
}

// Corrected-contact DEV grid: read-only and independent of inference readiness.
export function correctedGridProgress(report) {
  const value = report?.corrected_grid_evaluation;
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const counts = value.counts ?? {};
  return { status: value.status ?? 'pending', phase: value.phase ?? 'preparing',
    // The evidence retains its last queue observation after lock acquisition.
    queue: value.status === 'waiting' || value.phase === 'waiting_for_4090'
      ? value.waiting_for_4090 ?? value.queue ?? null : null,
    cells: correctedGridRows(report),
    geometryRequests: contactCount(counts.geometry_requests),
    nativeAttempts: contactCount(counts.native_attempts),
    nativeReleases: contactCount(counts.native_releases),
    confirmationAttempts: contactCount(counts.confirmation_attempts),
    qualified: contactCount(counts.qualified_positions),
    connected: contactCount(counts.common_home_connected),
    homeClear: contactCount(counts.home_clear_positions),
    settings: contactCount(counts.qualified_settings),
    multipleChoices: contactCount(counts.positions_with_multiple_qualified),
    audits: value.audits ?? {},
    stages: value.stage_statistics ?? null,
    verification: value.verification?.status ?? 'pending',
    scope: value.scope ?? 'Seeded development grid; no new policy training',
    reason: value.reason ?? '',
    fingerprint: value.contract_fingerprint ?? 'pending',
  };
}
export function correctedTrainingReadiness(report) {
  const progress = correctedGridProgress(report);
  const pending = {readyToLaunch:false,title:'Training decision pending',text:'Wait for the complete study and both contract-matched audits.'};
  if (!progress || progress.status !== 'completed') return pending;
  if (progress.verification !== 'passed' || !['native','supplemental'].every(name => progress.audits[name]?.status === 'passed'))
    return {...pending,title:'Study finished; audit clearance required'};
  if (!(progress.qualified > 0)) return {...pending,title:'More feasible settings needed',text:'No position passed the declared path and robustness checks.'};
  return {readyToLaunch:false,title:'Audited evidence ready for a limited pilot',
    text:`${progress.settings ?? 'Qualified'} settings at ${progress.qualified}/${progress.cells.length} development positions; ${progress.multipleChoices ?? 'unknown'} positions have multiple qualified choices. Next: deduplicate relative geometry prototypes, adapt and smoke-test the learner with corrected physics and shared-start admission, then compare against uniform feasible selection. Table-wide training and held-out generalization remain unvalidated.`};
}
export function correctedGridRows(report) {
  const value = report?.corrected_grid_evaluation;
  return (value?.cells ?? []).filter(row => row && typeof row.id === 'string' && finiteVector(row.bucket_xy_m,2)).map(row => {
    const groups = Array.isArray(row.confirmations) ? row.confirmations : [];
    const qualified = row.qualification?.qualified === true || groups.some(g => g.qualified === true);
    const eligible = contactCount(row.geometry?.endpoint_eligible ?? row.geometry?.eligible);
    const attempts = contactCount(row.search?.attempts), captures = contactCount(row.search?.captures);
    const motion = row.motion?.connected === true;
    const homeBlocked = row.home_geometry?.passed === false;
    const state = qualified ? 'qualified' : captures > 0 ? 'captured' : homeBlocked ? 'home_blocked' : attempts > 0 ? 'missed'
      : eligible > 0 ? 'endpoint' : eligible === 0 ? 'rejected' : 'pending';
    return { id:row.id,xy:row.bucket_xy_m,eligible,attempts,captures,qualified,motion,state,
      reason:row.reason ?? (homeBlocked ? row.home_geometry.category ?? row.home_geometry.reason : null) ?? row.status ?? 'Pending',confirmations:groups };
  });
}
