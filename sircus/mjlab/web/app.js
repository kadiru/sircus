import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';
import { contactStudy, contactGroups, contactConfigurations, transitRoutes, routePreview, configurationsCsv } from '/state.mjs';
import { correctedGridProgress, correctedTrainingReadiness } from '/state.mjs';
import { plannerReady, plannerOnly, searchReady, modeReady, plannerFarmRows, plannerGridRows, plannerGridSummary, plannerGridProgress, plannerGridCsv } from '/state.mjs';
import { sweepRlStudy, sweepRlSummary, sweepRlProgress, sweepRlCompletedWorkers, sweepRlPoints, sweepRlCurves, sweepRlFinalRows, sweepRlFinalAggregate, sweepRlStatusLabel, sweepRlCsv, sweepRlAdmissionRows, sweepRlHistory, historicalPlannerIsPreV2 } from '/state.mjs';
import { studyReady, buildRequest, methodLabel, coverageRows, finiteVector, friendlyError, coverageSummary, comparisonRows, learningPoints, inStudiedRegion, BOARD_BODY_SIZE, bankComparisonRows, policyRouteLabel, poseInputValues, bankCoverageRows, bankCoverageStatus, formatMetric, solveButtonState, expansionTargetRows, expansionProgress, expandedComparisonRows, stableIkProgress, stableIkTargets, stableIkAuditRows, stableIkControlRows } from '/state.mjs';

const $ = id => document.getElementById(id);
const canvas = $('world');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.setClearColor(0x121b20);
const world = new THREE.Scene();
world.fog = new THREE.Fog(0x121b20, 5, 11);
const camera = new THREE.PerspectiveCamera(38, 1, .01, 30);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * .48;
controls.minDistance = 1;
controls.maxDistance = 5;
function resetView() { camera.position.set(2.6, -2.8, 2.15); controls.target.set(.67, -.23, .35); controls.update(); }
resetView();
$('resetView').onclick = resetView;
world.add(new THREE.HemisphereLight(0xd1e9e5, 0x3f4549, 2));
const light = new THREE.DirectionalLight(0xffefdc, 3.5);
light.position.set(1.6, -1.3, 4); light.castShadow = true;
light.shadow.mapSize.set(2048, 2048); light.shadow.camera.left = -2;
light.shadow.camera.right = 2; light.shadow.camera.top = 2; light.shadow.camera.bottom = -2;
light.shadow.bias = -.0003;
world.add(light);
const material = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: .65, metalness: .12, ...extra });
function box(size, color, pos) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(...size), material(color));
  m.position.fromArray(pos); m.castShadow = m.receiveShadow = true; world.add(m); return m;
}
box([8, 8, .04], 0x0f161c, [.67, -.25, -.84]);
const table = box([1.8, 1.25, .05], 0x243b40, [.67, -.25, -.025]);
for (const x of [-.1, 1.44]) for (const y of [-.75, .25]) box([.045, .045, .75], 0x2a383d, [x, y, -.425]);
const axes = new THREE.AxesHelper(.22); axes.position.set(-.15, -.77, .005); world.add(axes);
const board = box(BOARD_BODY_SIZE, 0x57b4aa, [.65, 0, .4]);
board.rotation.z = Math.PI / 2; // Schematic landscape mount; replaced by native quaternion.
const boardEdge = new THREE.LineSegments(new THREE.EdgesGeometry(board.geometry), new THREE.LineBasicMaterial({ color: 0x92ded3 })); board.add(boardEdge);
const ball = new THREE.Mesh(new THREE.SphereGeometry(.0375, 24, 18), material(0xf889a7));
ball.castShadow = true; ball.position.set(.65, 0, .6875); world.add(ball);
const releaseMarker = new THREE.Mesh(new THREE.SphereGeometry(.013, 16, 12), material(0xffd6e2, { transparent: true, opacity: .55 }));
world.add(releaseMarker); releaseMarker.visible = false;
const bucketGroup = new THREE.Group(); world.add(bucketGroup);
function bucketWalls(insideTop, insideBottom, outsideTop, outsideBottom, depth, bottom) {
  const vertices = [], indices = [], n = 64;
  for (const [radius, z] of [[outsideBottom, 0], [outsideTop, depth + bottom], [insideTop, depth + bottom], [insideBottom, bottom]]) {
    for (let i = 0; i < n; i++) { const a = i * Math.PI * 2 / n; vertices.push(radius * Math.cos(a), radius * Math.sin(a), z); }
  }
  for (const [a, b] of [[0, 1], [1, 2], [2, 3]]) for (let i = 0; i < n; i++) {
    const j = (i + 1) % n, v = a * n + i, w = b * n + i;
    indices.push(v, a * n + j, w, a * n + j, b * n + j, w);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setIndex(indices); geometry.computeVertexNormals();
  return geometry;
}
const bucket = new THREE.Mesh(bucketWalls(.0885, .075, .1, .079, .175, .008), material(0xe6aa49, { side: THREE.DoubleSide }));
bucket.castShadow = bucket.receiveShadow = true; bucketGroup.add(bucket);
const bucketBase = new THREE.Mesh(new THREE.CylinderGeometry(.079, .079, .008, 64), material(0xd39a3e));
bucketBase.rotation.x = Math.PI / 2; bucketBase.position.z = .004; bucketGroup.add(bucketBase);
let sceneSpec = {}, report = {}, pending = false, activeRequest = null, currentTrajectory = [], playTime = 0, playing = false, pathLine;
let physics = null, physicsViewerResult, physicsProgress, savedPhysicsRecordings, clearanceExpansionSummary, neuralShotSummary, planningBudgetSummary, fastPlanningSummary, interactiveFastAcceptanceSummary, armCheckSpeedSummary, directPathFirstSummary, physicsSceneActive = false, physicsViewKey = '';
let savedReplayRows = [], savedReplaySignature = '';
const physicsSelected = () => $('mode').value === 'physics' && physics?.available === true;
let methodChosen = false, farmSignature = '', gridStudySignature = '', benchmarkSignature = '';
let transitFrames = [], transitTime = 0, transitName = "";
let bucketXY = [.65, -.4];
const armGroups = [new THREE.Group(), new THREE.Group()]; armGroups.forEach(g => world.add(g));
function armLinks(points, group, color) {
  group.children.forEach(c => { c.geometry?.dispose(); c.material?.dispose(); });
  group.clear();
  for (let i = 0; i < points.length - 1; i++) {
    const a = new THREE.Vector3(...points[i]), b = new THREE.Vector3(...points[i + 1]);
    const delta = b.clone().sub(a), length = delta.length();
    if (length < 1e-5) continue;
    const link = new THREE.Mesh(new THREE.CylinderGeometry(i === points.length - 2 ? .025 : .037, .043, length, 14), material(color));
    link.position.copy(a).add(b).multiplyScalar(.5);
    link.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize());
    link.castShadow = true; group.add(link);
    const joint = new THREE.Mesh(new THREE.SphereGeometry(.046, 16, 12), material(0x708b8b));
    joint.position.copy(a); joint.castShadow = true; group.add(joint);
  }
}
function schematic(base, end) {
  return [base, [base[0], base[1], .26], [base[0] * .7 + end[0] * .3, base[1] * .7 + end[1] * .3, Math.max(.65, end[2] + .28)], [end[0], end[1], end[2] + .14], end];
}
function drawArms(config = {}) {
  const b = config.board_center ?? board.position.toArray();
  const d = config.release_center ?? config.dropper_center ?? ball.position.toArray();
  const native = finiteVector(config.link_positions10?.[0], 3) && finiteVector(config.link_positions14?.[0], 3);
  armLinks(native ? config.link_positions10 : schematic([0, 0, 0], d), armGroups[0], 0xb6c9c7);
  armLinks(native ? config.link_positions14 : schematic([1.33731, 0, 0], b), armGroups[1], 0xd09249);
  $('geometryLabel').textContent = native ? 'Native kinematic link positions · simplified link surfaces' : 'Endpoint schematic · actual dynamics reported separately';
}
function setBucket(xy, updateInputs = true) {
  bucketXY = xy; bucketGroup.position.set(xy[0], xy[1], 0);
  $('regionWarning').hidden = inStudiedRegion(xy);
  if (updateInputs) { $('bucketX').value = (xy[0] * 1000).toFixed(0); $('bucketY').value = (xy[1] * 1000).toFixed(0); }
}
setBucket(bucketXY); drawArms();
function clearResult(message, preservePhysics = false) {
  if (!preservePhysics && physics?.available) physics.invalidate('Scene changed. Plan a new shot.');
  if (physicsSceneActive) {
    board.position.fromArray(sceneSpec.board?.center || [.65, 0, .4]);
    board.rotation.set(0, 0, Math.PI / 2);
    ball.position.fromArray(sceneSpec.release_center || [.65, 0, .6875]);
    drawArms(); physicsSceneActive = false; physicsViewKey = '';
  }
  $('sceneOutcome').textContent = message; $('sceneOutcome').className = 'scene-outcome';
  transitFrames = []; ball.visible = true;
  playing = false; currentTrajectory = []; $('play').disabled = true; $('timeline').disabled = true;
  if (pathLine) { world.remove(pathLine); pathLine.geometry.dispose(); pathLine.material.dispose(); pathLine = null; }
  $('resultTitle').textContent = message;
  $('resultText').textContent = 'A new solve is required after changing the scene or arm constraints.';
  $('resultDot').className = 'status-dot'; $('metrics').replaceChildren();
  $('savedReplayMarkers').replaceChildren();
}
const raycaster = new THREE.Raycaster(), mouse = new THREE.Vector2(), dragPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), dragPoint = new THREE.Vector3();
let dragging = false;
function pointerRay(event) {
  const rect = canvas.getBoundingClientRect();
  mouse.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
  raycaster.setFromCamera(mouse, camera);
}
canvas.addEventListener('pointerdown', event => {
  if (pending || physics?.busy) return;
  pointerRay(event);
  if (raycaster.intersectObjects(bucketGroup.children).length) {
    dragging = true; controls.enabled = false; canvas.setPointerCapture(event.pointerId); event.preventDefault();
  }
});
canvas.addEventListener('pointermove', event => {
  if (!dragging || pending || physics?.busy) return;
  pointerRay(event);
  if (raycaster.ray.intersectPlane(dragPlane, dragPoint)) {
    const bounds = (physicsSelected() && physics.capabilities.bucket_bounds) || sceneSpec.bucket_bounds || { x: [-.13, 1.47], y: [-.775, .275] };
    setBucket([THREE.MathUtils.clamp(dragPoint.x, ...bounds.x), THREE.MathUtils.clamp(dragPoint.y, ...bounds.y)]);
    clearResult('Bucket moved — ready for a new solve');
  }
});
for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(name, () => { dragging = false; controls.enabled = true; });
function number(id) { const value = $(id).value.trim(); return value === '' ? NaN : Number(value) / 1000; }
function vector(prefix) { return ['X', 'Y', 'Z'].map(axis => number(prefix + axis)); }
function updateConstraintPreview() {
  updateMethodHelp();
  for (const id of ['boardLock', 'dropLock']) $(id).disabled = pending || physics?.busy === true || physicsSelected();
  $('boardFields').classList.toggle('inactive', !$('boardLock').checked);
  $('dropFields').classList.toggle('inactive', !$('dropLock').checked);
  for (const prefix of ['board', 'drop']) for (const axis of ['X', 'Y', 'Z']) $(prefix + axis).disabled = pending || physics?.busy === true || physicsSelected() || !$(prefix + 'Lock').checked;
  if (!physicsSelected() && $('boardLock').checked && finiteVector(vector('board'), 3)) board.position.fromArray(vector('board'));
  releaseMarker.visible = !physicsSelected() && $('dropLock').checked;
  if (releaseMarker.visible && finiteVector(vector('drop'), 3)) releaseMarker.position.fromArray(vector('drop'));
}
for (const id of ['bucketX', 'bucketY']) $(id).addEventListener('input', () => { if (physics?.available) { const xy = [number('bucketX'), number('bucketY')]; if (finiteVector(xy, 2)) setBucket(xy, false); clearResult('Bucket edited — plan again'); } });
for (const id of ['boardLock', 'dropLock', 'boardX', 'boardY', 'boardZ', 'dropX', 'dropY', 'dropZ']) $(id).addEventListener('input', () => { if (physics?.available) clearResult('Scene edited — plan again'); });
for (const id of ['bucketX', 'bucketY']) $(id).addEventListener('change', () => { if (Number.isFinite(number('bucketX')) && Number.isFinite(number('bucketY'))) setBucket([number('bucketX'), number('bucketY')], false); clearResult('Bucket changed — ready for a new solve'); });
for (const id of ['boardLock', 'dropLock', 'boardX', 'boardY', 'boardZ', 'dropX', 'dropY', 'dropZ']) $(id).addEventListener('change', () => { updateConstraintPreview(); clearResult('Arm constraints changed'); });
updateConstraintPreview();
function updateMethodHelp() {
  const search = $('mode').value === 'search', planner = plannerReady(report);
  $('physicsControls').hidden = !physicsSelected();
  $('physicsPlannerControls').hidden = !physicsSelected();
  $('positionConstraints').hidden = physicsSelected();
  $('methodHelp').textContent = physicsSelected() ? 'Plan both arm endpoints first, then explicitly run one native physics drop. Position locks belong to the historical modes.' : !search ? policyRouteLabel(report, $('boardLock').checked, $('dropLock').checked) : planner
    ? 'Sweep planner · corrected native MuJoCo · endpoint-held. Exact saved targets can replay recorded evidence; new targets require bounded search and fresh native verification. No solution within budget does not mean impossible.'
    : 'Explicit per-scene search. This is measured separately from frozen-policy performance.';
  $('searchBudgetFields').hidden = !(search && planner);
  $('policyOption').disabled = !studyReady(report);
  $('searchOption').disabled = !searchReady(report);
}
$('mode').onchange = () => { methodChosen = true; if (physics?.available) clearResult('Method changed — ready for a new request'); updateConstraintPreview(); syncSolveButtons(); };
function syncSolveButtons() {
  const state = physicsSelected() ? { disabled: pending || physics.busy, mainLabel: physics.busy ? 'Physics request in progress… ' : 'Plan shot ', topLabel: physics.busy ? 'Physics request in progress…' : 'Plan shot ↗' } : solveButtonState(report, pending || physics?.busy === true, $('mode').value);
  for (const id of ['solve', 'solveTop']) $(id).disabled = state.disabled;
  $('solve').firstChild.textContent = state.mainLabel;
  $('solveTop').textContent = state.topLabel;
  const dropDisabled = pending || !physicsSelected() || !physics.canDrop;
  for (const id of ['physicsDrop', 'physicsDropTop']) $(id).disabled = dropDisabled;
  $('physicsDropTop').hidden = dropDisabled;
  $('physicsClear').disabled = pending || physics?.busy === true || !physicsSelected() || ['idle', 'stale', 'unavailable'].includes(physics.snapshot.state);
  const blocked = pending || physics?.busy === true;
  for (const id of ['bucketX', 'bucketY', 'mode', 'searchBudget']) $(id).disabled = blocked;
  $('physicsPlanner').disabled = blocked || !physicsSelected() || physics.plannerOptions.choices.length < 2;
  for (const button of $('plannerFarmGrid').children) button.disabled = blocked;
  for (const button of $('plannerGridTable').querySelectorAll('button')) button.disabled = blocked;
  $('savedReplaySelect').disabled = pending || physics?.busy === true;
  $('savedReplayLoad').disabled = pending || physics?.busy === true || !savedReplayRows.length;
  for (const button of $('savedReplayMarkers').children) button.disabled = pending || physics?.busy === true;
}
function setPending(value) {
  pending = value; syncSolveButtons();
  for (const id of ['bucketX', 'bucketY', 'boardLock', 'dropLock', 'mode', 'searchBudget']) $(id).disabled = value || physics?.busy === true;
  for (const button of $('plannerFarmGrid').children) button.disabled = value || physics?.busy === true;
  for (const button of $('plannerGridTable').querySelectorAll('button')) button.disabled = value || physics?.busy === true;
  updateConstraintPreview();
}
function addMetrics(pairs, target) {
  const dl = document.createElement('dl');
  for (const [key, value] of pairs) { const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = key; dd.textContent = String(value); dl.append(dt, dd); }
  target.replaceChildren(dl);
}
function applyResult(result, { recordedPlayback = false, physicsPreview = false, physicsResult = false } = {}) {
  const config = result.configuration || {}, b = config.board_center, n = config.board_normal;
  if (finiteVector(b, 3)) board.position.fromArray(b);
  if (finiteVector(config.board_quaternion_wxyz, 4)) {
    const [w, x, y, z] = config.board_quaternion_wxyz; board.quaternion.set(x, y, z, w).normalize();
  } else if (finiteVector(n, 3)) {
    board.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(...n).normalize())
      .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2));
  }
  const release = config.release_center ?? config.dropper_center;
  if (finiteVector(release, 3)) ball.position.fromArray(release);
  for (const [prefix, position] of [['board', b], ['drop', release]]) {
    const values = poseInputValues(position);
    if (!$(prefix + 'Lock').checked && values) ['X', 'Y', 'Z'].forEach((axis, i) => { $(prefix + axis).value = values[i]; });
  }
  drawArms(config);
  currentTrajectory = (Array.isArray(result.trajectory) ? result.trajectory : []).filter(p => Number.isFinite(p.time) && finiteVector(p.ball, 3));
  if (currentTrajectory.length > 1) {
    const geometry = new THREE.BufferGeometry().setFromPoints(currentTrajectory.map(p => new THREE.Vector3(...p.ball)));
    pathLine = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0xf8a7b9, transparent: true, opacity: .48 })); world.add(pathLine);
    $('play').disabled = false; $('timeline').disabled = false; playTime = currentTrajectory[0].time; playing = true;
  }
  const success = result.success === true;
  const recorded = recordedPlayback || ['catalogue', 'catalog_lookup', 'cached', 'dataset_lookup'].includes(result.method) || result.provenance?.cache_hit === true;
  const busy = result.status_native === 'busy';
  $('sceneOutcome').textContent = success ? recorded ? '✓ Saved verified shot · recorded native evidence' : '✓ Verified capture · native simulation' : busy ? '4090 busy · no new shot executed' : 'No verified solution within budget · see result';
  $('sceneOutcome').className = `scene-outcome ${success ? 'ready' : 'failed'}`;
  $('resultDot').className = `status-dot ${success ? 'ready' : 'failed'}`;
  $('resultTitle').textContent = success ? recorded ? 'Recorded verified shot at this exact target' : 'Verified capture in simulation' : busy ? 'Simulation GPU is busy' : 'No verified shot for this request';
  const provenance = methodLabel(result.method);
  $('resultText').textContent = `${provenance}. ${result.message ?? result.reason ?? 'Board bounce, whole-ball entry and one continuous second retained are required.'}`;
  const metrics = result.metrics || {}, pairs = [['Method', provenance], ['Request', result.request_id?.slice(0, 10) ?? '—']];
  if (finiteVector(b, 3)) pairs.push(['Board center Z', `${(b[2] * 1000).toFixed(0)} mm`]);
  if (finiteVector(release, 3)) pairs.push(['Release center Z', `${(release[2] * 1000).toFixed(0)} mm`]);
  for (const [key, label] of [['tilt_deg', 'Board tilt'], ['board_bounces', 'Board bounces'], ['retention_s', 'Retained'], ['quality', 'Entry quality'], ['attempts', 'Native attempts'], ['native_candidate_count', 'Native candidate count'], ['eligible_candidates', 'Eligible candidates']]) if (metrics[key] !== undefined) pairs.push([label, formatMetric(key, metrics[key])]);
  if (physicsPreview || physicsResult) {
    const outcome = ({ direct: 'Direct capture', table: 'Table-assisted capture', unclassified_capture: 'Capture · route unclassified', arm_assisted_capture: 'Robot-assisted capture', flight_miss: 'Miss', invalid_physics: 'Invalid physics' })[result.outcome] || 'Outcome unknown';
    const label = physicsPreview ? 'Pose preview · no drop run yet' : `${outcome} · measured native simulation`;
    $('sceneOutcome').textContent = label; $('resultTitle').textContent = label;
    $('sceneOutcome').className = `scene-outcome ${physicsPreview ? 'preview' : success ? 'ready' : 'failed'}`;
    $('resultDot').className = `status-dot ${physicsPreview ? '' : success ? 'ready' : 'failed'}`;
    $('geometryLabel').textContent = 'Native FK · pale arm 10 drops · amber arm 14 holds board';
    $('resultText').textContent = physicsPreview
      ? 'Both planned arm poses, from native forward kinematics. Run one drop to measure the flight.'
      : 'Replay of the measured simulator trajectory. Replay speed changes the display only.';
    const plannerLabel = result.planner === 'fast' ? 'Faster planner' : result.planner === 'classic' ? 'Original planner' : 'Physics planner';
    pairs.splice(0, pairs.length, ['Method', `${plannerLabel} · endpoint-held`], ['Plan', result.request_id?.slice(0, 12) || '—'], ['Candidate pool', result.candidate_pool || 'Unknown']);
    if (Number.isFinite(result.timing?.planning_wall_s) && result.timing.planning_wall_s >= 0) pairs.push(['Server planning time', `${result.timing.planning_wall_s.toFixed(1)} s`]);
    if (Number.isFinite(config.tilt_deg)) pairs.push(['Board tilt', `${config.tilt_deg.toFixed(1)}°`]);
    if (Number.isFinite(config.gap_m)) pairs.push(['Release gap', `${(config.gap_m * 1000).toFixed(1)} mm`]);
    if (physicsPreview) pairs.push(['Prediction only', String(result.prediction?.predicted_route || result.route_intent || 'Unknown').replaceAll('_', ' ')]);
    else {
      pairs.push(['Observed outcome', outcome], ['Captured', result.capture === true ? 'Yes' : result.capture === false ? 'No' : 'Unknown']);
      if (Number.isFinite(result.retention_s)) pairs.push(['Retained', `${result.retention_s.toFixed(2)} s`]);
      if (Number.isInteger(result.provenance?.substeps)) pairs.push(['Native substeps', result.provenance.substeps]);
    }
  }
  addMetrics(pairs, $('metrics'));
}
async function requestSolve() {
  if (physicsSelected()) {
    if (pending || physics.busy) return;
    // Drag/grid coordinates retain full precision; inputs are rounded for display.
    const xy = finiteVector([number('bucketX'), number('bucketY')], 2) ? bucketXY.slice() : [NaN, NaN];
    clearResult('Planning both arm poses', true);
    await physics.plan(xy); return;
  }
  if (physics?.busy) return;
  if (!modeReady(report, $('mode').value) || pending) return;
  try {
    const budget = $('mode').value === 'search' && plannerReady(report) ? Number($('searchBudget').value) : undefined;
    const payload = buildRequest($('mode').value, bucketXY, $('boardLock').checked, vector('board'), $('dropLock').checked, vector('drop'), crypto.randomUUID(), budget);
    clearResult('Finding a posture'); setPending(true);
    const response = await fetch('/api/solve', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw Error(data.error || `HTTP ${response.status}`);
    activeRequest = data.job_id;
    pollSolution();
  } catch (error) { setPending(false); $('sceneOutcome').textContent = 'Simulation unavailable'; $('resultTitle').textContent = 'Unable to run this request'; $('resultText').textContent = friendlyError(error.message); $('resultDot').className = 'status-dot failed'; }
}
for (const id of ['solve', 'solveTop']) $(id).onclick = requestSolve;
async function pollSolution() {
  if (!pending) return;
  try {
    const response = await fetch('/api/solution', { cache: 'no-store' });
    if (!response.ok) throw Error('Unable to read simulation status');
    const data = await response.json();
    if (data.job_id !== activeRequest) throw Error('Simulation response belongs to another request');
    if (['running', 'queued', 'waiting'].includes(data.status)) {
      $('sceneOutcome').textContent = data.status === 'running' ? 'Checking geometry and native shot verification…' : 'Waiting for the shared 4090 · no workload interrupted';
      setTimeout(pollSolution, 700); return;
    }
    if (data.status === 'error') throw Error(data.error || 'Simulation failed');
    if (data.status !== 'completed') throw Error('Unexpected simulation state');
    applyResult(data.result); setPending(false);
  } catch (error) { setPending(false); $('sceneOutcome').textContent = 'Simulation unavailable'; $('resultTitle').textContent = 'Solve did not complete'; $('resultText').textContent = friendlyError(error.message); $('resultDot').className = 'status-dot failed'; }
}
function showPlannerFarm() {
  const rows = plannerFarmRows(report), target = $('plannerFarmGrid');
  $('plannerFarm').hidden = !rows.length;
  const signature = JSON.stringify(rows);
  if (signature !== farmSignature) {
    farmSignature = signature;
    target.replaceChildren();
    for (const row of rows) {
      const button = document.createElement('button');
      const label = `${Math.floor(row.index / 7) + 1}·${row.index % 7 + 1}`;
      button.type = 'button'; button.className = 'quiet'; button.style.cssText = 'padding:10px 3px;min-width:0';
      button.textContent = `${label}${row.cached ? ' ✓' : ''}`;
      button.title = `${row.id}: X ${(row.xy[0] * 1000).toFixed(1)}, Y ${(row.xy[1] * 1000).toFixed(1)} mm · ${row.status.replaceAll('_', ' ')}${row.cached ? ' · saved verified recipe' : ''}`;
      button.setAttribute('aria-label', button.title);
      button.dataset.bucketX = row.xy[0]; button.dataset.bucketY = row.xy[1];
      button.onclick = () => {
        if (pending || physics?.busy) return;
        setBucket(row.xy.slice()); clearResult(`Farm target ${label} selected`); showPlannerFarm();
      };
      target.append(button);
    }
  }
  for (const button of target.children) {
    const selected = Math.abs(Number(button.dataset.bucketX) - bucketXY[0]) < 1e-9 && Math.abs(Number(button.dataset.bucketY) - bucketXY[1]) < 1e-9;
    button.setAttribute('aria-pressed', String(selected)); button.disabled = pending || physics?.busy === true;
    button.style.borderColor = selected ? 'var(--teal)' : 'var(--line)';
  }
  $('plannerFarmSummary').textContent = report.planner_grid_study
    ? `${rows.filter(row => row.cached).length} / ${rows.length} targets have an original catalogue recipe. The study below shows new settings and repeat results for every table.`
    : `${rows.filter(row => row.cached).length} / ${rows.length} targets have a saved verified recipe. Select one to inspect, then solve. Other positions remain available by dragging the bucket.`;
}
const gridMillimetres = value => Number.isFinite(value) ? `${(value * 1000).toFixed(1)}` : '—';
const gridVector = value => finiteVector(value, 3) ? value.map(gridMillimetres).join(', ') : '—';
function gridRepeats(value) {
  if (!value) return 'Pending / unknown';
  return `${value.captures}/${value.attempts}${value.attempts ? ` · ${(100 * value.captures / value.attempts).toFixed(1)}%` : ' · no trials'}`;
}
function gridQuality(value) {
  if (!value || value.clean === null && value.assisted === null) return 'Unknown';
  const scope = [value.cohort ? `${value.cohort} cohort` : 'cohort unknown', value.source === 'saved_reference' ? 'saved reference' : value.source === 'fresh_study' ? 'current study' : value.source === 'frozen_validation' ? 'frozen validation' : 'source unknown'];
  return `${value.clean ?? '—'} / ${value.assisted ?? '—'}\n${scope.join(' · ')}`;
}
async function viewPlannerGridResult(row) {
  if (pending || !row.resultUrl || !row.xy) return;
  try {
    clearResult('Loading recorded shot');
    setPending(true);
    const response = await fetch(row.resultUrl, { cache: 'no-store' });
    const data = await response.json();
    if (!response.ok) throw Error(data.error || `HTTP ${response.status}`);
    const result = data.result ?? data;
    if (!result || typeof result !== 'object' || !Array.isArray(result.trajectory)) throw Error('Recorded shot has no native trajectory');
    $('boardLock').checked = false; $('dropLock').checked = false;
    setBucket(row.xy.slice()); clearResult(`Loading table ${row.label} recorded shot`);
    applyResult(result, { recordedPlayback: true });
    $('resultText').textContent = `Table ${row.label} · recorded native study evidence. ${result.message ?? 'No new search or physics trial was run.'}`;
    $('world').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    $('resultTitle').textContent = 'Recorded shot unavailable'; $('resultText').textContent = friendlyError(error.message);
    $('resultDot').className = 'status-dot failed';
  } finally { setPending(false); }
}
function showPlannerGridStudy() {
  if (typeof report.planner_grid_error === 'string') {
    $('plannerGridStudy').hidden = false; gridStudySignature = '';
    $('plannerGridStatus').textContent = `Study evidence unavailable · ${friendlyError(report.planner_grid_error)}`;
    $('plannerGridScope').textContent = 'The study report could not be verified. Recorded trial results are unavailable until a valid report is restored.';
    $('plannerGridSummary').replaceChildren(); $('plannerGridTable').replaceChildren(); $('plannerGridDownload').disabled = true;
    return;
  }
  const summary = plannerGridSummary(report), rows = plannerGridRows(report);
  $('plannerGridStudy').hidden = !summary;
  if (!summary) return;
  const signature = JSON.stringify([report.planner_grid_study, report.planner_grid_blockers, report.planner_grid_parent]);
  if (signature === gridStudySignature) return;
  gridStudySignature = signature;
  const progress = plannerGridProgress(summary);
  $('plannerGridStatus').textContent = `${summary.status.replaceAll('_', ' ')} · ${progress}${summary.updated ? ` · updated ${summary.updated}` : ''}`;
  addMetrics([['Qualified recipes available', `${summary.qualified}/${summary.total}`], ['Fresh study / saved reference', `${summary.freshQualified} / ${summary.savedQualified}`], ['Unresolved / blocked', summary.unresolved],
    ['Frozen nominal captures', `${gridRepeats(summary.frozenNominal)}${summary.frozenNominal ? ` · ${summary.frozenNominal.reported} cells` : ''}`],
    ['Frozen perturbed captures', `${gridRepeats(summary.frozenFresh)}${summary.frozenFresh ? ` · ${summary.frozenFresh.reported} cells` : ''}`]], $('plannerGridSummary'));
  $('plannerGridScope').textContent = summary.scope || 'Corrected native MuJoCo sweep search. These are development targets; repeated captures are separate from exploratory search and saved reference evidence.';
  $('plannerGridParentNote').hidden = !report.planner_grid_study.protocol?.inherited_baseline;
  const table = document.createElement('table'), head = document.createElement('thead'), header = document.createElement('tr'), body = document.createElement('tbody');
  for (const label of ['Table / bucket XY mm', 'Study status', 'Frozen validation', 'Board XYZ mm', 'Tilt', 'Ball release Z mm', 'Gap / distance mm', 'Qualification nominal', 'Qualification perturbed', 'Clean / assisted', 'Search effort']) {
    const th = document.createElement('th'); th.textContent = label; th.scope = 'col'; header.append(th);
  }
  head.append(header); table.append(head, body);
  for (const row of rows) {
    const tr = document.createElement('tr'); tr.dataset.cellId = row.id;
    const cell = text => { const td = document.createElement('td'); td.style.whiteSpace = 'pre-line'; td.textContent = text; tr.append(td); return td; };
    const identity = cell(`${row.label}\n${row.xy ? row.xy.map(gridMillimetres).join(', ') : 'Unknown XY'}`);
    identity.className = 'grid-cell-identity';
    const status = cell(`${row.status.replaceAll('_', ' ')}${row.displayStage ? `\n${row.displayStage}` : ''}`);
    status.className = 'grid-cell-status';
    if (row.reason) {
      status.title = row.reason;
      const details = document.createElement('details'), summary = document.createElement('summary'), reason = document.createElement('p');
      summary.textContent = 'Details'; reason.textContent = row.reason; details.append(summary, reason); status.append(details);
    }
    if (['qualified', 'verified_fresh', 'saved_verified'].includes(row.status)) status.style.color = 'var(--teal)';
    const currentFrozen = !row.selected && ['unresolved', 'blocked', 'failed'].includes(row.status) ? 'Not evaluated · no selected shot'
      : !row.frozenExpected ? 'Saved recipe · no new frozen repeats' : row.frozen ? `${row.frozen.status.replaceAll('_', ' ')}\nNominal ${gridRepeats(row.frozen.nominal)}\nPerturbed ${gridRepeats(row.frozen.fresh)}\nClean / assisted: ${gridQuality(row.frozen.quality)}` : 'Pending / unknown';
    const previousFrozen = row.previousFrozen ? `\n\nPrevious study frozen repeats\nNominal ${gridRepeats(row.previousFrozen.nominal)}\nPerturbed ${gridRepeats(row.previousFrozen.fresh)}\nClean / assisted: ${gridQuality(row.previousFrozen.quality)}` : '';
    const frozen = cell(currentFrozen + previousFrozen);
    if (row.frozen?.configurationSha) frozen.title = `Frozen configuration SHA256: ${row.frozen.configurationSha}`;
    cell(`${gridVector(row.selected?.board)}${row.selected ? `\n${row.selected.source}` : ''}`);
    cell(row.selected?.tilt === null || !row.selected ? '—' : `${row.selected.tilt.toFixed(2)}°`);
    cell(gridMillimetres(row.selected?.release?.[2]));
    cell(`${gridMillimetres(row.selected?.gap)} / ${gridMillimetres(row.selected?.distance)}`);
    cell(`${gridRepeats(row.nominal)}${row.savedNominal ? `\nSaved: ${gridRepeats(row.savedNominal)}` : ''}`);
    cell(`${gridRepeats(row.fresh)}${row.savedFresh ? `\nSaved: ${gridRepeats(row.savedFresh)}` : ''}`);
    cell(gridQuality(row.quality));
    const searchCaptures = row.search.captures !== null && row.search.nativeRequests !== null
      ? `${row.search.captures}/${row.search.nativeRequests} search captures`
      : `${row.search.nativeRequests ?? '—'} native requests · captures unknown`;
    cell(`${searchCaptures}\n${row.search.rejected ?? '—'} geometry rejects${row.search.cpuCandidates !== null ? ` / ${row.search.cpuCandidates} proposals` : ''}`);
    if (row.resultUrl && row.xy) {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'quiet'; button.textContent = 'View recorded shot';
      button.setAttribute('aria-label', `View table ${row.label} recorded native shot`); button.disabled = pending || physics?.busy === true;
      button.onclick = () => viewPlannerGridResult(row); identity.append(button);
    } else {
      const unavailable = document.createElement('small'); unavailable.textContent = 'No saved trajectory'; identity.append(unavailable);
    }
    body.append(tr);
  }
  $('plannerGridTable').replaceChildren(table);
  $('plannerGridDownload').disabled = !rows.length;
}
$('plannerGridDownload').onclick = () => {
  const blob = new Blob([plannerGridCsv(report)], { type: 'text/csv;charset=utf-8' }), url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url; link.download = 'sircus-35-table-sweep.csv'; link.click(); URL.revokeObjectURL(url);
};
const benchmarkNumber = (value, suffix = '', digits = 1) => value === null || value === undefined ? 'Unknown' : `${Number(value).toFixed(digits)}${suffix}`;
const benchmarkCounts = value => value ? `${value.captures}/${value.attempts}${value.attempts ? ` · ${(100 * value.captures / value.attempts).toFixed(1)}%` : ' · no trials'}` : 'Unknown';
function drawBenchmarkCurve(target, curves, metric) {
  const valid = point => metric === 'rate' ? point.rate !== null : point.wallS !== null;
  const all = curves.flatMap(curve => curve.points.filter(valid));
  if (!all.length) { const text = document.createElement('p'); text.textContent = 'Waiting for measured validation milestones.'; target.replaceChildren(text); return; }
  const width = 420, height = 240, left = 48, right = 18, top = 18, bottom = 58;
  const maxX = Math.max(1, ...all.map(point => point.samples));
  const maxY = metric === 'rate' ? 100 : Math.max(1, ...all.map(point => point.wallS)) * 1.08;
  const px = value => left + value / maxX * (width - left - right), py = value => height - bottom - value / maxY * (height - top - bottom);
  const svg = makeSvg('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': metric === 'rate' ? 'Validation capture rate versus new training shot requests for sweep and residual PPO' : 'Setup and training elapsed time excluding validation versus equal shot-request budgets' });
  for (let i = 0; i <= 4; i++) {
    const value = maxY * i / 4, y = py(value);
    svg.append(makeSvg('line', { x1: left, x2: width - right, y1: y, y2: y, class: 'grid' }));
    const label = makeSvg('text', { x: left - 7, y: y + 3, 'text-anchor': 'end' }); label.textContent = metric === 'rate' ? `${value.toFixed(0)}%` : value >= 60 ? `${(value / 60).toFixed(1)}m` : `${value.toFixed(0)}s`; svg.append(label);
  }
  const colors = ['#a97312', '#128b8c', '#b75439'];
  const legend = document.createElement('div'); legend.className = 'benchmark-chart-legend';
  curves.forEach((curve, index) => {
    const points = curve.points.filter(valid), color = colors[index % colors.length];
    if (!points.length) return;
    const value = point => metric === 'rate' ? point.rate * 100 : point.wallS;
    svg.append(makeSvg('polyline', { points: points.map(point => `${px(point.samples)},${py(value(point))}`).join(' '), fill: 'none', stroke: color, 'stroke-width': 2.5, 'stroke-dasharray': index ? '6 3' : 'none' }));
    for (const point of points) {
      if (metric === 'rate' && point.seeds > 1 && point.minRate !== null && point.maxRate !== null) {
        const x = px(point.samples), y1 = py(point.minRate * 100), y2 = py(point.maxRate * 100);
        svg.append(makeSvg('line', { x1: x, x2: x, y1, y2, stroke: color, 'stroke-width': 1.4, opacity: .65 }));
        for (const y of [y1, y2]) svg.append(makeSvg('line', { x1: x - 4, x2: x + 4, y1: y, y2: y, stroke: color, 'stroke-width': 1.4 }));
      }
      const dot = makeSvg('circle', { cx: px(point.samples), cy: py(value(point)), r: 3.5, fill: color });
      const title = makeSvg('title'); title.textContent = `${curve.label} · ${point.samples} requests · ${point.seeds} seeds · ${point.captures}/${point.attempts} validation captures`; dot.append(title); svg.append(dot);
    }
    const entry = document.createElement('span'), swatch = document.createElement('i'), label = document.createElement('span');
    swatch.className = 'benchmark-legend-swatch'; swatch.style.borderTop = `3px ${index ? 'dashed' : 'solid'} ${color}`; swatch.setAttribute('aria-hidden', 'true');
    label.textContent = curve.label; entry.append(swatch, label); legend.append(entry);
  });
  const ticks = [...new Set([0, ...all.map(point => point.samples)])].sort((a, b) => a - b);
  for (const value of ticks) { const label = makeSvg('text', { x: px(value), y: height - bottom + 17, 'text-anchor': 'middle' }); label.textContent = value; svg.append(label); }
  const axis = makeSvg('text', { x: width / 2, y: height - 24, 'text-anchor': 'middle', class: 'axis-label' }); axis.textContent = 'New training shot requests / method / seed'; svg.append(axis);
  target.replaceChildren(svg, legend);
}
const benchmarkLegacyNotes = Object.fromEntries(['benchmarkMetricNote', 'benchmarkResourceNote', 'benchmarkQueryNote'].map(id => [id, $(id).textContent]));
function showSweepRlBenchmark() {
  if (typeof report.sweep_rl_error === 'string') {
    $('sweepRlBenchmark').hidden = false; benchmarkSignature = '';
    $('benchmarkStatus').textContent = `Benchmark evidence unavailable · ${friendlyError(report.sweep_rl_error)}`;
    for (const id of ['benchmarkSummary', 'benchmarkCaptureChart', 'benchmarkTimeChart', 'benchmarkResourceTable', 'benchmarkMilestones', 'benchmarkWorkerCosts', 'benchmarkFinalTable', 'benchmarkFinalAggregateTable']) $(id).replaceChildren();
    $('benchmarkScope').textContent = 'The benchmark report could not be verified. No benchmark measurements are displayed.';
    $('benchmarkPrior').textContent = ''; $('benchmarkProgress').textContent = ''; $('benchmarkAudit').textContent = ''; $('benchmarkFinalStatus').textContent = 'Final results unavailable.'; $('benchmarkDownload').disabled = true;
    for (const id of ['benchmarkRule', 'benchmarkAdmissionSection', 'benchmarkHistory', 'benchmarkFinalAggregate']) $(id).hidden = true;
    return;
  }
  const study = sweepRlStudy(report), summary = sweepRlSummary(report);
  $('sweepRlBenchmark').hidden = !summary;
  if (!summary) return;
  const signature = JSON.stringify([study, report.historical_sweep_rl_benchmark, report.historical_sweep_rl_error]);
  if (signature === benchmarkSignature) return;
  const firstDisplay = !benchmarkSignature;
  benchmarkSignature = signature;
  const curves = sweepRlCurves(report), points = sweepRlPoints(report), finals = sweepRlFinalRows(report), progress = sweepRlProgress(report), workers = sweepRlCompletedWorkers(report);
  $('benchmarkHeading').textContent = summary.calibrated ? 'Sweep versus RL · calibrated v2' : 'Sweep versus RL · original v1';
  $('benchmarkRule').hidden = !summary.calibrated;
  if (summary.calibrated) {
    const c = study.calibration?.score_config ?? {}, setup = study.setup ?? {};
    $('benchmarkRule').textContent = `Capture: board rebound → whole-ball bucket entry → ${benchmarkNumber(c.retention_s, ' s', 2)} continuously retained. Rim-assisted captures count. Calibrated contact allowances: wall ${benchmarkNumber(c.wall_tolerance_m === undefined ? null : c.wall_tolerance_m * 1000, ' mm', 3)}, floor ${benchmarkNumber(c.floor_tolerance_m === undefined ? null : c.floor_tolerance_m * 1000, ' mm', 3)}. Scorer ${study.protocol?.scorer_version ?? 'Unknown'}. Calibration ${study.calibration?.score_config_sha256 ?? 'Unknown'}.`;
    $('benchmarkMetricNote').textContent = `New v2 metric and setup contract; historical v1 labels remain unchanged. A new canonical simulator park replaces the previously obstructed initial pose; connectivity from the old park is not claimed. Shared geometry resolution allows at most ${setup.geometry_candidate_budget ?? 'Unknown'} source/offset candidates before prediction. Exact action endpoints and forward/reverse paths retain the 5 mm clearance threshold. No physical validation.`;
    $('benchmarkResourceNote').textContent = 'Instrumented wall/CPU include setup, geometry-only source search, exact-action admission and evidence I/O; evaluation is separate. Endpoint/home and sampled-path wall costs are reported separately where observed. Native GPU phase is elapsed simulation time, not kernel utilization. Full GPU reservation includes evaluation. RSS and Torch peaks include simulator/evaluation; process VRAM is sampled once per second. Memory is MiB; missing values stay unknown.';
    $('benchmarkQueryNote').textContent = 'Frozen prediction still pays geometry-only pre-policy source resolution; its measured time is shown separately from prediction/preparation. Neither method searches ball-flight outcomes at test time. These subtotals exclude artifact loading and later exact-action IK/path admission/native execution, so they are not end-to-end latency. PPO artifact size includes actor, critic and optimizer state.';
  } else for (const [id, text] of Object.entries(benchmarkLegacyNotes)) $(id).textContent = text;
  $('benchmarkStatus').textContent = `${sweepRlStatusLabel(summary)} · ${summary.completedRuns}/${summary.totalRuns} method/seed runs complete${summary.auditStatus ? ` · audit ${summary.auditStatus}` : ''}`;
  const observed = value => value === null ? 'Unknown' : value.toLocaleString(), runTotal = progress.declaredRuns ?? 'unknown';
  const trainingProgress = progress.requests === null ? 'Awaiting reported training-request counters.' : `Recorded training decisions: ${observed(progress.requests)} / ${observed(progress.declaredRequests)} declared.`;
  const runningProgress = progress.reportedRunning.map(worker => `${worker.label}, seed ${worker.seed}: ${observed(worker.requests)} / ${observed(progress.budget)} requests`).join('; ');
  $('benchmarkProgress').textContent = `${trainingProgress} Recorded native training releases: ${observed(progress.nativeReleases)}. Counter coverage: requests ${progress.requestCounterRuns}/${runTotal}, native ${progress.nativeCounterRuns}/${runTotal} runs.${runningProgress ? ` Reported running: ${runningProgress}.` : ''}${progress.conflictingRuns ? ` ${progress.conflictingRuns} conflicting worker counter record(s) omitted.` : ''} Validation and test requests are separate.`;
  addMetrics([['Equal training budget per method / seed', summary.budget === null ? 'Unknown' : summary.budget],
    ['Training seeds', summary.seedCount || 'Not reported'], ['Measured validation checkpoints', points.length],
    ['Verified final method / seed records', summary.status === 'completed' && (!summary.calibrated || summary.testDisclosed) ? finals.length : 'Pending']], $('benchmarkSummary'));
  $('benchmarkScope').textContent = summary.scope || 'Matched current-ball shot benchmark; source/physics and evaluation limits belong to the frozen contract.';
  $('benchmarkPrior').textContent = `Shared prior: ${summary.priorRecipes === null ? 'unknown number of' : summary.priorRecipes} recipes. Latest source collection: ${summary.priorReleases === null ? 'unknown' : summary.priorReleases.toLocaleString()} native releases, separate from the new training budget. ${summary.priorFullCostKnown ? '' : 'Full predecessor acquisition cost is not known. '}${summary.priorScope}`;
  drawBenchmarkCurve($('benchmarkCaptureChart'), curves, 'rate'); drawBenchmarkCurve($('benchmarkTimeChart'), curves, 'wallS');
  const latest = curves.filter(curve => curve.points.length).map(curve => ({ ...curve.points.at(-1), label: curve.label }));
  expansionTable($('benchmarkResourceTable'), ['Method / budget', 'Validation captures', 'Seeds reported', 'Native validation / rejected', 'Mean setup + training time', summary.calibrated ? 'Mean CPU / endpoint-home / path wall' : 'Mean CPU / endpoint-home wall', 'Mean GPU phase / reservation', 'Peak RSS / sampled VRAM', 'Torch allocated / reserved peak'], latest.map(row => [
    `${row.label}\n${row.samples} requests`, benchmarkCounts(row), `${row.seeds} / ${summary.seedCount || 'unknown'}`,
    `${row.nativeReleases ?? 'Unknown'} / ${row.invalid ?? 'Unknown'}`, benchmarkNumber(row.wallS, ' s'),
    `${benchmarkNumber(row.cpuS, ' s')} / ${benchmarkNumber(row.geometryS, ' s')}${summary.calibrated ? ` / ${benchmarkNumber(row.pathS, ' s')}` : ''}`, `${benchmarkNumber(row.gpuS, ' s')} / ${benchmarkNumber(row.gpuReservationS, ' s')}`,
    `${benchmarkNumber(row.peakRssMb, ' MiB')} / ${benchmarkNumber(row.peakVramMb, ' MiB')}`,
    `${benchmarkNumber(row.torchAllocatedMb, ' MiB')} / ${benchmarkNumber(row.torchReservedMb, ' MiB')}`]));
  expansionTable($('benchmarkMilestones'), ['Method', 'Seed', 'Training requests', 'Validation capture', 'Native / rejected', 'Setup + training', 'CPU / endpoint-home wall', 'GPU phase / reservation', 'Peak RSS / sampled VRAM', 'Validation time'], points.map(row => [
    row.label, row.seed, row.samples, benchmarkCounts(row.counts), `${row.counts.nativeReleases ?? 'Unknown'} / ${row.counts.invalid ?? 'Unknown'}`,
    benchmarkNumber(row.resources.wall_s, ' s'), `${benchmarkNumber(row.resources.cpu_s, ' s')} / ${benchmarkNumber(row.resources.geometry_s, ' s')}`,
    `${benchmarkNumber(row.resources.gpu_s, ' s')} / ${benchmarkNumber(row.resources.gpu_reservation_s, ' s')}`, `${benchmarkNumber(row.resources.peak_rss_mb, ' MiB')} / ${benchmarkNumber(row.resources.peak_vram_mb, ' MiB')}`,
    benchmarkNumber(row.resources.evaluation_wall_s, ' s')]));
  expansionTable($('benchmarkWorkerCosts'), ['Completed worker / seed', 'Spent TRAIN requests / native releases', 'Setup + TRAIN wall / CPU', 'Validation + test wall', 'Full GPU reservation', 'Lifetime peak RSS / sampled VRAM', 'Torch allocated / reserved peak'], workers.map(row => [
    `${row.label}\nSeed ${row.seed}`, `${row.requests ?? 'Unknown'} / ${row.nativeReleases ?? 'Unknown'}`,
    `${benchmarkNumber(row.resources.wall_s, ' s')} / ${benchmarkNumber(row.resources.cpu_s, ' s')}`,
    benchmarkNumber(row.resources.evaluation_wall_s, ' s'), benchmarkNumber(row.resources.gpu_reservation_s, ' s'),
    `${benchmarkNumber(row.resources.peak_rss_mb, ' MiB')} / ${benchmarkNumber(row.resources.peak_vram_mb, ' MiB')}`,
    `${benchmarkNumber(row.resources.torch_allocated_mb, ' MiB')} / ${benchmarkNumber(row.resources.torch_reserved_mb, ' MiB')}`]));
  const admission = sweepRlAdmissionRows(report);
  $('benchmarkAdmissionSection').hidden = !summary.calibrated;
  expansionTable($('benchmarkAdmissionTable'), ['Method / seed', 'TRAIN requested', 'Native releases / setup rejects', 'Fixed-base rejects', 'Unresolved pre-policy anchors', 'Native numeric failures', 'Setup rejection reasons'], admission.map(row => [
    `${row.label}\nSeed ${row.seed}`, row.requested, `${row.native} / ${row.rejected}`, row.fixedBase ?? 'Unknown', row.anchorUnresolved ?? 'Unknown', row.invalidPhysics ?? 'Unknown', row.reasons.map(([reason, count]) => `${reason}: ${count}`).join('\n') || 'None observed']));
  $('benchmarkFinalStatus').textContent = summary.status === 'completed' && (!summary.calibrated || summary.testDisclosed) ? `${finals.length} frozen method/seed records. Test requests are separate from training and checkpoint-selection validation.` : summary.calibrated ? 'TEST results remain hidden until the full benchmark completes and its independent byte-bound audit passes. Validation is not final test performance.' : 'Final results remain pending until the benchmark completes. Validation is not final test performance.';
  const aggregate = sweepRlFinalAggregate(report);
  $('benchmarkFinalAggregate').hidden = !aggregate;
  expansionTable($('benchmarkFinalAggregateTable'), ['Method', 'Captured / requested', 'Capture rate', 'Training seeds', 'Clean captures'],
    aggregate ? aggregate.methods.map(row => [row.label, `${row.captures} / ${row.attempts}`, `${row.percent.toFixed(1)}%`, row.seeds, row.clean]) : []);
  $('benchmarkFinalAggregateNote').textContent = aggregate ? `All requested cases, including admission rejects, remain in the denominator.${aggregate.ppoMinusSweepPp === null ? '' : ` Observed PPO − sweep difference: ${aggregate.ppoMinusSweepPp >= 0 ? '+' : ''}${aggregate.ppoMinusSweepPp.toFixed(1)} percentage points.`}` : '';
  expansionTable($('benchmarkFinalTable'), ['Frozen method', 'Seed', 'Selected training budget', 'Test captures', 'Clean', 'Native / rejected', summary.calibrated ? '24-layout prediction / pre-policy geometry' : '24-layout prediction + preparation', 'Artifact SHA256'], finals.map(row => [row.label, row.seed, row.selectedSamples ?? 'Unknown',
    benchmarkCounts(row.counts), row.counts.clean ?? 'Unknown', `${row.counts.nativeReleases ?? 'Unknown'} / ${row.counts.invalid ?? 'Unknown'}`,
    `${benchmarkNumber(row.queryS, ' s', 3)}${summary.calibrated ? ` / ${benchmarkNumber(row.contextResolutionS, ' s', 3)}` : ''}`, row.artifactSha]));
  $('benchmarkAudit').textContent = `${summary.auditStatus ? `Independent audit: ${summary.auditStatus}. ` : 'Independent audit: pending or not reported. '}Contract: ${summary.fingerprint ?? 'not reported'}. This panel does not activate a checkpoint or launch training.`;
  $('benchmarkDownload').disabled = !points.length && !finals.length;
  const history = sweepRlHistory(report);
  $('benchmarkHistory').hidden = !history && !report.historical_sweep_rl_error;
  $('benchmarkHistoryNote').textContent = history ? `${history.metric ?? 'Original v1 metric'}. ${history.limitation ?? ''}` : report.historical_sweep_rl_error ?? '';
  const historicalRows = history ? sweepRlFinalRows({ sweep_rl_benchmark: history }) : [];
  expansionTable($('benchmarkHistoryTable'), ['Historical method', 'Seed', 'Original v1 test capture', 'Native / rejected'], historicalRows.map(row => [row.label, row.seed, benchmarkCounts(row.counts), `${row.counts.nativeReleases ?? 'Unknown'} / ${row.counts.invalid ?? 'Unknown'}`]));
  if (firstDisplay && location.hash === '#sweepRlBenchmark') $('sweepRlBenchmark').scrollIntoView({ block: 'start' });
}
$('benchmarkDownload').onclick = () => {
  const blob = new Blob([sweepRlCsv(report)], { type: 'text/csv;charset=utf-8' }), url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url; link.download = 'sircus-sweep-rl-measurements.csv'; link.click(); URL.revokeObjectURL(url);
};
$('play').onclick = () => { if (!currentTrajectory.length) return; if (playTime >= currentTrajectory.at(-1).time) playTime = currentTrajectory[0].time; playing = !playing; };
$('timeline').oninput = () => { playing = false; if (currentTrajectory.length) playTime = currentTrajectory[0].time + Number($('timeline').value) / 1000 * (currentTrajectory.at(-1).time - currentTrajectory[0].time); };
function makeSvg(tag, attributes = {}) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
}
function drawChart(target, points, valueKey, showValidation = false) {
  const clean = points.filter(p => Number.isFinite(p[valueKey]));
  if (!clean.length) { const p = document.createElement('p'); p.textContent = 'Waiting for PPO batches.'; target.replaceChildren(p); return; }
  const width = 330, height = 170, left = 37, right = 13, top = 15, bottom = 29;
  const values = clean.map(p => p[valueKey]);
  if (showValidation) values.push(...points.filter(p => p.validationReward !== null).map(p => p.validationReward));
  let min = Math.min(0, ...values), max = Math.max(...values);
  if (max - min < .001) max = min + 1;
  const maxUpdate = Math.max(1, ...clean.map(p => p.update)),
    px = v => left + v / maxUpdate * (width - left - right),
    py = v => height - bottom - (v - min) / (max - min) * (height - top - bottom);
  const svg = makeSvg('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': valueKey === 'captures' ? 'Training captures per PPO update' : 'Training and validation mean shaped reward per PPO update' });
  for (let i = 0; i <= 3; i++) {
    const value = min + (max - min) * i / 3, y = py(value);
    svg.append(makeSvg('line', { x1: left, x2: width - right, y1: y, y2: y, class: 'grid' }));
    const text = makeSvg('text', { x: left - 6, y: y + 3, 'text-anchor': 'end' }); text.textContent = valueKey === 'captures' ? value.toFixed(0) : value.toFixed(2); svg.append(text);
  }
  const polyline = makeSvg('polyline', { points: clean.map(p => `${px(p.update)},${py(p[valueKey])}`).join(' '), class: 'series' }); svg.append(polyline);
  for (const point of clean) svg.append(makeSvg('circle', { cx: px(point.update), cy: py(point[valueKey]), r: 2.5, class: 'point' }));
  if (showValidation) for (const point of points.filter(p => p.validationReward !== null)) svg.append(makeSvg('circle', { cx: px(point.update), cy: py(point.validationReward), r: 3.2, class: 'validation' }));
  for (const value of [0, maxUpdate]) { const text = makeSvg('text', { x: px(value), y: height - 12, 'text-anchor': 'middle' }); text.textContent = value; svg.append(text); }
  const axis = makeSvg('text', { x: width / 2, y: height - 3, 'text-anchor': 'middle', class: 'axis-label' }); axis.textContent = 'PPO update'; svg.append(axis);
  target.replaceChildren(svg);
}
function showBankCoverage() {
  const rows = bankCoverageRows(report), context = $('bankCoverage').getContext('2d'), width = 720, height = 230;
  context.clearRect(0, 0, width, height); context.fillStyle = '#edf0e7'; context.fillRect(0, 0, width, height);
  const px = x => 35 + (x - .1) / 1.14 * (width - 70), py = y => height - 28 - (y + .65) / .9 * (height - 50);
  context.strokeStyle = '#cbd6ca'; context.lineWidth = 1;
  for (let i = 0; i <= 6; i++) { const x = 35 + i * (width - 70) / 6; context.beginPath(); context.moveTo(x, 15); context.lineTo(x, height - 28); context.stroke(); }
  const counts = { captured: 0, missed: 0, unresolved: 0 }, colors = { captured: '#128b8c', missed: '#ec7657', unresolved: '#9aaba0' };
  for (const row of rows) {
    const status = bankCoverageStatus(row); counts[status]++;
    context.fillStyle = colors[status]; context.beginPath(); context.arc(px(row.xy[0]), py(row.xy[1]), 5, 0, Math.PI * 2); context.fill();
  }
  context.fillStyle = '#6e8180'; context.font = '12px sans-serif'; context.fillText('Fresh test XY · X100–1240mm / Y−650–250mm', 20, height - 8);
  $('bankCoverageCount').textContent = rows.length ? `${counts.captured} captured · ${counts.missed} missed · ${counts.unresolved} unresolved / ${rows.length}` : 'No fresh test coverage yet';
}
function showBankComparison() {
  const bank = report.bank_evaluation;
  $('bankResults').hidden = !bank;
  if (!bank) return;
  showBankCoverage();
  $('bankStatus').textContent = `Relative-bank study ${bank.status ?? 'pending'} · separate fresh evaluation`;
  const pairs = [];
  for (const [key, label] of [['train_episodes', 'Bank training attempts'], ['updates', 'Bank PPO updates'], ['test_layouts', 'Fresh held-out bucket positions']]) if (bank[key] !== undefined) pairs.push([label, bank[key]]);
  if (pairs.length) addMetrics(pairs, $('bankSummary'));
  $('bankScope').textContent = bank.source_scope || 'Frozen TRAIN-derived relative configurations and geometry filtering restrict the action family. The selector learns among these proposals; it is not unconstrained joint-space learning.';
  const rows = bankComparisonRows(report);
  if (!rows.length) { $('bankComparison').replaceChildren(); return; }
  const table = document.createElement('table'), head = document.createElement('thead'), heading = document.createElement('tr');
  for (const label of ['Method / fresh split', 'Captures / shots', 'Capture %', 'Invalid', 'Mean reward']) { const th = document.createElement('th'); th.textContent = label; heading.append(th); }
  head.append(heading); table.append(head);
  const body = document.createElement('tbody');
  for (const row of rows) {
    const tr = document.createElement('tr'); if (row.key === 'bank_selected') tr.className = 'selected';
    for (const value of [row.label, `${row.captures} / ${row.attempts}`, row.attempts ? `${(100 * row.captures / row.attempts).toFixed(1)}%` : '—', row.invalid, row.meanReward === null ? '—' : row.meanReward.toFixed(3)]) { const td = document.createElement('td'); td.textContent = String(value); tr.append(td); }
    body.append(tr);
  }
  table.append(body); $('bankComparison').replaceChildren(table);
}
function expansionTable(target, headings, rows, selected = () => false) {
  const table = document.createElement('table'), head = document.createElement('thead'), header = document.createElement('tr');
  for (const label of headings) { const cell = document.createElement('th'); cell.textContent = label; header.append(cell); }
  head.append(header); table.append(head);
  const body = document.createElement('tbody');
  for (const row of rows) {
    const tr = document.createElement('tr'); if (selected(row)) tr.className = 'selected';
    for (const value of row) { const td = document.createElement('td'); td.textContent = String(value); tr.append(td); }
    body.append(tr);
  }
  table.append(body); target.replaceChildren(table);
}
function showExpansion() {
  const progress = expansionProgress(report);
  $('expandedStudy').hidden = !progress;
  if (!progress) return;
  $('expansionStatus').textContent = `Expanded-bank study · ${progress.status}`;
  $('expansionActivePolicy').textContent = progress.activePolicy;
  $('expansionPhase').textContent = `Search: ${progress.searchPhase} · Training: ${progress.trainingPhase}${progress.queued ? ` · Waiting for shared 4090 (${Math.round(progress.waitSeconds)} s)` : ''}`;
  addMetrics([
    ['Development targets with a promoted template', `${progress.promoted} / ${progress.targetCount || '—'}`],
    ['Recorded native search throws · all stages', progress.nativeSearchAttempts], ['Perturbed confirmation attempts · all stages', progress.confirmationAttempts],
    ['CPU geometry proposals', progress.geometryProposals], ['PPO training attempts', progress.trainAttempts],
    ['Native training releases', progress.nativeTrainReleases],
    ['PPO updates', `${progress.updates}${progress.updateBudget ? ` / ${progress.updateBudget}` : ''}`],
  ].filter(([, value]) => value !== null), $('expansionSummary'));
  const plan = progress.trainLayouts === null ? 'Fresh train/validation/test splits will be declared before training.' : `${progress.trainLayouts} training layouts · ${progress.validationLayouts ?? '—'} validation · ${progress.testLayouts ?? '—'} fresh test × ${progress.testRepeats ?? '—'} nominal repeats.`;
  const geometry = progress.geometry ? ` Current CPU audit: ${progress.geometry.group}, ${progress.geometry.completed} / ${progress.geometry.total} queries.` : '';
  const sampling = progress.eligibleContexts === null ? '' : ` Training samples ${progress.eligibleContexts} endpoint-eligible contexts; ${progress.excludedContexts ?? '—'} excluded contexts remain in complete evaluation denominators.`;
  $('expansionPlan').textContent = plan + geometry + sampling;
  const targets = expansionTargetRows(report);
  if (targets.length) expansionTable($('expansionTargets'), ['Converted DEV target', 'X / Y (mm)', 'Search captures / attempts, or retained sources', 'Promoted confirmation', 'Status / source'], targets.map(row => [
    row.id, row.xy.map(v => (v * 1000).toFixed(1)).join(' / '), row.retainedSources && !row.attempts ? `${row.retainedSources} prior captured sources` : `${row.captures} / ${row.attempts}`,
    row.promoted ? `${row.confirmationCaptures} / ${row.confirmationAttempts}` : '—',
    row.promoted ? `${row.phase || 'Confirmed'} · ${row.sourceId}` : row.unresolvedReason,
  ]));
  else $('expansionTargets').textContent = 'The twelve converted-development coordinates will appear when the search report is available.';
  const rows = expandedComparisonRows(report);
  if (rows.length) expansionTable($('expansionComparison'), ['Frozen method', 'Captures / shots', 'Overall %', 'Clean captures', 'Eligible shots', 'Conditional %', 'Positions captured', 'Unresolved positions'], rows.map(row => [
    row.label, `${row.captures} / ${row.attempts}`, row.attempts ? `${(100 * row.captures / row.attempts).toFixed(1)}%` : '—', row.clean,
    row.eligibleAttempts, row.conditionalCaptureRate === null ? '—' : `${(100 * row.conditionalCaptureRate).toFixed(1)}%`,
    `${row.positionsWithCapture} / ${row.layouts}`, row.unresolvedPositions,
  ]), row => row[0] === 'Expanded bank · learned');
  else $('expansionComparison').textContent = 'Fresh frozen-checkpoint comparisons will appear after the full training study completes.';
  const paired = progress.training?.paired_comparisons?.learning_vs_uniform;
  const interval = paired?.descriptive_95pct_layout_bootstrap;
  $('expansionConclusion').textContent = interval?.length === 2 && Number.isFinite(paired.capture_rate_difference)
    ? `Learned minus uniform capture rate: ${(100 * paired.capture_rate_difference).toFixed(1)} percentage points; descriptive 95% paired layout interval ${(100 * interval[0]).toFixed(1)} to ${(100 * interval[1]).toFixed(1)}. ${interval[0] <= 0 && interval[1] >= 0 ? 'No established capture-rate advantage.' : 'Single seeded simulation study.'} Clean capture counts are a separate quality measure.`
    : '';
  const points = learningPoints(progress.training);
  drawChart($('expansionCaptureChart'), points, 'captures');
  drawChart($('expansionRewardChart'), points, 'reward', true);
}
// BEGIN READ-ONLY WINDOW STUDY
const WINDOW_METHODS = {adaptive_surrogate:'Supervised success model',sac:'Terminal SAC',uniform_bounded:'Uniform bounded gaps'};
function windowStatistics(value = {}) {
  const n = Number.isInteger(value.attempts) ? value.attempts : Number.isInteger(value.requests) ? value.requests : null;
  const k = Number.isInteger(value.captures) ? value.captures : null;
  const valid = n !== null && k !== null && n >= 0 && k >= 0 && k <= n;
  return {requests:n,captures:valid?k:null,released:Number.isInteger(value.native_releases)?value.native_releases:null,
    fraction:valid?`${k} / ${n}`:'—',rate:valid&&n?k/n:null,
    reward:Number.isFinite(value.mean_learning_reward)?value.mean_learning_reward:Number.isFinite(value.mean_reward)?value.mean_reward:null,
    clean:Number.isInteger(value.clean)?value.clean:null,invalid:Number.isInteger(value.invalid)?value.invalid:null,
    interval:Array.isArray(value.descriptive_95pct_wilson)&&value.descriptive_95pct_wilson.length===2&&value.descriptive_95pct_wilson.every(Number.isFinite)?value.descriptive_95pct_wilson:null};
}
function windowBoardCoverage(study) {
  const groups=study.settings?.contexts??{},raw=study.prepared_contexts??[];
  const prepared=Array.isArray(raw)?raw:Object.values(raw).filter(Array.isArray).flat();
  const exactLegacy=study.version==='sircus-fixed-bucket-gap-study-v1'&&study.settings?.version==='sircus-fixed-bucket-gap-study-v1';
  const summarize=rows=>{
    let resolved=0,unresolved=0,unknown=0;
    for(const row of rows){
      const matches=prepared.filter(p=>(p.id??p.context_id)===row.id&&p.board_height_m===row.board_height_m&&p.tilt_deg===row.tilt_deg);
      const p=matches.length===1?matches[0]:null;
      let outcome=null;
      if(p&&p.prepared!==false){
        if(typeof p.board_admitted==='boolean')outcome=p.board_admitted;
        else if(exactLegacy&&Object.hasOwn(p,'reason')){
          if(p.reason===null)outcome=true;else if(typeof p.reason==='string')outcome=false;
        }
      }
      if(outcome===true)resolved++;else if(outcome===false)unresolved++;else unknown++;
    }
    return {total:rows.length,resolved,unresolved,unknown};
  };
  const rows=['train','validation','test'].map(name=>Array.isArray(groups[name])?groups[name]:[]);
  return {all:summarize(rows.flat()),train:summarize(rows[0]),validation:summarize(rows[1]),test:summarize(rows[2])};
}
function windowConfirmationView(study) {
  const value=study.window_confirmations;if(!value||typeof value!=='object')return null;
  if(typeof study.contract_fingerprint!=='string'||!study.contract_fingerprint||value.parent_contract_fingerprint!==study.contract_fingerprint)
    return {status:'failed',reason:'Window confirmation parent fingerprint does not match this learning study',recipes:[]};
  return value;
}
function windowStudyView(value) {
  if (!value || typeof value !== 'object') return null;
  const groups=value.settings?.contexts??{}, identities=[], coordinates=[];
  const split=['train','validation','test'].map(name=>{
    const rows=Array.isArray(groups[name])?groups[name]:[];
    for(const r of rows){identities.push(r.id);coordinates.push(`${r.board_height_m}:${r.tilt_deg}`);}
    return rows.length;
  });
  const disjoint=identities.length>0&&identities.every(v=>typeof v==='string')&&new Set(identities).size===identities.length&&new Set(coordinates).size===coordinates.length;
  return {raw:value,split,disjoint,boardCoverage:windowBoardCoverage(value),curve:(value.learning_curve??[]).filter(r=>r.method in WINDOW_METHODS&&Number.isFinite(r.milestone))
    .map(r=>({...r,stats:windowStatistics(r.validation)})),test:Object.entries(value.test??{}).filter(([name])=>name in WINDOW_METHODS)
    .map(([name,stats])=>({name,stats:windowStatistics(stats)}))};
}
function windowTrajectoryView(study, trace) {
  const board=study.settings?.board_xy_m,bucket=study.settings?.bucket_xy_m;
  if(!finiteVector(board,2)||!finiteVector(bucket,2)||!Array.isArray(trace?.position_m)||trace.position_m.length<2||!trace.position_m.every(p=>finiteVector(p,3))) return null;
  const distance=Math.hypot(bucket[0]-board[0],bucket[1]-board[1]);
  if(distance<1e-9) return null;
  const axis=[(bucket[0]-board[0])/distance,(bucket[1]-board[1])/distance];
  return {distance,points:trace.position_m.map(p=>[(p[0]-board[0])*axis[0]+(p[1]-board[1])*axis[1],p[2]]),
    lateral:Math.max(...trace.position_m.map(p=>Math.abs(-(p[0]-board[0])*axis[1]+(p[1]-board[1])*axis[0])))};
}
function windowPlot(canvas, xlow, xhigh, ylow, yhigh, xlabel, ylabel) {
  const c=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
  c.clearRect(0,0,w,h);c.fillStyle='#edf0e7';c.fillRect(0,0,w,h);
  const x=v=>55+(v-xlow)/Math.max(1e-9,xhigh-xlow)*(w-80),y=v=>h-42-(v-ylow)/Math.max(1e-9,yhigh-ylow)*(h-68);
  c.font='11px sans-serif';c.textAlign='center';
  for(let i=0;i<=4;i++){
    const vx=xlow+(xhigh-xlow)*i/4,vy=ylow+(yhigh-ylow)*i/4;
    c.strokeStyle='#d3ddd0';c.beginPath();c.moveTo(x(vx),y(ylow));c.lineTo(x(vx),y(yhigh));c.stroke();
    c.beginPath();c.moveTo(x(xlow),y(vy));c.lineTo(x(xhigh),y(vy));c.stroke();
    c.fillStyle='#6e8180';c.fillText(vx.toFixed(xhigh>10?0:2),x(vx),h-25);c.fillText(vy.toFixed(yhigh>10?0:2),30,y(vy)+4);
  }
  c.fillStyle='#496f67';c.fillText(xlabel,w/2,h-7);c.textAlign='left';c.fillText(ylabel,55,15);
  return {c,x,y};
}
function drawWindowCurve(view) {
  const points=view.curve.filter(r=>r.stats.rate!==null),planned=view.raw.settings?.budgets?.milestones??[504,1008];
  const low=Math.min(...planned,...points.map(r=>r.milestone)),high=Math.max(...planned,...points.map(r=>r.milestone));
  const {c,x,y}=windowPlot($('windowLearningChart'),Number.isFinite(low)?low:504,Number.isFinite(high)?high:1008,0,100,'Training shot requests charged to each learner','Validation capture %');
  for(const [method,color] of [['adaptive_surrogate','#128b8c'],['sac','#ec7657']]){
    const series=points.filter(r=>r.method===method).sort((a,b)=>a.milestone-b.milestone);
    c.strokeStyle=color;c.fillStyle=color;c.lineWidth=2;c.beginPath();series.forEach((r,i)=>{if(i)c.lineTo(x(r.milestone),y(100*r.stats.rate));else c.moveTo(x(r.milestone),y(100*r.stats.rate));});c.stroke();
    for(const row of series){c.beginPath();c.arc(x(row.milestone),y(100*row.stats.rate),4,0,Math.PI*2);c.fill();}
  }
  if(!points.length){c.fillStyle='#6e8180';c.fillText('Validation checkpoints pending — no result inferred',80,95);}
}
function drawWindowRuns(study) {
  const canvas=$('windowWindows'),rows=(study.observed_windows??[]).filter(r=>Array.isArray(r.points)),bounds=study.settings?.gap_bounds??[.02,.65];
  canvas.height=Math.max(180,Math.min(1600,rows.length*24+55));
  const c=canvas.getContext('2d'),w=canvas.width,x=v=>185+(v-bounds[0])/(bounds[1]-bounds[0])*(w-215);
  c.clearRect(0,0,w,canvas.height);c.fillStyle='#edf0e7';c.fillRect(0,0,w,canvas.height);c.font='10px sans-serif';c.textAlign='left';
  rows.forEach((row,i)=>{const y=23+i*24;c.fillStyle='#496f67';c.fillText(row.context_id??'unknown context',10,y+3);
    c.strokeStyle='#c9d5ca';c.lineWidth=1;c.beginPath();c.moveTo(x(bounds[0]),y);c.lineTo(x(bounds[1]),y);c.stroke();
    for(const run of row.observed_hit_runs??[]){c.strokeStyle='#78b6a2';c.lineWidth=4;c.beginPath();c.moveTo(x(run.low_observed_gap_m),y);c.lineTo(x(run.high_observed_gap_m),y);c.stroke();}
    for(const p of row.points){if(!Number.isFinite(p.gap_m))continue;c.fillStyle=p.native_releases===0?'#8a9990':p.captures===p.attempts&&p.attempts>0?'#128b8c':p.captures>0?'#d49a31':'#ec7657';c.beginPath();c.arc(x(p.gap_m),y,3.5,0,Math.PI*2);c.fill();}
  });
  c.fillStyle='#496f67';c.fillText(rows.length?`Release gap: ${(1000*bounds[0]).toFixed(0)} to ${(1000*bounds[1]).toFixed(0)} mm · sampled points only`:'Observed training gaps pending',185,canvas.height-12);
}
function drawWindowTrace(study,index) {
  const trace=study.trace_examples?.[index],view=windowTrajectoryView(study,trace),canvas=$('windowTrace');
  if(!view){const {c}=windowPlot(canvas,0,1,0,1,'Projected distance (m)','Ball-center height above table (m)');c.fillText('Saved native trajectory pending',80,110);$('windowTraceCaption').textContent=`No trajectory is reconstructed from a prediction or ideal flight model. ${(study.trace_warnings??[]).join(' ')}`;return;}
  const xs=view.points.map(p=>p[0]),zs=view.points.map(p=>p[1]);
  const {c,x,y}=windowPlot(canvas,Math.min(-.1,...xs),Math.max(view.distance+.15,...xs),Math.min(0,...zs),Math.max(.3,...zs)+.05,'Distance along board → bucket axis (m)','Ball-center height above table (m)');
  c.strokeStyle='#a47320';c.lineWidth=2;c.beginPath();c.moveTo(x(view.distance-.1),y(.183));c.lineTo(x(view.distance-.079),y(.008));c.lineTo(x(view.distance+.079),y(.008));c.lineTo(x(view.distance+.1),y(.183));c.stroke();
  c.strokeStyle=trace.route_success?'#128b8c':'#ec7657';c.lineWidth=2;c.beginPath();view.points.forEach((p,i)=>{if(i)c.lineTo(x(p[0]),y(p[1]));else c.moveTo(x(p[0]),y(p[1]));});c.stroke();
  c.fillStyle='#173c3d';const first=view.points[0];c.beginPath();c.arc(x(first[0]),y(first[1]),4,0,Math.PI*2);c.fill();
  $('windowTraceCaption').textContent=`${trace.method??'Native example'} · ${trace.role??'role unreported'} · ${trace.context_id??'Context unknown'} · gap ${Number.isFinite(trace.gap_m)?(trace.gap_m*1000).toFixed(1)+' mm':'unknown'} · ${trace.route_success===true?'captured':trace.route_success===false?'missed':'outcome unknown'} · ${view.points.length} saved samples · maximum off-plane displacement ${(1000*view.lateral).toFixed(1)} mm. Projection does not hide lateral error from the scorer. ${trace.selection_scope??''} ${(study.trace_warnings??[]).join(' ')}`;
}
function showWindowStudy() {
  const view=windowStudyView(report.window_study_evaluation);$('windowStudy').hidden=!view;if(!view)return;
  const study=view.raw,shown=v=>v===null||v===undefined?'—':v,mm=v=>Number.isFinite(v)?(v*1000).toFixed(1):'—';
  $('windowStatus').textContent=`${study.version??'Gap study'} · ${study.status??'pending'} · ${study.phase??'pending'}`;
  $('windowScope').textContent=`${study.scope??study.settings?.scope??'Fixed bucket and board XY; held-out height/tilt contexts only.'} ${study.active_policy??'Previous interactive solver remains active.'}`;
  const coverage=view.boardCoverage,covered=row=>row.total?`${row.resolved} / ${row.total}${row.unknown?' · '+row.unknown+' unknown':''}`:'—';
  const pairs=[['Train / validation / test contexts',`${view.split.join(' / ')} · ${view.disjoint?'disjoint declared contexts':'split identity not verified'}`],
    ['Board IK resolved under current budget · all contexts',covered(coverage.all)],
    ['Board IK resolved · train / validation / test',[coverage.train,coverage.validation,coverage.test].map(covered).join(' · ')],
    ['Fixed bucket XY (mm)',(study.settings?.bucket_xy_m??[]).map(mm).join(' / ')||'—'],['Fixed board XY (mm)',(study.settings?.board_xy_m??[]).map(mm).join(' / ')||'—'],
    ['Native evidence audit',study.audit?.status??'pending']];
  for(const [key,value] of Object.entries(study.counts??{}))if(Number.isFinite(value))pairs.push([key.replaceAll('_',' '),value]);
  addMetrics(pairs,$('windowSummary'));drawWindowCurve(view);drawWindowRuns(study);
  const native=stats=>stats.released===null?'unreported':`${stats.released} / ${shown(stats.requests)}`;
  const reward=stats=>stats.reward===null?'—':stats.reward.toFixed(3);
  expansionTable($('windowCurve'),['Method','Training requests','Validation captures / requests','Native releases / requests','Mean shaped reward'],view.curve.map(r=>[WINDOW_METHODS[r.method],r.milestone,r.stats.fraction,native(r.stats),reward(r.stats)]));
  expansionTable($('windowComparison'),['Frozen method','Captures / all requests','Native releases / requests','Clean','Invalid','Mean shaped reward','95% descriptive interval'],view.test.map(r=>[WINDOW_METHODS[r.name],r.stats.fraction,native(r.stats),shown(r.stats.clean),shown(r.stats.invalid),reward(r.stats),r.stats.interval?r.stats.interval.map(v=>(v*100).toFixed(1)+'%').join(' – '):'—']));
  $('windowUncertainty').textContent='All requests stay in the capture denominator, including geometry rejects. Descriptive binomial intervals do not account for dependence between repeated shots at the same context. This single seeded study does not establish broad state-space coverage or a statistically proven method advantage. The surrogate probability is uncalibrated.';
  const rows=[];for(const row of study.test_contexts??[])for(const [method,value] of Object.entries(row.methods??{}))if(method in WINDOW_METHODS){const nominal=windowStatistics(value.nominal),fresh=windowStatistics(value.fresh),total=windowStatistics(value.total);rows.push([row.context_id,`${mm(row.board_height_m)} / ${shown(row.tilt_deg)}°`,WINDOW_METHODS[method],mm(value.gap_m),nominal.fraction,fresh.fraction,total.fraction,native(total)]);}
  expansionTable($('windowTestContexts'),['Test context','Board H mm / tilt','Method','Gap mm','Nominal captures / requests','Fresh captures / requests','All captures / requests','Native / all'],rows);
  const select=$('windowTraceSelect'),previous=select.value,traces=study.trace_examples??[];
  select.replaceChildren(...traces.map((trace,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=`${trace.method??'native'} · ${trace.role??'role unreported'} · ${trace.context_id??'Unknown'} · ${mm(trace.gap_m)} mm · ${trace.route_success?'capture':'miss'}`;return option;}));
  if(previous!==''&&Number(previous)<traces.length)select.value=previous;select.disabled=!traces.length;select.onchange=()=>drawWindowTrace(study,Number(select.value));drawWindowTrace(study,Number(select.value));
  const checkpoints=Object.entries(study.selected_checkpoints??{}).map(([name,value])=>`${WINDOW_METHODS[name]??name}: ${value?.milestone??value?.training_requests??'checkpoint pending'} training requests`).join('; ');
  $('windowAudit').textContent=`Audit: ${study.audit?.status??'pending'}${study.audit?.reason?' · '+study.audit.reason:''}. Contract: ${study.contract_fingerprint??'unreported'}. ${checkpoints}`;
  const confirmation=windowConfirmationView(study);$('windowConfirmations').hidden=!confirmation;
  if(confirmation){
    $('windowConfirmationStatus').textContent=`${confirmation.status??'pending'} · separate repeatability audit: ${confirmation.audit?.status??'pending'}${Number.isInteger(confirmation.audit?.checks_count)?' · '+confirmation.audit.checks_count+' checks':''}${confirmation.reason?' · '+confirmation.reason:''}`;
    $('windowConfirmationScope').textContent=confirmation.scope??'Gap-only repeatability at frozen TRAIN configurations; no continuous-window or policy-generalization claim.';
    expansionTable($('windowConfirmationTable'),['Recipe','Board mm / °','Gap / release mm','Distance mm','Nominal','Fresh','Total','Native','Clean'],(confirmation.recipes??[]).map(row=>{
      const nominal=windowStatistics(row.nominal),fresh=windowStatistics(row.fresh),total=windowStatistics(row.total);
      return [row.id??'recipe',`${mm(row.board_height_m)} / ${Number.isFinite(row.tilt_deg)?row.tilt_deg.toFixed(2):'—'}°`,`${mm(row.gap_m)} / ${mm(row.release_center_height_m)}`,mm(row.distance_m),nominal.fraction,fresh.fraction,total.fraction,native(total),total.clean===null?'unreported':`${total.clean} / ${shown(total.requests)}`];
    }));
  }
}
// END READ-ONLY WINDOW STUDY
function showCorrectedGrid() {
  const progress = correctedGridProgress(report);
  $('correctedGrid').hidden = !progress;
  if (!progress) return;
  const shown = value => value === null ? '—' : value;
  const display = value => String(value).replaceAll('_',' ');
  $('correctedGridStatus').textContent = `${display(progress.status)} · ${display(progress.phase)}${progress.queue ? ' · Waiting for shared 4090' : ''}`;
  $('correctedGridScope').textContent = `${progress.scope}. ${progress.reason} Contract: ${progress.fingerprint}`;
  const readiness = correctedTrainingReadiness(report);
  $('correctedGridReadinessTitle').textContent = readiness.title;
  $('correctedGridReadiness').textContent = readiness.text;
  const auditText = audit => !audit ? 'pending' : `${audit.status}${Number.isInteger(audit.checks) ? ` · ${audit.checks.toLocaleString()} checks` : ''}`;
  addMetrics([['Declared bucket positions',progress.cells.length],['Geometry requests · including confirmations',shown(progress.geometryRequests)],
    ['Shot test requests',shown(progress.nativeAttempts)],['Actual native releases',shown(progress.nativeReleases)],['Fresh confirmation requests',shown(progress.confirmationAttempts)],
    ['Robust qualified positions',progress.qualified === null ? '—' : `${progress.qualified} / ${progress.cells.length}`],
    ['Qualified settings',shown(progress.settings)],['Positions with multiple qualified choices',shown(progress.multipleChoices)],
    ['Clear shared-start layouts',shown(progress.homeClear)],['Common-start connected positions',shown(progress.connected)],
    ['Native evidence audit',auditText(progress.audits.native)],['Admission coverage audit',auditText(progress.audits.supplemental)]],$('correctedGridSummary'));
  const stages = progress.stages;
  expansionTable($('correctedGridStages'),['Stage','Shot requests','Native releases','Captures'],stages ? [
    ['Search',stages.search],['Nominal repeats',stages.nominal],['Fresh perturbations',stages.confirmation]
  ].map(([name,stage])=>[name,stage.requests,stage.releases,stage.captures]) : []);
  const canvas = $('correctedGridMap'), ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  const px = x => 45 + (x + .23) / 1.8 * (w - 90), py = y => h - 35 - (y + .875) / 1.25 * (h - 60);
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#edf0e7';ctx.fillRect(0,0,w,h);
  ctx.strokeStyle='#bdcfc3';ctx.lineWidth=1;ctx.strokeRect(45,25,w-90,h-60);
  const colors={pending:'#d8dfd7',rejected:'#626e69',home_blocked:'#b9a385',endpoint:'#8bb8ba',missed:'#ec7657',captured:'#d49a31',qualified:'#128b8c'};
  for (const row of progress.cells) {
    const x=px(row.xy[0]),y=py(row.xy[1]);ctx.beginPath();ctx.arc(x,y,9,0,Math.PI*2);ctx.fillStyle=colors[row.state];ctx.fill();
    ctx.fillStyle='#355654';ctx.font='10px sans-serif';ctx.textAlign='center';ctx.fillText(row.id.replace('grid-',''),x,y-14);
  }
  for (const [x,label] of [[0,'10'],[1.33731,'14']]) { ctx.fillStyle='#455f60';ctx.fillRect(px(x)-9,py(0)-9,18,18);ctx.font='bold 10px sans-serif';ctx.fillStyle='#fff';ctx.fillText(label,px(x),py(0)+4); }
  ctx.fillStyle='#607c74';ctx.textAlign='left';ctx.font='11px sans-serif';ctx.fillText('Tabletop map · bucket centers · square markers = arm bases',50,17);
  const fraction=g=>g && Number.isFinite(g.captures) && Number.isFinite(g.attempts) ? `${g.captures}/${g.attempts}`:'—';
  expansionTable($('correctedGridCells'),['Position','Bucket X / Y (mm)','Endpoint options','Search captures / requests','Nominal · fresh per source','Status'],progress.cells.map(row=>[
    row.id,row.xy.map(x=>(1000*x).toFixed(1)).join(' / '),shown(row.eligible),row.attempts===null?'—':row.attempts===0?'No native shot':`${shown(row.captures)} / ${row.attempts}`,
    row.confirmations.map(g=>`${fraction(g.nominal)} · ${fraction(g.fresh)}`).join('; ')||'—',row.qualified?'Robust and connected':row.reason,
  ]));
}
function showContactStudy() {
  const study = contactStudy(report);
  $('contactRepeatability').hidden = !study;
  if (!study) return;
  const shown = value => value === null || value === undefined ? '—' : value;
  const mm = value => Number.isFinite(value) ? (1000 * value).toFixed(2) : '—';
  const fraction = row => row.attempts !== null && row.captures !== null ? `${row.captures} / ${row.attempts}` : '—';
  $('contactStatus').textContent = `${study.version ?? 'Contact study'} · ${study.status ?? 'pending'}`;
  $('contactConclusion').textContent = study.conclusion ?? 'Investigating numerical repeatability before more training. Previous policy remains active.';
  const diagnosis = (study.diagnosis ?? []).filter(value => typeof value === 'string');
  $('contactDiagnosis').replaceChildren(...diagnosis.map(value => { const li = document.createElement('li'); li.textContent = value; return li; }));
  addMetrics([['Previous perturbed confirmation', '13 / 20'], ['New training updates', 0],
    ['New native throws', shown(study.total_attempts)], ['Independent audit', study.verification?.status ?? 'Pending']], $('contactSummary'));
  const groups = contactGroups(report);
  const cells = row => [
    row.label ?? row.id, fraction(row), shown(row.clean), shown(row.invalid),
    Number.isFinite(row.max_joint_excursion_rad) ? `${row.max_joint_excursion_rad.toExponential(2)} rad` : '—',
  ];
  const headers = ['Protocol / group', 'Captures / attempts', 'Clean', 'Invalid', 'Max hinge excursion'];
  const fresh = groups.filter(row => row.stage === 'confirmation' && row.family === 'fresh_perturbation');
  if (fresh.length) expansionTable($('contactFreshGroups'), headers, fresh.map(cells));
  else $('contactFreshGroups').textContent = 'Fresh confirmation is pending.';
  if (groups.length) expansionTable($('contactGroups'), headers, groups.map(cells));
  else $('contactGroups').textContent = 'Controlled contact comparisons are pending.';
  const configurations = contactConfigurations(report);
  $('contactDownload').disabled = !configurations.length;
  // Keep the page compact; the CSV includes every distinct tested pose.
  if (configurations.length) expansionTable($('contactConfigurations'), ['Configuration', 'Board X / Y / H (mm)', 'Tilt', 'Gap / release H (mm)', 'Bucket X / Y (mm)', 'Distance (mm)'], configurations.slice(0, 8).map(row => [
    row.id, row.board_xyz_m.map(mm).join(' / '), `${row.tilt_deg.toFixed(3)}°`,
    `${mm(row.drop_gap_m)} / ${mm(row.release_center_m[2])}`, row.bucket_xy_m.map(mm).join(' / '),
    mm(Math.hypot(row.board_xyz_m[0] - row.bucket_xy_m[0], row.board_xyz_m[1] - row.bucket_xy_m[1])),
  ]));
  else $('contactConfigurations').textContent = 'Exact tested poses will be listed after capture.';
  $('contactDownload').onclick = () => {
    const url = URL.createObjectURL(new Blob([configurationsCsv(report)], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = 'sircus-contact-configurations.csv'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const transit = study.transit ?? {}, routes = transitRoutes(report);
  $('transitScope').textContent = transit.scope ?? 'Transit verification pending; endpoint IK alone does not validate the route.';
  if (routes.length) expansionTable($('contactTransit'), [`Route · first ${Math.min(5, routes.length)} of ${routes.length}`, 'Geometric audit', 'Samples', 'Clearance lower bound', 'Reason / scope'], routes.slice(0,5).map(row => [
    row.id, row.accepted === true ? 'Passed' : row.accepted === false ? 'Rejected' : 'Pending', shown(row.path_samples),
    Number.isFinite(row.min_clearance_mm) ? `${row.min_clearance_mm.toFixed(2)} mm` : '—', row.reason ?? '',
  ]));
  else $('contactTransit').textContent = 'No transit path has passed yet.';
  $('transitPreviews').replaceChildren(...routes.filter(row => routePreview(row).length).map(row => {
    const button = document.createElement('button'); button.className = 'quiet'; button.textContent = `Preview ${row.id}`;
    button.onclick = () => {
      if (pending || physics?.busy) return;
      clearResult('Sampled FK route · geometric audit only');
      transitFrames = routePreview(row); transitTime = transitFrames[0].time; transitName = row.id;
      if (Array.isArray(row.bucket_xy_m)) setBucket(row.bucket_xy_m);
      ball.visible = false;
      $('sceneOutcome').textContent = `Audited transit ${row.id} · geometric preview`;
      $('world').scrollIntoView({ behavior: 'smooth', block: 'center' });
    };
    return button;
  }));
}
function showStableIk() {
  const progress = stableIkProgress(report);
  $('ikStability').hidden = !progress;
  if (!progress) return;
  const shown = value => value === null || value === undefined ? '—' : value;
  const flag = value => value === null ? 'Not yet verified' : value ? 'Passed' : 'Failed';
  $('stableIkStatus').textContent = `IK stability study · ${progress.status} · ${progress.phase}${progress.queued ? ' · Waiting for shared 4090' : ''}`;
  $('stableIkPolicy').textContent = 'Previous policy remains active. This study does not train or deploy a policy.';
  addMetrics([
    ['Selected development targets', progress.targets], ['New qualified settings', `${progress.qualified} / ${progress.targets || '—'}`],
    ['Native search throws', progress.nativeSearchAttempts], ['First confirmation attempts', progress.confirmationAttempts],
    ['Final confirmation attempts', progress.finalConfirmationAttempts], ['CPU geometry proposals', progress.geometryProposals],
    ['Order / batch invariance', flag(progress.invariance)], ['Known valid branch preservation', flag(progress.preservation)],
  ].filter(([, value]) => value !== null), $('stableIkSummary'));
  const audit = stableIkAuditRows(report);
  if (audit.length) expansionTable($('stableIkAudit'), ['Endpoint audit', 'Accepted / queries', 'Order mask changes', 'Batch mask changes', 'Max joint Δ (rad)', 'Saved branches preserved'], audit.map(row => [
    row.label, `${shown(row.accepted)} / ${shown(row.queries)}`, shown(row.orderChanges), shown(row.batchChanges),
    row.maxDelta === null ? '—' : row.maxDelta.toExponential(2), shown(row.preserved),
  ]));
  else $('stableIkAudit').textContent = 'Before / after endpoint audit has not completed.';
  const controls = stableIkControlRows(report);
  if (controls.length) expansionTable($('stableIkControls'), ['Same measured source controls', 'Direct captures / attempts', 'Clean captures', 'Rejected endpoints'], controls.map(row => [
    row.label, `${shown(row.captures)} / ${shown(row.attempts)}`, shown(row.clean), shown(row.invalid),
  ]));
  else $('stableIkControls').textContent = 'Native control repeats have not completed.';
  const targets = stableIkTargets(report);
  const fraction = value => value && Number.isFinite(value.captures) && Number.isFinite(value.attempts) ? `${value.captures} / ${value.attempts}` : '—';
  const confirmationStatus = row => {
    if (row.promoted) return `Qualified · ${row.sourceId}`;
    if (row.multipleFinalGroups) return `Multiple final groups; inspect evidence · ${row.confirmationSourceId}`;
    const final = row.finalConfirmation;
    if (final?.status) return `Final ${final.status} · ${row.confirmationSourceId}`;
    if (row.firstConfirmation?.status === 'failed') return `Screening failed · ${row.confirmationSourceId}`;
    return row.confirmationSourceId ? `${row.reason} · ${row.confirmationSourceId}` : row.reason;
  };
  if (targets.length) expansionTable($('stableIkTargets'), ['Remaining DEV target', 'X / Y (mm)', 'Direct captures / search', 'Assisted entries', 'Best screening / final', 'Status / source'], targets.map(row => [
    row.id, row.xy.map(x => (1000 * x).toFixed(1)).join(' / '), `${shown(row.captures)} / ${shown(row.attempts)}`,
    shown(row.assisted), `${fraction(row.firstConfirmation)} · ${fraction(row.finalConfirmation)}`,
    confirmationStatus(row),
  ]));
  else $('stableIkTargets').textContent = 'The ten remaining development coordinates will appear once pinned to the previous geometry cache.';
}
function showComparison() {
  const rows = comparisonRows(report), points = learningPoints(report);
  if (rows.length) {
    const table = document.createElement('table'), head = document.createElement('thead'), tr = document.createElement('tr');
    for (const label of ['Checkpoint / condition', 'Captures / shots', 'Capture %', 'Invalid', 'Mean reward']) { const th = document.createElement('th'); th.textContent = label; tr.append(th); }
    head.append(tr); table.append(head);
    const body = document.createElement('tbody');
    for (const row of rows) {
      const tr = document.createElement('tr'); if (row.key === 'selected') tr.className = 'selected'; if (row.key.endsWith('_locked')) tr.className = 'locked';
      for (const value of [row.label, `${row.captures} / ${row.attempts}`, row.attempts ? `${(100 * row.captures / row.attempts).toFixed(1)}%` : '—', row.invalid, row.meanReward === null ? '—' : row.meanReward.toFixed(3)]) { const td = document.createElement('td'); td.textContent = String(value); tr.append(td); }
      body.append(tr);
    }
    table.append(body); $('comparisonTable').replaceChildren(table);
  }
  drawChart($('captureChart'), points, 'captures'); drawChart($('rewardChart'), points, 'reward', true);
  $('studyScope').textContent = report.scope || 'Privileged simulator geometry · shot endpoints initialized and held · no learned grasping or transit.';
}
function showCoverage() {
  const rows = coverageRows(report), summary = coverageSummary(report), context = $('coverage').getContext('2d'), width = 720, height = 230;
  context.clearRect(0, 0, width, height); context.fillStyle = '#edf0e7'; context.fillRect(0, 0, width, height);
  const bounds = sceneSpec.bucket_bounds || { x: [-.13, 1.47], y: [-.775, .275] }, px = x => 35 + (x - bounds.x[0]) / (bounds.x[1] - bounds.x[0]) * (width - 70), py = y => height - 28 - (y - bounds.y[0]) / (bounds.y[1] - bounds.y[0]) * (height - 50);
  context.strokeStyle = '#cbd6ca'; context.lineWidth = 1;
  for (let i = 0; i <= 6; i++) { const x = 35 + i * (width - 70) / 6; context.beginPath(); context.moveTo(x, 15); context.lineTo(x, height - 28); context.stroke(); }
  context.strokeStyle = '#8ca99a'; context.setLineDash([4, 4]); context.strokeRect(px(.1), py(.25), px(1.24) - px(.1), py(-.65) - py(.25)); context.setLineDash([]);
  for (const row of rows) {
    context.fillStyle = row.captures > 0 || row.success === true ? '#128b8c' : '#9aaba0'; context.beginPath(); context.arc(px(row.xy[0]), py(row.xy[1]), 5, 0, Math.PI * 2); context.fill();
    if (row.confirmation?.attempts > 0 && row.confirmation.captures === row.confirmation.attempts) { context.beginPath(); context.arc(px(row.xy[0]), py(row.xy[1]), 8, 0, Math.PI * 2); context.strokeStyle = '#d6982d'; context.lineWidth = 2; context.stroke(); }
  }
  context.fillStyle = '#6e8180'; context.font = '12px sans-serif'; context.fillText(rows.length ? 'Exploration grid · dashed box = studied region' : 'Awaiting layout evidence', 20, height - 8);
  $('coverageCount').textContent = rows.length ? `${summary.resolved} / ${summary.total} explored layouts captured` : 'No layout evidence loaded';
  const evaluation = report.frozen_evaluation || {}, pairs = [];
  for (const [key, label] of [['train_episodes', 'Training attempts'], ['updates', 'Policy updates'], ['selected_update', 'Selected update'], ['seed', 'Training seed']]) if (report[key] !== undefined) pairs.push([label, report[key]]);
  if (rows.length) pairs.push(['Resolved explored cells', `${summary.resolved} / ${summary.total}`], ['Unresolved cells', summary.unresolved], ['All confirmation repeats captured', `${summary.confirmed} / ${summary.total}`], ['Total confirmation trials', summary.confirmationTrials]);
  if (evaluation.captures !== undefined) { const attempts = evaluation.trials ?? evaluation.attempts ?? 0; pairs.push(['Frozen test captures', `${evaluation.captures} / ${attempts}`], ['Frozen test capture rate', attempts ? `${(100 * evaluation.captures / attempts).toFixed(1)}%` : '—']); }
  if (evaluation.layouts !== undefined) pairs.push(['Frozen test layouts', evaluation.layouts]);
  if (pairs.length) addMetrics(pairs, $('studySummary'));
  else $('studySummary').textContent = report.summary ?? 'Study metrics will appear when published. Unresolved cells remain visible.';
  showComparison(); showBankComparison(); showExpansion(); showStableIk(); showContactStudy(); showCorrectedGrid(); showWindowStudy();
}
async function loadStatus() {
  try {
    const [s, r] = await Promise.all([fetch('/api/scene', { cache: 'no-store' }).then(v => v.json()), fetch('/api/study', { cache: 'no-store' }).then(v => v.json())]);
    const initial = !sceneSpec.version; sceneSpec = s; report = r;
    const standalone = plannerOnly(r);
    const historicalPlanner = historicalPlannerIsPreV2(r);
    $('plannerHistoricalNotice').hidden = !historicalPlanner;
    $('plannerFarmHeading').textContent = historicalPlanner ? 'Historical farm targets · pre-v2' : "Select a table's bucket position";
    $('plannerGridHeading').textContent = historicalPlanner ? 'Historical shot settings and repeats · pre-v2' : 'Shot settings. Repeated captures.';
    $('historicalStudies').hidden = standalone;
    const benchmarkView = !!sweepRlStudy(r) || typeof r.sweep_rl_error === 'string';
    document.title = benchmarkView ? 'Sircus · Sweep vs RL benchmark' : standalone ? 'Sircus · Sweep shot planner' : 'Sircus · mjlab learning lab';
    $('workspaceHeading').textContent = benchmarkView ? 'Sweep versus RL benchmark' : standalone ? 'Sweep shot planner' : 'Place a target. Find a posture.';
    if (!methodChosen && !physicsSelected() && plannerReady(r)) $('mode').value = 'search';
    if (initial) {
      if (s.bucket?.xy) setBucket(s.bucket.xy);
      if (s.table?.size) { table.scale.set(...s.table.size.map((v, i) => v / [1.8, 1.25, .05][i])); table.position.fromArray(s.table.center); }
      if (s.board?.center) ['X', 'Y', 'Z'].forEach((axis, i) => { $('board' + axis).value = s.board.center[i] * 1000; });
      if (s.release_center) ['X', 'Y', 'Z'].forEach((axis, i) => { $('drop' + axis).value = s.release_center[i] * 1000; });
    }
    updateMethodHelp();
    const ready = physicsSelected() || modeReady(r, $('mode').value), planner = plannerReady(r);
    $('studyTitle').textContent = physicsSelected() ? 'Physics planner · simulation only' : r.benchmark_read_only ? 'Read-only benchmark · simulation only' : planner ? 'Sweep planner ready · simulation only' : ready ? 'Studies complete · frozen evaluations ready' : r.bank_evaluation && r.bank_evaluation.status !== 'completed' ? `Bank study ${r.bank_evaluation.status ?? 'pending'}` : `Study ${String(r.status ?? 'unavailable').replaceAll('_', ' ')}`;
    $('studyDetail').textContent = physicsSelected() ? 'Plan both arm poses, then run one drop.' : r.benchmark_read_only ? 'Read-only benchmark. The scene and recorded grid replays are historical; this page cannot launch training or new solves.' : planner ? 'Corrected native MuJoCo · endpoint-held. Learned inference stays separately gated.' : ready ? 'Manual simulation testing is now available.' : 'Manual solves remain locked until evaluation finishes.';
    $('studyState').querySelector('.status-dot').className = `status-dot ${ready ? 'ready' : ''}`;
    syncSolveButtons();
    $('fingerprint').textContent = `Scene fingerprint: ${r.scene_fingerprint ?? r.contract_fingerprint ?? r.fingerprint ?? s.fingerprint ?? 'not reported'}`;
    if (ready && !physicsSelected() && $('resultTitle').textContent === 'Waiting for the learning study') { $('resultTitle').textContent = 'Ready to test a bucket position'; $('resultText').textContent = 'Choose a bucket position and optional arm locks, then solve.'; }
    showSavedPhysicsReplays();
    showClearanceExpansion();
    showNeuralShotStudy();
    showPlanningBudgetStudy();
    showFastPlanningStudy();
    showInteractiveFastAcceptance();
    showArmCheckSpeed();
    showDirectPathFirst();
    showPlannerFarm();
    showPlannerGridStudy();
    showSweepRlBenchmark();
    if (!standalone) showCoverage();
  } catch (error) { $('studyTitle').textContent = 'Study connection unavailable'; $('studyDetail').textContent = error.message; for (const id of ['solve', 'solveTop']) $(id).disabled = true; if (physicsSelected()) syncSolveButtons(); }
}
function showSavedPhysicsReplays() {
  if (!savedPhysicsRecordings) return;
  const rows = savedPhysicsRecordings(report), signature = JSON.stringify(rows);
  $('savedPhysicsReplays').hidden = rows.length === 0;
  if (signature !== savedReplaySignature) {
    const previous = $('savedReplaySelect').value;
    savedReplayRows = rows; savedReplaySignature = signature;
    $('savedReplaySelect').replaceChildren(...rows.map(row => {
      const option = document.createElement('option'); option.value = row.id; option.textContent = row.label; return option;
    }));
    if (rows.some(row => row.id === previous)) $('savedReplaySelect').value = previous;
    $('savedReplayTitle').textContent = typeof report.interactive_recordings?.title === 'string'
      ? report.interactive_recordings.title : 'Saved first-drop replays';
  }
  syncSolveButtons();
}
function showDirectPathFirst() {
  if (!directPathFirstSummary) return;
  const view = directPathFirstSummary(report);
  $('directPathFirst').hidden = !view;
  if (!view) return;
  const number = value => value === null ? '—' : String(value);
  const seconds = value => value === null ? '—' : `${value.toFixed(1)} s`;
  const phases = { initialized: 'Comparison initialized', preparing: 'Checking both planners at new positions',
    prepared: 'All shot choices saved before testing', waiting_for_4090: 'Waiting for the shared 4090',
    controls_pre: 'Checking simulator controls before drops', test_baseline: 'Testing reference-path-first planning',
    test_direct_first: 'Testing candidate-path-first planning', controls_post: 'Checking final simulator controls',
    diagnostic_baseline: 'Replaying the previous timeout · reference path first',
    diagnostic_direct_first: 'Replaying the previous timeout · candidate path first',
    native_completed: 'Drops finished · verifying saved results', auditing: 'Verifying saved results', completed: 'Drops finished · final verification pending' };
  $('directPathFirstStatus').textContent = view.complete ? 'Comparison complete · independent audit passed'
    : view.status === 'failed' ? 'Comparison stopped · completed evidence preserved'
    : view.status === 'completed' && view.audit === 'passed' && !view.accounting_valid ? 'Reported totals disagree · final results withheld'
    : phases[view.phase] || 'Comparison in progress · final results pending';
  $('directPathFirstProgress').textContent = `Plans completed: ${number(view.prepared)} / ${number(view.planned)} · Actual drops completed: ${number(view.releases)} / ${number(view.planned_releases)}`;
  $('directPathFirstScope').textContent = `${number(view.total)} fresh bucket positions · ${number(view.budget)}-second decision limit for both · same frozen model · at most one drop per method and position.`;
  $('directPathFirstDiagnostic').replaceChildren(); $('directPathFirstDiagnosticNote').textContent = '';
  $('directPathFirstTable').replaceChildren(); $('directPathFirstPositions').replaceChildren();
  $('directPathFirstPair').textContent = ''; $('directPathFirstPositionDetails').hidden = !view.complete;
  const table = (target, headers, rows) => {
    const element = document.createElement('table'), head = document.createElement('thead'), body = document.createElement('tbody'), heading = document.createElement('tr');
    for (const label of headers) { const cell = document.createElement('th'); cell.scope = 'col'; cell.textContent = label; heading.append(cell); }
    head.append(heading);
    for (const cells of rows) {
      const row = document.createElement('tr');
      cells.forEach((value, i) => { const cell = document.createElement(i === 0 ? 'th' : 'td'); if (i === 0) cell.scope = 'row'; cell.textContent = value; row.append(cell); }); body.append(row);
    }
    element.append(head, body); $(target).append(element);
  };
  if (view.complete) {
    const measurements = [
      ['First-drop captures', method => `${method.captures} / ${method.positions}`],
      ['Checked shots released', method => `${method.admitted} / ${method.positions}`],
      ['No checked shot within the limit', method => number(method.no_candidate)],
      ['Released shots that missed', method => number(method.flight_misses)],
      ['Mean planning time', method => seconds(method.timing.mean_s)],
      ['Median planning time', method => seconds(method.timing.median_s)],
      ['90th-percentile planning time', method => seconds(method.timing.p90_s)],
    ];
    table('directPathFirstTable', ['Measurement', ...view.methods.map(method => method.label)],
      measurements.map(([name, value]) => [name, ...view.methods.map(value)]));
    $('directPathFirstPair').textContent = `Candidate path first gained ${view.paired.gained} captures and lost ${view.paired.lost}; both captured at ${view.paired.both} positions. All ${view.total} positions remain in each denominator. Planning time includes worker initialization and result handling, but excludes browser/network time and drop simulation.`;
    const labels = { direct: 'Direct capture', table: 'Table-assisted capture', unclassified_capture: 'Capture · route unclassified',
      setup_rejected: 'No checked shot', flight_miss: 'Released · missed', invalid_physics: 'Invalid physics', arm_assisted_capture: 'Robot-assisted · separate' };
    if (view.diagnostic) {
      table('directPathFirstDiagnostic', ['Previously timed-out target', ...view.diagnostic.methods.map(method => method.label)], [
        ['Checked shot available', ...view.diagnostic.methods.map(method => method.checked_shot ? 'Yes' : 'No')],
        ['Planning time', ...view.diagnostic.methods.map(method => seconds(method.planning_s))],
        ['Drop outcome', ...view.diagnostic.methods.map(method => method.outcome === null ? 'Not run' : labels[method.outcome] || method.outcome.replaceAll('_', ' '))],
      ]);
      $('directPathFirstDiagnosticNote').textContent = `Bucket (${view.diagnostic.bucket_xy_m.map(value => (value * 1000).toFixed(1)).join(', ')}) mm. This previously inspected target is separate from the ${view.total} fresh positions above.`;
    }
    table('directPathFirstPositions', ['Bucket X, Y (mm)', ...view.methods.map(method => method.label)],
      view.positions.map(row => [row.bucket_xy_m.map(value => (value * 1000).toFixed(1)).join(', '),
        ...view.methods.map(method => labels[row.outcomes[method.id]] || row.outcomes[method.id].replaceAll('_', ' '))]));
  }
  $('directPathFirstNext').textContent = view.error || view.next;
}
function showArmCheckSpeed() {
  if (!armCheckSpeedSummary) return;
  const view = armCheckSpeedSummary(report);
  $('armCheckSpeed').hidden = !view;
  if (!view) return;
  const number = value => value === null ? '—' : String(value);
  const seconds = value => value === null ? '—' : `${value.toFixed(1)} s`;
  const phases = { initialized: 'Comparison initialized', preparing: 'Checking both planners at new positions',
    prepared: 'All shot choices saved before testing', waiting_for_4090: 'Waiting for the shared 4090',
    controls_pre: 'Checking simulator controls before drops', test_baseline: 'Testing current arm checks',
    test_optimized: 'Testing faster arm checks', controls_post: 'Checking final simulator controls',
    native_completed: 'Drops finished · verifying saved results', auditing: 'Verifying saved results', completed: 'Drops finished · final verification pending' };
  $('armCheckSpeedStatus').textContent = view.complete ? 'Comparison complete · independent audit passed'
    : view.status === 'failed' ? 'Comparison stopped · completed evidence preserved'
    : view.status === 'completed' && view.audit === 'passed' && !view.accounting_valid ? 'Reported totals disagree · final results withheld'
    : phases[view.phase] || 'Comparison in progress · final results pending';
  $('armCheckSpeedProgress').textContent = `Plans completed: ${number(view.prepared)} / ${number(view.planned)} · Actual drops completed: ${number(view.releases)} / ${number(view.planned_releases)}`;
  $('armCheckSpeedScope').textContent = `${number(view.total)} fresh bucket positions · ${number(view.budget)}-second decision limit for both · same frozen model · at most one drop per method and position.`;
  $('armCheckSpeedTable').replaceChildren(); $('armCheckSpeedPositions').replaceChildren();
  $('armCheckSpeedPair').textContent = ''; $('armCheckSpeedPositionDetails').hidden = !view.complete;
  const table = (target, headers, rows) => {
    const element = document.createElement('table'), head = document.createElement('thead'), body = document.createElement('tbody'), heading = document.createElement('tr');
    for (const label of headers) { const cell = document.createElement('th'); cell.scope = 'col'; cell.textContent = label; heading.append(cell); }
    head.append(heading);
    for (const cells of rows) {
      const row = document.createElement('tr');
      cells.forEach((value, i) => { const cell = document.createElement(i === 0 ? 'th' : 'td'); if (i === 0) cell.scope = 'row'; cell.textContent = value; row.append(cell); }); body.append(row);
    }
    element.append(head, body); $(target).append(element);
  };
  if (view.complete) {
    const measurements = [
      ['First-drop captures', method => `${method.captures} / ${method.positions}`],
      ['Checked shots released', method => `${method.admitted} / ${method.positions}`],
      ['No checked shot within the limit', method => number(method.no_candidate)],
      ['Released shots that missed', method => number(method.flight_misses)],
      ['Mean planning time', method => seconds(method.timing.mean_s)],
      ['Median planning time', method => seconds(method.timing.median_s)],
      ['90th-percentile planning time', method => seconds(method.timing.p90_s)],
    ];
    table('armCheckSpeedTable', ['Measurement', ...view.methods.map(method => method.label)],
      measurements.map(([name, value]) => [name, ...view.methods.map(value)]));
    $('armCheckSpeedPair').textContent = `Faster checks gained ${view.paired.gained} captures and lost ${view.paired.lost}; both captured at ${view.paired.both} positions. All ${view.total} positions remain in each denominator. Planning time includes worker initialization and result handling, but excludes browser/network time and drop simulation.`;
    const labels = { direct: 'Direct capture', table: 'Table-assisted capture', unclassified_capture: 'Capture · route unclassified',
      setup_rejected: 'No checked shot', flight_miss: 'Released · missed', invalid_physics: 'Invalid physics', arm_assisted_capture: 'Robot-assisted · separate' };
    table('armCheckSpeedPositions', ['Bucket X, Y (mm)', ...view.methods.map(method => method.label)],
      view.positions.map(row => [row.bucket_xy_m.map(value => (value * 1000).toFixed(1)).join(', '),
        ...view.methods.map(method => labels[row.outcomes[method.id]] || row.outcomes[method.id].replaceAll('_', ' '))]));
  }
  $('armCheckSpeedNext').textContent = view.error || view.next;
}
function showInteractiveFastAcceptance() {
  if (!interactiveFastAcceptanceSummary) return;
  const view = interactiveFastAcceptanceSummary(report);
  $('interactiveFastAcceptance').hidden = !view;
  if (!view) return;
  const value = n => n === null ? '—' : String(n);
  const seconds = n => n === null ? '—' : `${n.toFixed(1)} s`;
  $('interactiveFastAcceptanceStatus').textContent = view.complete ? 'Live comparison complete · saved results verified'
    : view.status === 'failed' ? `Live comparison stopped · ${value(view.completed)} / ${value(view.planned)} requests finished · saved evidence preserved`
    : view.status === 'completed' && view.audit === 'passed' && !view.accounting_valid ? 'Reported totals disagree · final results withheld'
    : `Live comparison ${view.status || 'pending'} · ${value(view.completed)} / ${value(view.planned)} requests finished · final results pending`;
  $('interactiveFastAcceptanceScope').textContent = view.scope || 'Same fresh bucket placements, one plan and at most one drop per planner. Native simulation at 64 substeps; arms held at planned endpoints.';
  $('interactiveFastAcceptanceTable').replaceChildren();
  $('interactiveFastAcceptancePair').textContent = '';
  if (view.complete) {
    const table = document.createElement('table'), head = document.createElement('thead'), row = document.createElement('tr'), body = document.createElement('tbody');
    for (const label of ['Measurement', ...view.methods.map(method => method.label)]) {
      const cell = document.createElement('th'); cell.scope = 'col'; cell.textContent = label; row.append(cell);
    }
    head.append(row);
    for (const [label, getValue] of [
      ['First-drop captures', method => `${method.captures} / ${method.positions}`],
      ['Measured drops completed', method => `${method.releases} / ${method.positions}`],
      ['No checked setup / failed requests', method => `${method.no_setup} / ${method.failed}`],
      ['Mean wait for plan', method => seconds(method.planning.mean_s)],
      ['90th-percentile wait for plan', method => seconds(method.planning.p90_s)],
      ['Mean server planning time', method => seconds(method.backend_request.mean_s)],
      ['Requests reusing a worker', method => value(method.worker_reused)],
    ]) {
      const tr = document.createElement('tr'), th = document.createElement('th'); th.scope = 'row'; th.textContent = label; tr.append(th);
      for (const method of view.methods) { const td = document.createElement('td'); td.textContent = getValue(method); tr.append(td); }
      body.append(tr);
    }
    table.append(head, body); $('interactiveFastAcceptanceTable').append(table);
    $('interactiveFastAcceptancePair').textContent = `Faster versus original: ${view.paired.gained} gained captures, ${view.paired.lost} lost. Both captured at ${view.paired.both} placements. All ${view.total} placements remain in each capture denominator.`;
  }
  $('interactiveFastAcceptanceNext').textContent = view.gpu_wait_expired
    ? `The GPU wait limit expired; this comparison is no longer running. ${view.methods.map(method => `${method.label}: ${value(method.planning_samples)} measured plan${method.planning_samples === 1 ? '' : 's'}${method.planning_samples > 0 ? `, mean ${seconds(method.planning.mean_s)}` : ''}`).join('; ')}. Capture and matched timing comparisons remain unavailable. Resume the saved test when the 4090 is free.`
    : view.error || view.next;
}
function showFastPlanningStudy() {
  if (!fastPlanningSummary) return;
  const view = fastPlanningSummary(report);
  $('fastPlanningStudy').hidden = !view;
  if (!view) return;
  const phases = { initialized: 'Comparison initialized', preparing: 'Preparing three independent plans per position',
    preparing_test: 'Preparing three independent plans per position', prepared: 'Plans prepared · freezing choices',
    choices_frozen: 'All three methods’ choices frozen', waiting_for_4090: 'Waiting for the shared 4090',
    controls_pre: 'Checking simulator controls before drops', test: 'Testing one drop per method and position',
    test_cold_30: 'Testing shots planned by fresh workers', test_warm_30: 'Testing shots planned by reused workers',
    test_warm_early: 'Testing shots from the earlier-return rule',
    controls_post: 'Checking final simulator controls', native_completed: 'Drops finished · checking saved measurements',
    auditing: 'Verifying saved measurements', completed: 'Drops finished · awaiting audited results' };
  $('fastPlanningStatus').textContent = view.complete ? 'Comparison complete · independent audit passed'
    : view.status === 'failed' ? 'Comparison interrupted · completed evidence is preserved'
    : view.status === 'completed' && view.audit === 'passed' && !view.accounting_valid ? 'Reported totals disagree · final results withheld'
    : phases[view.phase] || 'Comparison in progress · final results pending';
  const number = value => value === null ? '—' : String(value);
  const seconds = value => value === null ? '—' : `${value.toFixed(1)} s`;
  $('fastPlanningLimit').textContent = `${number(view.total)} fresh bucket positions · ${number(view.budget)}-second decision limit · same frozen learned selector · at most one drop per method and position.`;
  const cards = [['Plans prepared', `${number(view.prepared)} / ${number(view.planned)}`, 'Three methods, same positions.'],
    ['Native drops completed', `${number(view.releases)}${view.planned_releases === null ? '' : ` / ${view.planned_releases} planned`}`, 'No checked shot means no release.']];
  $('fastPlanningProgress').replaceChildren(...cards.map(([label, value, detail]) => {
    const card = document.createElement('div'), title = document.createElement('strong'), count = document.createElement('p'), note = document.createElement('p');
    title.textContent = label; count.textContent = value; note.textContent = detail; card.append(title, count, note); return card;
  }));
  $('fastPlanningTable').replaceChildren(); $('fastPlanningProfiles').replaceChildren(); $('fastPlanningPaired').replaceChildren();
  $('fastPlanningProfileDetails').hidden = !view.complete;
  if (view.complete) {
    const matrix = (target, rows) => {
      const table = document.createElement('table'), head = document.createElement('thead'), header = document.createElement('tr'), body = document.createElement('tbody');
      for (const label of ['Measurement', ...view.methods.map(row => row.label)]) {
        const th = document.createElement('th'); th.scope = 'col'; th.textContent = label; header.append(th);
      }
      head.append(header);
      for (const [label, getValue] of rows) {
        const tr = document.createElement('tr'), th = document.createElement('th'); th.scope = 'row'; th.textContent = label; tr.append(th);
        for (const method of view.methods) { const td = document.createElement('td'); td.textContent = String(getValue(method)); tr.append(td); }
        body.append(tr);
      }
      table.append(head, body); $(target).append(table);
    };
    matrix('fastPlanningTable', [
      ['First-drop captures', row => `${row.captures} / ${view.total} (${(100 * row.captures / view.total).toFixed(0)}%)`],
      ['Checked shot ready', row => `${row.releases} / ${view.total}`],
      ['No checked shot', row => row.no_candidate],
      ['Deadline reached', row => row.deadline_reached],
      ['Returned early', row => row.early_return],
      ['Robot-contact shots', row => row.robot_contacts],
      ['Mean decision time', row => seconds(row.decision.mean_s)],
      ['Median decision time', row => seconds(row.decision.median_s)],
      ['90th-percentile decision time', row => seconds(row.decision.p90_s)],
      ['Mean measured planning request', row => seconds(row.request.mean_s)],
      ['Median measured planning request', row => seconds(row.request.median_s)],
      ['90th-percentile measured planning request', row => seconds(row.request.p90_s)],
    ]);
    matrix('fastPlanningProfiles', [
      ...view.methods[0].stages.map(stage => [stage.label, row => {
        const value = row.stages.find(item => item.id === stage.id);
        return `${seconds(value.aggregate_s)}${value.fraction === null ? '' : ` · ${(value.fraction * 100).toFixed(0)}%`}`;
      }]),
      ['Median worker initialization', row => seconds(row.startup.median_s)],
      ['Workers initialized', row => number(row.workers?.cold_starts ?? null)],
      ['Requests reusing a worker', row => number(row.workers?.reused_requests ?? null)],
      ['Replacement workers', row => number(row.workers?.restarts ?? null)],
    ]);
    for (const pair of view.paired.filter(row => row.valid)) {
      const item = document.createElement('li');
      item.textContent = `${pair.alternative} versus ${pair.base}: ${pair.gained} gained captures, ${pair.lost} lost; both captured at ${pair.both} positions.`;
      $('fastPlanningPaired').append(item);
    }
  }
  $('fastPlanningResultNote').textContent = view.complete
    ? 'Captures use all tested positions as the denominator, including missing setups. Deadline counts can include ready shots. The 90th percentile means nine out of ten requests finished within that time. A dash means the measurement is unavailable.'
    : 'Capture and timing comparisons appear only after all drops and the independent audit finish. Partial progress does not identify a winning method.';
  $('fastPlanningNext').textContent = view.next; $('fastPlanningError').textContent = view.error;
}
function showPlanningBudgetStudy() {
  if (!planningBudgetSummary) return;
  const view = planningBudgetSummary(report);
  $('planningBudgetStudy').hidden = !view;
  if (!view) return;
  const phases = { initialized: 'Comparison initialized', preparing: 'Planning under the shared time limit',
    preparing_test: 'Planning under the shared time limit', prepared: 'Plans prepared · freezing final choices',
    choices_frozen: 'Both planners’ choices frozen', waiting_for_4090: 'Waiting for the shared 4090',
    controls_pre: 'Checking simulator controls before drops', test: 'Testing one drop per planner and position',
    controls_post: 'Checking final simulator controls', native_completed: 'Drops finished · checking saved measurements',
    auditing: 'Verifying saved measurements', completed: 'Drops finished · awaiting audited results' };
  $('planningBudgetStatus').textContent = view.complete ? 'Comparison complete · independent audit passed'
    : view.status === 'failed' ? 'Comparison interrupted · completed evidence is preserved'
    : phases[view.phase] || 'Comparison in progress · final results pending';
  const number = value => value === null ? '—' : String(value);
  $('planningBudgetLimit').textContent = view.budget === null ? 'Planning time limit not yet reported.'
    : `${view.budget}-second decision budget per planner and bucket position · ${number(view.total)} fresh positions · at most one drop per planner at each position.`;
  const cards = [
    ['Plans prepared', `${number(view.prepared)} / ${number(view.planned)}`, 'Each position needs one independent plan from each planner.'],
    ['Native drops completed', `${number(view.releases)}${view.planned_releases === null ? '' : ` / ${view.planned_releases} planned`}`, 'Only actual ball releases; missing setups are counted separately.'],
  ];
  $('planningBudgetProgress').replaceChildren(...cards.map(([label, value, detail]) => {
    const card = document.createElement('div'), title = document.createElement('strong'), count = document.createElement('p'), note = document.createElement('p');
    title.textContent = label; count.textContent = value; note.textContent = detail;
    card.append(title, count, note); return card;
  }));
  $('planningBudgetTable').replaceChildren(); $('planningBudgetPaired').textContent = ''; $('planningBudgetTiming').textContent = '';
  if (view.complete) {
    const table = document.createElement('table'), head = document.createElement('thead'), header = document.createElement('tr'), body = document.createElement('tbody');
    for (const label of ['Planner', 'First-drop captures', 'Shot ready', 'Deadline reached', 'No checked shot', 'Robot contacts', 'Median decision time', '90th-percentile decision time', 'Maximum decision time']) {
      const th = document.createElement('th'); th.scope = 'col'; th.textContent = label; header.append(th);
    }
    head.append(header);
    const seconds = value => value === null ? '—' : `${value.toFixed(1)} s`;
    for (const row of view.methods) {
      const tr = document.createElement('tr');
      const values = [row.label, `${row.captures} / ${view.total} (${(100 * row.captures / view.total).toFixed(1)}%)`,
        `${row.releases} / ${view.total}`, row.deadline_reached, row.no_candidate, row.robot_contacts,
        seconds(row.median_s), seconds(row.p90_s), seconds(row.max_s)];
      for (const value of values) { const td = document.createElement('td'); td.textContent = String(value); tr.append(td); }
      body.append(tr);
    }
    table.append(head, body); $('planningBudgetTable').append(table);
    $('planningBudgetPaired').textContent = `The learned selector captured at ${view.paired.gained} positions where the current planner missed, and missed at ${view.paired.lost} positions the current planner captured. Both captured at ${view.paired.both}; neither captured at ${view.paired.neither}.`;
    const lifecycle = view.methods.filter(row => row.startup_median_s !== null || row.end_to_end_median_s !== null);
    if (lifecycle.length) $('planningBudgetTiming').textContent = `Measured process timings: ${lifecycle.map(row => `${row.label} — median startup ${seconds(row.startup_median_s)}, median worker lifecycle ${seconds(row.end_to_end_median_s)}`).join('; ')}. Worker lifecycle runs from startup through worker shutdown. It excludes parent evidence verification and archive writes, is not capped at the decision budget, and is not total UI request latency.`;
  }
  $('planningBudgetResultNote').textContent = view.complete
    ? 'Both frozen planners used the same new positions and the same deadline. The 90th percentile means nine out of ten planning decisions finished within that time. Deadline and no-shot counts can overlap.'
    : 'Final capture and timing results appear after all drops and the independent audit finish. Partial counts do not identify a winning planner.';
  $('planningBudgetNext').textContent = view.next; $('planningBudgetError').textContent = view.error;
}
function showNeuralShotStudy() {
  if (!neuralShotSummary) return;
  const view = neuralShotSummary(report);
  $('neuralShotStudy').hidden = !view;
  if (!view) return;
  const phases = { initialized: 'Study initialized · preparation has not finished',
    preparing: 'Preparing candidate arm poses', preparing_collection: 'Preparing training and validation shots',
    preparing_train: 'Preparing training shots', collection_prepared: 'Training and validation shots prepared',
    train: 'Collecting training drops', training: 'Training the shot selector', fit: 'Training the shot selector',
    validation: 'Collecting drops at separate validation positions', preparing_validation: 'Preparing validation shots',
    waiting_for_4090: 'Waiting for the shared 4090', waiting_for_4090_collection: 'Waiting for the 4090 to collect training and validation drops',
    waiting_for_4090_test: 'Waiting for the 4090 to test the frozen planners',
    controls_pre: 'Checking simulator controls', collection_controls_pre: 'Checking controls before data collection',
    collection_controls_post: 'Checking controls after data collection', collection_native_completed: 'Collection finished · preparing the model',
    model_frozen: 'Model trained and frozen · final testing still pending',
    preparing_test: 'Preparing the frozen comparison', preparing_baseline: 'Preparing current and wider-search test shots',
    preparing_neural: 'Preparing learned-selector test shots', baseline_prepared: 'Current and wider-search test shots prepared',
    neural_prepared: 'Learned-selector test shots prepared', test_frozen: 'All three planners’ test choices frozen',
    choices_frozen: 'All test choices frozen', test: 'Testing one drop per planner and position',
    test_baseline: 'Running current and wider-search test drops', test_neural: 'Running learned-selector test drops',
    controls_post: 'Checking final simulator controls', test_controls_pre: 'Checking controls before final testing',
    test_controls_post: 'Checking controls after final testing', test_native_completed: 'Test drops finished · checking saved measurements',
    auditing: 'Verifying saved measurements', completed: 'Final drops finished; waiting for audited results' };
  $('neuralShotStatus').textContent = view.complete ? 'Comparison complete · independent audit passed'
    : view.status === 'failed' ? 'Study interrupted · completed evidence is preserved'
    : phases[view.phase] || 'Study in progress · final comparison pending';
  const number = value => value === null ? '—' : String(value);
  $('neuralShotProgress').replaceChildren(...view.splits.map(row => {
    const card = document.createElement('div'), title = document.createElement('strong'), positions = document.createElement('p'), drops = document.createElement('p');
    title.textContent = row.label;
    positions.textContent = row.id === 'test' && (row.baseline_prepared !== null || row.neural_prepared !== null)
      ? `Positions prepared: existing planners ${number(row.baseline_prepared)} / ${number(row.positions)} · learned selector ${number(row.neural_prepared)} / ${number(row.positions)}`
      : `Positions prepared: ${number(row.prepared)} / ${number(row.positions)}`;
    drops.textContent = `Drops completed: ${number(row.released)}${row.planned === null ? '' : ` / ${row.planned} planned`}`;
    card.append(title, positions, drops); return card;
  }));
  $('neuralShotBudget').textContent = view.gpu_seconds === null ? ''
    : `4090 reserved time used: ${(view.gpu_seconds / 3600).toFixed(2)} hours${view.gpu_budget_seconds === null ? '' : ` / ${(view.gpu_budget_seconds / 3600).toFixed(1)} hour cap`}. Waiting for the GPU is separate.`;
  $('neuralShotTable').replaceChildren();
  if (view.complete) {
    const table = document.createElement('table'), head = document.createElement('thead'), header = document.createElement('tr'), body = document.createElement('tbody');
    for (const label of ['Planner', 'First-drop captures', 'Feasible setups', 'Robot contacts', 'Median plan time', '90th-percentile plan time']) {
      const th = document.createElement('th'); th.scope = 'col'; th.textContent = label; header.append(th);
    }
    head.append(header);
    const seconds = value => value === null ? '—' : `${value.toFixed(1)} s`;
    for (const row of view.methods) {
      const tr = document.createElement('tr');
      const values = [row.label, `${row.captures} / ${view.total} (${(100 * row.captures / view.total).toFixed(1)}%)`,
        `${row.admitted} / ${view.total}`, row.robot_contacts, seconds(row.median_s), seconds(row.p90_s)];
      for (const value of values) { const td = document.createElement('td'); td.textContent = String(value); tr.append(td); }
      body.append(tr);
    }
    table.append(head, body); $('neuralShotTable').append(table);
  }
  $('neuralShotResultNote').textContent = view.complete
    ? 'All three planners used the same new bucket positions and at most one release per position. The 90th-percentile time means nine out of ten plans finished within that time.'
    : 'Final capture and timing comparison appears after testing and the independent audit finish. No winning model has been selected.';
  $('neuralShotNext').textContent = view.next;
  $('neuralShotError').textContent = view.error;
}
function showClearanceExpansion() {
  if (!clearanceExpansionSummary) return;
  const view = clearanceExpansionSummary(report);
  $('clearanceExpansionStudy').hidden = !view;
  if (!view) return;
  const phases = { preparing: 'Checking candidate poses and paths', waiting_for_4090: 'Waiting for the shared 4090',
    controls_pre: 'Checking simulator controls', choices_frozen: 'Both planners frozen for testing',
    test: 'Testing one drop per planner', controls_post: 'Checking final simulator controls',
    auditing: 'Verifying saved measurements', completed: 'Checking final measurements' };
  const state = view.status === 'failed' ? 'Comparison interrupted; final results pending'
    : view.complete ? 'Comparison complete'
    : phases[view.phase] || 'Comparison in progress; final results pending';
  $('clearanceExpansionStatus').textContent = state + (view.audit === 'passed' ? ' · independent audit passed' : '');
  const number = value => value === null ? '—' : String(value);
  $('clearanceExpansionProgress').textContent = `Prepared positions ${number(view.prepared)} / ${number(view.total)} · Completed releases ${number(view.completed)} / ${number(view.planned)}`;
  $('clearanceExpansionTable').replaceChildren();
  if (view.complete) {
    const table = document.createElement('table'), head = document.createElement('thead'), header = document.createElement('tr'), body = document.createElement('tbody');
    for (const label of ['Planner', 'First-drop captures', 'Feasible setups', 'Unresolved', 'Released misses', 'Robot contacts']) {
      const th = document.createElement('th'); th.scope = 'col'; th.textContent = label; header.append(th);
    }
    head.append(header);
    for (const row of view.methods) {
      const tr = document.createElement('tr');
      const values = [row.label, `${row.captures} / ${view.total} (${(100 * row.captures / view.total).toFixed(0)}%)`,
        `${row.admitted} / ${view.total}`, row.unresolved, row.flight_misses, row.robot_contacts];
      for (const value of values) { const td = document.createElement('td'); td.textContent = String(value); tr.append(td); }
      body.append(tr);
    }
    table.append(head, body); $('clearanceExpansionTable').append(table);
  }
  $('clearanceExpansionPaired').textContent = view.complete && view.paired
    ? `On the same positions, wider search gained ${view.paired.gained} captures and lost ${view.paired.lost}. Both captured at ${view.paired.both}; neither captured at ${view.paired.neither}.`
    : 'Paired capture results remain pending.';
  const timed = view.methods.filter(row => row.median_s !== null);
  $('clearanceExpansionTiming').textContent = timed.length
    ? 'CPU planning time, median · ' + timed.map(row => `${row.label} ${row.median_s.toFixed(1)} s`).join(' · ') : '';
  $('clearanceExpansionScope').textContent = view.scope;
  $('clearanceExpansionNext').textContent = view.next;
}
function replaySavedPhysics(row) {
  if (pending || physics?.busy || !row || !physicsViewerResult) return;
  clearResult('Loading saved native shot');
  $('boardLock').checked = false; $('dropLock').checked = false;
  setBucket(row.result.bucket_xy_m.slice());
  $('speed').value = '0.25';
  applyResult(physicsViewerResult(row.result, row.id), { recordedPlayback: true, physicsResult: true });
  physicsSceneActive = true; physicsViewKey = `recorded:${row.id}`;
  $('savedReplayExplanation').textContent = `${row.label}. ${row.explanation} Recorded at 64 substeps. Playback is ¼ speed.`;
  $('resultText').textContent = row.explanation || 'Replay of the saved native simulator trajectory.';
  const markers = row.markers.map(marker => {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'quiet';
    button.textContent = `${marker.time.toFixed(3)} s · ${marker.label}`;
    button.onclick = () => { if (!pending && !physics?.busy && physicsViewKey === `recorded:${row.id}`) { playTime = marker.time; playing = false; } };
    return button;
  });
  $('savedReplayMarkers').replaceChildren(...markers);
  syncSolveButtons();
}
$('savedReplayLoad').onclick = () => replaySavedPhysics(savedReplayRows.find(row => row.id === $('savedReplaySelect').value));
function updatePhysics(snapshot) {
  if (physics.available) {
    $('physicsOption').hidden = false; $('physicsOption').disabled = false;
    if (!methodChosen) { $('mode').value = 'physics'; methodChosen = true; }
  }
  $('physicsStatus').textContent = snapshot.message || snapshot.progress?.message || ({ submitting: 'Submitting shot plan…', queued: 'Waiting to plan…', planning: 'Planning both arm poses…', submitting_drop: 'Starting one drop…', running: 'Running one simulated drop…', failed: 'Physics request failed. Plan again to retry.', stale: 'Scene changed. Plan a new shot.', idle: 'Place the bucket, then plan a shot.', ready: 'Preview ready. Run one drop when ready.', no_setup: 'No feasible setup found within this plan.', completed: 'Drop complete. Replay the measured flight, or plan another shot.', expired: 'Plan expired. Plan a new shot.' })[snapshot.state] || 'Physics status unavailable.';
  $('physicsProgress').textContent = physicsProgress(snapshot);
  const options = physics.plannerOptions, select = $('physicsPlanner');
  if ([...select.options].map(option => option.value).join(',') !== options.choices.map(option => option.id).join(',')) {
    select.replaceChildren(...options.choices.map(choice => {
      const option = document.createElement('option'); option.value = choice.id; option.textContent = choice.label; return option;
    }));
  }
  select.value = physics.planner;
  $('physicsPlannerHelp').textContent = physics.planner === 'fast'
    ? 'Returns a checked shot early when possible. The ~30 s decision budget excludes worker startup, queue waits and result checks. No automatic fallback.'
    : options.advertised ? 'Original wider search. It can take longer. It runs only when you select it and press Plan shot.'
    : 'This server offers the original planner. Faster planning is not configured here.';

  if (physicsSelected()) {
    if (snapshot.state === 'idle' && $('resultTitle').textContent === 'Waiting for the learning study') {
      $('resultTitle').textContent = 'Ready to plan a shot';
      $('resultText').textContent = 'Place the bucket, plan both arm poses, then run one simulated drop.';
    }
    const key = `${snapshot.id}:${snapshot.state}`;
    if (key !== physicsViewKey && ['ready', 'completed'].includes(snapshot.state)) {
      clearResult(snapshot.state === 'ready' ? 'Pose preview' : 'Measured drop', true);
      applyResult(physicsViewerResult(snapshot.state === 'ready' ? snapshot.preview : snapshot.result, snapshot.id), { physicsPreview: snapshot.state === 'ready', physicsResult: snapshot.state === 'completed' });
      physicsSceneActive = true; physicsViewKey = key;
    } else if (!['ready', 'completed', 'stale', 'unavailable', 'idle'].includes(snapshot.state)) {
      $('sceneOutcome').textContent = $('physicsStatus').textContent;
      const stopped = ['failed', 'expired', 'no_setup'].includes(snapshot.state);
      $('sceneOutcome').className = `scene-outcome ${stopped ? 'failed' : ''}`;
      if (stopped) {
        $('resultTitle').textContent = snapshot.state === 'no_setup' ? 'No feasible shot' : 'Simulation request unavailable';
        $('resultText').textContent = $('physicsStatus').textContent;
        $('resultDot').className = 'status-dot failed';
      }
    }
  }
  if (physicsSelected()) {
    $('studyTitle').textContent = 'Physics planner · simulation only';
    $('studyDetail').textContent = 'Plan both arm poses, then run one drop.';
    $('studyState').querySelector('.status-dot').className = 'status-dot ready';
  }
  updateConstraintPreview(); syncSolveButtons();
}
async function initializePhysics() {
  try {
    // Older servers may serve this viewer but not the optional control module.
    const response = await fetch('/api/interactive-physics', { cache: 'no-store' });
    if (!response.ok || (await response.json()).enabled !== true) return;
    const module = await import('/interactive-physics.mjs');
    physicsViewerResult = module.physicsViewerResult; physicsProgress = module.physicsProgress;
    savedPhysicsRecordings = module.savedPhysicsRecordings;
    clearanceExpansionSummary = module.clearanceExpansionSummary;
    neuralShotSummary = module.neuralShotSummary;
    planningBudgetSummary = module.planningBudgetSummary;
    fastPlanningSummary = module.fastPlanningSummary;
    interactiveFastAcceptanceSummary = module.interactiveFastAcceptanceSummary;
    armCheckSpeedSummary = module.armCheckSpeedSummary;
    directPathFirstSummary = module.directPathFirstSummary;
    physics = new module.InteractivePhysics({ onChange: updatePhysics });
    await physics.initialize();
    showSavedPhysicsReplays();
    showClearanceExpansion();
    showNeuralShotStudy();
    showPlanningBudgetStudy();
    showFastPlanningStudy();
    showInteractiveFastAcceptance();
    showArmCheckSpeed();
    showDirectPathFirst();
  } catch { /* Historical viewing stays available without this optional API. */ }
}
for (const id of ['physicsDrop', 'physicsDropTop']) {
  $(id).onclick = () => { if (!pending && physicsSelected()) void physics.drop(); };
}
$('physicsClear').onclick = () => { if (!pending && !physics?.busy) clearResult('Preview cleared — plan a new shot'); };
$('physicsPlanner').onchange = () => {
  if (pending || !physics?.selectPlanner($('physicsPlanner').value)) { if (physics) $('physicsPlanner').value = physics.planner; return; }
  clearResult('Planner changed — plan again', true);
  syncSolveButtons();
};
void initializePhysics();
loadStatus(); setInterval(loadStatus, 5000);
const resize = new ResizeObserver(() => { const r = canvas.getBoundingClientRect(); renderer.setSize(r.width, r.height, false); camera.aspect = r.width / r.height; camera.updateProjectionMatrix(); }); resize.observe(canvas);
let last = performance.now();
function animate(now) {
  requestAnimationFrame(animate); const dt = Math.min(.1, (now - last) / 1000); last = now; controls.update();
  if (currentTrajectory.length) {
    const start = currentTrajectory[0].time, end = currentTrajectory.at(-1).time;
    if (playing) playTime = Math.min(end, playTime + dt * Number($('speed').value));
    if (playTime >= end) playing = false;
    let i = 0; while (i < currentTrajectory.length - 2 && currentTrajectory[i + 1].time < playTime) i++;
    const a = currentTrajectory[i], b = currentTrajectory[Math.min(i + 1, currentTrajectory.length - 1)], f = b.time === a.time ? 0 : THREE.MathUtils.clamp((playTime - a.time) / (b.time - a.time), 0, 1);
    ball.position.fromArray(a.ball).lerp(new THREE.Vector3(...b.ball), f);
    $('timeline').value = end === start ? 0 : (playTime - start) / (end - start) * 1000;
    $('replayTime').textContent = `${playTime.toFixed(2)} s`; $('play').textContent = playing ? 'Ⅱ Pause' : '▶ Replay';
  }
  if (transitFrames.length) {
    transitTime = Math.min(transitFrames.at(-1).time, transitTime + dt);
    let i = 0; while (i < transitFrames.length - 1 && transitFrames[i + 1].time <= transitTime) i++;
    const frame = transitFrames[i];
    board.position.fromArray(frame.board_center);
    if (finiteVector(frame.ball_center_m, 3)) { ball.visible = true; ball.position.fromArray(frame.ball_center_m); }
    const q = frame.board_quaternion_wxyz; board.quaternion.set(q[1], q[2], q[3], q[0]);
    drawArms(frame);
    $('geometryLabel').textContent = `Sampled native FK · ${frame.phase ?? transitName} · geometric audit only`;
    $('replayTime').textContent = `${transitTime.toFixed(2)} s`;
    if (transitTime >= transitFrames.at(-1).time) { transitFrames = []; ball.visible = true; }
  }
  renderer.render(world, camera);
}
requestAnimationFrame(animate);
