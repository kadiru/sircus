import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';
import { contactStudy, contactGroups, contactConfigurations, transitRoutes, routePreview, configurationsCsv } from '/state.mjs';
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
function clearResult(message) {
  $('sceneOutcome').textContent = message; $('sceneOutcome').className = 'scene-outcome';
  transitFrames = []; ball.visible = true;
  playing = false; currentTrajectory = []; $('play').disabled = true; $('timeline').disabled = true;
  if (pathLine) { world.remove(pathLine); pathLine.geometry.dispose(); pathLine.material.dispose(); pathLine = null; }
  $('resultTitle').textContent = message;
  $('resultText').textContent = 'A new solve is required after changing the scene or arm constraints.';
  $('resultDot').className = 'status-dot'; $('metrics').replaceChildren();
}
const raycaster = new THREE.Raycaster(), mouse = new THREE.Vector2(), dragPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), dragPoint = new THREE.Vector3();
let dragging = false;
function pointerRay(event) {
  const rect = canvas.getBoundingClientRect();
  mouse.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
  raycaster.setFromCamera(mouse, camera);
}
canvas.addEventListener('pointerdown', event => {
  if (pending) return;
  pointerRay(event);
  if (raycaster.intersectObjects(bucketGroup.children).length) {
    dragging = true; controls.enabled = false; canvas.setPointerCapture(event.pointerId); event.preventDefault();
  }
});
canvas.addEventListener('pointermove', event => {
  if (!dragging) return;
  pointerRay(event);
  if (raycaster.ray.intersectPlane(dragPlane, dragPoint)) {
    const bounds = sceneSpec.bucket_bounds || { x: [-.13, 1.47], y: [-.775, .275] };
    setBucket([THREE.MathUtils.clamp(dragPoint.x, ...bounds.x), THREE.MathUtils.clamp(dragPoint.y, ...bounds.y)]);
    clearResult('Bucket moved — ready for a new solve');
  }
});
for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(name, () => { dragging = false; controls.enabled = true; });
function number(id) { const value = $(id).value.trim(); return value === '' ? NaN : Number(value) / 1000; }
function vector(prefix) { return ['X', 'Y', 'Z'].map(axis => number(prefix + axis)); }
function updateConstraintPreview() {
  updateMethodHelp();
  $('boardFields').classList.toggle('inactive', !$('boardLock').checked);
  $('dropFields').classList.toggle('inactive', !$('dropLock').checked);
  for (const prefix of ['board', 'drop']) for (const axis of ['X', 'Y', 'Z']) $(prefix + axis).disabled = pending || !$(prefix + 'Lock').checked;
  if ($('boardLock').checked && finiteVector(vector('board'), 3)) board.position.fromArray(vector('board'));
  releaseMarker.visible = $('dropLock').checked;
  if (releaseMarker.visible && finiteVector(vector('drop'), 3)) releaseMarker.position.fromArray(vector('drop'));
}
for (const id of ['bucketX', 'bucketY']) $(id).addEventListener('change', () => { if (Number.isFinite(number('bucketX')) && Number.isFinite(number('bucketY'))) setBucket([number('bucketX'), number('bucketY')], false); clearResult('Bucket changed — ready for a new solve'); });
for (const id of ['boardLock', 'dropLock', 'boardX', 'boardY', 'boardZ', 'dropX', 'dropY', 'dropZ']) $(id).addEventListener('change', () => { updateConstraintPreview(); clearResult('Arm constraints changed'); });
updateConstraintPreview();
function updateMethodHelp() { $('methodHelp').textContent = $('mode').value === 'policy' ? policyRouteLabel(report, $('boardLock').checked, $('dropLock').checked) : 'Explicit per-scene search. This is measured separately from frozen-policy performance.'; }
$('mode').onchange = updateMethodHelp;
function syncSolveButtons() {
  const state = solveButtonState(report, pending);
  for (const id of ['solve', 'solveTop']) $(id).disabled = state.disabled;
  $('solve').firstChild.textContent = state.mainLabel;
  $('solveTop').textContent = state.topLabel;
}
function setPending(value) {
  pending = value; syncSolveButtons();
  for (const id of ['bucketX', 'bucketY', 'boardLock', 'dropLock', 'mode']) $(id).disabled = value;
  updateConstraintPreview();
}
function addMetrics(pairs, target) {
  const dl = document.createElement('dl');
  for (const [key, value] of pairs) { const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = key; dd.textContent = String(value); dl.append(dt, dd); }
  target.replaceChildren(dl);
}
function applyResult(result) {
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
  $('sceneOutcome').textContent = success ? '✓ Qualified capture · native verification' : 'No qualified capture · see result';
  $('sceneOutcome').className = `scene-outcome ${success ? 'ready' : 'failed'}`;
  $('resultDot').className = `status-dot ${success ? 'ready' : 'failed'}`;
  $('resultTitle').textContent = success ? 'Qualified capture in simulation' : 'No qualified capture for this request';
  const provenance = methodLabel(result.method);
  $('resultText').textContent = `${provenance}. ${result.message ?? result.reason ?? 'Board bounce, whole-ball entry and one continuous second retained are required.'}`;
  const metrics = result.metrics || {}, pairs = [['Method', provenance], ['Request', result.request_id?.slice(0, 10) ?? '—']];
  if (finiteVector(b, 3)) pairs.push(['Board center Z', `${(b[2] * 1000).toFixed(0)} mm`]);
  if (finiteVector(release, 3)) pairs.push(['Release center Z', `${(release[2] * 1000).toFixed(0)} mm`]);
  for (const [key, label] of [['tilt_deg', 'Board tilt'], ['board_bounces', 'Board bounces'], ['retention_s', 'Retained'], ['quality', 'Entry quality'], ['attempts', 'Native attempts'], ['native_candidate_count', 'Native candidate count'], ['eligible_candidates', 'Eligible candidates']]) if (metrics[key] !== undefined) pairs.push([label, formatMetric(key, metrics[key])]);
  addMetrics(pairs, $('metrics'));
}
async function requestSolve() {
  if (!studyReady(report) || pending) return;
  try {
    const payload = buildRequest($('mode').value, bucketXY, $('boardLock').checked, vector('board'), $('dropLock').checked, vector('drop'), crypto.randomUUID());
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
    if (data.status === 'running') { setTimeout(pollSolution, 700); return; }
    if (data.status === 'error') throw Error(data.error || 'Simulation failed');
    if (data.status !== 'completed') throw Error('Unexpected simulation state');
    applyResult(data.result); setPending(false);
  } catch (error) { setPending(false); $('sceneOutcome').textContent = 'Simulation unavailable'; $('resultTitle').textContent = 'Solve did not complete'; $('resultText').textContent = friendlyError(error.message); $('resultDot').className = 'status-dot failed'; }
}
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
      if (pending) return;
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
  showComparison(); showBankComparison(); showExpansion(); showStableIk(); showContactStudy();
}
async function loadStatus() {
  try {
    const [s, r] = await Promise.all([fetch('/api/scene', { cache: 'no-store' }).then(v => v.json()), fetch('/api/study', { cache: 'no-store' }).then(v => v.json())]);
    const initial = !sceneSpec.version; sceneSpec = s; report = r;
    if (initial) {
      if (s.bucket?.xy) setBucket(s.bucket.xy);
      if (s.table?.size) { table.scale.set(...s.table.size.map((v, i) => v / [1.8, 1.25, .05][i])); table.position.fromArray(s.table.center); }
      if (s.board?.center) ['X', 'Y', 'Z'].forEach((axis, i) => { $('board' + axis).value = s.board.center[i] * 1000; });
      if (s.release_center) ['X', 'Y', 'Z'].forEach((axis, i) => { $('drop' + axis).value = s.release_center[i] * 1000; });
    }
    updateMethodHelp();
    const ready = studyReady(r);
    $('studyTitle').textContent = ready ? 'Studies complete · frozen evaluations ready' : r.bank_evaluation && r.bank_evaluation.status !== 'completed' ? `Bank study ${r.bank_evaluation.status ?? 'pending'}` : `Study ${String(r.status ?? 'unavailable').replaceAll('_', ' ')}`;
    $('studyDetail').textContent = ready ? 'Manual simulation testing is now available.' : 'Manual solves remain locked until evaluation finishes.';
    $('studyState').querySelector('.status-dot').className = `status-dot ${ready ? 'ready' : ''}`;
    syncSolveButtons();
    $('fingerprint').textContent = `Scene fingerprint: ${r.scene_fingerprint ?? r.contract_fingerprint ?? r.fingerprint ?? s.fingerprint ?? 'not reported'}`;
    if (ready && $('resultTitle').textContent === 'Waiting for the learning study') { $('resultTitle').textContent = 'Ready to test a bucket position'; $('resultText').textContent = 'Choose a bucket position and optional arm locks, then solve.'; }
    showCoverage();
  } catch (error) { $('studyTitle').textContent = 'Study connection unavailable'; $('studyDetail').textContent = error.message; for (const id of ['solve', 'solveTop']) $(id).disabled = true; }
}
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
