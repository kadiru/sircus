import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';

// This viewer only composes cameras around an immutable, exported native scene.
const byId = (id) => document.getElementById(id);
const viewport = byId('viewport');
const loading = byId('loading');
const loadingText = byId('loadingText');
const presetSelect = byId('preset');
const enabledButtons = ['resetButton', 'fitButton', 'focusButton', 'applyButton', 'saveButton', 'copyButton'];
const fields = ['eyeX', 'eyeY', 'eyeZ', 'targetX', 'targetY', 'targetZ'];
let data, scene, camera, controls, renderer, farmBounds, cameraKey, defaultView;
let persistTimer, framePending = false;
let dragMode = 'orbit';
const defaultPresetId = '__selected_view06__';
const geometries = new Map();

function fail(error) {
  console.error(error);
  loading.hidden = true;
  const card = byId('webglError');
  card.textContent = `Unable to display the camera scout.\n${error.message || error}\nNo simulation or hardware state was changed.`;
  card.hidden = false;
  byId('sceneStatus').textContent = 'Scene unavailable';
  enabledButtons.forEach((id) => { byId(id).disabled = true; });
}

function finiteVector(value) {
  return Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
}

function parseBinarySTL(buffer) {
  if (buffer.byteLength < 84) throw new Error('Truncated STL mesh.');
  const source = new DataView(buffer);
  const triangles = source.getUint32(80, true);
  if (triangles > 5000000 || buffer.byteLength !== 84 + triangles * 50) {
    throw new Error('Expected an original binary STL mesh.');
  }
  const positions = new Float32Array(triangles * 9);
  const normals = new Float32Array(triangles * 9);
  for (let i = 0; i < triangles; i += 1) {
    const offset = 84 + i * 50;
    for (let vertex = 0; vertex < 3; vertex += 1) {
      for (let axis = 0; axis < 3; axis += 1) {
        const out = i * 9 + vertex * 3 + axis;
        positions[out] = source.getFloat32(offset + 12 + vertex * 12 + axis * 4, true);
        normals[out] = source.getFloat32(offset + axis * 4, true);
        if (!Number.isFinite(positions[out]) || !Number.isFinite(normals[out])) {
          throw new Error('Non-finite values in an STL mesh.');
        }
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.computeBoundingSphere();
  return geometry;
}

function bucketGeometry(spec) {
  const positions = [];
  const segments = spec.segments || 64;
  const height = spec.height_m;
  const bottom = spec.bottom_thickness_m;
  // The native asset has an open rim, individual straight-sided wall wedges,
  // interior bottom at bottom_thickness, and its local outer bottom at Z=0.
  const radii = [[spec.inner_bottom_diameter_m / 2, bottom],
    [spec.outer_bottom_diameter_m / 2, 0],
    [spec.inner_top_diameter_m / 2, height],
    [spec.outer_top_diameter_m / 2, height]];
  const faces = [[0, 1, 5, 4], [2, 6, 7, 3], [0, 2, 3, 1],
    [4, 5, 7, 6], [0, 4, 6, 2], [1, 3, 7, 5]];
  for (let i = 0; i < segments; i += 1) {
    const a = i * Math.PI * 2 / segments;
    const b = (i + 1) * Math.PI * 2 / segments;
    const vertices = radii.flatMap(([r, z]) => [[r * Math.cos(a), r * Math.sin(a), z], [r * Math.cos(b), r * Math.sin(b), z]]);
    for (const [v0, v1, v2, v3] of faces) {
      for (const index of [v0, v1, v2, v0, v2, v3]) positions.push(...vertices[index]);
    }
  }
  const base = new THREE.CylinderGeometry(spec.outer_bottom_diameter_m / 2,
    spec.outer_bottom_diameter_m / 2, bottom, segments, 1, false);
  base.rotateX(Math.PI / 2);
  base.translate(0, 0, bottom / 2);
  const expandedBase = base.toNonIndexed();
  positions.push(...expandedBase.getAttribute('position').array);
  base.dispose();
  expandedBase.dispose();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

async function buildGeometry(spec) {
  if (spec.type === 'stl') {
    const response = await fetch(spec.url);
    if (!response.ok) throw new Error(`Mesh unavailable: ${spec.url} (${response.status})`);
    return parseBinarySTL(await response.arrayBuffer());
  }
  if (spec.type === 'box') return new THREE.BoxGeometry(...spec.size_m);
  if (spec.type === 'sphere') return new THREE.SphereGeometry(spec.radius_m, 32, 20);
  if (spec.type === 'bucket') return bucketGeometry(spec);
  throw new Error(`Unsupported visual geometry: ${spec.type}`);
}

function installInstances() {
  const groups = new Map();
  for (const instance of data.instances) {
    if (!geometries.has(instance.geometry)) throw new Error(`Unknown geometry: ${instance.geometry}`);
    if (!Array.isArray(instance.matrix) || instance.matrix.length !== 16 || !instance.matrix.every(Number.isFinite)) {
      throw new Error('Invalid exported world matrix.');
    }
    // Display-only match to the selected native gray board; no geometry changes.
    const color = instance.semantic === 'board' ? [0.15, 0.17, 0.19] : (instance.color || [0.4, 0.44, 0.47]);
    const key = `${instance.geometry}:${color.join(',')}:${instance.semantic === 'table' ? 'table' : 'object'}`;
    if (!groups.has(key)) groups.set(key, { geometry: instance.geometry, color, instances: [] });
    groups.get(key).instances.push(instance);
  }
  farmBounds = new THREE.Box3();
  const matrix = new THREE.Matrix4();
  for (const group of groups.values()) {
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setRGB(...group.color, THREE.LinearSRGBColorSpace),
      roughness: 0.72, metalness: 0.04, side: THREE.DoubleSide,
    });
    const mesh = new THREE.InstancedMesh(geometries.get(group.geometry), material, group.instances.length);
    mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    group.instances.forEach((instance, index) => {
      matrix.fromArray(instance.matrix); // Exporter supplies Three-compatible column-major data.
      mesh.setMatrixAt(index, matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingBox();
    mesh.computeBoundingSphere();
    mesh.frustumCulled = true;
    scene.add(mesh);
    farmBounds.union(mesh.boundingBox);
  }
  if (farmBounds.isEmpty()) throw new Error('The snapshot contains no visible objects.');
  const floorColor = data.floor?.color || [0.22, 0.23, 0.24];
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(240, 240),
    new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(...floorColor, THREE.LinearSRGBColorSpace), roughness: 0.88, metalness: 0 }));
  const center = farmBounds.getCenter(new THREE.Vector3());
  floor.position.set(center.x, center.y, data.floor?.z_m ?? -0.775);
  scene.add(floor);
}

function cameraPayload() {
  return {
    schema: 'sircus-camera-scout-camera-v1',
    source_fingerprint: data.source_fingerprint,
    scope: 'Camera only; static saved poses; validate with native Isaac render',
    eye_m: camera.position.toArray(), target_m: controls.target.toArray(), up_axis: 'Z',
    focal_length_mm: data.default_camera.focal_length_mm,
    horizontal_aperture_mm: data.default_camera.horizontal_aperture_mm,
    vertical_aperture_mm: data.default_camera.vertical_aperture_mm,
    resolution: [1920, 1080],
  };
}

function updateFields() {
  const values = [...camera.position.toArray(), ...controls.target.toArray()];
  fields.forEach((id, i) => { if (document.activeElement !== byId(id)) byId(id).value = values[i].toFixed(4); });
  byId('cameraReadout').textContent = `Height above tabletop: ${camera.position.z.toFixed(2)} m · Aim distance: ${camera.position.distanceTo(controls.target).toFixed(2)} m`;
}

function requestFrame() {
  if (framePending) return;
  framePending = true;
  requestAnimationFrame(() => {
    framePending = false;
    if (!renderer || !camera) return;
    const changing = controls.update();
    renderer.render(scene, camera);
    if (changing) requestFrame();
  });
}

function changed() {
  updateFields();
  requestFrame();
  clearTimeout(persistTimer);
  persistTimer = setTimeout(() => {
    try { localStorage.setItem(cameraKey, JSON.stringify(cameraPayload())); } catch { /* Browsing remains available without storage. */ }
  }, 180);
}

function applyView(view) {
  if (!finiteVector(view.eye_m) || !finiteVector(view.target_m)) throw new Error('Camera coordinates must contain three finite numbers.');
  const eye = new THREE.Vector3(...view.eye_m);
  const target = new THREE.Vector3(...view.target_m);
  if (eye.distanceTo(target) < 0.02) throw new Error('Camera eye and aim must be at least 2 cm apart.');
  // Flush the previous gesture before installing a preset. Otherwise a pending
  // damped pan/orbit delta is applied to the new eye and aim during update().
  const damping = controls.enableDamping;
  controls.enableDamping = false;
  try {
    controls.update();
    camera.position.copy(eye);
    controls.target.copy(target);
    camera.up.set(0, 0, 1);
    camera.lookAt(target);
    controls.update();
  } finally {
    controls.enableDamping = damping;
  }
  changed();
}

function fitFarm() {
  const center = farmBounds.getCenter(new THREE.Vector3());
  const size = farmBounds.getSize(new THREE.Vector3());
  const radius = size.length() / 2;
  const halfAngle = THREE.MathUtils.degToRad(camera.fov / 2);
  const distance = radius / Math.sin(halfAngle) * 1.05;
  const direction = new THREE.Vector3(0.8, -1.3, 1.1).normalize();
  applyView({ eye_m: center.clone().addScaledVector(direction, distance).toArray(), target_m: center.toArray() });
  presetSelect.value = '';
}

function focusFront() {
  const frontPreset = data.presets?.find((preset) => /^(front|foreground)$/.test(String(preset.id)));
  if (frontPreset) { applyView(frontPreset); presetSelect.value = String(frontPreset.id); return; }
  // The selected native view is the known front workcell, including its source/display permutation.
  applyView(defaultView);
  presetSelect.value = defaultPresetId;
}

function setDragMode(mode) {
  dragMode = mode;
  controls.mouseButtons.LEFT = mode === 'pan' ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
  controls.touches.ONE = mode === 'pan' ? THREE.TOUCH.PAN : THREE.TOUCH.ROTATE;
  for (const name of ['orbit', 'pan']) {
    const button = byId(`${name}ModeButton`);
    button.classList.toggle('is-active', mode === name);
    button.setAttribute('aria-pressed', String(mode === name));
  }
  byId('dragInstructions').innerHTML = `Left-drag <strong>${mode}</strong><span>·</span> Right-drag / Shift-drag <strong>pan</strong><span>·</span> Scroll <strong>zoom</strong>`;
}

function resize() {
  if (!renderer) return;
  const rect = viewport.getBoundingClientRect();
  // Keep the Isaac aperture framing even in fullscreen / non-16:9 containers.
  let width = rect.width, height = rect.width * 9 / 16;
  if (height > rect.height) { height = rect.height; width = height * 16 / 9; }
  renderer.setSize(Math.max(1, Math.round(width)), Math.max(1, Math.round(height)), false);
  renderer.domElement.style.width = `${width}px`;
  renderer.domElement.style.height = `${height}px`;
  renderer.domElement.style.position = 'absolute';
  renderer.domElement.style.left = `${(rect.width - width) / 2}px`;
  renderer.domElement.style.top = `${(rect.height - height) / 2}px`;
  requestFrame();
}

function notify(message, error = false) {
  byId('exportStatus').textContent = message;
  byId('exportStatus').style.color = error ? '#efb4ba' : '';
}

function bindUI() {
  presetSelect.innerHTML = '';
  const custom = new Option('Custom camera', '');
  presetSelect.add(custom);
  presetSelect.add(new Option('Selected view 06 · adjusted', defaultPresetId));
  for (const preset of data.presets || []) presetSelect.add(new Option(preset.name || String(preset.id), String(preset.id)));
  presetSelect.disabled = false;
  presetSelect.addEventListener('change', () => {
    if (presetSelect.value === defaultPresetId) { applyView(defaultView); return; }
    const view = data.presets.find((preset) => String(preset.id) === presetSelect.value);
    if (view) applyView(view);
  });
  controls.addEventListener('start', () => { presetSelect.value = ''; });
  controls.addEventListener('change', changed);
  byId('resetButton').addEventListener('click', () => { applyView(defaultView); presetSelect.value = defaultPresetId; notify('Selected view 06 restored.'); });
  byId('orbitModeButton').addEventListener('click', () => setDragMode('orbit'));
  byId('panModeButton').addEventListener('click', () => setDragMode('pan'));
  // OrbitControls reverses modified mouse actions. Keep Shift-left as pan in
  // either explicit mode without modifying the vendored control implementation.
  renderer.domElement.addEventListener('pointerdown', (event) => {
    if (event.button === 0) controls.mouseButtons.LEFT = event.shiftKey || event.ctrlKey || event.metaKey
      ? THREE.MOUSE.ROTATE : (dragMode === 'pan' ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE);
  }, { capture: true });
  byId('fitButton').addEventListener('click', fitFarm);
  byId('focusButton').addEventListener('click', focusFront);
  byId('cameraForm').addEventListener('submit', (event) => {
    event.preventDefault();
    const values = fields.map((id) => byId(id).value.trim() === '' ? NaN : Number(byId(id).value));
    try { applyView({ eye_m: values.slice(0, 3), target_m: values.slice(3) }); presetSelect.value = ''; notify('Camera coordinates applied.'); }
    catch (error) { notify(error.message, true); }
  });
  byId('saveButton').addEventListener('click', () => {
    const blob = new Blob([`${JSON.stringify(cameraPayload(), null, 2)}\n`], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `sircus-camera-${new Date().toISOString().replaceAll(':', '-').slice(0, 19)}.json`;
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    notify('Camera file saved. Validate this view with a native Isaac still.');
  });
  byId('copyButton').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(JSON.stringify(cameraPayload(), null, 2)); notify('Camera JSON copied.'); }
    catch { notify('Clipboard unavailable. Use Save camera JSON instead.', true); }
  });
  byId('fullscreenButton').addEventListener('click', async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await viewport.requestFullscreen(); }
    catch { notify('Full screen is unavailable in this browser.', true); }
  });
  document.addEventListener('fullscreenchange', resize);
  new ResizeObserver(resize).observe(viewport);
  enabledButtons.forEach((id) => { byId(id).disabled = false; });
}

async function main() {
  const response = await fetch('scene.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`Saved scene unavailable (${response.status}).`);
  data = await response.json();
  if (data.schema !== 'sircus-camera-scout-v1') throw new Error('Unsupported scene snapshot version.');
  if (!Array.isArray(data.instances) || !data.default_camera) throw new Error('Incomplete saved scene.');
  defaultView = data.default_camera;
  cameraKey = `sircus-camera-scout:${data.source_fingerprint || 'unidentified'}`;
  scene = new THREE.Scene();
  scene.background = new THREE.Color().setRGB(...(data.floor?.color || [0.22, 0.23, 0.24]), THREE.LinearSRGBColorSpace);
  const lens = defaultView;
  const fov = THREE.MathUtils.radToDeg(2 * Math.atan(lens.vertical_aperture_mm / (2 * lens.focal_length_mm)));
  camera = new THREE.PerspectiveCamera(fov, 16 / 9, 0.025, 500);
  camera.up.set(0, 0, 1);
  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.12;
  renderer.debug.onShaderError = () => fail(new Error('WebGL could not compile a scene material. Try a browser with hardware acceleration enabled.'));
  viewport.prepend(renderer.domElement);
  renderer.domElement.addEventListener('webglcontextlost', (event) => { event.preventDefault(); fail(new Error('WebGL context lost. Reload this scouting page to recover.')); });
  renderer.domElement.addEventListener('contextmenu', (event) => event.preventDefault());
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.screenSpacePanning = true;
  controls.minDistance = 0.12;
  controls.maxDistance = 220;
  controls.maxPolarAngle = Math.PI - 0.01;
  controls.zoomSpeed = 0.8;
  scene.add(new THREE.HemisphereLight(0xf0f5fa, 0x92979d, 2.1));
  const key = new THREE.DirectionalLight(0xffffff, 2.0);
  key.position.set(-8, -12, 22); scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.9);
  fill.position.set(12, 8, 15); scene.add(fill);
  const specs = Object.entries(data.geometries || {});
  let completed = 0;
  await Promise.all(specs.map(async ([id, spec]) => {
    geometries.set(id, await buildGeometry(spec));
    completed += 1; loadingText.textContent = `Loading original geometry ${completed} / ${specs.length}…`;
  }));
  installInstances();
  bindUI();
  let initial = defaultView;
  try {
    const saved = JSON.parse(localStorage.getItem(cameraKey));
    if (saved?.source_fingerprint === data.source_fingerprint && finiteVector(saved.eye_m) && finiteVector(saved.target_m)) initial = saved;
  } catch { /* Invalid or unavailable browser storage must not block the snapshot. */ }
  applyView(initial);
  const matchesDefault = ['eye_m', 'target_m'].every((key) => initial[key].every((value, i) => Math.abs(value - defaultView[key][i]) < 1e-9));
  presetSelect.value = matchesDefault ? defaultPresetId : '';
  resize();
  loading.hidden = true;
  const cells = new Set(data.instances.map((instance) => instance.display_slot).filter(Boolean)).size;
  byId('sceneStatus').textContent = `${cells || 35} workcells · Original meshes · Saved native world transforms`;
  byId('sourceNote').textContent = `Source: ${data.source_fingerprint || 'Unspecified'}${data.scope ? ` · ${data.scope}` : ''}`;
  if (initial !== defaultView) notify('Your previous camera was restored. Reset view 06 returns to the selected native view.');
  window.addEventListener('error', (event) => { if (/webgl|shader/i.test(event.message)) fail(new Error(event.message)); });
}

main().catch(fail);
