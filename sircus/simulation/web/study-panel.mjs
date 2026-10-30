// Numerical policy preview only. No Viam or physical endpoint is accepted here.
export const MAX_JSON_BYTES = 1_000_000;
const validSeed = seed => Number.isSafeInteger(seed) && seed >= 0 && seed < 2**32;

export function sceneFromFile(value) {
  const scene = value?.scene ?? value;
  if (!scene || typeof scene !== 'object' || Array.isArray(scene) ||
      !scene.board || !scene.bin || !scene.ball || !scene.robot14) {
    throw new Error('Choose a scene JSON or one study episode containing scene.');
  }
  return scene;
}

export function validatePreview(value) {
  if (!value || value.data_origin !== 'synthetic' || value.physically_verified !== false ||
      value.native_paths_checked !== false || !value.selected_scene || !value.release_action ||
      !value.checkpoint_id || value.scoring_version !== 'sircus-board-bounce-capture-1s-v1' ||
      value.physics_version !== 'sircus-compliant-board-bucket-impulse-v1' ||
      !value.result || value.result.scoring_version !== value.scoring_version ||
      value.result.physics_version !== value.physics_version) {
    throw new Error('Incompatible numerical preview response. No replay loaded.');
  }
  if (value.result.frames !== undefined && !Array.isArray(value.result.frames)) {
    throw new Error('Invalid numerical trajectory.');
  }
  sceneFromFile(value.selected_scene);
  if (!['roll_deg', 'pitch_deg', 'u_mm', 'v_mm', 'gap_mm'].every(k => Number.isFinite(value.action?.[k]))) {
    throw new Error('Invalid predicted action.');
  }
  let previous = -1;
  if ((value.result.frames || []).length > 2000) throw new Error('Numerical trajectory exceeds preview limit.');
  for (const frame of value.result.frames || []) {
    if (!Number.isFinite(frame.t) || frame.t < 0 || frame.t < previous || !Array.isArray(frame.ball_xyz_mm) ||
        frame.ball_xyz_mm.length !== 3 || !frame.ball_xyz_mm.every(Number.isFinite)) {
      throw new Error('Invalid numerical trajectory frame.');
    }
    previous = frame.t;
  }
  return value;
}

export class StudyPanel {
  constructor({document, fetchImpl = globalThis.fetch.bind(globalThis),
    onPreview, onExit, onInvalidate = () => {}, onPending = () => {}, isBusy = () => false}) {
    this.document = document; this.fetch = fetchImpl; this.onPreview = onPreview;
    this.onExit = onExit; this.onInvalidate = onInvalidate; this.isBusy = isBusy; this.onPending = onPending;
    this.e = Object.fromEntries(['studyCheckpoint', 'studyScene', 'studySeed',
      'studyPreview', 'studyReplay', 'studyExit', 'studyStatus', 'studyAction']
      .map(id => [id, document.getElementById(id)]));
    this.checkpoint = null; this.scene = null; this.preview = null;
    this.pending = false; this.revision = 0; this.loads = {checkpoint: 0, scene: 0};
    this.e.studyCheckpoint.addEventListener('change', () => void this.load('checkpoint', this.e.studyCheckpoint.files[0]));
    this.e.studyScene.addEventListener('change', () => void this.load('scene', this.e.studyScene.files[0]));
    this.e.studySeed.addEventListener('change', () => this.invalidate());
    this.e.studyPreview.addEventListener('click', () => void this.run());
    this.e.studyReplay.addEventListener('click', () => this.preview && this.onPreview(this.preview));
    this.e.studyExit.addEventListener('click', () => { this.invalidate(); void this.onExit(); });
    this.render();
  }

  invalidate() {
    this.revision++; this.preview = null; this.onInvalidate();
    this.e.studyAction.textContent = 'No prediction loaded.';
    this.render();
  }

  async load(kind, file) {
    this.invalidate(); this[kind] = null;
    const revision = ++this.loads[kind];
    try {
      if (!file) { this.render(); return; }
      if (file.size > MAX_JSON_BYTES) throw new Error('JSON file exceeds one megabyte.');
      const value = JSON.parse(await file.text());
      if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('A JSON object is required.');
      if (revision !== this.loads[kind]) return;
      const scene = kind === 'scene' ? sceneFromFile(value) : null;
      if (kind === 'scene' && value.scene && ('seed' in value || 'rollout_seed' in value)) {
        const seed = 'seed' in value ? value.seed : value.rollout_seed;
        if (!validSeed(seed)) throw new Error('Imported episode seed must be an integer from 0 to 2^32−1.');
        this.e.studySeed.value = String(seed);
      }
      this[kind] = kind === 'scene' ? scene : value;
      this.e.studyStatus.textContent = 'File loaded. Prediction checks model, geometry and source compatibility.';
    } catch (error) {
      if (revision === this.loads[kind]) this.e.studyStatus.textContent = error.message;
    }
    this.render();
  }

  render() {
    const blocked = this.pending || this.isBusy();
    this.e.studyPreview.disabled = blocked || !this.checkpoint || !this.scene;
    this.e.studyReplay.disabled = blocked || !this.preview?.result?.frames?.length;
    this.e.studyExit.disabled = blocked;
    this.e.studyCheckpoint.disabled = blocked; this.e.studyScene.disabled = blocked;
    this.e.studySeed.disabled = blocked;
  }

  async run() {
    if (this.pending || this.isBusy() || !this.checkpoint || !this.scene) return;
    const seed = Number(this.e.studySeed.value);
    if (this.e.studySeed.value.trim() === '' || !validSeed(seed)) {
      this.e.studyStatus.textContent = 'Seed must be an integer from 0 to 2^32−1.'; return;
    }
    this.invalidate(); this.pending = true; this.onPending(true); this.render();
    const revision = this.revision;
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), 40000);
    this.e.studyStatus.textContent = 'Predicting once, checking endpoint IK, then running numerical physics…';
    try {
      const body = JSON.stringify({checkpoint: this.checkpoint, scene: this.scene, seed});
      if (new TextEncoder().encode(body).length > MAX_JSON_BYTES) throw new Error('Combined checkpoint and scene exceed one megabyte.');
      const response = await this.fetch('/api/study/preview', {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body, signal: abort.signal});
      const value = await response.json();
      if (!response.ok || value.error) throw new Error(value.error || 'Numerical preview failed.');
      validatePreview(value);
      if (revision !== this.revision) return;
      this.preview = value;
      const a = value.action, f = value.feasibility;
      this.e.studyAction.textContent = `Roll ${a.roll_deg.toFixed(2)}° · pitch ${a.pitch_deg.toFixed(2)}°\nU ${a.u_mm.toFixed(2)} mm · V ${a.v_mm.toFixed(2)} mm · gap ${a.gap_mm.toFixed(2)} mm\n` +
        `Endpoint IK: ${f?.arm_reachable && f?.board_arm_reachable ? 'reachable' : 'not reachable / unavailable'} · native paths: unchecked`;
      this.e.studyStatus.textContent = `Frozen checkpoint ${String(value.checkpoint_id).slice(0, 16)} · ${value.result.outcome}. One-second scoring; simulation prediction only.`;
      this.onPreview(value);
    } catch (error) {
      if (revision === this.revision) {
        this.preview = null;
        this.e.studyStatus.textContent = error.name === 'AbortError' ? 'Numerical preview timed out. No robot command was sent.' : error.message;
      }
    } finally { clearTimeout(timer); this.pending = false; this.onPending(false); this.render(); }
  }
}
