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
  if (Number.isFinite(p.elapsed_s) && p.elapsed_s >= 0) parts.push(`Elapsed ${Math.floor(p.elapsed_s)} s`);
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
export class InteractivePhysics {
  constructor({
    fetcher = (...args) => globalThis.fetch(...args),
    onChange = () => {},
    schedule = (callback, delay) => globalThis.setTimeout(callback, delay),
    cancel = timer => globalThis.clearTimeout(timer),
  } = {}) {
    this.fetcher = fetcher; this.onChange = onChange; this.schedule = schedule; this.cancel = cancel;
    this.epoch = 0; this.timer = null; this.capabilities = null;
    this.snapshot = { state: 'unavailable', message: 'Physics planner unavailable.' };
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
    const epoch = ++this.epoch, target = xy.slice();
    if (this.timer !== null) this.cancel(this.timer);
    this.timer = null;
    this.publish({ state: 'submitting', bucket_xy_m: target, message: 'Submitting shot plan…' });
    try {
      const job = await this.json('/api/interactive-physics/plan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ bucket_xy_m: target }) });
      if (epoch !== this.epoch) return false;
      if (typeof job.id !== 'string' || !job.id) throw Error('Planner did not return a job ID.');
      this.publish({ ...job, state: 'queued', bucket_xy_m: target, message: 'Planning geometry and both arm poses…' });
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
      if (job.state === 'ready' && (!hasNativePreview(job.preview) || !sameXY(job.preview.bucket_xy_m, xy))) throw Error('Plan has no complete native arm-pose preview.');
      if (job.state === 'completed' && (!job.result || !sameXY(job.result.bucket_xy_m, xy))) throw Error('Drop result does not match this target.');
      this.publish(job);
      if (!terminal.has(job.state)) this.timer = this.schedule(() => { this.timer = null; void this.poll(epoch, id, xy); }, 700);
    } catch (error) { if (epoch === this.epoch) this.publish({ state: 'failed', id, bucket_xy_m: xy, message: error.message }); }
  }
}
