// Browser-independent contract tests. No server, Viam or physical runtime starts.
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {StudyPanel, sceneFromFile, validatePreview, MAX_JSON_BYTES} from '../web/study-panel.mjs';

const scene = {board: {}, bin: {}, ball: {}, robot14: {}};
const checkpoint = {checkpoint_id: 'frozen-test'};
const copy = value => JSON.parse(JSON.stringify(value));
const envelope = () => ({
  data_origin: 'synthetic', physically_verified: false, native_paths_checked: false,
  checkpoint_id: 'frozen-test', selected_scene: copy(scene),
  scoring_version: 'sircus-board-bounce-capture-1s-v1',
  physics_version: 'sircus-compliant-board-bucket-impulse-v1',
  action: {roll_deg: 8, pitch_deg: -3, u_mm: 2, v_mm: 4, gap_mm: 160},
  release_action: {u_mm: 2, v_mm: 4, gap_mm: 160},
  feasibility: {arm_reachable: true, board_arm_reachable: true},
  result: {outcome: 'captured', success: true,
    scoring_version: 'sircus-board-bounce-capture-1s-v1',
    physics_version: 'sircus-compliant-board-bucket-impulse-v1',
    frames: [{t: 0, ball_xyz_mm: [0, 0, 200]}, {t: 2, ball_xyz_mm: [0, 0, 40]}]}
});
const file = value => ({size: JSON.stringify(value).length, text: async () => JSON.stringify(value)});
const deferred = () => {let resolve; const promise = new Promise(r => {resolve = r;}); return {promise, resolve};};

function harness(fetchImpl = async () => ({ok: true, json: async () => envelope()})) {
  const elements = {};
  const document = {getElementById(id) {
    return elements[id] ??= {disabled: false, textContent: '', value: id === 'studySeed' ? '42' : '', files: [],
      listeners: {}, addEventListener(type, fn) {this.listeners[type] = fn;}};
  }};
  const calls = [], previews = [], exits = [], pending = [];
  let busy = false;
  const panel = new StudyPanel({document, fetchImpl: async (...args) => {calls.push(args); return fetchImpl(...args);},
    onPreview: value => previews.push(value), onExit: () => exits.push(true),
    onPending: value => pending.push(value), isBusy: () => busy});
  return {panel, elements, calls, previews, exits, pending, setBusy: value => {busy = value; panel.render();}};
}

async function load(h, episode = scene) {
  await h.panel.load('checkpoint', file(checkpoint));
  await h.panel.load('scene', file(episode));
}

test('explicit imports and prediction; replay sends no network or robot requests', async () => {
  const h = harness();
  assert.equal(h.elements.studyPreview.disabled, true);
  assert.equal(h.elements.studyReplay.disabled, true);
  await load(h, {scene, seed: 2765472149});
  assert.equal(h.elements.studySeed.value, '2765472149');
  assert.equal(h.elements.studyPreview.disabled, false);
  assert.equal(h.calls.length, 0);
  await h.panel.run();
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0][0], '/api/study/preview');
  assert.deepEqual(JSON.parse(h.calls[0][1].body), {checkpoint, scene, seed: 2765472149});
  assert.equal(h.calls[0][1].method, 'POST');
  assert.equal(h.previews.length, 1);
  assert.equal(h.elements.studyReplay.disabled, false);
  h.elements.studyReplay.listeners.click();
  assert.equal(h.previews.length, 2);
  assert.equal(h.calls.length, 1);
  assert.match(h.elements.studyAction.textContent, /native paths: unchecked/);
  assert.deepEqual(h.pending, [true, false]);
});

test('uint32 episode seeds preserve evaluation reproducibility and reject invalid imports', async () => {
  const h = harness();
  await load(h, {scene, rollout_seed: 2**32-1});
  assert.equal(h.elements.studySeed.value, String(2**32-1));
  await h.panel.run();
  assert.equal(JSON.parse(h.calls[0][1].body).seed, 2**32-1);
  for (const seed of [2**32, -1, true, '42']) {
    await h.panel.load('scene', file({scene, seed}));
    assert.equal(h.panel.scene, null);
    assert.equal(h.elements.studyPreview.disabled, true);
    assert.match(h.elements.studyStatus.textContent, /episode seed/);
  }
});

test('busy state prevents preview and unavailable audio-independent API has no fallback', async () => {
  const h = harness(async () => {throw new Error('Numerical service unavailable');});
  await load(h);
  h.setBusy(true);
  await h.panel.run();
  assert.equal(h.calls.length, 0);
  h.setBusy(false);
  await h.panel.run();
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0][0], '/api/study/preview');
  assert.equal(h.previews.length, 0);
  assert.match(h.elements.studyStatus.textContent, /unavailable/);
  assert.equal(h.panel.pending, false);
});

test('duplicate clicks cannot launch concurrent inference', async () => {
  const d = deferred(), h = harness(() => d.promise);
  await load(h);
  const run = h.panel.run();
  await h.panel.run();
  assert.equal(h.calls.length, 1);
  d.resolve({ok: true, json: async () => envelope()});
  await run;
  assert.equal(h.previews.length, 1);
});

test('delayed preview cannot replace a newly loaded scene or explicit exit', async () => {
  for (const change of ['scene', 'checkpoint', 'exit', 'seed']) {
    const d = deferred(), h = harness(() => d.promise);
    await load(h);
    const run = h.panel.run();
    if (change === 'scene') await h.panel.load('scene', file({...scene, variant: 2}));
    if (change === 'checkpoint') await h.panel.load('checkpoint', file({checkpoint_id: 'new'}));
    if (change === 'exit') h.elements.studyExit.listeners.click();
    if (change === 'seed') h.elements.studySeed.listeners.change();
    d.resolve({ok: true, json: async () => envelope()});
    await run;
    assert.equal(h.previews.length, 0, change);
    assert.equal(h.panel.preview, null, change);
    assert.equal(h.elements.studyReplay.disabled, true, change);
  }
});

test('slow file reading cannot replace newer checkpoint contents', async () => {
  const d = deferred(), h = harness();
  const slow = h.panel.load('checkpoint', {size: 20, text: () => d.promise});
  await h.panel.load('checkpoint', file({checkpoint_id: 'new'}));
  d.resolve(JSON.stringify({checkpoint_id: 'old'}));
  await slow;
  assert.equal(h.panel.checkpoint.checkpoint_id, 'new');
});

test('unknown/legacy/provenance-incompatible previews fail closed', async () => {
  for (const mutation of [
    value => {value.scoring_version = 'legacy';},
    value => {value.result.physics_version = 'legacy';},
    value => {value.physically_verified = true;},
    value => {value.native_paths_checked = true;},
    value => {value.data_origin = 'hardware';},
    value => {value.action.gap_mm = Infinity;},
    value => {value.result.frames[1].t = -1;},
    value => {value.result.frames[1].ball_xyz_mm[2] = NaN;},
    value => {value.result.frames = Array(2001).fill(value.result.frames[0]);}
  ]) {
    const value = envelope(); mutation(value);
    assert.throws(() => validatePreview(value));
    const h = harness(async () => ({ok: true, json: async () => value}));
    await load(h); await h.panel.run();
    assert.equal(h.previews.length, 0);
    assert.equal(h.panel.preview, null);
  }
});

test('size and seed limits reject requests without compute; input is JSON only', async () => {
  const h = harness();
  await load(h);
  for (const seed of ['', '-1', '1.2', String(2**32)]) {
    h.elements.studySeed.value = seed; await h.panel.run();
    assert.equal(h.calls.length, 0);
  }
  await h.panel.load('checkpoint', {size: MAX_JSON_BYTES+1, text: async () => {throw new Error('must not read');}});
  assert.equal(h.panel.checkpoint, null);
  assert.match(h.elements.studyStatus.textContent, /megabyte/);
  await h.panel.load('checkpoint', file('/tmp/model.pkl'));
  assert.equal(h.panel.checkpoint, null);
  assert.throws(() => sceneFromFile('/tmp/scene.json'));
});

// Run the real dashboard state/handlers with only rendering stubbed. This
// catches isolation/provenance errors outside the standalone study panel.
async function dashboard() {
  const elements = {}, calls = []; let tick, responder;
  const element = id => elements[id] ??= {id, value: ({studySeed:'42',speed:'1',dropU:'12',dropV:'23',dropGap:'140'})[id] ?? '',
    textContent: '', innerHTML: 'Original Viam copy', style: {}, files: [], disabled: false,
    classList: {toggle() {}, add() {}, remove() {}}, parentElement: {classList: {toggle() {}}},
    listeners: {}, addEventListener(type, callback) {this.listeners[type] = callback;},
    closest() {return null;}, getAttribute() {return 'Original Viam canvas';}, setAttribute() {}};
  const document = {getElementById: element, querySelector: element, querySelectorAll: () => []};
  responder = async url => ({ok: true, json: async () => url === '/api/arm-model' ? {} :
    {config: {}, status: 'idle', connection: {backend: 'offline', connected: true}}});
  const browserFetch = async (...args) => {calls.push(args); return responder(...args);};
  const BrowserStudyPanel = class extends StudyPanel {constructor(options) {super({...options, fetchImpl: browserFetch});}};
  const context = vm.createContext({document, StudyPanel: BrowserStudyPanel, performance, console,
    fetch: browserFetch,
    setInterval: fn => {tick = fn;}, requestAnimationFrame: () => {}, setTimeout, clearTimeout});
  const source = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8')
    .replace(/^import .*;\n/gm, '').replace(/\ninit\(\);\s*$/, '\n');
  vm.runInContext(source + `
    createScene=()=>{armGroup={visible:true};controls={update(){}};renderer={render(){}};};
    syncInputs=()=>{};buildEnvironment=()=>{};renderArm=()=>{};
    updateTargetPoint=()=>{};clearTrail=()=>{};prepareTrail=()=>{};placePhysics=()=>{};
    globalThis.dashboardAPI={init,connect,setBusy,showStudyPreview,animate,
      panel:()=>studyPanel,
      prepareConnected:()=>{offline=false;viamConnected=true;config={original:true};sceneDirty=false;playing=false;},
      enterStudy:()=>{studyMode=true;studyEnvelope={data_origin:'synthetic'};result={synthetic:true};config={numerical:true};},
      snapshot:()=>({studyMode,result,config,elapsed})};`, context);
  await context.dashboardAPI.init();
  return {api: context.dashboardAPI, calls, elements, tick: () => tick(), setResponder: value => {responder = value;}};
}

test('dashboard failed reconnect clears numerical evidence before provenance changes', async () => {
  const h = await dashboard();
  h.api.enterStudy();
  h.setResponder(async () => {throw new Error('Viam unavailable');});
  await h.api.connect();
  assert.equal(h.api.snapshot().studyMode, false);
  assert.equal(h.api.snapshot().result, null);
  assert.equal(h.elements.downloadRun.disabled, true);
});

test('dashboard cannot cancel a Viam task from numerical pending state', async () => {
  const h = await dashboard(), d = deferred();
  h.api.prepareConnected();
  await h.api.panel().load('checkpoint', file(checkpoint));
  await h.api.panel().load('scene', file(scene));
  h.setResponder(() => d.promise);
  const run = h.api.panel().run();
  assert.equal(h.elements.cancelButton.disabled, true);
  await h.elements.cancelButton.listeners.click();
  assert.equal(h.calls.some(([url]) => url === '/api/stop'), false);
  d.resolve({ok: true, json: async () => envelope()});
  await run;
  assert.equal(h.api.snapshot().studyMode, true);
  assert.equal(h.calls.filter(([url]) => url.startsWith('/api/study/')).length, 1);
});

test('late native polling success or failure does not overwrite numerical view', async () => {
  for (const fail of [false, true]) {
    const h = await dashboard(), d = deferred();
    h.api.prepareConnected();
    h.setResponder(async () => {await d.promise; if(fail)throw new Error('late native error');
      return {ok: true, json: async () => ({config: {native: true}, status: 'idle'})};});
    const poll = h.tick();
    h.api.enterStudy();
    h.elements.phase.textContent = 'Numerical preview complete';
    d.resolve(); await poll;
    assert.equal(h.api.snapshot().config.numerical, true);
    assert.equal(h.elements.phase.textContent, 'Numerical preview complete');
  }
});

test('first numerical RAF frame cannot enter an empty native arm sequence', async () => {
  const h = await dashboard();
  h.api.showStudyPreview(envelope());
  // RAF was queued before play reset lastTime using performance.now().
  assert.doesNotThrow(() => h.api.animate(0));
  assert.equal(h.api.snapshot().elapsed, 0);
  h.api.showStudyPreview(envelope());
  assert.doesNotThrow(() => h.api.animate(0));
  assert.equal(h.api.snapshot().elapsed, 0);
});

test('return restores prior release action unless the runtime supplies its own', async () => {
  for (const supplied of [false, true]) {
    const h = await dashboard();
    h.api.showStudyPreview(envelope());
    assert.equal(h.elements.dropU.value, '2.00');
    // A second prediction must not replace the saved pre-study action.
    h.api.showStudyPreview(envelope());
    h.setResponder(async url => ({ok:true,json:async()=>url==='/api/arm-model'?{}:
      {config:{},connection:{backend:'offline',connected:true},
        ...(supplied?{default_action:{u_mm:50,v_mm:60,gap_mm:170}}:{})}}));
    await h.api.connect();
    assert.equal(h.elements.dropU.value, supplied?'50.00':'12.00');
    assert.equal(h.elements.dropV.value, supplied?'60.00':'23.00');
    assert.equal(h.elements.dropGap.value, supplied?'170.00':'140.00');
  }
});
