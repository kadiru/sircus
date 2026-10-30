import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source = await readFile(new URL('../web/audio.js', import.meta.url), 'utf8');
const {AudioPanel} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const IDS = ['audioPanel', 'audioFX', 'audioFXLabel', 'audioVolume', 'audioVolumeValue',
  'audioConnection', 'audioMessage', 'audioMuteWarning'];
function fixture(fetchImpl) {
  const nodes = Object.fromEntries(IDS.map(id => [id, {id, checked: false, disabled: false,
    value: '0.35', hidden: true, textContent: '', listeners: {}, classList: {toggle() {}},
    addEventListener(event, fn) {this.listeners[event] = fn;},
    removeEventListener(event) {delete this.listeners[event];}}]));
  const robotStop = {disabled: false, clicks: 0};
  const requestedIds = [];
  const document = {getElementById(id) {requestedIds.push(id); return nodes[id];}};
  const panel = new AudioPanel({document, fetchImpl,
    setTimer: () => 1, clearTimer: () => {}, clientId:'00000000-0000-4000-8000-000000000001'});
  return {panel, nodes, requestedIds, robotStop};
}
const response = value => ({ok: true, json: async () => value});
const ready = {configured: true, enabled: false, requested_enabled: false, volume: 0.35,
  connection: 'connected', action_token: 'local-csrf', mute_confirmed: null, pending: false};
const deferred = () => {let resolve, reject; const promise = new Promise((a,b) => {resolve=a; reject=b;}); return {promise, resolve, reject};};

// Startup only observes status, even when an old UI or storage could have been ON.
const calls = [];
const f = fixture(async (path, options) => {calls.push({path, options}); return response(ready);});
assert.equal(f.nodes.audioFX.checked, false);
assert.equal(f.nodes.audioFX.disabled, true);
await f.panel.poll();
assert.deepEqual(calls.map(c => [c.path, c.options.method]), [['/api/audio/status', 'GET']]);
assert.equal(f.nodes.audioFX.checked, false);
assert.equal(f.nodes.audioVolumeValue.textContent, '35%');
assert.ok(f.requestedIds.every(id => id.startsWith('audio')));
assert.equal(f.robotStop.disabled, false);
f.panel.destroy();

// Audio unavailable cannot disable robot Stop or other physical controls.
const unavailable = fixture(async () => {throw new Error('offline with private diagnostic');});
await unavailable.panel.poll();
assert.equal(unavailable.nodes.audioFX.checked, false);
assert.equal(unavailable.nodes.audioConnection.textContent, 'Audio unavailable');
assert.match(unavailable.nodes.audioMessage.textContent, /unavailable/);
assert.doesNotMatch(unavailable.nodes.audioMessage.textContent, /private diagnostic/);
assert.equal(unavailable.robotStop.disabled, false);
unavailable.panel.destroy();

// A delayed pre-control status cannot undo the operator's later OFF intent.
const pollGate = deferred(), enableGate = deferred();
const controls = [];
const race = fixture((path, options) => {
  if (path.endsWith('/status')) return pollGate.promise;
  const payload = JSON.parse(options.body); controls.push({payload, options});
  if (payload.enabled) return enableGate.promise;
  return Promise.resolve(response({...ready, mute_confirmed: true}));
});
race.panel.accept(ready);
const oldPoll = race.panel.poll();
race.panel.state.requested_enabled = true;
const enabling = race.panel.control({action: 'fx', enabled: true});
assert.equal(race.nodes.audioFX.disabled, false, 'OFF must remain usable while enable is pending');
race.panel.state.requested_enabled = false;
const disabling = race.panel.control({action: 'fx', enabled: false});
await disabling;
enableGate.resolve(response({...ready, enabled: true, requested_enabled: true}));
pollGate.resolve(response({...ready, enabled: true, requested_enabled: true}));
await Promise.all([oldPoll, enabling]);
assert.equal(race.nodes.audioFX.checked, false);
assert.equal(race.nodes.audioFXLabel.textContent, 'off');
assert.equal(race.nodes.audioMuteWarning.hidden, true);
assert.equal(controls[0].options.headers['X-Sircus-Audio-Token'], 'local-csrf');
assert.equal(controls[0].options.headers.Authorization, undefined);
assert.equal(race.robotStop.disabled, false);
race.panel.destroy();

// A failed mute remains explicit through ordinary reconnect; only a positive
// mute acknowledgment clears it. No automatic retry/reenable or robot request.
const failedMuteCalls = [];
const muted = fixture(async (path, options) => {
  failedMuteCalls.push(path);
  if (options.method === 'POST') throw new Error('backend offline');
  return response(ready);
});
muted.panel.accept({...ready, enabled: true, requested_enabled: true});
await muted.panel.control({action:'fx', enabled:false});
assert.equal(muted.nodes.audioFX.checked, false);
assert.equal(muted.nodes.audioMuteWarning.hidden, false);
await muted.panel.poll();
assert.equal(muted.nodes.audioMuteWarning.hidden, false);
muted.panel.accept({...ready, mute_confirmed:true});
assert.equal(muted.nodes.audioMuteWarning.hidden, true);
assert.deepEqual(failedMuteCalls, ['/api/audio/control', '/api/audio/status']);
muted.panel.destroy();

// Volume control is explicit, bounded and never invokes robot endpoints.
const volumeCalls = [];
const volume = fixture(async(path, options) => {volumeCalls.push({path, options}); return response({...ready,volume:0.5});});
volume.panel.accept(ready);
volume.nodes.audioVolume.value = '0.5';
volume.nodes.audioVolume.listeners.input();
assert.equal(volume.nodes.audioVolumeValue.textContent, '50%');
volume.nodes.audioVolume.listeners.change();
await new Promise(resolve => setImmediate(resolve));
assert.deepEqual(JSON.parse(volumeCalls[0].options.body), {action:'volume',volume:0.5,
  client_id:'00000000-0000-4000-8000-000000000001',sequence:1});
volume.nodes.audioVolume.value = '2';
volume.nodes.audioVolume.listeners.change();
assert.equal(volumeCalls.length, 1);
assert.equal(volume.robotStop.disabled, false);
volume.panel.destroy();

// Requests can reach the backend out of order, independently of response order.
// Model the observer's per-client sequence gate rather than only delaying replies.
const lateOn = deferred(); let remoteSequence = 0, remoteEnabled = false;
const orderingCalls = [];
const ordering = fixture(async(path, options) => {
  if (path.endsWith('/status')) return response({...ready,enabled:remoteEnabled,requested_enabled:remoteEnabled});
  const body = JSON.parse(options.body); orderingCalls.push(body);
  if (body.enabled) await lateOn.promise;
  if (body.sequence > remoteSequence) {remoteSequence=body.sequence;remoteEnabled=body.enabled;}
  return response({...ready,enabled:remoteEnabled,requested_enabled:remoteEnabled,mute_confirmed:!remoteEnabled});
});
ordering.panel.accept(ready);
const oldOn = ordering.panel.control({action:'fx',enabled:true});
await ordering.panel.control({action:'fx',enabled:false});
assert.equal(ordering.nodes.audioFX.disabled,false);
lateOn.resolve(); await oldOn; await ordering.panel.poll();
assert.deepEqual(orderingCalls.map(value=>value.sequence),[1,2]);
assert.equal(orderingCalls[0].client_id,orderingCalls[1].client_id);
assert.equal(remoteEnabled,false);
assert.equal(ordering.nodes.audioFX.checked,false);
assert.equal(ordering.nodes.audioMuteWarning.hidden,true);
ordering.panel.destroy();

// A failed OFF followed by an ON status is not fresh operator consent to enable.
const ambiguous = fixture(async(path,options)=>{
  if(options.method==='POST')throw new Error('lost mute packet');
  return response({...ready,enabled:true,requested_enabled:true,mute_confirmed:null});
});
ambiguous.panel.accept({...ready,enabled:true,requested_enabled:true});
await ambiguous.panel.control({action:'fx',enabled:false});
await ambiguous.panel.poll();
assert.equal(ambiguous.nodes.audioFX.checked,false);
assert.equal(ambiguous.nodes.audioMuteWarning.hidden,false);
ambiguous.panel.accept({...ready,enabled:true,requested_enabled:true,mute_confirmed:true});
assert.equal(ambiguous.nodes.audioMuteWarning.hidden,false,'contradictory ON status cannot confirm mute');
ambiguous.panel.accept({...ready,enabled:false,requested_enabled:false,mute_confirmed:true});
assert.equal(ambiguous.nodes.audioMuteWarning.hidden,true);
ambiguous.panel.destroy();

// Mini-local OFF/restart is a settled OFF, not permanently checked/pending.
let miniStatus={...ready,enabled:true,requested_enabled:true};const miniCalls=[];
const mini=fixture(async(path,options)=>{miniCalls.push([path,options.method]);return response(miniStatus);});
await mini.panel.poll();assert.equal(mini.nodes.audioFX.checked,true);
miniStatus={...ready,mute_confirmed:true,error:'FX was disabled on the Mini; enable it here to resume cues'};
await mini.panel.poll();
assert.equal(mini.nodes.audioFX.checked,false);assert.equal(mini.nodes.audioFXLabel.textContent,'off');
assert.equal(mini.nodes.audioMuteWarning.hidden,true);
// Worker reports Mini ON separately as uncertainty, without adopting its remote intent.
miniStatus={...ready,mute_confirmed:false,error:'Remote audio may be on; mute is unconfirmed'};
await mini.panel.poll();
assert.equal(mini.nodes.audioFX.checked,false);assert.equal(mini.nodes.audioFXLabel.textContent,'off');
assert.equal(mini.nodes.audioMuteWarning.hidden,false);
assert.ok(miniCalls.every(([path,method])=>path==='/api/audio/status'&&method==='GET'));
// A settled failure is unavailable; only genuinely pending work is labelled pending.
mini.panel.accept({...ready,requested_enabled:true,pending:false,error:'Optional audio unavailable'});
assert.equal(mini.nodes.audioFXLabel.textContent,'unavailable');
mini.panel.destroy();

// WebKit timer methods require the Window receiver. Default adapters must not
// invoke an unbound native method with AudioPanel as its `this` value.
const originalSetTimeout = globalThis.setTimeout, originalClearTimeout = globalThis.clearTimeout;
const timerCalls = [], cleared = [];
let timerPanel;
try {
  globalThis.setTimeout = function(callback, delay) {
    assert.equal(this, globalThis, 'setTimeout lost its Window receiver');
    timerCalls.push(delay); return timerCalls.length;
  };
  globalThis.clearTimeout = function(timer) {
    assert.equal(this, globalThis, 'clearTimeout lost its Window receiver');
    cleared.push(timer);
  };
  const timerFixture = fixture(async()=>response(ready));
  timerFixture.panel.destroy();
  timerPanel = new AudioPanel({document:{getElementById:id=>timerFixture.nodes[id]},
    fetchImpl:async()=>response(ready),clientId:'timer-test'});
  await timerPanel.poll();
  timerPanel.start();
  await new Promise(resolve=>setImmediate(resolve));
  assert.ok(timerCalls.includes(2500), 'request timeout was scheduled');
  assert.ok(timerCalls.includes(1500), 'next status poll was scheduled');
  assert.ok(cleared.length>=2, 'request timeouts were cleared');
  timerPanel.destroy();
} finally {
  globalThis.setTimeout = originalSetTimeout;
  globalThis.clearTimeout = originalClearTimeout;
}
console.log('Audio panel: default OFF, isolated failure, ordered controls, stale replies, mute uncertainty, and volume passed');
