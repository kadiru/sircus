import assert from 'node:assert/strict';
import {SircusAudio} from '../integration/sircus-audio-client.mjs';
import {DashboardAudio} from '../integration/dashboard-worker.mjs';

const tick=()=>new Promise(resolve=>setTimeout(resolve,0));
const wait=async worker=>{for(let i=0;i<100&&worker.controlling;i++)await tick();assert.equal(worker.controlling,false);};
const event=(event_id='e',trial_id='t')=>({schema:'sircus.audio.v1',event_id,trial_id,
  event_type:'release',source:'operator',evidence:'operator_action',outcome:'unknown',confirmed:false,
  contact_object:'unknown',timestamp:10,time_basis:'host_monotonic_s',clock_id:'test',provenance:'explicit mock action'});
const calls=[],heard=[],seen=new Set(),statuses=[];
const audio=new SircusAudio({baseURL:'http://example.invalid',token:'TEST_ONLY',fetchImpl:async(url,opts)=>{
  const data=opts.body&&JSON.parse(opts.body);calls.push({url,data});
  if(url.endsWith('/event')){
    const key=data.trial_id+':'+data.event_id;
    if(seen.has(key))return {ok:true,json:async()=>({accepted:false,reason:'duplicate_or_capacity'})};
    seen.add(key);heard.push(data);
  }
  return {ok:true,json:async()=>url.endsWith('/status')?{connected:false,audio_ready:true,fx_enabled:true}:{accepted:true}};
}});
const worker=new DashboardAudio({audio,publish:s=>statuses.push(s),now:()=>1000});
worker.handle({op:'emit',event:event(),queued_unix_ms:1000,revision:0});assert.equal(calls.length,0);
await worker.poll();assert.equal(worker.connection,'connected'); // MIDI disconnected != HTTP disconnected.
worker.handle({op:'fx',enabled:true,revision:1});await wait(worker);assert.equal(worker.enabled,true);
for(let i=0;i<2;i++)worker.handle({op:'emit',event:event(),queued_unix_ms:1000,revision:1});
while(audio.running)await tick();assert.equal(heard.length,1);assert.equal(heard[0].evidence,'operator_action');
worker.handle({op:'emit',event:event('stale'),queued_unix_ms:-1,revision:1});assert.equal(heard.length,1);
worker.handle({op:'stop',trial_id:'t',revision:2});await wait(worker);
worker.handle({op:'emit',event:event('late'),queued_unix_ms:1000,revision:2});await tick();assert.equal(heard.length,1);
worker.handle({op:'fx',enabled:false,revision:3});await wait(worker);assert.equal(worker.muteConfirmed,true);
worker.handle({op:'emit',event:event('disabled','new'),queued_unix_ms:1000,revision:3});assert.equal(heard.length,1);
assert.ok(calls.some(call=>call.url.endsWith('/fx')&&call.data.enabled===false));

// An unavailable service never throws on producer calls or leaks raw transport error text.
const unavailable=new SircusAudio({baseURL:'http://example.invalid',token:'TEST_ONLY',fetchImpl:async()=>{throw Error('SECRET_URL_AND_TOKEN');}});
const failureStates=[];const failure=new DashboardAudio({audio:unavailable,publish:s=>failureStates.push(s)});
assert.equal(failure.handle({op:'fx',enabled:true,revision:1}),undefined);await wait(failure);
assert.equal(failure.enabled,false);assert.equal(failure.connection,'error');
assert.equal(JSON.stringify(failureStates).includes('SECRET_URL_AND_TOKEN'),false);

// Delayed enable response cannot relatch FX after OFF. Abort is not remote cancellation.
let rejectEnable, remoteEnabled=false;
const raceAudio={onError:()=>{},cancel(){if(rejectEnable){rejectEnable(Error('aborted'));rejectEnable=null;}},
  async setVolume(){},setFX(enabled){
    if(enabled)return new Promise((_,reject)=>{rejectEnable=reject;});
    remoteEnabled=false;return Promise.resolve({accepted:true});
  },async stop(){},emit(){throw Error('must remain disabled');},async status(){return {fx_enabled:remoteEnabled};}};
const race=new DashboardAudio({audio:raceAudio});race.handle({op:'fx',enabled:true,revision:1});await tick();
race.handle({op:'fx',enabled:false,revision:2});await wait(race);
remoteEnabled=true; // Delayed request can still execute on Mini after its client was aborted.
await race.poll();
assert.equal(race.enabled,false);assert.equal(race.muteConfirmed,false);
assert.equal(race.error,'remote_mute_unconfirmed');

// Independent control revisions: newer volume/stop must not supersede explicit FX.
const independentCalls=[];
const independentAudio={onError:()=>{},cancel(){},async setVolume(v){independentCalls.push(['volume',v]);},
  async setFX(value){independentCalls.push(['fx',value]);},async stop(t){independentCalls.push(['stop',t]);},emit(){}};
const independent=new DashboardAudio({audio:independentAudio});
independent.handle({op:'fx',enabled:true,revision:1});
independent.handle({op:'volume',volume:.6,revision:2});await wait(independent);
assert.equal(independent.enabled,true);assert.ok(independentCalls.some(row=>row[0]==='fx'&&row[1]===true));
// Python prioritizes scoped stop, so OFF can legitimately arrive with an older global revision.
independent.handle({op:'stop',trial_id:'old',revision:4});
independent.handle({op:'fx',enabled:false,revision:3});
independent.handle({op:'volume',volume:.2,revision:5});await wait(independent);
assert.equal(independent.enabled,false);assert.equal(independent.muteConfirmed,true);
assert.ok(independentCalls.some(row=>row[0]==='fx'&&row[1]===false));

// Scoped cancellation aborts client requests, so an interrupted OFF is reasserted.
let offReject, offCount=0;
const offAudio={onError:()=>{},cancel(){if(offReject){offReject(Error('aborted'));offReject=null;}},
  async setVolume(){},setFX(enabled){assert.equal(enabled,false);offCount++;
    if(offCount===1)return new Promise((_,reject)=>{offReject=reject;});return Promise.resolve();},
  async stop(){},emit(){}};
const offWorker=new DashboardAudio({audio:offAudio});
offWorker.handle({op:'fx',enabled:false,revision:1});await tick();
offWorker.handle({op:'stop',trial_id:'cancelled',revision:2});await wait(offWorker);
assert.equal(offCount,2);assert.equal(offWorker.muteConfirmed,true);

// A full control queue fails closed to global mute instead of dropping cancellation.
let releaseBusy;
const capacityCalls=[];
const capacityAudio={onError:()=>{},cancel(){},setVolume(){return new Promise(resolve=>{releaseBusy=resolve;});},
  async setFX(value){capacityCalls.push(value);},async stop(){},emit(){}};
const capacity=new DashboardAudio({audio:capacityAudio});
capacity.handle({op:'volume',volume:.3,revision:1});
for(let i=2;i<50;i++)capacity.handle({op:'stop',trial_id:'t'+i,revision:i});
assert.ok(capacity.controlQueue.length<=32);assert.equal(capacity.enabled,false);
releaseBusy();await wait(capacity);assert.ok(capacityCalls.includes(false));

// A delayed unavailable transport times out entirely off the producer call stack.
const timeoutAudio=new SircusAudio({baseURL:'http://example.invalid',token:'TEST_ONLY',timeoutMs:10,
  fetchImpl:async(_url,options)=>new Promise((_,reject)=>options.signal.addEventListener('abort',()=>reject(Error('timeout'))))});
const timeoutWorker=new DashboardAudio({audio:timeoutAudio});
const started=performance.now();timeoutWorker.handle({op:'fx',enabled:true,revision:1});
assert.ok(performance.now()-started<50);await wait(timeoutWorker);assert.equal(timeoutWorker.enabled,false);
assert.equal(timeoutWorker.connection,'error');
console.log('Dashboard worker mocks passed: opt-in, provenance, duplicate/late events, unavailable/delayed audio, independent control revisions, cancellation and priority mute');
