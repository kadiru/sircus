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
  return {ok:true,json:async()=>url.endsWith('/status')?{connected:false,audio_ready:true,fx_enabled:true,control_epoch:'TEST_ONLY_EPOCH'}:{accepted:true}};
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
// Mini-local controls/restart reconcile the full published intent, not only enabled.
const remoteCalls=[], remoteStates=[];let miniEnabled=false;
const remoteAudio={onError:()=>{},cancel(){remoteCalls.push('cancel');},async setVolume(){},
  async setFX(value){remoteCalls.push(['fx',value]);miniEnabled=value;},async stop(){},emit(){remoteCalls.push('cue');},
  async status(){return {fx_enabled:miniEnabled};}};
const remoteWorker=new DashboardAudio({audio:remoteAudio,publish:value=>remoteStates.push(value)});
remoteWorker.handle({op:'fx',enabled:true,revision:1});await wait(remoteWorker);
miniEnabled=false; // Ghost Booth local OFF or a fresh FX-off service after restart.
await remoteWorker.poll();
assert.equal(remoteStates.at(-1).status.requested_enabled,false);
assert.equal(remoteStates.at(-1).status.enabled,false);
assert.equal(remoteStates.at(-1).status.pending,false);
assert.equal(remoteWorker.error,'audio_disabled_remotely');
remoteWorker.handle({op:'emit',event:event('after-mini-off'),queued_unix_ms:Date.now(),revision:1});
assert.equal(remoteCalls.includes('cue'),false);
assert.deepEqual(remoteCalls.filter(row=>Array.isArray(row)),[['fx',true]],'status must not auto-command FX');
miniEnabled=true; // Another Mini client turns FX on after our previous mute observation.
await remoteWorker.poll();
assert.equal(remoteStates.at(-1).status.requested_enabled,false);
assert.equal(remoteStates.at(-1).status.enabled,false);
assert.equal(remoteStates.at(-1).status.mute_confirmed,false);
assert.equal(remoteStates.at(-1).status.error,'remote_mute_unconfirmed');
remoteWorker.handle({op:'fx',enabled:false,revision:2});await wait(remoteWorker);
assert.equal(remoteWorker.muteConfirmed,true);
miniEnabled=true;await remoteWorker.poll();
assert.equal(remoteWorker.muteConfirmed,false,'remote ON invalidates an acknowledged OFF');

// An idle status read can overlap a control; it cannot overwrite its new revision.
let finishOldPoll;
remoteAudio.status=()=>new Promise(resolve=>{finishOldPoll=resolve;});
const stalePoll=remoteWorker.poll();
remoteWorker.handle({op:'fx',enabled:true,revision:3});await wait(remoteWorker);
finishOldPoll({fx_enabled:false});await stalePoll;
assert.equal(remoteWorker.enabled,true);assert.equal(remoteWorker.requested,true);

// A poll started during enable can carry the same revision yet sample pre-enable OFF.
let finishDuringPoll, finishVolume;
const pendingAudio={onError:()=>{},cancel(){},setVolume(){return new Promise(resolve=>{finishVolume=resolve;});},
  async setFX(){},async status(){return await new Promise(resolve=>{finishDuringPoll=resolve;});}};
const pendingWorker=new DashboardAudio({audio:pendingAudio});
pendingWorker.handle({op:'fx',enabled:true,revision:1});
const duringPoll=pendingWorker.poll();finishVolume();await wait(pendingWorker);
finishDuringPoll({fx_enabled:false});await duringPoll;
assert.equal(pendingWorker.enabled,true);assert.equal(pendingWorker.requested,true);
console.log('Dashboard worker mocks passed: opt-in, provenance, duplicate/late events, unavailable/delayed audio, independent control revisions, cancellation and priority mute');

// Four-ball mappings require a fresh native/configuration/selection observation,
// not protocol support or the earlier configure ACK alone.
const digest='a'.repeat(64),changedDigest='b'.repeat(64);
const balls=['pink','purple','green','gray'].map(color=>({ball_id:'ball-'+color,bound:true,connected:true}));
const showState=()=>({audio_ready:true,route:'native',fx_enabled:false,show_audio:{
  schema:'sircus.show-audio.v1',readiness_schema:'sircus.show-readiness.v1',multiball_event_routing:true,
  configured:true,config_digest:digest,selection_matches:true,ball_inputs:balls.map(row=>({...row}))}});
const showCalls=[], showStates=[];let showFails=true,remoteShow=showState();
const showAudio={onError:()=>{},cancel(){},async setVolume(){},async setFX(value,options){showCalls.push(['fx',value,options]);remoteShow.fx_enabled=value;},
  async configureShow(config){showCalls.push(['config',config]);if(showFails)throw Error('unavailable');return {...remoteShow.show_audio,config_digest:digest};},
  async status(){return structuredClone(remoteShow);},emit(e){showCalls.push(['event',e]);}};
const showWorker=new DashboardAudio({audio:showAudio,publish:value=>showStates.push(value)});
showWorker.handle({op:'show_config',config:{test:true},revision:1});await wait(showWorker);
await showWorker.poll();
showWorker.handle({op:'fx',enabled:true,revision:2});await wait(showWorker);
assert.equal(showWorker.enabled,false);assert.equal(showCalls.some(c=>c[0]==='fx'&&c[1]),false);
showFails=false;showWorker.handle({op:'show_config',config:{test:true},revision:3});await wait(showWorker);
assert.equal(showWorker.showReady,false,'configuration ACK is not proof of native input selection');
showWorker.handle({op:'fx',enabled:true,revision:4});await wait(showWorker);
assert.equal(showWorker.showReady,true);assert.equal(showWorker.enabled,true);
assert.deepEqual(showCalls.at(-1),['fx',true,{showConfigDigest:digest}]);
const colored={...event('four-ball'),ball_id:'ball-pink',sound_id:'bone_low',source_trial_id:'original'};
showWorker.handle({op:'emit',event:colored,queued_unix_ms:Date.now(),revision:4});
assert.deepEqual(showCalls.at(-1),['event',colored]);

// Service restart clears actual bindings. Polling withdraws the obsolete green
// readiness and never reapplies inputs, enables FX, or changes robot state.
remoteShow=showState();Object.assign(remoteShow.show_audio,{configured:false,config_digest:null,selection_matches:false});
await showWorker.poll();
assert.equal(showWorker.mappingMatches,false);assert.equal(showWorker.enabled,false);assert.equal(showWorker.requested,false);
assert.equal(showWorker.showReady,false);assert.equal(showStates.at(-1).status.connected_ball_count,0);
const configs=showCalls.filter(c=>c[0]==='config').length;
showWorker.handle({op:'fx',enabled:true,revision:5});await wait(showWorker);
assert.equal(showWorker.enabled,false);assert.equal(showCalls.filter(c=>c[0]==='config').length,configs,'only explicit reapply configures inputs');

// Explicit reapply + enable recovers. A partial connection is reported accurately
// (two balls, not four); this is not acoustic verification or a robot motion gate.
remoteShow=showState();remoteShow.show_audio.ball_inputs[2].connected=false;remoteShow.show_audio.ball_inputs[3].connected=false;
showWorker.handle({op:'show_config',config:{test:true},revision:6});await wait(showWorker);
showWorker.handle({op:'fx',enabled:true,revision:7});await wait(showWorker);
assert.equal(showWorker.enabled,true);assert.equal(showStates.at(-1).status.bound_ball_count,4);
assert.equal(showStates.at(-1).status.connected_ball_count,2);
remoteShow.show_audio.config_digest=changedDigest;remoteShow.fx_enabled=true;
await showWorker.poll();
assert.equal(showWorker.mappingMatches,false);assert.equal(showWorker.enabled,false);assert.equal(showWorker.muteConfirmed,false);
showWorker.handle({op:'emit',event:colored,queued_unix_ms:Date.now(),revision:7});
assert.equal(showCalls.filter(c=>c[0]==='event').length,1,'another client mapping must not receive our ball event');

// Native failure and a legacy server cannot claim readiness. The explicit reapply
// has to be acknowledged by the new readiness protocol before enabling.
remoteShow=showState();remoteShow.audio_ready=false;
showWorker.handle({op:'show_config',config:{test:true},revision:8});await wait(showWorker);
showWorker.handle({op:'fx',enabled:true,revision:9});await wait(showWorker);assert.equal(showWorker.enabled,false);
remoteShow=showState();delete remoteShow.show_audio.readiness_schema;
showWorker.handle({op:'show_config',config:{test:true},revision:10});await wait(showWorker);
showWorker.handle({op:'fx',enabled:true,revision:11});await wait(showWorker);assert.equal(showWorker.enabled,false);

// A pre-control poll cannot overwrite a newer successful mapping and enable.
remoteShow=showState();
let finishStaleMappingPoll;const realShowStatus=showAudio.status;
showAudio.status=()=>new Promise(resolve=>{finishStaleMappingPoll=resolve;});
const staleMappingPoll=showWorker.poll();showAudio.status=realShowStatus;
showWorker.handle({op:'show_config',config:{test:true},revision:12});await wait(showWorker);
showWorker.handle({op:'fx',enabled:true,revision:13});await wait(showWorker);
const obsolete=showState();obsolete.show_audio.config_digest=changedDigest;
finishStaleMappingPoll(obsolete);await staleMappingPoll;
assert.equal(showWorker.mappingMatches,true);assert.equal(showWorker.enabled,true);

// A poll begun during config control may carry the same revision but sample the
// previous config. It cannot overwrite the control's accepted digest/readiness.
let finishConfig,finishConfigPoll;
const savedConfigure=showAudio.configureShow;
showAudio.configureShow=()=>new Promise(resolve=>{finishConfig=resolve;});
showWorker.handle({op:'show_config',config:{test:true},revision:14});
showAudio.status=()=>new Promise(resolve=>{finishConfigPoll=resolve;});const pendingConfigPoll=showWorker.poll();
finishConfig({...showState().show_audio});await wait(showWorker);
finishConfigPoll(obsolete);await pendingConfigPoll;
assert.equal(showWorker.expectedConfigDigest,digest);assert.equal(showWorker.showReady,false);
showAudio.configureShow=savedConfigure;showAudio.status=realShowStatus;
showWorker.handle({op:'fx',enabled:true,revision:15});await wait(showWorker);assert.equal(showWorker.enabled,true);

// Connection loss revokes readiness immediately; no stale "connected inputs".
showAudio.status=async()=>{throw Error('private host failure');};await showWorker.poll();
assert.equal(showWorker.enabled,false);assert.equal(showWorker.mappingMatches,false);assert.equal(showWorker.engineReady,null);
assert.deepEqual(showWorker.ballInputs,[]);
console.log('Show readiness mocks passed: restart, config replacement, partial inputs, native failure, legacy protocol, explicit recovery, stale polls and loss of service');

// A new mapping supersedes an older in-flight enable even if its status request
// ignores abort. Configuration must never let that previous ON reach the service.
let releaseOldShowStatus;const supersedeCalls=[];
const supersedeAudio={onError:()=>{},cancel(){},async setVolume(){},
  async setFX(value){supersedeCalls.push(value);},
  async configureShow(){return {...showState().show_audio};},
  status(){return new Promise(resolve=>{releaseOldShowStatus=resolve;});},emit(){}};
const supersedeWorker=new DashboardAudio({audio:supersedeAudio});
supersedeWorker.handle({op:'show_config',config:{test:true},revision:1});await wait(supersedeWorker);
supersedeWorker.handle({op:'fx',enabled:true,revision:2});await tick();
supersedeWorker.handle({op:'show_config',config:{test:true},revision:3});
releaseOldShowStatus(showState());await wait(supersedeWorker);
assert.deepEqual(supersedeCalls,[]);assert.equal(supersedeWorker.enabled,false);
