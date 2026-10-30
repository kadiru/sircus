import assert from 'node:assert/strict';
import {FXControls} from '../integration/fx-controls.mjs';
import {SircusAudio} from '../integration/sircus-audio-client.mjs';
const tick=()=>new Promise(resolve=>setTimeout(resolve,0));

// A slow epoch read must not hold up OFF or later submit an obsolete ON.
for(const panic of [false,true]){
  let resolveStatus;const calls=[];
  const control=new FXControls({clientId:'panel',request:(path,body)=>{
    calls.push({path,body});
    if(path==='status')return new Promise(resolve=>{resolveStatus=resolve;});
    return Promise.resolve({fx_enabled:false});
  }});
  const on=control.setFX(true);const rejected=assert.rejects(on,/superseded/);
  await (panic?control.panic():control.setFX(false));
  assert.equal(calls.length,2);assert.equal(calls[1].path,panic?'panic':'fx');
  assert.equal(calls[1].body.control_sequence,2);
  resolveStatus({control_epoch:'epoch'});await rejected;
  assert.equal(calls.length,2);
}

// Requests already dispatched carry revisions, so reverse server arrival is detectable.
let resolveOn;const delivered=[];
const ordered=new FXControls({clientId:'panel',request:async(path,body)=>{
  if(path==='status')return {control_epoch:'observed'};
  delivered.push({path,body});
  if(body.enabled===true)await new Promise(resolve=>{resolveOn=resolve;});
  return {fx_enabled:body.enabled===true};
}});
const on=ordered.setFX(true);await tick();await ordered.panic();
assert.equal(delivered[0].body.control_sequence,1);
assert.equal(delivered[0].body.control_epoch,'observed');
assert.equal(delivered[1].body.control_sequence,2);
resolveOn();await on;

// The provided server-side client uses the same protocol; it cannot enable an old service.
const calls=[];
const audio=new SircusAudio({baseURL:'http://example.invalid',token:'TEST_ONLY',fetchImpl:async(url,options)=>{
  calls.push({url,body:options.body&&JSON.parse(options.body)});
  return {ok:true,json:async()=>url.endsWith('/status')?{control_epoch:'epoch'}:{fx_enabled:false}};
}});
await audio.setFX(true);await audio.setFX(false);await audio.panic();
assert.deepEqual(calls.map(row=>row.url.split('/').pop()),['status','fx','fx','panic']);
const controls=calls.filter(row=>row.body);
assert.ok(controls.every(row=>row.body.control_client_id===controls[0].body.control_client_id));
assert.ok(controls[0].body.control_sequence<controls[1].body.control_sequence);
assert.ok(controls[1].body.control_sequence<controls[2].body.control_sequence);
const old=new SircusAudio({baseURL:'http://example.invalid',fetchImpl:async()=>({ok:true,json:async()=>({})})});
await assert.rejects(old.setFX(true),/does not support ordered/);
await old.setFX(false); // Mute never waits for compatible status.

// Cancellation during status acquisition also invalidates the not-yet-posted enable.
let releaseStatus;const cancelCalls=[];
const cancelled=new SircusAudio({baseURL:'http://example.invalid',fetchImpl:async(url)=>{
  cancelCalls.push(url);
  return {ok:true,json:()=>new Promise(resolve=>{releaseStatus=resolve;})};
}});
const pending=cancelled.setFX(true);const rejection=assert.rejects(pending,/superseded/);
await tick();cancelled.cancel();releaseStatus({control_epoch:'epoch'});await rejection;
assert.equal(cancelCalls.length,1);
console.log('Ordered FX controls passed: delayed/reversed ON, OFF/panic priority, shared client metadata, legacy fail-closed and cancellation');
