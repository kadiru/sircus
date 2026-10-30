import assert from 'node:assert/strict';
import {BoostControls, BOOST_DB} from '../web/boost-controls.mjs';

const ready={route:'native',audio_ready:true,peak_limiter_ready:true,
  boost_db:0,boost_requested_db:0,boost_pending:false,fx_enabled:false,volume:0.35};
const defer=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};

// Import/constructor/status readback never sends any mutation or starts sound.
const calls=[];
const control=new BoostControls({request:async(path,body)=>{
  calls.push({path,body});return {...ready,boost_db:body.boost_db,boost_requested_db:body.boost_db};
}});
assert.equal(control.view().checked,false);
assert.equal(control.view().disabled,true);
control.observe(ready);
assert.equal(control.view().checked,false);
assert.equal(control.view().disabled,false);
assert.equal(calls.length,0);
await control.setBoost(true);
assert.deepEqual(calls,[{path:'boost',body:{boost_db:BOOST_DB}}]);
assert.equal(control.view().checked,true);
assert.match(control.view().text,/3× gain enabled/);
assert.equal(control.state.fx_enabled,false,'boost never implicitly enables FX');
assert.equal(control.state.volume,0.35,'master volume is preserved');
await control.setBoost(false);
assert.deepEqual(calls.at(-1),{path:'boost',body:{boost_db:0}});
assert.equal(control.view().checked,false);

// HTTP acknowledgment only reports pending; native readback establishes applied gain.
const post=defer();
const pending=new BoostControls({request:()=>post.promise});
pending.observe(ready);
const setting=pending.setBoost(true);
assert.equal(pending.view().pending,true);
assert.equal(pending.view().disabled,true);
assert.match(pending.view().text,/waiting for native confirmation/);
assert.doesNotMatch(pending.view().text,/gain enabled/);
post.resolve({...ready,boost_requested_db:BOOST_DB,boost_pending:true});
await setting;
assert.equal(pending.view().pending,true);
assert.equal(pending.state.boost_db,0);
assert.doesNotMatch(pending.view().text,/gain enabled/);
await assert.rejects(pending.setBoost(false),/pending boost change/);
pending.observe({...ready,boost_db:BOOST_DB,boost_requested_db:BOOST_DB});
assert.equal(pending.view().pending,false);
assert.match(pending.view().text,/gain enabled/);

// Unavailable/Logic/older backends cannot alter gain. Mute/Stop remain separate.
for(const status of [null,{...ready,route:'logic'},{...ready,audio_ready:false},
  {...ready,peak_limiter_ready:false},{route:'native',audio_ready:true}]){
  let mutated=false;
  const unavailable=new BoostControls({request:async()=>{mutated=true;}});
  if(status)unavailable.observe(status);
  assert.equal(unavailable.view().disabled,true);
  await assert.rejects(unavailable.setBoost(true),/Native audio/);
  assert.equal(mutated,false);
}
control.observe({...ready,boost_db:BOOST_DB,boost_requested_db:BOOST_DB,peak_limiter_ready:false});
assert.equal(control.view().disabled,true,'normal gain also requires a ready limiter');
const callsBeforeLostLimiter=calls.length;
await assert.rejects(control.setBoost(false),/Native audio/);
assert.equal(calls.length,callsBeforeLostLimiter,'no false normal-gain request when limiter is unavailable');
assert.equal(control.view().checked,true,'last confirmed gain stays visible; no false cleared ACK');

// A failed change is never represented as confirmed gain; no automatic retries.
let attempted=0;
const failed=new BoostControls({request:async()=>{attempted++;throw new Error('network');}});
failed.observe(ready);
await assert.rejects(failed.setBoost(true),/network/);
assert.equal(failed.view().checked,false);
assert.equal(failed.view().disabled,true);
assert.match(failed.view().error,/not confirmed/);
assert.equal(attempted,1);
failed.observe(ready);
assert.equal(failed.view().checked,false);
assert.equal(attempted,1);

// A partial legacy response cannot be mistaken for native confirmation.
const legacy=new BoostControls({request:async()=>({accepted:true})});
legacy.observe(ready);
await assert.rejects(legacy.setBoost(true),/confirmation is unavailable/);
assert.equal(legacy.view().checked,false);
assert.match(legacy.view().error,/not confirmed/);
console.log('Boost controls passed: default OFF, explicit native-only gain, pending ACK, mute/volume independence, failures and no playback');
