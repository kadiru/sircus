import test from 'node:test';
import assert from 'node:assert/strict';
import {TapTestControls,testView} from '../web/tap-test.mjs';
const status={control_epoch:'fresh',audio_ready:true,route:'native',show_audio:{config_digest:'digest',selection_matches:true,connected_ball_count:4},tap_test:{supported:true,active:false,balls:[]}};
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
test('start uses fresh epoch and duration; no synthetic tap',async()=>{
  const calls=[];const c=new TapTestControls({clientId:'test',request:async(...args)=>{calls.push(args);return status;}});
  await c.start();assert.equal(calls[1][1].duration_s,15);assert.equal(calls[1][1].control_epoch,'fresh');assert.equal(calls[1][0],'tap-test');assert.equal(c.pending,false);
});
test('stop overtakes delayed readiness without launching test',async()=>{
  const gate=deferred(),calls=[];const c=new TapTestControls({clientId:'test',request:async(path,body)=>{calls.push([path,body]);return path==='status'?gate.promise:status;}});
  const start=c.start(30);await c.stop();gate.resolve(status);await assert.rejects(start,/cancelled/);assert.equal(calls.length,2);assert.equal(calls[1][1].action,'stop');
});
test('lost start reply triggers mute and never retries',async()=>{
  const calls=[];const c=new TapTestControls({clientId:'test',request:async(path,body)=>{calls.push([path,body]);if(body?.action==='start')throw new Error('timeout');return status;}});
  await assert.rejects(c.start(20),/timeout/);assert.deepEqual(calls.map(c=>c[1]?.action),[undefined,'start','stop']);assert.equal(c.pending,false);
});
test('invalid duration performs no HTTP calls',async()=>{
  const c=new TapTestControls({request:()=>assert.fail('unexpected HTTP')});for(const n of [0,121,1.2,NaN])await assert.rejects(c.start(n));
});
test('connected is distinct from responding, disconnect overrides old tap',()=>{
  const s={...status,tap_test:{supported:true,active:true,remaining_s:14.2,balls:[{bound:true,connected:true,taps:0},{bound:true,connected:true,taps:3,last_tap_age_s:.2},{bound:true,connected:false,taps:2}]}};
  const v=testView(s);assert.equal(v.canStart,false);assert.match(v.text,/15 seconds/);assert.equal(v.balls[0].detail,'Waiting for tap');assert.match(v.balls[1].detail,/3 taps/);assert.equal(v.balls[2].detail,'Disconnected');
});
