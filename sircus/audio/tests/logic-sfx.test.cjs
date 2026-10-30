// Mocked host checks; these do not claim real Logic audio or BLE validation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
let sent = [], now = 0;
const parameters = {'Input channel':16, 'Output channel':1, 'Mute and release':0};
class Event { send() { sent.push({...this, type:this.constructor.name}); } }
class NoteOn extends Event {} class NoteOff extends Event {}
class ControlChange extends Event {} class PitchBend extends Event {}
const context = vm.createContext({NoteOn, NoteOff, ControlChange, PitchBend,
  GetParameter:name=>parameters[name], Date:{now:()=>now}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../logic-sfx.js'),'utf8'),context);
function send(Type, fields) { context.HandleMIDI(Object.assign(new Type(),{channel:16},fields)); }
function notes(type) { return sent.filter(e=>e.type===type); }
send(NoteOn,{channel:1,pitch:60,velocity:40});
send(ControlChange,{number:5,value:127});
assert.equal(sent.length,0,'other inputs and raw ODD CCs must be consumed');
send(NoteOn,{pitch:60,velocity:127});
assert.equal(notes('NoteOn')[0].velocity,90);
assert.equal(notes('NoteOn')[0].channel,1);
send(NoteOn,{pitch:60,velocity:30});
assert.equal(notes('NoteOff').length,1,'retrigger releases first');
send(NoteOn,{pitch:60,velocity:0});
assert.equal(notes('NoteOff').length,2,'velocity-zero releases');
send(NoteOn,{pitch:65,velocity:30});
parameters['Output channel']=2; context.ParameterChanged(1,2);
assert.equal(notes('NoteOff').at(-1).channel,1,'release owns original channel');
send(NoteOn,{pitch:67,velocity:30}); now=2501; context.ProcessMIDI();
assert.equal(notes('NoteOff').at(-1).pitch,67,'watchdog releases missing off');
parameters['Mute and release']=1; context.ParameterChanged(2,1); sent=[];
send(NoteOn,{pitch:72,velocity:80}); send(PitchBend,{value:8191});
assert.equal(sent.length,0,'mute blocks notes and bends');
parameters['Mute and release']=0;
send(PitchBend,{value:10000}); assert.equal(sent.at(-1).value,8191);
send(NoteOn,{pitch:72,velocity:50}); send(ControlChange,{number:120,value:0});
assert.equal(notes('NoteOff').at(-1).pitch,72);
assert.ok(sent.some(e=>e.type==='ControlChange' && e.number===123));
assert.equal(sent.at(-1).value,0,'panic centers bend');
console.log('PASS: Scripter isolation, velocity cap, retrigger, note ownership, watchdog, mute and panic');
