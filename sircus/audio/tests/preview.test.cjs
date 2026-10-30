// Test generated preview logic with fake Web Audio. No sound or browser needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = {'#level':{value:'.15'}, '#mute':{checked:false}, '#status':{}, '#stop':{}};
const buttons = ['boing','squeak','plop','spring','celebration','wong','captured','missed','unknown']
  .map(name=>({dataset:{play:name}}));
let created = 0, oscillators = [], timers = new Map(), seq = 0;
const parameter = () => ({value:0,setValueAtTime(){},linearRampToValueAtTime(){}});
class AudioContext {
  constructor(){created++;this.currentTime=0;this.destination={};}
  async resume(){}
  createGain(){return {gain:parameter(),connect(to){return to;},disconnect(){}};}
  createDynamicsCompressor(){return {threshold:parameter(),ratio:parameter(),connect(to){return to;}};}
  createOscillator(){const o={frequency:parameter(),connect(to){return to;},start(){},stop(){this.stopped=true;}};oscillators.push(o);return o;}
}
const context = vm.createContext({AudioContext, document:{querySelector:s=>elements[s],querySelectorAll:()=>buttons,addEventListener(){}},
  window:{addEventListener(){}},setTimeout:fn=>{timers.set(++seq,fn);return seq;},clearTimeout:id=>timers.delete(id)});
vm.runInContext(source,context);
(async()=>{
  assert.equal(created,0,'loading page must not create an audio context');
  elements['#mute'].checked=true;
  await buttons[0].onclick(); assert.equal(created,0,'muted click must stay silent');
  elements['#mute'].checked=false;
  for(const button of buttons){
    oscillators=[]; await button.onclick();
    assert.ok(oscillators.length>0,button.dataset.play+' should schedule tones');
    elements['#stop'].onclick();
    assert.ok(oscillators.every(o=>o.stopped),'stop cancels every oscillator');
    assert.equal(timers.size,0,'stop cancels pending status timers');
  }
  console.log('PASS: preview has no autoplay; all nine buttons, mute and stop work with fake Web Audio');
})().catch(error=>{console.error(error);process.exitCode=1;});
