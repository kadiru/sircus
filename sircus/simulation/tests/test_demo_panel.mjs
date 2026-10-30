import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
import {DemoPanel,LocalDemoAudio,normalizeDemo,learningCurveMarkup,DEMO_BUNDLE_URL,DEMO_SOUNDS} from '../web/demo.js';

const saved=JSON.parse(await readFile(new URL('../web/media/demo-reel/showcase.json',import.meta.url),'utf8'));
const copy=value=>structuredClone(value),response=value=>({ok:true,json:async()=>copy(value)});
const deferred=()=>{let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};};
const turn=()=>new Promise(resolve=>setImmediate(resolve));
class Node{
  constructor(){this.value='';this.textContent='';this.innerHTML='';this.checked=false;this.disabled=false;this.hidden=false;this.children=[];this.listeners={};this.attributes={};this.className='';this.style={setProperty:(k,v)=>{this.style[k]=v;}};this.classList={toggle(){},remove(){}};}
  addEventListener(name,fn){this.listeners[name]=fn;}setAttribute(name,value){this.attributes[name]=value;}append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=nodes;}scrollIntoView(){}
}
function fixture({audio,fetchImpl}={}){
  const nodes={},requests=[],frames=new Map(),views=[],sounds=[],node=id=>nodes[id]??=new Node();let now=0,frameId=0;
  const document={getElementById:node,createElement:()=>new Node(),createTextNode:text=>({textContent:text})};
  const viewer={replay:(clip,state)=>views.push({clip,state}),pause:()=>views.push({pause:true}),seek:time=>views.push({time}),setBallColor:color=>views.push({color})};
  const localAudio=audio||{setEnabled:value=>sounds.push(['enabled',value]),setVolume:value=>sounds.push(['volume',value]),begin:()=>{sounds.push(['begin']);return Promise.resolve(false);},nextClip:()=>sounds.push(['next']),stop:()=>sounds.push(['stop']),carryAt:value=>sounds.push(['carry',value]),contact:(...args)=>sounds.push(['contact',...args])};
  const panel=new DemoPanel({document,viewer,audio:localAudio,fetchImpl:(...args)=>{requests.push(args);return fetchImpl?fetchImpl(...args):Promise.resolve(response(saved));},now:()=>now,requestFrame:fn=>{const id=++frameId;frames.set(id,fn);return id;},cancelFrame:id=>frames.delete(id)});
  return {panel,nodes,node,requests,frames,views,sounds,async load(){return panel.load();},advance(ms){now=ms;const callbacks=[...frames.values()];frames.clear();callbacks.forEach(fn=>fn(now));}};
}
function audioFixture({resume=Promise.resolve(),sampleFetch}={}){
  const requests=[],voices=[],statuses=[],gains=[],contexts=[];
  const context={destination:{},currentTime:0,resume:()=>resume,decodeAudioData:async bytes=>({bytes}),createGain(){const gain={gain:{value:1},connect(){},disconnect(){}};gains.push(gain);return gain;},createBufferSource(){const voice={started:0,stopped:0,connect(){},disconnect(){},start(){this.started++;},stop(when){if(when!==undefined){this.stopAt=when;return;}this.stopped++;this.onended?.();}};voices.push(voice);return voice;}};
  const audio=new LocalDemoAudio({contextFactory:()=>{contexts.push(context);return context;},fetchImpl:(...args)=>{requests.push(args);return sampleFetch?sampleFetch(...args):Promise.resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(8)});},onStatus:text=>statuses.push(text)});
  return {audio,requests,voices,statuses,gains,contexts};
}

test('saved demo validates provenance, fixed action, paired misses and separate selected success',()=>{
  const bundle=normalizeDemo(copy(saved));assert.deepEqual(bundle.fixed.clips.map(c=>c.ball.label),['Pink','Purple','Green','Gray']);assert.equal(bundle.fixed.clips.length,4);assert.deepEqual(bundle.learning.same_layout_clips.map(c=>c.train_scenes),[8,24,64,128]);assert.ok(bundle.learning.same_layout_clips.every(c=>!c.result.success));assert.equal(bundle.learning.success_example.result.success,true);
  for(const mutate of [b=>{b.hardware_enabled=true;},b=>{b.fixed.clips[1].action.gap_mm++;},b=>{b.fixed.clips[0].events[1].event_id=b.fixed.clips[0].events[0].event_id;},b=>{b.fixed.clips[0].events[0].trial_id='wrong-trial';},b=>{b.learning.same_layout_clips.reverse();}]){const invalid=copy(saved);mutate(invalid);assert.throws(()=>normalizeDemo(invalid));}
});

test('loading reads one static bundle with sound disabled and sends no control request',async()=>{
  const h=fixture();assert.equal(h.node('demoSound').checked,false);assert.equal(h.requests.length,0);assert.equal(await h.load(),true);assert.deepEqual(h.requests.map(row=>row[0]),[DEMO_BUNDLE_URL]);assert.ok(h.requests.every(([,options])=>!options.method));assert.equal(h.sounds.filter(row=>row[0]==='begin').length,0);assert.equal(h.nodes.playDemo.disabled,false);assert.equal(h.nodes.stopDemo.disabled,true);assert.match(h.nodes.bundleEvidence.textContent,/front-negative-y/);assert.doesNotMatch(h.nodes.bundleEvidence.textContent,/object Object/);h.panel.destroy();
});

test('one play runs exactly four independent numerical shots and never repeats automatically',async()=>{
  const h=fixture();await h.load();assert.equal(h.panel.play(),true);assert.equal(h.panel.play(),false);assert.equal(h.panel.selectMode('learning'),false);assert.equal(h.nodes.modelResult.textContent,'Awaiting replay');let clock=0;
  h.advance(1400);assert.equal(h.sounds.at(-1)[0],'carry');assert.equal(h.sounds.at(-1)[1],true);assert.match(h.nodes.stageStatus.textContent,/Carry/);
  h.advance(3000);assert.match(h.nodes.stageStatus.textContent,/Settle/);assert.equal(h.sounds.at(-1)[1],true);
  for(let i=0;i<4;i++){clock+=(h.panel.duration+1.3)*1000;h.advance(clock);}
  assert.equal(h.panel.playing,false);assert.equal(h.panel.index,3);assert.equal(h.nodes.reelProgress.value,1);assert.equal(h.nodes.stopDemo.disabled,true);assert.equal(h.frames.size,0);assert.equal(h.sounds.filter(row=>row[0]==='begin').length,1);assert.match(h.nodes.demoMessage.textContent,/Four independent simulations finished/);assert.deepEqual(h.views.filter(row=>row.color).slice(-4).map(row=>row.color),saved.fixed.clips.map(c=>c.ball.color));assert.equal(h.requests.length,1);h.panel.destroy();
});

test('Stop invalidates queued animation and local audio before another user-started replay',async()=>{
  const h=fixture();await h.load();h.panel.play();const stale=[...h.frames.values()][0];h.panel.stop();const count=h.views.length,soundCount=h.sounds.length;stale(90000);assert.equal(h.views.length,count);assert.equal(h.sounds.length,soundCount);assert.equal(h.frames.size,0);assert.equal(h.nodes.stopDemo.disabled,true);assert.match(h.nodes.stageStatus.textContent,/Stopped/);assert.equal(h.panel.play(),true);assert.equal(h.panel.index,0);h.panel.destroy();
});

test('learning mode preserves paired failures and labels the successful clip as a separate layout',async()=>{
  const h=fixture();await h.load();assert.equal(h.panel.selectMode('learning'),true);assert.equal(h.nodes.learningEvidence.hidden,false);assert.equal(h.nodes.fixedEvidence.hidden,true);assert.match(h.nodes.clipTitle.textContent,/N = 8/);assert.match(h.nodes.clipDescription.textContent,/saved miss is retained/);assert.equal(h.nodes.learnedScore.textContent,'33 / 48');assert.match(h.nodes.learnedInterval.textContent,/54.7%–80.1%/);h.panel.showClip(4);assert.equal(h.nodes.clipTitle.textContent,'A separate successful example.');assert.match(h.nodes.clipDescription.textContent,/different held-out layout/);assert.ok(h.nodes.learningChart.innerHTML.includes('59.0%'));assert.ok(h.nodes.learningChart.innerHTML.includes('69.4%'));const curve=learningCurveMarkup(saved.learning.learning_curve_aggregate);assert.match(curve,/95% layout interval/);h.panel.destroy();
});

test('expired final contacts use real elapsed time after a render stall',async()=>{
  const h=fixture();await h.load();h.panel.play();h.advance((h.panel.duration+.5)*1000);const contacts=h.sounds.filter(row=>row[0]==='contact');assert.ok(contacts.length);for(const row of contacts)assert.equal(row.at(-1),h.panel.duration+.5);h.panel.destroy();
});

test('local audio stays inert until Play and mute during delayed resume prevents sample work',async()=>{
  const delayed=deferred(),h=audioFixture({resume:delayed.promise});assert.equal(h.contexts.length,0);h.audio.setEnabled(true);assert.equal(h.contexts.length,0);const starting=h.audio.begin();assert.equal(h.contexts.length,1);assert.equal(h.requests.length,0);h.audio.setEnabled(false);delayed.resolve();assert.equal(await starting,false);assert.equal(h.requests.length,0);assert.equal(h.voices.length,0);assert.equal(h.audio.enabled,false);
});

test('late sample completion cannot restart playback after Stop or mute',async()=>{
  const delayed=deferred(),h=audioFixture({sampleFetch:()=>delayed.promise});h.audio.setEnabled(true);const starting=h.audio.begin();await turn();assert.equal(h.requests.length,5);assert.ok(h.requests.every(([url])=>url.startsWith('/demo-audio/')));h.audio.carryAt(true);h.audio.setEnabled(false);delayed.resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(8)});assert.equal(await starting,false);assert.equal(h.voices.length,0);assert.equal(h.audio.ready,false);assert.ok(h.requests.every(([,options])=>options.signal.aborted));
});

test('ready local sound uses original samples, deduplicates events, expires stale cues and mutes every voice',async()=>{
  const h=audioFixture();h.audio.setEnabled(true);assert.equal(await h.audio.begin(),true);assert.deepEqual(h.requests.map(([url])=>url).sort(),Object.values(DEMO_SOUNDS).map(file=>'/demo-audio/'+file).sort());h.audio.setVolume(.2);assert.equal(h.gains[0].gain.value,.2);h.audio.carryAt(true);const drum=h.voices.at(-1);assert.equal(drum.loop,true);assert.equal(drum.stopAt,3);h.audio.carryAt(true);assert.equal(h.voices.length,1);h.audio.carryAt(false);assert.equal(drum.stopped,1);
  const event=saved.fixed.clips[0].events.find(e=>e.type==='contact');assert.equal(h.audio.contact(event,'ball-pink',event.t+.02),true);assert.equal(h.audio.contact(event,'ball-pink',event.t+.03),false);assert.equal(h.audio.contact({...event,event_id:'late'},'ball-pink',event.t+2),false);assert.equal(h.audio.contact({...event,event_id:'future'},'ball-pink',event.t-.1),false);assert.equal(h.audio.contact({...event,event_id:'physical',evidence:'physical'},'ball-pink',event.t),false);assert.equal(h.voices.length,2);h.audio.carryAt(true);h.audio.setEnabled(false);assert.ok(h.voices.every(v=>v.stopped===1));assert.equal(h.audio.voices.size,0);assert.match(h.statuses.at(-1),/Sound is off/);
});

test('unavailable optional audio fails silently without blocking numerical playback',async()=>{
  const local=audioFixture({sampleFetch:async()=>({ok:false})});local.audio.setEnabled(true);const h=fixture({audio:local.audio});await h.load();assert.equal(h.panel.play(),true);await turn();assert.equal(h.panel.playing,true);assert.match(local.statuses.at(-1),/continues silently/);assert.equal(local.voices.length,0);h.advance(3000);assert.match(h.nodes.stageStatus.textContent,/Settle/);h.panel.destroy();
});


test('an audio device failure during a cue never interrupts the numerical reel',async()=>{const h=audioFixture();h.audio.setEnabled(true);await h.audio.begin();h.contexts[0].createBufferSource=()=>{throw Error('Audio device closed');};const event=saved.fixed.clips[0].events.find(e=>e.type==='contact');assert.doesNotThrow(()=>h.audio.contact(event,'ball-pink',event.t));assert.equal(h.audio.active,false);assert.equal(h.audio.voices.size,0);assert.match(h.statuses.at(-1),/visual replay continues silently/);});
