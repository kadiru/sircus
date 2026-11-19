import {createViewer,replayTimeline} from './operations.js';

export const DEMO_BUNDLE_URL='/media/demo-reel/showcase.json';
export const DEMO_SOUNDS=Object.freeze({'ball-pink':'cartoon_boing.wav','ball-purple':'cartoon_spring.wav','ball-green':'boing.wav','ball-gray':'cartoon_squeak.wav',carry:'drum_roll.wav'});
const PHASE_NAMES={grip:'Grip · illustrated',orient_board:'Aim board · endpoint illustration',move_dropper:'Carry · endpoint illustration',settle:'Settle',release:'Release',ball_flight:'Numerical ball flight'};
const EVENT_TYPES=new Set(['carry_start','carry_stop','release','contact','outcome']);
const pct=value=>`${(value*100).toFixed(1)}%`;
const displayOutcome=clip=>clip.result.success?'Captured in model':String(clip.result.outcome||'unknown').replaceAll('_',' ');
const comparePose=(a,b)=>JSON.stringify(a)===JSON.stringify(b);

export function normalizeDemo(value){
  if(value?.schema!=='sircus-demo-reel-v1'||value.data_origin!=='synthetic'||value.hardware_enabled!==false||value.native_paths_checked!==false||value.physically_verified!==false)throw Error('The replay bundle must contain explicitly synthetic, non-hardware evidence.');
  const fixed=value.fixed?.clips,paired=value.learning?.same_layout_clips,success=value.learning?.success_example;
  if(!Array.isArray(fixed)||fixed.length!==4||!Array.isArray(paired)||paired.length!==4||!success)throw Error('The fixed and learning replay sets are incomplete.');
  const ids=new Set(),events=new Set();
  for(const clip of [...fixed,...paired,success]){
    if(typeof clip.id!=='string'||!clip.id||ids.has(clip.id)||!clip.trial_id||clip.data_origin!=='synthetic'||clip.native_paths_checked!==false||clip.physically_verified!==false||typeof clip.result?.success!=='boolean'||!clip.scene?.board||!clip.scene?.table||!clip.scene?.bin||!clip.scene?.ball)throw Error('A replay has invalid identity, scene or provenance.');
    ids.add(clip.id);const timeline=replayTimeline(clip);
    if(!timeline.choreography||!timeline.choreography.initial_board_pose||!Array.isArray(timeline.choreography.initial_arm10_boxes))throw Error('A replay is missing its illustrated endpoint sequence.');
    if(!Object.hasOwn(DEMO_SOUNDS,clip.ball?.id)||!/^#[0-9a-f]{6}$/i.test(clip.ball?.color||''))throw Error('A replay ball identity or color is invalid.');
    if(!Number.isFinite(clip.board_tilt_deg)||!Number.isFinite(clip.drop_height_mm)||!Array.isArray(clip.events))throw Error('A replay is missing its action or contact record.');
    let previous=-Infinity;
    for(const event of clip.events){if(typeof event.event_id!=='string'||events.has(event.event_id)||event.trial_id!==clip.trial_id||!Number.isFinite(event.t)||event.t<0||event.t<previous||!EVENT_TYPES.has(event.type)||event.evidence!=='synthetic'||event.physically_verified!==false)throw Error('A replay contains invalid, duplicated or misattributed events.');events.add(event.event_id);previous=event.t;}
  }
  if(new Set(fixed.map(clip=>clip.ball.id)).size!==4||fixed.some(clip=>!comparePose(clip.scene,fixed[0].scene)||!comparePose(clip.action,fixed[0].action)))throw Error('The four-ball routine must preserve one fixed scene and action.');
  if(paired.some((clip,i)=>clip.train_scenes!==[8,24,64,128][i]||!clip.paired_layout||!clip.checkpoint_id))throw Error('The learning progression must preserve the four paired frozen checkpoints.');
  if(!Array.isArray(value.learning.learning_curve)||!Array.isArray(value.learning.learning_curve_aggregate))throw Error('Learning evidence is missing.');
  for(const row of value.learning.learning_curve_aggregate){if(!Number.isSafeInteger(row.train_scenes)||row.train_scenes<1||!Number.isFinite(row.mean_success_fraction)||row.mean_success_fraction<0||row.mean_success_fraction>1||!Array.isArray(row.layout_ci95)||row.layout_ci95.length!==2||row.layout_ci95.some(x=>!Number.isFinite(x)||x<0||x>1)||row.layout_ci95[0]>row.layout_ci95[1])throw Error('Learning curve values or intervals are invalid.');}
  return value;
}

export function learningCurveMarkup(rows){
  const sorted=[...rows].sort((a,b)=>a.train_scenes-b.train_scenes),w=660,h=270,left=55,right=20,top=26,bottom=49;
  if(!sorted.length)return '';
  const lo=Math.log2(sorted[0].train_scenes),hi=Math.log2(sorted.at(-1).train_scenes),x=n=>left+(hi===lo?.5:(Math.log2(n)-lo)/(hi-lo))*(w-left-right),y=n=>h-bottom-n*(h-top-bottom);
  const lines=[0,.25,.5,.75,1].map(n=>`<line x1="${left}" x2="${w-right}" y1="${y(n)}" y2="${y(n)}" stroke="#e2e6dc"/><text x="${left-12}" y="${y(n)+4}" text-anchor="end" fill="#667e7b" font-size="11">${Math.round(n*100)}%</text>`).join('');
  const path=sorted.map((r,i)=>`${i?'L':'M'}${x(r.train_scenes)},${y(r.mean_success_fraction)}`).join(' ');
  const points=sorted.map(r=>`<g><title>N ${r.train_scenes}: ${pct(r.mean_success_fraction)}; 95% layout interval ${pct(r.layout_ci95[0])}–${pct(r.layout_ci95[1])}</title><line x1="${x(r.train_scenes)}" x2="${x(r.train_scenes)}" y1="${y(r.layout_ci95[0])}" y2="${y(r.layout_ci95[1])}" stroke="#9ebfb1" stroke-width="3"/><circle cx="${x(r.train_scenes)}" cy="${y(r.mean_success_fraction)}" r="6" fill="#14858a" stroke="#fffef8" stroke-width="2"/><text x="${x(r.train_scenes)}" y="${y(r.mean_success_fraction)-13}" text-anchor="middle" font-size="12" font-weight="700" fill="#153d42">${pct(r.mean_success_fraction)}</text><text x="${x(r.train_scenes)}" y="${h-bottom+22}" text-anchor="middle" font-size="12" fill="#667e7b">${r.train_scenes}</text></g>`).join('');
  return `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">${lines}<path d="${path}" fill="none" stroke="#14858a" stroke-width="2.5"/>${points}<text x="${w/2}" y="${h-5}" text-anchor="middle" font-size="11" fill="#667e7b">Training layouts · N</text></svg>`;
}

/** Optional, browser-local original samples. No robot or service API is used. */
export class LocalDemoAudio {
  constructor({fetchImpl=(...args)=>globalThis.fetch(...args),contextFactory=()=>new (globalThis.AudioContext||globalThis.webkitAudioContext)(),onStatus=()=>{},setTimeoutImpl=(fn,ms)=>globalThis.setTimeout(fn,ms),clearTimeoutImpl=id=>globalThis.clearTimeout(id)}={}){
    this.fetch=fetchImpl;this.factory=contextFactory;this.onStatus=onStatus;this.setTimeout=setTimeoutImpl;this.clearTimeout=clearTimeoutImpl;this.enabled=false;this.volume=.5;this.context=null;this.gain=null;this.buffers=new Map();this.voices=new Set();this.requests=new Set();this.seen=new Set();this.generation=0;this.active=false;this.ready=false;this.carry=null;this.wantsCarry=false;
  }
  say(text){this.onStatus(text);}
  setEnabled(value){this.enabled=!!value;if(!this.enabled){this.stop();this.say('Sound is off. Enable it, then press Play.');}else this.say('Local samples selected. Press Play to enable sound.');}
  setVolume(value){this.volume=Math.max(0,Math.min(1,Number(value)||0));if(this.gain)this.gain.gain.value=this.volume;}
  async sample(file,epoch){
    if(this.buffers.has(file))return;
    const controller=new AbortController();this.requests.add(controller);let timer;
    const deadline=new Promise((_,reject)=>{timer=this.setTimeout(()=>{controller.abort();reject(Error('Local sample request timed out'));},8000);});
    try{const buffer=await Promise.race([(async()=>{const response=await this.fetch('/demo-audio/'+file,{cache:'force-cache',signal:controller.signal});if(!response.ok)throw Error('Local sample unavailable');return this.context.decodeAudioData(await response.arrayBuffer());})(),deadline]);if(epoch===this.generation)this.buffers.set(file,buffer);}
    finally{this.clearTimeout(timer);this.requests.delete(controller);}
  }
  begin(){
    this.stop();if(!this.enabled)return Promise.resolve(false);const epoch=this.generation;this.active=true;this.say('Loading local demo samples; visual replay continues.');
    try{if(!this.context){this.context=this.factory();this.gain=this.context.createGain();this.gain.connect(this.context.destination);this.gain.gain.value=this.volume;}}
    catch{this.active=false;this.context=null;this.gain=null;this.say('Local sound is unavailable in this browser. Visual replay still works.');return Promise.resolve(false);}
    // Called from the Play gesture before any fetch or await, for autoplay rules.
    let resume;try{resume=this.context.resume();}catch(error){resume=Promise.reject(error);}
    return Promise.resolve(resume).then(()=>epoch===this.generation&&this.active?Promise.all(Object.values(DEMO_SOUNDS).map(file=>this.sample(file,epoch))):false).then(()=>{
      if(epoch!==this.generation||!this.enabled||!this.active)return false;this.ready=true;this.say('Local demo samples · synthetic contacts only.');if(this.wantsCarry)this.startCarry();return true;
    }).catch(()=>{if(epoch===this.generation){this.stop();this.say('Local sound unavailable; replay continues silently. Press Play to retry.');}return false;});
  }
  voice(file,{loop=false,scale=1,maxSeconds=null}={}){
    if(!this.enabled||!this.active||!this.ready||!this.buffers.has(file))return null;
    if(this.voices.size>=10){const oldest=[...this.voices].find(voice=>voice!==this.carry);if(oldest)this.stopVoice(oldest);}
    let source,gain;
    try{source=this.context.createBufferSource();gain=this.context.createGain();source.buffer=this.buffers.get(file);source.loop=loop;gain.gain.value=scale;source.connect(gain);gain.connect(this.gain);this.voices.add(source);source.onended=()=>{this.voices.delete(source);try{source.disconnect();gain.disconnect();}catch{}};source.start();if(Number.isFinite(maxSeconds))source.stop(this.context.currentTime+maxSeconds);return source;}
    catch{if(source)this.stopVoice(source);this.stop();this.say('Local sound stopped unexpectedly; visual replay continues silently.');return null;}
  }
  stopVoice(source){this.voices.delete(source);try{source.stop();}catch{}if(source===this.carry)this.carry=null;}
  startCarry(){if(!this.carry)this.carry=this.voice(DEMO_SOUNDS.carry,{loop:true,scale:.42,maxSeconds:3});}
  carryAt(active){this.wantsCarry=!!active;if(this.wantsCarry)this.startCarry();else if(this.carry)this.stopVoice(this.carry);}
  contact(event,ballId,now){
    if(!event||event.type!=='contact'||event.evidence!=='synthetic'||this.seen.has(event.event_id))return false;this.seen.add(event.event_id);
    // A delayed render/background tab must never produce a backlog of old cues.
    if(!Number.isFinite(now)||now-event.t<0||now-event.t>.15)return false;
    return !!this.voice(DEMO_SOUNDS[ballId],{scale:.7});
  }
  nextClip(){this.carryAt(false);this.seen.clear();for(const source of [...this.voices])this.stopVoice(source);}
  stop(){this.generation++;this.active=false;this.ready=false;this.wantsCarry=false;this.seen.clear();for(const request of this.requests)request.abort();for(const voice of [...this.voices])this.stopVoice(voice);this.carry=null;}
}

export class DemoPanel {
  constructor({document,viewer=null,audio=null,fetchImpl=(...args)=>globalThis.fetch(...args),now=()=>globalThis.performance.now(),requestFrame=fn=>globalThis.requestAnimationFrame(fn),cancelFrame=id=>globalThis.cancelAnimationFrame(id)}={}){
    this.document=document;this.$=id=>document.getElementById(id);this.viewer=viewer;this.audio=audio||new LocalDemoAudio({onStatus:message=>{this.$('audioStatus').textContent=message;}});this.fetch=fetchImpl;this.now=now;this.requestFrame=requestFrame;this.cancelFrame=cancelFrame;this.bundle=null;this.mode='fixed';this.index=0;this.playing=false;this.generation=0;this.frame=null;this.eventIndex=0;this.disposed=false;this.loadEpoch=0;
    this.$('fixedMode').addEventListener('click',()=>this.selectMode('fixed'));this.$('learningMode').addEventListener('click',()=>this.selectMode('learning'));this.$('playDemo').addEventListener('click',()=>this.play());this.$('stopDemo').addEventListener('click',()=>this.stop());
    this.$('demoSound').checked=false;this.$('demoSound').addEventListener('change',()=>{this.audio.setEnabled(this.$('demoSound').checked);this.$('soundSetting').textContent=this.$('demoSound').checked?'ON':'OFF';});this.$('demoVolume').addEventListener('input',()=>{const value=Number(this.$('demoVolume').value);this.audio.setVolume(value/100);this.$('volumeValue').textContent=`${Math.round(value)}%`;});
  }
  message(value,error=false){this.$('demoMessage').textContent=value;this.$('demoMessage').classList.toggle('error',error);}
  clips(){return this.bundle?(this.mode==='fixed'?this.bundle.fixed.clips:[...this.bundle.learning.same_layout_clips,this.bundle.learning.success_example]):[];}
  setViewer(viewer){this.viewer=viewer;if(this.bundle&&!this.playing)this.showClip(0);this.updateControls();}
  async load(){const epoch=++this.loadEpoch,controller=new AbortController();const timer=setTimeout(()=>controller.abort(),12000);try{const response=await this.fetch(DEMO_BUNDLE_URL,{cache:'no-store',signal:controller.signal});if(!response.ok)throw Error('Saved replay bundle unavailable');const bundle=normalizeDemo(await response.json());if(this.disposed||epoch!==this.loadEpoch)return false;this.bundle=bundle;this.renderEvidence();this.showClip(0);this.message('Ready. These are saved simulation records; no device is connected.');this.updateControls();return true;}catch(error){if(!this.disposed&&epoch===this.loadEpoch)this.message(error.message+'. Reload this page to retry.',true);return false;}finally{clearTimeout(timer);}}
  updateControls(){this.$('demoWorkspace').classList.toggle('playing',this.playing);this.$('playDemo').disabled=!this.bundle||!this.viewer||this.playing;this.$('stopDemo').disabled=!this.playing;for(const id of ['fixedMode','learningMode'])this.$(id).disabled=this.playing;this.$('playLabel').textContent=this.mode==='fixed'?'Play four balls':'Play learning reel';}
  selectMode(mode){if(this.playing||!['fixed','learning'].includes(mode))return false;this.audio.stop();this.mode=mode;this.index=0;this.$('fixedMode').setAttribute('aria-selected',String(mode==='fixed'));this.$('learningMode').setAttribute('aria-selected',String(mode==='learning'));this.$('fixedEvidence').hidden=mode!=='fixed';this.$('learningEvidence').hidden=mode!=='learning';this.$('reelEyebrow').textContent=mode==='fixed'?'THE FIXED ROUTINE':'THE LEARNING EXPERIMENT';if(this.bundle)this.showClip(0);this.updateControls();this.message(mode==='fixed'?'One fixed action, four independent model releases.':'Four paired checkpoints, followed by one separately selected successful example.');return true;}
  renderEvidence(){const b=this.bundle;this.$('fixedScore').textContent=`${b.fixed.clips.filter(c=>c.result.success).length} / ${b.fixed.clips.length}`;this.$('learningChart').innerHTML=learningCurveMarkup(b.learning.learning_curve_aggregate);const summary=b.learning.summary||{},row=b.learning.learning_curve.find(r=>r.seed===summary.default_seed&&r.train_scenes===summary.default_train_scenes&&r.method==='frozen_policy');this.$('learnedScore').textContent=row?`${row.captures} / ${row.episodes}`:'Not reported';this.$('learnedInterval').textContent=row?.ci95?`95% interval ${pct(row.ci95[0])}–${pct(row.ci95[1])} · synthetic held-out layouts`:'Uncertainty not reported';for(const [id,key]of [['fixedBaseline','fixed_baseline'],['searchBaseline','limited_search_oracle']]){const value=summary[key];this.$(id).textContent=value?`${value.captures} / ${value.episodes}`:'Not reported';}this.$('bundleEvidence').textContent=`${b.schema} · ${b.profile?.profile_id||'front-bucket profile'} · ${b.manifest?.clips?.length||9} source-linked clips. Original baselines and scoring definitions are preserved.`;}
  showClip(index){this.index=index;this.eventIndex=0;const clips=this.clips(),clip=clips[index];if(!clip)return;this.clip=clip;this.timeline=replayTimeline(clip);this.duration=this.timeline.frames.at(-1).t;this.audio.nextClip();
    this.$('clipCounter').textContent=`${index+1} / ${clips.length}`;this.$('clipTitle').textContent=this.mode==='fixed'?`${clip.ball.label} takes the stage.`:clip.paired_layout?`N = ${clip.train_scenes}. Same test layout.`:'A separate successful example.';
    this.$('clipDescription').textContent=this.mode==='fixed'?'One unchanged board pose and release. A fresh numerical shot for each ball.':clip.paired_layout?'Frozen checkpoint prediction. No per-scene search; the saved miss is retained.':'A different held-out layout, explicitly selected to illustrate a successful frozen prediction.';
    const list=clips.map((row,i)=>{const li=this.document.createElement('li'),dot=this.document.createElement('span');li.className=(i===index?'active ':'')+(this.mode==='learning'?'paired':'');dot.className='clip-dot';dot.style.setProperty('--ball-color',row.ball.color);if(this.mode==='learning')dot.textContent=row.paired_layout?String(row.train_scenes):'★';li.append(dot,this.document.createTextNode(this.mode==='fixed'?row.ball.label:row.paired_layout?`N ${row.train_scenes}`:'Example'));return li;});this.$('clipList').replaceChildren(...list);
    this.$('boardTilt').textContent=`${clip.board_tilt_deg.toFixed(1)}°`;this.$('releaseGap').textContent=`${clip.drop_height_mm.toFixed(0)} mm`;this.$('modelResult').textContent='Awaiting replay';this.$('modelResult').classList.remove('miss');this.$('resultDetail').textContent='Board bounce + whole-ball entry + one continuous second inside.';this.$('reelProgress').value=index/clips.length;this.$('stageClock').textContent='0.0 s';
    this.$('clipProvenance').textContent=`Trial ${clip.trial_id}. ${clip.checkpoint_id?'Checkpoint '+clip.checkpoint_id+'. ':''}${clip.provenance?.selection||clip.provenance?.inference||'Fresh numerical rollout of the fixed preset.'} Release gap is measured from the nominal contact ball-center height, not from a calibrated physical TCP.`;
    if(this.viewer){this.viewer.replay({...clip,selected_scene:clip.scene},{scene:clip.scene});this.viewer.pause();this.viewer.setBallColor?.(clip.ball.color);this.viewer.seek(0);}this.$('stageStatus').textContent='Ready · saved numerical replay';
  }
  play(){if(!this.bundle||!this.viewer||this.playing||this.disposed)return false;this.playing=true;const epoch=++this.generation;this.showClip(0);this.clipStarted=this.now();this.updateControls();this.message('Playing saved numerical evidence. Stop cancels the reel and every local sound.');void this.audio.begin();this.$('demoScene').scrollIntoView?.({behavior:'smooth',block:'center'});this.frame=this.requestFrame(time=>this.tick(time,epoch));return true;}
  tick(time,epoch){if(!this.playing||epoch!==this.generation||this.disposed)return;const elapsed=Math.max(0,(time-this.clipStarted)/1000),clip=this.clip,shown=Math.min(elapsed,this.duration);this.viewer.seek(shown);this.$('stageClock').textContent=`${shown.toFixed(1)} s`;
    const stage=this.timeline.segments.find(row=>shown>=row.start_s&&shown<row.end_s)||this.timeline.segments.at(-1),carrying=this.timeline.segments.some(row=>['move_dropper','settle'].includes(row.stage)&&shown>=row.start_s&&shown<row.end_s);
    this.audio.carryAt(carrying&&elapsed<=this.duration);this.$('stageStatus').textContent=stage?.stage==='settle'?`Settle · ${Math.max(0,stage.end_s-shown).toFixed(1)} s`:(PHASE_NAMES[stage?.stage]||'Saved numerical replay');
    while(this.eventIndex<clip.events.length&&clip.events[this.eventIndex].t<=shown){const event=clip.events[this.eventIndex++];if(event.type==='contact')this.audio.contact(event,clip.ball.id,elapsed);}
    const completed=elapsed>=this.duration;this.$('modelResult').textContent=completed?displayOutcome(clip):shown<this.timeline.releaseTime?'Release pending':'Evaluating model contact';this.$('modelResult').classList.toggle('miss',completed&&!clip.result.success);if(completed){this.$('stageStatus').textContent=displayOutcome(clip)+' · simulation';this.$('resultDetail').textContent=clip.result.success?'The saved scorer verified one-second containment after a board bounce.':'This saved shot did not satisfy the complete-shot capture rule.';}
    this.$('reelProgress').value=(this.index+Math.min(elapsed/(this.duration+1.25),1))/this.clips().length;
    if(elapsed>=this.duration+1.25){if(this.index+1<this.clips().length){this.showClip(this.index+1);this.clipStarted=time;}else{this.playing=false;this.audio.stop();this.$('reelProgress').value=1;this.updateControls();this.message(this.mode==='fixed'?'Four independent simulations finished. Press Play to replay; no physical result is claimed.':'Paired misses and the separate successful example are all retained. View the full study for uncertainty and failure cases.');return;}}
    this.frame=this.requestFrame(next=>this.tick(next,epoch));
  }
  stop(reason='Stopped. No queued replay or local sound remains.'){this.generation++;this.playing=false;if(this.frame!==null)this.cancelFrame(this.frame);this.frame=null;this.audio.stop();this.viewer?.pause();this.$('stageStatus').textContent='Stopped · no pending cues';this.updateControls();this.message(reason);}
  destroy(){this.stop();this.disposed=true;this.loadEpoch++;}
}

if(typeof document!=='undefined'&&document.getElementById('demoScene')){
  const panel=new DemoPanel({document});void panel.load();createViewer(document,{canvasId:'demoScene',resetId:'demoViewReset',topId:'demoViewTop',pauseId:'unusedDemoPause'}).then(viewer=>panel.setViewer(viewer)).catch(()=>{document.getElementById('viewerError').hidden=false;panel.message('The 3D viewer is unavailable. Use the saved film below.',true);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&panel.playing)panel.stop('Paused because the page was hidden. Press Play to restart cleanly.');});window.addEventListener('pagehide',()=>panel.destroy(),{once:true});
}
