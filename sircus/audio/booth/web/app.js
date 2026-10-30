'use strict';
import {FXControls} from '/fx-controls.mjs';
const fxControls=new FXControls({request:(path,body)=>act(path,body),clientId:crypto.randomUUID()});
let fxPending=0;
const $=id=>document.getElementById(id);
const titles={boing:'Haunted BOING',bone_low:'Low bone clack',bone_high:'High bone clack',eerie_bell:'Eerie bell',pinball:'Ghost pinball',transport:'Energy hum',release:'Falling whistle',capture:'Spooky victory',miss:'Oops, a ghost'};
let state=null,sounds=[],events=[],assignments=null,selected='impact',snapshot='',applied='',cursor=0,lines=[],mappingDirty=false,polling=false;
const clone=x=>JSON.parse(JSON.stringify(x));
const title=name=>sounds.find(s=>s.name===name)?.title||titles[name]||name;
async function request(path,body){
  const response=await fetch('/api/'+path,{...(body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),signal:AbortSignal.timeout(4000)});
  const data=await response.json();if(!response.ok)throw new Error(data.error||'Request failed');return data;
}
function fail(error){$('error').textContent=error.message||String(error);}
function bind(id,fn,event='click'){$(id).addEventListener(event,()=>{Promise.resolve().then(fn).catch(fail);});}
async function act(path,body){$('error').textContent='';const result=await request(path,body);if(result.accepted===false&&result.reason!=='not_mapped')throw new Error(result.reason.replaceAll('_',' '));return result;}
function noteDirty(){const value=JSON.stringify(assignments);$('dirty').textContent=value===snapshot?'Saved profile':value===applied?'Applied · not saved':'Changes not applied';renderEvents();}
function describe(row){if(!row.sounds.length||row.gain===0)return 'Silent';return row.selection==='fixed'?title(row.sounds[0]):row.sounds.length+' sounds · '+row.selection;}
function renderEvents(){
  $('events').replaceChildren();
  for(const event of events){const b=document.createElement('button');b.className=event.key===selected?'active':'';b.setAttribute('aria-pressed',String(event.key===selected));const strong=document.createElement('b'),small=document.createElement('small');strong.textContent=event.title;small.textContent=describe(assignments[event.key]);b.append(strong,small);b.onclick=()=>{selected=event.key;renderEditor();};$('events').append(b);}
}
function renderEditor(){
  const row=assignments[selected],event=events.find(e=>e.key===selected);renderEvents();$('event-title').textContent=event.title;$('event-description').textContent=event.description;
  $('choices').replaceChildren();
  for(const sound of sounds){const label=document.createElement('label');label.className='choice';const box=document.createElement('input');box.type='checkbox';box.checked=row.sounds.includes(sound.name);box.onchange=()=>{if(box.checked)row.sounds.push(sound.name);else row.sounds=row.sounds.filter(n=>n!==sound.name);noteDirty();renderOrder();};label.append(box,document.createTextNode(title(sound.name)));$('choices').append(label);}
  $('selection').value=row.selection;$('avoid-repeat').checked=row.avoid_repeat;$('gain').value=Math.round(row.gain*100);$('pitch').value=row.pitch;$('loop').checked=row.loop;$('max-seconds').value=row.max_seconds;$('loop-controls').hidden=!['carry','flight'].includes(selected);renderOrder();
}
function renderOrder(){const r=assignments[selected];$('gain-value').textContent=Math.round(r.gain*100)+'%';$('pitch-value').textContent=(r.pitch>0?'+':'')+r.pitch+' semitones';$('avoid-repeat').disabled=r.selection!=='random';$('max-seconds').disabled=!r.loop;$('order').textContent=r.sounds.length?(r.selection==='fixed'?'Plays: '+title(r.sounds[0]):'Selection order: '+r.sounds.map(title).join(' → ')):'No sound assigned.';}
function setProfile(profile){
  assignments=clone(profile.assignments);snapshot=JSON.stringify(assignments);applied=snapshot;
  $('profile').replaceChildren(...profile.names.map(n=>new Option(n,n)));$('profile').value=profile.active;
  $('profile-name').value=profile.builtins.includes(profile.active)?'':profile.active;renderEditor();noteDirty();if(profile.error)fail(profile.error);
}
async function applyProfile(){
  if(JSON.stringify(assignments)!==applied){await act('profile',{action:'apply',assignments});applied=JSON.stringify(assignments);noteDirty();}
}
function paintStatus(){
  if(!fxPending)$('fx').checked=state.fx_enabled;
  $('badge').textContent=fxPending?'FX CHANGE PENDING':state.fx_enabled?'FX LIVE':'FX OFF';$('badge').className=state.fx_enabled?'live':'';
  if(document.activeElement!==$('volume'))$('volume').value=Math.round(state.volume*100);$('volval').textContent=Math.round(state.volume*100)+'%';$('mode').value=state.mode;$('meter').style.width=Math.min(100,state.audio_meter_peak*100)+'%';
  const key=JSON.stringify(state.inputs);if($('input').dataset.key!==key){$('input').replaceChildren(new Option('No MIDI input · use previews',''));for(const p of state.inputs){const option=new Option(p.name+(p.online?'':' · offline')+' · ID '+p.id,p.id);option.disabled=!p.online;$('input').append(option);}$('input').dataset.key=key;}$('input').value=state.selected_input??'';
  $('connection').textContent=(state.audio_ready?'Mac audio ready':'Audio unavailable')+' · '+(state.connected?'Selected ball connected':'No ball selected / connected')+(state.last_midi_age_s===null?'':' · Last MIDI '+state.last_midi_age_s+'s ago')+(state.recording?' · Recording MIDI':'');
  $('trial').textContent=(state.domain==='simulation'?'Synthetic replay. Click “Return to live ball” to resume taps.':state.domain==='recorded_midi'||state.playback_source==='events'?'Recorded trace selected. Click “Return to live ball” to resume taps.':state.armed?'Live trial armed.':'Live ball ready; '+(state.mode==='free'?'every tap uses Bounce / impact.':'click Release / arm live trial before taps.'))+(state.loop_event?' Loop: '+state.loop_event+'.':'');
  for(const button of document.querySelectorAll('[data-play]'))button.disabled=!state.fx_enabled;
  $('replay').disabled=!state.fx_enabled;
  if(!mappingDirty){$('note').value=state.mapping.impact_note;$('channel').value=state.mapping.channel;$('dedup').value=state.mapping.exact_duplicate_ms;}
  if(state.last_error)fail(state.last_error);
}
async function refresh(){
  if(polling||fxPending)return;polling=true;
  const sequence=fxControls.sequence;
  try{const current=await request('status');if(sequence!==fxControls.sequence)return;state=current;if($('error').textContent.startsWith('Bridge disconnected:'))$('error').textContent='';events=state.events;if(!assignments)setProfile(state.profile);paintStatus();const m=await request('monitor?after='+cursor);cursor=m.seq;for(const e of m.events)lines.push(JSON.stringify(e));lines=lines.slice(-100);$('monitor').textContent=lines.join('\n');$('monitor').scrollTop=$('monitor').scrollHeight;}
  catch(error){$('badge').textContent='DISCONNECTED';$('fx').checked=false;for(const b of document.querySelectorAll('[data-play]'))b.disabled=true;fail('Bridge disconnected: '+error.message);}
  finally{polling=false;}
}
async function changeFX(panic=false){
  const enabled=$('fx').checked;fxPending++;$('badge').textContent='FX CHANGE PENDING';
  if(panic)$('fx').checked=false;
  try{if(panic)await fxControls.panic();else await fxControls.setFX(enabled);}
  finally{fxPending--;await refresh();}
}
bind('fx',()=>changeFX(),'change');bind('panic',()=>changeFX(true));bind('stop',()=>act('stop',{}));
bind('volume',()=>{$('volval').textContent=$('volume').value+'%';},'input');bind('volume',()=>act('volume',{volume:Number($('volume').value)/100}),'change');
bind('input',async()=>{await act('input',{id:$('input').value===''?null:Number($('input').value)});await refresh();},'change');bind('mode',()=>act('mode',{mode:$('mode').value}),'change');
bind('load',async()=>setProfile(await act('profile',{action:'load',name:$('profile').value})));
bind('save',async()=>setProfile(await act('profile',{action:'save',name:$('profile-name').value.trim(),assignments})));
bind('apply-profile',applyProfile);bind('none',()=>{assignments[selected].sounds=[];renderEditor();noteDirty();});
for(const [id,key,kind] of [['selection','selection','string'],['avoid-repeat','avoid_repeat','boolean'],['gain','gain','gain'],['pitch','pitch','number'],['loop','loop','boolean'],['max-seconds','max_seconds','number']]){
  bind(id,()=>{assignments[selected][key]=kind==='boolean'?$(id).checked:kind==='string'?$(id).value:Number($(id).value)/(kind==='gain'?100:1);renderOrder();noteDirty();},id==='gain'||id==='pitch'?'input':'change');
}
bind('preview',async()=>{await applyProfile();await act('preview',{event:selected});});bind('audition',()=>act('event',{type:'audition'}));
bind('live',()=>act('mode',{mode:$('mode').value}));
bind('release',async()=>{await applyProfile();await act('mode',{mode:'trial'});await act('event',{type:'release',trial_id:'manual-'+crypto.randomUUID()});await refresh();});
bind('bounce',()=>act('mock-midi',{status:144+state.mapping.channel-1,data:[state.mapping.impact_note,90],input:'mock'}));
function syntheticSequence(outcome){
  const trial='synthetic-'+crypto.randomUUID();
  return [[0,'pickup'],[.4,'carry_start'],[1.5,'release'],[1.7,'flight_start'],[2.5,'impact','board'],[3.1,'impact','table'],[3.7,'impact','bucket-rim'],[4.5,'outcome']].map(([timestamp,event_type,contact_object='unknown'],i)=>({schema:'sircus.audio.v1',event_id:trial+'-'+i,trial_id:trial,source:'simulation',event_type,contact_object,timestamp,time_basis:'simulation_elapsed_s',clock_id:'ghost-booth-synthetic',provenance:'Hand-authored sound rehearsal; no robot or physics run',evidence:'model_prediction',outcome:event_type==='outcome'?outcome:'unknown',confirmed:false,intensity:.7}));
}
bind('sequence',async()=>{await applyProfile();await act('replay-events',{events:syntheticSequence($('outcome').value)});await refresh();});
for(const id of ['note','channel','dedup'])bind(id,()=>{mappingDirty=true;},'input');
bind('apply-mapping',async()=>{await act('mapping',{impact_note:Number($('note').value),channel:Number($('channel').value),exact_duplicate_ms:Number($('dedup').value)});mappingDirty=false;});
bind('record',()=>act('record',{enabled:true}));bind('endrecord',()=>act('record',{enabled:false}));
bind('replay',async()=>{const file=$('replay').files[0];if(!file)return;if(file.size>1000000)throw new Error('Trace must be under 1 MB');const text=await file.text();const rows=text.trim().startsWith('[')?JSON.parse(text):text.trim().split('\n').filter(Boolean).map(JSON.parse);await applyProfile();await act(rows[0]?.schema?'replay-events':'replay',{events:rows});$('replay').value='';},'change');
(async()=>{sounds=await request('sounds');$('catalog-summary').textContent=sounds.length+' original sounds, available for every event. Click a tile to hear it on its own.';for(const sound of sounds){const b=document.createElement('button');b.className='pad'+(sound.family==='Cartoon'?' cartoon':'');b.dataset.play='';b.disabled=true;const strong=document.createElement('b'),small=document.createElement('small');strong.textContent=title(sound.name);small.textContent=(sound.family||'Haunted')+' · '+sound.seconds.toFixed(2)+' s';b.append(strong,small);b.onclick=()=>act('event',{type:'test',sound:sound.name,velocity:85}).catch(fail);$('pads').append(b);}$('audition').textContent='▶ Audition all · '+Math.ceil(sounds.reduce((n,s)=>n+s.seconds+.15,0))+' seconds';await refresh();setInterval(refresh,700);})().catch(fail);
