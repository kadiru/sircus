import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
import {ShowPanel,bindingValue,workspaceURL} from '../web/show.js';
import {mountPhysicalPanel} from '../web/physical.js';
const copy=value=>JSON.parse(JSON.stringify(value));
const reply=value=>({ok:true,json:async()=>value});
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
class Node{
 constructor(){this.children=[];this.value='';this.textContent='';this.disabled=false;this.hidden=false;this.checked=false;this.dataset={};this.style={};this.attrs={};this.listeners={};this.classList={toggle(){}};}
 addEventListener(name,fn){this.listeners[name]=fn;}removeEventListener(name){delete this.listeners[name];}
 append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=nodes;}
 setAttribute(k,v){this.attrs[k]=String(v);}getAttribute(k){return this.attrs[k]??null;}removeAttribute(k){delete this.attrs[k];}
 querySelectorAll(){return [];}querySelector(){return null;}
}
function documentFixture(){const nodes={};const document={activeElement:null,getElementById(id){return nodes[id]??=new Node();},createElement(){return new Node();},querySelectorAll(){return [];}};return {document,nodes};}
function status(mode='dry_run'){return {schema:'sircus.show.v1',action_token:'show-test-token',physical_mode:mode,modes:{mock:{available:true,url:'/operations.html'},simulator:{available:true,url:'/operations.html'},trained:{available:true,url:'/operations.html'},deterministic:{available:true,mode}},audio_show:{schema:'sircus.show-audio.v1',catalog:['boing','transport'],config:{schema:'sircus.show-audio.v1',impact_window_s:12,balls:['pink','purple','green','gray'].map((name,i)=>({ball_id:'ball-'+name,label:name,color:['#ed8cae','#9a74b8','#7aab70','#8c9594'][i],midi_binding:null,sounds:{surface:'boing',bucket:'boing',collision:'boing',unknown:'boing',carry:'transport'}}))}}};}
function harness(fetchImpl=async()=>reply(status())){const {document,nodes}=documentFixture(),commands=[],selections=[],requests=[];const physical={busy:false,allowed:()=>true,selectView:value=>selections.push(value),command:async(...args)=>{commands.push(args);return true;}};const panel=new ShowPanel({document,fetchImpl:(...args)=>{requests.push(args);return fetchImpl(...args);},physical});return {panel,nodes,document,physical,commands,selections,requests};}

test('show startup reads configuration without a physical command and embeds only permitted numerical workspace',async()=>{const h=harness();assert.equal(h.requests.length,0);await h.panel.refresh();assert.equal(h.requests[0][0],'/api/show/status');assert.equal(h.requests[0][1].method,undefined);assert.deepEqual(h.commands,[]);assert.deepEqual(h.selections,[]);assert.equal(h.nodes.simulatorFrame.getAttribute('src'),'/operations.html?embed=simulator');assert.equal(h.nodes.trainedFrame.getAttribute('src'),null);assert.equal(h.nodes.physicalPanel.hidden,true);assert.deepEqual(h.panel.state.audio_show.config.balls.map(row=>row.ball_id),['ball-pink','ball-purple','ball-green','ball-gray']);});

test('hardware mode cannot embed local training or claim a numerical backend',()=>{const h=harness();h.panel.accept(status('hardware'));h.panel.selectMode('trained');assert.equal(h.nodes.trainedFrame.getAttribute('src'),null);assert.equal(h.nodes.trainedUnavailable.hidden,false);assert.match(h.nodes.showRuntime.textContent,/PHYSICAL/);assert.throws(()=>workspaceURL('https://evil.test/operations.html','simulator'));assert.throws(()=>workspaceURL('http://localhost/admin','simulator'));});

test('changing acts is blocked during an active routine so Stop remains reachable',()=>{const h=harness();h.panel.accept(status());assert.equal(h.panel.selectMode('deterministic'),true);h.physical.busy=true;assert.equal(h.panel.selectMode('simulator'),false);assert.equal(h.panel.mode,'deterministic');assert.equal(h.nodes.physicalPanel.hidden,false);assert.match(h.nodes.showMessage.textContent,/Stop remains available/);assert.equal(h.commands.length,0);});

test('ball choice is operator supplied and only deterministic choice changes physical trial identity',async()=>{const h=harness();h.panel.accept(status());await h.panel.selectBall('ball-purple');assert.equal(h.panel.selectedBall,'ball-purple');assert.equal(h.commands.length,0);h.panel.selectMode('deterministic');await h.panel.selectBall('ball-gray');assert.deepEqual(h.commands,[['set_ball',{ball_id:'ball-gray'}]]);h.physical.busy=true;assert.equal(await h.panel.selectBall('ball-pink'),false);assert.equal(h.commands.length,1);});

test('identity binding remains unknown when blank and refuses partial or invalid identifiers',()=>{assert.equal(bindingValue(['','','']),null);assert.deepEqual(bindingValue(['-2','1','60']),{input_id:-2,channel:1,note:60});assert.deepEqual(bindingValue(['2','1','60']),{input_id:2,channel:1,note:60});for(const values of [['2','','60'],['0','1','60'],['2','17','60'],['2','1','128']])assert.throws(()=>bindingValue(values));});

test('configuration edit preserves numerical sound assignments and sends no playback or robot command',async()=>{const h=harness(async(_url,options)=>reply(options?.body?{...status(),audio_show:{...status().audio_show,config:JSON.parse(options.body).config}}:status()));h.panel.accept(status());const fields=h.panel.editors.get('ball-pink');fields.label.value='Pink comet';fields.input_id.value='4';fields.channel.value='2';fields.note.value='61';fields.unknown.value='boing';h.panel.dirty=true;assert.equal(await h.panel.saveConfig(),true);const post=h.requests.find(([,options])=>options.method==='POST');const body=JSON.parse(post[1].body);assert.equal(post[0],'/api/show/action');assert.equal(post[1].headers['X-Sircus-Show-Token'],'show-test-token');assert.equal(body.action,'configure_audio');assert.equal(body.config.balls[0].label,'Pink comet');assert.equal(body.config.balls[0].sounds.surface,'boing');assert.equal(body.config.balls[0].midi_binding.note,61);assert.equal(h.commands.length,0);});

test('stale status cannot overwrite accepted newer configuration and edits survive ordinary polling',async()=>{const old=deferred();const next=status();next.audio_show.config.balls[0].label='New pink';const h=harness(url=>url.endsWith('/status')?old.promise:Promise.resolve(reply(next)));h.panel.accept(status());const pending=h.panel.refresh();await h.panel.action({action:'configure_audio',config:next.audio_show.config});old.resolve(reply(status()));await pending;assert.equal(h.panel.state.audio_show.config.balls[0].label,'New pink');h.panel.editors.get('ball-pink').label.value='Unsaved';h.panel.dirty=true;h.panel.accept(status());assert.equal(h.panel.editors.get('ball-pink').label.value,'Unsaved');});

test('physical controller mount stays inert until explicit view selection and keeps state API reusable',async()=>{const {document,nodes}=documentFixture(),calls=[];const panel=mountPhysicalPanel({document,window:{location:{hash:''},addEventListener(){},dispatchEvent(){}},manageTabs:false,fetchImpl:async(...args)=>{calls.push(args);return reply({mode:'dry_run',connected:false,allowed_actions:[],calibration:{}});},setIntervalImpl:()=>1,clearIntervalImpl(){}});assert.equal(calls.length,0);await panel.refreshStatus();assert.equal(calls.length,0);panel.selectView(true);await new Promise(resolve=>setImmediate(resolve));assert.equal(calls[0][0],'/api/physical/status');assert.equal(calls.length,1);assert.equal(panel.status.mode,'dry_run');assert.equal(nodes.physicalCameraSource.textContent,'Dry-run camera · no hardware');assert.match(nodes.physicalCamera.alt,/Mock camera preview; no physical camera/);panel.destroy();});

test('operator markup has no manual contact cues and exposes only measured generic physical audio',async()=>{const html=await readFile(new URL('../web/show.html',import.meta.url),'utf8');assert.doesNotMatch(html,/data-show-contact|preview_contact|Board bounce<\/button|Bucket bounce<\/button/);assert.match(html,/measured generic bounce sounds/);assert.match(html,/does not identify board, bucket or ball contact/);assert.match(html,/id="audioFX"[^>]*disabled/);assert.doesNotMatch(html,/id="audioFX"[^>]*checked/);});

test('simulation audio replay needs a completed numerical trial and FX; it never invokes motion',async()=>{const h=harness();h.panel.accept(status());h.panel.numericalTrial={mode:'simulation_only',hardware_enabled:false,phase:'complete',trial_id:'synthetic-trial-7'};assert.equal(h.panel.replayAudio(),false);h.panel.getAudioState=()=>({enabled:true,show_audio_capable:true});assert.equal(await h.panel.replayAudio(),true);const body=JSON.parse(h.requests[0][1].body);assert.deepEqual(body,{action:'replay_simulation',trial_id:'synthetic-trial-7',ball_id:'ball-pink'});assert.equal(h.commands.length,0);h.panel.numericalTrial.hardware_enabled=true;assert.equal(h.panel.replayAudio(),false);h.panel.numericalTrial.hardware_enabled=false;h.panel.selectMode('deterministic');assert.equal(h.panel.replayAudio(),false);});

test('audio replay Stop supersedes delayed start acknowledgment without touching physical controls',async()=>{const delayed=deferred();const h=harness((_url,options)=>JSON.parse(options.body).action==='replay_simulation'?delayed.promise:Promise.resolve(reply(status())));h.panel.accept(status());const starting=h.panel.action({action:'replay_simulation',trial_id:'sim-1',ball_id:'ball-pink'});assert.equal(h.panel.pending,true);await h.panel.action({action:'stop_audio_replay'});const stale=status();stale.audio_show.replay_running=true;delayed.resolve(reply(stale));assert.equal(await starting,false);assert.equal(h.panel.pending,false);assert.notEqual(h.panel.state.audio_show.replay_running,true);assert.equal(h.commands.length,0);});

test('pose configuration uses the shared controller API and does not send movement commands',async()=>{const {document,nodes}=documentFixture(),calls=[];const state={mode:'dry_run',connected:false,phase:'idle',action_token:'physical-test',allowed_actions:['configure_poses','stop'],calibration:{}};const controller=mountPhysicalPanel({document,window:{location:{hash:''},addEventListener(){},dispatchEvent(){}},manageTabs:false,fetchImpl:async(...args)=>{calls.push(args);return reply(state);},setIntervalImpl:()=>1,clearIntervalImpl(){}});controller.selectView(true);await new Promise(resolve=>setImmediate(resolve));for(const [prefix,values]of [['Loading',[150,-600,200,0,0,-1,0]],['Drop',[600,-150,600,0,0,-1,0]]])['X','Y','Z','OX','OY','OZ','Theta'].forEach((suffix,i)=>nodes['physical'+prefix+suffix].value=String(values[i]));nodes.physicalPoseConfirm.checked=true;assert.equal(await controller.command('configure_poses'),true);const post=calls.find(([,options])=>options?.method==='POST');const value=JSON.parse(post[1].body);assert.equal(value.action,'configure_poses');assert.deepEqual(value.loading_pose,{x:150,y:-600,z:200,o_x:0,o_y:0,o_z:-1,theta:0});assert.equal(value.drop_pose.z,600);assert.equal(value.confirmed,true);assert.equal(post[1].headers['X-Sircus-Token'],'physical-test');assert.equal(calls.filter(([,options])=>options?.method==='POST').length,1);controller.destroy();});


test('unselected physical ball remains visibly unknown rather than adopting the simulator default',()=>{const h=harness();h.panel.accept(status());assert.equal(h.panel.selectedBall,'ball-pink');h.panel.selectMode('deterministic');h.panel.physicalStatus({selected_ball_id:null,busy:false});assert.equal(h.panel.selectedBall,null);assert.match(h.nodes.selectedBallStatus.textContent,/No physical ball identity selected/);h.panel.accept(status());assert.equal(h.panel.selectedBall,null);h.panel.selectMode('simulator');assert.equal(h.panel.selectedBall,'ball-pink');assert.equal(h.commands.length,0);});


test('replay requires verified service capability and renders replay failures across status polls',async()=>{
 const failed=status();failed.audio_show.error='Audio replay not scheduled: enable a compatible audio service first';failed.audio_show.replay_running=false;
 const trial={mode:'simulation_only',hardware_enabled:false,phase:'complete',trial_id:'sim-capability-check'};
 const h=harness(async url=>reply(url==='/api/operations/status'?trial:failed));h.panel.accept(status());h.panel.numericalTrial=trial;
 for(const show_audio_capable of [undefined,false]){h.panel.getAudioState=()=>({enabled:true,show_audio_capable});h.panel.render();assert.equal(h.nodes.replaySimulationAudio.disabled,true);assert.match(h.nodes.simulationAudioStatus.textContent,/not verified four-ball replay support/);assert.equal(h.panel.replayAudio(),false);}
 assert.equal(h.requests.length,0);h.panel.getAudioState=()=>({enabled:true,show_audio_capable:true});h.panel.render();assert.equal(h.nodes.replaySimulationAudio.disabled,false);
 assert.equal(await h.panel.replayAudio(),false);assert.equal(h.nodes.simulationAudioStatus.textContent,failed.audio_show.error);assert.equal(h.nodes.showMessage.textContent,failed.audio_show.error);assert.equal(h.panel.pending,false);
 await h.panel.refresh();assert.equal(h.nodes.simulationAudioStatus.textContent,failed.audio_show.error);assert.equal(h.commands.length,0);
 const recovered=status();recovered.audio_show.error=null;recovered.audio_show.replay_running=true;h.panel.accept(recovered);assert.equal(h.nodes.simulationAudioStatus.textContent,'Synthetic audio replay in progress.');
});


test('four modes reuse isolated workspaces and preserve the manual controller route',async()=>{
 const h=harness();h.panel.accept(status());assert.equal(h.panel.selectMode('mock'),true);assert.equal(h.nodes.mockFrame.getAttribute('src'),'/operations.html?embed=mock');assert.equal(h.nodes.showMockPane.hidden,false);assert.equal(h.nodes.simulatorFrame.getAttribute('src'),null);assert.equal(h.nodes.physicalPanel.hidden,true);assert.equal(h.commands.length,0);h.panel.accept(status('hardware'));assert.equal(h.nodes.mockFrame.getAttribute('src'),null);assert.equal(h.nodes.mockUnavailable.hidden,false);const html=await readFile(new URL('../web/show.html',import.meta.url),'utf8');for(const title of ['Manual operation','Trained operation','Simulation','Mock simulation'])assert.ok(html.includes(title));assert.doesNotMatch(html,/18766\/operations\.html\?embed=(mock|trained)/);assert.match(html,/saved-poses" open/);
});
test('numerical job and trusted active-frame activity keep mode changes locked until idle',()=>{
 const h=harness(),busy=status();busy.numerical_busy=true;h.panel.accept(busy);assert.equal(h.panel.selectMode('deterministic'),false);assert.equal(h.nodes.modeDeterministic.disabled,true);h.panel.accept(status());const source={};h.nodes.simulatorFrame.contentWindow=source;assert.equal(h.panel.workspaceStatus({origin:'https://other.test',source,data:{type:'sircus-workspace-activity',busy:true}},'http://localhost:18770'),false);assert.equal(h.panel.selectMode('mock'),true);h.nodes.mockFrame.contentWindow=source;assert.equal(h.panel.workspaceStatus({origin:'http://localhost:18770',source,data:{type:'sircus-workspace-activity',busy:true}},'http://localhost:18770'),true);assert.equal(h.panel.selectMode('trained'),false);h.panel.workspaceStatus({origin:'http://localhost:18770',source,data:{type:'sircus-workspace-activity',busy:false}},'http://localhost:18770');assert.equal(h.panel.selectMode('trained'),true);assert.equal(h.commands.length,0);
});

test('verified hardware status selects Manual once and never sends an actuator command',()=>{
 const h=harness();h.panel.accept(status('hardware'));assert.equal(h.panel.mode,'deterministic');assert.equal(h.nodes.physicalPanel.hidden,false);assert.equal(h.nodes.hardwarePreparation.hidden,false);assert.deepEqual(h.selections,[true]);assert.deepEqual(h.commands,[]);assert.equal(h.requests.length,0);
 h.panel.accept(status('hardware'));assert.deepEqual(h.selections,[true]);for(const mode of ['simulator','trained','mock'])assert.equal(h.panel.selectMode(mode),false);assert.equal(h.panel.mode,'deterministic');assert.equal(h.nodes.modeSimulator.disabled,true);
});
test('hardware setup distinguishes stored targets, live state and audio input readiness',()=>{
 const h=harness();h.panel.accept(status('hardware'));assert.match(h.nodes.hardwareRobotStatus.textContent,/Awaiting/);assert.match(h.nodes.hardwareBallStatus.textContent,/0\/4 saved bindings.*unknown/);
 h.panel.physicalStatus({connected:true,calibration:{loading_pose:{x:150},drop_pose:{x:600}},fixed_cycle_missing:[],selected_ball_id:null});assert.match(h.nodes.hardwarePoseStatus.textContent,/Saved targets/);
 const configured=status('hardware');for(const ball of configured.audio_show.config.balls)ball.midi_binding={input_id:4,channel:1,note:60};h.panel.getAudioState=()=>({configured:true,connection:'connected',engine_ready:true,mapping_matches:true,connected_ball_count:2});h.panel.accept(configured);assert.match(h.nodes.hardwareBallStatus.textContent,/4\/4 saved bindings.*2\/4 inputs connected/);assert.match(h.nodes.hardwareAudioHint.textContent,/listening/);assert.equal(h.nodes.modeDeterministic.disabled,false);
 h.panel.getAudioState=()=>({configured:true,connection:'disconnected',engine_ready:true,mapping_matches:true,connected_ball_count:4});h.panel.render();assert.match(h.nodes.hardwareEngineStatus.textContent,/not verified/);assert.match(h.nodes.hardwareBallStatus.textContent,/connection unknown/);assert.deepEqual(h.commands,[]);
});
test('explicit reapply of unchanged audio mappings is audio-only and blocked by unsaved edits or busy routine',async()=>{
 const h=harness();h.panel.getAudioState=()=>({configured:true});h.panel.accept(status('hardware'));assert.equal(h.nodes.reapplyAudioMappings.disabled,false);assert.equal(await h.panel.reapplyAudioMappings(),true);assert.equal(h.requests.length,1);assert.equal(h.requests[0][0],'/api/show/action');assert.equal(JSON.parse(h.requests[0][1].body).action,'configure_audio');assert.deepEqual(h.commands,[]);
 h.panel.dirty=true;assert.equal(await h.panel.reapplyAudioMappings(),false);h.panel.dirty=false;h.physical.busy=true;assert.equal(await h.panel.reapplyAudioMappings(),false);assert.equal(h.requests.length,1);
});

const pickupPose={x:150,y:-600,z:120,o_x:0,o_y:0,o_z:-1,theta:0};
function pickupState(extra={}){return {connected:true,busy:false,phase:'idle',selected_ball_id:'ball-pink',
 calibration:{pickup_poses:{'ball-pink':{...pickupPose}},drop_pose:{...pickupPose,x:600,z:600}},
 pickup_grasp_pose:{...pickupPose},pickup_approach_pose:{...pickupPose,z:220},loading_approach_pose:{...pickupPose,z:300},
 pickup_approach_offset_mm:100,pickup_station_missing:[],pickup_station_blockers:[],...extra};}
function pickupHarness(){const h=harness();h.panel.accept(status('hardware'));h.panel.physicalStatus(pickupState());return h;}

test('taped pickup is a separate opt-in mode and teaching never starts a routine',async()=>{
 const h=pickupHarness();assert.equal(h.panel.manualLoadMode,'hand');assert.equal(h.nodes.handLoadCycle.hidden,false);assert.equal(h.nodes.pickupStationPanel.hidden,true);assert.deepEqual(h.commands,[]);
 assert.equal(h.panel.selectLoadMode('pickup'),true);assert.equal(h.nodes.handLoadCycle.hidden,true);assert.equal(h.nodes.handLoadControls.hidden,true);assert.equal(h.nodes.pickupStationPanel.hidden,false);assert.equal(h.nodes.pickupCycle.disabled,true);assert.match(h.nodes.pickupPose.textContent,/"z": 120/);assert.match(h.nodes.pickupApproachPose.textContent,/"z": 220/);assert.deepEqual(h.commands,[]);
 assert.equal(await h.panel.pickupCommand('teach_pickup'),true);assert.deepEqual(h.commands,[['teach_pickup',{ball_id:'ball-pink'}]]);
 assert.equal(h.panel.selectLoadMode('hand'),true);assert.equal(h.nodes.handLoadCycle.hidden,false);assert.equal(h.commands.length,1);
});

test('pickup approach requires explicit pad clearance and never sends descend or tool commands',async()=>{
 const h=pickupHarness();h.panel.selectLoadMode('pickup');assert.equal(await h.panel.pickupCommand('go_pickup_approach'),false);assert.deepEqual(h.commands,[]);
 h.nodes.pickupSetupConfirm.checked=true;h.panel.render();assert.equal(h.nodes.pickupApproach.disabled,false);assert.equal(h.nodes.pickupCycle.disabled,true);
 assert.equal(await h.panel.pickupCommand('go_pickup_approach'),true);assert.deepEqual(h.commands,[['go_pickup_approach',{ball_id:'ball-pink',pickup_setup_confirmed:true}]]);assert.equal(h.nodes.pickupSetupConfirm.checked,false);
 h.nodes.pickupSetupConfirm.checked=true;assert.equal(await h.panel.pickupCommand('go_loading_approach'),true);assert.deepEqual(h.commands[1],['go_loading_approach',{pickup_setup_confirmed:true}]);
});

test('pickup cycle keeps selected identity, requires both confirmations and remains one press per attempt',async()=>{
 const h=pickupHarness();h.panel.selectLoadMode('pickup');h.nodes.pickupSetupConfirm.checked=true;assert.equal(await h.panel.pickupCommand('pickup_drop_cycle'),false);
 h.nodes.pickupFixedConfirm.checked=true;h.nodes.pickupPractice.checked=true;
 const gate=deferred();h.physical.command=async(...args)=>{h.commands.push(args);return gate.promise;};
 const first=h.panel.pickupCommand('pickup_drop_cycle');assert.equal(h.nodes.pickupCycle.disabled,true);assert.equal(h.nodes.pickupSetupConfirm.checked,false);assert.equal(h.nodes.pickupFixedConfirm.checked,false);assert.equal(h.panel.selectLoadMode('hand'),false);assert.equal(await h.panel.pickupCommand('pickup_drop_cycle'),false);
 assert.deepEqual(h.commands,[['pickup_drop_cycle',{ball_id:'ball-pink',pickup_setup_confirmed:true,fixed_setup_confirmed:true,practice_mode:true}]]);
 gate.resolve(true);assert.equal(await first,true);h.panel.physicalStatus(pickupState({trial:{id:'retained',saved:true,status:'completed'}}));assert.equal(h.nodes.pickupCycle.disabled,true);assert.equal(h.commands.length,1,'completed status cannot automatically repeat');
});

test('color, target, fault and readback changes invalidate pickup confirmations',()=>{
 const h=pickupHarness();h.panel.selectLoadMode('pickup');
 for(const changed of [pickupState({selected_ball_id:'ball-gray',pickup_grasp_pose:null,pickup_approach_pose:null}),pickupState({calibration:{drop_pose:{...pickupPose,z:650}}}),pickupState({connected:false}),pickupState({fault:true,error:'guard fault'}),pickupState({busy:true,phase:'pickup_descending'})]){
   h.panel.physicalStatus(pickupState());h.nodes.pickupSetupConfirm.checked=true;h.nodes.pickupFixedConfirm.checked=true;h.panel.physicalStatus(changed);assert.equal(h.nodes.pickupSetupConfirm.checked,false);assert.equal(h.nodes.pickupFixedConfirm.checked,false);assert.equal(h.nodes.pickupCycle.disabled,true);
 }
 assert.deepEqual(h.commands,[]);
});

test('pickup motion stays locked for an untaught color, unexpected offset and unsaved target edits',async()=>{
 const h=pickupHarness();h.panel.selectLoadMode('pickup');
 for(const changed of [pickupState({pickup_grasp_pose:null,pickup_approach_pose:null}),pickupState({pickup_approach_offset_mm:50})]){
  h.panel.physicalStatus(changed);h.nodes.pickupSetupConfirm.checked=true;h.nodes.pickupFixedConfirm.checked=true;assert.equal(await h.panel.pickupCommand('go_pickup_approach'),false);assert.equal(await h.panel.pickupCommand('pickup_drop_cycle'),false);
 }
 h.panel.physicalStatus(pickupState());h.physical.hasUnsavedPoseEdits=true;h.nodes.pickupSetupConfirm.checked=true;h.nodes.pickupFixedConfirm.checked=true;h.panel.render();assert.match(h.nodes.pickupStatus.textContent,/Save or discard/);
 for(const action of ['go_loading_approach','go_pickup_approach','pickup_drop_cycle'])assert.equal(await h.panel.pickupCommand(action),false);
 assert.deepEqual(h.commands,[]);
});

test('an immediate successful Stop with a new action epoch invalidates both pickup confirmations',()=>{
 const h=pickupHarness();h.panel.selectLoadMode('pickup');h.panel.physicalStatus(pickupState({action_epoch:10,action_token:'same-session'}));
 h.nodes.pickupSetupConfirm.checked=true;h.nodes.pickupFixedConfirm.checked=true;h.panel.render();assert.equal(h.nodes.pickupCycle.disabled,false);
 // Stop can finish before the next status poll: no busy/error phase need be seen.
 h.panel.physicalStatus(pickupState({action_epoch:11,action_token:'same-session',busy:false,phase:'idle',fault:false,error:null}));
 assert.equal(h.nodes.pickupSetupConfirm.checked,false);assert.equal(h.nodes.pickupFixedConfirm.checked,false);assert.equal(h.nodes.pickupCycle.disabled,true);assert.equal(h.commands.length,0);
});
