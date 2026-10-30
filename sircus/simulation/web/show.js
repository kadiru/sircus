import {mountPhysicalPanel} from './physical.js';
import {AudioPanel} from './audio.js';
const MODES=['deterministic','trained','simulator','mock'];
const NUMERICAL_MODES=['simulator','trained','mock'];
const SOUNDS=['unknown','carry'];
const BALL_IDS=['ball-pink','ball-purple','ball-green','ball-gray'];
const PICKUP_ACTIONS=['teach_pickup','go_pickup_approach','pickup_drop_cycle','go_loading_approach'];
const POSE_KEYS=['x','y','z','o_x','o_y','o_z','theta'];
const validPose=pose=>Boolean(pose&&POSE_KEYS.every(key=>Number.isFinite(pose[key])));
const copy=value=>JSON.parse(JSON.stringify(value));
export function workspaceURL(value,mode){
  if(!NUMERICAL_MODES.includes(mode))throw Error('Unknown numerical mode');
  const url=new URL(value||'/operations.html','http://127.0.0.1');
  if(!['http:','https:'].includes(url.protocol)||!['127.0.0.1','localhost'].includes(url.hostname)||url.pathname!=='/operations.html'||url.username||url.password)throw Error('Only the local numerical workspace can be embedded.');
  url.search='?embed='+mode;url.hash='';
  return value?.startsWith('/')?url.pathname+url.search:url.href;
}
export function bindingValue(values){
  const clean=values.map(value=>String(value).trim());if(clean.every(value=>!value))return null;
  const [input_id,channel,note]=clean.map(value=>value===''?NaN:Number(value));
  if(![input_id,channel,note].every(Number.isSafeInteger)||(input_id===0||input_id<-(2**31)||input_id>2**31-1)||channel<1||channel>16||note<0||note>127)throw Error('A binding needs a nonzero signed input ID, channel 1–16 and note 0–127. Leave all three blank for unbound.');
  return {input_id,channel,note};
}
function el(document,tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
export class ShowPanel{
  constructor({document,fetchImpl=(...args)=>globalThis.fetch(...args),physical=null,getAudioState=()=>null}){
    this.document=document;this.fetch=fetchImpl;this.physical=physical;this.getAudioState=getAudioState;this.numericalTrial=null;this.state=null;this.mode='simulator';this.hardwareViewSelected=false;this.selectedBall=null;this.token=null;this.pending=false;this.epoch=0;this.polling=false;this.workspaceActivity=new Map();this.dirty=false;this.editors=new Map();this.editorKey=null;this.$=id=>document.getElementById(id);
    this.manualLoadMode='hand';this.pickupSignature=null;this.pickupPending=false;
    this.$('allFourBalls')?.addEventListener('click',()=>{if(this.selectLoadMode('round'))this.$('physicalRoundStatus')?.scrollIntoView?.({behavior:'smooth',block:'center'});});
    this.$('manualFourBallRound')?.addEventListener('click',()=>this.selectLoadMode('round'));
    this.$('manualHandLoad')?.addEventListener('click',()=>this.selectLoadMode('hand'));
    this.$('manualTapedPickup')?.addEventListener('click',()=>this.selectLoadMode('pickup'));
    for(const id of ['pickupSetupConfirm','pickupFixedConfirm','pickupPractice','pickupNoCamera'])this.$(id)?.addEventListener('change',()=>this.render());
    for(const [id,action]of [['pickupTeach','teach_pickup'],['pickupApproach','go_pickup_approach'],['pickupCycle','pickup_drop_cycle'],['pickupLoadingApproach','go_loading_approach']])this.$(id)?.addEventListener('click',()=>void this.pickupCommand(action));
    for(const mode of MODES)this.$('mode'+mode[0].toUpperCase()+mode.slice(1)).addEventListener('click',()=>this.selectMode(mode));
    this.$('editBallBindings').addEventListener('click',()=>{const hidden=!this.$('ballSettings').hidden;this.$('ballSettings').hidden=hidden;this.$('editBallBindings').setAttribute('aria-expanded',String(!hidden));});
    this.$('saveBallSettings').addEventListener('click',()=>void this.saveConfig());
    this.$('reapplyAudioMappings')?.addEventListener('click',()=>void this.reapplyAudioMappings());
    this.$('replaySimulationAudio').addEventListener('click',()=>void this.replayAudio());this.$('stopSimulationAudio').addEventListener('click',()=>void this.action({action:'stop_audio_replay'}));
    this.render();
  }
  message(text,error=false){this.$('showMessage').textContent=text;this.$('showMessage').classList.toggle('error',error);}
  physicalStatus(value){this.physicalState=value;if(this.mode==='deterministic'&&value&&Object.hasOwn(value,'selected_ball_id'))this.selectedBall=value.selected_ball_id;const c=value?.calibration||{},signature=JSON.stringify([value?.selected_ball_id,c.pickup_poses?.[value?.selected_ball_id],c.loading_pose,c.drop_pose,c.board_joints,c.board_pose,c.setup_revision,value?.action_token,value?.action_epoch]);if((this.pickupSignature!==null&&signature!==this.pickupSignature)||!value?.connected||value?.fault||value?.error||value?.busy)this.resetPickupConfirmation();this.pickupSignature=signature;this.render();}
  async request(path,body){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),5000);try{const response=await this.fetch(path,{cache:'no-store',signal:controller.signal,...(body?{method:'POST',headers:{'Content-Type':'application/json','X-Sircus-Show-Token':this.token||''},body:JSON.stringify(body)}:{})});const value=await response.json();if(!response.ok||value.error)throw Error(value.error||'Show desk request unavailable');return value;}finally{clearTimeout(timer);}}
  accept(value){if(!value||typeof value!=='object'||!value.modes)throw Error('Show desk is not enabled on this server.');this.state=value;this.token=value.action_token||this.token;if(value.modes.deterministic?.mode==='hardware'&&!this.hardwareViewSelected){this.hardwareViewSelected=true;this.mode='deterministic';this.manualLoadMode='round';this.selectedBall=this.physicalState?.selected_ball_id||null;this.physical?.selectView(true);}const config=value.audio_show?.config;if(config){if(config.schema!=='sircus.show-audio.v1'||!Array.isArray(config.balls)||config.balls.length!==4||new Set(config.balls.map(row=>row.ball_id)).size!==4||config.balls.some(row=>!BALL_IDS.includes(row.ball_id)))throw Error('Four-ball configuration is unavailable.');if(!this.selectedBall&&this.mode!=='deterministic')this.selectedBall=config.balls[0].ball_id;const key=JSON.stringify(config);if(key!==this.editorKey&&!this.dirty){this.buildEditors(config,value.audio_show.catalog||[]);this.editorKey=key;}}
    this.render();
  }
  async refresh(){if(this.polling)return;this.polling=true;const epoch=this.epoch;try{const value=await this.request('/api/show/status');if(epoch===this.epoch){this.accept(value);if(this.available('simulator')&&this.state.modes.deterministic?.mode!=='hardware'){try{const trial=await this.request('/api/operations/status');if(epoch===this.epoch&&trial.mode==='simulation_only'&&trial.hardware_enabled===false){this.numericalTrial=trial;this.render();}}catch{this.numericalTrial=null;this.render();}}if(!this.pending)this.message('Choose a mode. Every routine starts with an operator action. Physical bounce sounds use measured generic impacts.');}}catch(error){if(epoch===this.epoch)this.message(error.message,true);}finally{this.polling=false;}}
  workspaceStatus(event,origin){if(event.origin!==origin||event.data?.type!=='sircus-workspace-activity'||typeof event.data.busy!=='boolean')return false;const mode=NUMERICAL_MODES.find(name=>name===this.mode&&this.$(name+'Frame')?.getAttribute('src')&&this.$(name+'Frame')?.contentWindow===event.source);if(!mode||event.source==null)return false;this.workspaceActivity.set(mode,event.data.busy);this.render();return true;}
  numericalBusy(){return this.state?.numerical_busy===true||this.numericalTrial?.busy===true||[...this.workspaceActivity.values()].some(Boolean);}
  selectMode(mode){if(!MODES.includes(mode))return false;if(mode!=='deterministic'&&this.state?.modes?.deterministic?.mode==='hardware'){this.message('This hardware server runs the saved-pose routine. Open a separate simulator for numerical experiments.');return false;}if(this.mode!==mode&&(this.physical?.busy||this.physicalState?.busy||this.numericalBusy())){this.message('Finish or stop the active routine before changing acts. Stop remains available in the current workspace.',true);return false;}this.mode=mode;if(mode!=='deterministic'&&!this.selectedBall)this.selectedBall=this.state?.audio_show?.config?.balls[0]?.ball_id||null;this.physical?.selectView(mode==='deterministic');this.render();return true;}
  available(mode){return this.state?.modes?.[mode]?.available===true;}
  render(){
    const modeData=this.state?.modes||{},physicalMode=modeData.deterministic?.mode||this.state?.physical_mode||'disabled';
    this.$('showRuntime').textContent=physicalMode==='hardware'?'PHYSICAL BACKEND ENABLED':physicalMode==='dry_run'?'DRY RUN · NO ROBOTS':'NO PHYSICAL BACKEND';this.$('showRuntime').classList.toggle('hardware',physicalMode==='hardware');this.$('deterministicBadge').textContent=physicalMode==='hardware'?'HARDWARE':physicalMode==='dry_run'?'DRY RUN':'UNAVAILABLE';
    for(const mode of MODES){const node=this.$('mode'+mode[0].toUpperCase()+mode.slice(1));node.classList.toggle('selected',this.mode===mode);node.setAttribute('aria-pressed',String(this.mode===mode));node.disabled=(physicalMode==='hardware'&&mode!=='deterministic')||(this.mode!==mode&&(this.physical?.busy||this.physicalState?.busy||this.numericalBusy()));}
    this.$('showSimulatorPane').hidden=this.mode!=='simulator';this.$('showTrainedPane').hidden=this.mode!=='trained';this.$('showMockPane').hidden=this.mode!=='mock';this.$('physicalPanel').hidden=this.mode!=='deterministic';
    for(const mode of NUMERICAL_MODES){const available=this.available(mode)&&physicalMode!=='hardware';this.$(mode+'Unavailable').hidden=available;const frame=this.$(mode+'Frame');frame.hidden=!available;if(available&&this.mode===mode&&!frame.getAttribute('src')){try{frame.setAttribute('src',workspaceURL(modeData[mode]?.url||'/operations.html',mode));}catch(error){this.message(error.message,true);}}if((!available||this.mode!==mode)&&frame.getAttribute('src')){frame.removeAttribute('src');this.workspaceActivity.delete(mode);}}
    const replayRunning=this.state?.audio_show?.replay_running===true,trialReady=this.numericalTrial?.phase==='complete'&&!!this.numericalTrial.trial_id;
    const audio=this.getAudioState(),audioEnabled=audio?.enabled===true,audioCompatible=audio?.show_audio_capable===true,replayError=this.state?.audio_show?.error;
    this.$('simulationAudioBar').hidden=this.mode==='deterministic'||physicalMode==='hardware';this.$('replaySimulationAudio').disabled=!this.token||!trialReady||!audioEnabled||!audioCompatible||replayRunning||this.pending;this.$('stopSimulationAudio').disabled=!this.token||(!replayRunning&&!this.pending);
    const replayStatus=this.$('simulationAudioStatus');replayStatus.classList.toggle('error',Boolean(replayError));
    replayStatus.textContent=replayError?String(replayError):replayRunning?'Synthetic audio replay in progress.':trialReady?(!audioEnabled?'Enable FX to hear the completed synthetic trial.':!audioCompatible?'Audio service has not verified four-ball replay support. Check the audio service connection.':'Completed trial '+this.numericalTrial.trial_id.slice(0,8)+' is ready for a separate audio replay.'):'Complete a simulated trial to hear its recorded contact sequence.';
    const config=this.state?.audio_show?.config,balls=config?.balls||[];const busy=this.pending||this.physical?.busy||this.physicalState?.busy;
    this.renderHardwarePreparation(physicalMode,balls,audio);const reapply=this.$('reapplyAudioMappings');if(reapply)reapply.disabled=!this.token||!config||busy||this.dirty||audio?.configured!==true;
    const fullRound=this.mode==='deterministic'&&this.manualLoadMode==='round';
    if(this.$('allFourBalls')){this.$('allFourBalls').hidden=this.mode!=='deterministic';this.$('allFourBalls').disabled=Boolean(busy);this.$('allFourBalls').setAttribute('aria-pressed',String(fullRound));}
    this.$('selectedBallStatus').textContent=fullRound?'All four: Gray → Purple → Pink → Green. Use the round buttons below; no individual selection is required.':this.mode==='deterministic'&&!this.physicalState?.selected_ball_id?'No physical ball identity selected. Choose a color to associate measured impacts.':'Operator selection · color is a label, not a camera detection.';
    this.$('ballChoices').replaceChildren(...balls.map(row=>{const button=el(this.document,'button');button.type='button';button.disabled=busy;button.classList.toggle('selected',!fullRound&&row.ball_id===this.selectedBall);button.setAttribute('aria-pressed',String(!fullRound&&row.ball_id===this.selectedBall));const sphere=el(this.document,'i',undefined,'ball-sphere');sphere.style.background=row.color;button.append(sphere,el(this.document,'span',row.label));button.addEventListener('click',()=>void this.selectBall(row.ball_id));return button;}));
    this.$('saveBallSettings').disabled=!this.token||!this.dirty||busy;this.$('bindingStatus').textContent=this.dirty?'Unsaved identity/sound edits.':this.state?.audio_show?.capabilities?.multi_input===false?'Saved. Multiple physical input endpoints still need commissioning.':'Bindings are operator-supplied. Unbound identities are not automatically recognized.';
    this.renderPickupStation(balls);
  }
  resetPickupConfirmation(){for(const id of ['pickupSetupConfirm','pickupFixedConfirm'])if(this.$(id))this.$(id).checked=false;}
  selectLoadMode(mode){
    if(!['round','hand','pickup'].includes(mode)||this.pickupPending||this.pending||this.physical?.busy||this.physicalState?.busy)return false;
    if(this.manualLoadMode!==mode){this.manualLoadMode=mode;this.resetPickupConfirmation();if(this.$('physicalCycleConfirm'))this.$('physicalCycleConfirm').checked=false;}
    if(this.$('physicalRoundConfirm'))this.$('physicalRoundConfirm').checked=false;
    this.render();return true;
  }
  pickupAllowed(action){
    if(!PICKUP_ACTIONS.includes(action)||this.mode!=='deterministic'||this.manualLoadMode!=='pickup'||this.pending||this.pickupPending||this.physical?.busy||this.physicalState?.busy||!this.physicalState?.connected||this.physicalState?.fault||this.physicalState?.error||!this.physical?.allowed(action))return false;
    if(action!=='teach_pickup'&&this.physical?.hasUnsavedPoseEdits)return false;
    if(!BALL_IDS.includes(this.selectedBall)||this.physicalState.selected_ball_id!==this.selectedBall)return false;
    if(action==='teach_pickup')return true;
    if(!this.$('pickupSetupConfirm')?.checked)return false;
    if(action==='go_loading_approach')return validPose(this.physicalState.loading_approach_pose);
    if(!validPose(this.physicalState.pickup_grasp_pose)||!validPose(this.physicalState.pickup_approach_pose)||this.physicalState.pickup_approach_offset_mm!==100)return false;
    return action!=='pickup_drop_cycle'||(this.$('pickupFixedConfirm')?.checked===true&&(!this.$('pickupNoCamera')?.checked||this.$('pickupPractice')?.checked===true));
  }
  renderPickupStation(balls){
    const panel=this.$('pickupStationPanel');if(!panel)return;
    const busy=this.pending||this.pickupPending||this.physical?.busy||this.physicalState?.busy,pickup=this.manualLoadMode==='pickup';
    panel.hidden=!pickup;this.$('handLoadCycle').hidden=this.manualLoadMode!=='hand';this.$('handLoadControls').hidden=this.manualLoadMode!=='hand';
    for(const [id,selected]of [['manualFourBallRound',this.manualLoadMode==='round'],['manualHandLoad',this.manualLoadMode==='hand'],['manualTapedPickup',pickup]]){this.$(id).disabled=Boolean(busy);this.$(id).setAttribute('aria-pressed',String(selected));}
    const s=this.physicalState,ball=balls.find(row=>row.ball_id===s?.selected_ball_id),grasp=s?.pickup_grasp_pose,approach=s?.pickup_approach_pose;
    this.$('pickupSelectedBall').textContent=ball?`${ball.label} · operator-selected identity. Each color has its own taught pickup pose.`:'Choose a physical ball color first.';
    this.$('pickupPose').textContent=validPose(grasp)?'Saved arm 10 flange pose in world\n'+JSON.stringify(grasp,null,2):'Pickup pose not taught for this color.';
    this.$('pickupApproachPose').textContent=validPose(approach)?'Approach flange pose in world · +100 mm Z\n'+JSON.stringify(approach,null,2):'Approach pose unavailable until pickup is taught.';
    const locked=Boolean(busy)||!s?.connected||Boolean(s?.fault||s?.error)||!ball||this.physical?.hasUnsavedPoseEdits;
    this.$('pickupSetupConfirm').disabled=locked;this.$('pickupFixedConfirm').disabled=locked;this.$('pickupPractice').disabled=Boolean(busy);if(this.$('pickupNoCamera')){if(s?.capabilities?.cameras_enabled===false)this.$('pickupNoCamera').checked=true;this.$('pickupNoCamera').disabled=Boolean(busy)||!this.$('pickupPractice').checked||s?.capabilities?.cameras_enabled===false;}
    for(const [id,action]of [['pickupTeach','teach_pickup'],['pickupApproach','go_pickup_approach'],['pickupCycle','pickup_drop_cycle'],['pickupLoadingApproach','go_loading_approach']])this.$(id).disabled=!this.pickupAllowed(action);
    const blockers=s?.pickup_station_blockers||s?.pickup_station_missing||[];
    let message='Choose a color, teach its pickup pose, then verify pad and route clearance.';
    if(!s?.connected)message='Fresh robot state is required. No pickup command is sent automatically.';
    else if(s?.fault||s?.error)message='Pickup blocked: '+(s.error||'controller fault; verify Stop and robot state');
    else if(busy){const steps={pickup_to_approach:'Moving to the pickup approach.',pickup_opening:'Opening above the pad.',pickup_descending:'Descending vertically to the taught pickup pose.',pickup_closing:'Closing and checking whether a ball is held.',pickup_lifting:'Lifting the held ball vertically by 100 mm.',pickup_clearance_lifting:'Raising vertically above the higher pickup stations before transfer.',cycle_to_drop:'Carrying to the unchanged saved drop pose.',cycle_release_checks:'Settling for 2 seconds and checking release readiness.',post_release_wait:'Release returned; waiting before ending the camera-off demo.',post_release_recording:'Release returned; finishing trial recording.',go_pickup_approach:'Moving above the selected pickup pad; gripper unchanged.',go_loading_approach:'Moving above the old loading target; gripper unchanged.',stopping:'Stopping arm 10 and waiting for stationary state.'};message=(steps[s?.phase]||'Request submitted; waiting for controller state.')+' Stop arm 10 remains available.';}
    else if(this.physical?.hasUnsavedPoseEdits)message='Save or discard unsaved loading/drop target edits before moving.';
    else if(blockers.length)message='Pickup blocked: '+blockers.join(' · ');
    else if(!validPose(grasp))message='Teach the selected ball’s pickup pose. Teaching reads only; it does not move the arm.';
    else if(this.pickupAllowed('pickup_drop_cycle'))message='Ready for one pickup-and-drop routine. The saved drop pose is unchanged.';
    else message='Confirm pad, neighboring-ball/jaw clearance, fixed board/bucket and route before this attempt.';
    this.$('pickupStatus').textContent=message;this.$('pickupStatus').classList.toggle('ready',this.pickupAllowed('pickup_drop_cycle'));
  }
  async pickupCommand(action){
    if(!this.pickupAllowed(action))return false;
    const payload=action==='go_loading_approach'?{}:{ball_id:this.selectedBall};
    if(action!=='teach_pickup')payload.pickup_setup_confirmed=this.$('pickupSetupConfirm').checked;
    if(action==='pickup_drop_cycle'){payload.fixed_setup_confirmed=this.$('pickupFixedConfirm').checked;payload.practice_mode=this.$('pickupPractice').checked;if(this.$('pickupNoCamera')?.checked)payload.recording_mode='operator_log_only';}
    this.pickupPending=true;this.resetPickupConfirmation();this.render();
    try{return await this.physical.command(action,payload);}
    catch(error){this.message(error.message||'Pickup request failed; check robot state. No retry was issued.',true);return false;}
    finally{this.pickupPending=false;this.render();}
  }
  renderHardwarePreparation(physicalMode,balls,audio){
    const panel=this.$('hardwarePreparation');if(!panel)return;panel.hidden=physicalMode!=='hardware';
    const robot=this.physicalState,liveAudio=audio?.configured===true&&audio?.connection==='connected';
    this.$('hardwareRobotStatus').textContent=robot?.connected===true?'Live state received':'Awaiting live state';
    const poses=robot?.calibration?.loading_pose&&robot?.calibration?.drop_pose&&Array.isArray(robot?.fixed_cycle_missing)&&robot.fixed_cycle_missing.length===0;
    this.$('hardwarePoseStatus').textContent=poses?'Saved targets present':'Loading/drop targets need checking';
    this.$('hardwareEngineStatus').textContent=!liveAudio?'Audio service not verified':audio.engine_ready===true?'Native sound engine ready':audio.engine_ready===false?'Native sound engine unavailable':'Native sound engine unknown';
    const bound=balls.filter(ball=>ball.midi_binding).length,connected=liveAudio&&audio.mapping_matches===true&&Number.isInteger(audio.connected_ball_count)&&audio.connected_ball_count>=0&&audio.connected_ball_count<=4?audio.connected_ball_count:null;
    this.$('hardwareBallStatus').textContent=`${bound}/4 saved bindings · ${connected===null?'connection unknown':connected+'/4 inputs connected'}`;
    this.$('hardwareAudioHint').textContent=!liveAudio?'Start Ghost Booth on this Mac, then reapply saved mappings. Logic Pro is not required.':audio.mapping_matches!==true?'Saved mappings are not confirmed by the audio service. Reapply them while idle, then check each ball.':'Verify each physical ball by tapping it and listening before running the routine. Connection alone does not verify a bounce or sound.';
  }
  async reapplyAudioMappings(){
    if(!this.token||!this.state?.audio_show?.config||this.pending||this.dirty||this.physical?.busy||this.physicalState?.busy||this.getAudioState()?.configured!==true)return false;
    return this.action({action:'configure_audio',config:copy(this.state.audio_show.config)});
  }
  async selectBall(ballId){if(this.pending||this.pickupPending||this.physical?.busy||this.physicalState?.busy)return false;if(!this.state?.audio_show?.config?.balls.some(row=>row.ball_id===ballId))return false;if(this.mode==='deterministic'){if(!this.physical?.allowed('set_ball')){this.message('Ball selection is locked until the deterministic backend is idle and ready.',true);return false;}this.resetPickupConfirmation();const ok=await this.physical.command('set_ball',{ball_id:ballId});if(!ok)return false;}this.selectedBall=ballId;if(this.mode==='deterministic'&&this.manualLoadMode==='round')this.selectLoadMode('pickup');this.render();return true;}
  async action(payload){if(!this.token||(this.pending&&payload.action!=='stop_audio_replay'))return false;const epoch=++this.epoch;this.pending=true;this.render();try{const value=await this.request('/api/show/action',payload);if(epoch!==this.epoch)return false;if(value.modes)this.accept(value);else if(value.status?.modes)this.accept(value.status);if(payload.action==='replay_simulation'&&this.state?.audio_show?.error)throw Error(this.state.audio_show.error);this.message(payload.action==='configure_audio'?'Ball identities and sound assignments saved.':payload.action==='stop_audio_replay'?'Audio replay stopped. Robot controls are unaffected.':'Synthetic audio replay requested; it does not move an arm or synchronize the viewer.');return true;}catch(error){if(epoch===this.epoch)this.message(error.message,true);return false;}finally{if(epoch===this.epoch){this.pending=false;this.render();}}}
  replayAudio(){if(this.mode==='deterministic'||!this.available('simulator')||this.state?.modes?.deterministic?.mode==='hardware'||this.numericalTrial?.mode!=='simulation_only'||this.numericalTrial?.hardware_enabled!==false||this.numericalTrial?.phase!=='complete'||this.getAudioState()?.enabled!==true||this.getAudioState()?.show_audio_capable!==true)return false;return this.action({action:'replay_simulation',trial_id:this.numericalTrial.trial_id,ball_id:this.selectedBall});}
  buildEditors(config,catalog){this.editors.clear();this.$('ballEditors').replaceChildren(...config.balls.map(row=>{
    const card=el(this.document,'section',undefined,'ball-editor');card.append(el(this.document,'h3',row.ball_id.replace('ball-','').toUpperCase()));const fields={};
    const field=(key,label,type,value,parent=card)=>{const wrapper=el(this.document,'label',label),input=el(this.document,'input');input.type=type;input.value=value??'';if(type==='text')input.maxLength=40;input.addEventListener('input',()=>{this.dirty=true;this.render();});wrapper.append(input);parent.append(wrapper);fields[key]=input;return input;};
    field('label','Display name','text',row.label);field('color','Color label','color',row.color);const grid=el(this.document,'div',undefined,'binding-grid');card.append(grid);field('input_id','Input ID','number',row.midi_binding?.input_id,grid);field('channel','Channel','number',row.midi_binding?.channel,grid);field('note','Note','number',row.midi_binding?.note,grid);
    for(const key of SOUNDS){const wrapper=el(this.document,'label',({surface:'Board bounce',bucket:'Bucket bounce',collision:'Ball collision',unknown:'Measured generic bounce',carry:'Carrying + 2 s settling'})[key]),select=el(this.document,'select');for(const name of catalog){const option=el(this.document,'option',String(name).replaceAll('_',' '));option.value=name;select.append(option);}select.value=row.sounds?.[key]||'';select.addEventListener('change',()=>{this.dirty=true;this.render();});wrapper.append(select);card.append(wrapper);fields[key]=select;}
    this.editors.set(row.ball_id,fields);return card;
  }));}
  async saveConfig(){try{const config=copy(this.state.audio_show.config);config.balls=config.balls.map(row=>{const f=this.editors.get(row.ball_id),label=f.label.value.trim(),color=f.color.value;if(!label||label.length>40||!/^#[0-9a-f]{6}$/i.test(color))throw Error('Each ball needs a short name and a valid color label.');const sounds={...row.sounds,...Object.fromEntries(SOUNDS.map(key=>[key,f[key].value]))};if(Object.values(sounds).some(value=>!this.state.audio_show.catalog.includes(value)))throw Error('Choose each sound from the available catalog.');return {...row,label,color,midi_binding:bindingValue([f.input_id.value,f.channel.value,f.note.value]),sounds};});const ok=await this.action({action:'configure_audio',config});if(ok){this.dirty=false;this.editorKey=null;await this.refresh();}return ok;}catch(error){this.message(error.message,true);return false;}}
}
if(typeof document!=='undefined'&&document.documentElement.hasAttribute('data-show-shell')){
  let show,audio;const physical=mountPhysicalPanel({document,manageTabs:false,onStatus:value=>show?.physicalStatus(value)});show=new ShowPanel({document,physical,getAudioState:()=>audio?.state});const activity=event=>show.workspaceStatus(event,window.location.origin);window.addEventListener('message',activity);audio=new AudioPanel({document}).start();void show.refresh();const timer=setInterval(()=>void show.refresh(),2500);window.addEventListener('pagehide',()=>{clearInterval(timer);window.removeEventListener('message',activity);physical.destroy();audio.destroy();},{once:true});
}
