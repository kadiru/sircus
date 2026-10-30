import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { StudyPanel } from './study-panel.mjs';

const $ = id => document.getElementById(id);
const clone = value => JSON.parse(JSON.stringify(value));
const mm = value => value / 1000;
const rad = value => value * Math.PI / 180;
const colors = { cream: 0xe7e6da, board: 0x439e97, ball: 0xf27a57, gold: 0xe6ba5b, ink: 0x174149 };
let config, initialConfig, result = null, resultConfig = null, resultAction = null, busy = false, playing = false, elapsed = 0, lastTime = 0;
let currentArm = [], armDuration = 0, armSequence = [], physicsFrames = [], armIndex = -1, physicsIndex = -1;
let renderer, scene, camera, controls, board, ball, targetPoint, support, staticGroup, armGroup, trail, trailGeometry;
let viamConnected=false, offline=false, nativeState=null, polling=null, pollingStatus=false, sceneDirty=false;
let displayedTrainingId=null;
let localOnly=false;
let studyMode=false, studyEnvelope=null, studyPanel=null;
let standardViewCopy=null;
let preStudyAction=null;
const armMeshes = new Map();
const fields = {
  boardX:['board','center_mm',0],boardY:['board','center_mm',1],boardZ:['board','center_mm',2],
  boardRoll:['board','rpy_deg',0],boardPitch:['board','rpy_deg',1],boardYaw:['board','rpy_deg',2],
  binX:['bin','center_xy_mm',0],binY:['bin','center_xy_mm',1],
  boardLength:['board','size_mm',0],boardWidth:['board','size_mm',1],boardThickness:['board','size_mm',2],
  boardMass:['board','mass_kg'],mountX:['board','support_offset_mm',0],mountY:['board','support_offset_mm',1],mountZ:['board','support_offset_mm',2],suctionOffset:['robot14','suction_offset_mm']
};

function message(text, error=false) { $('message').textContent=text; $('message').classList.toggle('error',error); }
function phase(text, active=false) { $('phase').textContent=text; $('phase').parentElement.classList.toggle('busy',active); }
async function request(url, body) {
  let response,data;
  try {
    response = await fetch(url, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    data = await response.json();
  } catch (error) {
    updateConnection({backend:offline?'offline':'viam',connected:false});
    throw new Error('Dashboard connection lost. Reconnect before starting another task.');
  }
  if(data.connection) updateConnection(data.connection);
  if (!response.ok || data.error) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function setBusy(value, label='') {
  busy=value;
  for (const el of document.querySelectorAll('.controls input,.controls textarea,.controls button,.preset')) if(!el.closest('#studyPanel'))el.disabled=value;
  $('replayButton').disabled=value || !result;
  $('downloadRun').disabled=value || !result;
  $('stopButton').disabled=!physicsFrames.length;
  $('cancelButton').disabled=!value || offline || studyMode || studyPanel?.pending;
  $('suctionOffset').disabled=value || offline;
  for(const id of ['dropButton','learnButton','applyJson'])$(id).disabled=value || !config || (!offline&&!viamConnected);
  if (label) {message(label);phase(label,true);}
  lockStudyControls();
}
function lockStudyControls(){
  if(studyMode){
    for(const el of document.querySelectorAll('.controls input,.controls textarea,.controls button,.preset')){
      if(!el.closest('#studyPanel')&&!['replayButton','downloadRun','stopButton','reconnectButton'].includes(el.id))el.disabled=true;
    }
  }
  studyPanel?.render();
}
function syncInputs() {
  for (const [id,path] of Object.entries(fields)) $(id).value=path.reduce((v,k)=>v?.[k],config) ?? '';
  $('suctionOffset').disabled=offline;
  $('configJson').value=JSON.stringify(config,null,2);
  $('boardSpec').textContent=`Board ${config.board.size_mm.join(' × ')} mm`;
  const bin=config.bin,bucket=bin.shape==='tapered_bucket';
  $('containerTitle').textContent=bucket?'Target bucket':'Target container';
  $('containerSpec').hidden=!bucket;
  if(bucket){
    $('containerDiameters').textContent=`Inside: Ø${bin.inner_top_diameter_mm} opening / Ø${bin.inner_bottom_diameter_mm} bottom. Outside: Ø${bin.outer_top_diameter_mm} top / Ø${bin.outer_bottom_diameter_mm} bottom. All mm.`;
    $('containerHeight').textContent=`Reported height: ${bin.reported_height_mm} mm. Overall height versus interior depth is awaiting clarification.`;
    $('containerAssumptions').textContent=`Model assumptions: ${bin.interior_depth_mm} mm interior depth; ${bin.bottom_thickness_mm} mm bottom thickness; straight taper. Placement is provisional.`;
  }
  document.querySelector('.ball-spec div span').textContent=`${config.ball.diameter_mm} mm · ${Math.round(config.ball.mass_kg*1000)} g`;
}
function readConfig() {
  const value=clone(config);
  for (const [id,path] of Object.entries(fields)) {
    if(id==='suctionOffset' && offline) continue;
    const number=Number($(id).value);
    if ($(id).value.trim()==='' || !Number.isFinite(number)) throw new Error('Enter finite board and container coordinates.');
    let parent=value;for(const key of path.slice(0,-1)){if(parent[key]===undefined)parent[key]={};parent=parent[key];}parent[path[path.length-1]]=number;
  }
  return value;
}
function action() {
  const output={u_mm:Number($('dropU').value),v_mm:Number($('dropV').value),gap_mm:Number($('dropGap').value)};
  if (['dropU','dropV','dropGap'].some(id=>$(id).value.trim()==='') || Object.values(output).some(v=>!Number.isFinite(v))) throw new Error('Enter a finite release position and gap.');
  return output;
}
function seed() {const value=Number($('seed').value);if (!Number.isSafeInteger(value)||value<0) throw new Error('Seed must be a nonnegative integer.');return value;}
function applyAction(value) {if(!value)return; $('dropU').value=Number(value.u_mm).toFixed(2); $('dropV').value=Number(value.v_mm).toFixed(2); $('dropGap').value=Number(value.gap_mm).toFixed(2);}
function setPosition(object, xyz) {object.position.set(...xyz.map(mm));}
function setRotation(object, rpy, matrix) {
  if (matrix) {const m=new THREE.Matrix4();m.set(matrix[0][0],matrix[0][1],matrix[0][2],0,matrix[1][0],matrix[1][1],matrix[1][2],0,matrix[2][0],matrix[2][1],matrix[2][2],0,0,0,0,1);object.quaternion.setFromRotationMatrix(m);}
  else object.rotation.set(rad(rpy[0]),rad(rpy[1]),rad(rpy[2]),'ZYX');
}
function disposeGroup(group) {while(group.children.length){const child=group.children[0];child.traverse(object=>{object.geometry?.dispose(); if(Array.isArray(object.material)) object.material.forEach(m=>m.dispose()); else object.material?.dispose();});group.remove(child);}}
function box(size,color,{opacity=1,edges=true,metalness=0,roughness=.6}={}) {
  const group=new THREE.Group();
  const geometry=new THREE.BoxGeometry(...size.map(mm));
  const material=new THREE.MeshStandardMaterial({color,roughness,metalness,transparent:opacity<1,opacity,depthWrite:opacity>=1});
  const mesh=new THREE.Mesh(geometry,material);mesh.castShadow=opacity>=1;mesh.receiveShadow=true;group.add(mesh);
  if(edges){const wire=new THREE.LineSegments(new THREE.EdgesGeometry(geometry),new THREE.LineBasicMaterial({color:color===colors.gold?0xf3d58e:0x7eada8,transparent:true,opacity:.35}));group.add(wire);}
  return group;
}
function line(points,color=colors.gold,opacity=.8) {const geometry=new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(...p.map(mm))));return new THREE.Line(geometry,new THREE.LineBasicMaterial({color,transparent:true,opacity}));}
function taperedBucket(bin) {
  // Revolve a closed radial section. Only the annular rim spans the top;
  // the inner opening is empty and the base lies below the interior floor.
  const bottom=bin.bottom_thickness_mm,depth=bin.interior_depth_mm;
  const profile=[
    [bin.outer_bottom_diameter_mm/2,-bottom],
    [bin.outer_top_diameter_mm/2,depth],
    [bin.inner_top_diameter_mm/2,depth],
    [bin.inner_bottom_diameter_mm/2,0],
    [0,0],[0,-bottom],[bin.outer_bottom_diameter_mm/2,-bottom]
  ];
  const geometry=new THREE.LatheGeometry(profile.map(([r,z])=>new THREE.Vector2(mm(r),mm(z))),96);
  geometry.rotateX(Math.PI/2);
  const material=new THREE.MeshStandardMaterial({color:colors.gold,roughness:.55,transparent:true,opacity:.38,depthWrite:false,side:THREE.DoubleSide});
  const group=new THREE.Group(),shell=new THREE.Mesh(geometry,material);shell.receiveShadow=true;group.add(shell);
  const floor=new THREE.Mesh(new THREE.CircleGeometry(mm(bin.inner_bottom_diameter_mm/2),96),new THREE.MeshStandardMaterial({color:colors.gold,roughness:.7,side:THREE.DoubleSide}));
  floor.position.z=.00005;floor.receiveShadow=true;group.add(floor);
  for(const [radius,z] of [[bin.inner_top_diameter_mm/2,depth],[bin.outer_top_diameter_mm/2,depth],[bin.inner_bottom_diameter_mm/2,0]]){
    const points=Array.from({length:97},(_,i)=>[radius*Math.cos(i*2*Math.PI/96),radius*Math.sin(i*2*Math.PI/96),z]);
    group.add(line(points,0xf3d58e,.6));
  }
  setPosition(group,[...bin.center_xy_mm,bin.floor_top_mm]);return group;
}
function buildEnvironment() {
  if(!scene || !config)return;
  disposeGroup(staticGroup);
  const table=box(config.table.size_mm,colors.cream,{roughness:.9});setPosition(table,config.table.center_mm);staticGroup.add(table);
  const bin=config.bin;
  if(bin.shape==='tapered_bucket')staticGroup.add(taperedBucket(bin));
  else{
    const [x,y]=bin.center_xy_mm,[w,d]=bin.inner_size_mm,t=bin.wall_thickness_mm,h=bin.wall_height_mm,f=bin.floor_top_mm;
    const slabs=[[[x,y,f-t/2],[w+2*t,d+2*t,t],.65],[[x-(w+t)/2,y,f+h/2],[t,d+2*t,h],.22],[[x+(w+t)/2,y,f+h/2],[t,d+2*t,h],.22],[[x,y-(d+t)/2,f+h/2],[w,t,h],.14],[[x,y+(d+t)/2,f+h/2],[w,t,h],.25]];
    for(const [center,size,opacity] of slabs){const part=box(size,colors.gold,{opacity});setPosition(part,center);staticGroup.add(part);}
  }
  if(board){scene.remove(board);board.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});}
  board=box(config.board.size_mm,colors.board);scene.add(board);setPosition(board,config.board.center_mm);setRotation(board,config.board.rpy_deg);
  const bw=config.board.size_mm[0],bd=config.board.size_mm[1],z=config.board.size_mm[2]/2+.6;
  board.add(line([[-bw/2+8,0,z],[bw/2-8,0,z]],0x9bd7c8,.35));board.add(line([[0,-bd/2+8,z],[0,bd/2-8,z]],0x9bd7c8,.35));
  for(const radius of [18,40]){const ring=new THREE.Mesh(new THREE.RingGeometry(mm(radius),mm(radius+1),64),new THREE.MeshBasicMaterial({color:0xbce4cf,transparent:true,opacity:.5,side:THREE.DoubleSide}));ring.position.z=mm(z);board.add(ring);}
  targetPoint=new THREE.Mesh(new THREE.RingGeometry(.006,.009,32),new THREE.MeshBasicMaterial({color:colors.gold,side:THREE.DoubleSide}));targetPoint.position.z=mm(z+1);board.add(targetPoint);updateTargetPoint();
  const mount=new THREE.Group();const offset=config.board.support_offset_mm || [0,0,0];mount.position.set(mm(offset[0]),mm(offset[1]),mm(-config.board.size_mm[2]/2-16));
  const bellows=new THREE.Mesh(new THREE.CylinderGeometry(.021,.027,.032,24),new THREE.MeshStandardMaterial({color:0x6d8c82,roughness:.7}));bellows.rotation.x=Math.PI/2;mount.add(bellows);board.add(mount);
  ball.geometry.dispose();ball.geometry=new THREE.SphereGeometry(mm(config.ball.diameter_mm/2),32,24);
  const tableTop=config.table.center_mm[2]+config.table.size_mm[2]/2;
  setPosition(ball,config.pickup_mm||[350,-220,tableTop+config.ball.diameter_mm/2]);
  clearTrail();physicsFrames=[];armSequence=[];armDuration=0;elapsed=0;playing=false;$('stopButton').disabled=true;
}
function updateTargetPoint(){if(targetPoint){targetPoint.position.x=mm(Number($('dropU').value)||0);targetPoint.position.y=mm(Number($('dropV').value)||0);}}
function renderArm(boxes, role='10') {
  if(!boxes)return;
  const prefix=role+':', names=new Set(boxes.map(b=>prefix+b.name));
  for(const [name,mesh] of armMeshes)if(name.startsWith(prefix)&&!names.has(name)){armGroup.remove(mesh);armMeshes.delete(name);mesh.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});}
  boxes.forEach((data,index)=>{
    const key=prefix+data.name;let mesh=armMeshes.get(key);
    if(!mesh){mesh=box(data.size_mm,role==='14'?(index%2===0?0x69afa2:0x458d84):(index%2===0?0xd4ddcf:0xb9cbc3),{metalness:.25,roughness:.4});armMeshes.set(key,mesh);armGroup.add(mesh);}
    setPosition(mesh,data.center_mm);setRotation(mesh,data.rpy_deg||[0,0,0],data.rotation_matrix||data.rotation);
  });
  if(role==='10')currentArm=boxes;
}
function renderNative(data){
  if(!data||studyMode)return;nativeState=data;
  const arms=data.arms||data.native?.arms;
  if(arms)for(const role of ['10','14']){if(arms[role]?.boxes)renderArm(arms[role].boxes,role);}
  if(data.board_pose&&!playing){const p=data.board_pose;if(p.center_mm)setPosition(board,p.center_mm);if(p.rpy_deg||p.rotation)setRotation(board,p.rpy_deg||config.board.rpy_deg,p.rotation);}
  if(data.ball_xyz_mm&&!playing)setPosition(ball,data.ball_xyz_mm);
  const collection=data.data_collection;
  if(collection){if(collection.enabled===false){$('dataStatus').textContent='Trial collection is disabled in Viam.';return;}const queued=collection.pending_files??collection.pending_count??collection.pending_sync_files;const trials=collection.local_trials??collection.trials??collection.records_written?.trial;
    $('dataStatus').textContent=[trials!==undefined?`This session: ${trials} trial${Number(trials)===1?'':'s'}`:'Trial logging active',localOnly?'Local records only · cloud sync disabled':(queued!==undefined?`${queued} files queued for cloud sync`:(collection.sync_configured?'Viam cloud sync configured':'Cloud sync not configured'))].join(' · ');}
}
function updateConnection(value){
  if(studyMode)return;
  localOnly=value.label?.startsWith('LOCAL FAKE-ONLY')===true;
  offline=value.backend==='offline';viamConnected=value.connected===true&&!offline;
  $('connectionLabel').textContent=offline?'OFFLINE SIMULATION':(viamConnected?(value.label||'VIAM · SIMULATION'):'VIAM DISCONNECTED');
  $('connectionBadge').classList.toggle('offline',offline);$('connectionBadge').classList.toggle('disconnected',!value.connected);
  $('viamLink').hidden=offline||!value.machine_url;if(value.machine_url)$('viamLink').href=value.machine_url;
  if(!value.connected)$('dataStatus').textContent='Viam unavailable. Commands are not sent to a local fallback.';
  else if(offline)$('dataStatus').textContent='Local physics only. Viam collection is not connected.';
  for(const id of ['dropButton','learnButton','applyJson'])$(id).disabled=busy || !config || (!offline&&!viamConnected);
}
const activeTask = data => ['running','training','positioning','stopping'].includes(data?.status);
function sceneSignature(value){
  if(Array.isArray(value))return '['+value.map(sceneSignature).join(',')+']';
  if(value&&typeof value==='object')return '{'+Object.keys(value).sort().map(k=>JSON.stringify(k)+':'+sceneSignature(value[k])).join(',')+'}';
  return JSON.stringify(value);
}
function clearLearning(){
  displayedTrainingId=null;$('learning').textContent='Not started';$('learningDetail').textContent='Release position + height';$('historyPanel').hidden=true;
}
function clearEvidence(){
  result=null;resultConfig=null;resultAction=null;playing=false;physicsFrames=[];armSequence=[];clearTrail();
  $('replayButton').disabled=true;$('downloadRun').disabled=true;$('stopButton').disabled=true;
  $('outcome').textContent='Ready to try';$('outcomeDetail').textContent='Run a drop or start a short search.';$('contact').textContent='—';clearLearning();
}
async function waitForTask(first,label){
  let data=first,deadline=Date.now()+20*60*1000;
  while(activeTask(data)){
    if(Date.now()>deadline)throw new Error('The task is still running in Viam. Check its Control tab before starting another task.');
    renderNative(data);const training=data.training||{};
    const detail=training.phase?`${training.phase}: ${training.trial||0}/${training.budget||24}`:(data.step||label);
    phase(detail.replaceAll('_',' '),true);
    if(training.phase){$('learning').textContent=`${training.trial||0} / ${training.budget||24}`;$('learningDetail').textContent=training.phase;if(training.history)historyChart(training.history);}
    await new Promise(resolve=>setTimeout(resolve,700));data=await request('/api/status');
  }
  renderNative(data);
  if(['aborted','cancelled','error','failed'].includes(data.status))throw new Error(data.error||data.message||`Viam task ${data.status}`);
  return data;
}
async function applyScene(next){
  if(studyMode)throw new Error('Return to the Viam scene before changing its configuration.');
  const started=await request('/api/config',{config:next});
  const finished=offline?started:await waitForTask(started,'Positioning board arm');
  const applied=finished.config||started.config||next;
  if(sceneSignature(config)!==sceneSignature(applied))clearEvidence();
  config=clone(applied);sceneDirty=false;syncInputs();buildEnvironment();renderNative(finished);return finished;
}
function clearTrail(){if(trail){scene.remove(trail);trail.geometry.dispose();trail.material.dispose();trail=null;}trailGeometry=null;}
function prepareTrail(frames){clearTrail();const positions=new Float32Array(frames.length*3);frames.forEach((f,i)=>{positions[i*3]=mm(f.ball_xyz_mm[0]);positions[i*3+1]=mm(f.ball_xyz_mm[1]);positions[i*3+2]=mm(f.ball_xyz_mm[2]);});trailGeometry=new THREE.BufferGeometry();trailGeometry.setAttribute('position',new THREE.BufferAttribute(positions,3));trailGeometry.setDrawRange(0,0);trail=new THREE.Line(trailGeometry,new THREE.LineBasicMaterial({color:0xffbe84,transparent:true,opacity:.65}));scene.add(trail);}
function placePhysics(frame){if(!frame)return;setPosition(ball,frame.ball_xyz_mm);if(frame.board_center_mm)setPosition(board,frame.board_center_mm);setRotation(board,frame.board_rpy_deg||config.board.rpy_deg,frame.board_rotation);}
function showMetrics(data) {
  if(data.scoring_version==='sircus-board-bounce-capture-1s-v1'){
    $('outcome').textContent=data.success===true?'Predicted capture · 1 s':String(data.outcome||'unknown').replaceAll('_',' ');
    $('outcome').style.color=data.success===true?'var(--teal)':'var(--ink)';
    $('outcomeDetail').textContent=`Numerical ${data.capture_class||'shot'} · continuous containment ${Number(data.current_containment_s||0).toFixed(3)} s · native paths unchecked`;
    $('contact').textContent=Array.isArray(data.impact_uv_mm)?data.impact_uv_mm.map(v=>Number(v).toFixed(1)).join(' / '):'—';
    return;
  }
  const retained=data.retained===true;
  $('outcome').textContent=retained?'Captured in model':(data.outcome||'Drop complete').replaceAll('_',' ');
  $('outcome').style.color=retained?'var(--teal)':'var(--ink)';
  const tilt=Number(data.max_support_tilt_deg);
  $('outcomeDetail').textContent=Number.isFinite(tilt)?`Peak support tilt ${tilt.toFixed(2)}°`:'Simulated result; no physical verification.';
  $('contact').textContent=Array.isArray(data.impact_uv_mm)?data.impact_uv_mm.map(v=>Number(v).toFixed(1)).join(' / '):'Board missed';
}
function play(data, replay=false) {
  if (!replay) {resultConfig=clone(config);resultAction=data.action ? clone(data.action) : action();}
  if(data.arms)renderNative(data);
  result=data;physicsFrames=data.frames||[];armSequence=data.arm_sequence||[];
  armDuration=armSequence.length?Number(armSequence[armSequence.length-1].t)||0:0;
  elapsed=0;armIndex=-1;physicsIndex=-1;playing=true;lastTime=performance.now();prepareTrail(physicsFrames);
  if(armSequence.length){const first=armSequence[0];renderArm(first.boxes);if(first.ball_xyz_mm)setPosition(ball,first.ball_xyz_mm);}else if(physicsFrames.length)placePhysics(physicsFrames[0]);
  $('replayButton').disabled=false;$('downloadRun').disabled=false;$('stopButton').disabled=false;$('stopButton').textContent='Pause replay';
  $('outcome').textContent='Drop in progress';$('outcomeDetail').textContent='Watching the simulated trajectory.';$('contact').textContent='—';
  phase(studyMode?'Numerical policy replay':armSequence.length?'Recorded arm motion':'Recorded physics replay',true);
}
function showStudyPreview(value){
  if(!studyMode&&!preStudyAction){try{preStudyAction=action();}catch{preStudyAction=null;}}
  studyMode=true;studyEnvelope=clone(value);sceneDirty=true;
  clearEvidence();config=clone(value.selected_scene);syncInputs();applyAction(value.release_action);
  buildEnvironment();armGroup.visible=false;
  setStudyPresentation(true);
  $('connectionLabel').textContent='NUMERICAL POLICY PREVIEW';$('viamLink').hidden=true;
  $('connectionBadge').classList.add('offline');$('connectionBadge').classList.remove('disconnected');
  $('dataStatus').textContent='Synthetic local rollout only. No Viam command or cloud trial was created.';
  $('learning').textContent='Frozen inference';$('learningDetail').textContent=String(value.checkpoint_id).slice(0,16);
  play({...value.result,action:value.release_action},false);
  message('Numerical prediction and trajectory only. Endpoint IK does not establish collision-free native motion.');
  lockStudyControls();
}
function setStudyPresentation(enabled){
  const hint=document.querySelector('.stage-hint'),scope=document.querySelector('#simulationPanel .footnote');
  document.querySelector('.stage-top').style.visibility=enabled?'hidden':'';
  if(!standardViewCopy)standardViewCopy={hint:hint.innerHTML,scope:scope.textContent,aria:$('scene').getAttribute('aria-label')};
  if(enabled){
    hint.textContent='Drag to orbit · Numerical trajectory · Native arm paths unchecked';
    scope.textContent='Frozen scene-conditioned prediction chooses board roll/pitch, U/V and release gap. This synthetic rollout uses versioned whole-ball, one-second capture scoring. Board mass, support dynamics and bucket height interpretation remain provisional; no native path or physical validation is claimed.';
    $('scene').setAttribute('aria-label','Numerical policy trajectory of a ball bouncing from the board toward a tapered bucket. No native arm paths are displayed.');
  }else{hint.innerHTML=standardViewCopy.hint;scope.textContent=standardViewCopy.scope;$('scene').setAttribute('aria-label',standardViewCopy.aria);}
}
function animate(now) {
  requestAnimationFrame(animate);
  // A queued RAF timestamp can predate play()'s performance.now() reset.
  const delta=Math.max(0,Math.min((now-lastTime)/1000,.1));lastTime=now;
  if(playing){
    elapsed+=delta*Number($('speed').value);
    if(armSequence.length&&elapsed<armDuration){
      let index=0;while(index+1<armSequence.length && armSequence[index+1].t<=elapsed)index++;
      if(index!==armIndex){armIndex=index;const frame=armSequence[index];renderArm(frame.boxes);if(frame.ball_xyz_mm)setPosition(ball,frame.ball_xyz_mm);phase((frame.phase||'Arm movement').replaceAll('_',' '),true);}
    }else if(physicsFrames.length){
      if(armSequence.length&&armIndex!==armSequence.length-1){armIndex=armSequence.length-1;renderArm(armSequence[armIndex].boxes);}
      const time=elapsed-armDuration;let index=0;while(index+1<physicsFrames.length&&physicsFrames[index+1].t<=time)index++;
      if(index!==physicsIndex){physicsIndex=index;placePhysics(physicsFrames[index]);trailGeometry?.setDrawRange(0,index+1);phase(time<.25?'Release & fall':'Bounce & settle',true);}
      if(time>physicsFrames[physicsFrames.length-1].t){playing=false;showMetrics(result);phase(studyMode?'Numerical preview complete':'Trial complete');message(studyMode?'Numerical replay complete. Native v2 replay remains blocked; no arm motion was sent.':'Trial complete. Replay it or change the setup.');$('stopButton').textContent='Resume replay';$('stopButton').disabled=true;}
    }else{playing=false;showMetrics(result);phase('No trajectory returned');}
  }
  controls.update();renderer.render(scene,camera);
}
function resetView(){camera.position.set(2.4,2.1,1.8);controls.target.set(.7,0,.35);controls.update();}
function createScene(){
  const canvas=$('scene');renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0x102d33);renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.2;
  scene=new THREE.Scene();scene.fog=new THREE.Fog(0x102d33,3.4,7);camera=new THREE.PerspectiveCamera(36,1,.01,15);camera.up.set(0,0,1);controls=new OrbitControls(camera,canvas);controls.enableDamping=true;controls.dampingFactor=.08;controls.minDistance=.55;controls.maxDistance=4.7;controls.maxPolarAngle=Math.PI*.89;resetView();
  scene.add(new THREE.HemisphereLight(0xe6f7e8,0x284953,2.2));const sun=new THREE.DirectionalLight(0xffedd3,3);sun.position.set(.5,-1.4,2.6);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-1;sun.shadow.camera.right=1.5;sun.shadow.camera.top=1.5;sun.shadow.camera.bottom=-1;sun.shadow.bias=-.001;scene.add(sun);
  const fill=new THREE.DirectionalLight(0x8bc5cf,1);fill.position.set(-1,1,1);scene.add(fill);
  const grid=new THREE.GridHelper(3,30,0x35606a,0x1d4149);grid.rotation.x=Math.PI/2;grid.position.z=-.031;grid.material.transparent=true;grid.material.opacity=.6;scene.add(grid);
  staticGroup=new THREE.Group();armGroup=new THREE.Group();scene.add(staticGroup,armGroup);
  ball=new THREE.Mesh(new THREE.SphereGeometry(.0375,32,24),new THREE.MeshStandardMaterial({color:colors.ball,roughness:.45,metalness:.05}));ball.castShadow=true;scene.add(ball);
  const resize=()=>{const {width,height}=$('stage').getBoundingClientRect();renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();};new ResizeObserver(resize).observe($('stage'));resize();lastTime=performance.now();requestAnimationFrame(animate);
}
function dirtyScene(){if(busy||studyMode)return;try{sceneDirty=true;config=readConfig();$('configJson').value=JSON.stringify(config,null,2);buildEnvironment();result=null;$('replayButton').disabled=true;$('downloadRun').disabled=true;$('outcome').textContent='Scene changed';$('outcomeDetail').textContent='Run a new drop for this configuration.';$('contact').textContent='—';$('learning').textContent='Not started';$('learningDetail').textContent='Release position + height';$('historyPanel').hidden=true;phase('Ready');message('Preview changed. Run or Learn will position simulated arm 14 in Viam.');}catch(error){message(error.message,true);}}
function historyChart(history){
  const rows=Array.isArray(history)?history:[];if(!rows.length)return;$('historyPanel').hidden=false;
  const values=rows.map(row=>Number(row.reward ?? row.score ?? row.result?.reward ?? 0));let low=Math.min(...values),high=Math.max(...values);if(high===low)high=low+1;
  const points=values.map((v,i)=>[12+(i/(Math.max(1,values.length-1)))*576,57-(v-low)/(high-low)*44]);
  const svg=$('historyChart');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';const make=(tag,attrs)=>{const element=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([key,value])=>element.setAttribute(key,value));svg.append(element);return element;};
  make('line',{x1:10,y1:59,x2:590,y2:59,stroke:'#d8dfd6','stroke-width':1});make('polyline',{points:points.map(p=>p.join(',')).join(' '),fill:'none',stroke:'#aec1b6','stroke-width':1.5});
  points.forEach(([x,y],i)=>{const dot=make('circle',{cx:x,cy:y,r:3.2,fill:rows[i].retained||rows[i].result?.retained?'#157f82':'#ed6c50'});const title=document.createElementNS(ns,'title');title.textContent=`Trial ${i+1}: reward ${values[i].toFixed(3)}`;dot.append(title);});
}
function trainingResult(state){return state?.training?.best_action?state.training:(state?.training?.result||state?.training?.output||state?.saved_training);}
function showSavedTrial(data){
  result=data;resultConfig=clone(config);resultAction=clone(data.action||action());applyAction(resultAction);updateTargetPoint();showMetrics(data);
  if(data.frames?.length)placePhysics(data.frames[data.frames.length-1]);
}
function showTraining(data,replay=true){
  displayedTrainingId=data.session_id||data.config_fingerprint||'current';
  applyAction(data.best_action);updateTargetPoint();historyChart(data.history);$('learning').textContent=`${data.history?.length??data.budget??24} proposals`;
  const test=data.test;let detail='Best release loaded';if(test){const n=test.trials??test.results?.length;const wins=test.retained_count??test.results?.filter(x=>x.retained).length;if(n!==undefined&&wins!==undefined)detail=`Held-out simulation: ${wins} / ${n} captured`;}$('learningDetail').textContent=detail;
  if(data.best_result){if(replay)play(data.best_result);else{result=data.best_result;resultConfig=clone(config);resultAction=clone(data.best_action);showMetrics(result);}}
  message(`Policy saved${data.session_id?' · '+data.session_id:''}. Run drop to validate its native Viam motion.`);
}
async function runDrop(){
  if(busy||studyMode||(!offline&&!viamConnected))return;
  try{const next=readConfig(),runAction=action();setBusy(true,offline?'Calculating contact dynamics…':'Positioning arm 14 through Viam…');result=null;playing=false;
    if(!offline)await applyScene(next);else{config=next;buildEnvironment();}
    message(offline?'Calculating contact dynamics…':'Arm 10 is executing the simulated pick and drop in Viam…');
    const started=await request('/api/simulate',{config,action:runAction,seed:seed()});
    const finished=offline?started:await waitForTask(started,'Running Viam trial');
    const data=offline?finished:finished.result;
    if(!data)throw new Error('Viam finished without a trial result. Inspect its Control tab.');
    if(data.arm_reachable===false){showMetrics(data);throw new Error(data.arm_error||'Release is outside the simulated arm workspace.');}
    play(data);message(offline?'Playing local simulation.':'Viam trial saved. Replaying its recorded physics.');
  }catch(error){message(error.message,true);phase('Trial did not complete');}finally{setBusy(false);}
}
async function train(){
  if(busy||studyMode||(!offline&&!viamConnected))return;
  try{const next=readConfig();setBusy(true,'Positioning the board before training…');result=null;playing=false;
    if(!offline)await applyScene(next);else{config=next;buildEnvironment();}
    $('outcome').textContent='Search in progress';$('outcomeDetail').textContent='Viam module evaluates simulated release proposals.';$('contact').textContent='—';$('learning').textContent='Starting…';$('learningDetail').textContent='One fixed board pose';
    const started=await request('/api/train',{config,seed:seed(),budget:24});
    const finished=offline?started:await waitForTask(started,'Training in Viam');
    const data=offline?finished:(trainingResult(finished)||finished.result);
    if(!data?.best_action)throw new Error('Viam finished without a saved policy. Inspect its Control tab.');
    showTraining(data);
  }catch(error){message(error.message,true);$('learning').textContent='Search stopped';$('learningDetail').textContent='See status below controls';phase('Training did not complete');}finally{setBusy(false);}
}
async function connect(){
  if(busy)return;
  // Clear numerical evidence before changing its provenance mode, including
  // when reconnect fails. A cached numerical result must never export as Viam.
  if(studyMode)clearEvidence();
  studyMode=false;studyEnvelope=null;armGroup.visible=true;
  setStudyPresentation(false);
  setBusy(true,'Connecting to the Viam simulation…');
  try{const received=await request('/api/config');if(!received.connection)throw new Error('The dashboard backend is being upgraded. Reconnect after its Viam bridge restarts.');clearEvidence();config=received.config||received;initialConfig=clone(config);sceneDirty=false;syncInputs();buildEnvironment();renderNative(received);applyAction(received.default_action||received.best_action||preStudyAction);preStudyAction=null;updateTargetPoint();
    if(offline){const arm=await request('/api/arm-model');renderArm(arm.boxes);if(arm.ball_xyz_mm)setPosition(ball,arm.ball_xyz_mm);}
    phase(activeTask(received)?received.status:'Ready');message(offline?'Explicit offline mode: no Viam commands or cloud collection.':(localOnly?'Connected to local fake-only simulator. Cloud sync disabled.':'Connected to sircus-sim. Both arms are simulated.'));
    if(!received.data_collection&&!offline)$('dataStatus').textContent='Trial collection runs in the Viam module. Cloud upload status is not yet reported.';
    const latest=activeTask(received)?await waitForTask(received,'Viam task'):received;
    if(latest!==received&&latest.config){config=clone(latest.config);syncInputs();buildEnvironment();renderNative(latest);}
    const saved=trainingResult(latest);if(saved?.best_action){showTraining(saved,false);phase('Policy ready');}
    if(latest.result?.frames?.length){showSavedTrial(latest.result);if(!saved)phase('Last trial loaded');}
  }catch(error){viamConnected=false;updateConnection({backend:'viam',connected:false});message(error.message,true);phase('Viam connection unavailable');}
  finally{setBusy(false);$('dropButton').disabled=!config||(!offline&&!viamConnected);$('learnButton').disabled=!config||(!offline&&!viamConnected);}
}
async function init(){
  createScene();
  studyPanel=new StudyPanel({document,onPreview:showStudyPreview,onExit:connect,
    isBusy:()=>busy||pollingStatus,
    onPending:value=>setBusy(value),
    onInvalidate:()=>{if(studyMode){clearEvidence();message('Choose Predict & preview to load a new numerical result.');lockStudyControls();}}});
  for(const id of Object.keys(fields))$(id).addEventListener('change',dirtyScene);
  for(const id of ['dropU','dropV','dropGap'])$(id).addEventListener('change',()=>{updateTargetPoint();message('Release action updated.');});
  $('dropButton').addEventListener('click',runDrop);$('learnButton').addEventListener('click',train);$('replayButton').addEventListener('click',()=>result&&play(result,true));
  $('stopButton').addEventListener('click',()=>{playing=!playing;$('stopButton').textContent=playing?'Pause replay':'Resume replay';phase(playing?(studyMode?'Numerical replay':'Recorded replay'):'Replay paused',playing);});
  $('cancelButton').addEventListener('click',async()=>{if(studyMode||studyPanel?.pending)return;try{await request('/api/stop',{});message('Stop requested from Viam.');}catch(error){message(error.message,true);}});
  $('reconnectButton').addEventListener('click',connect);
  $('resetView').addEventListener('click',resetView);$('topView').addEventListener('click',()=>{camera.position.set(.7,0,2.8);controls.target.set(.7,0,0);controls.update();});
  document.querySelectorAll('.preset').forEach(button=>button.addEventListener('click',()=>{if(!config)return;const kind=button.dataset.preset;if(kind==='baseline'){config.board.center_mm=clone(initialConfig.board.center_mm);config.board.rpy_deg=clone(initialConfig.board.rpy_deg);}if(kind==='turn')config.board.rpy_deg[2]+=15;if(kind==='raise')config.board.center_mm[2]+=50;syncInputs();dirtyScene();}));
  document.querySelectorAll('.contact-preset').forEach(button=>button.addEventListener('click',()=>{$('dropU').value=button.dataset.u;$('dropV').value=button.dataset.v;updateTargetPoint();}));
  $('applyJson').addEventListener('click',async()=>{if(busy||studyMode)return;try{const next=JSON.parse($('configJson').value);if(!next.board?.center_mm||!next.board?.rpy_deg||!next.bin?.center_xy_mm||!next.table?.size_mm||!next.ball?.diameter_mm)throw new Error('Configuration is missing required scene fields.');setBusy(true,'Applying scene through Viam…');await applyScene(next);result=null;message(offline?'Offline configuration saved.':'Scene saved and board positioned in Viam.');phase('Ready');}catch(error){message(error.message,true);phase('Scene rejected');}finally{setBusy(false);}});
  $('downloadRun').addEventListener('click',()=>{if(!result)return;const record=studyMode?{...studyEnvelope,source:'scene-conditioned-numerical',machine_id:null}: {source:offline?'offline':'viam',machine_id:offline?null:'5db5a9df-2e56-4711-9d39-b42cc86e4d79',config:resultConfig,action:resultAction,result};const blob=new Blob([JSON.stringify(record,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),anchor=document.createElement('a');anchor.href=url;anchor.download=studyMode?'sircus-numerical-policy-preview.json':'sircus-simulation-drop.json';anchor.click();URL.revokeObjectURL(url);});
  await connect();
  polling=setInterval(async()=>{
    if(studyMode||offline||busy||playing||sceneDirty||!config||pollingStatus)return;
    pollingStatus=true;
    try{
      const state=await request('/api/status');
      if(studyMode)return;
      if(state.config&&sceneSignature(state.config)!==sceneSignature(config)){
        clearEvidence();config=clone(state.config);initialConfig=clone(config);syncInputs();buildEnvironment();
        phase('Scene updated');message('Scene changed in Viam. Earlier results remain archived with their original geometry.');
      }
      renderNative(state);
      if(!activeTask(state)){
        const saved=trainingResult(state),savedId=saved?.session_id||saved?.config_fingerprint||'current';
        if(saved?.best_action&&savedId!==displayedTrainingId){showTraining(saved,false);phase('Policy ready');}
        else if(displayedTrainingId&&!saved?.best_action)clearLearning();
      }
      if(activeTask(state)){
        setBusy(true,'A task is running in the Viam simulation…');
        try{
          const finished=await waitForTask(state,'Viam task');
          if(finished.config){config=clone(finished.config);syncInputs();buildEnvironment();renderNative(finished);}
          const saved=trainingResult(finished);
          if(saved?.best_action){showTraining(saved,false);phase('Policy ready');}
          else{phase('Viam task complete');message('The Viam task completed. You can run the next trial.');}
          if(finished.result?.frames?.length)showSavedTrial(finished.result);
        }
        catch(error){message(error.message,true);phase('Viam task stopped');}
        finally{setBusy(false);}
      }
    }catch(error){if(!studyMode){updateConnection({backend:'viam',connected:false});phase('Viam disconnected');}}
    finally{pollingStatus=false;studyPanel?.render();}
  },2500);
}
init();
