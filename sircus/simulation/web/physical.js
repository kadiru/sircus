export function mountPhysicalPanel({document=globalThis.document,window=globalThis.window,fetchImpl=(...args)=>globalThis.fetch(...args),manageTabs=true,onStatus=()=>{},setIntervalImpl=(fn,ms)=>globalThis.setInterval(fn,ms),clearIntervalImpl=id=>globalThis.clearInterval(id),setTimeoutImpl=(fn,ms)=>globalThis.setTimeout(fn,ms),clearTimeoutImpl=id=>globalThis.clearTimeout(id),requestTimeouts={status:5000,action:35000,stop:45000,camera:12000}}={}) {
const $ = id => document.getElementById(id);
const panel = $('physicalPanel');
const controls = [...panel.querySelectorAll('[data-physical-action]')];
let selected = false;
let destroyed = false;
let commandEpoch=0;
const poseDirty=new Set();
let poseCommit=null;
const poseKeys=['x','y','z','o_x','o_y','o_z','theta'];
const poseSuffixes=['X','Y','Z','OX','OY','OZ','Theta'];
let status = null;
let pending = 0;
let pendingAction = null;
let commandUncertain = false;
let lastKnownBusy = false;
let lastActionEpoch = -1;
let setupSignature = null;
const requests = new Set();
let polling = false;
let cameraPending = false;
let cameraURL = null;
let currentTrialId = null;
let calibrationInitialized = false;
let commandError = '';
let lastActionToken = null;
let stopWasAvailable = false;

function showMessage(text, error = false) {
  $('physicalMessage').textContent = text;
  $('physicalMessage').classList.toggle('error', error);
}

function allowed(action) {
  if (action !== 'stop' && commandUncertain) return false;
  if (action === 'stop' && !status) return stopWasAvailable;
  const values = status?.allowed_actions || [];
  return Array.isArray(values) ? values.includes(action) : values[action] === true;
}

function pretty(value) { return JSON.stringify(value, null, 2); }
function present(value) { return value !== null && value !== undefined; }
function finiteInput(id) {
  const raw = $(id).value.trim();
  if (!raw || !Number.isFinite(Number(raw))) throw new Error('Enter a finite number for each required measurement.');
  return Number(raw);
}

async function request(path, options = {}, timeoutMs = requestTimeouts.status, consume = async response => {
  const value = await response.json();
  if (!response.ok) { const error = new Error(value.error || `Request rejected (${response.status})`); error.httpStatus = response.status; throw error; }
  return value;
}) {
  const controller = new AbortController(); requests.add(controller);
  let timer;
  const deadline = new Promise((_, reject) => { timer = setTimeoutImpl(() => { controller.abort(); const error = new Error('Request timed out'); error.transportUncertain = true; reject(error); }, timeoutMs); });
  try { return await Promise.race([Promise.resolve().then(() => fetchImpl(path, {...options, signal:controller.signal})).then(consume), deadline]); }
  catch (error) { if (!error.httpStatus) error.transportUncertain = true; throw error; }
  finally { clearTimeoutImpl(timer); requests.delete(controller); }
}

function cycleReadiness() {
  if (pendingAction === 'stop') return 'Stop requested. Waiting for confirmation; no tool will open automatically.';
  if (commandUncertain) return 'Command outcome is unconfirmed. Press Stop, verify robot state, then reconfirm the fixed setup.';
  if (!status) return 'Status unavailable. Cycle state is unknown; no automatic retry. Stop remains available after a connection was established.';
  if (status.fault || status.error) return 'Cycle blocked: '+(status.error || 'controller fault; verify Stop and robot state');
  if (!status.connected) return 'Fresh robot readback is required. Cycle controls are locked.';
  const phase=status.phase, trial=status.trial||{};
  const remaining=status.loading_seconds_remaining;
  const steps={cycle_to_loading:'Moving to the saved loading pose.',cycle_opening:'Opening at the loading pose.',
    cycle_loading_countdown:Number.isFinite(remaining)?(remaining>0?`Load now · automatic close in ${remaining} s (latest readback).`:'Automatic close is imminent. Keep fingers clear.'):'Loading countdown unavailable. Keep clear; use Stop if needed.',
    cycle_closing:'Closing gripper; waiting for the held-object check.',cycle_to_drop:'Carrying to the saved drop pose.',
    cycle_release_checks:'Settling at the drop pose for 2 continuous seconds; recording must stay ready.',
    post_release_recording:'Release command returned. Finishing the trial recording.',stopping:'Stopping arm 10; awaiting stationary readback.'};
  if(status.busy||pending) return steps[phase] || (pending ? 'Request submitted; acceptance is not completion. Waiting for controller state.' : 'Controller action in progress.');
  if(poseDirty.size)return 'Unsaved TCP target edits. Save or discard them before starting a cycle.';
  if(trial.id&&!trial.saved)return trial.status==='completed'?'Routine complete. Keep the record unlabeled or label and save before another cycle.':'Trial stopped or incomplete. Keep the record before another cycle; inspect any fault.';
  const blockers=status.fixed_cycle_blockers||status.fixed_cycle_missing||[];
  if(blockers.length)return 'Cycle blocked: '+blockers.join(' · ');
  if(!allowed('load_drop_cycle'))return 'Cycle is not ready. Check fresh stationary state, fixed board and saved poses.';
  const saved=trial.saved?(trial.status==='completed'?'Last routine retained locally; bucket outcome is '+(trial.outcome||'unknown')+'. ':'Last incomplete trial retained. '):'';
  return saved+($('physicalCycleConfirm').checked?'Ready for one operator-started cycle.':'Recheck the fixed setup and confirm before starting.');
}

function updateControls() {
  controls.forEach(button => {
    const action = button.dataset.physicalAction;
    button.disabled = !allowed(action) || (action !== 'stop' && (pending > 0 || (poseDirty.size>0&&['load_drop_cycle','go_loading','go_drop','drop'].includes(action))));
  });
  $('physicalCycle').disabled = !allowed('load_drop_cycle') || pending > 0 || poseDirty.size>0 || !$('physicalCycleConfirm').checked;
  $('physicalCycleConfirm').disabled = pending>0 || !!status?.busy || !status?.connected || !!status?.fault || !!status?.error || commandUncertain || poseDirty.size>0;
  $('physicalPractice').disabled = pending>0 || !!status?.busy;
  const canLabel = allowed('save_trial') && pending === 0;
  $('physicalOutcomes').disabled = !canLabel;
  $('physicalNotes').disabled = !canLabel;
  const outcome = panel.querySelector('input[name="physicalOutcome"]:checked')?.value;
  $('physicalMissDirection').disabled = !canLabel || outcome !== 'missed';
  $('physicalSave').disabled = !canLabel || !outcome;
  $('physicalSaveCalibration').disabled = !allowed('configure') || pending > 0 || !$('physicalCalibrationConfirm').checked;
  $('physicalHeight').disabled = pending > 0 || !!status?.busy || !!(status?.trial && !status.trial.saved);
  $('physicalCycleStatus').textContent=cycleReadiness();
  $('physicalCycleStatus').classList.toggle('ready',!$('physicalCycle').disabled);
  if($('physicalDiscardPoses'))$('physicalDiscardPoses').disabled=!poseDirty.size||pending>0||!!status?.busy;
  if($('physicalSavePoses'))$('physicalSavePoses').disabled=!allowed('configure_poses')||pending>0||!$('physicalPoseConfirm').checked;
}

function render(value) {
  if(!value||typeof value!=='object'||!value.allowed_actions)throw new Error('Invalid controller status');
  const newSession=value.action_token&&lastActionToken&&value.action_token!==lastActionToken;
  if(newSession){lastActionEpoch=-1;$('physicalCycleConfirm').checked=false;}
  if(Number.isSafeInteger(value.action_epoch)&&value.action_epoch<lastActionEpoch)return false;
  if(Number.isSafeInteger(value.action_epoch))lastActionEpoch=value.action_epoch;
  status = value;lastKnownBusy=!!value.busy;
  lastActionToken = value.action_token || lastActionToken;
  stopWasAvailable = allowed('stop');
  const trial = value.trial || {};
  const recording = value.recording || {};
  const calibration = value.calibration || {};
  if(poseCommit&&!value.busy){
    if(!value.fault&&!value.error&&value.connected){
      for(const [prefix,key]of [['Loading','loading_pose'],['Drop','drop_pose']]){
        const submitted=poseCommit[key],saved=calibration[key];
        const matches=poseKeys.every((name,i)=>saved?.[name]===submitted[name]&&$('physical'+prefix+poseSuffixes[i]).value.trim()!==''&&Number($('physical'+prefix+poseSuffixes[i]).value)===submitted[name]);
        if(matches)poseDirty.delete(prefix);
      }
    }
    poseCommit=null;
  }
  const robot = value.robot || value.robot_state || {};
  const nextTrialId = trial.id || trial.trial_id || value.trial_id || null;
  if (nextTrialId !== currentTrialId) {
    panel.querySelectorAll('input[name="physicalOutcome"]').forEach(input => { input.checked = false; });
    $('physicalNotes').value = '';
    $('physicalMissDirection').value = '';
    currentTrialId = nextTrialId;
  }
  const mode = value.mode || 'unknown';
  $('physicalMode').textContent = `${mode === 'hardware' ? 'PHYSICAL HARDWARE' : mode === 'dry_run' || mode === 'mock' ? 'DRY RUN · NO HARDWARE' : mode.toUpperCase()} · ${value.connected ? 'CONNECTED' : 'DISCONNECTED'}`;
  $('physicalMode').classList.toggle('hardware', mode === 'hardware');
  if(!value.connected)$('physicalCameraTime').textContent='Readback unavailable · last image may be stale';
  if($('physicalCameraSource'))$('physicalCameraSource').textContent=mode==='hardware'?'Camera readback · arm 10':'Dry-run camera · no hardware';
  $('physicalCamera').alt=mode==='hardware'?'Latest sfhack10 camera readback':'Mock camera preview; no physical camera connected';
  $('physicalTrialId').textContent = nextTrialId || 'No active trial';
  $('physicalArmState').textContent = value.phase || 'Unknown';
  if (value.fault || value.error || !value.connected) $('physicalCycleConfirm').checked = false;
  const signature=JSON.stringify([calibration.loading_pose,calibration.drop_pose,calibration.board_joints,calibration.board_pose,calibration.setup_revision]);
  if(setupSignature!==null&&signature!==setupSignature)$('physicalCycleConfirm').checked=false;setupSignature=signature;
  $('physicalRecording').textContent = recording.active ? (recording.ready ? 'Recording · ready' : 'Recording · preparing') : recording.ready ? 'Ready' : 'Not ready';
  const height = trial.height_mm ?? trial.requested_height_mm ?? value.requested_height_mm;
  $('physicalRequestedHeight').textContent = present(height) ? `${height} mm` : trial.behavior === 'fixed_pose_cycle' ? 'Taught pose · height unmeasured' : '—';
  $('physicalState').textContent = pretty(robot);
  $('physicalTrialDetails').textContent = Object.keys(trial).length ? pretty(trial) : 'No active trial.';
  $('physicalStateTime').textContent = value.observed_at || value.updated_at || robot.captured_at || robot.timestamp || 'Readbacks shown below';
  $('physicalRecordingDetails').textContent = pretty({recording, gripper_command_state:value.gripper_command_state, gripper_state_provenance:value.gripper_state_provenance, integrations: value.integrations || value.capabilities || {}});
  const missing = value.calibration_missing || [];
  $('physicalCalibrationMissing').textContent = missing.length ? `Missing: ${missing.join(' · ')}` : 'Saved calibration available. Review details before a physical trial.';
  $('physicalCalibrationMissing').classList.toggle('ready', missing.length === 0);
  $('physicalCalibrationDetails').textContent = pretty(calibration);
  for(const [prefix,key]of [['Loading','loading_pose'],['Drop','drop_pose']])if(!poseDirty.has(prefix))poseKeys.forEach((name,index)=>{const node=$('physical'+prefix+poseSuffixes[index]);if(node&&document.activeElement!==node)node.value=present(calibration[key]?.[name])?String(calibration[key][name]):'';});
  const limits = calibration.height_limits_mm || {};
  const minimum = calibration.min_height_mm ?? limits.min;
  const maximum = calibration.max_height_mm ?? limits.max;
  if (present(minimum) && present(maximum)) {
    $('physicalHeight').min = minimum;
    $('physicalHeight').max = maximum;
    $('physicalHeightLimits').textContent = `Allowed ${minimum}–${maximum} mm above reference Z ${calibration.reference_z_mm ?? 'unknown'} mm. Only Z changes.`;
  } else {
    $('physicalHeight').removeAttribute('min');
    $('physicalHeight').removeAttribute('max');
    $('physicalHeightLimits').textContent = 'Height limits have not been calibrated.';
  }
  if (!calibrationInitialized) {
    [['physicalReferenceZ', calibration.reference_z_mm], ['physicalMinHeight', minimum], ['physicalMaxHeight', maximum]].forEach(([id, val]) => { if (present(val)) $(id).value = val; });
    if (calibration.setup_revision) $('physicalSetupRevision').value = calibration.setup_revision;
    calibrationInitialized = true;
  }
  const blockers = value.blockers || value.drop_blockers || [];
  $('physicalReadiness').textContent = allowed('drop') ? 'Ready: intended pose reached, arm settled and recording ready.' : blockers.length ? blockers.join(' · ') : 'Drop is locked until pose, settling and recording checks pass.';
  $('physicalReadiness').classList.toggle('ready', allowed('drop'));
  if (commandError) showMessage(commandError, true);
  else if (value.error) showMessage(value.error, true);
  else showMessage(value.message || (value.connected ? 'Operator buttons send commands to the selected physical-trial backend.' : 'Physical-trial backend is not connected. Commands remain locked.'));
  updateControls();
  onStatus(value);
  return true;
}

async function refreshStatus() {
  if (destroyed || !selected || polling) return;
  polling = true;
  const epoch=commandEpoch;
  try {
    const value = await request('/api/physical/status', {cache:'no-store'},requestTimeouts.status);
    if(epoch===commandEpoch&&!destroyed)render(value);
  } catch (error) {
    if(epoch!==commandEpoch||destroyed)return;
    status = null;$('physicalCycleConfirm').checked=false;
    updateControls();
    $('physicalMode').textContent = 'STATUS UNAVAILABLE';
    $('physicalCameraTime').textContent='Status unavailable · last image may be stale';
    showMessage(`${error.message} No command will be retried automatically.`, true);
    onStatus({connected:false,busy:pending>0||lastKnownBusy||commandUncertain,phase:'status_unavailable'});
  } finally { polling = false; }
}

async function refreshCamera() {
  if (destroyed || !selected || cameraPending || !status?.connected) return;
  cameraPending = true;
  try {
    const blob = await request('/api/physical/camera',{cache:'no-store'},requestTimeouts.camera,async response=>{
      if(!response.ok||!(response.headers.get('content-type')||'').startsWith('image/'))throw new Error('Camera frame unavailable');
      return response.blob();
    });
    if(destroyed||!selected||!status?.connected)return;
    const next = URL.createObjectURL(blob);
    const previous = cameraURL;
    cameraURL = next;
    $('physicalCamera').src = next;
    $('physicalCamera').hidden = false;
    $('physicalCameraPlaceholder').hidden = true;
    $('physicalCameraTime').textContent = `Received ${new Date().toLocaleTimeString()}`;
    if (previous) URL.revokeObjectURL(previous);
  } catch {
    if(destroyed)return;
    $('physicalCameraTime').textContent = 'Preview unavailable / last image may be stale';
    if (!cameraURL) $('physicalCameraPlaceholder').textContent = 'Camera preview unavailable. Check the recording details.';
  } finally { cameraPending = false; }
}

async function command(action,extra={}) {
  if (destroyed || !allowed(action) || (action !== 'stop' && pending > 0) || (action==='stop'&&pendingAction==='stop') || (poseDirty.size>0&&['load_drop_cycle','go_loading','go_drop','drop'].includes(action))) return false;
  let body = {action,...extra,...(action!=='stop'&&Number.isSafeInteger(status?.action_epoch)?{expected_action_epoch:status.action_epoch}:{})};
  try {
    if (action === 'load_drop_cycle') { if(!$('physicalCycleConfirm').checked)throw new Error('Confirm the fixed setup before starting a cycle.'); body.fixed_setup_confirmed = $('physicalCycleConfirm').checked; body.practice_mode = $('physicalPractice').checked; }
    if (action === 'keep_unlabeled') body.trial_id = currentTrialId;
    if (action === 'go_drop') body.height_mm = finiteInput('physicalHeight');
    if (action === 'save_trial') {
      body = {...body, trial_id: currentTrialId, outcome: panel.querySelector('input[name="physicalOutcome"]:checked')?.value,
        miss_direction: $('physicalMissDirection').disabled ? '' : $('physicalMissDirection').value, notes: $('physicalNotes').value};
      if (!body.outcome) throw new Error('Choose Captured, Missed or Unsure.');
    }
    if(action==='configure_poses'){body.loading_pose=Object.fromEntries(poseKeys.map((key,i)=>[key,finiteInput('physicalLoading'+poseSuffixes[i])]));body.drop_pose=Object.fromEntries(poseKeys.map((key,i)=>[key,finiteInput('physicalDrop'+poseSuffixes[i])]));body.confirmed=$('physicalPoseConfirm').checked;}
    if (action === 'configure') {
      body = {...body, reference_z_mm: finiteInput('physicalReferenceZ'), min_height_mm: finiteInput('physicalMinHeight'),
        max_height_mm: finiteInput('physicalMaxHeight'), setup_revision: $('physicalSetupRevision').value.trim(), confirmed: $('physicalCalibrationConfirm').checked};
    }
  } catch (error) { showMessage(error.message, true); return false; }
  const epoch=++commandEpoch;
  pending = 1;pendingAction=action;
  if(action==='stop'){poseCommit=null;$('physicalCycleConfirm').checked=false;}
  if(action==='configure_poses'){poseDirty.add('Loading');poseDirty.add('Drop');}
  commandError = '';
  updateControls();
  showMessage(action === 'stop' ? 'Sending Stop to arm 10…' : 'Command submitted. Waiting for robot readback…');
  try {
    const value = await request('/api/physical/action', {
      method:'POST', headers:{'Content-Type':'application/json', ...(lastActionToken ? {'X-Sircus-Token':lastActionToken} : {})}, body:JSON.stringify(body)
    },action==='stop'?(requestTimeouts.stop??45000):requestTimeouts.action);
    if(value.error||value.accepted===false)throw new Error(value.error||'Command was not accepted');
    if(epoch!==commandEpoch||destroyed)return false;
    const returned=value.status||value;
    if(!returned.allowed_actions){const error=new Error('Controller response did not confirm its state');error.transportUncertain=true;throw error;}
    if(action==='stop'&&returned.connected===true&&returned.busy===false&&!returned.fault&&!returned.error)commandUncertain=false;
    if(action==='configure_poses'){poseCommit={loading_pose:body.loading_pose,drop_pose:body.drop_pose};$('physicalPoseConfirm').checked=false;}
    if(action==='teach_loading')poseDirty.delete('Loading');
    if(action==='teach_drop')poseDirty.delete('Drop');
    if (value.status) render(value.status);
    else if (value.allowed_actions) render(value);
    if (action === 'configure') $('physicalCalibrationConfirm').checked = false;
    if (action === 'stop' || (action === 'load_drop_cycle' && !body.practice_mode)) $('physicalCycleConfirm').checked = false;
    return true;
  } catch (error) {
    if(epoch!==commandEpoch||destroyed)return false;
    $('physicalCycleConfirm').checked=false;
    if(error.transportUncertain)commandUncertain=true;
    commandError = error.transportUncertain?`${error.message}. Command acceptance is unknown; it was not retried. Press Stop and verify state before continuing.`:`${error.message} The command was not retried. Check robot state and reconfirm setup before continuing.`;
    showMessage(commandError, true);
    return false;
  } finally {
    if(epoch===commandEpoch){pending=0;pendingAction=null;updateControls();await refreshStatus();}
  }
}

function selectView(physical) {
  selected = physical;
  if(manageTabs){$('simulationPanel').hidden = physical;
  panel.hidden = !physical;
  $('connectionBadge').hidden = physical;
  $('showSimulation').classList.toggle('selected', !physical);
  $('showPhysical').classList.toggle('selected', physical);
  $('showSimulation').setAttribute('aria-pressed', String(!physical));
  $('showPhysical').setAttribute('aria-pressed', String(physical));}
  panel.hidden=!physical;
  if (physical) { refreshStatus().then(refreshCamera); }
  else window.dispatchEvent(new Event('resize'));
}

if(manageTabs){$('showSimulation').addEventListener('click', () => selectView(false));
$('showPhysical').addEventListener('click', () => selectView(true));}
controls.forEach(button => button.addEventListener('click', () => command(button.dataset.physicalAction)));
panel.querySelectorAll('input[name="physicalOutcome"]').forEach(input => input.addEventListener('change', updateControls));
$('physicalCalibrationConfirm').addEventListener('change', updateControls);
$('physicalCycleConfirm').addEventListener('change',()=>{if($('physicalCycleConfirm').checked&&!commandUncertain)commandError='';updateControls();if(status&&!commandError)render(status);});
$('physicalPoseConfirm')?.addEventListener('change',updateControls);
for(const prefix of ['Loading','Drop'])for(const suffix of poseSuffixes)$('physical'+prefix+suffix)?.addEventListener('input',()=>{poseDirty.add(prefix);$('physicalCycleConfirm').checked=false;updateControls();});
$('physicalDiscardPoses')?.addEventListener('click',()=>{if(pending||status?.busy)return;poseDirty.clear();$('physicalPoseConfirm').checked=false;$('physicalCycleConfirm').checked=false;if(status)render(status);});
const timers=[setIntervalImpl(refreshStatus,1000),setIntervalImpl(refreshCamera,1500)];
function destroy(){if(destroyed)return;destroyed=true;commandEpoch++;for(const controller of requests)controller.abort();timers.forEach(clearIntervalImpl);if(cameraURL)URL.revokeObjectURL(cameraURL);}
window.addEventListener('pagehide',destroy,{once:true});
if(manageTabs&&window.location.hash==='#physical')selectView(true);
return {selectView,command,refreshStatus,allowed,destroy,get status(){return status;},get busy(){return pending>0||!!status?.busy||commandUncertain||(!status&&lastKnownBusy);}};
}
if(typeof document!=='undefined'&&document.getElementById('physicalPanel')&&!document.documentElement.hasAttribute('data-show-shell'))mountPhysicalPanel();
