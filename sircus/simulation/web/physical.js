export function mountPhysicalPanel({document=globalThis.document,window=globalThis.window,fetchImpl=(...args)=>globalThis.fetch(...args),manageTabs=true,onStatus=()=>{},setIntervalImpl=(fn,ms)=>globalThis.setInterval(fn,ms),clearIntervalImpl=id=>globalThis.clearInterval(id)}={}) {
const $ = id => document.getElementById(id);
const panel = $('physicalPanel');
const controls = [...panel.querySelectorAll('[data-physical-action]')];
let selected = false;
let destroyed = false;
let commandEpoch=0;
const poseDirty=new Set();
const poseKeys=['x','y','z','o_x','o_y','o_z','theta'];
const poseSuffixes=['X','Y','Z','OX','OY','OZ','Theta'];
let status = null;
let pending = 0;
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

function updateControls() {
  controls.forEach(button => {
    const action = button.dataset.physicalAction;
    button.disabled = !allowed(action) || (action !== 'stop' && pending > 0);
  });
  $('physicalCycle').disabled = !allowed('load_drop_cycle') || pending > 0 || !$('physicalCycleConfirm').checked;
  $('physicalCycleConfirm').disabled = !!status?.busy;
  $('physicalPractice').disabled = !!status?.busy;
  const canLabel = allowed('save_trial') && pending === 0;
  $('physicalOutcomes').disabled = !canLabel;
  $('physicalNotes').disabled = !canLabel;
  const outcome = panel.querySelector('input[name="physicalOutcome"]:checked')?.value;
  $('physicalMissDirection').disabled = !canLabel || outcome !== 'missed';
  $('physicalSave').disabled = !canLabel || !outcome;
  $('physicalSaveCalibration').disabled = !allowed('configure') || pending > 0 || !$('physicalCalibrationConfirm').checked;
  $('physicalHeight').disabled = pending > 0 || !!status?.busy || !!(status?.trial && !status.trial.saved);
  if($('physicalSavePoses'))$('physicalSavePoses').disabled=!allowed('configure_poses')||pending>0||!$('physicalPoseConfirm').checked;
}

function render(value) {
  status = value;
  lastActionToken = value.action_token || lastActionToken;
  stopWasAvailable = allowed('stop');
  const trial = value.trial || {};
  const recording = value.recording || {};
  const calibration = value.calibration || {};
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
  if($('physicalCameraSource'))$('physicalCameraSource').textContent=mode==='hardware'?'Camera readback · arm 10':'Dry-run camera · no hardware';
  $('physicalCamera').alt=mode==='hardware'?'Latest sfhack10 camera readback':'Mock camera preview; no physical camera connected';
  $('physicalTrialId').textContent = nextTrialId || 'No active trial';
  $('physicalArmState').textContent = value.phase || 'Unknown';
  const cycleLabels = {cycle_to_loading:'Moving to loading pose', cycle_opening:'Opening at loading pose',
    cycle_loading_countdown:`Load now · closes in ${value.loading_seconds_remaining ?? '…'} seconds`,
    cycle_closing:'Closing gripper', cycle_to_drop:'Moving to taught drop pose',
    cycle_release_checks:'Holding drop pose steady for 2 seconds; checking recording', post_release_recording:'Recording after release',
    awaiting_outcome: trial.status === 'completed' ? 'Cycle finished. Keep unlabeled or label and save.' : 'Trial stopped. Keep unlabeled or label and save before another cycle.'};
  if (value.fault || value.error) $('physicalCycleConfirm').checked = false;
  $('physicalCycleStatus').textContent = cycleLabels[value.phase] || (value.fixed_cycle_missing?.length
    ? value.fixed_cycle_missing.join(' · ') : trial.saved && trial.label_status === 'unlabeled' ? 'Kept locally, unlabeled. Ready for your next button press.' : 'Taught poses available. Confirm setup, then start one cycle.');
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
}

async function refreshStatus() {
  if (destroyed || !selected || polling) return;
  polling = true;
  const epoch=commandEpoch;
  try {
    const response = await fetchImpl('/api/physical/status', {cache:'no-store'});
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || 'Physical-trial service unavailable.');
    if(epoch===commandEpoch&&!destroyed)render(value);
  } catch (error) {
    if(epoch!==commandEpoch||destroyed)return;
    status = null;
    updateControls();
    $('physicalMode').textContent = 'STATUS UNAVAILABLE';
    showMessage(`${error.message} No command will be retried automatically.`, true);
  } finally { polling = false; }
}

async function refreshCamera() {
  if (destroyed || !selected || cameraPending || !status?.connected) return;
  cameraPending = true;
  try {
    const response = await fetchImpl('/api/physical/camera', {cache:'no-store'});
    if (!response.ok) throw new Error('Camera frame unavailable');
    if (!(response.headers.get('content-type') || '').startsWith('image/')) throw new Error('No image returned');
    const next = URL.createObjectURL(await response.blob());
    const previous = cameraURL;
    cameraURL = next;
    $('physicalCamera').src = next;
    $('physicalCamera').hidden = false;
    $('physicalCameraPlaceholder').hidden = true;
    $('physicalCameraTime').textContent = `Received ${new Date().toLocaleTimeString()}`;
    if (previous) URL.revokeObjectURL(previous);
  } catch {
    $('physicalCameraTime').textContent = 'Preview unavailable / last image may be stale';
    if (!cameraURL) $('physicalCameraPlaceholder').textContent = 'Camera preview unavailable. Check the recording details.';
  } finally { cameraPending = false; }
}

async function command(action,extra={}) {
  if (destroyed || !allowed(action) || (action !== 'stop' && pending > 0)) return false;
  let body = {action,...extra};
  try {
    if (action === 'load_drop_cycle') { body.fixed_setup_confirmed = $('physicalCycleConfirm').checked; body.practice_mode = $('physicalPractice').checked; }
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
  pending += 1;
  commandError = '';
  updateControls();
  showMessage(action === 'stop' ? 'Sending Stop to arm 10…' : 'Command submitted. Waiting for robot readback…');
  try {
    const response = await fetchImpl('/api/physical/action', {
      method:'POST', headers:{'Content-Type':'application/json', ...(lastActionToken ? {'X-Sircus-Token':lastActionToken} : {})}, body:JSON.stringify(body)
    });
    const value = await response.json();
    if (!response.ok || value.error) throw new Error(value.error || `Command rejected (${response.status})`);
    if(epoch!==commandEpoch||destroyed)return false;
    if(action==='configure_poses'){poseDirty.clear();$('physicalPoseConfirm').checked=false;}
    if(action==='teach_loading')poseDirty.delete('Loading');
    if(action==='teach_drop')poseDirty.delete('Drop');
    if (value.status) render(value.status);
    else if (value.allowed_actions) render(value);
    if (action === 'configure') $('physicalCalibrationConfirm').checked = false;
    if (action === 'stop' || (action === 'load_drop_cycle' && !body.practice_mode)) $('physicalCycleConfirm').checked = false;
    return true;
  } catch (error) {
    if(epoch!==commandEpoch||destroyed)return false;
    commandError = `${error.message} The command was not retried. Check robot state before continuing.`;
    showMessage(commandError, true);
    return false;
  } finally {
    pending -= 1;
    updateControls();
    await refreshStatus();
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
$('physicalCycleConfirm').addEventListener('change', updateControls);
$('physicalPoseConfirm')?.addEventListener('change',updateControls);
for(const prefix of ['Loading','Drop'])for(const suffix of poseSuffixes)$('physical'+prefix+suffix)?.addEventListener('input',()=>poseDirty.add(prefix));
const timers=[setIntervalImpl(refreshStatus,1000),setIntervalImpl(refreshCamera,1500)];
function destroy(){if(destroyed)return;destroyed=true;timers.forEach(clearIntervalImpl);if(cameraURL)URL.revokeObjectURL(cameraURL);}
window.addEventListener('pagehide',destroy,{once:true});
if(manageTabs&&window.location.hash==='#physical')selectView(true);
return {selectView,command,refreshStatus,allowed,destroy,get status(){return status;},get busy(){return pending>0||!!status?.busy;}};
}
if(typeof document!=='undefined'&&document.getElementById('physicalPanel')&&!document.documentElement.hasAttribute('data-show-shell'))mountPhysicalPanel();
