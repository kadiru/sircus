# Project state

## Revised Round 2 completed — October 30, 2026, 13:12–13:13 PDT

Following explicit reload/clearance authorization, all four pickup/drop cycles
completed using the revised Round2 pair (revision2): Gray, Purple, Pink, Green.
The board already matched its revised preset and was not commanded to move.
The run used the saved revised drop target, constrained pickup/lift, two-second
release settling, enabled FX and disabled cameras. No retries or recovery were
needed. All four local records are completed/unlabeled; capture outcomes and
acoustic success are not inferred from command completion. Audio controls
reported no error. Final readback: both arms stationary, controller idle without
fault/error, dropper open, FX muted, and no repeat queued. No code, service or
Viam configuration change occurred. Calibration and raw evidence remain local.


## Round 2 preset revised — October 30, 2026, 13:11 PDT

The operator supplied a new paired dropper/board pose and manually positioned
both arms. Screenshots matched fresh stationary world-frame readback. Read-only
`teach_drop` saved the revised drop target and board reference; the private Round2
preset is revision2. Previous Round2 pair was archived; Round1, loading and all
four pickup points were preserved. No actuator/camera command or service restart
was issued. This revised pair has not completed an automated round or validated
transition; the successful Round2 result below belongs to revision1. Coordinates
and joint calibration remain host-local. Next run requires a separate operator
request and fresh readiness checks.


## Round 2 completed — October 30, 2026, 13:05–13:06 PDT

After explicit operator reload/clearance confirmation, native motion moved only
the board arm to its taught Position 2. Pose/joint checks passed; vacuum holding
remained true during the transition and the dropper stayed stationary. The paired
Round2 drop target was activated through the controller's supported pose route.
Gray → Purple → Pink → Green then all completed pickup, constrained vertical lift,
transfer, two-second settle and release, without recovery/retries. FX was enabled
at maximum dashboard volume; no audio control error was reported. Acoustic output
and bucket capture were not independently scored; all four records are unlabeled.

Final readback: both arms stationary, dashboard idle/connected, no fault or error,
dropper open, board at Position 2, camera use disabled and FX muted. No next round
starts automatically. No service restart or Viam configuration change occurred.
Private trial/state logs remain local; software is unchanged from `0f44107`.


## Held Green recovery completed — October 30, 2026

Latest branch: `codex/macbook-held-ball-recovery`, implementation `0f44107`.
The explicit quiescent-recovery path passed Stop RPCs, two server-operation/arm
samples and pending-request checks. The original held-Green trial then resumed
from its pickup approach: no repeated pickup, no close command, one saved-pose
transfer, two-second settle and one release. The original trial ID and recovery
evidence were retained and saved unlabeled. Round 1 is complete; bucket outcome
remains unknown. Both arms reported stationary afterward, fault cleared, FX muted.
Round 2 has not started and awaits operator reload/readiness.

The dashboard was explicitly relaunched with `--disable-cameras`: no camera
lookup, preview, recording or capture-control attachment from this process.
Viam camera/frame resources were not removed or powered down, and unrelated
clients/services were not reconfigured. Board arm and suction were not commanded.
Validation: 146 physical/controller tests, 11 startup tests, 40 UI tests, and
repository checks passed. Runtime evidence/calibration remain local, not in Git.


## Latest handoff — October 30, 2026

Read [the cross-computer handoff](AGENT_HANDOFF.md) before acting on historical
sections below. Latest integration branch: `codex/macbook-no-camera-demo`; main
does not yet include the draft demo stack. Camera-off Round1 completed Gray,
Purple and Pink, then stopped during Green's lift following lost robot readback.
The operator and a read-only sensor check confirm Green remains held. Both arms
were stationary and no operations active at 19:49:58 UTC, but the controller's
planner-stop verification fault remains. Round2 has not started; FX is muted.
No automatic retry, gripper release or runtime reset is authorized by this status
record. See the handoff for tested code, branch/PR map and independent agent tasks.


## Authorized four-station physical run — October 30, 2026, 19:06–19:07 UTC

After operator clearance confirmation, the four approved host-local pickup poses
were imported and verified, and only the idle physical dashboard was restarted.
One supervised sequence completed Gray → Purple → Pink → Green. Each trial
passed driver grip, requested constrained vertical descent/lift, recording,
saved-drop settling and release checks, and was saved locally as completed with
unknown outcome. Native planning accepted this run; no bucket capture rate,
continuous path-clearance measurement or audio listening result is inferred.
The fixed drop target, original loading target and board joint reference were
preserved. Both arms read stationary afterward; arm14 received no motion command.

Following the separate operator request, one `vacuum-board.Grab` was sent.
It returned true. `IsHoldingSomething` was initially false, then a later read
returned true. No retry or release was issued; suction is left enabled. Both arms
read stationary afterward. Sensor acknowledgment does not measure holding force.
Host poses, private logs and RGB-D/state trial data remain excluded from Git.

## Per-color taped pickup stations — October 30, 2026

- `codex/macbook-fixed-pickup` builds on audio-loudness checkpoint `f8c0850`.
  Manual operation adds independent read-only teaching for each color, with a
  derived world-Z+100mm approach and the existing saved drop pose unchanged.
- A separate pickup cycle plans to approach, records, opens, requests constrained
  vertical descent, checks grip, lifts vertically, and uses the existing two-second
  drop settling/release path. Constraint rejection has no unconstrained fallback.
- Host calibration, station coordinates and observations remain excluded from
  Git. Physical route and neighboring-ball clearance are not established by
  mock tests. The operator is teaching Gray first; no new pickup pose or physical
  pickup has yet been recorded at this checkpoint.
- [Station workflow and limits](TAPED_PICKUP_STATIONS.md) distinguish requested
  endpoint constraints, measured robot readback and operator-confirmed placement.
- After explicit approval, only the idle physical dashboard was restarted with
  these controls. Camera/both-arm readback was verified and Gray selected; no
  robot/tool command or station teaching occurred. Final checks passed 131
  physical-controller and 37 physical/show UI tests after the full regression.

## Native audio loudness — October 30, 2026

- `codex/macbook-audio-loudness` builds on four-ball commissioning `81d927b`.
  Ghost Booth adds an explicit native-only 3× signal boost with a peak limiter
  after the combined voices. FX and boost default off; Logic routing, raw MIDI,
  per-ball identity and robot behavior are unchanged.
- Backend/UI distinguish a requested gain from native acknowledgment and require
  engine/limiter readiness. Offline renders verify gain, summed-voice limiting,
  mute/tail clearing and retained gain; mock tests cover controls and failures.
- The preceding manual-drop test reached the bucket according to the operator,
  but venue loudness was insufficient at the existing maximum. This observation
  is not an automatically scored capture-rate result. See
  [loudness instructions](AUDIO_LOUDNESS.md) and the dated audio STATUS evidence.
- 440 Python and 107 Node tests passed, plus Scripter and orientation assertions.
  Only Ghost Booth was restarted; saved four-ball mappings were restored and
  the 3× control was verified in the browser. The operator confirmed boosted
  hand-drop sounds were loud enough and clear. Robot/physical-dashboard,
  simulation and Viam configuration were untouched.

## MacBook four-ball audio commissioning — October 30, 2026

- `codex/macbook-ball-commissioning` builds on hardware-demo `ef27aca`.
  Native Ghost Booth now services the main run loop, fixing observed stale
  CoreMIDI discovery after Bluetooth pairing. A read-only A/B experiment isolated
  the issue, and live Green hot-plug was verified after the audio-only restart.
- Pink/boing, Purple/squeak, Green/spring and Gray/plop each produced measured
  MIDI input and user-confirmed sound. All four also passed one combined,
  explicitly armed 30-second manual audio test with separate ball/trial routing.
  This supersedes the earlier unassigned-binding/listening status for this Mac.
- Actual endpoint bindings, generic voice assignments and evidence were saved
  only in ignored host-local configuration/data. Ghost Booth remains running on
  loopback with FX muted; the temporary observation collector has stopped.
  Logic Pro and the Mac mini were not required for this test.
- Full regression: 422 Python and 106 Node tests passed, plus Scripter and
  orientation-contract assertions. No robot connection or actuation, Viam
  reconfiguration, physical-dashboard startup, or simulator restart occurred.
  Robot-cycle timing and physical capture/miss/surface classification remain
  separate commissioning work. See [audio evidence and limits](../sircus/audio/STATUS.md#macbook-ble-hot-plug-discovery-correction--october-30-2026).

## MacBook hardware-demo preparation — October 30, 2026

- `codex/macbook-hardware-demo-ready` builds on demo-reel `d28ade95`.
  The saved-pose routine now has an offline host-profile check and a separate
  explicit deployment launcher. It reuses original calibration and journals;
  credentials, host paths and generated runtime state remain ignored.
- Hardware startup reserves its HTTP port and exclusive same-user sfhack10 lease
  before creating SDK/audio clients. Bounded cleanup releases owned resources
  without issuing arm/tool commands. Remote/legacy controllers remain an operator
  ownership check. Source synchronization does not deploy anything.
- Hardware show UI opens Manual mode, separates saved targets from live readiness,
  and reports native audio engine plus bound/connected ball counts. An explicit
  Reapply action recovers saved mappings after Booth restart, muting audio first.
- Ghost Booth can run on this MacBook without Logic Pro or Mini connectivity.
  Matching mapping/engine status is verified at FX enable; stale/restarted state
  cannot claim readiness. Generic measured MIDI impacts retain ball/trial identity;
  audio does not gate robot operations. Actual four-ball bindings remain unassigned.
- Native Swift helper compiled without execution. Full regression passed420 Python
  and106 Node checks plus standalone Logic guards; final launcher regression
  passed17 focused tests after a profile-snapshot correction. Static browser QA
  verified Manual mode and disconnected controls. No hardware or live audio was commissioned;
  existing processes and all physical/cloud configurations remain unchanged.
- [Operator startup and remaining checks](HARDWARE_DEMO_READY.md) distinguish
  prepared source from live camera/state, clearance, recording, MIDI and listening
  acceptance. Fixed targets do not require the still-missing variable-height limits.


## Simple demo and learning film — October 30, 2026

- `codex/macbook-demo-reel` is stacked on deterministic reliability commit762133b.
  `/demo.html`, linked from the show desk, plays a fixed four-color routine and a
  recorded frozen-checkpoint learning sequence. `./start-demo.sh` starts a separate
  credential-free loopback preview on18772 with physical mode disabled.
- Fixed nominal preset: four predetermined fresh releases,4/4 numerical captures,
  no retuning/retries. Each shot resets an empty bucket; multi-ball interaction is
  not modeled. Browser-local original effects default off, with generic contact
  voices per color and carrying/settling drums. Stop/mute cancel pending playback.
- Learning replay uses the original N8/24/64/128 checkpoints and shared held-out
  layout, retaining all four misses plus a separately labeled successful example.
  Population mean59.0%→69.4%; default checkpoint33/48. No per-scene optimization
  during inference. Displayed arm positioning is illustrated endpoint movement;
  numerical ball traces do not establish valid native paths or physical transfer.
- Replay reconstruction permits only absolute1e-9 numerical rounding in frozen
  inference checks, then retains the original recorded action, noisy scene and
  trajectory exactly. Cross-CPU checks keep hashes/IDs/seeds/outcomes exact.
- Added an independent downloadable film and source-linked checksum manifests.
  Existing studies, geometry, scoring definitions, policies and original film
  remain unchanged. See [demo instructions](DEMO_REEL.md).
- Full regression passed383 Python and103 Node tests plus standalone Logic guards.
  Browser verification completed the fixed routine, initialized local audio at
  zero volume and exercised Stop. Real sound listening/ODD commissioning and real
  robot validation remain separate; no physical or cloud configuration was changed.

## Deterministic routine reliability — October 30, 2026

- `codex/macbook-deterministic-reliability` builds on draft four-mode commit
  `46fe121`. Manual operation still uses exact saved TCP poses, a two-second loading
  countdown and two-second drop settling, with another operator press for every repeat.
- Stop cannot be prevented by journal-write failure. Unresolved planner/actuator
  requests and interrupted Stop keep recovery locked; stale action epochs cannot
  restart a pre-Stop command. Readback loss invalidates all held-ball/open caches.
- Recording must belong to the current trial and be fresh from request start,
  including acquisition/write delay. Old canceled writes retain their original
  trial directory. Unknown native-capture ownership is reported and requires
  operator inspection. Failed calibration saves preserve the prior saved targets.
- Browser requests have deadlines and no automatic command retry. Ambiguous
  actions require verified Stop and renewed setup confirmation; old replies cannot
  unlock controls. Pending pose edits survive delayed/failed saves.
- All 368 Python tests, 92 Node tests and standalone Logic guards passed, including
  73 physical-controller/adapter tests using mocks. Numerical baselines, policies,
  geometry and scoring definitions are unchanged.
- A separate credential-free preview uses loopback port 18771. The existing 18766/18769/
  18770 services, physical services, Viam config and real audio service were not
  changed. [Acceptance instructions](DETERMINISTIC_RELIABILITY.md) include a
  no-SDK endurance runner and a future supervised physical test. Physical
  reliability and bucket capture accuracy remain unverified by this checkpoint.

## Four-mode bucket-facing workflow — October 30, 2026

- `codex/macbook-four-mode-workflow` builds on the draft three-mode show checkpoint
  `70e4edd`; main and earlier drafts remain separate. The four visible choices are
  Manual operation, Trained operation, Simulation and Mock simulation. Manual
  operation retains the existing saved TCP load/drop editor and controller.
- Mock and Trained share board-center entry/seeded XYZ-box placement, supplied
  bucket position, shot preparation, loading acknowledgment and Go. Horizontal
  board-normal heading points toward the bucket; tilt is measured from horizontal
  (0°) to vertical (90°), with configurable bounds strictly between them.
- Mock uses explicit bounded search. Trained runs one guarded frozen prediction
  plus a separately recorded geometric heading constraint, without search fallback.
  Raw prediction, transformed release, both endpoint checks and synthetic evidence
  remain visible. The previous33/48 benchmark does not apply to this new variant.
- Go illustrates grip, board orientation, carrying,2s settling, release and the
  returned numerical ball trajectory. Viewer focus and an overlay Stop make the
  sequence observable; interpolated arm endpoints are not validated native paths.
  Loading, dirty-scene, mode mismatch, cancellation and stale-result guards remain.
- [Small diagnostic](../baselines/facing-rehearsal-20261030/README.md): three
  prespecified layouts, five fresh noisy releases each. Search15/15; constrained
  frozen prediction10/15, retaining all five misses in the right/farther layout.
  No test-based refitting or action reselection; no physical validation claimed.
- Validation:340 Python tests,81 Node tests and standalone Logic guards passed.
  Browser QA completed a nominal mock capture, boxed randomization, a constrained
  learned capture and viewer Stop/cross-mode guards. An isolated credential-free
  mock preview uses loopback18770; existing18769/18766 services were not restarted.
  No physical commands, real audio playback or Viam cloud changes were performed.
  See [workflow details](BUCKET_FACING_WORKFLOW.md) and [show runbook](SHOW_RUNBOOK.md).

## Three-mode show integration — October 30, 2026

- Draft integration branch `codex/macbook-three-mode-show` combines the guided
  workspace and audio draft source without merging main or those draft PRs.
  `/show.html` exposes Simulator, Deterministic and Trained views. Default launch
  `./start-show.sh` uses mock arms/camera/recording on loopback18769; hardware mode
  is explicit and has no numerical workflow routes. See [show runbook](SHOW_RUNBOOK.md).
- Deterministic pose editing records operator-configured Viam TCP targets without
  motion or a calibration claim. The fixed routine uses two seconds to load and
  two seconds settled before release, with optional unlabeled practice records.
  Stop and conflicting-action guards retain the existing physical-controller logic.
- Pink/Purple/Green/Gray are explicit operator labels with unassigned MIDI bindings.
  A verified input routes measured generic impacts only; no manual contact buttons,
  object inference or automatic capture/miss labels. Older released balls retain
  their own trial/window while a different ball loads. FX starts off; lifecycle
  hooks enqueue without audio I/O, and cancellation cannot stop another ball's cue.
- An original bounded drum-roll loop spans carry and the two-second settle, ending
  immediately before release. Replay startup/mute races found by Ubuntu CI were corrected with an audio-only
  readiness epoch and nonblocking cancellation ownership. Four-input native MIDI parsing and scoped voice
  cancellation compile and pass isolated tests without opening devices. Real BLE,
  listening, private-LAN delivery and optional Logic commissioning are outstanding.
- Validation:333 Python unittest checks,71 Node tests and the standalone Logic
  guard assertions passed. Coverage includes unreachable/delayed audio, duplicate
  events, cancellation/lock contention, four identities, repeated synthetic replay,
  pose configuration, fixed-cycle mocks, UI routes, scoring and immutable baselines.
  A new mock-only preview was launched/refreshed; existing18766 operator service,
  physical services, cloud machine config and hardware were not changed.
- Current synthetic learning evidence remains33/48 frozen inference versus25/48
  fixed action and37/48 separately labeled bounded search. No learning result,
  fingerprint or compatibility guard was rewritten by show/audio integration.
- The existing presentation was reduced to eight spacious circus/Halloween slides,
  retaining real photos and adding current UI, measured study results and explicit
  uncommissioned perception/audio boundaries. Physical trained deployment is still
  blocked by calibration, detector and native-path requirements.

## Corrected front-bucket study and commissioning stubs — October 30, 2026

- Active draft PR5 / `codex/macbook-guided-training` keeps main and audio drafts separate.
  The operator workspace defaults to front-negative-y-v1: origin at dropper arm10,
  +X toward arm14, +Z up; bucket and operator face the board from negative Y.
  Screenshot TCP values remain historical evidence, not calibrated ball-center poses.
- Historical positive-Y assets/checkpoints and rectangular/bucket baselines are
  preserved. The corrected profile has separate fingerprint guards and immutable
  training/test evidence. Profile changes invalidate stale scans and plans.
- Initial corrected24/6/8 study: frozen3/8, fixed2/8, limited4-proposal search2/8;
  the frozen95% interval is13.7–69.4%. The small sample is inconclusive. See
  [the current study record](FRONT_BUCKET_STUDY.md) for the larger prespecified
  three-seed N8/24/64/128 curves, held-out challenge results and limitations.
- `/study-results.html` displays completed synthetic reports, uncertainty, actual
  ball trajectories, and video export. `/operations.html` provides guided rehearsal,
  profile-specific training and a read-only physical-readiness checklist. Numerical
  arm endpoints remain separate from native collision-path and dynamics validation.
- [Viam deployment stubs](VIAM_DEPLOYMENT.md) add typed scan/kinematics/planning/
  recording/motion interfaces, a blocked coordinator and unknown-result vision
  service. Actual localhost Viam1.14.3 RPC tests passed with no hardware components.
  A private unattached cloud fragment was saved with zero deployments; physical
  machine configuration and robot commands were not changed.
- Physical implementation still requires a real bucket detector, reference-based
  camera/world and board-mount calibration, single-controller ownership, taught
  scan/load poses, measured grasp offset/release limits, recording validation and
  collision-checked paths. Native v2 replay compatibility guards remain in place.

### Final overnight checkpoint

- Larger study: three training seeds, nested N8/24/64/128; 16 validation and 48
  shared disjoint test layouts. Mean captures:59.0%,67.4%,67.4%,69.4%. Prespecified
  default N128/seed20261030:33/48 (95%54.7–80.1%), fixed25/48, bounded search37/48.
  At ±200mm lateral offsets default frozen inference scored0/24; separate diagnostic
  search found18/24 and12/24. No diagnostic result was used to refit/select the model.
- Final suite:193 Python and53 Node tests passed, including physical-controller
  mocks, versioned scoring/contact, source/asset guards, all576 saved predictions,
  numerical replay, UI races and deployment-stub tests. Independent reconstruction
  matched all12 train-only model fits within1.89e-15.
- Only the dedicated offline operator process on port18766 was refreshed after
  confirming it was idle and preserving its status. It serves the published front
  checkpoint/report with physical controls disabled. It is manually launched via
  `./start-operator-sim.sh`; this is not physical deployment or a Viam cloud runtime.
  Temporary QA and native-stub servers are stopped when verification completes.
- The video is a rendered record of actual saved numerical trajectories and results,
  not a physical trial or validated native arm path. Browser video export remains
  capability-dependent; a standalone renderer is supplied for repeatable output.

## Historical guided-workspace checkpoint — October 30, 2026

- `codex/macbook-guided-training` adds `/operations.html` on an explicitly opted-in,
  local-only offline server. Start with `./start-operator-sim.sh` on unused port18766.
  This source change does not restart or deploy any existing runtime.
- Guided rehearsal: board pose → bucket placement → synthetic scan → fixed-board
  search or frozen learned-angle preview → operator loading acknowledgment → grip/drop
  and fresh numerical trajectory. The board plane comes from simulated arm14 joints
  and the provisional mount. Bucket evidence is synthetic; no real detector is claimed.
- Training runs actual ridge regression through the existing seeded/disjoint study,
  saves a frozen checkpoint and evaluates held-out layouts, with fixed-action and
  per-scene search comparisons separate. Jobs are bounded and cancellable; successful
  artifacts can be downloaded or selected in the guided workspace.
- Validation:150 Python and26 Node tests passed, including existing physical-controller
  mocks and scoring/contact/fingerprint tests. Browser verification completed scan,
  frozen prediction, load, fresh captured trial, real4/1/2 training and checkpoint
  selection. The tiny training smoke gave2/2 for all three methods; it shows no
  advantage over the baseline or physical readiness. Generated trial/training data
  stays ignored and does not replace published baselines.
- Read [operator boundaries](OPERATOR_WORKSPACE.md), [training contract](TRAINING_WORKSPACE.md)
  and [device synchronization](DEVICE_SYNC.md). Physical integration still needs a real
  bucket detector, verified camera/world and board-mount transforms, taught scan/load
  poses, release calibration, collision-aware paths and a matching native v2 adapter.
- Main remains separate from this task branch. Audio PR1 and PR4 remain drafts;
  this workspace does not merge them. Existing services, cloud configuration and
  hardware are untouched.

## Integration and learning checkpoint — October 30, 2026

- PR #2 was reviewed and merged at `6f00ac3faa4e34d508b005891e62d9911ead81cf`; see [review evidence](PR2_INTEGRATION_REVIEW.md). Scoring requires a board bounce, whole-ball entry and one continuous second inside, including rim-assisted captures. Earlier result definitions are preserved.
- New scoring is opt-in. The existing Viam runtime and running dashboard have not automatically switched. The v2 native adapter is still missing and compatibility guards still block replay; endpoint IK is not path validation.
- [Scene-conditioned study](SCENE_CONDITIONED_STUDY.md): 24 training, 6 validation and 8 disjoint test layouts; real ridge regression from training-only search labels; five explicit actions; frozen direct inference. The [published synthetic checkpoint](../baselines/scene-conditioned-20261030/README.md) scored 8/8 versus 6/8 fixed action and 8/8 separately labeled per-scene search. This narrow, eight-layout test does not establish broad generalization or physical performance.
- A separate numerical-preview API/UI loads the checkpoint and replays its trajectory in the existing viewer. It was tested on an isolated offline dashboard with physical controls disabled. No deployment or restart of existing services, hardware connection, Viam changes or audio merge accompanies this checkpoint.
- Next boundaries: independent Ubuntu reproduction and broader prespecified studies; a matching fake-only v2 native adapter and collision-aware replay; calibrated physical trials only under separate authorization. Audio remains on draft PR #1.

## Historical initialization snapshot — October 29, 2026

- Mac: physical sfhack10 hosts arm10/dropper and arm14/board holder. Operator dashboard on localhost8765. Practice cycles use taught poses,2s loading,2s settled release,2s post-release recording. Completed practice records are retained locally unlabeled. Successful panel Open commands can skip redundant loading Open; unknown/failed states do not. Saved hardware poses/credentials are deliberately not in Git.
- Ubuntu: isolated fake-only server8083 and local dashboard8765. New orientation/height scripts from the tested Varun transfer are now canonical in sircus/two-arm. Original transfer archives were not changed.
- Geometry: board200x300x5mm, landscape90° mount; base spacing1337.31mm. Tapered bucket inner177/150mm, outer200/158mm.175mm reported height remains unresolved; provisional simulation depth175mm and8mm base. Mount offsets, masses, support compliance and bucket placement remain provisional.
- Learning: old U/V/gap search has one fixed board pose. New CLI searches roll/pitch/gap at fixed release XY and fixed board/bucket centers. This is not a generalized policy. New schema/fingerprint protects replay; selected candidates require native path validation.
- Perception: camera on arm10 tool bracket; stationary capture works. Camera-to-world calibration and reliable ball/board/bucket localization remain incomplete. No current scene overlay is proof of calibration.
- Tests before repo initialization:34 physical-controller/adapter mock tests;46 two-arm baseline regressions plus orientation contract checks. New training smoke4 candidates/2 synthetic tests and8 successful native fake-arm motion checks on Mac. Linux CI is the next portability check; native Linux Viam runtime remains independently unverified.
- Native cloud simulation runtime previously on the Mac is no longer assumed running. Do not reuse its cloud identity without coordinating ownership. Local8083 needs no credentials.

Current detailed physical/perception STATUS files include historical PIDs and timestamps. Verify live state before acting. Read docs/UBUNTU_SETUP.md for repository startup; old transfer/local absolute paths are historical.

## Mac Mini audio — October 30, 2026

Ghost Booth is the unified local event-sound/rehearsal interface under
`sircus/audio/booth/`, with eight configurable event slots and sixteen original
sounds. See [audio status](../sircus/audio/STATUS.md). It runs independently with
FX off and no robot dependency. The main audio branch/draft PR carries this code;
no physical dashboard hook, Viam configuration or robot deployment is included.

## MacBook audio integration checkpoint — October 30, 2026

The separate `codex/macbook-dashboard-audio` task branch includes reviewed main
`6f00ac3faa4e34d508b005891e62d9911ead81cf` and the draft audio client. It adds
initially-off FX, volume, sanitized connection/mute status and nonblocking
physical-controller observers. [Integration details](DASHBOARD_AUDIO_INTEGRATION.md)
describe evidence, command ordering, cancellation, tests and private-LAN setup.
This is source integration only: no deployment, service restart, hardware call
or cross-computer validation. Audio PR #1 remains draft; learning PR #3 remains
separate. New scoring is opt-in and v2 native replay remains blocked.

## Reviewed learning merge and audio corrections — October 30, 2026

PR #3 was merged into main at `a376ad67968f6445d39a51c2a29325a204bb582a`.
The audio integration branch retains both the numerical study preview and
audio controls after resolving the shared server, test script and style changes.
Published study artifacts, scoring versions and native replay guards are unchanged.
Audio PRs #1 and #4 remain drafts; this is source integration, not deployment.
No existing services were restarted, no audio was auditioned, and no Viam or
physical robot connection was used. Real listening, ODD/BLE and authenticated
cross-computer commissioning remain separate follow-ups.
