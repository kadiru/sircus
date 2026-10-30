# Sircus cross-computer handoff — October 30, 2026

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

The earlier recovery/reload notes below are historical. Latest physical status
is the completed Round2 state above. Source branch remains
`codex/macbook-held-ball-recovery` (draft PR #15).

## Update: Green finished; reload pause

The earlier blocked state below is historical. `codex/macbook-held-ball-recovery`
(implementation `0f44107`) adds explicit verified quiescent recovery, held-ball
continuation and a dashboard-wide camera-disable flag. Recovery and Green's
remaining transfer/release completed physically, keeping the original trial and
without repeating pickup/close. Round 1 is complete; Round 2 remains paused for
operator reload. Both arms reported stationary, fault cleared, FX muted.
Camera use is disabled in the physical dashboard; Viam camera/frame configuration
and other clients are unchanged. No board or suction command was issued.
The active physical dashboard now uses the held-ball-recovery checkout. The
previous integration branch and stopped-state sections below are dated evidence,
not instructions to retry Green. Other computers should fetch the new branch for
current implementation. Headless work remains separate from deployment.

Read this first, then `AGENTS.md`, `CONTRIBUTING.md`, and `docs/PROJECT_STATE.md`.
This is a dated handoff, not a live robot status feed. Recheck actual runtime and
operator readiness before any separately authorized hardware work.

## Source of truth and checkout

- Repository: https://github.com/kadiru/sircus (private).
- Fetched `origin/main`: `a376ad67968f6445d39a51c2a29325a204bb582a` (PR #3 merge).
- Latest hardware/demo integration branch: **`codex/macbook-no-camera-demo`**.
- Implementation checkpoint: `321f3d1` (camera-off pickup mode).
- Physical outcome checkpoint: `b4b469e52ab8feb5d18095731dbf32755330d45e`.
- This handoff is added afterward on that same branch. Fetch its actual head.
- All 14 Mac Git worktrees were clean and matched their fetched remote branch
  before this documentation update. Ubuntu/Mini local checkouts were not inspected;
  a successful push does not mean those computers have pulled it.
- `main` is an ancestor of this branch, but does **not** contain the full latest
  demo/UI/audio work. Do not use a historical archive as the current source.

From an existing clone, without changing its checked-out files:

```sh
git fetch origin
git status --short --branch
git worktree add -b codex/YOURNAME-TASK ../sircus-YOURNAME-TASK \
  origin/codex/macbook-no-camera-demo
cd ../sircus-YOURNAME-TASK
```

Replace the placeholders with a unique owner/task. If a worktree already exists,
inspect it rather than overwrite or reset it. Use the latest integration branch
for work needing the newest UI/audio/physical code. Work specifically targeting
reviewed main may branch from `origin/main`, but it must not assume these newer
features are merged. Push a task branch and open a draft PR against the agreed
base. Do not push directly to another agent's branch or merge the draft stack.
Read `docs/DEVICE_SYNC.md` for the wider synchronization workflow.

## Immediate hardware state and interrupted task

The requested demo is four balls in pickup order **Gray → Purple → Pink → Green**,
Round 1, then a mandatory pause for manual reload before Round 2. Colors identify
sound inputs; the physical pickup order follows the four taught station positions.
Two paired board/drop presets have been taught; a third round is only planned.
The paired round sequencer currently lives as a host-local operator script, not
as a portable multi-round dashboard feature. Another computer must not run it.

Latest attempt: **19:47:39–19:48:45 UTC, October 30**:

- Explicit camera-off practice was requested by the operator. Gray, Purple and
  Pink completed grip, vertical lift, saved drop move, two-second settle and release.
  Their local trial records are completed/unlabeled; bucket outcomes are unknown.
- Green acknowledged grip and reached its 100 mm approach height. A robot-state
  readback failure during lift/settle interrupted the routine before transfer/release.
- The orchestration requested Stop and muted FX. The controller retains a
  **Stop/planner-cancellation verification fault**. No automatic retry was made.
- At **19:49:58 UTC**, a separate read-only inspection reported both arms
  stationary, no active Viam operations, dropper holding true and vacuum holding
  true. The operator also confirmed Green remains held. These observations do
  not clear the controller fault or authorize bypassing it.
- Board remains at Round 1; no board motion or suction command was issued by this
  camera-off run. Round 2 has not started. Do not open Green's gripper or restart
  pickup while it holds the ball. Reload is not yet the next physical action.
- Earlier, one four-ball station run completed at 19:06–19:07 UTC with camera
  recording. Keep that evidence separate from this interrupted run.

Hardware resources on `sfhack10`: `arm` = dropper 10, `arm-board` = board holder
14, `gripper` = dropper tool, `vacuum-board` = AS1200, `cam` = RealSense,
`builtin` = motion service. The camera is on arm10's tool bracket per operator.
The existing physical panel Stop controls the dropper; it is not a global
both-arm stop. Independent/legacy control of arm14 must be checked before future
board movements, including any `sfhack14` runtime. No exclusivity claim is made.

## What is implemented

### Deterministic physical demo

Files: `sircus/physical_trials/controller.py`, `viam_adapter.py`,
`sircus/simulation/web/show.{html,js}`, `physical.js`.

The Manual mode has hand-loading and per-color taped pickup workflows. Each taught
pickup is a full world-frame arm flange pose. Approach is world Z +100 mm at the
same XY/orientation; descent and lift request native linear constraints. Transfers
use the highest saved pickup approach as clearance. Native rejection has no
unconstrained fallback. Low starts must match the selected grasp/approach for
vertical recovery; arbitrary low lateral moves are rejected.

`pickup_drop_cycle` requires the selected `ball_id`,
`pickup_setup_confirmed:true`, and `fixed_setup_confirmed:true`.
`practice_mode:true` retains a completed trial without outcome labeling.
Optional **`recording_mode:"operator_log_only"`** is accepted only for an explicitly
unlabeled taped-pickup practice cycle. Default remains `camera_required`.
Camera-off skips starting camera recording and its freshness gate, but retains
fresh both-arm state, fixed-board joints, intended poses, grip acknowledgment,
settling, cancellation and Stop checks. Records retain discrete pickup/release
state, timestamps, IDs, mode and unknown outcome. Camera preview may still be
requested by an open browser; camera-off recording is not network isolation.

`GET /api/physical/status` supplies allowed actions and an action epoch/token.
`POST /api/physical/action` uses `X-Sircus-Token` and `expected_action_epoch`.
Treat tokens as session credentials, not code/config to share. Use existing
controller routes, not an alternate actuator client to evade a fault. There is
no commissioned "resume while holding after failed Stop" route.

### Sounds

Ghost Booth runs locally on the MacBook; Logic Pro and the Mini are not required.
Measured MIDI inputs for all four balls were individually and jointly heard by
the operator: **Pink boing, Purple squeak, Green spring, Gray pinball** (replaces
Gray's quieter historical plop). Volume 100% plus the peak-limited 3× native boost
was audibly commissioned. FX was enabled during the latest run and muted on abort;
we do not infer this run's acoustic result from successful API enqueue alone.

Carry drums, release cues and identified generic impacts use nonblocking audio
queues. Preserve trial/event IDs and provenance. Audio never gates robot motion
or Stop. Unknown outcomes stay silent. A measured ODD tap does not identify
board vs bucket or prove capture; the operator explicitly chose measured generic
bounce sounds only. Multiple balls can remain audibly active in their event
windows; numerical simulation does not model accumulated-ball collisions.
See `docs/AUDIO_LOUDNESS.md`, `docs/DASHBOARD_AUDIO_INTEGRATION.md`, and
`sircus/audio/EVENT_CONTRACT.md`. Actual MIDI bindings and credentials are local.

### Simulator, learning and film

The show desk has Manual, Trained, Simulation and Mock simulation modes. Numerical
modes do not command the physical backend. The guided UI supplies board/bucket
positions, ball-loaded acknowledgment, and Go. Geometric heading faces the bucket;
search/learned proposals select tilt, U/V and release gap. This does not establish
native collision-free trajectories or autonomous real-world perception.

Portable source/evidence already exists in this branch:

- `sircus/studies/`, `baselines/front-bucket-20261030/`,
  `docs/FRONT_BUCKET_STUDY.md`: disjoint seeded layouts, real ridge-model fitting,
  frozen inference and learning curves. Mean test capture over three seeds on the
  same 48 held-out layouts: N8 59.0%, N24 67.4%, N64 67.4%, N128 69.4%.
  Prespecified checkpoint 33/48; fixed baseline 25/48; separate search 37/48.
  These are synthetic results, not physical success rates. Heading-constrained
  rehearsal is a separate contract, not automatically the published 33/48 model.
- `docs/DEMO_REEL.md`, `sircus/simulation/web/media/demo-reel/`: fixed four-ball
  numerical presentation, saved learning replay, media/checksums and film guidance.
  `./start-demo.sh` starts a separate fake-only preview when explicitly desired.
- `docs/SCORING_CONTRACT.md`: opt-in complete-shot score requires board bounce,
  whole-ball entry and one continuous second inside. Rim-assisted capture counts;
  partial exits reset retention. Legacy rectangular/bucket results retain original
  scoring/fingerprints. V2 native replay remains blocked pending its matching
  adapter; do not remove compatibility guards.
- `docs/BUCKET_FACING_WORKFLOW.md`: tilt 0° horizontal, 90° vertical; explicit
  geometric heading and distinction between bounded search and frozen prediction.

### Geometry and perception boundary

Board 200×300×5 mm, landscape; bucket open-top straight taper, inner top/bottom
177/150 mm and outer top/bottom 200/158 mm. Reported height 175 mm is unresolved as
overall height vs interior depth. Contact uses inner dimensions, clearance outer.
Arm separation is operator-measured 1337.31 mm; full base and mount transforms,
bucket placement and compliant-contact parameters remain provisional.
Host wall-right was moved to Y=-900 mm; ceiling raised by 200 mm. These are
operator-directed configuration values, not a new geometric survey. Host obstacle
and calibration changes were not copied into historical simulation scenes.

Real board pose should eventually come from calibrated arm14 FK/tool mounting;
vision should locate the bucket. Camera-to-tool/world calibration, metric depth,
alignment and real bucket localization remain uncommissioned. `sircus/perception/`
contains reusable capture/inspection scripts. `sircus/deployment/` contains typed
contracts, mock coordinator and blocked Viam stubs—not a deployed autonomous
physical system. See `docs/VIAM_DEPLOYMENT.md` before claiming otherwise.

## Current Mac services (process inspection during this handoff)

| Loopback port | Role | Source checkout/branch |
|---|---|---|
| 8765 | Hardware show dashboard, faulted after Green | `codex/macbook-no-camera-demo`; loaded implementation `321f3d1` |
| 8766 | Ghost Booth audio bridge | running native audio service; no restart in this handoff |
| 18766 | Offline operations/training UI; physical disabled | `codex/macbook-guided-training` checkout |
| 18770 | Four-mode dry-run show | `codex/macbook-four-mode-workflow` checkout |

Presence of a listener is not application/hardware validation. The latest Git
branch is not necessarily loaded by these older processes. No service was
restarted, hardware accessed or cloud configuration changed for this handoff.
Do not launch the same Viam cloud part on another host. Another agent can run
headless tests or isolated fake-only previews without sharing these services.

## Draft integration stack (fetched from GitHub)

All listed PRs remain draft; none was merged by this handoff.

| PR | Head branch (`codex/` prefix) | Base |
|---|---|---|
| #1 | mac-mini-oddball-sfx | main |
| #4 | macbook-dashboard-audio | mac-mini-oddball-sfx |
| #5 | macbook-guided-training | main |
| #6 | macbook-three-mode-show | main |
| #7 | macbook-four-mode-workflow | macbook-three-mode-show |
| #8 | macbook-deterministic-reliability | macbook-four-mode-workflow |
| #9 | macbook-demo-reel | macbook-deterministic-reliability |
| #10 | macbook-hardware-demo-ready | macbook-demo-reel |
| #11 | macbook-ball-commissioning | macbook-hardware-demo-ready |
| #12 | macbook-audio-loudness | macbook-ball-commissioning |
| #13 | macbook-fixed-pickup | macbook-audio-loudness |

This handoff/camera-off branch is stacked on `codex/macbook-fixed-pickup`.
Do not independently merge overlapping PRs or assume every branch change exists
in every other branch; inspect ancestry/diffs. PR #2 and #3 are already in main.

## Work another agent can take without blocking this Mac

Choose one bounded task with the user; these are recommendations, not assignments.

1. **Mock-only Stop recovery review**: inspect `ViamAdapter.stop`, owned operation
   discovery, RPC cancellation and `Controller._stop_and_verify`. When a Move
   finishes or disappears during transport failure, `was_active` can require a
   cancellation acknowledgment that is unavailable. This is a hypothesis from
   source review, not a proven sole cause. Reproduce late completion, vanished
   operations, failed discovery, pending actuator requests and transport errors
   in mocks. Design evidence-based recovery; never clear faults merely because
   one snapshot is stationary or restart the process to erase ownership state.
2. **Portable multi-round UI/state machine**: replace the private operator runner
   with an explicit Round1 → pause/reload → Round2 flow, immutable paired presets,
   single motion owner, board transition guards, Stop, failure recovery, and
   no implicit release of a held ball. Mock first; do not ship host calibration.
3. **Ubuntu numerical work**: investigate distance/lateral generalization using
   frozen evaluation and held-out layouts; retain historical checkpoints. Or
   implement/test the matching v2 native *simulation* replay adapter without
   changing physical endpoints or disabling compatibility checks.
4. **Audio resilience**: test stale/duplicate/cancelled cues and multiple active
   ball windows against mocks. Preserve measured generic provenance and no audio
   dependencies in motion. Do not overwrite commissioned local MIDI bindings.
5. **Perception offline**: implement bucket estimation and calibration contracts
   with synthetic fixtures/publicly shareable data. Missing physical depth/extrinsic
   evidence stays unknown. Real observations require a separately arranged transfer.

The Mac operator task remains paused with Green held; only the authorized Mac
hardware owner should arrange recovery and eventually the manual Round2 reload.
Other computers should neither connect to sfhack10 nor deploy/restart this panel.

## Validation and excluded host artifacts

Latest implementation checks: **136 physical-controller Python tests, 38
physical/show Node tests**, repository credential/file scan, and `git diff --check`.
Those are mocks/UI checks, not proof of reliable hardware networking. To reproduce
on an installed pinned Python 3.12 environment, from repo root:

```sh
python -m unittest discover -s sircus/physical_trials -p 'test_*.py' -q
node --test sircus/simulation/tests/test_physical_panel.mjs \
  sircus/simulation/tests/test_show_panel.mjs
python scripts/check-repository.py
```

`setup-ubuntu.sh` and dependency specifications are versioned. Verify your selected
interpreter/dependencies; do not copy a Mac virtualenv. Broader tests: `scripts/test.sh`.

Excluded deliberately: credentials/tokens, calibration.json, paired round poses,
actual MIDI bindings, raw RGB-D/state logs, runtime profile/config, upload queues,
virtualenvs, and machine-local scripts containing calibration/control assumptions.
The Mac owner can locate the private evidence under `.runtime/two-round-demo/`,
`.runtime/pickup-station/`, `.runtime/board-round-test/`, and
`sircus/physical_trials/data/hardware/`. Use these as local evidence references,
not instructions to publish those directories or to recreate physical poses.
All reusable controller/UI/audio/learning code and compact published synthetic
baselines are already on pushed branches; no hidden private runner is required
for single-ball UI operation or the numerical study.
