# Sircus shared-development handoff — 2026-10-30

## Start here

Latest handoff branch: `codex/macbook-handoff-slides`, based on
`codex/macbook-round-operator-ui` at `17cc4a5a9a54b13fc878d91ac41e635818fe89b9`.
The operator implementation is `5ef1b88`. PR #16 remains draft and the latest
physical UI is **not merged into main**. Do not use main alone as this baseline.
This documentation branch does not merge the independent draft PRs.

The Mac audit found all 16 existing worktrees clean, with HEAD equal to their
fetched upstream. That verifies published Mac source, not the current state of
Ubuntu or Mac mini checkouts. Credentials, calibration, logs, queues, recordings
and generated runtime configurations intentionally remain outside Git.

On Ubuntu or Mac mini, preserve local changes and existing services. From an
existing clone, create a separate task worktree (choose an unused branch/path):

```sh
git status --short --branch
git fetch origin
git worktree add -b codex/ubuntu-next-task ../sircus-next-task origin/codex/macbook-handoff-slides
cd ../sircus-next-task
```

Read AGENTS.md, README.md, CONTRIBUTING.md, docs/PROJECT_STATE.md,
[AGENT_HANDOFF.md](AGENT_HANDOFF.md), and [ROUND_OPERATOR_UI.md](ROUND_OPERATOR_UI.md).
Source synchronization is not deployment. Do not start the same Viam cloud part
on another computer, restart existing services, or connect to physical arms as
part of synchronization. Use mocks/fake-only training and fresh dependencies.
Commit and push your task branch, report its exact commit and tests, and keep
unrelated changes and draft PRs separate.

## What exists now

- Mac hardware dashboard: `http://127.0.0.1:8765/show.html`, launched from the
  `sircus-round-operator-ui` worktree with cameras disabled. Ghost Booth remains
  separately hosted on port 8766. These are historical deployment observations,
  not guarantees of current process state. This handoff task restarts neither.
- Manual operation: saved two-round board/drop pose pairs; four taped pickup
  points in Gray → Purple → Pink → Green order. Each pickup uses a 100 mm
  approach, straight descent and lift, planned transfer and a two-second settled
  release. Reload between rounds requires a separate operator confirmation.
- All four sound identities were physically heard: Pink boing, Purple squeak,
  Green spring, Gray pinball. Measured generic impact events do not distinguish
  board from bucket or prove capture. Drums accompany carry and settling. Audio
  stays asynchronous and cannot gate motion, release or Stop. Wake all four
  inputs before a demonstration; the last idle UI check had only three awake.
- Earlier physical execution completed eight drops over two rounds, with one
  separately retained interrupted Green approach and a later Green-only retry.
  Outcomes remain unlabeled; this is not a measured capture rate.
- The new backend/UI round coordinator passed 156 controller/adapter mock tests,
  42 UI tests, 11 startup checks and a browser mock rehearsal of both rounds.
  Its first physical run remains outstanding. Prior script-driven physical runs
  are not relabeled as validation of the new coordinator.

## Fault boundary and recovery

Green has succeeded repeatedly but has also stopped at different stages:
post-grip/lift in one run, approach before gripping in another. It is fourth and
farthest; that correlation does not establish the cause. The new coordinator
removes the extra chat-side cached-connection abort path. Genuine readback
failures still stop execution; no freshness, settling or release guard was removed.
Per-resource/method readback timings and error classes support further diagnosis.

Never restart all four to recover just Green. Preserve completed trials. A held
ball uses the guarded stationary/held-ball recovery with its original trial ID;
an empty failed approach uses a single selected pickup after fresh confirmation.
No automatic retry or restart-resume is implemented. See the operator guide.

## Code and contracts to inspect

- `sircus/physical_trials/controller.py`: serialized rounds, pickup/release guards,
  round journal, held-ball recovery.
- `sircus/physical_trials/viam_adapter.py`: owned native motion, vacuum readback,
  cancellation, per-RPC diagnostics. Round transitions do not actuate suction.
- `sircus/physical_trials/mock_adapter.py`: isolated fake hardware.
- `sircus/simulation/web/show.html`, `physical.js`, `show.css`: operator panel.
- `docs/ROUND_OPERATOR_UI.md`: exact operating and validation boundaries.
- Existing learning/scoring guidance in PROJECT_STATE.md: preserve versioned
  evaluators, scene fingerprints, historical results and replay compatibility.

Good independent Ubuntu work: reproduce the frozen-checkpoint numerical learning
study, improve simulation-only policy/replay adapters, or add fault-injection
coverage. Do not convert fixed-scene optimization into claims of generalization.
Numerical trajectories, endpoint IK and native collision-checked arm paths are
separate evidence. Hardware calibration and perception remain future work; the
current manual routine needs no camera or bucket localization.

## Manual UI quick reference

Open Manual operation → **Run the complete act**. Empty dropper must start at a
taught round drop pose and board at a taught board pose. Load all four marks,
confirm clear paths and board/bucket setup, choose sounds, then **Run Round 1**.
After completion, reload all four and confirm again before **Reloaded · run Round
2**. Keep the browser open: closing it does not stop the backend. **Stop demo
motion** cancels owned motion without opening a gripper or releasing suction.

The two round buttons use paired host-only round presets, not merely the global
load/drop editor. Editing a single drop pose does not reteach the paired rounds.
Local preset values must not be copied into Git. Source sync does not supply a
calibrated hardware setup to another computer.

## Presentation

The existing native deck was updated in place; see
[PRESENTATION_STATUS.md](PRESENTATION_STATUS.md) for its current narrative and
validation boundaries. Theme and apparatus photographs were preserved.
