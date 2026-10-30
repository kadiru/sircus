# Project state

## Guided operator and training workspace — October 30, 2026

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
