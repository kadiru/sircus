# Guided operator and training workspace

This source checkpoint provides a separate **simulation-only** page in the existing
Sircus dashboard. It cannot select a physical adapter, connect to a Viam
machine, command an arm/tool, or deploy a policy. Its server requires both
`--offline` and `--physical-mode disabled`; HTTP handlers enforce the same boundary.

## Start a local session

Use Python 3.12 and the pinned `sircus/simulation/requirements-viam.txt` environment.
On Ubuntu, run `./setup-ubuntu.sh` once in this checkout. Do not copy the Mac venv.

```bash
./start-operator-sim.sh
# Open http://127.0.0.1:18766/operations.html
```

If using an already verified Python environment, set `SIRCUS_PYTHON` to its Python
executable. `SIRCUS_OPERATOR_PORT` selects a different unused loopback port. A port
conflict is an error, not permission to kill an existing service. The launcher
starts only this new local process and never a Viam cloud part. Existing simulation,
Ghost Booth and physical dashboard services are independent. Output stays under
ignored `.runtime/operator-lab/`; no credentials are needed.

## Guided trial

1. Position arm14's board in simulation. Its impact plane is reconstructed from
   simulated joint kinematics and the provisional suction/mount definition. No
   board vision is used. Input pose changes invalidate downstream work.
2. Place the bucket. Its measured inner177/150mm and outer200/158mm diameters and
   unresolved reported175mm height are preserved. Bucket placement remains provisional.
3. Scan the bucket. Here this returns explicitly **synthetic** scene evidence.
   It does not claim that the real RGB-D detector or camera calibration is working.
4. Preview a plan. **Hold board pose** searches releaseU/V and height for that fixed
   pose. This is fixed-scene optimization. **Learned board angle** uses a frozen
   checkpoint to predict roll/pitch/U/V/height without per-scene optimization.
   The latter rejects layouts outside the checkpoint's declared distribution.
5. Confirm that the ball is loaded. This is an operator acknowledgment in rehearsal,
   not a sensor claim that a physical gripper holds the ball.
6. Grip & drop runs the versioned numerical shot and exposes its real trajectory
   for playback. Events retain trialID, scene revision, plan and synthetic provenance.
   Stop cancels the simulation operation; it never actuates either tool.

Changing geometry/checkpoint, reset or cancellation invalidates the relevant scan,
plan/loading authorization. A completed older job cannot revive canceled state.
Endpoint IK is checked; it does not validate collision-free paths or real motion.

## Training

The training tab launches the existing scene-conditioned trainer in an isolated,
bounded subprocess. It saves seeded disjoint train/validation/test layouts before
teacher searches, fits ridge-regression weights, selects regularization using
validation alone, freezes the checkpoint, and evaluates on held-out layouts.
Frozen inference, fixed-action baseline and per-scene search oracle are reported
separately. The default small run is a pipeline smoke, not a performance study.
Training progress, cancellation and artifact downloads operate on that real job.
Completed checkpoints can be loaded into the guided workspace.

Board center and bucket offset vary within the existing narrow study distribution;
roll/pitch/U/V/release gap are explicit actions. This is not general-purpose RL,
full robot dynamics or a calibrated sim-to-real guarantee. Board compliance and
contact coefficients are still provisional. The original published checkpoint and
historical rectangular/bucket results are not rewritten. See TRAINING_WORKSPACE.md.

## Required before tomorrow's physical operation

The intended physical sequence is: operator positions board and bucket → dropper
moves to a taught, collision-checked scan pose → stationary RGB-D bucket detection
and operator review → dropper moves to a taught loading pose → human loads ball →
explicit Grip & drop → close → move to validated release → open → observe outcome.
The board stays still throughout a fixed-board trial. Learned-angle operation needs
an explicit preview/confirmation of the board movement before loading/release.

Physical execution is deliberately unavailable until all of these are provided
and tested in a separate authorized integration:

- Verified arm bases/common world frame and arm14 joint readback; measured flange
  to suction/board impact-surface transform (including landscape90° mount).
  Knowing joints alone does not calibrate this transform or board flex.
- Camera serial/intrinsics, metric depth scale, RGB/depth registration and calibrated
  camera-to-arm10 transform. Current stationary observations do not establish them.
- A real bucket detector with top-opening center/normal, timestamps, frameID,
  uncertainty and calibrated transformation to the common world. Reject occlusion,
  missing depth, ambiguous detections and stale scans; moving bucket invalidates scan.
- Taught scan/loading poses, measured ball center relative to gripper, release
  latency and height limits. Ball insertion still needs operator clearance.
- Collision-checked paths for both arms, board, gripper, bucket, table and walls;
  a matching v2 native replay adapter. Existing compatibility guards must remain.
- A small physical calibration/validation set to measure bounce response versus
  contact location, board angle and release height. Simulation success alone cannot
  establish tomorrow's physical success; fit the model before expanding trials.

The physical adapter must also record pre-release camera/state evidence and actual
release pose, and must stop motion without opening grippers or releasing suction.
Reuse the existing physical controller and stationary capture components rather
than bypassing them. No hardware motion is authorized by running this workspace.
