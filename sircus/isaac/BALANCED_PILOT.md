# Solver fix and short distance-balanced PPO pilot

This opt-in mode fixes endpoint IK, verifies the board's **actual simulator pose**,
and tests a short learning update before authorizing any longer study. It uses
only the RTX4090. No hardware, Viam service or quasi-perception pipeline changes.

## Solver correction

The board task constrains flange position and its facing normal, leaving spin
about that normal free. On this UF850 chain, joint6 changes only that spin. The
old solver optimized six variables for a task with five effective variables,
using numerical derivatives and a small stopping budget. This could stall or
stop early even for feasible targets.

`robust_ik.py` preserves the old attempts first. If needed, it uses an analytic
Jacobian, removes the verified zero-effect last variable, uses stricter numerical
convergence thresholds and larger bounded budgets, then deterministic alternative
starts. The **acceptance norm stays below1e-4**; joint limits are unchanged. The
dropper retains all six pose constraints. No endpoint result proves collision-safe
travel. Unresolved targets remain rejected, not silently clamped or moved.

The offline historical audit exercises all62 board failures plus16 valid controls.
This is regression evidence, not a newly held-out task-performance score.

## Measured settling and release gate

`SettledShot` extends the existing simulated shot controller. Gravity remains on,
with model-based PhysX gravity compensation applied as feed-forward joint torques
on both arms. Original drive stiffness, damping, effort limits, geometry and
contact materials remain unchanged. This is ideal simulator model information,
not a demonstrated physical torque controller or a camera measurement.

After endpoint initialization and120 physics ticks, measure the board center,
normal and arm joint velocities, then observe a60-tick stability window (250ms at
240Hz). Release requires all of:

- Board center within2mm and normal within0.5° of the requested target.
- Both arms' maximum joint speed ≤0.005rad/s.
- Board movement ≤0.1mm and normal change ≤0.05° during the stability window.

Up to three bounded feedback corrections can change arm14's position targets,
with120 ticks between corrections. No joint-state teleport occurs during
correction. A failed gate parks the ball and returns an invalid shot before any
release-state write. A mock test executes the real shot method and checks this.
Gravity compensation is active throughout the subsequent bounce as well.

Twelve development probes cover four scenes in each distance band. At least
three IK-valid poses per band must pass, with no unsettled valid pose. A legitimate
endpoint rejection does not count as verified settling and remains explicitly
reported. It is not excluded from study outcome denominators. The same checks
apply to every training and evaluation throw.

Earlier development passes without compensation are retained as unsuccessful
controller experiments; none are training results. They showed slow drift and
why checking a single near-target pose was insufficient.

## Learning and paired comparison

- **384 additional PPO throws**, fine-tuning the previously selected100-layout
  policy. The old reference networks remain frozen.
-96 training layouts:32 per near/medium/far band. Each48-throw batch samples16
  distinct layouts from each band. Eight optimizer batches total.
-18 separate validation layouts (six per band); validate at0,192 and384 throws.
  Select by captures, then mean reward. The initial actor remains eligible, so
  an unsuccessful update is not promoted as a better policy.
-36 fresh test layouts (12 per band), evaluated once after checkpoint selection
  for the selected pilot, both historical policies and a fixed warm action.
- Fresh split seeds:161101/171101/181101 for train/validation/test; development
  seed191101. IDs and geometry are disjoint. Training RNG seed43.
- Same11 normalized privileged geometry inputs and same three outputs: board
  tilt, heading offset from bucket-facing, and release gap. No per-scene action
  search or grid labels are used. No point clouds enter the policy yet.
- Same contextual one-step PPO loss, Adam0.0007, four epochs/batch, clip0.2,
  entropy0.001, value coefficient0.5, gradient norm limit1, terminal reward/15.
  Exploration log-standard-deviation starts at−0.7 and is bounded[−2.8,−0.2].
- Success still requires board contact/rebound, whole-ball entry and one
  continuous second inside. The rebound direction uses the measured pre-release
  normal. IK/gate failures receive−2 and remain in outcome counts.

All four policies are compared under the **same new solver/settling contract**.
These are explicit transfer evaluations of the legacy weights; do not confuse
those scores with their historical256-scene scores. The new fingerprint records
controller, settling, split, exploration and scoring-normal changes. Legacy
fingerprints are checked before allowing this explicit adapter. No compatibility
guard is removed from other replay paths.

One short run and12 test scenes per distance cannot establish robust improvement.
Report selected versus latest weights, actor changes, paired wins/losses,
per-distance captures, invalid causes and action ranges. Preserve the historical
results and all development failures. Increasing budget is a separate decision.

## Launch and output

Use the pinned Isaac Sim5.1 / Lab2.3.2 runtime and generated UF850 assets:

```bash
export SIRCUS_ISAAC_SOURCE=/workspace/studies/YOUR_IMMUTABLE_SNAPSHOT
# With the existing scene lock, only one active simulator in this workspace:
python.sh launch_scene.py --headless --enable_cameras --device cuda:0 \
  --balanced-pilot --pilot-episodes 384 \
  --reference-policy RUN_100/selected.pt \
  --reference-policy RUN_1000/selected.pt
```

Add `--pilot-smoke` for development probes only. Pilot budgets must be multiples
of48 within48–768. The executed job additionally uses a90-minute timeout and3GiB
free-space floor. The 4070 is not involved.

`balanced-pilot-UTC/` contains immutable source hashes, contract, splits,
pre-release pose records, every training trial, validation results, latest and
selected checkpoints, and paired tests. These runtime files stay outside Git.
`pilot_report.py` rejects incomplete/unbalanced/mismatched evidence before
exporting aggregates and checksums. The read-only interface's `#balanced-pilot`
panel displays progress, validation selection and the paired comparison; old
studies remain below it. Sensor imagery is stale unless separately refreshed.

Completed results: [paired pilot comparison](../../baselines/isaac-balanced-pilot-20261031/README.md). The solver repair passed; the short policy update is not promoted.
