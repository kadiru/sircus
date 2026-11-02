# mjlab learning and coverage

The task is cooperative **shot configuration selection**. One arm holds a board;
the other releases a ball vertically above it. The board should redirect the
ball into a tapered bucket. This task currently uses simulator geometry as its
observation. Point-cloud perception, pickup, grasping, collision-free transit,
and moving-board interception are separate tasks.

## What coverage means

The continuous combination of bucket position, board position/orientation,
release position, and joint configuration cannot be exhaustively tested. A study
must declare a finite tabletop region, bucket grid, action bounds, seeds, and
arm-position constraints. Report attempted cells, cells with a valid IK endpoint,
cells with an observed success, and repeated-shot reliability separately.

An endpoint search failure means **unresolved by this search**, not physically
impossible. A layout outside a conservative reach bound can be labelled outside
that bound; it still does not prove that every possible bouncing strategy fails.
Buckets can be outside the dropper's direct reach while remaining reachable by
a board bounce.

## Learning and evaluation boundary

The implemented study uses a 7 × 5 training grid, 20 validation coordinates and
50 distinct test coordinates within X = 100–1240 mm and Y = −650–250 mm. The
continuous test positions measure interpolation within the same region, not
performance in disjoint spatial cells or across the entire physical table.

The actor and critic each have two 128-unit tanh hidden layers. Ten inputs encode
bucket XY, two positional-lock masks and the two optional XYZ values. Five
continuous outputs choose board X/Y/height, tilt from horizontal and vertical
drop gap. Board yaw faces the bucket and release XY equals board-center XY. This
center-impact family does not cover arbitrary surface contact points, every board
orientation, or every arm joint configuration.

Training starts with native-simulation exploration using geometric/ballistic
proposals, followed by independently perturbed confirmation of the best training
settings. A return-weighted imitation step fits one confirmed teacher per grid
position. Actual on-policy PPO then updates the Gaussian actor and critic from
fresh simulated throws. Imitation warm-start results are reported separately so
that any gain from PPO can be assessed rather than assumed. Validation mean
shaped reward selects the checkpoint; the test set never selects it.

PPO uses Adam at 0.0003, six epochs per batch, ratio clipping at 0.2, value weight
0.5, latent entropy weight 0.001 and gradient-norm clipping at 1.0. Each throw is
a terminal one-decision episode, so returns are observed rewards without GAE or
bootstrapping. Frozen evaluation uses the actor mean, not sampled actions.

The declared training reward is −1 for rejected endpoints. Otherwise it adds
0.15 for a board rebound and 0.5 exp(−closest opening distance / 0.12 m). Detected
external contacts subtract 0.5 and suppress all capture/retention bonuses. Direct
routes can earn up to 0.5 for retention, 3 for capture, 0.3 for clean entry and
0.1 for exactly one board contact episode. These shaping terms do not alter the
historical terminal scoring contract.

The finite wrist-distance proposal prior rejects wasteful seeds before full IK.
It is not a reachability certificate: actual joint limits, independent native
FK, endpoint collision tests and measured collider transforms remain necessary.
The gripper proxies are initialized open (25 mm on each finger) and the ball is
initialized at its release center. This is not a trained grasp-and-release.

Each episode contains one configuration decision followed by an independently
reset native mjlab/MuJoCo-Warp ball simulation. The arm endpoint IK solution is
validated against joint limits and forward kinematics. Initializing or holding
those endpoints does not train the arm transit or grasping trajectory.

A neural policy must be updated from observed native simulation rewards, saved,
and evaluated while frozen. Geometry-only feasible-action proposals and analytic
ballistic priors are allowed, but their restrictions must be reported. If the
policy chooses among a finite candidate set, that is candidate selection rather
than unrestricted continuous arm control.

Training layouts drive optimization. Validation layouts select a checkpoint and
hyperparameters. Test layouts remain untouched until the selected checkpoint is
frozen. Exact split seeds, coordinates, and checkpoint hashes belong in the study
contract/evidence. Held-out nearby jitter and held-out spatial cells answer
different questions; label each accordingly.

Per-scene rollout search or refinement is an **oracle/search baseline**, not
policy inference. The interface must distinguish a frozen policy proposal from
any later simulation-assisted search. A failed policy proposal must remain a
failure in policy evaluation even if an optional search subsequently finds a
successful shot.

## Success and reward

The unchanged historical capture rule requires an observed board rebound, whole
ball entry using the actual polygonal tapered interior, and one continuous second
inside. Partial exits reset retention. Rim-assisted captures count. The additive
`sircus-board-route-v1` rule rejects any detected table, floor, other-robot, or
unattributed ball contact. Clean entry requires no pre-entry/rim-band contact,
no later partial exit, and at least 1 mm mouth clearance. Exactly one board
contact episode is a separate metric.

`learning_score.BatchScore` is a batched Torch transcription of the existing
`ScoreStream`, not a new success rule. It samples contact-force maxima over the
inner physics substeps at the original 240 Hz scoring cadence. It freezes at the
first historical capture or ball-center Z below −0.2 m. Scalar/batched tests cover
retention boundaries, rim assistance, external contacts, multiple board contacts,
polygon wall geometry, and terminal freezing. Float64 counters preserve the
one-second retention boundary.

A training reward may add dense proximity, entry, or retention shaping, but
captures through disallowed external routes must not receive a capture bonus.
Document the reward separately from terminal success. Reward improvements alone
do not establish capture-rate improvements.

## Interface constraints

Locking an arm's position fixes its declared Cartesian endpoint, not every joint
angle. An orientation or joint lock is a different constraint. Validate the lock
after IK/FK and make its frame and tolerances visible. Contradictory constraints
should yield an explanatory unresolved/infeasible response rather than silently
moving a locked endpoint or resetting the bucket.

Before each replay, verify the actual bucket collision transforms (the bucket is
a mocap body), arm endpoints, and board normal. A displayed requested pose does
not establish that the physical simulation geometry is there. Preserve the
reported 175 mm bucket-height uncertainty and provisional tool/board transforms.

The manual placement interface is gated on a completed study/report. Mock UI
tests may run while training is underway. Do not start its human/manual tests
before the learning results have been reviewed.

Locked evaluation is conditional: the board/release XYZ anchors come from the
same frozen free-policy endpoints at held-out bucket coordinates. It tests lock
handling and conditional self-consistency, not arbitrary XYZ-lock reliability.
Repeated nominal throws measure numerical repeatability; the training-catalogue
perturbations are a separate development robustness check.

## Commands and local interface

The completed categorical follow-up uses256 relative proposals with geometry-only
endpoint masks,99/20/50 fresh train/validation/test layouts and2,048 further native
PPO attempts. It samples78 eligible training contexts; empty-mask test cases stay
in the failure denominator. A shared9→64→64→1 actor scores each candidate and a
bucket-XY critic estimates return. Validation captures, then reward, select the
checkpoint. The [audited comparison](../../baselines/mjlab-learning-20261102/README.md)
scored62.0% learned versus62.7% uniform feasible-bank selection: this does not
demonstrate a PPO advantage over the geometric baseline.

Run a new categorical study with `python -m sircus.mjlab.bank_study --parent-study
COMPLETED_CONTINUOUS_STUDY --assets ASSETS --gpu-lock LOCK --output NEW_STUDY
--batch 128 --updates 16 --test-repeats 3`, using the pinned environment and only
the4090 UUID. `verify_bank.py` audits masks, candidate IDs, saved behavior log
probabilities, checkpoints, frozen choices and complete test denominators on CPU.
Cached audited joint branches prevent replay from silently choosing another IK
solution. Keep original executed source bytes, including formatting, immutable.

For the two-model interface, set `SIRCUS_MJLAB_BANK_STUDY` as well as the parent
study variables below. After both auditors pass, run `prepare_interface` with
`--parent-study`, `--bank-study` and a new `--output` report; point the web server
at that report. Free requests use the bank selector; positional-lock requests use
the continuous model. Explicit native search is separate and always receives a
fresh verification throw. Original reports and policy failures are never replaced.

Use the already isolated mjlab environment, not Isaac Sim's Python. From a source
checkout containing the shared scene and scorer files:

```bash
CUDA_VISIBLE_DEVICES=YOUR_4090_GPU_UUID WANDB_MODE=disabled \
  "$HOME/sircus-mjlab/venv/bin/python" -m sircus.mjlab.learning_study \
  --benchmark baselines/mjlab-comparison-20261102/benchmark.json \
  --assets "$HOME/sircus-isaac/assets" \
  --gpu-lock "$HOME/sircus-isaac/scene.lock" \
  --output "$HOME/sircus-mjlab/output/NEW_UNIQUE_STUDY" \
  --batch 128 --explore 32 --updates 32 --test-repeats 3
```

The existing shared GPU lock serializes native studies. The runner refuses an
occupied 4090 and never stops another workload. A new output directory is
required; old studies are not overwritten. A report is complete only after all
frozen comparisons finish. Raw JSONL, trajectories, generated models, runtime
paths and virtual environments remain outside Git.

After completion and artifact verification, configure only the local backend:

```bash
export SIRCUS_MJLAB_STUDY=/absolute/path/to/completed/study
export SIRCUS_MJLAB_ASSETS="$HOME/sircus-isaac/assets"
export SIRCUS_MJLAB_GPU_LOCK="$HOME/sircus-isaac/scene.lock"
"$HOME/sircus-mjlab/venv/bin/python" -m sircus.mjlab.interface \
  --study-report "$SIRCUS_MJLAB_STUDY/report.json" \
  --solver sircus.mjlab.learning_interface:solve --port 18781
```

An immutable artifact-provenance file binds source, scorer, model assets and
checkpoint. Each explicit solve launches a finite child process, acquires the
GPU lock, selects only the 4090 and releases GPU memory when it exits. The web
server does not occupy a GPU between solves. The child timeout terminates its
process group; other simulation jobs remain untouched. Use an SSH localhost
forward to view this new interface from the MacBook. See [UI/API details](web/README.md).

## Compute isolation

Use the existing pinned mjlab environment and explicit RTX 4090 UUID selection.
The runner must reject an unexpected GPU. Serialize GPU work with the agreed
host lock and inspect actual GPU processes before launch. Do not stop, reset, or
reconfigure the 4070 workload or the independent Isaac study. Headless source
tests use CPU with both GPUs hidden. Avoid duplicating environments or large
assets on the nearly full host disk.

## What the earlier comparison video showed

The paired video used the **same recorded joint endpoints and ball releases** in
Isaac PhysX and mjlab. Each engine independently simulated ball contacts and the
same scorer evaluated the trajectories. A common MuJoCo renderer displayed both
traces. It compared contact behavior at held arm endpoints, not learned policies,
RL convergence, arbitrary-table generalization, or the original Isaac viewport.
The physics difference, especially around the rim, is precisely why a policy
must be trained and evaluated again in mjlab.
