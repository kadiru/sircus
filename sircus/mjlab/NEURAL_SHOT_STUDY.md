# Learn to choose promising shots before solving the arms

This study aims to make **place bucket → plan → drop** faster and more reliable.
The last matched comparison captured 40/50 with the current planner and 42/50
with wider search. Wider search increased median planning time from 33.7 to
39.5 seconds and took up to 383 seconds. Those are historical results;
[the new study](neural_shot_study.py) must earn its own scores.

## What is being learned

A small neural network ranks proposed shots **before** expensive arm-pose
checks. A shot specifies board-center X/Y, board height, tilt, release gap,
heading, board twist, and an intended direct or table-bounce route. Saved
source joint angles may supply an IK starting hint; they are not the solved
joint angles for the new target.

The network has 44 input features, hidden layers of 64 and 32 units, and two
outputs: estimated chance of a feasible setup, and estimated chance of capture
given a feasible setup. Their product orders candidates. Features encode the
bucket and proposed shot geometry, relative distance/direction, route intent,
and optional source joint hints. They exclude solved query joints, measured
ball trajectories, collision outcomes, and future test results.

Training is **supervised learning**, not PPO or another reinforcement-learning
run. Simulated attempts provide labels; Adam minimizes two binary prediction
losses. Each bucket position contributes equal total weight to each output's
loss, so positions with more attempted shots cannot dominate. Training runs
on one CPU thread; the 4090 supplies the native simulated drops.

- Setup refusals train only the feasibility output.
- Admitted proposals without a release also supply setup evidence only.
- Valid released misses and captures train the capture output.
- Invalid physics never becomes a physical-miss training label.
- Direct and table-assisted captures count. Robot-assisted captures remain
  separately recorded and are not standard captures in this comparison.

The 100-, 250-, and 500-epoch checkpoints are compared using position-balanced
loss on validation data. Validation labels do not enter gradient updates.
The selected weights are exported as a plain JSON checkpoint and frozen.

## Data collection and independent splits

The runner freezes **160 training, 40 validation, and 200 final test bucket
positions** before collecting outcomes. Positions span the usable table
interior, edges, and regions near fixed robot bases. Bucket footprint/base
geometry gates still apply. New positions are at least 5 mm from inventoried
historical positions and one another. This is separation between positions,
not a claim of testing entirely new physical layouts or all possible targets.

At each training or validation position, the proposal menu includes up to
eight existing model proposals, 32 transferred shot geometries, and 16
bounded height/tilt/gap variations. Duplicate geometry is removed. Exact
endpoint, collision, and sampled path checks decide which candidates can
release. Of the admitted shots, four with strong predicted flight scores are
kept, then a deterministic shuffle selects diverse remaining candidates.

| Split | Positions | Native release cap per position | Maximum releases |
|---|---:|---:|---:|
| Training | 160 | 36 | 5,760 |
| Validation | 40 | 16 | 640 |
| Final test | 200 | 1 per planner | 600 across three planners |

These are **caps, not completed counts**. Fewer feasible candidates produce
fewer drops. Every original proposal, setup refusal, admission, selected
release, measured flight, and outcome is preserved with its position and ID.

## How the learned selector plans one test shot

1. Build the same finite geometry menu without IK or native releases.
2. Rank it with the frozen neural network.
3. Check the first four proposals using the existing exact arm/path checks.
4. If no feasible shot has positive predicted flight clearance, check the
   next four, up to 16 original proposals total.
5. If clearance remains poor, try alternative board-arm joint branches for
   up to four promising checked geometries. Keep the dropper joints and
   board/release targets fixed during that branch search.
6. Use the existing physics flight score to select among checked, admitted
   candidates. Release at most one ball. No test-drop outcome guides selection.

The neural network prioritizes computation; it does not authorize a pose or
replace the existing physics ranking and admission checks. The current
planner and wider-search planner retain their existing rules as baselines.
Each planner is timed during independent preparation, so the learned method
cannot inherit the baseline's prepared arm solutions.

## Frozen comparison and evidence

All three planners receive the same 200 untouched positions and one drop per
position at **64 substeps**. The checkpoint, source versions, layouts, candidate
plans, and selected commands are bound by hashes before final releases.
No final test data trains the model or changes its settings.

Capture requires a measured board rebound, whole-ball entry, and one
continuous second inside the bucket. Rim assistance and a table bounce are
allowed. The report gives first-drop captures over all 200 positions, feasible
setups, released misses, robot contacts, and median/90th-percentile planning
time. Setup refusals and invalid physics remain in the test denominator.

Matched simulator controls run around each native stage. An independent
CPU audit checks evidence, frozen choices, split separation, exact execution,
scoring, and accounting. Final UI scores remain hidden until that audit passes.
There is no automatic promotion to the live planner.

## Execution and limits

The runner is `python -m sircus.mjlab.neural_shot_study`; use `--help` for the
required audited archive, scene, asset, and output paths. It extends the input
chain in [CLEARANCE_EXPANSION.md](CLEARANCE_EXPANSION.md) with the completed
clearance-expansion study and audit. Use one fresh output directory and retain
it throughout these ordered stages:

1. `--stage initialize`: freeze split layouts, source manifest, and inputs.
2. `--stage prepare-collection`: prepare training and validation on CPU.
3. `--stage run-collection`: acquire the shared 4090 lease, check controls,
   collect admitted native drops, and check controls again.
4. `--stage fit`: build labeled datasets and freeze the selected model.
5. `--stage prepare-baseline`: prepare both existing planners on test positions.
6. `--stage prepare-neural`: independently prepare the learned selector.
7. `--stage bind-test`: freeze all final test choices and model/data bindings.
8. `--stage run-test`: acquire the lease and run the matched one-drop comparison.
9. `--stage finalize`: summarize results and freeze the evidence inventory.
10. `--stage audit --audit-output <fresh-path-outside-study>`: verify evidence
    before publishing final results.

CPU preparation uses four low-priority workers with CUDA hidden. Native runs
select only the 4090 UUID, use the shared GPU lock, and wait rather than
interrupt another workload. Collection has a four-hour leased-GPU cap; testing
has a three-hour cap. Together they cannot exceed seven leased GPU hours.
CPU preparation and waiting for a busy GPU are separate from that budget.
The 4070 and Isaac setup are not modified. Storage is checked during work;
raw evidence, logs, generated configs, and runtime checkpoints stay outside Git.
A partially started native stage must not be silently rerun as new attempts.

`--display-report` updates only the additive `neural_shot_study` section of the
existing derived report. The UI anchor is `#neuralShotStudy`. It distinguishes
prepared positions, planned releases, and completed native drops, and does not
change the Plan shot button or restart its service.

## Perception comes after a dependable planner

The current inputs are simulator geometry. There is no camera or point-cloud
input to this model. Earlier Isaac RGB-D work detected the table, ball, board,
and bucket in simple scenes, but its randomized validation exposed substantial
lighting/depth-noise failures; it is not a ready replacement for known geometry.
See [Isaac perception validation](../isaac/PERCEPTION_VALIDATION.md).

A later phase can produce object estimates from rendered depth/point clouds,
first with simulator-assisted object cropping, then with independent detection.
Keep ground truth separate for scoring and error measurement, and preserve
missing/occluded detections instead of silently substituting true positions.
Testing the same frozen shot planner with estimated versus known geometry will
separate perception errors from planning errors.

This study does not add sensor noise or 128-substep comparisons. It uses the
accepted fixed-64 contract. Native shots hold both arms at selected endpoints;
CPU path checks are not a demonstration of dynamically executed arm motion,
physical grasping, or real-world deployment.
