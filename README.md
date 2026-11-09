# Sircus — Simple Circus

**Latest mjlab training:** [seven-hour-budget, 70-position comparison](baselines/mjlab-coverage-training-20261109/README.md).
All 2,757 planned requests completed in 235.1 worker minutes. New imitation and
selected new PPO each captured **28/70 (40%)** on unseen bucket positions;
PPO gained one capture and lost one, with no net improvement. Previous PPO
weights captured 31/70 under the same expanded prior. The independent audit
and publication follow-up are recorded with the compact results. No policy
was promoted; the 4070 and Isaac setup remain unchanged.


**Latest mjlab experiment:** [Case 00 tilt/drop-gap sweep](baselines/mjlab-case00-tilt-gap-20261109/README.md).
Of 70 fixed-grid proposals, 42 passed setup admission and two captured:
**26° and 32° tilt, both at 251.819 mm drop gap and 460 mm board height**.
Each captured **3/3 fresh nominal repeats**, with rim assistance and no robot
contact. The unchanged 28° setting captured 0/3. All 18 controls and
27,547 independent checks passed. The `#case00TiltGapStudy` UI contains the
complete matrix and saved-trajectory before/after replay. This is a local
fixed-scene search, not a newly trained policy or a proven robust window.
No PPO, 4070 workload or Isaac changes.

**Earlier mjlab experiment:** [four-case height/tilt admission and drop results](baselines/mjlab-table-height-tilt-20261108/README.md).
The fixed 120-candidate grid found **58 admitted setups**, but their native64
drops produced **0 captures**: all bounced the board, none entered the bucket,
and 42 contacted a robot. The 12 unchanged original repeats also missed.
All 18 controls and 36,371 independent checks passed. Exact settings, all
declines and separate flight outcomes are in `#tableHeightTiltStudy`.
No capture qualified for fresh repeats. Next: investigate ball-path clearance
and admitted shot geometry before more PPO. No policy, 4070 or Isaac changes.

**Earlier mjlab diagnosis:** [31 flight misses, four setup declines and the tilt-only test](baselines/mjlab-table-training-diagnosis-20261108/README.md).
The 42-request correction test completed with controls and independent audit:
original shots captured **0/12**; changing only tilt to 32° captured **0/12**
because all 12 changed requests failed board-arm IK admission. No corrected
flight was executed, so this did not demonstrate improved contact behavior.
Recorded trajectories and setup proof details are in `#tableTrainingDiagnosis`.
No new training or policy activation; the 4070 and Isaac remain untouched.

**Latest completed mjlab training:** [table-assisted 50-position comparison](baselines/mjlab-table-assisted-training-20261108/README.md).
On the same fresh bucket positions at fixed 64 substeps, old imitation captured
**9/50**, old PPO **8/50**, new imitation **14/50**, and new PPO **15/50**.
The complete 766-request study passed its independent audit; both new policies
made six captures with table contact. Most of the observed gain is already in
new imitation; PPO adds only one capture. This is one seed and one shot per
position, with different training exposure from the old models. No policy was
promoted. Final counts and all 50 matched outcomes are in `#tableTraining`.

**Table-assisted task definition:** [version 3 scoring and reward](sircus/mjlab/SCORING_TABLE_ASSISTED.md)
accept board → table → bucket captures after one continuous second inside.
Historical v2 labels remain intact; CPU rescoring is separate from policy
training, and no new policy is activated by this change.
[Completed saved-shot audit](baselines/mjlab-table-assisted-scoring-20261108/README.md):
8,037 native trajectories; accepted captures **4,409 → 4,532** (+123).

**Earlier completed selector comparison:** [arm-clearance results](baselines/mjlab-arm-clearance-20261108/README.md)
and [frozen protocol](sircus/mjlab/ARM_CLEARANCE.md). On 50 new paired bucket
positions at fixed64, captures fell from **20/50 to 16/50**. Robot-contact trials
fell from 12/44 releases to 5/32, largely because the gate withheld 12 shots.
All 112 requested slots and 12 controls passed a 1,461,935-check audit. The
current interactive selector remains unchanged; the candidate is not promoted.

**Next bounded selector comparison:** [frozen velocity + aperture ranking](sircus/mjlab/CONSISTENT_FLIGHT.md),
30 fresh bucket positions, four selectors, 132 total requests including controls.
No new fit or scoring change; queued/native results are separately audited.

**Ensemble flight diagnosis:** [concrete recorded misses and prediction errors](baselines/mjlab-ensemble-flight-diagnosis-20261106/README.md).
Only 4/13 ensemble requests scored at least 99% captured. All 25 native misses
failed whole-ball entry; the UI separates observed flights, independent model
heads and conditional ballistic diagnostics. Case 36 changed only drop gap by
36 mm relative to a successful BC shot and missed. CPU analysis only: no new
training or native shots; the occupied 4090, Isaac and 4070 were left untouched.

**Physics-ensemble comparison completed:** [checkpoint, results and audit](baselines/mjlab-physics-ensemble-20261106/README.md).
All 264 fixed64 requests completed; the independent audit passed 897,304 checks.
On the same 60 fresh TEST bucket positions, BC captured 23/60, PPO 26/60 and
ensemble-ranked selection 24/60. The ensemble reduced setup rejections but did
not demonstrate a capture improvement over the saved policies. All forecasts
were frozen before evaluation; its extra candidate/geometry budget remains
explicit. No all-table, 128-substep or physical validation is claimed.

**Completed PPO diagnosis:** [paired setup/flight evidence and post-hoc interventions](baselines/mjlab-policy64-diagnosis-20261106/README.md).
Five of seven lost imitation captures became setup rejects. Restoring board height
recovered all five in one fresh test each; the two native flight losses repeated.
The 96-slot fixed64 diagnosis passed its independent audit. See `#policy64Diagnosis`
for exact configurations and recorded flights. No new model or WAM was trained.

**Completed fixed-64 learning study:** [Results and trained checkpoints](baselines/mjlab-policy64-20261105/README.md).
Historical TRAIN successes bootstrapped imitation, followed by 1,024 fresh PPO
requests. On 50 new test bucket positions: prior 19 captures, nearest success 18,
imitation 20, selected PPO 18. PPO did not improve capture rate. All 1,284 fresh
requests passed independent evidence checks; results and native flight examples
are in the UI's `#policy64Study` panel. Earlier 128 failures remain intact.

**Latest mjlab diagnostic:** the [bounded impact-point study](baselines/mjlab-impact650-20261105/README.md) completed 189 requests / 146 native releases in 9.1 worker-minutes. Five selected settings captured 15/15 at 64 and 0/15 at 128; none qualified. Controls and all 54,829 independent audit checks passed. No new PPO or noise run was started.

**Existing policy benchmark, historical definitions:** [completed matched sweep/PPO benchmark](baselines/mjlab-sweep-rl-benchmark-20261103/README.md). Frozen formal test scores are sweep **67/216 (31.0%)**, PPO **58/216 (26.9%)**. The 1 µm wall-overlap rule is contact-sensitive, so these labels do not establish an ordinary capture or adaptation winner. The earlier [35-position planner](baselines/mjlab-grid35-expanded-20261103/README.md) retains 17 selected recipes, 12 unresolved targets and 6 clearance blockers; its historical definitions/results remain unchanged.

Two UFACTORY850 arms learn to bounce a ball from a suction-mounted board into a bucket. Arm10 picks/releases; arm14 positions the board. This private repository is the shared source for Mac physical work and Ubuntu simulation work.

The current goal is a simulation showcase with an interactively placed bucket and optional constraints on either arm. [The mjlab/Isaac comparison](baselines/mjlab-comparison-20261102/README.md) establishes the second engine's two-arm geometry and shared scoring.

![Architecture](docs/sircus-architecture.png)

## Start here

- **Bounded impact-point study:** [protocol and 189-request plan](sircus/mjlab/IMPACT650.md).
  Three fixed source families, real off-center releases, matched 64/128 checks.
  Board centers stay 650 mm from buckets; nominal impact flight distance varies.
  [Completed audited results](baselines/mjlab-impact650-20261105/README.md): 135 coarse
  requests / 92 native releases, five rim captures and zero clean. All five selected
  settings captured 3/3 at 64 and 0/3 at 128. Both control gates and 54,829 independent
  checks passed; 189 total requests and 544.3 worker-seconds. No candidate qualified.

- **Remaining 30° verification:** [bounded 54-request protocol](sircus/mjlab/TILT650.md).
  [Completed audited results](baselines/mjlab-tilt30-650-20261105/README.md): all five
  exact recipes captured 3/3 at 64 and 0/3 at 128; no recipe qualified. All 54
  native releases and 14,410 independent checks passed validity/accounting.
  Its subsequent bounded impact-point study is reported separately above.

- **Translated XY650 development search:** [bounded protocol](sircus/mjlab/XY650.md).
  Board and bucket move together at fixed 650 mm distance/bearing; fresh strict
  endpoints, first-flight-only ranking and matched 64/128 qualification. This
  follows three fixed-XY searches with no qualified clean 650 recipe. The
  [completed translated study](baselines/mjlab-xy650-20261105/README.md) found 24
  rim-assisted discovery captures but 0 clean; five selected settings failed at 128
  after 3/3 rim at 64. Corrected audit v2 passed 80,138 checks. No noise or new PPO.

- **Sweep versus RL benchmark:** [matched current-ball protocol](sircus/mjlab/SWEEP_RL_BENCHMARK.md).
  All six workers completed: 1,024 spent TRAIN requests per method/seed, 6,144 TRAIN
  + 864 validation + 432 test requests, using the same 17-recipe historical prior.
  The [checksummed results and six selected artifacts](baselines/mjlab-sweep-rl-benchmark-20261103/README.md)
  passed the corrected independent audit: 902,252 checks, 7,440 requests / 3,738
  native releases. Per-worker setup + TRAIN wall averaged 2,097 s for sweep and
  2,100 s for PPO; lifetime sampled VRAM was 762 MiB for both. These are
  instrumented implementation costs, not isolated algorithm timings. Strict
  containment remains under contact-boundary review; calibrate and version the
  scorer before an ordinary capture/adaptation comparison. Current read-only
  results: [18789](http://127.0.0.1:18789/#sweepRlBenchmark); earlier services remain separate.

- **35-table planner study:** [expanded sweeps and separately frozen repeats](sircus/mjlab/GRID_STUDY.md).
  Per-table settings, saved versus current qualification, and independent 3 + 30
  repeat counts appear in the study interface and [all-35 CSV](baselines/mjlab-grid35-expanded-20261103/per-table-settings.csv).
  Both native studies completed and passed their independent audits. Unresolved
  and blocked cells stay visible; no new policy or catalogue was deployed.

- **Sweep shot planner:** [place a bucket, search and replay a confirmed shot](sircus/mjlab/SWEEP_PLANNER.md).
  Independent of learned-policy readiness, with exact saved recipes for12 farm
  targets and corrected native search for new coordinates. Finite search can
  remain unresolved; arbitrary query support is not all-table success coverage.

- **Contact repeatability:** [gravity/time-step diagnosis, 1,708 throws, exact settings and collision-aware transit](baselines/mjlab-contact-transit-20261102/README.md).100 fresh perturbed requests capture at corrected 32/64 substeps; all are rim-assisted. Numerical diagnostics and geometric paths stay separate from learned-policy performance.

- **mjlab interactive shot study:** [bounded learning protocol](sircus/mjlab/LEARNING.md) and [drag-bucket/position-lock interface](sircus/mjlab/web/README.md). Frozen policy inference and native per-scene search are separate modes; the interface stays gated until the study and evaluation complete. Simulator-only endpoint configurations, without physical control or perception.

- [Compact dark-floor farm film](sircus/isaac/DARK_FARM_FILM.md) — one-minute native Isaac recording, driven arm transitions, four configurations at one front table, trimmed impacts and a 10× station cut. [Verified replay results](baselines/isaac-dark-farm-20261102/README.md).
- [35-table Isaac training film and front-row close-ups](sircus/isaac/CYLINDER_FILM.md) — camera flight, 7 × 5 bucket grid, four learned shot parameters, and reproduction commands.
- **Workspace feasibility:** [32-target board/release-position search](sircus/isaac/FEASIBILITY_MAP.md), with [6,212-attempt results and exact settings](baselines/isaac-feasibility-map-20261101/README.md), direct-reach audit and fresh perturbation tests. Read-only `/feasibility.html` separates searched success from unresolved regions; this is not new RL training.
- **Native Isaac recording:** [direct application-window capture](sircus/isaac/NATIVE_VIEWPORT.md), showing the live 32-table / 64-arm farm in orthographic, elevated perspective or low close-up views without composite panels.
- **32-table PPO farm:** [actual training, video and reproduction](sircus/isaac/FARM_PPO.md). A shared neural policy selects board height, drop gap and tilt in 32 simultaneous Isaac Lab workcells. Recorded early/middle/late batches are training data; arm pickup and transit are not learned.
- **Isaac Lab / RTX Ubuntu:** [isolated two-arm RGB-D foundation](sircus/isaac/README.md) and [verified setup status](sircus/isaac/STATUS.md). Uses a dedicated GPU, separate environment and new physics contract. Privileged shot learning and perception diagnostics are available; full grasp/path learning remains separate.
- **Parallel Isaac world:** [32 workcells / 64 arms, fixed distinct targets](sircus/isaac/PARALLEL_WORLD.md), with [completed 1, 090-attempt experiment and exact settings](baselines/isaac-parallel-world-20261101/README.md). Live read-only world/video: `/parallel-world.html`. Six targets passed 10/10 fresh perturbations; farther feasibility remains unresolved. Parameter exploration, not a new RL policy.
- **Spatial bucket grid:** [nine-position settings and fresh confirmation](baselines/isaac-bucket-grid-20261101/README.md), with [contact-aware search protocol and RL compute examples](sircus/isaac/BUCKET_GRID.md). Three near cells passed 30/30 each; far feasibility remains unresolved. Development search, not a new learned policy.
- **Bucket distance study:** [method](sircus/isaac/DISTANCE_CATALOGUE.md) and [completed settings/results at 300–650 mm](baselines/isaac-distance-catalogue-20261101/README.md). Fresh confirmation is separate from search; farther captures have intermediate-rebound warnings and are not certified direct shots.
- **Dense shot catalogue:** [exact settings, clean/assisted scoring and repeated tests](sircus/isaac/SHOT_CATALOGUE.md). Fixed400 mm distance; nominal repeatability is separated from perturbation robustness.
- **Fixed-plane feasibility sweep:** [board height, drop gap and tilt at one bucket distance](sircus/isaac/PLANE_SWEEP.md). Simulation-only grid search with repeated selected shots; no cross-distance training.
- **Current short learning pilot:** [solver fix, measured board settling and balanced PPO comparison](sircus/isaac/BALANCED_PILOT.md). Synthetic controller validation and fresh paired policy evaluation before a longer run.
- **Isaac randomized learning study:** [methodology, budgets and privileged/cloud boundary](sircus/isaac/LEARNING_STUDY.md). Centralized PPO learns shot parameters on seeded layouts; full grasp/path control remains separate.
- **Isaac fixed-scene RL:** [cooperative orientation/height training](sircus/isaac/RL_FIXED_SCENE.md). A one-shot policy chooses board tilt/heading and release height in PhysX. Endpoint initialization is separate from future grasping and trajectory learning.
- **Ubuntu / Varun:** [setup, training and local dashboard](docs/UBUNTU_SETUP.md). Run `./setup-ubuntu.sh`, then `./start-local.sh`. In another terminal run `sircus/simulation/.venv/bin/python sircus/two-arm/local_dashboard.py`.
- **Orientation/height experiment:** `sircus/simulation/.venv/bin/python sircus/two-arm/train_orientation_height.py --budget 32 --tests 10 --output experiments/run-001`. No server or hardware required. The search holds release XY, board center/yaw and bucket fixed; it optimizes roll, pitch and gap with endpoint IK checks. V2 uses [one-second capture scoring](docs/SCORING_CONTRACT.md). Native replay rejects these policies until a matching runtime adapter is integrated; endpoint IK alone does not validate a motion path.
- **Scene-conditioned learning:** [study contract and commands](docs/SCENE_CONDITIONED_STUDY.md), [frozen checkpoint and synthetic results](baselines/scene-conditioned-20261030/README.md). A trained ridge model predicts roll/pitch/U/V/gap on disjoint layouts without search at inference. The opt-in dashboard panel previews numerical trajectories only; native v2 replay remains blocked.
- **Mac / physical sfhack10:** [operator panel instructions](sircus/physical_trials/README.md). Physical control is explicit and separate; cloning/pulling never starts hardware or reloads a running dashboard.
- **All contributors and agents:** [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md), and [current project summary](docs/PROJECT_STATE.md).

## Implementation audit and next experiment

Read the [Mac system audit, UI/API contracts, and learning integration plan](docs/audits/2026-10-29/README.md) before new development. These distinguish implemented capabilities from historical observations and proposed work; raw local evidence and credentials are excluded.

## Close-up simulation video

[Orthographic replay](sircus/isaac/OVERHEAD_REPLAY.md): four workcells and synchronized top/front close-ups at half and quarter speed. Includes reproduction commands and verification evidence; frozen settings, not a new learned policy.

## Source layout

| Directory | Purpose |
|---|---|
| `sircus/two-arm/` | Shared measured/provisional scene, two-arm physics, kinematics, Viam module, trainers and fake-only guard |
| `sircus/simulation/` | Browser dashboard, Python bridge, pinned dependencies, shared viewer assets |
| `sircus/physical_trials/` | Mac operator panel backend and mock tests; credentials/calibration/records remain local |
| `sircus/perception/` | Stationary RGB-D capture/inspection; calibration is incomplete |
| `sircus/isaac/` | Separate Isaac Lab 2.3.2 / Sim 5.1 articulated scene, generated assets, RGB-D capture and isolated launch scripts |
| `baselines/` | Small dated synthetic reference results; not current hardware evidence |
| `docs/` | Setup, architecture, collaboration, credential templates and limits |

Python3.12 is the target for the existing dashboard/numerical backend; the separate Isaac backend uses Sim's Python3.11. Run `scripts/test.sh` with the project venv available. Tests use mocks or local numerical models; they never command a physical robot. CI runs these checks on Ubuntu. GPU Isaac smoke tests are separate and explicit.

## What synchronization means

GitHub contains **committed and pushed** changes. Agents must fetch before work, use their own branch/checkout, and push tested changes before handoff. `main` is the shared reviewed baseline; active work is on `codex/<owner>-<task>` branches. See `scripts/sync.sh`. There is no background auto-commit, auto-merge or live robot deployment.

Credentials, local calibration, observations, upload queues, virtual environments, generated configs, photos, and transfer archives are ignored. Share those separately when needed. An agent on another account needs repository access; a private Git URL alone does not grant it.

Bucket dimensions and provisional measurements are preserved in `sircus/two-arm/scene.json` and `sircus/measurements.json`. Simulation success does not establish hardware performance. The175mm bucket height interpretation is unresolved.

Third-party licenses/provenance stay beside their model/viewer sources. No new license is granted for project-authored code by this private repository.
