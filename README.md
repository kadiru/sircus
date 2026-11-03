# Sircus — Simple Circus

Two UFACTORY850 arms learn to bounce a ball from a suction-mounted board into a bucket. Arm10 picks/releases; arm14 positions the board. This private repository is the shared source for Mac physical work and Ubuntu simulation work.

![Architecture](docs/sircus-architecture.png)

## Start here

- [Broad shot search and shared-policy learning](sircus/isaac/FARM_SHOT_LEARNING.md) — [completed per-table results and checkpoints](baselines/isaac-farm-shot-learning-20261102/README.md): successful searched recipes at 19/35 targets, 8,960 PPO throws, and frozen sampled improvement from 96/560 to 183/560. Greedy capture remains 10/35, so the model is not ready for promotion or unseen-target claims.
- [All 35 farm sites releasing in the expanded PPO study](sircus/isaac/EXPANDED_FARM_POSES.md) — 350/350 releases, 111 qualified captures and 24 native-qualified options per table, from the unchanged selected camera. [Results and evidence](baselines/isaac-expanded-farm-20261102/README.md). A short training pass with scripted transit and privileged inputs; no held-out generalization claim.
- [Fixed-camera PPO training and action-only editing](sircus/isaac/ACTION_TRAINING_FILM.md) — records every batch of a fresh ten-batch run at the selected angle; keeps setup/drop/outcome while preserving full scoring evidence and explaining parked cells.
- [User-selected one-minute farm film](sircus/isaac/CHOSEN_FARM_FILM.md) — native silver-gray Isaac orbit, then eight drops at the exact exported front-row camera. [Verified recording evidence](baselines/isaac-chosen-farm-20261102/README.md); replay of selected PPO actions, not a new training benchmark.
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
