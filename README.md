# Sircus — Simple Circus

Two UFACTORY850 arms learn to bounce a ball from a suction-mounted board into a bucket. Arm10 picks/releases; arm14 positions the board. This private repository is the shared source for Mac physical work and Ubuntu simulation work.

![Architecture](docs/sircus-architecture.png)

## Start here

- **Ubuntu / Varun:** [setup, training and local dashboard](docs/UBUNTU_SETUP.md). Run `./setup-ubuntu.sh`, then `./start-local.sh`. In another terminal run `sircus/simulation/.venv/bin/python sircus/two-arm/local_dashboard.py`.
- **Orientation/height experiment:** `sircus/simulation/.venv/bin/python sircus/two-arm/train_orientation_height.py --budget 32 --tests 10 --output experiments/run-001`. No server or hardware required. The search holds release XY, board center/yaw and bucket fixed; it optimizes roll, pitch and gap with endpoint IK checks. V2 uses [one-second capture scoring](docs/SCORING_CONTRACT.md). Native replay rejects these policies until a matching runtime adapter is integrated; endpoint IK alone does not validate a motion path.
- **Scene-conditioned learning:** [study contract and commands](docs/SCENE_CONDITIONED_STUDY.md), [frozen checkpoint and synthetic results](baselines/scene-conditioned-20261030/README.md). A trained ridge model predicts roll/pitch/U/V/gap on disjoint layouts without search at inference. The opt-in dashboard panel previews numerical trajectories only; native v2 replay remains blocked.
- **Mac / physical sfhack10:** [operator panel instructions](sircus/physical_trials/README.md). Physical control is explicit and separate; cloning/pulling never starts hardware or reloads a running dashboard.
- **All contributors and agents:** [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md), and [current project summary](docs/PROJECT_STATE.md).

## Implementation audit and next experiment

Read the [Mac system audit, UI/API contracts, and learning integration plan](docs/audits/2026-10-29/README.md) before new development. These distinguish implemented capabilities from historical observations and proposed work; raw local evidence and credentials are excluded.

## Source layout

| Directory | Purpose |
|---|---|
| `sircus/two-arm/` | Shared measured/provisional scene, two-arm physics, kinematics, Viam module, trainers and fake-only guard |
| `sircus/simulation/` | Browser dashboard, Python bridge, pinned dependencies, shared viewer assets |
| `sircus/physical_trials/` | Mac operator panel backend and mock tests; credentials/calibration/records remain local |
| `sircus/perception/` | Stationary RGB-D capture/inspection; calibration is incomplete |
| `baselines/` | Small dated synthetic reference results; not current hardware evidence |
| `docs/` | Setup, architecture, collaboration, credential templates and limits |

Python3.12 is the target. Run `scripts/test.sh` with the project venv available. Tests use mocks or local numerical models; they never command a physical robot. CI runs these checks on Ubuntu.

## What synchronization means

GitHub contains **committed and pushed** changes. Agents must fetch before work, use their own branch/checkout, and push tested changes before handoff. `main` is the shared reviewed baseline; active work is on `codex/<owner>-<task>` branches. See `scripts/sync.sh`. There is no background auto-commit, auto-merge or live robot deployment.

Credentials, local calibration, observations, upload queues, virtual environments, generated configs, photos, and transfer archives are ignored. Share those separately when needed. An agent on another account needs repository access; a private Git URL alone does not grant it.

Bucket dimensions and provisional measurements are preserved in `sircus/two-arm/scene.json` and `sircus/measurements.json`. Simulation success does not establish hardware performance. The175mm bucket height interpretation is unresolved.

Third-party licenses/provenance stay beside their model/viewer sources. No new license is granted for project-authored code by this private repository.
