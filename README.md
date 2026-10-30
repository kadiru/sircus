# Sircus — Simple Circus

Two UFACTORY850 arms learn to bounce a ball from a suction-mounted board into a bucket. Arm10 picks/releases; arm14 positions the board. This private repository is the shared source for Mac physical work and Ubuntu simulation work.

![Architecture](docs/sircus-architecture.png)

## Start here

- **Latest multi-computer handoff:** [current demo status, paused Green trial, draft branches and parallel tasks](docs/AGENT_HANDOFF.md). Read this before historical status sections; syncing main alone does not fetch the latest demo into your checkout.

- **Ubuntu / Varun:** [setup, training and local dashboard](docs/UBUNTU_SETUP.md). Run `./setup-ubuntu.sh`, then `./start-local.sh`. In another terminal run `sircus/simulation/.venv/bin/python sircus/two-arm/local_dashboard.py`.
- **Orientation/height experiment:** `sircus/simulation/.venv/bin/python sircus/two-arm/train_orientation_height.py --budget 32 --tests 10 --output experiments/run-001`. No server or hardware required. The search holds release XY, board center/yaw and bucket fixed; it optimizes roll, pitch and gap with endpoint IK checks. V2 uses [one-second capture scoring](docs/SCORING_CONTRACT.md). Native replay rejects these policies until a matching runtime adapter is integrated; endpoint IK alone does not validate a motion path.
- **Scene-conditioned learning:** [study contract and commands](docs/SCENE_CONDITIONED_STUDY.md), [frozen checkpoint and synthetic results](baselines/scene-conditioned-20261030/README.md). A trained ridge model predicts roll/pitch/U/V/gap on disjoint layouts without search at inference. The opt-in dashboard panel previews numerical trajectories only; native v2 replay remains blocked.
- **Saved-pose demo on MacBook:** [offline preparation and explicit hardware/audio startup](docs/HARDWARE_DEMO_READY.md). Logic Pro is optional; Ghost Booth can play locally. The four MacBook MIDI inputs and voices have been commissioned; robot-cycle timing remains separate. FX starts off; [native loudness boost](docs/AUDIO_LOUDNESS.md) is an explicit audio-only control.
- **Mac / physical sfhack10:** [operator panel instructions](sircus/physical_trials/README.md). Physical control is explicit and separate; cloning/pulling never starts hardware or reloads a running dashboard.
- **Mac Mini / audio:** [Ghost Booth event sound library and rehearsal desk](sircus/audio/README.md). Playback is opt-in; the show branch adds nonblocking controller cues and measured MIDI routing. There is no automatic capture detector.
- **MacBook audio integration:** [optional dashboard observer and commissioning boundary](docs/DASHBOARD_AUDIO_INTEGRATION.md). FX starts off; robot motion never waits for audio. Source-only integration is not deployed. Same-Mac loopback is supported; a Mini connection requires separate explicit private-LAN setup.
- **All contributors and agents:** [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md), and [current project summary](docs/PROJECT_STATE.md).

## Four-mode show desk

Run `./start-show.sh` and open `http://127.0.0.1:18769/show.html` for Manual operation, Trained operation, Simulation and Mock simulation. The default is mock-only. The [show runbook](docs/SHOW_RUNBOOK.md) covers editable TCP poses, the one-button two-second routine, four measured-impact ball identities, optional drums/FX and the remaining physical commissioning steps. [Bucket-facing rehearsal](docs/BUCKET_FACING_WORKFLOW.md) adds entered/seeded boxed board positions, geometric heading, tilt/release planning and the load-then-Go sequence. Hardware deployment is explicit and separate from source synchronization.

For repeatable fixed-pose runs, see [deterministic reliability checks and physical acceptance](docs/DETERMINISTIC_RELIABILITY.md). The no-hardware endurance runner preserves the real two-second waits, verifies recording and repeated operator presses, and retains unknown outcomes separately from routine completion.

## Guided operator and training workspace

[Operator workspace](docs/OPERATOR_WORKSPACE.md) adds a separate offline workflow for board placement, bucket scan rehearsal, loading, numerical grip/drop and real local model-training jobs. Launch `./start-operator-sim.sh` and open `http://127.0.0.1:18766/operations.html`. The default view places the bucket on the operator’s negative-Y side. Open `/study-results.html` for the [learning curve and challenge report](docs/FRONT_BUCKET_STUDY.md), saved numerical replays and video export. [Viam commissioning stubs](docs/VIAM_DEPLOYMENT.md) are blocked until physical integration is validated. Hardware execution remains disabled. [Device sync instructions](docs/DEVICE_SYNC.md) preserve existing runtimes and draft branches.

## Implementation audit and next experiment

Read the [Mac system audit, UI/API contracts, and learning integration plan](docs/audits/2026-10-29/README.md) before new development. These distinguish implemented capabilities from historical observations and proposed work; raw local evidence and credentials are excluded.

## Source layout

| Directory | Purpose |
|---|---|
| `sircus/two-arm/` | Shared measured/provisional scene, two-arm physics, kinematics, Viam module, trainers and fake-only guard |
| `sircus/simulation/` | Browser dashboard, Python bridge, pinned dependencies, shared viewer assets |
| `sircus/physical_trials/` | Mac operator panel backend and mock tests; credentials/calibration/records remain local |
| `sircus/studies/` | Corrected front-bucket learning curves, held-out challenges and frozen model contracts |
| `sircus/deployment/` | Blocked Viam coordinator, unknown-result vision service and typed commissioning adapters |
| `sircus/audio/` | Ghost Booth, native audio/ODD MIDI, event sound profiles, shared event contract and record converters |
| `sircus/perception/` | Stationary RGB-D capture/inspection; calibration is incomplete |
| `baselines/` | Small dated synthetic reference results; not current hardware evidence |
| `docs/` | Setup, architecture, collaboration, credential templates and limits |

Python3.12 is the target. Run `scripts/test.sh` with the project venv available. Tests use mocks or local numerical models; they never command a physical robot. CI runs these checks on Ubuntu.

## What synchronization means

GitHub contains **committed and pushed** changes. Agents must fetch before work, use their own branch/checkout, and push tested changes before handoff. `main` is the shared reviewed baseline; active work is on `codex/<owner>-<task>` branches. See `scripts/sync.sh`. There is no background auto-commit, auto-merge or live robot deployment.

Credentials, local calibration, observations, upload queues, virtual environments, generated configs, photos, and transfer archives are ignored. Share those separately when needed. An agent on another account needs repository access; a private Git URL alone does not grant it.

Bucket dimensions and provisional measurements are preserved in `sircus/two-arm/scene.json` and `sircus/measurements.json`. Simulation success does not establish hardware performance. The175mm bucket height interpretation is unresolved.

Third-party licenses/provenance stay beside their model/viewer sources. No new license is granted for project-authored code by this private repository.
