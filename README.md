# Sircus — Simple Circus

Can two robot arms bounce a ball into a bucket after the bucket moves? One UFACTORY850 arm holds a tilted board. The other releases the ball above it. The planner chooses the board pose and drop height from the known scene geometry. A capture requires the whole ball to stay in the bucket for one continuous second after bouncing off the board.

![Saved MuJoCo replay with one arm holding the ball above the board and the second arm tilting the board toward the bucket](docs/images/two-arm-replay.webp)

*An archived MuJoCo/mjlab simulation frame. Both arms hold their selected poses during the drop. [Replay provenance](https://github.com/kadiru/sircus/blob/fd950dc49fc5634f70e021f40619d4e8a8f21356/baselines/mjlab-yaw-tilt-gap-20261109/video.json).*

## Start here

| If you want to… | Start with… |
| --- | --- |
| Understand how a shot is chosen and what worked | [Illustrated one-drop planner guide](docs/ONE_DROP_PLANNER.md) |
| Inspect the November showcase implementation | [mjlab planner](https://github.com/kadiru/sircus/blob/9b59b84b8820b40790ebe20f1a036c1ea8679282/sircus/mjlab/intergrid_system_plan.py) and [interactive replay and results](https://github.com/kadiru/sircus/blob/9b59b84b8820b40790ebe20f1a036c1ea8679282/sircus/mjlab/web/README.md) at the revision used in the showcase |
| Run a small numerical experiment on `main` | [Ubuntu setup and fixed-scene experiment](docs/UBUNTU_SETUP.md) or the commands below |
| Inspect a trained model without running the simulation | [October checkpoint, test scorecard and trajectory preview](baselines/scene-conditioned-20261030/README.md) |
| Explore the physical setup | [Mac operator panel and mock adapter](sircus/physical_trials/README.md) and [stationary RGB-D capture](sircus/perception/CAPTURE_README.md) |

## Run the CPU experiment on main

With Python 3.12 on Ubuntu, run these commands from the repository root. This fixed-scene search tries board roll, pitch and release gap while holding the bucket and release XY fixed. It checks inverse kinematics at both arm endpoints and runs a numerical ball rollout. The experiment uses no GPU, Viam server or robot.

```bash
./setup-ubuntu.sh
sircus/simulation/.venv/bin/python sircus/two-arm/train_orientation_height.py \
  --budget 4 --tests 2 --seed 42 --output experiments/first-shot
```

Open `experiments/first-shot/summary.json` for the result. Give each run a fresh output directory. For the optional local viewer, follow the [Viam server and dashboard steps](docs/UBUNTU_SETUP.md). The dashboard can preview scene-conditioned numerical trajectories. Native replay of those v2 policies requires a matching runtime adapter.

## Where the code lives on main

| Area | What you will find |
| --- | --- |
| [Two-arm simulation](sircus/two-arm/) | [Scene geometry](sircus/two-arm/scene.json), [two-arm kinematics](sircus/two-arm/two_arm_kinematics.py), [ball rollout](sircus/two-arm/shot_rollout.py) and [capture scoring](sircus/two-arm/capture_scoring.py) |
| [Dashboard and models](sircus/simulation/) | Browser viewer, Python bridge, Viam integration and pinned dependencies |
| [Learning experiments](docs/SCENE_CONDITIONED_STUDY.md) | Training and evaluation commands, layout splits and comparisons with fixed actions and per-scene search |
| [Reference results](baselines/) | Dated synthetic checkpoints, scorecards and experiment inputs |
| [Physical operator](sircus/physical_trials/) and [perception](sircus/perception/) | Mac control panel, mock tests, stationary camera capture and alignment tools |

Hardware calibration is incomplete. The scene records provisional values, including whether the measured 175 mm bucket height describes its interior depth or overall height. Simulation capture rates apply to the specified layouts and scoring rules. [Measurements](sircus/measurements.json) · [Scoring contract](docs/SCORING_CONTRACT.md).

## Development

Run `scripts/test.sh` for the mock and numerical checks after setup. Use [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) for the shared branch and review workflow. Pushing code and operating the robot are separate actions. Local credentials, calibration, raw observations and generated runtime files stay outside Git.

[System architecture](docs/sircus-architecture.png) · [October integration checkpoint](docs/PROJECT_STATE.md) · [System audit and integration plan](docs/audits/2026-10-29/README.md)

Third-party license and provenance files sit beside their models and viewer assets. No additional license is granted for project-authored code.
