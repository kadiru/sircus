# Fixed-scene cooperative shot RL

This task runs both articulated arms and ball dynamics in Isaac Lab / PhysX on
the dedicated4090. The4070 workload is stopped at the user's request. No hardware,
Viam configuration, audio service or existing numerical results are changed.

## What learns

A shared Gaussian policy chooses three parameters once per shot:

| Action | Range |
|---|---|
| Board tilt from horizontal | 12–45 degrees |
| Heading offset from directly facing bucket | -15–15 degrees |
| Vertical ball clearance above board center + ball radius | 60–250mm |

The policy uses REINFORCE with an episodic return baseline, Adam updates in
batches of16 and a small entropy bonus. It learns the mean and exploration
spread of a bounded action distribution; it is a fixed-layout, single-decision
RL problem, not a scene-conditioned neural network or joint-trajectory policy.
There is no precomputed optimizer label dataset or per-scene search at inference.

The board center, bucket, arm bases and release XY stay fixed. Endpoint IK places
both simulated arms; they settle under joint controllers before release. Invalid
IK receives a penalty. The ball is initialized in the open gripper proxy at the
release location. Thus grasping, approach paths and grasp-to-release timing are
**not learned or validated** by this task. Initializing endpoints is a simulator
reset, not a physical motion plan. Arm self-collision remains disabled.

This first policy uses privileged fixed-scene geometry. The RGB-D camera renders
the real simulated scene for inspection, but depth/point clouds are not yet
policy inputs. Perception-driven observations are a later stage.

## Physics and scoring

Task contract `sircus-isaac-fixed-shot-v5` fingerprints the scene, action bounds,
scoring and physics choices. The foundation's average restitution combine is
replaced by **max combine** for the provisional ball restitution0.65. Board
mounting is still rigid; these are uncalibrated contact assumptions, not measured
real-ball behavior. Earlier foundation and numerical results remain separate.

Board contact comes from an Isaac Lab filtered PhysX contact sensor on the ball,
filtered to the attached impact board. A bounce requires contact followed by
separation and positive normal rebound speed. A whole sphere must enter and
remain inside the tapered bucket for one continuous second. Partial exits reset
retention. Rim contact does not disqualify a capture. Exact signed distances to the64 collision wall planes determine side clearance; floor contact has a1mm tolerance. Side/top checks use an explicit1µm numerical
tolerance for float32 resting-contact jitter; it is included in the fingerprint.
The maximum rollout is4 seconds. Shaping rewards board rebound, proximity and
retention; only the separate success flag represents a completed shot.

See the [contact sensor API](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/core-concepts/sensors/contact_sensor.html).

## Run and inspect

On Ubuntu, from `~/sircus-isaac/source`:

```bash
bash sircus/isaac/run.sh status
# Stop ONLY the Sircus scene if one is running; preview process can remain.
bash sircus/isaac/run.sh stop
bash sircus/isaac/run.sh train 256 42
```

Runs are saved under `~/sircus-isaac/output/rl-<UTC>/`:

- `contract.json`: provenance, fixed geometry, limits and task fingerprint.
- `heuristic_baseline.json`: midpoint action, separately labeled.
- `baseline.json`: untrained policy's deterministic action.
- `episodes.jsonl`: actions, reward/outcome, contact force, ball trajectory, release
  and arm targets. No synthetic camera data is mislabeled as physical evidence.
- `checkpoint.json`: actor parameters, action, training count and compatibility
  contract, replaced atomically after each16-shot update.
- `evaluation.json` / `summary.json`:8 final frozen-action trials with no updates.

For independent frozen-checkpoint replay after training has exited:

```bash
bash sircus/isaac/run.sh evaluate /workspace/output/rl-UTC/checkpoint.json 8
```

An incompatible scene/physics/scoring fingerprint rejects the checkpoint. A
replay does not restart training or optimize an action. These repeated trials
use the same deterministic layout; they do not measure generalization. Positions
must remain fixed for this stage. Randomized train/validation/test layouts come
after the fixed task and model checks pass.

Preview: Ubuntu `http://127.0.0.1:18860`, forwarded Mac
`http://127.0.0.1:18780`. It displays the rendered scene, depth and learning status;
it has no physical command endpoint. Training exits after the requested budget;
the preview then becomes stale. Nothing starts automatically on reboot.

## Verification boundary

The first16-episode v1 pipeline smoke saved a changed policy, observed PhysX board
contacts/rebounds, and scored8/8 deterministic frozen trials. The midpoint
heuristic also scored2/2: this smoke demonstrates working updates/replay, **not
an RL advantage**. Its local run is `rl-20261031T024024Z`; it is preserved as a
distinct earlier contract. The final v5 task uses exact polygon wall-plane containment, deterministic IK
initialization and a weaker untrained actor initialization; it retains the
midpoint heuristic as an explicit comparator. A board rebound must occur within
50ms of board contact, preventing a later floor bounce from qualifying. Earlier
v2/v3/v4 diagnostic runs remain local and separate; their partial results are not
reported as final training evidence.

Eleven dependency-light geometry/scoring tests cover frame conversion, board
orientation, tapered geometry, rebound requirement, whole-ball containment,
continuous retention reset, action bounds and fingerprint changes. The full
existing mock/numerical suite is also run separately. Neither establishes
real-world safety or contact accuracy. Timestep convergence, grasping,
compliance and collision-constrained trajectories remain pending.
