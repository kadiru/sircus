# Broad feasibility and continuous PPO pilot

This separate, opt-in study preserves the earlier bucket geometry, scoring,
camera, and completed benchmarks. It is a single-seed terminal shot-selection
pilot, not learned grasping, reactive interception, or motor-level path control.

Five continuous actions select bucket-relative bearing offset (−180 to 180°),
radius (280–1200 mm), board center height (140–600 mm), vertical drop gap
(80–560 mm), and tilt (12–52° from horizontal). Gap is ball-bottom Z minus
board-center Z; ball-center release height adds the provisional 37.5 mm radius.
The heading faces the bucket; release XY is above board center. Full-circle
bounds do not imply that both arms can reach a full orbit.

The geometry-only map retains 512 independently scrambled Sobol requests per
bucket site (17,920 total). It checks board/table bounds, URDF outer reach,
bounded numerical IK, independent FK/joint limits, and provisional joint-chain
capsule clearances. A finite failed IK attempt is unresolved, not impossible.
Native trials add a 16-point joint-interpolation proxy and unchanged measured
settling/release checks. These do not certify all mesh/self-collision or tracked
paths; the inherited assets still disable self-collision.

The 35 bucket sites stay distinct. Twenty-five train sites, five validation
sites, and five test sites are assigned before rewards. Four geometry-only
coverage batches are filmed and excluded from PPO. The actor samples directly
in the continuous domain, without lookup, projection, or retrying rejected
actions. Every active failed attempt remains in the training denominator and
buffer. A declared geometric initialization aims toward the shared workspace
reference (670, −100 mm), with radius set from bucket-to-reference distance;
this prior is not learned or fitted to capture labels.

The finite pilot uses 64 × 25 = 1,600 training attempts, 16 PPO updates, four
epochs/update, learning rate 0.0003, clip 0.2, value weight 0.5, squashed entropy
weight 0.005 and gradient norm 1. The reward is the unchanged native board-route
capture reward. A fresh 2×64 tanh actor/critic saves all continuous latents,
likelihoods, rewards, optimizer membership and checkpoints. Initial and final
deterministic policies each receive eight repeated native trials per site.
Repeated identical layouts measure numerical repeatability; five held-out
sites are a small spatial test, not table-wide generalization evidence.

All filmed setup is normal-time 30 fps actual articulation drive motion from
the preceding measured pose. There are no per-trial joint-state teleports.
Ball placement at release is initialized; pickup and joint paths are scripted.
The exact saved user camera and silver-gray appearance remain fixed. Coverage
and on-policy footage are labeled separately; no outcome selects the clips.
Full-rate release telemetry records all balls, arm joints, board poses and
contacts. Raw media and telemetry remain outside Git.

Use the pinned Isaac Sim 5.1 / Isaac Lab 2.3.2 environment from README. Run a
four-batch smoke before the 64-batch pilot, through the existing external scene
lock and dedicated 4090 supervisor. The optional CLI is:

```bash
python.sh film_cylinder_farm.py --headless --enable_cameras --device cuda:0 \
  --assets /workspace/assets --output /workspace/output/NEW_RUN \
  --continuous-source /workspace/output/expanded-training-20261102-v4 \
  --continuous-camera-json /workspace/EXACT_SAVED_CAMERA.json \
  --continuous-feasibility /workspace/output/NEW_MAP/feasibility.json \
  --continuous-batches 64 --appearance silver
```

The read-only `continuous_pilot_server.py --output EXPORT --port 18794` serves
the map, split-specific frozen results, per-cell failures and native video.
It exposes no simulator or hardware controls and never replaces an existing
dashboard. Source synchronization is not deployment. Do not restart services,
use the other GPU, or touch physical/Viam/audio resources to reproduce this study.

Results are pending native validation. Bucket inside top/bottom are 177/150 mm,
outside top/bottom 200/158 mm. The reported 175 mm height still has unresolved
overall-height versus interior-depth meaning; the existing interior-depth
approximation and 8 mm bottom remain provisional.
