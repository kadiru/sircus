# Heading and robot rebound search at (1050, 25) mm

The initial inventory contains 1,546 saved drops at the last unresolved
location. The prior contact analysis found that the board arm frequently
intercepts the ball on its way to the bucket. This study requests 2,000
additional simulated releases with the bucket fixed.

The planner starts from 16 measured flights whose pre-contact trajectories
line up with the opening and eight robot-contact trajectories whose later
motion approaches it. It interleaves the original bucket-facing direction
with small steering changes, wider headings up to 45 degrees off target,
and board twist about its normal. The twist changes the reachable joint
configurations while preserving the requested surface normal. The samples
around measured rim grazes start with smaller individual changes before
varying multiple coordinates. The dropper keeps its bucket-facing yaw.

Each candidate can yield several distinct endpoint branches. Before release,
the solver checks the full end-effector poses, endpoint collisions and a
sampled setup path. Equivalent commands and earlier releases are excluded
from the 2,000. The original simulator contact model, native timestep,
settled endpoint behavior and bucket calibration are retained.

The standard route score is recorded alongside an experimental robot-assisted
label. The latter requires an observed board rebound, subsequent robot contact
and valid whole-ball entry followed by one continuous second of retention.
Robot contacts must end by the beginning of the retained run. Table and rim
contacts are permitted. The force trace provides contact order and observed
velocity change. Six settings at most, up to three from each accepted route,
are selected for three exact-command repeats apiece. The report counts a
confirmed result when all three repeats earn the same accepted route.

Run the study from pinned source in the pinned simulation environment, using
the completed and audited six-hour parent, matching assets and a fresh output
directory. Use the shared lease for the 4090. The independent audit runs
after the simulation with CUDA visibility disabled.

```sh
CUDA_VISIBLE_DEVICES="$AUTHORIZED_4090_UUID" python -m sircus.mjlab.cell1050_rebound_study \
  --parent-study "$PARENT_STUDY" --parent-audit "$PARENT_AUDIT" \
  --reference "$NATIVE_REFERENCE" --assets "$SCENE_ASSETS" \
  --output "$REBOUND_OUTPUT" --gpu-lock "$SHARED_GPU_LEASE"

CUDA_VISIBLE_DEVICES='' python -m sircus.mjlab.cell1050_rebound_audit \
  --study "$REBOUND_OUTPUT" --parent-study "$PARENT_STUDY" \
  --parent-audit "$PARENT_AUDIT" --assets "$SCENE_ASSETS" --audit "$REBOUND_AUDIT"
```
