# Coupled-arm feasibility and saved-policy reanalysis

The new `joint_feasibility.py` API separates geometry admission from shot
learning and measured motion execution. `SettledTransitShots.prepare` now uses
this shared API for its initial action qualification. Its scripted movement,
correction logic, settling tolerances and release guards are unchanged.
The frozen benchmark producer source, experiment, checkpoints and results are
historical inputs and were not edited or restarted.

## What is implemented

`preflight(scene, cell, action, links, starts, *, seeds=None)` accepts one exact
five-parameter shot and the actual six-joint positions of **both** arms. It:

1. Checks the original action bounds and consistent derived geometry.
2. Constructs the coupled board and dropper flange targets.
3. Applies the original board/table/bucket and ball/board clearance checks,
   both-arm necessary outer reach bounds, robust IK, independent FK and joint
   limits, and equivalent-joint representation relative to measured starts.
4. Checks the original endpoint capsules and 16-sample joint-interpolation path
   proxy from those starts.
5. Returns the unchanged action, targets, fits, path, status, rejection reason
   and fingerprints of the scene, chains, context, action, starts, seeds, solver
   protocol and source files. `release_authorized` is always false.

The gap remains **ball-bottom release Z minus board-center Z**, at least 80 mm.
Ball radius is 37.5 mm, and the provisional dropper flange offset is another
95 mm. Thus minimum-gap release places that flange 212.5 mm above board center.
The tilted board's normal clearance is checked separately. Both arms must
reach their different compatible poses; reaching board XYZ alone is insufficient.

`prepare_candidates(...)` exposes every candidate and every admitted index in a
finite, geometry-only support set. It does not rank by reward, pick one anchor,
claim feasibility throughout a box, project an actor output, retry or remove
failures from accounting. Empty support means no candidate qualified within the
finite supplied set and solver budget. It does not prove the bucket impossible.

`validate_preflight(...)` rejects stale inputs and independently rechecks
admitted FK and path evidence without optimizing IK. A prepared record is
geometry evidence, not permission to move or release. Native execution still
uses actual starts and all original measured pose, velocity and drift guards.

## Current training boundary

The existing actor, critic, observation features, reward and checkpoint bytes
are unchanged. The native controller now calls the shared exact-action layer;
the candidate API can also run without importing Torch, Isaac or a controller
process. No native job was launched for this change.

The old continuous PPO distribution has **not** been replaced with a policy over
finite support. It can still propose invalid actions. Feeding its output through
nearest-candidate projection or repeated resampling would change the policy;
old PPO log probabilities would no longer describe the executed action. A future
training decoder/distribution must declare and account for that change. This
task implements the shared layer and reanalyzes the current policy; it does not
claim reduced training cost or improved capture rates.

The board-goal input interface, network changes, continuation training and the
from-scratch-versus-transfer experiment remain deferred for discussion.

## Verified reanalysis

See [compact results](../../baselines/isaac-joint-feasibility-20261108/README.md).
All **420 final frozen PPO requests** were checked: three seeds times
35 known-TRAIN diagnostic +35 validation +70 TEST positions. The layer replayed
the exact saved finite IK answers, independently recomputed FK, geometry and
path proxies, and matched original admission on **420/420**. No solver search,
actor inference, optimizer update or new ball flight was performed in that
shadow analysis. The original passed audits remain the evidence for ball-flight
scores and checkpoint predictions.

The analyzer verifies the published experiment/splits, original independent
audit receipts, progress/scene/checkpoint/proof/trial hashes, exact context
membership, unchanged final model state and layout coverage. URDF bytes are
bound to the previously committed reach-audit geometry receipt. Missing fits,
changed short-circuit sequences or admission/outcome inconsistencies fail closed.

Separately, the first ten final known-TRAIN requests of seed 20261122 received
fresh CPU robust-IK solves, in fixed slot order, with **10/10 admission parity**:
eight admitted, one board IK unresolved and one outside a necessary outer reach
bound. This verifies the live solver path; it is not ten new training shots.

Example, using a read-only mirror of original evidence and generated assets:

```bash
python sircus/isaac/joint_feasibility_analysis.py \
  --benchmark /path/to/frozen-benchmark-mirror \
  --assets /path/to/original-assets \
  --published baselines/isaac-sweep-ppo-10k-20261106/final-comparison.json \
  --output /path/to/separate-new-shadow.json
```

The output must be new and outside the input experiment. Raw trial files and
generated runtime inputs stay outside Git. This analysis requires NumPy and
SciPy, not Isaac, Torch, a GPU or any robot connection.

## Limits

IK failure is unresolved, not a proof of impossibility. Clearance proxies can
reject safe poses. The path is finitely sampled, self-collision is still disabled
in the inherited native assets, and this is not a mesh collision certificate or
an obstacle-avoiding planner. Board IK constrains center and normal, leaving
in-plane twist free. A mechanically admitted shot may still miss the bucket.
The geometry layer binds joint positions, not velocity, observation age or
human motion; a future moving-board supervisor needs those additional checks.
No new native rollout or physical validation accompanies this source change.
