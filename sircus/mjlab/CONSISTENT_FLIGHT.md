# Frozen velocity + aperture selector

This bounded follow-up changes **ranking**, not learned weights or native physics.
It follows the high-confidence flight-miss diagnosis: the classifier, distance
and velocity heads can disagree. The old ensemble TEST positions are now
**development data**. A changed selector is evaluated at 30 new positions.

## What changes

Use the same sixteen candidates from saved imitation/PPO, the geometric anchor,
TRAIN examples and fixed perturbations. For each candidate, transform all five
members' predicted rebound velocities into world coordinates. Combine them with
requested board geometry, ball radius and gravity to compute the ideal flight
through the bucket's compiled opening. The nominal origin is the tangent point
of a vertically falling sphere, not an observed rebound from a future trial.

Rank supported board contact, complete downward rim passages, worst member
clearance, mean clearance, then the original rank and candidate ID. Each ranked
method checks at most four candidates with the unchanged strict setup/path
admission, choosing the first admitted. No new native flight is searched at
inference. Negative margins are **not hard rejects**: a rim-assisted capture
may still succeed. All failed setups remain in the denominator.

The aperture margin uses the full ball radius throughout its downward passage
across rim ± radius. This is conservative geometric plausibility, not a capture
probability. It is distinct from the old closest-3D-opening-distance head and
from single ball-center-at-rim clearance. Velocities were trained after finite
contact, whereas the new origin is idealized; this mismatch remains explicit.

No robot-flight collision, bucket-wall bounce or retention is predicted by the
ballistic formula. Existing static setup and path geometry checks remain intact.
Robot-interception filtering is a separate next ablation. Below-rim launches are
handled mathematically, but possible earlier outer-wall contact is not modeled.
Only actual unchanged native scoring determines entry and one-second capture.

## Frozen experiment

- Four methods: saved imitation, saved PPO, original ensemble rank, new rank.
- Thirty fresh bucket positions: seed `20261701`, X 100–1240 mm, Y −650–250 mm.
  Every new position is at least 10 mm from declared ancestral positions, all
  old ensemble validation/test positions and every other new position.
- One nominal request per method/layout: 120 TEST requests. Twelve matched
  positive/native-miss control requests before/after give **132 total slots**.
- Same frozen checkpoints, candidate bank, geometry limits, native64 solver,
  scoring and measured/provisional scene. No new fitting, noise or 128 checks.
- All actions, rankings, geometry proofs and hashes frozen before GPU acquisition
  and before any fresh outcome. No retry, outcome-conditioned selection or
  substitution. Setup rejection and invalid dynamics stay visible.
- Exclusive existing 4090 lock plus foreign-process check. Never signal another
  workload. Queue expires after 24 hours; acquired worker capped at one hour;
  at least 8 GiB free disk must remain. The 4070 and Isaac remain untouched.

This is a small paired pipeline comparison, not an all-table success claim or
an equal-compute comparison with the direct actors. Both ranked selectors use
16 model candidates and up to four geometry checks. Model disagreement is not
calibrated uncertainty. All previous scores/fingerprints are preserved.

## Interface and reproduction

The additive `#consistentFlightStudy` panel shows original measured development
examples alongside nominal aperture forecasts. Changed candidates have unknown
outcomes until the new native experiment. Queued/running counts remain
provisional; complete results require a byte-bound independent audit.
The running interactive solver is not switched or restarted.

```bash
CUDA_VISIBLE_DEVICES=GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4 \
  /path/to/pinned-mjlab-python -m sircus.mjlab.consistent_flight_pipeline \
  --parent-study /path/to/policy64-v1 \
  --parent-audit /path/to/policy64-v1-audit.json \
  --ensemble-study /path/to/physics-ensemble-v1 \
  --ensemble-audit /path/to/physics-ensemble-v1-audit.json \
  --native-source /path/to/frozen-policy64-source \
  --assets /path/to/assets --output /path/to/new-consistent-flight-v1 \
  --gpu-lock /path/to/existing/scene.lock \
  --display /path/to/existing-interface-report.json \
  --queue-timeout 86400 --wall-time-limit 3600
```

A new output namespace is mandatory. Raw native traces, host paths, logs and
runtime configs stay outside Git. Source synchronization does not deploy a
physical dashboard or cloud part. No hardware is required or accessed.
