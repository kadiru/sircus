# Consistent-flight residual diagnosis

The completed comparison captured **17/30** with velocity/aperture ranking,
versus imitation 14/30, PPO 14/30 and the old ensemble rank 13/30. The full native
audit passed 485,047 checks across 132 requests, including 12 separate controls.
The [curated scalar results](../../baselines/mjlab-flight-residual-20261107/README.md)
retain exact evidence hashes and all 30 consistent-selector request outcomes.

This diagnosis adds **no shots and no training**. It verifies the completed
report, manifest, contract, full audit, frozen plan, requested actions and every
consumed evidence file before reconstructing dense first-flight segments on CPU.
Its 802 derived checks supplement the original native audit; they do not rescore
capture or replace that audit.

## What remains wrong

The selector had 26 native releases: 17 complete captures, nine route failures,
and four additional setup rejects. Eight failed native shots never entered the
bucket. **Case20 did enter and retain for one second**, but robot contact invalidated
the complete board-to-bucket route. These are different failure types. First
post-board flight cutoffs were bucket five, table three and robot one. Bucket
contact alone does not identify a rim hit or prove final capture.

Three setup rejects report `finite_ik_unresolved`; one reports
`strict_endpoint_clearance_or_pose_rejected`. All top-four checks failed for each.
A finite solver failure does not establish physical unreachability.

All sixteen candidates had negative model aperture margins in every native
failure layout. The bank may lack useful gaps even after better ranking. This
motivates the [gap-coverage comparison](GAP_COVERAGE.md); it is a hypothesis, not
evidence that expanded gaps will succeed. The nominal-origin offset for these
misses is only about 1.03–1.71 mm and is not established as their cause.

## Concrete examples and curve meaning

| Case | Board height / tilt / gap | Observed result |
|---|---|---|
| 01 | 274.148 mm / 27.534° / 96.365 mm | No entry; bucket-contact cutoff; old capture head scored 92.35% |
| 19 | 416.209 mm / 34.095° / 148.146 mm | No entry; measured center-at-rim clearance −92.18 mm |
| 20 | 401.637 mm / 37.277° / 81.687 mm | One-second retention, but robot contact makes the complete route fail |
| 02 | 446.605 mm / 26.676° / 168.584 mm | No native release; all four geometry checks unresolved by finite IK |

Case01's direct distance head predicted 13.29 mm versus the measured first-flight
closest-opening distance of 133.06 mm. Conditional endpoint error was 6.31 mm
using predicted velocity from the observed rebound origin, versus 0.066 mm using
measured velocity. These are distinct quantities, not alternative capture scores.

The UI `#flightResidualDiagnosis` separates the recorded native segment from
three gravity-only forecasts: measured origin/model velocity, measured
origin/measured velocity, and nominal tangent origin/model velocity. Each ends
before the first later contact response. They do not predict wall collisions,
robot interception or retention. Missing rim crossing remains unknown.

## Reproduce without a GPU

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  /path/to/pinned-mjlab-python -m sircus.mjlab.consistent_flight_diagnosis \
  --study /path/to/immutable/consistent-flight-v2 \
  --audit /path/to/consistent-flight-v2-audit.json \
  --output /path/to/new/flight-residual-diagnosis
```

Use a new directory outside the immutable study. The complete audit and dense
observations remain local; Git contains only curated scalar diagnostics and
hashes. One nominal shot per method/layout does not establish reliable windows
or an architecture winner. These formerly held-out layouts are now development
data. Native64, privileged geometry and held endpoints remain the scope; no new
128/noise, perception, driven arm transit, grasping or hardware claim is made.
