# Ensemble flight misses — recorded-flight diagnosis

The ensemble's high scores did not reliably predict capture. This report
reconstructs its original fixed64 recordings before any more training.
[Method and commands](../../sircus/mjlab/ENSEMBLE_FLIGHT_DIAGNOSIS.md);
[read-only UI](http://127.0.0.1:18781/#ensembleFlightDiagnosis).

## Failure breakdown

| Original ensemble requests | Count |
|---|---:|
| All requested | 60 |
| Actual native releases | 49 |
| Captures | 24 |
| Native flight misses | 25 |
| Setup rejects | 11 |

All 25 flight misses had **no whole-ball entry and zero retention**. Their first
post-board free flight ended at bucket contact in 16 cases, robot contact in
three and table contact in six. These are observed contact channels, not a new
failure-scoring rule. Bucket contact does not by itself identify a rim hit.

Scores at least 90% yielded **18 captures from 36 requested shots**: 13 native
flight misses and five setup rejects remain in that denominator. Scores at
least 99% yielded **4 captures from 13** despite a mean model score of 99.54%.
Model scores and member disagreement remain uncalibrated.

## Examples to inspect in the UI

| TEST case | H / tilt / gap | Model score | Direct predicted / observed first-flight closest distance |
|---|---|---:|---:|
| 29 | 268.099 mm / 31.520° / 261.560 mm | 99.654% | 8.4 / 107.9 mm |
| 49 | 262.108 mm / 33.125° / 264.839 mm | 99.495% | 15.8 / 113.7 mm |
| 36 | 413.012 mm / 31.014° / 184.645 mm | 98.752% | 14.5 / 72.5 mm |
| 35 | 284.167 mm / 23.896° / 193.984 mm | 99.312% | 8.6 / 91.2 mm |

These cases all missed. Case 36 keeps the successful BC board XY, height and
tilt identical, increasing only drop gap by 36 mm (148.645 → 184.645 mm).
BC/PPO captured; the ensemble missed. One original sample per action cannot
establish a repeatable safe gap window. Cases 29 and 49 missed with all three
methods. Case 35 improved admission over rejected BC/PPO setups but still missed.

Case 29's measured setup errors were tiny: approximately 0.00022 mm board
position error, 0° normal error, zero recorded board drift and zero bucket
collider placement error. Its predicted vertical rebound speed was 0.4604 m/s,
versus measured 0.5426 m/s. This points to flight prediction/ranking error,
without proving one root cause for every miss.

## How to read the curves

Solid traces are saved native simulation. Conditional ballistic curves start
at the **observed** rebound origin and use either modeled or measured velocity
plus gravity, with no later contacts. They are not newly simulated shots or
learned full trajectories. The model-velocity curve's case29 endpoint error is
about 16.97 mm; the measured-velocity check is about 0.07 mm.

Those endpoint errors differ from the direct opening-distance head's error.
Closest 3D opening distance is not signed rim clearance. If contact cuts off a
flight before descending rim-plane crossing, the crossing/clearance remains
unknown; the complete scorer independently establishes no whole-ball entry.

The independently supervised capture/distance/velocity heads have no ballistic
consistency constraint. Five members can share a blind spot. Normalized
nearest-TRAIN distances describe local support, not causal proof or calibrated
confidence. The UI retains both the exact observations and these limitations.

The derived diagnosis passed **600 reconstruction checks**. These supplement
the original 897,304-check native audit and do not replace its scoring authority.

## Evidence and boundary

- [Scalar summary](summary.json) contains the failure funnel, score bins, exact
  cases, all 25 misses and provenance; raw trajectories remain outside Git.
- [Reconstruction checks](checks.json) and [checksums](SHA256SUMS) bind this
  derived publication. Their scope is separate from the original native audit.
- Original report SHA256: `9d76ed2225f5e9a659d5b8ef18839207553df87e14980cf0a290707de437b5a6`.
- Original full audit SHA256: `72e5cc9fd4c424896a0c41588b3eec0bd85317ce931cd76e90bd850474f87ebb`;
  897,304 original checks. This diagnosis does not replace that proof.
- Original contract: `90a7d5b7e3d1fb1e738a36a7cf30930116516da0d6d6435c06b98eb97be8c376`.

This is CPU analysis with **zero new native shots and zero training**. The
occupied 4090 workload was not interrupted; Isaac, the 4070, old results and
the interactive solver remain unchanged. Only fixed64, privileged geometry and
held endpoints are covered. No noise, 128 convergence, physical validation,
perception or driven-transit claim. The 175 mm bucket-height interpretation
remains provisional.

Next, compare a physically consistent flight/aperture score with the capture
head on these development cases. No such tuning was performed here. A changed
selector needs a new frozen held-out evaluation; these former TEST examples
are now diagnostic development data.
