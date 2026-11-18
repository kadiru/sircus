# Clearance mismatch: no catch-label changes in the saved-data diagnostic

The separate location-comparison recordings contain **18,520 paid requests in
539 batches**. Recalculating their score/quality/route fields changed **zero catch
labels and zero clean-catch labels**. Totals remain **6,742 catches, 1,588 clean
catches, 7,175 misses and 4,603 rejected requests**. The 345 inactive padding slots
are excluded. These totals combine TRAIN, monitoring and final TEST requests;
they are not a held-out performance rate.

This is a CPU numerical diagnostic of saved data, **not a passed full audit**.
The [original shutdown failure](../isaac-seed22-comparison-shutdown-20261118/README.md)
and [failed recovery audit](../isaac-comparison-recovery-failure-20261118/README.md)
remain preserved. No model was promoted. No training, simulator startup, new shot,
scoring-rule change or frozen-benchmark modification occurred.

## The discrepancy

Only one request disagreed with the original-style float64 auditor arithmetic:
TRAIN batch `00123`, slot `2`, in the original-location copy. It was already a
**miss**, with no clean entry. The differing field was
`minimum_mouth_clearance_mm`:

| Calculation | Clearance (mm) |
| --- | ---: |
| Saved producer value | -387.38491162892745 |
| Auditor float64 coordinate subtraction | -387.38485266941296 |
| Float32 subtraction, then double-precision scoring | -387.38491162892745 |

The absolute disagreement is **0.00005895951449 mm**. The diagnostic isolates
rounding when converting world coordinates into a workcell's local coordinates:
the producer subtracts float32 tensors; the reader subtracts float64 JSON origins.
The discrepant shot remained a miss under every arithmetic path. All paths stopped
its scoring after the same 351 observations.

The original absolute field tolerance remains **1e-7** (for this field, mm).
The clean-entry requirement remains **at least 1 mm**, alongside all original
contact/exit conditions. Neither threshold was relaxed.

## Complete coverage and numerical controls

| Path through unchanged original scoring helpers | Requests with a field mismatch | Catch-label changes | Clean-label changes |
| --- | ---: | ---: | ---: |
| Float64 auditor subtraction and scoring | 1 | 0 | 0 |
| Float32 subtraction and float32 scorer input on the current Mac libraries | 13,785 | 0 | 0 |
| Float32 subtraction followed by double-precision scorer input | 0 | 0 | 0 |

All 539 batches and all 18,520 unique paid request IDs were checked. Every path
had **zero incomplete scoring intervals**. Comparisons use the original scorer
classes, original `same(..., 1e-7)` helper and original per-cell capture/fall
stopping rule. Per-request comparisons prevent opposite changes from cancelling
out in aggregate totals.

The Mac run used Python 3.12.14, NumPy 2.5.4 and SciPy 1.18.1; the original Linux
producer used different versions. The raw float32 path is therefore a
version-confounded control, not a claim of producer-runtime parity. Its additional
numeric differences concern 13,714 clearance fields and 4,265 historical
`ShotScore.reward` fields. Those are **not** independently recomputed PPO shaping
returns or optimizer updates. The third arithmetic path matched every returned
saved score/quality/route field within the unchanged tolerance.

## Final TEST labels, unchanged in all three arithmetic paths

Each model faced the same 70 broad and 25 nearby positions. Columns below are
**catch / clean catch / rejected request**, with clean catches a subset of catches.
These are diagnostic counts; full model/physics audit acceptance is still absent.

| Model | Broad: 70 requests | Nearby: 25 requests |
| --- | ---: | ---: |
| Parent seed 22 | 24 / 8 / 6 | 9 / 3 / 3 |
| Original-location continuation | 23 / 5 / 22 | 14 / 3 / 4 |
| Broader-location continuation | 25 / 3 / 14 | 12 / 2 / 3 |

TRAIN labels are also unchanged: original-location copy **3,340 catches / 877
clean / 2,289 rejected**, and broader-location copy **3,208 / 660 / 2,165**, each
from 8,960 requested shots. Monitoring labels are included separately in
[summary.json](summary.json).

## Evidence and limits

- All **25,059 sealed files / 4,330,228,832 bytes** matched their hashes while
  streaming the stopped container's saved data to the Mac. Media and model files
  were hashed but not retained by this numerical copy; no weights were loaded.
- The diagnostic opened **1,079 numerical files / 1,402,492,374 bytes** and checked
  their hashes before and after calculation. All 217 original Python source files
  matched the pinned producer manifest.
- Measured local diagnostic cost: **453.86 s wall**, **445.98 s user CPU**, **5.40 s
  system CPU**, **629.17 MiB peak child RSS**. No GPU or simulator runtime was used.
- **24 synthetic tests**, independent source review and two independent report
  reaggregations passed. Raw observations and the detailed per-request report
  remain private; compact report/source/input hashes are in [summary.json](summary.json).
- Ubuntu remained below its unchanged 6 GiB storage floor. No remote audit was
  launched. Automatic approval review rejected downloading an isolated matching
  Mac runtime; this diagnostic instead used the existing Mac environment.

The work does not re-audit policy likelihoods, optimizer updates, release/geometry
guards, startup physics, camera/video bindings or shutdown. A future corrected
reader and full CPU audit would need to retain the same tolerance and separately
record its outcome. This report does not amend or replace either failed audit.

Implementation: [saved-data diagnostic](../../sircus/isaac/LOCATION_COMPARISON_CLEARANCE.md).
