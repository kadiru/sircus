# Matched +60 mm release-gap correction — November 7, 2026

This is a completed **development intervention**, not policy training or a fresh held-out evaluation. It retests cases 12, 24 and 48 from the original 50-position comparison. Only release gap changes; board XYZ, board tilt and bucket XY remain fixed.

| Result | Original | Gap +60 mm |
|---|---:|---:|
| Diagnostic requests/native releases | 9/9 | 9/9 |
| Complete captures | 0/9 | 9/9 |
| Clean captures | 0/9 | 0/9 |
| Post-board-rebound robot-contact trials | 9/9 | 0/9 |

Each case changes from 0/3 to 3/3 complete captures. All 9/9 corrected captures have recorded rim-band/bucket contact; none are clean. The scorer labels them `assisted_or_marginal_capture`. Cases 12 and 24 briefly exit after entry before later retaining for one continuous second; case 48 does not. All nine original repetitions enter and retain the ball for one second, but intervening robot contact invalidates the route. Corrected shots have no robot contact anywhere in the four-second recording. Capture and contact evidence remain separate.

All 30 requested shots released: 18 diagnostic shots plus 12 matched pre/post controls, with no setup rejections or invalid physics. Controls behaved as expected: six clean positive captures and six negative misses. The independent native audit passed **46,625 checks**; the separate contact derivative passed **771 checks**. Worker wall time was **225.928 seconds**, including overhead; pure GPU time was not measured.

| Case | Board height | Tilt from horizontal | Original gap | Corrected gap |
|---|---:|---:|---:|---:|
| 12 | 386.114 mm | 39.5118° | 69.548 mm | 129.548 mm |
| 24 | 391.871 mm | 36.8016° | 68.644 mm | 128.644 mm |
| 48 | 268.710 mm | 32.4491° | 56.096 mm | 116.096 mm |

[summary.json](summary.json) contains sanitized scalar counts, exact settings and source/audit hashes. Dense trajectories, caches, host calibration and host paths are excluded. The prior queued baseline and original 50-position results remain unchanged.

The native logger's robot channel is aggregate. Its contact threshold is force strictly greater than 0.01 N; no specific robot link is claimed as a native measurement. The contact derivative independently checks recorded arrays and the existing scalar scorer. Success requires the unchanged complete board route, whole-ball entry and continuous one-second retention.

`sircus/mjlab/gap_flight_correction_verify.py` verifies the completed native study. `sircus/mjlab/gap_flight_results.py` derives the contact summary on CPU after authenticating that proof, the manifest and every consumed native cache. Reproduce the derivative with `--study STUDY --audit AUDIT --output NEW_DIRECTORY`; the output must be outside the immutable study. This does not launch another simulation.

Limits: three post-hoc selected development cases; exact repeats without noise; fixed64 physics; held endpoints rather than driven arm transit. Increased gap changes both clearance and impact energy, so this intervention does not isolate a clearance-only mechanism. No new policy was trained or activated. The 4070 and Isaac setup were untouched.

Next: freeze a clearance-aware selection change and evaluate it on new held-out bucket positions before claiming broad improvement. This small intervention does not revise the original 24/50 selector score.
