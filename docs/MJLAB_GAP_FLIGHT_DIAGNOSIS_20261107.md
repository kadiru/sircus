# Gap comparison: diagnosis and one queued correction

The completed 50-position comparison remains **24/50 captures for both the original and wider-gap selectors**. Wider proposals gained case 07 and lost case 38. This task diagnoses the original selector's **18 native misses and 8 setup rejections** before more training. It does not rescore the study or train a model.

The CPU reconstruction passed **702 evidence-binding and reconstruction checks**, backed by the original **1,441,075-check native audit v2**. It verifies the frozen source hashes, complete artifact manifest, unchanged requested actions and dense cache columns. All 50 original requests and their 50 wider-gap counterparts remain visible in the interface.

## What failed

| Evidence | Count | Meaning |
|---|---:|---|
| First subsequent contact was a robot | 9 | Five contacts occurred within 50 ms of board rebound. |
| First subsequent contact was the bucket | 6 | This aggregate channel alone does not identify a rim hit. |
| First subsequent contact was the table | 3 | Recorded first flight reached the table. |
| Finite IK search unresolved | 3 | No board-arm branch found in that search; not proof of unreachability. |
| Fixed-base clearance rejected | 2 | Bucket placement conflicts with the fixed dropper base. |
| Endpoint clearance/pose rejected | 2 | Saved branch attempts include board-arm self-clearance failures. |
| Action bounds rejected | 1 | Requested shot lies outside the configured action bounds. |

Four native misses—12, 24, 25 and 48—actually entered and retained the ball for one second, but robot contact invalidated the required complete board route. These remain failures under the unchanged scoring contract.

Only five of the 18 misses have an observed rim-height crossing before another contact. All five crossed short of the bucket center by **158–241 mm**. The other 13 have no measured pre-contact crossing. We do not extend a ballistic curve through a contact and call that extension an observation.

## Concrete examples

| Case | Recorded evidence |
|---|---|
| 48 | Board height **268.710 mm**, tilt **32.449°**, gap **56.096 mm**, board-to-bucket distance **156.118 mm**. Predicted capture score **0.965**; actual robot contact **8.464 ms** after rebound. Ball retained 1 s, complete route failed. |
| 27 | Board height **294.193 mm**, tilt **17.204°**, gap **271.578 mm**. Recorded rim crossing was **241.496 mm short** of center; first subsequent contact was the table; no entry. |
| 16 | First saved branch had negligible endpoint position error, but only **1.419 mm** clearance between board-arm links 4 and 6. Setup rejected; no native flight. |
| 01 | Finite search found **10 dropper-arm branches, 0 board-arm branches**. No native flight. |
| 06 | Saved home clearance witness identifies the fixed dropper base and bucket wall; conservative minimum clearance is **0 mm**. |
| 41 | Requested release gap **36.948 mm**; rejected by action bounds. |

Capture scores are model outputs, not calibrated probabilities. The dense logger records an aggregate robot-contact channel, not the exact robot link. Its original link identity remains unknown; no reconstructed link is presented here as a native measurement.

## Next test: frozen and queued, results pending

One correction increases only the release gap by **60 mm** on three diagnosed early robot-contact cases. Board XYZ, board tilt and bucket placement stay fixed.

| Case | Original gap | Corrected gap |
|---|---:|---:|
| 12 | 69.548 mm | 129.548 mm |
| 24 | 68.644 mm | 128.644 mm |
| 48 | 56.096 mm | 116.096 mm |

The fixed64 test requests three repetitions of each original and corrected setting, plus 12 matched pre/post controls: **30 requests maximum**. It is queued for the 4090; the 4070 and Isaac workload remain untouched. The increased gap changes impact energy as well as clearance, so improvement is not assumed. These are development examples selected after diagnosis, not fresh held-out evidence. No new training is scheduled by this test.

## Code and evidence

- CPU diagnosis: `sircus/mjlab/gap_flight_diagnosis.py`.
- Scalar baseline and source/audit hashes: `baselines/mjlab-gap-flight-diagnosis-20261107/summary.json`.
- Interface section: `#gapFlightDiagnosis`; includes all 50 requests, paired outcomes and contact-limited example trajectories.
- Validation: complete `scripts/test.sh` passed; 38 focused diagnosis/correction/publication tests and eight UI tests passed. The frozen plan and source preflight passed 6,810 CPU checks; native correction outcomes are pending.

Raw trajectories, native caches, host paths, credentials and host calibration are excluded from the baseline. The original study remains immutable. No 128-substep convergence, noise robustness, driven arm transit, grasp, perception or physical validation is established here.

## Continue the queued experiment

Use the persistent new `gap_flight_pipeline` process; do not start a duplicate.
Its separate pipeline JSON records child PID, queue/error state and audit result.
The worker output namespace is `gap-flight-correction-v1`; diagnostic input is
`gap-flight-diagnosis-v1`. The derived report publisher refreshes the existing
viewer while waiting and after execution, without restarting that viewer.

If the bounded queue expires, inspect the saved error and GPU ownership first.
Do not overwrite this contract or rerun in its evidence directory. A new attempt
requires a new namespace. A completed raw report is not enough: the separate
`gap_flight_correction_verify` proof must pass before publishing successful counts.
