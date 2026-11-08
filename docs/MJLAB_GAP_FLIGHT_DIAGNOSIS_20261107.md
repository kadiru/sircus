# Gap comparison: diagnosis and completed correction

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

## Completed targeted correction

One correction increases only the release gap by **60 mm** on three diagnosed early robot-contact cases. Board XYZ, board tilt and bucket placement stay fixed.

| Case | Original gap | Corrected gap |
|---|---:|---:|
| 12 | 69.548 mm | 129.548 mm |
| 24 | 68.644 mm | 128.644 mm |
| 48 | 56.096 mm | 116.096 mm |

All **30/30 native requests completed**, with no setup rejects or invalid physics. Each original and corrected setting received three repetitions; the remaining 12 requests were matched pre/post controls. Independent native audit passed **46,625 checks**.

| Case | Original complete captures | +60 mm gap complete captures | Corrected clean captures |
|---|---:|---:|---:|
| 12 | 0/3 | 3/3 | 0/3 |
| 24 | 0/3 | 3/3 | 0/3 |
| 48 | 0/3 | 3/3 | 0/3 |
| Total | **0/9** | **9/9** | **0/9** |

All **9/9 corrected captures had recorded rim-band/bucket contact; none were clean**. The scorer labels them `assisted_or_marginal_capture`. Cases 12 and 24 briefly exited after entry before later retaining the ball for one continuous second; case 48 did not exit. All nine original repetitions retained the ball for one second but failed the complete route. Recorded robot contact after board rebound occurred in **9/9 originals and 0/9 corrected shots**; corrected shots also had no robot contact anywhere in their four-second traces. All 12 controls behaved as expected: six clean positive captures and six negative misses.

Worker wall time was **225.928 s**; this includes worker overhead and is not a pure GPU-compute measurement. The 4090 ran this bounded experiment; the 4070 and Isaac workload were untouched.

This is a local improvement on three post-hoc development cases, not a new 50-position policy result. Repeats used exact settings, without noise. Increased gap changes impact energy as well as clearance, so this does not prove clearance alone caused the gain. Endpoints were held during the shot; driven arm transit remains untested. No new training or policy activation occurred.

## Code and evidence

- CPU diagnosis: `sircus/mjlab/gap_flight_diagnosis.py`.
- Scalar baseline and source/audit hashes: `baselines/mjlab-gap-flight-diagnosis-20261107/summary.json`.
- Completed correction, exact settings and provenance: `baselines/mjlab-gap-flight-correction-20261107/summary.json`. The earlier queued baseline remains historical and unchanged.
- Interface section: `#gapFlightDiagnosis`; includes all 50 requests, paired outcomes and contact-limited example trajectories.
- Native evidence validation: the frozen plan/source preflight passed 6,810 CPU checks; the completed correction's independent audit passed 46,625 checks and its contact derivative passed 771 checks. Code and publication checks are reported separately at their commit boundary.

Raw trajectories, native caches, host paths, credentials and host calibration are excluded from the baseline. The original study remains immutable. No 128-substep convergence, noise robustness, driven arm transit, grasp, perception or physical validation is established here.

## Next boundary

The bounded experiment is finished; do not resume or rerun in its evidence directory. Preserve the original `gap-flight-correction-v1` and its separate independent proof. Any follow-up needs a new namespace and frozen protocol.

Use the diagnosis to define one clearance-aware selection change, then compare frozen selectors on fresh held-out bucket positions before more training. The current correction was manually targeted to three already observed failures; it is not yet a learned or automatically deployed rule.
