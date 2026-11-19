# Calibrated 650 mm engineering replay

This finite experiment replays the exact historical shallow shot with the frozen
current-ball simulated contact calibration. It does not refit the calibration,
relabel historical results, train a policy, or validate arm transit paths.

The candidate holds the board center at **380 mm**, tilt **28° from horizontal**,
and board-to-bucket horizontal distance **650 mm**. The exact gap is **386.8 mm**;
the ball center starts at **804.3 mm**. Nominal source joints and explicit bucket
coordinates are preserved. The board is prescribed at its endpoint throughout
the flight, rather than being moved by a learned actuator controller.

## Frozen protocol

Three original ordered groups each run three nominal copies at both 64 and 128
native substeps per 240 Hz outer interval: a previously calibrated positive
control, the candidate, and the historical 750 mm negative control. The scoring
uses the same frozen compiled cavity/contact envelope at both resolutions.
Capture needs board rebound, downward whole-ball mouth entry, and one continuous
second of containment. Partial exits reset retention; external assistance and
invalid physics cannot qualify. The independent CPU oracle rescans dense native
observations and contact geometry.

The gap window opens only if all of these pass:

- Positive control and candidate: 3/3 captures at **each** resolution.
- Negative control: 0/3 captures at each resolution, with valid native releases.
- Every replay has valid physics and agrees with the independent scorer.
- Candidate ball positions at 350 ms differ by at most 5 mm across resolutions.

Only then run gaps **376.8, 380.8, 384.8, 386.8, 388.8 mm**, three nominal copies
per resolution. Board joints, height, tilt and bucket remain fixed; only the
dropper solves the changed gap on a local branch within 0.10 rad. All geometry,
including conditional requests, is frozen before outcomes. Budget: 18 initial,
at most 48 total requests. Rejected endpoints remain in denominators. Identical
repeats are engineering checks, not fresh-noise reliability estimates.

Every endpoint needs fresh CPU 5 mm clearance, full bucket tabletop footprint and
native collider/pose agreement. Calibration, assets, packages, source and evidence
are checksummed. The historical and positive-control scenes differ only in an
unused default bucket center field; this one explicit exception requires a full
canonical reference-scene hash match. Actual shot bucket coordinates never change.
The reported bucket height remains 175 mm with overall-vs-interior interpretation
unconfirmed; the model retains its prior provisional 175 mm interior + 8 mm base.

## Run on Ubuntu

Use a new output/source namespace and the pinned environment in `requirements.txt`.
The worker enforces the 4090 UUID and shared GPU lock, and does not touch the 4070
or existing Isaac runtime. The live viewer receives a new derived report key;
no server restart or policy activation is required.

```bash
python -m sircus.mjlab.replay650_pipeline \
  --old-study OLD_SHALLOW_EVIDENCE \
  --positive-study CALIBRATED_CONTROL_EVIDENCE \
  --positive-audit CALIBRATED_CONTROL_AUDIT \
  --calibration-report FROZEN_CALIBRATION \
  --verification-report FROZEN_VERIFICATION \
  --assets ASSET_DIRECTORY --output NEW_EVIDENCE_DIRECTORY \
  --gpu-lock SHARED_4090_LOCK --display EXISTING_DISPLAY_JSON \
  --batch 16 --queue-timeout 14400 --wall-time-limit 7200
```

The independent audit is `python -m sircus.mjlab.replay650_verify --help`.
Auditor v2 reads cadence from the actual frozen fit/verification fixture records;
the original v1 checker incorrectly expected a timestep inside the contact-only
score configuration. The revised checker runs from a separate snapshot with
`--source` pointing to the original immutable producer. It records its own and
imported dependency hashes separately. No native observations or scoring rules
change, and no physics retry is needed for this offline correction.
The additive HTML/CSS/module fragment is `web/replay650.html`; inline its explicit
style/script placeholders when the existing viewer does not serve new filenames.
The panel makes a read-only GET to `/api/study` and requires the exact completed
report, matching fingerprint, nonempty passed full audit and byte bindings before
labeling results verified. Older study panels remain intact.

Raw dense observations and joint arrays stay outside Git. Curated results and
proof hashes belong in `baselines/mjlab-calibrated650-20261104/`.

Startup attempt `replay650-v1` stopped before evidence allocation or native
flights because the historical package dictionary also records SciPy1.15.3.
The corrected producer/auditor pin that exact eighth dependency; no runtime
package was installed, changed or downgraded. Its startup logs remain separate.
