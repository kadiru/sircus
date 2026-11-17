# Prepared seed22 location-breadth comparison — November 16, 2026

**Preparation only: zero simulator starts, zero new shot requests and zero
learning steps. No performance improvement has been measured by this work.**

Two future copies of the original seed22 model would each receive **8,960 new
TRAIN requests**. A repeats **31 usable original locations**; B uses **8,960
distinct valid locations**. Both inherit the same trained weights and Adam
state. The detailed [protocol](../../sircus/isaac/LOCATION_COMPARISON.md) fixes
shared learning, geometry, assessment and failure rules.

The original Isaac 35 are different from mjlab's fixed grid. Static source
geometry identified **four bucket/base overlaps**, not the two positions in
mjlab. Excluded IDs and conservative enclosing-circle clearances:

| Original TRAIN ID | Nearest base | Enclosure clearance (mm) | Positive overlap witness radius (mm) |
|---|---|---:|---:|
| train-021 | Arm10 | -95.582 | 4.000 |
| train-027 | Arm14 | -126.958 | 4.000 |
| train-028 | Arm10 | -102.602 | 4.000 |
| train-034 | Arm14 | -23.920 | 1.433 |

Negative enclosure clearance alone is not an overlap proof. The final column
comes from independent source-convex-geometry witnesses. This does not certify
native cooked PhysX geometry, physical calibration or successful arm motion.
Historical results retain their original target counts and classifications.

Prepared measurements:

| Cohort | Positions | Purpose |
|---|---:|---|
| Monitoring | 35 | Learning curve; no checkpoint selection |
| Primary fresh TEST | 70 | Capture and clean-capture rates on new targets |
| Secondary local TEST | 25 | Sensitivity to a 50 mm bucket displacement |

The local cohort could not place a fresh valid offset for `train-003`,
`train-017`, `train-020`, `train-024`, `train-025`, `train-032` within the
predeclared finite geometry search. These six are recorded as unsampled, not
treated as failures or replaced with easier locations.

The 130 held-out points were checked against **1,131 known historical
coordinates**. Minimum historical distance is **25.025 mm**; minimum held-out
pair distance **25.340 mm**; minimum TRAIN/held-out distance **25.0004 mm**.
New broad TRAIN and primary TEST cover 33 and 25 of the original spatial tiles,
respectively; base occupancy and freshness exclusions condition the domain.
The primary denominator remains 70 for each model.

Proposed accounting: **17,920 TRAIN + 315 monitoring + 285 final assessment =
18,520 total requests**. Rejects after policy dispatch remain charged. Invalid
bucket placements are filtered before any request schedule exists. Training
budgets are identical; A's targets receive 289 or 290 requests each. All target
choices were made without reading outcome values.

Verified by 18 synthetic/mock tests, deterministic regeneration of the actual
package, a separate scalar distance/placement check, and independent review.
Review found a re-signed geometry-receipt loophole; the preparer now pins the
verified receipt fingerprint, with a regression check for radius tampering.
Preparation is not a production audit or launch approval.

Private package: `.runtime/seed22-location-comparison-prep-20261116/package-v1/`.
Its full coordinate schedules, source receipt and history inventory remain
outside Git. Compact identities are in [summary.json](summary.json).
Original checkpoint, raw evidence, benchmark source and runtime services are
unchanged. No simulator or training launcher was added or invoked.
