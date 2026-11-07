# Frozen flight selector — development checkpoint

This follow-up changes candidate ranking, not the trained models. It combines
the five saved rebound-velocity predictions with requested board geometry,
gravity and the finite ball's passage through the bucket opening.
[Protocol](../../sircus/mjlab/CONSISTENT_FLIGHT.md);
[read-only interface](http://127.0.0.1:18781/#consistentFlightStudy).

## What the original examples show

These are **post-hoc development diagnostics**, not fresh test results. All
60 original ensemble TEST requests retain their original measured outcomes.
Model-only reranking changes the top candidate in 64 of the 84 original
VAL/TEST banks; those different candidates have unknown outcomes.

| Original native outcomes | Requests | Negative predicted opening margin |
|---|---:|---:|
| Captured | 24 | 13 |
| Flight missed | 25 | 25 |
| Setup rejected | 11 | 9 |

Among model scores at least 99%, all five native misses have a negative margin,
but so does one of four captures. Therefore this proxy **must not be used as
a hard capture/miss classifier**. It is conservative, does not simulate rim
contacts or retention, and uses an ideal contact origin with a velocity model
trained after finite contact. Its ability to improve selection remains unproven.

| Original case | Old model score | New conservative margin | Recorded outcome |
|---|---:|---:|---|
| 29 | 99.654% | −71.5 mm | Miss |
| 39 | 99.811% | +5.1 mm | Capture |
| 55 | 99.774% | −65.7 mm | Capture |

The margin uses the worst available ensemble member over the whole-ball
downward rim passage. It is not the old closest-3D-opening-distance prediction.
[Scalar data and source hashes](development-summary.json) preserve these
figures without publishing raw native traces or runtime configuration.

## Fresh comparison

The bounded protocol freezes 30 new bucket positions, comparing saved imitation,
saved PPO, original ensemble ranking and the new ranking. There are 120 TEST
requests plus 12 matched controls, fixed 64 substeps, no noise or retraining.
Ranked methods use identical 16-candidate and at-most-four-geometry-check budgets.
All decisions freeze before GPU acquisition and fresh outcomes; setup rejects
remain in the denominator. Results require a separate dense native audit.

The first preparation attempt stopped before any native request on an old
literal candidate with a −0.984 mm drop gap. Its evidence is preserved. The
fix retains that candidate unchanged, records unavailable nominal flight data
and ranks it last. It does not clamp the gap, fabricate a flight or claim a
native rejection. The corrected isolated run uses source
`efdacccd4fb587312f272d68307e607a406ca4ff`.

At this development checkpoint, fresh native results remain pending. Consult
the UI and live worker state for progress. The exclusive 4090 queue expires
after 24 hours and acquired execution is limited to one hour; existing jobs
are not interrupted. The 4070, Isaac and interactive solver are unchanged.
This work remains draft PR #83, stacked on diagnosis PR #82.

Validation: 42 focused Python tests on Mac and the pinned Ubuntu environment,
six UI tests, the full mock/headless project suite and GitHub CI. The UI was
visually checked against original case 29. None of these tests substitutes
for the fresh comparison. No hardware or service restart was involved.
