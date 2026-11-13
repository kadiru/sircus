# Table-bounce failures and paired shot collection

The [matched test](MJLAB_MATCHED_ROUTE_TEST.md) gave each of three frozen
planners one direct and one table-bounce attempt at the same 50 new bucket
positions. This follow-up diagnoses the saved table attempts and collects
new training data around positions that already supported both routes.

## What failed

**Setup.** The table route had 52 failed setups in 150 requests, compared
with 23 for direct. In 32 of the 52, the model produced no candidate inside
the allowed shot ranges. In 19, the available predictions were either
out of bounds or exceeded a wrist reach bound. One also encountered an
endpoint clearance or pose rejection.

Across the four candidate slots per request, 190 of 600 table predictions
were outside the configured ranges and 127 more exceeded a wrist reach
bound. Of the out-of-bounds predictions, 179 exceeded the board Y range.
The dropper was involved in 107 of the wrist failures. The median source
recipe was 327 mm from its table-shot target, compared with 63 mm for direct.
The table models were adapting from farther away. This was an observed
association, and the setup results rule on the proposed settings only.
For example, at bucket XY (192.4, 160.0) mm all three planners lost all four
table candidates to the action limits, with source buckets 723–868 mm away.

**After release.** Of 98 released table attempts, 36 captured, 61 valid
drops missed and one failed a physics check. All 61 valid misses had a
verified table rebound and none fully entered the bucket.

| First bucket or robot contact relative to the table rebound | Valid misses |
|---|---:|
| Bucket after rebound | 32 |
| Bucket before rebound | 20 |
| Robot before rebound, with no bucket contact | 8 |
| Robot after rebound, with no bucket contact | 1 |

All 32 misses that reached the bucket after the rebound first hit its near
side, and 24 were still rising. Ten struck below the rim-height band. Those
32 shots rebounded a median 202 mm from the bucket center, compared with
251 mm for the 36 successes. This comparison is descriptive across the
sampled layouts and settings. At (586.7, −393.0) mm the ball struck the side
four milliseconds after the rebound, with its center 41.5 mm high and the
rim 183 mm high. The invalid trajectory at (564.9, −287.3) mm was flagged
for crossing the bucket wall after entry.

The setup diagnosis reconstructs the frozen candidate menu and checks
the generated settings against the original bounds and wrist tests. The
flight diagnosis verifies the saved ledger and dense trajectory archive
hashes and uses force episodes and measured velocity reversal to order
the table, bucket and robot contacts. Both use the audited
`intergrid-matched-routes-v1` archive. Reproduce the counts on CPU with
`python -m sircus.mjlab.intergrid_paired_route_diagnosis --help`.

## Collection protocol

Use every exact XY with verified captures by both routes across the
audited training archive and the completed matched test. Four positions
come from the training archive and seven from the matched test. Those
seven now serve as development locations for this collection. Preserve
the old test results as recorded and exclude these coordinates from any
future unseen comparison of models trained on this collection.

Freeze 16 novel direct and 16 novel table proposals per position.
Independently perturb known same-route settings using board position,
height, tilt, heading and release gap. Exclude geometry already present
in the two input archives. Run the existing pose and path checks and
release every admitted shot once in MuJoCo at 64 substeps. Record the
intended route separately from the measured result. Save valid successes
and misses with trace provenance for future training.

The producer is `python -m sircus.mjlab.intergrid_paired_collection_study --help`.
Reproduce the audit with
`python -m sircus.mjlab.intergrid_paired_collection_audit --help`.
The model, trajectories and raw rows remain in the Ubuntu archive.

## New collection results

The 4090 ran 346 drops from 352 proposed settings. Six direct proposals
failed setup. Of the released shots, 137 of 170 direct attempts and 154 of
176 table attempts captured by the requested route. All 11 positions
produced new captures by both routes. These are local variations at proven
positions and measure data yield under this collection strategy.

| Bucket XY (mm) | Drops | Direct captures | Table captures |
|---|---:|---:|---:|
| 215.4, −551.8 | 32 | 16 | 16 |
| 228.6, −612.8 | 32 | 13 | 15 |
| 229.2, −561.8 | 32 | 16 | 15 |
| 353.4, −580.4 | 32 | 9 | 14 |
| 457.0, −608.9 | 32 | 16 | 13 |
| 545.4, −632.3 | 32 | 16 | 15 |
| 756.5, −608.2 | 32 | 16 | 14 |
| 948.8, −286.0 | 32 | 12 | 12 |
| 976.3, −286.8 | 28 | 12 | 12 |
| 1050.0, −425.0 | 32 | 7 | 16 |
| 1050.0, 250.0 | 30 | 4 | 12 |
| **Total** | **346** | **137** | **154** |

The training ledger holds 342 valid new releases, including 291 captures
and 51 misses. Four trajectories failed the physics checks and remain in
the raw ledger. The dataset is
`intergrid-paired-collection-v1/new-training-dataset.jsonl` on Ubuntu.

The independent CPU audit passed 62,087 checks, including both control
sets, all releases, and reconstruction of the training ledger. The
mjlab regression suite ran 2,269 tests with seven skipped.

- Report SHA-256: `1ad93b9680d95624d5f5e5c1580ec367ae7ceb5445a4c1caf768af08ea2f5009`
- Evidence manifest SHA-256: `f47674cf49bd6ee16c2ce0ab7865dc97aa6e743659103c32219dec1cf8a40bbf`
- Training ledger SHA-256: `10b0b5e294c6af85ec0f5569a7f4298ad3b10a685c2fb4eba92dbd1ed56f9b5e`
