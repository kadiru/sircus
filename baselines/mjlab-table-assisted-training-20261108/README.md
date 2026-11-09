# Table-assisted imitation and short PPO comparison

Completed native MuJoCo Warp study, producer `b11e8b9`, fixed **64 substeps**.
The independent audit passed **823,442 checks**. Source evidence, full traces
and model files remain outside this compact repository package.

All four frozen policies made one request at each of the **same 50 fresh bucket
positions**, using the new v3 rule: a measured board rebound, whole-ball bucket
entry and one continuous second retained. Table and bucket contacts are allowed;
robot, floor and unattributed contacts still prevent success. Every setup decline
stays in the requested-position denominator. Historical v2 results are unchanged.

| Frozen policy | Captures / 50 | Direct / table-contact captures | Clean entries | Setup declines | Released misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old imitation | 9 (18%) | 9 / 0 | 4 | 6 | 35 |
| Old PPO | 8 (16%) | 8 / 0 | 3 | 5 | 37 |
| New imitation | 14 (28%) | 8 / 6 | 5 | 4 | 32 |
| New PPO | **15 (30%)** | **9 / 6** | **8** | **4** | **31** |

The larger improvement accompanies the new demonstrations and scoring. The
additional PPO gain over new imitation is just **one capture** in this sample.

## What was trained

The new bootstrap used all **7,168 historical TRAIN requests**, including 115
newly credited captures, for critic fitting. The actor used **one selected
successful teacher per TRAIN layout**: 27 teachers, including five table-assisted
teachers, versus 22 in the old imitation model. Sixteen teacher selections changed.
The 115 relabeled outcomes are not 115 actor demonstrations.

PPO then made **512 new TRAIN requests in 16 updates**. Validation captured
5/12 at initialization, 5/12 after 256 requests and 6/12 after 512. The prescribed
validation rule selected the 512-request checkpoint before any new TEST outcome.
Including validation, controls and four frozen test methods, the study made
**766 requests** in **4,585.6 seconds (76.4 minutes)**. Direct-capture,
table-contact-capture and miss controls each passed three repeats before and after.

Exact-action IK, clearance checks, geometric priors and sampled setup paths were
preserved. The policy was not projected to another action or resampled after a
rejection. Flights used held robot endpoints; the experiment did not learn or
validate pickup, carrying or driven arm transit.

## Paired position comparison

| New PPO compared with | New-only captures | Reference-only captures | Net gain | Descriptive 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Old imitation | 8 | 2 | +12 percentage points | 0 to +24 points |
| Old PPO | 8 | 1 | +14 points | +4 to +26 points |
| New imitation | 2 | 1 | +2 points | −4 to +10 points |

Intervals use 100,000 paired layout bootstrap resamples, with NumPy PCG64 seeds
20261109, 20261110 and 20261111 respectively. Each resample draws 50 shared
positions with replacement, preserving the two policies' outcomes at each
position; the interval is the 2.5th–97.5th percentile of capture-rate differences.
These are posthoc descriptive intervals for this one run and this position sample,
not uncertainty over training seeds, repeated drops or multiple-comparison-adjusted
hypothesis tests.

## What remains limited

- **Coverage:** the coordinate generator excluded 1,037 inventoried historical
  coordinates within 5 mm. These are new positions under that explicit rule,
  not a balanced table-wide test. Thirty of the 50 lie at X > 1 m; 22 of 35
  descriptive 7×5 bins contain a test position.
- **Far captures:** the six new-PPO table-contact captures occur 643–703 mm from
  the board. Their board heights are 423–428 mm, tilt approximately 32°, and drop
  gaps 141–158 mm. They cluster in one far-table region. A successful single
  attempt is not a reliable-shot qualification.
- **Contact attribution:** “table-assisted” labels record table force intervals
  above 0.01 N before the scoring terminal. The labels alone do not prove causal
  board → table → bucket ordering or that a particular table contact was a rebound.
- **Remaining failures:** new PPO has 31 released misses: 13 with forbidden
  contacts and 18 without. All 31 bounced from the board; none achieved whole-ball
  entry. Its four setup declines are two unresolved finite IK searches and two
  fixed-base clearance declines. No TEST shot had invalid physics.
- **Attribution:** new imitation has more historical training exposure and a
  different teacher set than old imitation. New PPO adds 512 attempts; old PPO
  previously used 1,024. This is not an equal-budget or reward-only causal study.
- **Scope:** one training seed, one attempt per test position, privileged geometry,
  fixed64 simulation. No timestep-convergence, noise, perception, physical transfer,
  universal feasibility or all-table success claim follows.

## Compact evidence

- [summary.json](summary.json): audited counts, validation checkpoints, paired
  comparisons, uncertainty method, coordinate occupancy and evidence/model hashes.
- [test-positions.csv](test-positions.csv): derived per-position binary outcomes
  for all four methods, sufficient to reconstruct the paired comparisons.
- [Training contract](../../sircus/mjlab/TABLE_ASSISTED_TRAINING.md) and
  [independent checker](../../sircus/mjlab/table_training_verify.py).

Report SHA-256: `4605bcb801351ae91ad16b71a033ceb3866027db621f9d84227f23b1ef257e95`.
Independent audit SHA-256:
`6e50d87ddb7b2d4b37debbc2a5ac96507ff40bc236223277121acd385b8ec91b`.
