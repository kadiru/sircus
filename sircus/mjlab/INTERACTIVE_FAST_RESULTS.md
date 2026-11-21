# Live fast-planner comparison results

The live comparison finished on **November 21, 2026**. On the same ten new
bucket positions, the faster planner captured **7/10**, versus **8/10** for
the original planner. Mean planning wait fell from **46.12 to 21.16 seconds
(54.1%)**, but one position lost a capture because the faster planner did not
finish checking a shot before its deadline. All 20 attempts terminated and
the independent audit passed for all 17 actual releases.

## What was compared

The [frozen protocol](INTERACTIVE_FAST_ACCEPTANCE.md) tests the two deployed
HTTP planning paths: the frozen learned selector with worker reuse, early
return and a roughly 30-second decision limit, versus the original live
planner without that decision limit. This measures their interactive speed
and capture tradeoff; it does not isolate the effect of early return alone.
No model, contact handling or scoring was changed.

There were four interior, four grid-edge and two near-base bucket positions.
The seed was `2026120101`; coordinates and the exclusion inventory were frozen
before testing, with at least 5 mm separation from the recorded earlier
positions. Planner order alternated. Each planner got one plan and at most one
drop per position at 64 substeps; an unavailable setup stayed in the denominator.

| Measurement | Faster planner | Original planner |
|---|---:|---:|
| Captures / all positions | **7/10 (70%)** | **8/10 (80%)** |
| Actual releases | 7 | 10 |
| No checked shot ready | 3 | 0 |
| Released shots without capture | 0 | 2 |
| Technical failures | 0 | 0 |
| Direct / table-assisted captures | 6 / 1 | 6 / 2 |
| Mean HTTP planning wait | **21.16 s** | **46.12 s** |
| Median HTTP planning wait | 20.21 s | 26.46 s |
| 90th-percentile HTTP planning wait | 30.20 s | 76.94 s |
| Longest HTTP planning wait | 30.21 s | 172.40 s |
| Mean native drop request | 44.47 s | 43.66 s |

Planning wait is measured from HTTP submission through observed ready/declined
status, including polling. It excludes waiting for a free GPU and the drop.
The native drop request includes startup, simulation and scoring. These are
loopback service measurements, not browser rendering latency. Each planning
summary contains ten samples; drop summaries contain seven and ten samples.

## Which outcomes differed

Both planners captured at seven positions. The faster planner gained none and
lost one capture; neither captured at two positions.

| Bucket (X, Y), mm | Faster planner | Original planner |
|---|---|---|
| (971.2, −285.4), case 02 | No checked shot by 30.17 s; no release | Direct capture after 172.40 s planning |
| (986.1, 115.8), case 04 | No checked shot by 30.18 s; no release | Released miss after 76.94 s planning |
| (1119.4, −5.7), case 10 | No checked shot by 30.21 s; no release | Released miss after 41.85 s planning |

The faster planner's three declines mean its finite search did not produce a
checked shot in time, not that the arms cannot reach those locations. All seven
shots it did release captured in this batch, but reporting 7/7 alone would hide
those three unresolved positions. Case 02 supports retaining the original
planner as an explicit fallback when the operator accepts a longer wait.

## Interrupted run and scope

The first fast plan was prepared on November 17 in **16.079 seconds** and saved
while another task occupied the 4090. The November 21 resume reused that exact
plan, without repeating a drop or adding the multi-day pause to planning time.
The report retains two interruptions: the original one-hour foreign-GPU wait
expired, and the first resume failed its preflight because the explicit 4090
selector was missing. Neither submitted a drop. After correcting that environment
setting, the run resumed the same saved plan. Excluding case 01 from both
methods gives nine-position mean waits of **21.73 versus 46.15 seconds**;
the speed difference therefore does not depend on the older first timing.

Capture requires a board rebound, whole-ball entry and one continuous second
inside the bucket. Table and rim assistance count; robot-assisted outcomes
remain separate. Inputs are simulator geometry and the native drops hold both
arms at the selected endpoints. This is not perception, driven-transit or
physical-robot validation, and ten positions do not establish all-table
reliability. The existing audited physics controls were retained; no new control
drop set or calibration was run for this interface check.

Keep the fast interactive default with the original planner available explicitly.
Do not claim that speed improved without any capture loss: this batch shows a
54.1% shorter mean wait and one fewer first-drop capture. The next focused speed
improvement is to shorten arm configuration/path checks that consume the deadline,
while preserving their validity. No additional training is implied by this result.

## Evidence and reproduction

- Finalized **November 21, 2026, 00:36:31 PST**; UI section: `#interactiveFastAcceptance`.
- Study: `/home/kadir/sircus-mjlab/output/interactive-fast-acceptance-v1`.
- Runtime evidence: `/home/kadir/sircus-mjlab/studies/interactive-fast-v1/evidence`.
- `positions.json` SHA-256: `4e821761713f3acb951b1659d2fb25ec6796e4963b0b0afb5f127e04b4078ca8`.
- `report.json` SHA-256: `aad7375ddfdaa3feef9f9c7819b5f86b6e57017d349a6bd250b052f3ffcc0eae`.
- `audit.json` SHA-256: `fff5e0ca13a35bdd4a09fb1f40a43c3be81d36d1dbfbb8465aa324ee34fa573c`.
- Audited results digest: `66f9ae8300991f16080400b33ee8f49c65003b7fabd0453cb3e9d65e79ebcfac`.
- Frozen learned checkpoint SHA-256: `3ea83197149204bb92782f9efd9a7ee985d6c7bbbfb0823bd068642eebdd1a1e`.
- The frozen file also binds the system/fast-study source hashes, exclusions,
  scene and scoring inputs. The audit binds all **17** unique native traces to
  their preview, bucket, exact joint commands and release request, independently
  re-scores retention/routes, checks the native64 clock, and verifies displayed
  trajectories. Stale and duplicate drop requests were rejected.
- Reproduce or rerun the CPU-only audit using the commands in the
  [acceptance protocol](INTERACTIVE_FAST_ACCEPTANCE.md). Raw observations and
  runtime artifacts remain outside Git. The 4070 workload and Isaac setup were
  untouched.
