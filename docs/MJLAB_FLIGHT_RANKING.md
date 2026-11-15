# Predicting which bounce shot to use

The frozen direct-first policy tries the first admitted direct shot and falls
back to a table-bounce shot when no direct setup passes. Replaying other
admitted commands found captures at 10 of the 16 positions where the earlier
route selector had released and missed. This study tests whether predicting the
ball path can pick a working command before release.
[Earlier replay protocol and results](MJLAB_MISSED_SHOT_REPLAY.md).

For each candidate, the ranker places both arms at their admitted joint angles.
It computes the contact point and incoming speed for a vertical drop onto the
board, applies the fitted rebound velocity ratios, and projects the resulting
arc to the bucket rim. If the first arc lacks a clear rim passage and reaches
the tabletop, it projects a second arc starting at the predicted table impact.
The score is the smallest predicted clearance along the route, including the
ball's radius at the rim and the distance to the held arms after the board
rebound. Larger clearance scores rank first. Rim impacts, spin, and retention
are outside the trajectory model. The native scorer still measures whole-ball
entry and one continuous second inside the bucket.

The velocity ratios came from audited native traces at other bucket positions.
For each isolated contact, velocities from the free flight before and after
contact were adjusted to the contact midpoint using gravity. The fit takes
the median within each position and then across positions.

| Surface | Isolated rebounds | Bucket positions | Normal speed ratio | Tangential speed ratio |
|---|---:|---:|---:|---:|
| Board | 85 | 78 | 0.6510 | 0.8197 |
| Table | 14 | 13 | 0.6480 | 0.8874 |

## Screen on the saved replay menus

With 73 shots across 16 positions, the flight rule chose **9/16** captures.
Uniform choice from the same options would average **5.275/16**. The model
ranked a capture above a miss in 93.7% of within-position pairs when weighting
the ten positions with mixed outcomes equally. All nine top-ranked captures
used table rebounds. At `(971, −611)` mm it favored a miss over two measured
captures, illustrating the error in approximate rebound and rim clearance
estimates.

The method and coefficients were fixed before ranking the replay menus. The
screen used previously selected miss positions. The holdout uses 50 placements
excluded from all archived coordinates using the existing placement separation
rule. Both methods share up to four candidate proposals per route and select at
most one drop at each position. The native run uses 64 substeps and nine
controls before and after.

- Frozen rule SHA-256 `3d22d455000dcb2269adc76f84fbb63da68952c17580a51dc24ce06ee751c134`
- Replay screen report SHA-256 `3966dbc612cff546a0cbcc6257fa5473845e7f4342f98bfe95b47f5ed59f369b`

## Fresh comparison preparation

The same candidate pool produced a certified setup for both methods at 47
positions. The physics rule changed the selected command at 33 of those.
It kept the intended route at 24 and changed it at nine. The remaining 14
positions used an identical command. A CPU check of the frozen choices and
their setup proofs passed 2,525 assertions.

- Frozen choices SHA-256 `2257ba8526b035ab588f5f6bd72e39abd1ba1870d9bf3c553b6d34d6cf6db304`

## Feasible setups for five earlier refusals

A CPU search applied the existing setup checks to 16 nearby-shot variations
per route at each of the five placements. Each result includes a certified arm
pose and approach path. The search released zero balls.

| Bucket X, Y (mm) | Feasible direct setups | Feasible table setups |
|---|---:|---:|
| 604, −168 | 6 | 3 |
| 725, −88 | 8 | 4 |
| 1029, −19 | 1 | 3 |
| 775, −28 | 10 | 3 |
| 618, 139 | 3 | 2 |
| Total | 28 | 15 |

The independent check passed 1,056 assertions across 160 proposals. The
expanded proposal search was applied to these five development positions.
The matched holdout uses the original proposal budget for both selectors.

- Pose search plan SHA-256 `f03f2033e895a1c7358589258cb344f4b5cb0c66dde5b4f735568cbcc912b105`
- Pose search audit SHA-256 `50b27bbf23bf260e56f4c77e33f02c5dee39db3244fece646857f4d1658630ad`

## Reproduction

Use the pinned Ubuntu mjlab environment and the authenticated
`fixed35-sixhour-v2`, `corrected-grid-v1`, `intergrid-route-fallback-v1` and
`intergrid-candidate-replay-v1` archives. Keep evidence outputs outside Git.

1. Run `python -m sircus.mjlab.intergrid_flight_ranking_screen --help`. With
   CUDA hidden, supply the two study/audit pairs, assets and fresh output
   path. The program writes the predictions and rule before joining them to
   the saved outcome rows. The report binds the exact predictions and rule.
2. Run `python -m sircus.mjlab.intergrid_flight_ranking_study --help` for the
   50-position comparison. Pass the screen report to `--rule`, the previous
   matched comparison study/audit pairs, the replay study/audit pair, and the
   pinned runtime inputs. Use the 4090 UUID and existing cooperative GPU
   lease. Repeat the same command with `--audit` and CUDA hidden to verify
   frozen choices and rescore the native traces independently.
3. Run `python -m sircus.mjlab.intergrid_setup_recovery --help` for the
   CPU-only pose search. Pass the parent, fallback and route-refit
   study/audit pairs, the pinned reference and assets, and a fresh output
   folder. Use the same arguments plus `--audit` to verify the saved proofs.
