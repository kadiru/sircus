# Replaying feasible shots at missed bucket positions

The [100-position comparison](MJLAB_DIRECT_FIRST_COMPARISON.md) recorded
16 ordinary flight misses for the route selector, five positions where
neither route passed setup, and one invalid trajectory. This follow-up
tests whether a different shot was already available at each ordinary miss.

- Reconstruct the original candidates and selections from the audited
  study. Freeze the complete list of 16 missed positions and their
  admitted commands before running the follow-up.
- Repeat the selected shot once at each position. Run every other distinct
  admitted command once, using the archived pose, joints and path proof.
  Candidate IDs that lead to the same command share one replay.
- Record which commands previously ran under direct-first. Compare their
  repeated results against the old outcomes and count newly tested
  alternatives by requested route. A capture requires a board rebound
  and one continuous second with the whole ball inside the bucket.
- For the five setups that failed before release, report whether each route
  produced any candidates and why the arm checks refused each one.
- Use the original 64-substep simulator and the same nine controls before
  and after. The CPU audit reconstructs the command list, verifies the
  archived setup proofs and rescores the recorded trajectories.

The selected positions serve as development evidence. The producer is
`python -m sircus.mjlab.intergrid_candidate_replay_study --help` and the
verifier is `python -m sircus.mjlab.intergrid_candidate_replay_audit --help`.
Use the audited `intergrid-route-fallback-v1` and `fixed35-sixhour-v2`
archives on Ubuntu. Native simulation uses the 4090 via the shared GPU
lock. Keep the `intergrid-candidate-replay-v1` archive outside Git.

## Audited replay results

| Shots | Drops | Captures | Positions with a capture |
|---|---:|---:|---:|
| Repeat of the selector's chosen shot | 16 | 0 | 0/16 |
| Previously untested alternatives | 55 | 28 | 10/16 |
| Alternatives already tried by direct-first | 2 | 1 | 1/2 |

The 28 new captures included 26 observed table bounces and two direct
shots. Five positions had a new capture using the selector's intended
route and seven had one using the other route. Two positions had both.
The two direct-first replays agreed with their archived outcomes. The
remaining six positions produced zero captures from novel alternatives.
One of those six had the already-known direct-first capture.

| Earlier position | Bucket XY, mm | Direct setup | Table setup |
|---|---:|---|---|
| 29 | 603.8, −168.2 | Four board placements violated the 150 mm minimum board-to-bucket distance | All four predicted board Y values exceeded the upper bound |
| 31 | 725.5, −88.2 | All four predicted release gaps were below the minimum. One also exceeded the board height cap | All four predicted board Y values exceeded the upper bound |
| 33 | 1028.9, −19.0 | Wrist reach rejected three candidates. The endpoint clearance or pose check rejected one | Wrist reach rejected all four |
| 81 | 775.2, −27.8 | All four predicted release gaps were below the minimum | Two board tilts were too low and two board Y values were above the upper bound |
| 90 | 618.4, 138.6 | The endpoint clearance or pose check rejected all four candidates | All four board tilts were too low. Three also fell below the board Y bound |

The setup table combines the archived rejection reasons with a CPU
reconstruction of the frozen model predictions before action filtering.
The reconstruction applied the same specialist coefficients to the
original four starting shots per route, using the expanded table library.

## Ranking screen

The [previous fixed-grid ranker](MJLAB_INTERGRID_COMPARISON.md) picked
three captures when scored against the complete measured candidate set
at these 16 miss positions. A seven-feature logistic model picked four.
A uniform choice among the same candidates would average 5.275 captures.
The measured candidate set contained a capture at ten positions.

The logistic model used the intended route, source-to-target distance,
normalized change in shot parameters, board-to-bucket distance, board
height, tilt, and release gap. The data combined the archived 100-position
study and this replay. It excluded invalid trajectories, combined exact
repeat commands, and omitted conflicting labels. This left 158 labeled
commands at 94 positions. Each position contributed equal total weight.
Features were standardized from training data in each fold. The fit
minimized weighted binary cross-entropy plus 0.05 times the squared
non-intercept coefficients. Each scored position and all its drops were
held out of its fit. The mean within-position AUC over the ten positions
with both a capture and a miss was 0.37.

These positions were selected for having missed with the incumbent shot.
The screen therefore measures ranking on difficult development cases.
Both ranking approaches fell below the uniform expectation on these
cases, so neither advanced to a fresh one-drop comparison.

The 4090 ran all 73 replays and 18 controls. Both control sets and
the 16,665-check CPU audit passed. The mjlab suite ran 2,281 tests with
seven skipped and zero failures. The 4070 and Isaac were left alone.

- Producer commit `79ec5061412597998f31133d0c187842865cb4f8`
- Report SHA-256 `97b2cafa32c53591fadea0b797dc778501bd69d7fdb3472b39341365d9ba9ea1`
- Evidence manifest SHA-256 `6e3d4f69e2a89dee3a08b8140a821320c763804d1dd3d9bcd07f14da992a6f0b`
- Replay plan SHA-256 `b32fc4f0a3d55ae6602179704c9cc3ffd031fe1d446f6a121dcae95c64547791`
- Audit SHA-256 `17c2daa3763f46d7d3a3f4a640b6d22e608e28ec727fc0dea9ace1840a6aad0c`
