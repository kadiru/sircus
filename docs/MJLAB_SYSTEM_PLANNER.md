# One-drop planning for a new bucket position

The goal is to place the bucket at a new point on the simulated table and have
both arms set up one shot that captures the ball on its first release. The
[previous benchmark](MJLAB_FLIGHT_RANKING.md) captured at 41 of 50 fresh
positions. It found a valid arm setup at 47, so three positions received no
drop and six released shots missed.

This study tests the two gaps separately on the same 50 new positions:

| Planner | How it chooses a shot |
|---|---|
| Physics baseline | Checks up to four shot candidates per route. Ranks the feasible ones by predicted clearance at the arms and bucket rim. |
| Wider setup search | Uses exactly the baseline shot if one exists. Otherwise checks up to 16 additional candidates per route and uses the same physics ranking. |
| Learned shot ranker | Uses the wider planner's feasible candidates and learns when to change their order. It enters the final test only if it beats the physics ranking on development positions. |

A candidate specifies board position and tilt and where to release the ball
above it. Both arms must pass the existing joint and motion path checks before
release. Direct board-to-bucket and board-to-table-to-bucket routes are both
eligible. A capture requires a board bounce, whole-ball entry and one
continuous second in the bucket, with no disqualifying robot or floor contact.

## Data collection and training

- Sample 50 development positions and 50 evaluation positions from the same
  fixed table geometry, away from archived positions and each other. Commit
  the coordinates and their provenance to the run's evidence before releases.
- Replay up to eight distinct feasible shots per development position at
  64 physics substeps. The simulator records the arm setup, ball trajectory,
  contacts, and capture result. Refused setups remain separate from misses.
- Fit a small linear ranking model to compare captures and misses at the same
  location. The inputs are the candidate geometry, selected joint angles,
  and physics estimates such as rim and arm clearance. The training target is
  the observed result. The output adjusts the existing physics score before
  release. No sensor observations or released ball trajectories enter inference.
- During five-fold validation, hold out entire bucket positions and compare
  each ranker's one selected shot there. If the learned ranker captures more,
  refit it on all development data and freeze the weights before evaluation.
- Run each qualifying planner independently with up to one release per fresh
  position. Report captures and feasible setups over all 50 positions, including
  refusals, plus the time spent on CPU pose and path checks. Use matched
  controls before and after and independently rescore the saved traces.

## Reproduction

Use the pinned Ubuntu mjlab environment. Run
`python -m sircus.mjlab.intergrid_system_study --help` for the required prior
study/audit pairs, frozen physics rule, reference scene, assets, cooperative
GPU lease, and output path. Use the RTX 4090 UUID for native simulation.
The run keeps the 4070 and Isaac jobs untouched, waits for its own lease,
and caps its use of that lease at seven hours. The CPU audit uses the same
study arguments with `--audit` and CUDA hidden. Store raw trajectories,
per-run files, and checkpoints outside Git.
