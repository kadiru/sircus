# Compare shot planners under the same 30-second limit

The prior learned selector captured 173/200 versus the current planner's
169/200, but its 90th-percentile planning time increased from 67 to 168 seconds.
This experiment asks whether it retains a capture advantage when both planners
have **30 seconds to choose a shot**. It does not retrain either model.

## Frozen comparison

- **100 new bucket positions**, covering the usable table interior, edges and
  regions near fixed arm bases. Each is at least 5 mm from inventoried previous
  positions, all 400 positions of the preceding neural study, and the other new
  positions. This does not establish all-table coverage.
- Current physics planner versus the frozen learned selector from
  [the completed study](NEURAL_SHOT_RESULTS.md).
- **One drop per planner per position**, at 64 substeps; at most 200 test drops
  plus 18 matched controls. No simulation outcome influences shot selection.
- Count captures over all 100 positions for each planner, including positions
  where no checked shot is ready before the deadline.

The candidate libraries, trained weights, exact arm checks, physics ranking and
capture definition stay fixed. Both planners check candidates incrementally so
they can keep a completed choice while working on another. The neural selector
orders candidates before arm checks. Identical physics predictions are cached
within each job; caching changes no predicted values or ranking rules.

## What the deadline covers

Each planning request owns an isolated CPU process. Common engine initialization
finishes before a parent/child start handshake and is measured separately.
The 30-second clock covers candidate generation, network inference, IK and path
checks, physics ranking, and publication of the complete choice.

After each candidate, the child writes an immutable complete evidence snapshot.
Only snapshots received by the parent before the deadline are eligible. At the
deadline, the parent stops only its own CPU child and keeps its last fully
checked choice. It never uses an unfinished or late candidate. If no complete
choice exists, that position has **no release** and counts as unsuccessful.
Unexpected worker errors fail the preparation stage rather than being hidden as
ordinary timeouts.

Reported decision time is bounded by 30 seconds. Startup, actual elapsed
planning/worker shutdown, and their combined worker lifecycle time are also
reported. Worker lifecycle excludes parent snapshot verification, hash/readback
checks and archive writes; it is not total UI request latency. A cold request
is therefore not claimed to finish within exactly 30 seconds. Four
low-priority CPU jobs run concurrently, with CUDA hidden. Method order is
deterministically shuffled. Separate proof directories prevent one planner from
reusing another planner's already solved arm configuration.

## Execution and evidence

`python -m sircus.mjlab.planning_budget_study --help` lists required archived
inputs. Use a fresh output directory and these stages in order:

1. `--stage initialize`: verify the preceding audit and dependencies; freeze
   positions, libraries, checkpoint, protocol and computational source hashes.
2. `--stage prepare`: run both deadline-limited planners on CPU, preserving all
   accepted snapshots and selected commands before any native test drop.
3. `--stage run`: cooperatively acquire the 4090, run controls, alternate the
   two planners' frozen drop batches, then run controls again.
4. `--stage finalize`: summarize and freeze the full evidence inventory.
5. `--stage audit --audit-output <fresh path outside study>`: independently
   verify deadlines, complete arm-check evidence, frozen commands, native ball
   traces, retention scoring and accounting before publishing results.

Capture requires a board rebound, whole-ball entry and one continuous second
inside the bucket. Rim and table assistance count. Robot-assisted captures are
recorded separately. Invalid physics remains in the denominator and is not
treated as evidence of a valid physical miss.

Compare first-drop captures, setup-ready counts, deadline expirations with and
without a checked shot, robot contacts, and median/p90/max decision time. Report
startup and worker lifecycle time separately, plus paired gains/losses.

Native simulation is capped at one leased GPU-hour and waits up to 12 hours for
the shared 4090 lease. It does not interrupt other work. The 4070 and Isaac setup
remain untouched. Raw observations, checkpoints, logs and generated settings
stay outside Git. A partial native stage cannot silently rerun as new attempts.

Results/progress appear in `#planningBudgetStudy` on the existing mjlab UI.
No automatic planner promotion or server restart occurs. Inputs are privileged
simulator geometry, and native drops hold settled endpoints; this is not a test
of perception, driven arm transit or physical deployment.
