# Guarded board-position fallback

## Problem and correction

The preceding [50-position comparison](BOARD_FALLBACK_STUDY.md) captured 41/50
with the current planner and 40/50 with an experimental board-pose fallback.
At the lost target, the fallback interrupted the normal search after 12.47 s,
before it could find the clear shot found by the current planner at 16.64 s.
The fallback selected an alternative with predicted negative arm clearance;
the native ball then hit the robot. Those original results remain unchanged.

This version makes two specified changes:

1. Let the original hybrid search finish naturally. Its existing reference-path
   stall handling remains unchanged; there is no additional 12-second stop.
   A board-pose fallback can use only time left within the original absolute
   30-second decision budget, after normal planning returns a risky checked shot.
2. Replace that shot only when the new shot has a finite predicted ball-to-arm
   clearance of at least 5 mm for its predicted route and a strictly better
   finite flight score. Full endpoint IK and existing path checks still apply.

If no qualifying replacement finishes before the deadline, retain the last
checked shot from the original search. This guards replacements; it does not
claim every original shot or predicted clearance will succeed in simulation.
The flight predictor, candidate geometry menu, models, contact settings,
scorer and original interactive planner are unchanged.

## Frozen test

- Two methods: current `StalledReferencePlanner` and the new
  `GuardedBoardFallbackPlanner`, using separate reusable CPU workers and the
  same frozen learned ordering and 30-second decision allowance.
- Twenty fresh seeded target positions: eight interior, eight edge and four
  near fixed robot bases. Exclude the previous 50 targets, all recorded prior
  training/development/evaluation positions and current interactive history.
  This is a stratified comparison, not a uniform table-wide estimate.
- Each method gets one choice and at most one drop per fresh target. A setup
  decline remains unsuccessful in its /20 denominator.
- Separately, independently plan the known regression target
  **(948.252, −97.816) mm** with each method, then repeat each chosen shot three
  times. These diagnostic repetitions do not enter the fresh-position score.
- Freeze all 42 planning decisions before any new native drop. Use no observed
  outcome to rerank, retry, repair or choose a shot during this comparison.
- Execute at 64 substeps on the 4090, under the existing shared GPU lease,
  with a 30-minute native lease cap and at most 64 releases: 40 fresh, six
  diagnostic, and 18 controls. Controls bracket native execution.

Success requires a board bounce, whole-ball entry and one continuous second
inside the bucket. Direct and table-assisted captures count. Robot-assisted
retentions remain separately reported. Historical scoring is not rewritten.

## Verification and interpretation

An independent CPU audit reconstructs target separation, frozen choices and
deadlines, natural completion before board search, replacement conditions,
full pose/path evidence, native contacts and retention scores. Report setup
availability, captures out of 20, paired gains/losses, route counts, and mean,
median and 90th-percentile planning request times. Show diagnostic repeats
separately. Full request time includes worker startup and cleanup; it can
exceed the decision limit and excludes GPU waiting and drop simulation.

This is a planner correction test, not new imitation or PPO training. It uses
simulator geometry, not camera or point-cloud observations. The native ball
flies with arms held at checked endpoints; sampled geometric paths are not
driven-arm dynamics or physical validation. The live planner and seven known
successful settings are preserved. No service restart, 4070 workload or
Isaac setup change is part of this task.

## Execution status

Implementation and mock checks are in progress. No new native capture result
is claimed until the frozen test and independent audit complete. Results will
appear in the existing UI at `#guardedBoardFallback`, with the previous
`#boardFallbackStudy` comparison preserved.
