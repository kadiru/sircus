# Bounded board-pose fallback

The current interactive planner captured at 17 of 20 previously unseen targets.
All three misses struck the board arm after rebound. A separate search that
moved the board found seven settings at two of those targets; each captured
three fresh repeats. Those discoveries are development evidence, not a new
policy evaluation. See [the completed search](BOARD_POSE_RECOVERY.md).

This task tests whether a bounded version of that geometry search improves
first-drop capture on new targets without removing arm checks or increasing
the 30-second decision budget.

## Known-target aiming experiment

At bucket (1226.0, 183.4) mm, the closest prior arm-free miss passed about
83 mm to the side of the bucket. The new experiment fixes its board center,
height, tilt, drop gap and twist while testing headings from −4° to +8°.
Only headings that fail setup admission receive small ±20/40 mm X/Y changes;
the finite preparation budget is 160 checks, at most 40 native settings, and
ten minutes of CPU preparation. The original setting receives three matched
repeats. Every qualifying new capture receives three fresh confirmations.
Two nine-shot control blocks bracket the native experiment.

### Completed diagnostic

Changing heading alone did not recover this target. The original arm-free
setting missed in all three matched repeats. Of the 13 new heading settings,
11 produced robot contact and two cleared the arms but still missed. No new
capture qualified for confirmation repeats.

| Test | Releases | Captures | Releases with robot contact |
|---|---:|---:|---:|
| Original setting, matched repeats | 3 | 0 | 0 |
| New heading settings | 13 | 0 | 11 |

All 18 control shots passed their expected outcomes: 12 captures and six
negative controls. The experiment therefore completed **34 native releases**,
including controls, in **129.46 seconds of leased GPU time**. The independent
audit passed **7,364 checks**. This establishes that this heading sweep did not
solve the remaining target; it does not establish that the target is impossible.

The producer was commit `316d6682d825e7995104936898427dd2827d025e`;
the frozen runtime and output are named `remaining-heading-v1`. Audited bindings:

- Report SHA-256: `f9b0af0ad7743acb1595e7755eafd6b2566c1b0c7117c37bef7fb80d825fc105`
- Preparation SHA-256: `dd4746f8e3bcc4f5fc42110869df0e189774b62c3d471cfb9bd44258bb260cbe`
- Evidence-manifest SHA-256: `26a5f190fb254efeff688a80d24d05dae168908b8b0113ba88accb8fb377f851`

This known target is excluded from the new 50-position test. Its outcomes do
not select among candidates at those new positions.

## Frozen comparison

- **Baseline:** the current stalled-reference fallback planner and frozen
  learned shot ordering, with its usual 30-second decision budget.
- **Candidate:** the same initial planner. Once it has a checked shot with
  predicted arm clearance below 5 mm or no finite arm-clearance prediction,
  it may end the initial search after 12 seconds and use the remaining time
  for alternate board positions/headings. If the initial search finishes
  earlier with such a shot, the fallback may begin then.
- The menu translates the board by up to 150 mm radially and laterally,
  changes heading/twist, and includes bounded height/tilt/gap proposals. It
  screens necessary reach before at most 16 full checks and two multistart
  retries. Full-pose IK and the existing geometric path checks remain required.
- Only a strictly better finite score from the unchanged approximate flight
  model replaces the checked initial choice. A timeout or failed fallback
  retains the last fully checked, published choice. Predicted quality is not
  evidence of a capture.
- Both methods use one absolute 30-second clock. The new method may explore
  fewer original candidates to reserve time for board changes; it does not
  receive a second planning budget or the independent baseline's final answer.

Fifty new seeded bucket positions use a **stratified sample: 20 interior,
20 near grid edges, and 10 near fixed robot bases**. This deliberately tests
different parts of the admitted table region; it is not a uniform table sample.
Prior training, development, evaluation and recorded interactive positions are
excluded. Each method prepares one choice per position using separate reusable
CPU workers. All 100 decisions are frozen before any test drop. A missing setup
remains an unsuccessful position in the denominator.

Native execution uses the 4090, 64 substeps, and the existing calibrated
simulator/contact/scoring contract. Each method gets at most one release at
each position. Direct and board→table→bucket routes qualify after whole-ball
entry and one continuous second inside. Robot-assisted retained outcomes
remain separately visible. Controls run before and after the experiment.

Report paired gains/losses, first-drop captures out of 50, setup availability,
route counts, fallback activations and changed choices, plus mean, median and
90th-percentile planning time. An independent audit recomputes evidence
bindings, selection rules, setup proofs and saved native outcome scores.

## Scope

This is planner engineering and evaluation, not new PPO or imitation training.
Inputs are privileged simulator geometry. Native ball flights hold the arms
at checked endpoints; sampled geometric paths do not establish driven-arm
simulation or physical safety. The seven existing successful settings and all
historical results are preserved. The live planner default remains unchanged
while comparison results are reviewed. The 4070 and Isaac are untouched.

The 50-position planner comparison is pending; the completed heading diagnostic
above is separate development evidence. Comparison results will be added after
independent validation. Raw traces and runtime configuration remain outside Git.
