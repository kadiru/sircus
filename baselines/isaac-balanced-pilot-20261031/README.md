# Short balanced pilot: solver repaired; policy improvement not established

Completed on RTX 4090, October 31, 2026 (UTC run `balanced-pilot-20261101T012440Z`).
384 additional PPO throws took about 18 minutes including development probes,
validation and four frozen comparisons. The 4070 was untouched. The finite job
exited; no longer run was started.

[Method and reproduction](../../sircus/isaac/BALANCED_PILOT.md) ·
[Verified aggregates and artifact hashes](results.json) · [IK regression](ik-regression.json)

## Engineering results

- All **62 historical board IK failures recovered**, with 16 valid controls still
  passing: **78/78**. Original acceptance norm <1e-4 and joint limits preserved.
  Strict convergence, analytic derivatives and removal of a redundant wrist-spin
  variable fixed the numerical issue. This is historical regression, not new
  generalization evidence or collision-path validation.
- Development probes: 11 measured stable poses, one rejected board endpoint.
  Rejection was retained and no ball released; it does not prove that target is
  globally unreachable. Earlier failed settling probes remain separate locally.
- **144/144 paired test shots passed IK and settling gates**. Worst measured board
  center error was 0.0862 mm; normal error 0.0128 degrees. Maximum measured joint
  speed was below 1.8e-8 rad/s and center drift rounded to zero over 250 ms.
  These ideal simulation readings use model-based gravity compensation; they are
  not physical calibration accuracy. No feedback corrections were required.
- Training had two endpoint rejections out of 384; these remained failures in the
  denominator. All other training shots passed the release gate.

## Learning result

96 distinct training layouts, 128 throws per distance band; 18 disjoint validation
layouts; 36 fresh test layouts (12 per band). Fine-tuned the selected 100-layout
actor. Validation captures progressed **9/18 → 10/18 → 13/18** at 0/192/384 throws,
selecting episode 384. Actor weights changed (L1 change 9.7518). Evaluation used
frozen direct inference with no scene-specific optimization.

| Policy | Total | Near | Medium | Far | Invalid |
|---|---:|---:|---:|---:|---:|
| Short balanced pilot | 21/36 (58.3%) | 7/12 | 11/12 | 3/12 | 0 |
| Existing 100-layout | 22/36 (61.1%) | 3/12 | 10/12 | 9/12 | 0 |
| Existing 1,000-layout | 22/36 (61.1%) | 11/12 | 10/12 | 1/12 | 0 |
| Fixed warm action | 28/36 (77.8%) | 12/12 | 12/12 | 4/12 | 0 |

All four use the new controller/settling contract on identical test layouts.
Historical 256-layout scores remain unchanged and must not be compared directly
against this table. The fixed action was inherited from the legacy contract,
not selected using these test outcomes.

The pilot won 7 layouts and lost 8 relative to its starting policy; it won 2 and
lost 9 relative to the fixed action. Confidence intervals are wide (pilot 95%
Wilson interval 42.2–72.9%). One seed and 12 tests per distance do not establish
statistical superiority. The update improved the small validation set but did
not improve overall test capture count and lost far-distance competence.

The pilot barely adapts its actions: tilt 15.16–15.89 degrees, heading offset
1.19–1.58 degrees, release gap 114.4–122.3 mm across test scenes. Together with
zero invalid endpoints, this points to weak scene conditioning/action selection
as the next issue to investigate; it does not establish a single causal bug.

## Decision and next bounded work

Keep the historical policies and the new pilot separately; **do not promote the
pilot or start another long run**. Audit normalized scene-feature sensitivity and
policy output changes on development layouts. Compare the reward landscape and
feasible tilt/height actions by distance under the repaired controller. Then test
one targeted conditioning correction with a larger, fresh validation pool and
fresh final test layouts. Treat these 36 test layouts as now exposed; do not tune
on them and claim they are untouched. Quasi-perception remains deferred.

## Evidence and scope

Raw trials, joints, checkpoints, splits and immutable source remain outside Git:
`~/sircus-isaac/output/balanced-pilot-20261101T012440Z/` and
`~/sircus-isaac/studies/balanced-pilot-v1/` on Ubuntu. The source hashes, reference
checkpoint hashes, selected checkpoint hash and evidence hashes are in results.json.
A local ignored evidence copy is retained by the Mac integration checkout.

This is privileged-state articulated simulation, with endpoint initialization,
not learned grasping/travel, camera-driven behavior, quasi-perception or physical
validation. Existing geometry uncertainties remain unchanged. No hardware,
Viam cloud configuration or audio behavior was changed.

Validation: 52 Isaac unit tests, 3 diagnostic/pilot DOM tests, the full project
mock/numerical suite, Ruff and repository checks passed. The completed interface
was checked in Chrome, including comparison tables and distance bars.
