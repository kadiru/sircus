# Completed privileged-state study

Both runs completed 4,096 exploratory training throws, followed by frozen evaluation
on the same 256 previously unseen test layouts. Selected checkpoints were chosen
from validation, not from test performance. Success means board rebound, whole-ball
entry and one continuous second retained in the bucket.

| Training layouts / GPU | Selected at throw | Test captures | 95% Wilson interval | Board IK failures | Wall time |
|---|---:|---:|---:|---:|---:|
| 100 / RTX4070 | 3072 | 142/256 (55.5%) | 49.3–61.4% | 26 | 2.44h |
| 1000 / RTX4090 | 4096 | 119/256 (46.5%) | 40.5–52.6% | 36 | 2.09h |

Fixed warm action:126/256 on4070,128/256 on4090. Fixed midpoint:123/256 on both.
The small difference in repeated fixed-action outcomes illustrates contact/device
sensitivity. One seed and different GPUs prevent causal claims that fewer training
layouts are better. At the equal budget,100 layouts received about41 visits each;
1000 received about4. There is no demonstrated benefit from adding layouts here.

| Learned policy | Near200–267mm | Medium267–333mm | Far333–400mm |
|---|---:|---:|---:|
| 100 layouts | 19/90 (21.1%) | 65/79 (82.3%) | 58/87 (66.7%) |
| 1000 layouts | 50/90 (55.6%) | 43/79 (54.4%) | 26/87 (29.9%) |

The100-layout policy improves medium/far captures while sacrificing near captures.
Its tilt varies only16.00–16.72° and gap126.20–132.84mm across the test set. The1000-
layout policy varies tilt21.14–21.32° and gap157.19–159.15mm. Outputs are scene-
dependent, but the small variation is consistent with weak adaptation. All failed
endpoint solves were on Arm14; solver failure does not prove physical infeasibility.

**Decision: defer quasi-perception training.** This is post-study engineering judgment,
not a preregistered statistical threshold. The privileged policy is not reliable
across distance groups; adding noisy clouds now would confound the diagnosis.

Recommended next experiment:

1. Audit board IK on development layouts using deterministic alternative seeds;
   distinguish solver failures from geometric reachability without removing failures
   from reported denominators.
2. Strengthen scene conditioning and test a near/medium/far curriculum. Select using
   all validation layouts plus group coverage, rather than a small selection subset
   with50% far layouts. Freeze the selection rule before running.
3. Repeat seeds and compare selected policies on one evaluation GPU. Use fresh
   held-out layouts for the next final test: this test has now informed diagnosis.
4. Once privileged performance covers all groups, introduce padded crops of actual
   rendered depth, preserving occlusion/missing points and labeling oracle cropping.

`comparison.json` contains aggregate curves, paired counts, action/failure diagnostics,
contract fingerprint and SHA256 of both selected checkpoints. Raw rows/checkpoints
remain in runtime storage, not Git. Rebuild using `sircus/isaac/learning_report.py`
against both original run directories (see `sircus/isaac/LEARNING_STUDY.md`).
No additional training, quasi-perception run, hardware connection or cloud deployment
was started during this comparison.
