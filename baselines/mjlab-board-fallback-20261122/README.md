# Bounded board-position fallback: audited results

Both planners used frozen models and a 30-second decision budget on the same
50 new bucket positions, one drop per method/position at 64 substeps.

| Measurement | Current planner | Board fallback |
|---|---:|---:|
| Captures | 41/50 | 40/50 |
| Direct / table-assisted | 30 / 11 | 29 / 11 |
| Checked shots released | 49 | 49 |
| Mean full planning request | 12.87 s | 14.39 s |

Forty shared captures, no gains, one loss, nine shared failures. The missing
setup in each method remains in its denominator. The fallback activated at 11
positions and changed six choices. Keep the current interactive default.

Sample: 20 interior, 20 edge, 10 near fixed bases; not a uniform table estimate.
98 test releases plus 18 controls used 973.419 leased GPU seconds. Both control
blocks and 152,638 independent audit checks passed. No robot-assisted retention
or invalid-physics outcome occurred in this comparison.

The separately reported known-target heading diagnostic captured 0/13 new
settings and 0/3 unchanged repeats. It used 34 native releases including
18 controls, 129.458 GPU seconds, and passed 7,364 audit checks. These are
development results and are excluded from the 50-position comparison.

[summary.json](summary.json) contains sanitized counts, timing, paired targets,
frozen producer/source hashes and independent audit bindings. No raw traces,
host calibration, model weights, credentials or runtime logs are included.

[Method, diagnosis and reproduction](../../sircus/mjlab/BOARD_FALLBACK_STUDY.md).
The live planner and seven previously confirmed settings remain unchanged.
This study uses simulator geometry and held-endpoint ball dynamics; it does
not train a new model or validate perception or driven-arm motion. The 4070
and Isaac setup were left alone.
