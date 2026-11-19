# Two flight losses recovered by restoring drop gap

Both selected recovery-imitation flight misses captured **3/3 nominal repeats**
after restoring their matched parent drop gaps. Exact parent recipes retained
**6/6 captures**; unchanged recovery recipes remained **0/6**. All successful
diagnostic shots used a table rebound. No model was retrained or activated.

| Variant | Captures / requests | Releases | Setup declines | Invalid physics |
| --- | ---: | ---: | ---: | ---: |
| Exact parent recipes | 6/6 | 6 | 0 | 0 |
| Exact recovery recipes | 0/6 | 6 | 0 | 0 |
| Parent drop gap restored | 6/6 | 6 | 0 | 0 |

Only drop gap changed on the **original recovery action**; this did not combine
the earlier board-height correction. Board X/Y, height, tilt and bucket X/Y
remained fixed. The actual board-arm joint vectors, settled board centers and
normals also matched their saved recovery sources exactly. Dropper joints
changed to achieve the lower release heights.

| Context | Bucket X/Y mm | Board X/Y mm | Board height mm | Tilt | Original → corrected gap mm | Actual release-center Z mm | Captures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r0-c5-n1` | 1053.282, -619.288 | 648.164, -196.698 | 424.899 | 31.478° | 133.998 → 103.431 | 565.830 | 3/3 |
| `r0-c6-n1` | 1124.875, -593.199 | 652.624, -198.229 | 443.245 | 30.546° | 147.678 → 109.932 | 590.677 | 3/3 |

Coordinates are in the existing simulation frame; board height is the command
parameter and release-center Z is the first recorded native ball center. Values
are rounded for display; [summary.json](summary.json) retains full-precision
action parameters. These are two successful recipes, not an established gap
interval or all-table policy.

All **36 requested shots** ran at 64 substeps in **170.2 worker seconds**:
18 diagnostic releases and 18 formal controls. Both control blocks passed,
and the independent audit passed **16,072 checks**. The six robot-contact
misses were intentional negative controls; diagnostic shots had none.

[Protocol](../../sircus/mjlab/REGRESSION_DROP_GAP.md) ·
[Reproduction and handoff](../../docs/MJLAB_REGRESSION_DROP_GAP_20261109.md) ·
[Draft PR #110](https://github.com/kadiru/sircus/pull/110).
The existing UI shows these results and saved native samples at
`#regressionDropGap`; historical panels and their movie remain intact.

The original recovery policy score stays **37/70**. These formerly held-out
positions are now development data, so the correction is not a new policy
benchmark. No new PPO, noise trial, 128-substep qualification, learned arm
transit or physical validation is claimed. Parent recipes, historical results,
the 4070 workload and Isaac setup remain unchanged. Raw traces, proofs, logs
and runtime configuration stay outside Git.
