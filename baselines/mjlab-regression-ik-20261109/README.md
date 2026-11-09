# Two IK declines recovered with nearby feasible recipes

Both selected recovery-imitation IK declines now have changed recipes that
captured **3/3 nominal repeats** at fixed **64 substeps**. Restoring only each
successful parent's board X/Y coordinates worked for both cases. The original
successful parent recipes still captured **6/6**. No policy was retrained or
activated, and the IK solver was not changed.

| Frozen variant | Case `r1-c1-n1` | Case `r2-c4-n0` | Total captures / requests |
| --- | --- | --- | ---: |
| Exact parent | 3/3 captures | 3/3 captures | 6/6 |
| Exact recovery | Cached IK decline | Cached IK decline | 0/6 |
| Restore parent X/Y | 3/3 captures | 3/3 captures | 6/6 |
| Restore parent height + gap | 3/3 captures | IK decline | 3/6 |
| Restore parent tilt | IK decline | IK decline | 0/6 |
| Blend 25% toward parent | 3/3 captures | 3/3 captures | 6/6 |
| Blend 50% toward parent | 3/3 captures | 3/3 captures | 6/6 |
| Blend 75% toward parent | 3/3 captures | 3/3 captures | 6/6 |

Every row received three requests per case, fixed before outcomes. A refusal
has no released ball or invented trajectory. Original refusals reuse their
historical proofs; each novel action has one strict admission proof shared by
its three requests. These are nominal flight repeats, not repeated independent
IK searches or noise robustness tests.

## Two concrete successful corrections

| Context | Bucket X/Y mm | Corrected board X/Y mm | Board height mm | Tilt | Drop gap mm | X/Y change from rejected recovery mm | Valid branches: dropper / board | Captures |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| `r1-c1-n1` | 378.083 / −438.148 | 635.421 / −392.848 | 257.834 | 32.196° | 236.932 | −5.892 / +23.525 | 0/9 → 10/8 | 3/3 |
| `r2-c4-n0` | 898.306 / −244.167 | 579.343 / −59.661 | 478.303 | 25.631° | 182.232 | +13.693 / +2.605 | 8/0 → 7/7 | 3/3 |

The first original failure lacked a dropper solution; the second lacked a
board-arm solution in the finite search. Zero solutions found does not prove
that a target is unreachable. These small X/Y corrections retain the rejected
recovery's height, tilt and gap parameters; bucket X/Y stays fixed. Target-facing
yaw and both arms' solved joint configurations can change with board X/Y.
Coordinates use the existing simulation frame. Rounded settings above are for
reading; [summary.json](summary.json) contains the exact parent, recovery and
all changed recipes plus first-repeat admission and realization diagnostics.

Across the **48 diagnostic requests**, **33 released and all 33 captured**;
15 declined setup. Nine of twelve novel configurations captured all three
repeats (**27/36 novel requests**); the other three configurations declined.
All diagnostic captures were direct board → bucket routes, with no robot
contact; direct does not mean rim-free. Successful parent recipes and all
historical outcomes remain intact.

The complete **66-slot** study finished in **203.8 worker seconds**. Both formal
control blocks passed: each had three direct captures, three table-assisted
captures and three expected misses. The independent audit passed **25,329
checks**. The 15 declines include six cached original refusals and nine new
action refusal slots; all 51 released shots include the 18 formal controls.
The six robot-contact misses belong only to intentional negative controls.

[Protocol](../../sircus/mjlab/REGRESSION_IK.md) ·
[Handoff and evidence identities](../../docs/MJLAB_REGRESSION_IK_20261109.md) ·
[Draft PR #111](https://github.com/kadiru/sircus/pull/111).
The existing `#regressionIK` interface shows all 16 configurations, original
failure reasons, controls and actual first-repeat ball/posture samples. Browser
checks confirmed admitted replay and disabled flight replay for declined poses.

This is local development evidence, not a new policy score: the original
recovery-imitation **37/70** remains unchanged. Together with the earlier height
and drop-gap corrections, all five inspected lost captures now have nominally
successful local recipes. Generalization still requires a separate frozen
policy/selector evaluation on new bucket positions. No new PPO, 128-substep or
noise qualification, learned grasp/transit, physical validation or automatic
promotion. The 4090 worker exited; the 4070 workload and Isaac setup were untouched,
and the existing viewer process was not restarted. Raw traces, proofs and logs stay out of
Git.
