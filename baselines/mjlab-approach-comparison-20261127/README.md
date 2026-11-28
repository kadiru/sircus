# Fresh-position bucket-approach comparison — November 27, 2026

**Keep the current planner.** On 50 fresh bucket positions, current captured
**43/50 (86%)**; expanded search plus full-approach ranking captured **32/50
(64%)**. The proposed method gained one position and lost twelve: 31 shared
captures and six shared failures. No additional experiment started.

| Measurement | Current | Proposed approach |
| --- | ---: | ---: |
| Captures / all positions | 43/50 | 32/50 |
| Setup declines | 0 | 9 |
| Released flight misses | 7 | 8 |
| Invalid-physics releases | 0 | 1 |
| Decision mean / median / p90, seconds | 11.845 / 5.723 / 30.000 | 13.686 / 10.353 / 30.000 |
| Full request mean / median / p90, seconds | 12.121 / 5.827 / 30.042 | 14.003 / 10.911 / 30.055 |

Declines and invalid physics remain in the denominator. Full-request time
includes worker startup/cleanup; GPU wait and execution are separate.

Both trained checkpoints stayed frozen, with a 30-second decision limit,
one attempted shot per target and 64 physics substeps. The proposed method
combines its expanded library/selector, local tilt/drop-gap candidates and
full-approach ranking. This compares complete methods, not the bucket check
alone. Both lanes received the same nested-IK adapter correction; the live
service remains unchanged.

Success requires the first confirmed rebound to be on the board, whole-ball
entry and one continuous second inside the bucket. Later table, robot and bucket contacts are permitted; a
partial exit resets retention. Arms remain held at checked endpoints during
ball simulation.

The 50 targets were separated by at least 5 mm from 5,166 inventoried historical
coordinates, including both earlier attempts. Commands froze before execution
in batches of up to 32 independent worlds. **109 native releases** completed:
91 evaluation drops and 18 matched controls. All controls passed. The final
independent audit passed **196,747 checks**; acquired 4090 time was
**366.846 seconds**.

Earlier evidence remains intact. V1's CPU preparation failed after 78 requests,
with zero drops. V2's singleton packets projected beyond the execution budget:
it saved 16 evaluation rows and nine controls; one request lacked a completed
row. Neither attempt contributes to these rates; target outcomes did not inform planner changes.
The final auditor correction recognizes the adapter's existing invalid-reason
label; native evidence, detailed physics reasons and capture definitions remain
unchanged.

[Compact results and hashes](summary.json) · [Reproduction protocol](../../docs/MJLAB_APPROACH_COMPARISON.md) · [Draft PR #169](https://github.com/kadiru/sircus/pull/169).
Producer: `003b84f587a3027d5d2b54526d864b3e66ca7205`.
Auditor: `8859f60e538ecd79dba29a14b7110eff5e49df3b`.

Results appear at `#approachComparison`; all 17 existing videos remain.
No retraining, live promotion, service restart, 4070 workload or Isaac setup
change occurred. This validates simulated endpoint-held shots, not perception,
driven arm motion, hardware, or guaranteed capture anywhere on the table.
