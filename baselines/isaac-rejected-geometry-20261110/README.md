# Rejected TRAIN proposals: frozen geometry repair results

All **6,344 original seed-22 TRAIN rejections** were checked offline with the
same frozen [continuous geometry layer](../../sircus/isaac/REJECTED_TRAIN_GEOMETRY.md).
The full independent CPU geometry audit passed.

| Original population | Passed unchanged | Adjusted and passed geometry | No verified repair |
|---|---:|---:|---:|
| All 6,344 rejected requests | 348 | 4,175 | 1,821 |
| Included 5,996 geometry/path rejections | 0 | 4,175 | 1,821 |
| Included 348 movement/settling rejections | 348 | 0 | 0 |

The layer geometrically repaired **69.6% of the original geometry/path
rejections**. The 348 unchanged cases are exactly the original 188 incomplete
transits, 147 non-quiet setups and 13 unsettled endpoints. They already passed
geometry, path and home eligibility in the original evidence. Their old native
failures remain untested dynamically; unchanged geometry does not fix or clear
those failures.

## Why 1,821 remained unresolved

The following are the final reached gate causes after the bounded attempt, not
the original reasons or a proof that no solution exists anywhere:

| Final cause | Requests |
|---|---:|
| Dropper arm IK unresolved | 445 |
| Bouncer arm IK unresolved | 1,077 |
| Board/table sampled-path proxy overlap | 181 |
| Board/bucket clearance/path proxy overlap | 105 |
| Bouncer arm/table endpoint proxy overlap | 5 |
| Proposed adjustment outside necessary outer reach bound | 8 |
| **Total** | **1,821** |

Across these same unresolved cases, 1,738 reached the local optimizer's fixed
function-evaluation limit; 83 had another optimizer return but did not pass all
final gates. This is a separate numerical-termination description, not an
additional population. A larger budget or different method might produce a
different answer; this check did not tune the layer or establish impossibility.

For comparison by the original rejection reason, repairs were 934/1,259 outer
reach declines, 1,727/2,186 dropper IK declines, 1,507/2,479 bouncer IK declines
and 7/27 board/bucket declines. The four arm/table and 41 board/table declines
remained rejected. [The compact evidence](summary.json) retains the complete
original-to-current classification matrix and final cause counts.

## Fixed method, cost and verification

Fresh IK used each original request's recorded measured starts for both arms.
No saved successful endpoint answer, candidate bank, model inference, ball-flight
score or held-out shot packet entered repair. All 289 paid TRAIN packets and
10,000 original requests, the original passed audit and final model bytes were
verified read-only. These proposals span the training history, not new outputs
of the final checkpoint.

The layer's defaults stayed at one local continuous repair, 24 optimizer function
evaluations, 512 residual evaluations and at most 10% of each original action
span. All original admission thresholds stayed fixed. The original 80 mm
ball-bottom-to-board-center gap definition is unchanged.

Median absolute changes among the 4,175 repairs were approximately 10.24 degrees
bearing, 19.69 mm radius, 17.57 mm board height, 15.19 mm gap and 1.72 degrees
tilt. The changes can affect ball flight. No capture label is attached to a
repaired setting, and no capture or generalization improvement was measured.

The nine-case timing sample and full CPU invocation together took **1,400.88 s
wall (23.35 minutes)** with four workers, **5,508.72 aggregate completed-worker
CPU-seconds** and 27.82 coordinator CPU-seconds. The nine saved sample records
were independently checked and reused, so all 6,344 requests were computed once.
These receipts exclude input loading, resumed-row validation, final report
assembly and the independent audit; they are not a total application or native
physics timing. Python 3.12.14 / NumPy 2.5.4 / SciPy 1.18.1 match the previous
successful-action geometry check.

All **38 focused cases passed without skips**, including full synthetic
profile/resume/audit integration, immutable source/data joins, native-stage
classification, bounded repair, atomic output, corrupt-cache and symlink cases.
Full `scripts/test.sh` passed: 1,400 Isaac cases (121 optional dependency skips),
34 physical mocks, five simulation checks, 59 Node cases and all nine numerical
scripts. Independent code and evidence reviews are recorded with the task.

The 445-file before/after inventory matched exactly, including frozen geometry
code, models, original benchmark artifacts and existing report. No training,
simulator, new native shot, reader/container restart, model promotion or original
benchmark change occurred. Raw coordinates and per-request evidence remain
private. Summary hashes bind the full analysis, independent audit, manifest,
durable row inventory, invocation receipts and test log.

Admission retains the original capsule/enclosing-volume and 16-sample path
proxies; it is not a full mesh/self-collision or driven-motion certificate.
Native execution and capture remain unverified for every repaired setting.
