# Frozen arm-clearance comparison — results

The same frozen models and candidate bank were evaluated on 50 new paired bucket positions. This is a selection-rule comparison; no model was trained.

| Metric | Current selector | Arm-clearance selector |
|---|---:|---:|
| Complete captures / requested | 20/50 | 16/50 |
| Clean captures / requested | 6/50 | 4/50 |
| Native releases | 44 | 32 |
| Native misses | 24 | 16 |
| Invalid native shots | 0 | 0 |
| Other setup rejections | 6 | 6 |
| Clearance abstentions | 0 | 12 |

Paired outcomes: **0 gains, 4 losses**, 16 shared captures and 30 shared failures. Selected action changed at 5/50 positions. 3 baseline captures were lost through clearance abstention.

| Measured contact | Current selector | Arm-clearance selector |
|---|---:|---:|
| Post-rebound robot contact / observed native shots | 12/44 | 5/32 |
| Before scoring terminal / observed native shots | 12/44 | 5/32 |
| After scoring terminal / observed native shots | 0/44 | 0/32 |
| Native contact observations missing | 0 | 0 |

Among 32 pairs with measured native contact on both sides, 0 lost robot contact and 1 gained it. A nonexecuted clearance abstention is not an observed contact-free shot.

Post-rebound contacts per requested layout were 12/50 versus 5/50. The lower total includes 12 additional nonexecuted abstentions; it is not evidence that the rule corrected seven trajectories. Among paired executed shots, it removed zero robot-contact outcomes and added one.

## Concrete paired examples

### arm-clearance-test-04 — capture lost

Bucket X/Y: 784.193 / -422.549 mm.

| Setting / observation | Current selector | Arm-clearance selector |
|---|---|---|
| Outcome | non-clean capture | clearance abstention |
| Board XYZ, mm | 624.375 / -250.353 / 387.672 | 624.375 / -250.353 / 387.672 |
| Board tilt, degrees | 36.209 | 36.209 |
| Release gap, mm | 85.581 | 85.581 |
| Predicted minimum arm clearance, mm | unknown | -10.805 |
| Observed post-rebound robot contact | no | unknown |
| Observed retention, seconds | 1.000 | unknown |

### arm-clearance-test-07 — capture lost

Bucket X/Y: 439.606 / 151.701 mm.

| Setting / observation | Current selector | Arm-clearance selector |
|---|---|---|
| Outcome | clean capture | clearance abstention |
| Board XYZ, mm | 592.385 / 101.498 / 269.285 | 592.385 / 101.498 / 269.285 |
| Board tilt, degrees | 27.267 | 27.267 |
| Release gap, mm | 99.131 | 99.131 |
| Predicted minimum arm clearance, mm | unknown | -3.632 |
| Observed post-rebound robot contact | no | unknown |
| Observed retention, seconds | 1.000 | unknown |

### arm-clearance-test-00 — clearance abstention

Bucket X/Y: 1110.204 / -11.765 mm.

| Setting / observation | Current selector | Arm-clearance selector |
|---|---|---|
| Outcome | native miss | clearance abstention |
| Board XYZ, mm | 603.039 / -124.312 / 314.601 | 603.039 / -124.312 / 314.601 |
| Board tilt, degrees | 16.423 | 16.423 |
| Release gap, mm | 271.877 | 271.877 |
| Predicted minimum arm clearance, mm | unknown | -75.186 |
| Observed post-rebound robot contact | yes | unknown |
| Observed retention, seconds | 0.000 | unknown |

### arm-clearance-test-39 — clean capture rejected

Bucket X/Y: 609.297 / -198.562 mm.

| Setting / observation | Current selector | Arm-clearance selector |
|---|---|---|
| Outcome | clean capture | clearance abstention |
| Board XYZ, mm | 632.944 / -49.450 / 274.148 | 632.944 / -49.450 / 274.148 |
| Board tilt, degrees | 27.534 | 27.534 |
| Release gap, mm | 96.365 | 96.365 |
| Predicted minimum arm clearance, mm | unknown | -7.474 |
| Observed post-rebound robot contact | no | unknown |
| Observed retention, seconds | 1.000 | unknown |

### arm-clearance-test-44 — alternative loses capture and adds robot contact

Bucket X/Y: 430.810 / -102.857 mm.

| Setting / observation | Current selector | Arm-clearance selector |
|---|---|---|
| Outcome | non-clean capture | native miss |
| Board XYZ, mm | 632.944 / -274.450 / 274.148 | 660.306 / -284.381 / 267.520 |
| Board tilt, degrees | 27.534 | 33.118 |
| Release gap, mm | 96.365 | 122.287 |
| Predicted minimum arm clearance, mm | unknown | 25.772 |
| Observed post-rebound robot contact | no | yes |
| Observed retention, seconds | 1.000 | 0.000 |

### arm-clearance-test-01 — predicted clear, observed robot contact

Bucket X/Y: 1122.558 / 128.726 mm.

| Setting / observation | Current selector | Arm-clearance selector |
|---|---|---|
| Outcome | native miss | native miss |
| Board XYZ, mm | 608.776 / -131.401 / 327.538 | 608.776 / -131.401 / 327.538 |
| Board tilt, degrees | 17.629 | 17.629 |
| Release gap, mm | 277.104 | 277.104 |
| Predicted minimum arm clearance, mm | unknown | 12.052 |
| Observed post-rebound robot contact | yes | yes |
| Observed retention, seconds | 0.000 | 0.000 |

## Verification and limits

Frozen producer source: commit `84d2ffb`. The scalar [summary.json](summary.json) records the exact rule, contract, plan, native report, manifest and auditor hashes. The fetched 79,382-byte compact report matched its remote SHA-256 (`f7422889bd84f698f2b1795d0c9f4cc845720d24021a05c037b34fe7f3e15305`).

Native proof: 1,461,935 checks. Both pre/post control gates passed. All 112 requests remain accounted for. Raw trajectories, local host paths and calibration configuration are excluded from this summary.

Fixed64 physics; one requested shot per method per layout; held endpoints and privileged simulator geometry. The prediction check is post-bounce only and does not certify incoming drop, dynamic arm transit, later bucket contact, retention or model accuracy. No noise or 128-substep qualification was added. The original +60 mm intervention and all earlier results remain separate.

Do not infer a universal winner from fewer contacts alone. Review capture rate, paired losses and abstention cost together. No active interactive policy is changed by this report.

## Decision and next step

**Do not promote this rule.** Capture fell from 40% to 32% with zero paired gains. Of the four lost captures, cases 04, 07 and 39 were rejected with conservative predicted clearances of −10.805, −3.632 and −7.474 mm even though the baseline native shots captured without post-bounce robot contact. Case 44 selected a different shot and lost capture while adding a robot contact. Case 01 passed a +12.052 mm prediction but still hit a robot. Closest predicted arm-link names are not published in the compact audit; native contact logging is aggregate.

The next targeted diagnosis is the conservative finger bounds and the time horizon used for predicted clearance. This is a hypothesis to test, not proof of an arithmetic bug. No further training was started, the frozen rule is preserved, and the previous selector remains the default.

All numbers above are synthetic MuJoCo results. A complete capture uses the unchanged board-bounce, whole-ball-entry and continuous one-second retention evaluator; a non-clean capture is not automatically a robot contact.
