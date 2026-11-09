# Two flight losses: matched drop-gap restoration

This is one frozen **36-request development experiment**, not training or a
new held-out policy evaluation. The two inspected layouts are already
posthoc development data. The original 37/70 recovery-policy score and all
successful parent recipes stay unchanged.

## Hypothesis and exact intervention

The saved recovery policy increased drop gap for two formerly captured shots,
and both released without forbidden contact but failed whole-ball entry.
The previous height restoration did not recover either. Here we restore
**only action coordinate4, `drop_gap_m`, to the corresponding parent value**.
The other four action coordinates remain the **original recovery** values;
we do not combine this with the previous height correction.

| Development context | Original recovery gap (mm) | Restored parent gap (mm) | Change (mm) |
| --- | ---: | ---: | ---: |
| `recovery-test-r0-c5-n1` | 133.998498 | 103.431221 | −30.567276 |
| `recovery-test-r0-c6-n1` | 147.677536 | 109.932245 | −37.745291 |

| Context | Board X/Y/height (mm), fixed recovery values | Tilt, fixed | Bucket X/Y (mm), fixed |
| --- | --- | ---: | --- |
| `r0-c5-n1` | 648.164348 / −196.698065 / 424.899238 | 31.477652° | 1053.282197 / −619.287729 |
| `r0-c6-n1` | 652.624210 / −198.228558 / 443.245177 | 30.545644° | 1124.874942 / −593.198695 |

Rounded values are documentation only. The worker derives full precision
from authenticated original rows. Changing gap moves the requested ball-release
center by the same vertical amount; board pose and bucket commands are fixed.
A fresh strict two-arm admission may nevertheless select a different joint
branch. Actual q10/q14 changes and settled board-center/normal deltas relative
to the authenticated saved source are recorded, independently audited and
shown separately. We do not force old joints or claim physically identical
board realization when the evidence differs.

[Literal rule](regression-drop-gap-correction.json) binds the CPU diagnosis
SHA-256 `511e9f82004af0d603628e25adf8fab71284faa8f11de9307b4c8a611d049ab7`.
The complete five-loss diagnosis is preserved; only its first two released
flight losses are eligible here. No other case or coordinate is searched.

## Frozen plan

1. Nine original formal controls: three direct captures, three table-assisted
   captures and three expected misses, retaining their exact 17-source branches.
2. For each of the two contexts, three exact parent requests, three exact
   recovery requests and three corrected requests: 18 diagnostic requests.
3. Repeat the same nine formal controls after the diagnostic requests.

All36 slots are fixed before outcomes. Both original diagnostic variants were
admitted flights. Original branch/proof files remain exact; each corrected
action gets one new strict proof reused unchanged for three nominal requests.
If changed setup fails admission, it remains in the denominator with no
fabricated flight. No failed case is replaced, and there is no outcome-dependent
retry, search, winner selection, fitting or policy promotion.

The six parent-recipe requests are positive diagnostic comparisons, not a
hard gate that hides an unstable recipe. Only the formal bracketing control
blocks gate progression. All successful recipes, all model weights and all
historical reports remain intact regardless of this experiment's outcome.

## Physics, admission and independent verification

Diagnostics use the unchanged 19-source geometry bank and `native/` namespace.
Formal controls use the exact original 17-source bank and `native-controls/`.
Both share one lazily allocated native64/v3 environment. No proof can migrate
between banks. Geometry, native timing,64 substeps and strict admission guards
stay unchanged. This evaluates held endpoint poses; it does not train or
validate arm transit or real hardware.

Capture requires a board bounce, whole-ball bucket entry and one continuous
second retained. Table and rim assistance remain valid. Robot, floor or
unattributed contacts and invalid physics cannot claim capture. Setup declines,
ordinary flight misses and invalid-physics requests stay distinct.

The independent CPU auditor reconstructs the plan without calling its builder,
checks the only changed action coordinate, both admission namespaces, exact
historical branches, fresh changed-action proofs, actual initial joints and
settled poses. It rescores all released dense trajectories using the independent
v3 oracle and verifies complete request/control/manifest accounting. All
source snapshots and evidence are bound by hashes; old evidence is read-only.

Parent report SHA-256:
`6a255223820656059fe16a5de2944ba2593f7ac3603b287f4955979cd4677504`.
Parent audit SHA-256:
`e2ba653792b37d302425ba4af6cb22460c3d5b426274b3c1b3cbce270a857a9b`.
Parent fingerprint:
`f54c341eba5258b427e4e1eb05a1b2e93bf76518d9f94a2fe8cf6bad744f6082`.

## Finite runtime and scope

The worker may acquire only RTX4090 UUID
`GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4`, refuses foreign4090 work, waits at
most900 seconds, runs at most3,600 acquired seconds, and preserves8GiB free
storage. Its supervisor has one original6,300-second total deadline with a
1,800-second CPU-audit reserve. It can signal only its own child; failure never
extends the budget or restarts a partial study under the same name.

Entry point:

```bash
python -m sircus.mjlab.regression_drop_gap_pipeline \
  --parent-study /path/to/audited/coverage-recovery-v1 \
  --parent-audit /path/to/coverage-recovery-v1-audit.json \
  --diagnosis /path/to/coverage-regression-diagnosis-v1/diagnosis.json \
  --rule sircus/mjlab/regression-drop-gap-correction.json \
  --reference /path/to/native-reference.json --assets /path/to/assets \
  --output /path/to/fresh/regression-drop-gap-v1 \
  --gpu-lock /path/to/shared/4090.lock
```

Absolute locations are resolved by the operator from existing verified runtime
configuration. Cloning or syncing does not launch anything. Source and preflight
must be reviewed and frozen before the single authorized native run.

The4070 fountain, Isaac setup, physical hardware, Viam configuration and running
services are outside this task. No noise,128-substep qualification, model
retraining or policy activation is included. The UI publishes additive audited
results without restarting a service. Raw evidence, logs, generated configs,
full branch proofs and movies remain outside Git.
