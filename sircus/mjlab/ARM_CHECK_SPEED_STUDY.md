# Faster arm checks: diagnosis and completed comparison

On **November 21, 2026**, faster arm checks reduced mean planning time by
**36%** on 20 fresh bucket positions. Both versions captured at the same
**17/20 positions**. The original difficult position still times out.
The live planner has not switched to these opt-in changes.

## Results on new positions

| Measurement | Current checks | Faster checks |
|---|---:|---:|
| Captures / fresh positions | 17/20 | 17/20 |
| Direct / table-assisted captures | 10 / 7 | 10 / 7 |
| Checked shots released | 18 | 19 |
| Released shots that missed | 1 | 2 |
| No checked shot within the limit | 2 | 1 |
| Mean full planning request | 21.96 s | 14.05 s |
| Median full planning request | 23.32 s | 8.85 s |
| 90th-percentile request | 31.37 s | 31.31 s |

Seventeen positions captured with both methods, three captured with neither,
and there were **zero gains or losses**. Finding one additional checked
setup did not add a capture. This batch supports faster typical planning;
it does not prove equal performance everywhere on the table.

Both methods use the same frozen model, shot menu, 30-second decision limit,
arm-clearance requirements and native physics at **64 substeps**. All 40
choices froze before any test drop. Each method had at most one release per
position; missing setups remain in the capture denominator. Full request
time includes worker initialization and result handling, excluding browser,
network, GPU waiting and drop simulation. Initialization explains why some
requests exceed the decision limit.

The comparison completed **37 test drops and 18 matched controls**. Both
control blocks and the independent **45,116-check audit passed**. Native
work used **186.10 leased seconds on the 4090**. The 4070 workload and Isaac
setup were untouched. No model was retrained.

Capture requires a board rebound and one continuous second fully inside
the bucket; direct and table-assisted routes count. The experiment uses
simulator geometry, sampled collision checks along planned paths, and ball
dynamics with arms held at the selected endpoints. It does not validate
moving-arm trajectory execution, cameras or real hardware.

The UI section is `#armCheckSpeed`. A compact committed result is
[summary.json](../../baselines/mjlab-arm-check-speed-20261121/summary.json).

## Observed timeout

Case02 places the bucket at **(971.201, -285.388) mm**. Fast planning returned
no checked shot after **30.168 s** HTTP wait; the original planner captured
after **172.395 s** planning. The fast worker was reused with **0 s startup**.
Its recorded CPU stages were:

| Stage | Seconds |
|---|---:|
| Generate shot menu | 0.058255 |
| Neural ordering | 0.013297 |
| First arm setup check, interrupted at deadline | 29.927377 |
| Physics ranking | 0.000237 |

No candidate or request proof completed before the deadline. Arm checks used
99.76% of the 30-second decision budget; the profile does not establish which
internal instruction was executing at interruption.

The original planner's completed proof isolates the expensive common setup:
**0.384298 s endpoint IK**, **85.733423 s path search**, **86.136828 s total**.
The endpoint is reachable. Both direct serialized movement orders violate
the required 5 mm clearance: bouncer link3 comes within **3.824940 mm** of
bucket wall36 on the forward segment; the reverse segment encounters link2
at **4.910049 mm** from bucket wall12. Six vertical/translate/orient
alternatives fail before bounded RRT finds a 22-waypoint path, with minimum
clearance **5.016732 mm**.

Both rejected direct orders consumed 543 gate checks each. RRT extensions
consumed 538 bouncer and 102 dropper checks; its final forward/reverse path
audit consumed 1,110 checks. The eventual selected shot instead passed a
direct home-to-shot path with 3 waypoints and 1,186 checks, after its
anchor-to-shot suffix failed. Thus this successful route did not use the
expensive reference path. The completed original run explains a plausible
shared bottleneck; the interrupted fast run has no complete internal proof.

## Historical diagnosis evidence

Runtime evidence remains outside Git on Ubuntu. Read these JSON records;
do not rerun or overwrite the historical requests:

```text
/home/kadir/sircus-mjlab/output/interactive-fast-acceptance-v1/report.json
/home/kadir/sircus-mjlab/studies/interactive-fast-v1/evidence/fast/interactive-24853e1aa5ee4a0d9b4d245359fed1e5/plan.json
/home/kadir/sircus-mjlab/studies/interactive-fast-v1/evidence/fast/cpu-lane-eddef504288d4b8e844f7d7e1ebaedcf/fast-journals/interactive-24853e1aa5ee4a0d9b4d245359fed1e5/
/home/kadir/sircus-mjlab/studies/interactive-fast-v1/evidence/interactive-dbec7cba8c084e7bb20dd3259bf555c8/plan.json
/home/kadir/sircus-mjlab/studies/interactive-fast-v1/evidence/interactive-dbec7cba8c084e7bb20dd3259bf555c8/native-heading/resolutions/8ceed83067685323435839f183ad39e3839e1fa0df756b5cad083359b02c91db.json
/home/kadir/sircus-mjlab/studies/interactive-fast-v1/evidence/interactive-dbec7cba8c084e7bb20dd3259bf555c8/native-heading/heading-connections/3298d1391c04326e458552e71d47d586be3cbdbeeb57cab99c0cba552c508936.json
```

The resolution record's `timing` provides the IK/path split. Its
`proof.attempts` records the rejected direct/Cartesian paths and RRT. Count
executed samples as `total_samples` for a passed edge or `sample + 1` for
a failed edge. Count a path once: its final proof is also repeated inside
the attempt history. The selected heading connection's `proof` is the
actual accepted route; `path_attempts` retains the rejected suffix.

## Changes and equivalence checks

Two per-instance, opt-in optimizations preserve the historical audited
implementation and its accepted-shot checks:

1. Vectorize conservative AABB distances for the same geometry pairs,
   retaining the exact distance/SAT/contact/corridor checks near obstacles.
2. Stop a path audit after its first failed segment. Audit every segment in
   both directions for an accepted path, in the original order with the
   original 0.5-degree sample spacing. Mark partial rejection evidence
   explicitly; it is never an accepted certificate.

The path installer is `arm_path_speed.install_path_short_circuit(planner)`.
It changes only the supplied `SerializedPlanner` instance. Historical
source files, thresholds, IK branches, model weights, physics and scoring
remain unchanged. CPU tests compare accepted certificates and every gate
sample against the original, and check early failures in both directions.

The geometric installer is `arm_check_speed.install_vectorized_gate(gate)`.
Real-scene CPU validation matched the original check records at **64 joint
states and 8 path segments**. Mean gate-check time was **24.105 ms versus
18.394 ms** (about 24% lower). The new test positions were kept separate from
this development check and the previously inspected target.

The diagnostic replay at **(971.2, −285.4) mm** still found no checked shot
within 30 seconds using either version (full requests: **31.34 / 31.41 s**).
The next focused change is to try the candidate's own direct home path
before obligating the expensive reference-path search, retaining the
complete accepted-path audit and fallback. The speed result above does
not establish that this unresolved case is fixed.

## Saved evidence and reproduction

Final evidence is outside Git:

```text
/home/kadir/sircus-mjlab/output/arm-check-speed-v2/
/home/kadir/sircus-mjlab/output/arm-check-speed-v2-audit.json
/home/kadir/sircus-mjlab/output/arm-check-speed-v1/cpu-equivalence.json
/home/kadir/sircus-mjlab/studies/arm-check-speed-v1/.study-args.json
```

The final directory binds source files, frozen inputs, positions, all plans,
selected requests and native traces. Its hashes are:

| Artifact | SHA-256 |
|---|---|
| `report.json` | `bdba220d0b41a1bf025da32d14f42d0c19ca42e78b2f06616ae0adc3e238e093` |
| `choices.json` | `53e399a4a03d6d2e0180ec00eb0029e8bdfa116e84c3c41d0a036f027b19794a` |
| `evidence-manifest.json` | `2cb1ee72d30573d01d1e5925d8912dc874fb869ca81b7aae8a60f1613c4d607f` |
| External audit | `a25c3a1f57a3cdbd1f66ed2cd9fd3446dc0a1605a8559d83ff8f41bff188dfa9` |

The driver is `python -m sircus.mjlab.arm_check_speed_study`. Its required
arguments are `--parent-study`, `--parent-audit`, `--reference`, `--assets`,
`--fast-study`, `--fast-audit`, `--acceptance-study`, `--acceptance-audit`,
`--evidence-root`, `--equivalence-report`, `--output`, `--gpu-lock`, and
`--stage`. The archived `.study-args.json` records every exact input path;
`--audit-output` supplies a separate fresh audit receipt. Assets are read
from the existing Isaac asset directory without changing that setup.

For a **new isolated repeat**, keep the audited runtime and inputs, choose
unused output/audit paths, and omit the live display destination. From
`/home/kadir/sircus-mjlab/studies/arm-check-speed-v1`, use the pinned Python:

```python
import json
import os
from pathlib import Path
import subprocess

runtime = Path('/home/kadir/sircus-mjlab/studies/arm-check-speed-v1')
python = '/home/kadir/sircus-mjlab/venv/bin/python'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
args = json.loads((runtime / '.study-args.json').read_text())
for flag, value in {
    '--output': '/home/kadir/sircus-mjlab/output/arm-check-speed-repeat',
    '--audit-output': '/home/kadir/sircus-mjlab/output/arm-check-speed-repeat-audit.json',
}.items():
    args[args.index(flag) + 1] = value
if '--display-report' in args:
    index = args.index('--display-report')
    del args[index:index + 2]
for stage in ('initialize', 'diagnose', 'prepare', 'run', 'finalize', 'audit'):
    subprocess.run([python, '-m', 'sircus.mjlab.arm_check_speed_study',
                    *args, '--stage', stage], cwd=runtime, env=env, check=True)
```

`run` waits for the shared 4090 lease, respects the one-hour native cap and
8 GiB disk reserve, and does not stop another workload. Do not point a new
run at the completed output. A repeat uses the same seeded protocol; it is
not automatically a new unseen-position benchmark if its positions were
already evaluated. A fresh benchmark needs a new frozen coordinate
inventory and seed before seeing outcomes.

Implementation checkpoints: `c526474` and `f6b9a94`,
[draft PR #153](https://github.com/kadiru/sircus/pull/153), based on draft
PR #152. The results panel is additive; it does not change the live planner.
