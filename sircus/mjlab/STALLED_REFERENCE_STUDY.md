# A bounded fallback when reference-path search stalls

On **November 22, 2026**, the delayed fallback preserved the same **17/20
first-drop captures** as the reference-first baseline. Mean planning time was
**11.52 versus 11.44 seconds** (0.7% slower). At the separately tested old timeout,
it returned a checked shot in **21.53 seconds** and captured via the table;
the baseline returned no checked shot in **31.27 seconds**. All three frozen
rollout criteria and the independent **57,230-check audit passed**. This
qualified the implementation for a separate live-interface check. That check
subsequently captured **3/3** and the UI now defaults to this planner, while
retaining the fast and classic alternatives. The fresh-position benchmark
still shows equal capture counts, not an improvement.

This experiment keeps the ordinary reference-first planner, but changes what
happens when its reference-path search consumes **5 seconds** without returning
a checked shot. It then tries the candidate-first planner within the **same
30-second decision deadline**. The model, shot ordering, ball physics and
capture scoring stay frozen. No new policy is trained.

A reference path moves the arms from their home pose to a previously known
shot pose. Searching this path can be expensive even when the selected shot
ultimately uses a direct home-to-shot path. The previous experiment recovered
one such target with candidate-first planning, but slowed the average request
on fresh positions. The new experiment tests a delayed fallback instead of
changing the order for every placement.

## Completed comparison

| Measurement | Reference-first baseline | Delayed fallback |
|---|---:|---:|
| Captures / fresh positions | 17/20 | 17/20 |
| Direct / table-assisted captures | 14 / 3 | 14 / 3 |
| Checked shots released | 19 | 20 |
| No checked shot within the limit | 1 | 0 |
| Released shots that missed | 2 | 3 |
| Mean full planning request | 11.437 s | 11.518 s |
| Median full planning request | 6.520 s | 6.582 s |
| 90th-percentile request | 30.046 s | 30.044 s |

Both captured at the **same 17 positions**: zero gained captures and zero
lost captures. The delayed fallback activated at **1/20 fresh positions**,
**(1195.024, 160.539) mm**. It found a checked setup where the baseline had
none, but that released ball missed. Thus the extra feasible setup was not
an extra capture. The other two misses were shared.

The old timeout **(971.201, −285.388) mm** also activated the fallback:

| Method | Full planning request | Native result |
|---|---:|---|
| Reference-first baseline | 31.267 s | No checked shot; no drop |
| Delayed fallback | 21.535 s | Table-assisted capture |

This development target is separate from the fresh-position denominator.
It was tested once here, so this is not a repeatability result.

The run completed **39 fresh releases, one diagnostic release and 18 controls
(58 total)** at 64 substeps. Both control blocks passed. Native work used
**293.412 leased seconds on the 4090**. No invalid-physics result or
robot-assisted capture appeared in the fresh comparison. The model was not
retrained; the 4070 workload and Isaac setup were unchanged.

| Predeclared rollout condition | Result |
|---|---|
| Capture count at least baseline | **Pass:** 17 ≥ 17 |
| Mean request time no more than 10% slower | **Pass:** 11.518 / 11.437 = 1.007 |
| Old timeout captured | **Pass:** table-assisted capture |
| Independent evidence verification | **Pass:** 57,230 checks |

The baseline in this experiment includes the earlier optimized collision/path
checks. It is not a new timing measurement of the historical live fast option.
After the separate functional check below, the running interface on **port
18781** was promoted to the new `hybrid` default. The original fast and classic
implementations remain available as explicit alternatives.

## Frozen comparison

| Item | Protocol |
|---|---|
| Baseline | Reference-first planning with optimized geometry/path checks |
| Hybrid | Same baseline; switch after 5 seconds of actual reference-path search |
| Total decision budget | 30 seconds, shared across both phases |
| Fresh evaluation | 20 new bucket positions, one selected drop per method per position |
| Diagnostic | Previously timed-out target, separate from the 20 fresh positions |
| Accepted arm paths | Same 5 mm clearance, detailed collisions, 0.5° samples in both directions |
| Native drops | MuJoCo, 64 substeps, arms held at the selected endpoints |
| Controls | Matched saved controls before and after comparison drops |

The fallback is cooperative and sequential. It interrupts only before a
selected shot has been published; it does not run two planners in a race.
The decision deadline does not restart when the fallback begins. Full request
time also includes worker initialization and result handling, which can take
it beyond 30 seconds. Browser/network time, GPU waiting and native drop time
are separate measurements.

Candidate-first planning can use a different initial joint-angle hint from
reference-first planning. It therefore may find different arm configurations
for the same requested board and release position. Every accepted endpoint
and complete path must still pass the same checks. No clearance test is
relaxed to recover a timeout.

Capture means a board rebound followed by one continuous second with the
whole ball inside the bucket. Direct and table-assisted captures count;
robot-assisted outcomes remain separately labeled. No-setup positions remain
in the denominator. A feasible setup or a predicted capture is not a measured
capture. This experiment does not validate camera perception or moving-arm
execution.

## When the interactive planner may change

Before observing results, the rollout criteria are:

1. The independent evidence audit passes.
2. Hybrid captures at least as many fresh positions as the baseline.
3. Its mean full planning request is no more than **10% slower**.
4. Its selected native drop captures at the old timed-out target.

All four are required. Twenty fresh positions provide limited comparison
evidence, not a guarantee across the table. An adapter must separately verify
the frozen source, input and proof identities before advertising the hybrid
planner. The existing fast and original planners remain selectable. A failed
request never starts another drop or changes planners silently.

## Separate live-interface acceptance and rollout

The staged API completed the exact plan → preview → one-drop flow for three
placements. Saved native traces were independently rescored and their request,
preview, flight and frozen-source identities checked.

| Placement | Bucket X, Y (mm) | API planning wait | Outcome |
|---|---:|---:|---|
| Known old timeout | 971.201, −285.388 | 23.695 s | Table-assisted capture |
| New southwest placement | 374.134, −355.879 | 5.542 s | Direct capture |
| New northeast placement | 972.722, −101.860 | 16.589 s | Direct capture |

These **3/3 captures are a functional check**, separate from the 20-position
paired benchmark. The waits above include API polling and are not the study's
server-only request times. All three duplicate-release requests and both
stale-job requests were rejected. This small check does not establish overall
capture reliability.

The verified server now runs on **18781 with `hybrid` selected by default**;
`fast` and `classic` remain available. Deployment kept the frozen computational
source at the study commit. Only the previous idle simulation server and this
task's staging instance were gracefully stopped. The launch and rollback
arguments are preserved outside Git. The 4070 and Isaac setup were unchanged.

```text
/home/kadir/sircus-mjlab/output/stalled-reference-live-v1/acceptance-report.json
/home/kadir/sircus-mjlab/output/stalled-reference-live-v1/live-launch.json
/home/kadir/sircus-mjlab/output/stalled-reference-live-v1/rollback-launch.json
```

Acceptance receipt SHA-256:
`cccec2cac92a9e905d438b2106f4e6ad279595a83a0f4b693e4351a809afa3d1`.
The compact baseline also records launch and metadata-restoration receipt hashes.

## Results and reproduction

The task driver is `python -m sircus.mjlab.stalled_reference_study`. It freezes
inputs, coordinates and selected commands before native outcomes are observed.
The output report uses `stalled_reference_study`, method IDs `baseline` and
`hybrid`, separate per-position outcomes, and a separate diagnostic replay.
Raw evidence, generated configurations and logs remain outside Git.

The UI section is **`#stalledReference`**. It displays progress while running.
Final counts appear only after the report is complete, the audit passed and
its totals reconcile with the per-position records. Missing native outcomes
remain “Not run.” The UI report alone never enables a planner; available
choices come from the running server's verified capabilities.

The compact checked aggregate is
[`baselines/mjlab-stalled-reference-20261122/summary.json`](../../baselines/mjlab-stalled-reference-20261122/summary.json).
The preserved raw report retains `audit_status: pending`; its separate passed
audit receipt is the verification authority. No result or selection was edited
after native execution to make that receipt pass.

Frozen producer: **`be97c2ab6c62bb36a93b90a224024aa4c1814789`**.
The later commit fixing optional-Torch test collection changed only tests;
the experiment continued using the frozen producer.

```text
/home/kadir/sircus-mjlab/studies/stalled-reference-v1/
/home/kadir/sircus-mjlab/studies/stalled-reference-v1/.study-args.json
/home/kadir/sircus-mjlab/output/stalled-reference-v1/
/home/kadir/sircus-mjlab/output/stalled-reference-v1-audit.json
```

| Artifact | SHA-256 |
|---|---|
| Raw report | `e8cd7009362af9c80f7fe6a54b70937b6cbef8f4182eaf04d22e6b7e8dbac2e1` |
| Frozen choices | `bab9c33832564689f28532d4fc04d3087467f3cedf4cab51f83e4b1dc325849b` |
| Evidence manifest | `b10cc44774f3d1e822a8beaaa4db7698a11ba47597200c57795518d2bb91a394` |
| Contract | `3f36fe9c893f63505c54092bc5d81b98057e0a325822396e78b261b677efac58` |
| External audit receipt | `3e98667dd739ed9a390d54ae0176a8b22bb400edbe0c39f61f9a0e1b8fcb07fe` |
| Verifier source | `c048a0b7b319f4bc95ab270071847a3bfab663f7355113f87907db91d73657e5` |
| Launch argument manifest | `52fa6a9d5682615658ed770b868f69775c463fc107695c6fca8249323ec297f8` |

The aggregate also records layout, plan and unchanged model hashes. Raw flight
arrays, logs and generated runtime configurations remain outside Git.

To repeat the **protocol** in the pinned Ubuntu environment, retain the frozen
runtime and dependencies; choose two new output paths and remove live display
publication from the saved arguments. Run the phases in order:

```python
import json
import os
from pathlib import Path
import subprocess

runtime = Path('/home/kadir/sircus-mjlab/studies/stalled-reference-v1')
args = json.loads((runtime / '.study-args.json').read_text())
for flag, value in {
    '--output': '/path/to/new-unused-output',
    '--audit-output': '/path/to/new-unused-audit.json',
}.items():
    args[args.index(flag) + 1] = value
if '--display-report' in args:
    i = args.index('--display-report')
    del args[i:i + 2]
env = dict(os.environ,
    CUDA_VISIBLE_DEVICES='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
for stage in ('initialize', 'diagnose', 'prepare', 'run', 'finalize', 'audit'):
    subprocess.run([
        '/home/kadir/sircus-mjlab/venv/bin/python',
        '-m', 'sircus.mjlab.stalled_reference_study',
        *args, '--stage', stage,
    ], cwd=runtime, env=env, check=True)
```

The argument manifest contains the exact audited parent studies, original
assets, shot libraries, neural checkpoint and shared GPU lease path. The
initialization inventories existing coordinates to prevent test leakage.
An updated inventory can change a later run's sampled positions; the exact
published positions and commands remain in this run's `layouts.json` and
`choices.json`. Repeating previously observed positions does not make them
unseen again. Native work waits for the shared 4090 lease and retains the
one-hour GPU cap and 8 GiB free-space reserve.

For **live adapter startup only**, the historical fast-parent source guard
also requires the byte-identical Apple metadata artifact
`sircus/mjlab/._neural_shot_model.py`, which is absent from the Git archive.
The staged runtime restored that single artifact from its prior runtime and
verified its expected SHA-256; no computational Python source or guard changed.
The restoration receipt is kept outside Git at
`/home/kadir/sircus-mjlab/output/stalled-reference-live-v1/historical-metadata-restoration.json`.
This is not a prerequisite for the study computation itself. Do not fabricate
metadata bytes or weaken the guard to start the server.
