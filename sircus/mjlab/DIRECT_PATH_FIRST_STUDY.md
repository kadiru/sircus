# Check the candidate path before the reference path

On **November 21, 2026**, checking the candidate path first recovered the
previously timed-out target: a checked shot was ready in **15.96 seconds**
and its one native drop captured via a table bounce. On **20 fresh bucket
positions**, both methods captured at the same **16/20 positions**. Candidate
first was slower on average: **14.06 versus 11.87 seconds**. Keep the current
live planner as the default; this result supports a targeted alternative,
not a general speed improvement.

The change affects the order of arm-path checks and how their initial
joint-angle hint is obtained. It does not retrain the model or change ball
physics.

## Completed results

| Measurement | Reference path first | Candidate path first |
|---|---:|---:|
| Captures / fresh positions | 16/20 | 16/20 |
| Direct / table-assisted captures | 12 / 4 | 12 / 4 |
| Checked shots released | 20 | 20 |
| Released shots that missed | 4 | 4 |
| No checked shot within the limit | 0 | 0 |
| Mean full planning request | 11.87 s | 14.06 s |
| Median full planning request | 7.13 s | 9.22 s |
| 90th-percentile request | 30.07 s | 30.05 s |

There were **zero gained or lost captures** on the fresh positions. All four
misses occurred at the same positions. Full request time includes worker
initialization and result handling, but excludes browser/network time, GPU
waiting and drop simulation. Both methods retain a 30-second decision limit;
request overhead can take total time beyond that limit.

The separately inspected target **(971.201, −285.388) mm** produced:

| Method | Full planning request | Outcome |
|---|---:|---|
| Reference path first | 31.30 s | No checked shot; no release |
| Candidate path first | 15.96 s | One release; table-assisted capture |

This one development-target capture is **outside** the 20-position test
counts and does not establish repeatability. The test completed **40 fresh
comparison drops, 1 diagnostic drop and 18 matched controls**. Both control
blocks and the independent **54,351-check audit passed**. Native work used
**215.94 leased seconds on the 4090**. The 4070 and Isaac setup were unchanged.

## Why change the order?

A proposed shot specifies where the board and ball release should be.
Inverse kinematics finds the arms' joint angles for those endpoints. A
path check then asks whether the arms can move there from their home pose
without violating the 5 mm clearance requirement.

The earlier planner first finds a path to a previously known reference
shot, then connects that reference pose to the proposed shot. At the
previously timed-out bucket position, **(971.201, −285.388) mm**, the original
completed plan spent **85.73 seconds** searching for the reference path.
Its eventual successful shot used a different, direct home-to-shot path.
The previous speed improvements still could not return a checked shot at
this target within the 30-second decision limit.

The new order checks the actual candidate's direct path first. If that
path fails, the reference-path search remains available as a fallback.
“Direct” here describes the **arms' path**; it does not restrict the ball
to a direct board-to-bucket flight. Table-assisted captures still count.

## Controlled comparison

| Item | Protocol |
|---|---|
| Baseline | Optimized geometry checks; reference path searched first |
| Change | Same checks; candidate's direct home path checked first |
| Decision limit | 30 seconds per method and bucket position |
| Fresh evaluation | 20 new bucket positions; at most one drop per method at each |
| Development target | Old timeout tested separately, outside the fresh-position denominator |
| Frozen inputs | Same model, shot ordering, geometry and physics settings |
| Accepted paths | Same 5 mm clearance, detailed collisions, 0.5° motion samples in both directions |
| Native simulation | MuJoCo at 64 substeps, arms held at selected endpoints |
| Controls | Same saved controls before and after the test drops |

The neural model, proposed shot menu and candidate ordering stay frozen.
However, the candidate-first version has not yet solved a reference pose:
its IK solver starts from a deterministic joint-angle hint associated with
the source geometry, or explicit candidate joints when supplied. The
reference-first version can use the solved reference joints. These different
starting hints can lead IK to different joint configurations for the same
requested endpoint. Each accepted connection records its hint and source
provenance. This is therefore not a claim of identical arm configurations
with only a reordered stopwatch; all accepted configurations must still pass
the same endpoint and full-path checks. No new policy is trained.

All selections are saved before native outcomes are observed. No-setup
positions stay in each method's denominator. Compare both capture counts
and full planning time; finding a feasible arm pose alone does not prove
a capture. The old target is development evidence, not a fresh test.

Capture requires a board rebound followed by one continuous second with
the whole ball inside the bucket. Direct and table-assisted routes count;
robot-assisted outcomes remain separate. This experiment does not train a
new model, use camera perception, or validate moving-arm execution.

## Evidence and interface

The driver is `python -m sircus.mjlab.direct_path_first_study`. It freezes
source hashes, input hashes, test coordinates and selections; records
separate reference-first and candidate-first evidence; then independently
audits the native results. The output directory and exact launch arguments
are recorded with the run. Raw observations, generated runtime files and
logs stay outside Git.

The additive UI section is **`#directPathFirst`**, sourced from
`direct_path_first_study`. It shows planning/drop progress while running.
Capture counts appear only when the report is complete, its independent
audit passed, and per-position outcomes reconcile with the totals. The
previously timed-out target has its own table. A missing native outcome
is shown as “Not run,” never a predicted capture.

The live UI service was not running after the Ubuntu reboot. It was resumed
with its previous launch arguments to show these results; no active service
was interrupted and the planner default did not change. The 4070 workload
and Isaac setup were not modified.

Saved evidence remains outside Git:

```text
/home/kadir/sircus-mjlab/output/direct-path-first-v1/
/home/kadir/sircus-mjlab/output/direct-path-first-v1-audit-v2.json
/home/kadir/sircus-mjlab/studies/direct-path-first-v1/.study-args.json
/home/kadir/sircus-mjlab/studies/direct-path-first-audit-v2/.audit-v2.py
```

The producer runtime is frozen at implementation commit `c44b2a6`. The first
audit rejected a naming mismatch for the diagnostic attempt with **no
release**. Verifier v2 corrects that label handling and verifies the original
producer source tree using `frozen_source_root`; it records its own source
hash separately. No shot was rerun, no physics/scoring changed, and the raw
results remained immutable. Consequently, the raw report still says audit
pending; the separate passed receipt supplies the UI's verified status.

| Artifact | SHA-256 |
|---|---|
| Raw `report.json` | `a4dce424945906b1d84005e40ed1c7c599e9ddad9b517d3f77ce029cc7c9034b` |
| Frozen `choices.json` | `5111f6f9b26a697b64a0263aaf302fdb1a81e62fe8472f9483b973d5d96463e4` |
| `evidence-manifest.json` | `8f8a8a1e2f8a8c74a014a927c8bee592ed718087304a315bff94a35a0bb912f4` |
| External audit receipt | `4ef646a7a673bb0a456275383634f2749f3efa1a4b51e22bf61def568660411c` |
| Verifier source | `b99cd03075845ed0e17b25a32538d751b8b029d7420df6ae3917108d8d4338be` |

## Reproduce the run

Use the frozen runtime and its pinned Ubuntu Python environment. Keep the
completed output immutable. The driver accepts these exact arguments:

| Required argument | Input |
|---|---|
| `--parent-study` | Audited native study supplying geometry, controls and scoring inputs |
| `--parent-audit` | That study's passed audit receipt |
| `--reference` | Original corrected-grid reference data |
| `--assets` | Existing read-only robot/scene assets |
| `--fast-study` | Completed fast-planning study supplying the frozen selector and shot libraries |
| `--fast-audit` | That study's passed audit receipt |
| `--acceptance-study` | Completed live acceptance comparison, including the old timed-out target |
| `--acceptance-audit` | That comparison's passed audit receipt |
| `--speed-study` | Completed optimized-arm-check comparison |
| `--speed-audit` | That comparison's passed audit receipt |
| `--evidence-root` | Existing interactive evidence, inventoried to exclude previously tested coordinates |
| `--equivalence-report` | Passed CPU geometry/path-check equivalence report |
| `--output` | New, unused output directory |
| `--gpu-lock` | Existing shared 4090 lease path |
| `--stage` | One of `initialize`, `diagnose`, `prepare`, `run`, `finalize`, `audit` |

`--audit-output` supplies a new receipt path **outside** the output directory
and is required by the audit stage. `--gpu-wait-s` optionally changes the
maximum wait for the shared GPU (default 43,200 seconds). `--display-report`
is optional; omit it for a repeat that should not publish to the live UI.

The archived `.study-args.json` records the exact input paths used for the
reported run. To repeat that frozen experiment, use the original producer
for the five simulation stages and verifier v2 for the final audit. Run this
with `/home/kadir/sircus-mjlab/venv/bin/python`, filling the two new output
paths below. The current branch's driver also supports ordinary new runs
with verifier v2; do not use the old frozen producer's `--stage audit`.

```python
import json
import os
from pathlib import Path
import subprocess
import sys

runtime = Path('/home/kadir/sircus-mjlab/studies/direct-path-first-v1')
verifier = Path('/home/kadir/sircus-mjlab/studies/direct-path-first-audit-v2')
output = '/path/to/new-unused-output'
audit_output = '/path/to/new-unused-audit.json'
args = json.loads((runtime / '.study-args.json').read_text())
for flag, value in {'--output': output, '--audit-output': audit_output}.items():
    args[args.index(flag) + 1] = value
if '--display-report' in args:
    index = args.index('--display-report')
    del args[index:index + 2]
env = dict(os.environ,
           CUDA_VISIBLE_DEVICES='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
for stage in ('initialize', 'diagnose', 'prepare', 'run', 'finalize'):
    subprocess.run([
        '/home/kadir/sircus-mjlab/venv/bin/python',
        '-m', 'sircus.mjlab.direct_path_first_study',
        *args, '--stage', stage,
    ], cwd=runtime, env=env, check=True)

# Audit the original producer with the separately versioned verifier.
sys.path.insert(0, str(verifier))
from sircus.mjlab.direct_path_first_study import argument_parser
from sircus.mjlab.direct_path_first_audit import audit
from sircus.mjlab.sweep_worker import atomic
audit_args = argument_parser().parse_args([*args, '--stage', 'audit'])
audit_args.frozen_source_root = runtime
if Path(audit_output).exists():
    raise ValueError('Use a fresh audit receipt path')
result = audit(audit_args)
atomic(Path(audit_output), result)
assert result['status'] == 'passed'
```

This repeats the frozen coordinate-generation protocol; it does not make
already tested coordinates unseen again. A genuinely new benchmark must
freeze an updated coordinate exclusion inventory and a new seed before
observing outcomes. Native work waits for the shared 4090 lease and respects
the one-hour GPU cap and 8 GiB storage reserve without stopping another job.
