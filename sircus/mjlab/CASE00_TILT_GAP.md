# Case 00: fixed-position tilt/drop-gap search

This is a bounded development experiment around the recorded contact-free miss
`bouncer-clearance-table-training-test-00-g14`. It does not train PPO, install a
new selector or change the scorer. Failed setup requests and native misses remain
evidence; a finite IK failure does not establish geometric impossibility.

The original board center stays at
`(0.5979922029820652, -0.023557257454087432, 0.46) m` in the table frame. The bucket
stays at `(1.1095395651982425, -0.22146373203137287) m`. Board tilt is measured from
horizontal; the existing bucket-facing heading convention remains unchanged.

The frozen grid contains all **70** combinations of:

- Tilt: **20, 22, 24, 26, 28, 30, 32 degrees**.
- Gap offset: **−60, −40, −20, −10, 0, +10, +20, +40, +60, +80 mm**, added to the
  exact original `0.25181883224993984 m` gap. Actual gaps therefore span about
  **191.819–331.819 mm**; full values are retained in the request ledger.

Board XYZ, bucket XY, orientation convention, scoring, physics and solver limits
stay fixed. The ball's release position changes with the requested gap according
to the existing model; arm endpoint poses must pass the same exact-action
admission checks. The unchanged 28-degree/zero-offset cell retains the original
connection proof and joint branches, as do the three baseline repetitions.

All 70 CPU admission outcomes are written before flight and then hash-locked.
Every admitted candidate is flown once at **64 substeps**, using its exact saved
connection proof. There is no predicted-clearance or reward filter on discovery
flights. An admitted pose may still have a rebound collision or miss, and those
outcomes are retained rather than replaced.

After the discovery phase, freeze the complete set of **all valid native capture
rows**, ordered by original grid index. Each gets **three fresh nominal repeats**.
A capture uses the unchanged v3 evaluator: board rebound, whole-ball entry and
one continuous second retained, with table and rim assistance accepted and
robot/floor/unattributed contacts excluded. No near miss or invalid-physics row
qualifies as a capture. If no candidate captures, no repeat slots are fabricated.
Individual discovery and repeat results remain separate, including any failures
in confirmation; repeat outcomes never change the selected set retroactively.

Maximum requested configurations/repeats are:

| Stage | Maximum requests |
| --- | ---: |
| Tilt/gap proposals, including unflown declines | 70 |
| Exact original baseline repetitions | 3 |
| Direct/table/miss controls, three each before and after | 18 |
| Three fresh repeats of every discovered capture | 210 |
| Total | **301** |

Native execution uses batches of at most 32, preserves ordered request accounting,
and requires both bracketing control gates to pass. CPU admission and acquired
4090 execution are each capped at one hour, with a two-hour queue cap and 8 GiB
storage reserve. The pipeline only signals its own subprocesses at deadlines.
Partial evidence is retained and later phases do not start after a failure.

Run from an immutable source checkout with the existing pinned mjlab environment:

```bash
python -m sircus.mjlab.case00_tilt_gap_pipeline \
  --parent-study /home/kadir/sircus-mjlab/output/bouncer-clearance-v1 \
  --parent-audit /home/kadir/sircus-mjlab/output/bouncer-clearance-v1-audit.json \
  --reference /home/kadir/sircus-mjlab/output/corrected-grid-v1 \
  --assets /home/kadir/sircus-isaac/assets \
  --output /home/kadir/sircus-mjlab/output/case00-tilt-gap-v1 \
  --gpu-lock /home/kadir/sircus-isaac/scene.lock \
  --cpu-limit 3600 --native-limit 3600 --queue-timeout 7200
```

The parent report and independent 17,380-check audit are pinned by SHA-256.
Package versions, original scene/assets, compiled geometry, native physics options,
source files, requests, admission proofs and trajectories remain bound to the new
contract. CPU-only admission/audit and the leased 4090 native phase leave the 4070,
Isaac workload and existing viewer alone. No hardware, noise or 128-substep test
is included. UI/video publication requires the new independent audit; recorded
flight replay does not invent unrecorded arm transit or constitute physical
validation. This single inspected layout is not a fresh generalization test.
