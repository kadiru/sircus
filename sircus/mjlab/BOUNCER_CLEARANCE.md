# Bouncer clearance: fixed-parameter XY experiment

This is a bounded development search for useful counterexamples and successful
shots. A failed contact or miss remains data. Geometric admission establishes
only that the unchanged setup checks pass; native physics determines whether the
ball clears the robot and enters the bucket. This is not another PPO training run.

The three original misses are the authenticated height/tilt records below. Their
exact connection proofs and executed joint branches are preserved for baselines.

| Source record | Board center height | Tilt from horizontal | Drop gap |
| --- | ---: | ---: | ---: |
| `height-tilt-table-training-test-00-g23` | 460 mm | 28° | 251.819 mm |
| `height-tilt-table-training-test-18-g23` | 460 mm | 28° | 240.200 mm |
| `height-tilt-table-training-test-26-g27` | 480 mm | 24° | 242.833 mm |

For each source, try all 25 combinations of X/Y offsets
`[-120, -60, 0, +60, +120] mm` from its original board center. Preserve the bucket
position, board height, board tilt and release gap. The existing bucket-facing
normal convention remains in force, so changing board XY also changes its
heading and the dropper's corresponding release XY. Original values, bucket XY,
requested values, admission proofs, executed joints and outcomes are retained
in the report and row ledger. This experiment does not keep board-to-bucket
horizontal distance constant.

The unchanged finite IK and serial setup-path admission runs on CPU before any
native flight. Every one of the 75 proposals receives a matrix row, including
rejections or unattempted rows after a budget interruption. The completed matrix
and proofs are hash-locked; native execution loads those proofs without solving
again. Every admitted candidate is flown once at **64 substeps**. Neither an
analytical clearance prediction nor a better-looking pose counts as success.

The native phase includes three matched repeats of each original shot and three
repeats each of direct-capture, table-assisted-capture and miss controls before
and after the discovery batch. For each case, a candidate is eligible for three
fresh repeats only if it is a valid native release with a board bounce and zero
measured robot-contact ticks before its scoring terminal. Eligible candidates
rank by capture, route quality, unchanged v3 reward, then frozen grid index. A
contact-free miss may be repeated and explicitly remains a miss. If no candidate
qualifies, the case has no selected repeats; a colliding shot is never filled in
as a substitute. Independent audit also separates contacts before versus after
first table/bucket contact for diagnosis; this does not alter scoring.

Maximum accounting is 75 candidate proposals + 9 originals + 18 controls + 9
selected repeats = **111 requested configurations/repeats**. CPU admission is
capped at one hour, acquired 4090 native execution at one hour, and lease waiting
at two hours. The 8 GiB free-space reserve, unchanged package pins, scene/assets,
compiled geometry, native physics options, v3 scorer and one-second whole-ball
retention remain enforced. Table-assisted captures are accepted; robot and floor
contacts before capture remain disqualifying. No noise or 128-substep experiment
is included. No alternate same-pose IK branch is substituted: that would require
a separate, versioned branch-choice proof and audit.

Run from an immutable source checkout using the existing mjlab Python environment:

```bash
python -m sircus.mjlab.bouncer_clearance_pipeline \
  --parent-study /home/kadir/sircus-mjlab/output/table-height-tilt-v1 \
  --parent-audit /home/kadir/sircus-mjlab/output/table-height-tilt-v1-audit.json \
  --reference /home/kadir/sircus-mjlab/output/corrected-grid-v1 \
  --assets /home/kadir/sircus-isaac/assets \
  --output /home/kadir/sircus-mjlab/output/bouncer-clearance-v1 \
  --gpu-lock /home/kadir/sircus-isaac/scene.lock \
  --cpu-limit 3600 --native-limit 3600 --queue-timeout 7200
```

The pipeline owns only its subprocesses, uses CPU-only admission/audit and leases
only the authorized 4090 UUID for native execution. It leaves the 4070, Isaac
workload, existing viewer and historical result namespaces untouched. An
interrupted phase preserves partial evidence and prevents later phases from
starting. UI/video publication requires the independent audit, including dense
trace replay and proof bindings; discovery success is not a generalization claim.
