# Adapting nearby successful shots to five missed bucket positions

The preceding parallel-board comparison preserved the original selected shots
but recovered none of five development misses. This experiment changes the
starting configurations: use settings that actually captured at nearby bucket
positions, transfer them to each missed target, check both arms, and test the
resulting ball flights. It does not train or promote a new policy.

## Completed result — November 23, 2026

[Results and exact commands](../../baselines/mjlab-nearby-shot-transfer-20261123/README.md):
**three of five known missed positions recovered.** Search captures from 20
settings per target were **16 / 0 / 10 / 16 / 0** in the target order below.
All 42 qualifying settings passed **3/3 fresh repeats (126/126)**; unchanged
originals remained **0/15**. The third recovered target used the table-assisted
route. Both unresolved targets had robot contact in every tested flight.

All **259 native releases**, both control blocks and the **56,065-check final
audit** completed; **973.070 seconds** of leased 4090 time. One arm-assisted
retained-ball outcome remains explicitly unconfirmed and one invalid-physics
outcome remains in the search denominator. The UI publishes the audited result
at `#nearbyShotTransfer`. No live planner switch, retraining or service restart.

## Sources and fixed targets

The source bank binds raw commands, actual native joint vectors, trajectories,
prepared proofs, manifests and passing independent audits from three immutable
studies:

- `board-pose-recovery-v2`: seven settings, each with three successful fresh
  confirmations (six direct routes and one table-assisted route).
- `parallel-board-v2`: the known successful control, with three successful repeats.
- `guarded-board-fallback-v1`: fifteen successful settings with one observed
  capture each. These are explicitly weaker evidence than repeated sources.

That is **23 source records, eight with 3/3 repeat evidence**. Source records
can duplicate physical settings across studies; they are not asserted to be
23 independent configurations. Identical transferred target geometries are
deduplicated before native testing.

The targets are the five audited `previous_miss` roles in `parallel-board-v2`:

| Target suffix | Bucket X, Y (mm) |
|---|---:|
| 03 | 942.599, −31.690 |
| 09 | 1192.994, 147.423 |
| 13 | 966.733, −40.910 |
| 15 | 1209.589, −160.593 |
| 19 | 1127.555, −38.348 |

These are selected known misses, **not unseen positions**. Their exact original
commands and evidence stay intact and receive three matched repeats each.

## What changes

For each source setting and target bucket, construct three deterministic families:

1. Keep board center X/Y/Z and re-aim its heading toward the target bucket.
2. Translate board and release X/Y by the bucket's X/Y displacement.
3. Apply half that X/Y displacement as an intermediate choice.

The source heading offset relative to facing the bucket and board twist are
preserved. Each family also tests a fixed menu of small adjustments: board
height up to ±40 mm, tilt up to ±6°, release gap up to ±80 mm, and heading
offset up to ±8°. Selected paired adjustments use ±20 mm height, ±3° tilt and
±40 mm gap. All candidates remain within the existing action bounds and keep
board-to-bucket horizontal distance greater than 150 mm.

The release gap means released **ball bottom Z minus board-center Z**; release
center Z additionally includes the ball radius. A new target recalculates the
bucket-facing gripper orientation through the existing target construction.

Source joint vectors only initialize IK. Every proposed command needs new full
pose, endpoint-clearance and serialized-path proofs at the target bucket.
Neither an old proof nor a successful source label admits a transferred shot.

## Bounded checking and testing

- CPU preparation uses at most five processes, one per target, with a
  600-second preparation deadline, up to 96 regular checks and eight existing
  16-start IK retries per target. Necessary wrist reach screens are not proof
  of complete arm feasibility.
- Preparation reserves unchanged transfers from eight nearby source settings,
  including three with confirmed repeat evidence when available. Remaining
  slots alternate between source settings and transfer families rather than
  allocating the whole budget to one predicted winner.
- Up to 20 distinct admitted target geometries per target are frozen before
  the GPU starts. Selection reserves admitted unchanged source transfers and
  each transfer family, then fills with diverse ranked candidates. Approximate
  rim/arm-clearance predictions rank experiments; they do not prove capture.
- Native drops run at the existing calibrated **64 substeps** on the RTX 4090
  under the shared lease. The 4070 and Isaac workloads are untouched.
- Test each selected setting once. Repeat **every valid direct or table-assisted
  search capture three additional times**, regardless of its intended route.
  Robot-assisted retained balls remain separately labeled and do not enter
  this confirmation group. Keep all misses and invalid outcomes in evidence.
- Run three unchanged originals per target and nine matched controls before
  and after. Maximum: 100 search + 300 confirmations + 15 originals + 18
  controls = **433 native releases**. Leased GPU time is capped at 3,600 seconds.

All sources, proposals, admissions, selected commands, code and schedules are
hash-bound. Independent preflight must pass before the first native release.
Native-started is created exclusively **after** acquiring a free 4090 lease;
a bounded GPU wait that expires before any release can retry the same frozen
preparation. Once native execution starts, the runner refuses blind reruns.

The final audit independently checks the saved trajectories, direct/table
scoring, exact controls, original commands and every fresh confirmation before
publishing counts. Report search captures separately from 3/3-confirmed settings.

## Reproduction

Use the pinned mjlab environment, original assets/reference and parent controls;
do not install new versions into the Isaac environment. All source/output paths
must be distinct, with generated evidence outside Git. The study CLI requires:

```text
python -m sircus.mjlab.nearby_transfer_study
  --target-study <parallel-board-v2> --target-audit <its audit.json>
  --recovery-study <board-pose-recovery-v2> --recovery-audit <its audit.json>
  --guarded-study <guarded-board-fallback-v1> --guarded-audit <its audit.json>
  --parent-study <fixed35-sixhour-v2> --parent-audit <its audit.json>
  --reference <corrected-grid-v1> --assets <original assets>
  --output <new empty study directory> --gpu-lock <shared scene.lock>
  --stage prepare|freeze|run
```

Prepare with `CUDA_VISIBLE_DEVICES=''` and one BLAS/Torch thread per process.
Freeze after all five preparations finish. Run the independent preflight and
pass its path using `--preflight-audit` for the run stage. Native execution must
expose only the authorized RTX 4090 UUID; GPU wait is bounded to 1,800 seconds
by default. Follow with the independent full audit and publisher.

The read-only UI section is `#nearbyShotTransfer`, from the display field
`nearby_shot_transfer`. Publishing does not restart a service or change the live
planner. The native test holds arms at checked endpoints: it validates ball
dynamics, not driven robot motion, perception or physical hardware.
