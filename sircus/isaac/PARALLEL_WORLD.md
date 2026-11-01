# Parallel two-arm workcells

This opt-in mode builds one GPU PhysX world containing 32 independent workcells:
64 articulated UF850 arms, 32 wrist-attached boards, 32 fixed buckets, 32 tables,
and 32 dynamic balls. It uses Isaac Lab `InteractiveScene` cloning with environment
collision filtering. Each cell is translated 4.5 m from its neighbors; the 4 m
floor footprints do not overlap. It does not use Viam or physical hardware.

[Completed first experiment and exact settings](../../baselines/isaac-parallel-world-20261101/README.md).

## Fixed targets, varied actions

Distinct **table-local** bucket coordinates, in metres:

- X: 0.47, 0.57, 0.67, 0.87.
- Eight Y rows linearly spanning −0.40 to −0.70.
- Board/release XY: (0.67, −0.10) in every cell.
- Horizontal board-to-bucket distances: 300–632.46 mm.

“Keep the distance consistent” means the bucket XY and horizontal distance stay
fixed **within each cell across all trials**. Distances differ between cells.
World translation alone does not count as a distinct target. The runtime checks
actual bucket positions after initialization, contact probes, preparation and shots.

The board faces its own target. Tilt is measured from horizontal. The finite
experiment tries 28 H/gap/tilt combinations per target: H = 350/400/450 mm,
gap = 120/180/240 mm, tilt = 18/24/30°, plus H350/gap200/tilt24°.
The same list rotates between cells, so different cells try different combinations
simultaneously but eventually receive every candidate. Two declared control
batches precede this search. The best qualified candidate is frozen per cell and
receives ten seeded trials with H/gap ±2 mm and tilt ±0.5°. No confirmation shot
retunes its action after seeing its outcome.

H is the board-center height above tabletop Z=0. The ball-bottom release height
is H+gap; ball-center Z adds 37.5 mm. The provisional flange offset adds 95 mm.
Arms initialize at solved joint endpoints, settle under gravity compensation,
and pass measured pose/velocity/drift gates before release. Board corrections
change joint targets, not measured acceptance tolerances. Dropper flange position
must be within 2 mm. Full path planning, pickup, gripper actuation and collision-free
transit are not validated by this mode. Robot self-collision is disabled as in the
existing endpoint study; this is not a whole-motion safety test.

## Physics, contacts and scoring

Shared physics runs at 240 Hz. Per-ball contact matrices distinguish board,
bucket, table, floor, robot bodies and unattributed force. Each run probes bucket
bottom/rim, table, floor and robot attribution in every cell. A foreign-cell
probe places cell0's ball over cell1's bucket: the ball must pass through the
foreign bucket/table/floor without reporting contact. A failed probe aborts.

Historical capture remains board bounce + whole-ball entry + one continuous
second retained. Additive `sircus-board-route-v1` also rejects detected
non-board/non-bucket contacts; rim-assisted captures count. Exactly one board
contact episode is reported separately. Clean=2, assisted/marginal=1, miss=0;
invalid IK/settling is not a released miss. Force threshold is 0.01 N; 50 ms
contact-free separates board episodes. This discretization does not prove that
no subthreshold or unsampled contact occurred.

Preserved geometry: 300×200×5 mm landscape board, 1,337.31 mm arm-base spacing;
177/150 mm inner and 200/158 mm outer bucket top/bottom diameters. **175 mm is
reported height; overall height versus interior depth is still unconfirmed.**
The existing model assumes 175 mm interior depth and an 8 mm bottom. Ball size,
mass, material properties and rigid board mounting remain provisional. Older
rectangular and bucket baselines are unchanged.

## Run on the isolated 4090

Use the established Isaac Lab 2.3.2/Sim 5.1 container and generated arm assets from
[setup](README.md). First inspect actual GPU jobs and the scene lock. Do not
replace an active study or start the same cloud part on another host.
Copy the desired source to an immutable directory mounted below `/workspace/studies/`;
set `SIRCUS_ISAAC_SOURCE` to that directory. Example inside the container:

```bash
export SIRCUS_ISAAC_SOURCE=/workspace/studies/parallel-world-v4
flock -n /workspace/scene.lock "$SIRCUS_ISAAC_SOURCE/python.sh" \
  "$SIRCUS_ISAAC_SOURCE/parallel_world.py" \
  --headless --enable_cameras --device cuda:0 \
  --grid-count 32 --num-envs 4 --smoke

flock -n /workspace/scene.lock "$SIRCUS_ISAAC_SOURCE/python.sh" \
  "$SIRCUS_ISAAC_SOURCE/parallel_world.py" \
  --headless --enable_cameras --device cuda:0 \
  --grid-count 32 --num-envs 32 --batches 28 --confirmations 10 --video-batches 1
```

The container exposes only the 4090; `cuda:0` is its container-local index.
The finite run needs at least 3 GiB free disk. First experiment used about 6.2 GiB
of GPU memory, 20 MiB of run artifacts before video encoding, and 197.3 s after
application launch. These are measured for this workload, not guaranteed PPO
throughput. The 4070 workload is independent and was left running.

`--num-envs 1 --cell-start 0 --smoke` provides a small serial reference. Smoke
runs preserve their own evidence and do not replace the dashboard's latest study.
The grid contract allows 4/32/64 targets, but only the 32-target world and its
one-/four-cell subsets have been runtime tested here.

## Evidence and interface

- `parallel_contract.py`: frozen cells, actions, perturbations and fingerprint.
- `parallel_assets.py`: cloned arm/table/bucket/ball configuration and local bucket USD.
- `parallel_runtime.py`: batched endpoint preparation, physics, contact probes,
  measured gates and per-cell scorers. It is not yet a Gym/RL adapter.
- `parallel_world.py`: bounded study, per-run outputs and atomic latest report.
- `parallel_report.py`: validates contract, declared plans/seeds, source/artifact
  hashes, raw/compact records, settling and score consistency; exports JSON/CSV.
- `parallel_video.py`: combines matched overview/close-camera frames into an MP4
  only after completed evidence is verified. It leaves an unrelated latest run alone.

Validation example, using the same executed source snapshot:

```bash
python parallel_report.py /path/to/run /path/to/verified-export \
  --executed-source /path/to/executed-source
```

Publish the verified JSON/CSV to that run directory; then `python parallel_video.py
/path/to/run` encodes its saved frames. FFmpeg is a host-side dependency; it does
not need a new simulator installation. Raw trajectories, frames, generated USD,
logs and local runs stay outside Git. Compact synthetic results/hashes are under
`baselines/isaac-parallel-world-20261101/`.

The loopback read-only preview serves `/parallel-world.html`; the existing Mac
forward is `http://127.0.0.1:18780/parallel-world.html`. It shows actual rendered
views, optional cell labels, exact coordinates/heights, search versus confirmation
results, invalid counts, CSV and a run-matched video. Selection changes the detail
panel; the close camera stays on cell0 and is explicitly labeled. Both cameras
produce ideal RGB and axial depth internally, but this study uses simulator state
for scoring; it does not train on segmented point clouds or save a cloud dataset.
No browser endpoint can move a robot or start a simulator job.

## Remaining learning boundary

This is a parallel **parameter-search environment and development catalogue**,
not a newly trained RL policy or held-out generalization test. Far-target coverage
and dropper reachability remain limiting. One borderline control changed outcome
between one/four cells and 32 cells, so strict clone equivalence is not established.
The selected near setting still passed its independent perturbations.

Next: a feasibility-aware three-action batched policy adapter (H, gap, tilt),
with target XY/distance as observation, disjoint spatial splits, and frozen-policy
assessment. Preserve the current scorer and reject invalid actions explicitly.
Investigate far shots and leave the endpoint arm-travel boundary explicit before
running a large PPO study. Perception observations and full joint-trajectory
learning remain separate steps.
