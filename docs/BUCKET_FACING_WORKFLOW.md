# Bucket-facing rehearsal

The show desk has four modes. Manual operation reuses saved TCP loading and drop
poses. Simulation keeps the general numerical lab and original trained baseline.
Trained operation and Mock simulation share an operator workflow:

1. Enter the board center in world XYZ, or sample it reproducibly within an XYZ box.
2. Enter the bucket's world XY position on the table.
3. Prepare a shot. The bucket position is a supplied simulation fixture; no physical
   camera scan is claimed. Planning verifies endpoints with the provisional mounts.
4. Acknowledge that the ball is inside the open gripper. Only then is **Go!** enabled.
5. Go illustrates gripping, board positioning, carrying the ball, a two-second
   settle, release and the returned numerical ball trajectory. These illustrated
   transitions are not collision-checked native robot trajectories.

Changing the scene, random placement, planning settings or checkpoint invalidates
the old plan/loading acknowledgment. Stop cancels the numerical operation and
viewer sequence. Numerical modes have no route to a physical robot backend.

## Facing and tilt

Angles are defined from the tabletop: **0° is horizontal and 90° is vertical**.
The supported search interval lies strictly between these endpoints. The board's
upward-facing normal has a horizontal component pointing from the board center to
the bucket center. Thus the vertical plane containing that direction passes
through both centers. The board slopes down toward the bucket.

Heading is derived from `atan2(bucket_y - board_y, bucket_x - board_x)`. It is not
an independent search variable. Tilt, board-local contact U/V, and world-Z release
gap remain the numerical action variables. A board directly above the bucket has
no defined horizontal heading and must be repositioned. Reachability and finite
contact checks still apply; aiming geometrically does not guarantee capture.

## Two distinct planning methods

- `facing_search`: explicitly bounded per-scene search over tilt/U/V/gap. Each
  candidate uses the same versioned complete-shot evaluator (board bounce,
  whole-ball entry and one continuous second inside). This supplies rehearsal
  examples and potential future training labels; it is not a trained policy.
- `facing_learned`: validate and run the existing frozen checkpoint once, then
  apply an explicitly recorded geometric heading constraint. Recompute the world
  release location, final board/tool pose, endpoint IK and numerical evaluation.
  The raw prediction and constrained proposal remain distinguishable. There is no
  optimizer fallback or fit on the current scene.

The heading-constrained variant is new and unbenchmarked beyond its separate small
smoke experiment. The published **33/48** frozen-policy result belongs to its
original unconstrained action contract. Its files and scene/source fingerprint
guards are unchanged. Out-of-distribution inputs or incompatible checkpoints do
not silently become learned plans. A sampled box is a geometric bound, not a
guarantee of reachability, model coverage or successful capture.

## Reproduce the small diagnostic

Use the pinned Python environment installed by `setup-ubuntu.sh`:

```sh
sircus/simulation/.venv/bin/python sircus/studies/run_facing_rehearsal.py \
  --budget 32 --trials 5 --output experiments/facing-smoke-001.json
```

The three layouts are declared before running. Each method selects one action
with planning seed42 and evaluates it on fresh seeds1001–1005 without refitting or
reselecting. Results remain separate from earlier rectangular-container, bucket
and learning-curve evidence. The output path must be new to avoid overwriting
historical results. Five repeated releases on each of three layouts are a smoke
test, not a generalization study or physical validation.

## Deployment boundary

No new hardware adapter, physical camera scan, service restart or Viam cloud
change is required to use this numerical workflow. Board200×300×5mm and bucket
inner177/150mm, outer200/158mm remain unchanged. The reported175mm height meaning,
bucket placement, support compliance and tool transforms remain provisional.
Physical trained execution still requires commissioned perception, transforms,
ball/TCP offset, verified paths and separately authorized hardware trials.
