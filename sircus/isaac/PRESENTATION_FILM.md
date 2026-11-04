# One-minute native Isaac farm film

This filming mode records fresh Isaac Sim / PhysX trials at all 35 tables from
the user's saved perspective camera. The camera stays fixed, with a close
foreground table and the remaining farm behind it. The silver-gray tables,
neutral floor and soft lighting preserve the existing stage appearance.

This is a presentation of qualified shot configurations, not a new RL training
run. Arm transit is driven by the existing scripted articulation controller;
balls are initialized at the measured release poses. Ball motion and contacts
are newly simulated. No saved ball trajectories, generated imagery, learned
grasp, learned joint path or physical deployment are involved. The source PPO
study remains a separate, frozen experiment.

## Frozen source and admission checks

The configured source is `output/expanded-training-20261102-v4`, produced with
the immutable `studies/expanded-farm-20261102-v6` source. Its portable evidence is
under [baselines/isaac-expanded-farm-20261102](../../baselines/isaac-expanded-farm-20261102/README.md).
That study contains 35 distinct bucket placements, 24 native-qualified geometry
options per table, ten training batches and five PPO updates. Its capture counts
are training results, not an evaluation of this film.

The recorder requires the completed native source's original files:

- `contract.json` and `result.json`.
- `expanded-menu-qualified.json`, including the endpoint fits and measured
  native qualification evidence for each option.
- `selected-camera.json` and `camera-binding.json`.
- Passing `probes.json` and `rim-rays.json`.

It verifies the contract fingerprint, exact camera bytes, menu checksum, camera
binding, original bucket identities and original source dependency hashes.
`film_cylinder_farm.py` is the sole intentionally updated historical Python file:
its additive presentation route is recorded with both its old and current hash.
Historical physics, scoring, asset, camera and controller helpers must still
match the completed source. New recording helpers receive their own hashes.

The original contract is copied byte for byte and retains `training_batches=10`.
Its fingerprint is
`63362ef37a93881f769b39d1ee7456335f1cc7bf5db04ffa7e548ce9697455bf`.
The camera retains its original source fingerprint
`98ebe47b48a01b9028747ab03bc913a033d90611e44f03b912e58f41c87ff79a`;
the separate binding connects the unchanged camera to the expanded task.

Before scripted transit, every reused endpoint solution is checked against its
joint limits and forward kinematics before entering the original IK cache.
This reuses proven branches without repeating 840 static qualification holds.
Fresh contact/isolation probes, measured arm settling and per-attempt release
guards still run. Static qualification does not certify a collision-free path.
A fresh rejected release stays in the evidence; it is not replaced with a
successful saved outcome.

Geometry remains unchanged: the board is 200 × 300 × 5 mm, the bucket uses the
measured tapered inner/outer diameters, and the reported 175 mm bucket height
still has unresolved overall-height versus interior-depth interpretation.
The provisional depth and bottom-thickness assumptions remain explicit in the
frozen scene. This filming mode makes no new calibration claim.

## Camera and configuration variety

The exact selected camera JSON is required, including its original serialization.
The chosen file was `sircus-camera-2026-11-02T20-24-07.json`:

| Camera property | Value |
|---|---|
| Eye, metres | `[10.9706, -6.1943, 2.5406]` |
| Target, metres | `[7.8307, -3.4877, 0.3904]` |
| World up | Z |
| Focal length | 23 mm |
| Aperture | 36 × 20.25 mm |
| Native frame size | 1920 × 1080 |

The values above are readable rounded representations; use the saved JSON bytes
when reproducing the recording. Each frame logs the actual camera readback.
There is no camera orbit or ball following in this fixed-angle film.

The separate `plan.json` defaults to 30 cycles, or 1,050 attempted table shots.
The bounded CLI accepts 1–60 cycles; one cycle is useful for a native preview but
cannot satisfy the one-minute deliverable. All 35 cells remain enabled, with
their original distinct local bucket positions and the same whole-cell display
permutation used by the selected camera. The foreground is source cell index 1;
its display location does not change its bucket identity.

At each table, the plan covers five normalized geometry dimensions:

1. Bucket-relative board bearing.
2. Horizontal board-to-bucket distance.
3. Board-center height above the tabletop.
4. Vertical ball-bottom gap above board center.
5. Board tilt from horizontal, with heading prescribed toward the bucket.

A deterministic farthest-first order starts near the menu's geometric centroid,
then selects the remaining geometry farthest from the already selected set.
Every one of the table's 24 qualified configurations appears before repetition.
The final six cycles repeat the start of that coverage. Saved captures, rewards,
policy weights and action names do not choose the order. These are nearby
qualified operating regions, not exhaustive coverage of the robot workspace.
New arm movements or successful captures at every table are measured outcomes,
not assumptions of the plan.

## Recording, scoring and editing

Each cycle captures 36 measured setup frames during actual joint-driven transit.
Their timing compresses joint travel; the setup portion is not a real-time
controller execution recording. Release frames are rendered every eight native
physics steps at `1/240 s`, giving 30 frames per simulated second before the
final declared playback adjustment.

The original complete-shot evaluator is unchanged: a board bounce, whole-ball
entry and one continuous second retained inside the bucket are required.
Partial exits reset retention. Rim-assisted captures count. Detected external
contacts disqualify the intended board-to-bucket route. The full native scoring
horizon is retained even when the visual edit removes a quiet retention tail.
No optimizer or actor update runs during this capture.

Evidence is saved outside Git:

| Artifact | Contents |
|---|---|
| `plan.json` | Geometry-only cycle schedule and source bindings; its canonical hash is bound by the result and clips |
| `clip-XX.json` | All 35 actions and outcomes, frame map, full-rate foreground telemetry and hashes |
| `physics-XX.npz` | Full-rate ball position/velocity/contact forces, both arms' joints/velocities and board pose for all 35 cells |
| `frames/00000.jpg` onward | Original native camera frames in chronological order |
| `preview.jpg` | First actual setup render for early framing inspection |
| `progress.json`, `result.json` | Attempt counts, rejected releases, source code hashes and completion status |
| `probes.json`, `rim-rays.json` | Fresh native contact and isolation evidence |

`presentation_export.py` first audits every recorded cycle. It independently
reconstructs the 35 cells' outcomes from the full-rate native packet, checks
frame readbacks against ball/joint telemetry, verifies geometry and camera
identity, and compares the aggregate counts. Original source probes and fresh
recording probes have separate hashes.

The visual edit keeps complete chronological action chapters: arm setup, release,
bounce and the immediate foreground outcome. A rim touch alone is never a cut.
For a capture, the edit shows the final qualifying entry and roughly 0.2 seconds
of padding; the longer retention proof remains in telemetry. Misses retain
bucket interactions through subsequent table/floor contact, exit below the
workspace, or the complete unresolved horizon. Rejected attempts remain labeled.

The exporter selects the first complete chronological prefix long enough for
60 seconds, with at least ten foreground rebounds. It never skips a failed
chapter to select a successful one. Any omitted suffix remains audited in the
raw evidence. Board-contact points from at least two other cells must project inside the
camera frame in each selected chapter. This is a geometric visibility check;
it does not prove that every point is unoccluded in the rendered pixels. Selected
native frames also receive visual inspection. Both foreground arms must show
measured configuration changes across the selected sequence.

The output is exactly 1,800 frames at 30 fps. A uniform playback adjustment of
at most 1.10× may shorten the selected prefix to one minute. It selects distinct
original frames only: no looping, duplicate padding, interpolation or frozen
frames. Insufficient action footage fails export instead of being stretched.
The metadata declares both the playback multiplier and compressed setup timing.
The optional small text caption is the only non-simulation visual content.

## Reproduction in the configured Ubuntu container

Use the pinned [Isaac foundation environment](README.md), generated assets under
`/workspace/assets` and the isolated `sircus_isaac_4090` container. Freeze the
capture code first and retain that snapshot for export: recorded code hashes
must match when the evidence is audited. Do not edit a running source snapshot.
Do not start another scene while the shared scene lock is occupied, restart
existing services, change another GPU's workload or connect to physical hardware.

The example below runs from Ubuntu. Replace the source snapshot path and use a
fresh output name; these are container paths. The output parent must already
exist and have sufficient free storage. The recorder stops below 3 GiB free.

```bash
film_source=/workspace/studies/REPLACE_WITH_FROZEN_FILM_SOURCE/sircus/isaac
qualified_source=/workspace/output/expanded-training-20261102-v4
film_output=/workspace/output/REPLACE_WITH_NEW_PRESENTATION_RUN

docker exec -u 0 -e SIRCUS_ISAAC_SOURCE="$film_source" sircus_isaac_4090 \
  flock -n /workspace/scene.lock \
  "$film_source/python.sh" "$film_source/film_cylinder_farm.py" \
  --headless --enable_cameras --device cuda:0 --appearance silver \
  --presentation-source "$qualified_source" \
  --presentation-camera-json "$qualified_source/selected-camera.json" \
  --presentation-batches 30 --output "$film_output"
```

This uses the container's dedicated device, not host GPU-index guessing. A
separate one-cycle preview uses `--presentation-batches 1` and another fresh
output directory. Inspect its native `preview.jpg` before commissioning a
different camera; changing the selected camera requires a separately documented
binding rather than editing this recording's evidence.

After the complete capture, export on a CPU environment with NumPy, SciPy,
Pillow, `ffmpeg` and `ffprobe`, using the matching frozen source helpers:

```bash
python presentation_export.py NATIVE_RUN NEW_EXPORT --source QUALIFIED_SOURCE
```

Paths in this command must be accessible on that machine. Add `--no-caption`
for native imagery without text. Export refuses an existing destination or a
destination inside the raw run. It writes `farm-one-minute.mp4`, `film.json`,
`farm-poster.jpg` and `contact-sheet.jpg`; verifies H.264 dimensions, cadence,
chapter times, frame count and a complete decode; and preserves a source-frame
map and hashes. Keep raw RGB, NPZ telemetry and generated movies out of Git.

For checks without Isaac or a GPU, run from the repository root with the
project's Python environment:

```bash
python -m unittest discover -s sircus/isaac/tests -p 'test_presentation*.py' -v
python -m unittest discover -s sircus/isaac/tests -p test_action_training.py -v
python -m unittest discover -s sircus/isaac/tests -p 'test_expanded*.py' -v
git diff --check
```

The Mac verification environment is
`/Users/kadirumacbook/Documents/ChatGPT/hackathon/sircus/simulation/.venv/bin/python`.
Torch-only checkpoint checks require a compatible CPU Torch environment and may
be explicitly skipped where it is unavailable. Unit tests validate contracts
and evidence handling; the native render, fresh probes, full-rate rescoring and
visual review are separate completion checks.

## Final recording result

Pending. Record the completed native run and immutable source paths, source and
export commits, actual attempt/release/capture/rejection counts, retained
chronological cycles, playback multiplier, movie digest, contact-sheet review,
native verification results and final storage/runtime state here after capture
and export finish. No numerical result is claimed by this pending section.
