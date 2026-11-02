# Silver-gray camera choices

Ten native Isaac Sim stills compare fixed camera positions at the right-front corner of the 35-table farm. The selected silver-gray surfaces, broad studio fill and subtle floor shadows are preserved. A neutral gray RTX background replaces exposed black sky in low views; scene lights are unchanged. This uses NVIDIA's [background source override](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html).

The featured source workcell is the previously successful `r1c2` case. Its entire origin is exchanged with source `r1c7`, placing the demonstrated arms, board, bucket and ball together at the right-front display corner. All 35 distinct target assignments remain in the farm. The old rightmost case has unreachable dropper endpoints in the saved ten batches, so this arrangement provides a valid demonstration setup without suggesting that its original target was solved.

All ten renders use the same frozen articulation/object state, 23 mm lens and 1920 × 1080 resolution. Neither the ball nor a policy steers the camera. Heights in the comparison captions are above tabletop Z=0. These are composition previews; no new shots, training updates or capture-rate evaluation were run.

| Choice | Camera view | Height above tabletop |
|---|---|---:|
| 01 | Original perspective shifted to right-front corner | 1.85 m |
| 02 | Close diagonal | 1.45 m |
| 03 | Low diagonal | 1.15 m |
| 04 | Balanced diagonal | 1.65 m |
| 05 | Inward diagonal | 1.60 m |
| 06 | Across the front row | 1.35 m |
| 07 | Along the right aisle | 1.40 m |
| 08 | Elevated close diagonal | 2.40 m |
| 09 | Wide diagonal | 2.20 m |
| 10 | Low wide diagonal | 1.50 m |

The gallery supplies numbered comparisons, full-resolution images, and left/right keyboard navigation. Geometric projection checks do not establish pixel visibility; all ten native final images were inspected. [Saved parameters and verification](../../baselines/isaac-camera-options-20261102/README.md).

## Reproduction

Use the existing isolated 4090 environment, pinned Isaac Sim 5.1 / Isaac Lab 2.3.2 installation and assets. Set `SIRCUS_ISAAC_SOURCE` to an immutable source snapshot, then inside its container:

```bash
python.sh film_cylinder_farm.py --headless --enable_cameras --device cuda:0 \
  --dark-replay /workspace/output/cylinder-film-35-v5 \
  --appearance silver --camera-previews \
  --output /workspace/output/camera-options-new
```

With Pillow installed, build the comparison sheets and static gallery:

```bash
python3 camera_gallery.py output/camera-options-new
```

This finite command does not start a server or update an existing service. `camera-options.json` records exact poses, readback, frozen-state hash, source hashes, whole-cell permutation and image digests.

## Next recording

View 06 was selected, with a requested revision: shift 0.30 m to camera-right and set the eye to world Z=2.0 m, retaining the original aim. Right is the original view's horizontal screen-right direction, not world X. The new world eye is (11.278317, −6.458747, 2.0) m, aimed at (7.65, −4.15, 0.18) m. Original ten-view definitions remain unchanged; the adjustment is explicit and opt-in.

To render only this revision, add `--camera-option 6 --camera-right-m 0.30 --camera-z-m 2.0` to the preview command. This writes a single `camera-06.jpg` and its manifest; the ten-choice gallery generator is for the full set.

After review of that revised still, prepare a separate one-minute silver-gray film: circle the 35-table farm, then settle into the selected front-row view for several measured drops and visible arm configuration changes. Other tables should remain active in the background. The close view stays fixed in world coordinates and never follows a ball. Current preview code has not recorded or validated this future sequence; the existing silver station videos and historical dark tour remain separate.
