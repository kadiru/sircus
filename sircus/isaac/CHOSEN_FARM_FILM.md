# Farm film with the selected camera

The one-minute native Isaac recording uses the camera exported by the user from the interactive scout on November 2, 2026. The original JSON is preserved byte-for-byte in [the compact baseline](../../baselines/isaac-chosen-farm-20261102/README.md). Final images come from Isaac Sim 5.1 / Isaac Lab 2.3.2 on RTX 4090, using the selected silver-gray surfaces, medium-gray floor and soft neutral lighting.

## Camera and sequence

The camera completes a full farm orbit in 14 seconds, approaches the selected view over two seconds, then stays fixed for 44 seconds. Eight foreground drops use four previously sampled configurations twice, with measured arm configuration changes and concurrent background releases. There is no ball-following, generated replacement imagery or added ball trail.

| Saved parameter | Value |
|---|---|
| Eye, world metres | (10.9706, −6.1943, 2.5406) |
| Aim, world metres | (7.8307, −3.4877, 0.3904) |
| Up | Z |
| Focal length / horizontal / vertical aperture | 23 / 36 / 20.25 mm |
| Output | 1920 × 1080, 30 fps |
| Full movie | `farm-one-minute.mp4`, 1800 frames, 60 s |
| Fixed front-row movie | `front-row.mp4`, source frames 480–1799, 44 s |

The whole source cases `r1c2` and `r1c7` exchange display origins, as in the scout. The demonstrated case occupies the right-front slot with its arms, board, bucket and ball together. All 35 distinct table-local bucket assignments remain present. This does not establish feasibility of the old rightmost case, whose saved dropper endpoints failed IK.

## Motion, timing and results

The scene contains 35 tables and 70 articulated UF850 arms at 2.50 × 2.10 m spacing. The film replays selected PPO actions, with scripted joint transitions and ball initialization at release. There are no optimizer updates, learned grasps or newly validated arm paths in this recording. Motion frames sample measured joint movement; settling waits are omitted. The release phase samples every six 240 Hz physics steps during the orbit (0.75× physical speed at 30 fps) and every four during the fixed view (0.5× physical speed). There is no new 10× cut; the earlier twelve-drop silver station export retains its original timing and checks.

Across the selected/repeated farm batches: 420 attempted actions, 198 accepted releases and 29 retained captures. These are filming outcomes, not a held-out success benchmark. All eight featured releases bounced off the board: two captured, four reached the physical floor, and two missed after bucket/table contact. Each fixed-view chapter has at least 15 background rebounds projected into the camera. Projection alone does not establish pixel visibility; native release/contact/outcome samples were also inspected.

The original whole-ball entry and continuous one-second retention evaluator remains in use. Bucket geometry and previous results are unchanged: inner top/bottom diameters 177/150 mm, outer 200/158 mm. The reported height is 175 mm; its overall-height/interior-depth meaning remains unconfirmed. The provisional model retains its 183 mm rim including an 8 mm base.

## Reproduction

Use the existing isolated 4090 container and generated assets from [the Isaac setup](README.md), plus the original completed `cylinder-film-35-v5` source run. Freeze this source directory before capture. The saved camera loader rejects incompatible scene fingerprints and camera schemas rather than silently substituting a view.

Inside the configured container, using fresh output directories:

```bash
python.sh film_cylinder_farm.py --headless --enable_cameras --device cuda:0 \
  --dark-replay /workspace/output/cylinder-film-35-v5 \
  --appearance silver --recording-plan chosen-farm-v1 \
  --camera-json /workspace/selected-camera.json \
  --output /workspace/output/chosen-farm-new
```

To check only the selected still first, replace `--recording-plan chosen-farm-v1` with `--camera-previews --camera-option 6`. Keep the other flags. The camera JSON takes precedence over the numbered preview pose; its full-precision values and exact-byte digest are recorded.

On the Ubuntu host with NumPy, Pillow, FFmpeg and ffprobe:

```bash
python3 dark_export.py output/chosen-farm-new output/chosen-farm-new-export \
  --source output/cylinder-film-35-v5
```

The exporter checks source action lineage, original camera bytes, every camera frame, uninterrupted frame numbering, native bucket/rim/table/floor/robot contacts, foreign-cell collision isolation, measured foreground joint travel and simultaneous background rebounds. It requires all eight foreground board contacts, a qualified capture and a real floor arrival. Both H.264 files are decoded to verify frame counts, resolution, fps and chapter timing, and their digests are recorded in `film.json`.

The completed immutable capture/export source is `studies/chosen-camera-20261102-v1`; raw output and exports are in corresponding `output/chosen-camera-20261102-v1` and `output/chosen-camera-20261102-v1-export` directories under the configured Ubuntu workspace. Videos, raw frames and detailed joint/camera logs remain outside Git. The finite capture exited; the existing viewer, services and 4070 workload were preserved. No physical, Viam or audio actions were performed.
