# Film a 35-table Isaac training run

[Completed run, checkpoint and verified outcomes](../../baselines/isaac-cylinder-film-20261101/README.md)

Open `sircus-front-row.mp4` for two close-up drops at different front-row tables, with the camera looking back across the farm. The 9.6-second cut uses the same frames as **22.5–32.1 s** in `sircus-drone-farm.mp4`. The full 43.5-second film starts with one 360° orbit, descends into an aisle, then cuts from the two front tables to a bird’s-eye view of the farm and an overhead view of a single bounce attempt.

One perspective RTX camera renders 1920 × 1080 frames during the Isaac Sim rollout. Each chapter follows one simultaneous batch of throws. The front-row cameras choose two different tables with settled release poses **before** the balls drop. PhysX steps at 240 Hz. The 30 fps film samples every second physics step for the close-ups (quarter speed), and every fourth step for the wide shots (half speed). [Camera path and chapter timings](farm_flight.py)

## What the arms learn

Each of the 35 tables has two articulated UF850 arms, an attached board, one ball and one fixed bucket. The five rows of seven tables correspond to five rows of seven tiles on a single 1800 × 1250 mm tabletop. Each table uses its own tile’s bucket location in local coordinates. The row nearest the camera uses the tiles at the upper Y edge of the tabletop. Tile centers may shift within the tile for bucket footprint and base clearance. In the farm, neighboring table origins are 4.5 m apart.

A shared PPO policy receives the known bucket XY, its distance to a common reference point at (670, −100) mm, and the sine and cosine of that bearing. It proposes four values per attempt.

| Choice | Range | Meaning |
|---|---|---|
| Board azimuth | −180° to +180° | Offset around a 400 mm radius about the bucket, measured from the bucket-to-reference direction. |
| Board height | 105–445 mm | Board-center height above the table. |
| Drop gap | 100–425 mm | Ball-bottom height above the board center. Ball and board center share XY. |
| Tilt | 10–42° | Board inclination from horizontal. The controller aims the board toward the bucket. |

The model uses a 183 mm rim height from 175 mm assumed interior depth plus an 8 mm bottom. The physical measurement is still a **reported 175 mm height**. A board center can therefore settle below the modeled rim, and the ball can leave it on an upward trajectory. A scored capture requires a board rebound followed by whole-ball entry and one uninterrupted second in the bucket, with no detected intervening table, floor, robot or unidentified contact. Rim-assisted captures count.

Before training, each cell screens 15 bearings and four (board height, gap, tilt) combinations using both arms’ inverse kinematics and settling measurements. The four combinations in (mm, mm, degrees) are (380, 265, 26), (320, 240, 29), (320, 145, 24) and (150, 235, 22). These 60 checks use no bounce or capture outcomes. Each sampled policy action selects the nearest accepted option in the policy’s four latent coordinates. A cell with an empty menu tries the policy proposal directly. The arms initialize at the selected endpoints and must settle again before release. Rejections count as failed attempts.

The run schedules 10 batches of 35 attempts and five PPO updates. The learned decision is the shot proposal given simulator geometry. Grasping, joint trajectories, physical transfer and generalization to unseen bucket positions need separate tests. [Shot geometry](cylinder_farm.py) · [Training and capture](film_cylinder_farm.py)

## Reproduce on the configured Ubuntu simulator

Use the existing 4090-only container, generated UF850 assets at `/workspace/assets`, and its mounted Isaac Sim 5.1.0 binary with Isaac Lab 2.3.2. The container’s `python.sh` selects Python 3.11 and the bundled PyTorch 2.7.0+cu128. The filming script requires at least 3 GiB free at startup. [Pinned runtime, dependencies and asset setup](README.md)

From the Ubuntu host, in a Git checkout **at the film commit**, create a fresh source snapshot. The host workspace is mounted as `/workspace` inside the existing container.

```bash
set -euo pipefail
film_id="cylinder-film-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
film_root="$HOME/sircus-isaac"
film_source="$film_root/studies/$film_id"
film_run="$film_root/output/$film_id"
mkdir "$film_source"
git archive HEAD:sircus/isaac | tar -x -C "$film_source"

docker exec -u 0 -e "SIRCUS_ISAAC_SOURCE=/workspace/studies/$film_id" \
  sircus_isaac_4090 timeout 2400 flock -n /workspace/scene.lock \
  "/workspace/studies/$film_id/python.sh" \
  "/workspace/studies/$film_id/film_cylinder_farm.py" \
  --headless --enable_cameras --device cuda:0 --output "/workspace/output/$film_id"
```

The run checks contact attribution and isolation before filming. `rim-rays.json` contains the first contact on each candidate rim ray. `probes.json` records the completed preflight. `menu.json` records endpoint rejections and the resulting per-cell menus. Each `clip-*.json` contains release actions, outcomes, and per-frame camera poses. `result.json` and `cylinder-policy.pt` are completion artifacts.

Export from the Ubuntu host using Python with NumPy and Pillow, and host FFmpeg with libx264 and ffprobe on PATH. Keep the shell variables from the capture step.

```bash
python3 "$film_source/export_cylinder_film.py" "$film_run" "$film_run-export"
```

Open the MP4s in `$film_run-export`. Its `film.json` reports training counts, low-board attempts, chapter timings, focused table IDs and capture outcomes. The MP4s and raw frames live outside Git. The exporter verifies the frame sequence and its association with the training batches before encoding. [Export script](export_cylinder_film.py)
