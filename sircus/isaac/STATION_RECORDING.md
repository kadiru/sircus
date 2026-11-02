# Silver-gray station recording

The selected appearance uses silver-gray tabletops over a neutral medium-gray floor. The silver-soft-v1 setup raises the neutral dome fill to 1050, lowers the broad key to 320 and adds two broad fills at 200 and 160. Ambient occlusion is disabled. This gives a slightly brighter scene with softer shadows beneath the slabs and weaker shadows on top.

Hold a single world-space camera view for the whole repeated-drop sequence. Ball positions, bounces, joint configurations and shot numbers do not steer the camera. The view matches the perspective used for the appearance previews and includes the foreground board, gripper, bucket, nearby floor and tables behind it.

The plan schedules twelve separately simulated foreground releases at row 1, column 2. Four previously recorded shot configurations repeat three times. Other tables execute the same concurrent batches. The exporter requires at least ten distinct measured foreground drop/rebound events and at least two active projected background drops in each shot before publishing the station videos. Rejected approaches and copied video frames cannot satisfy that count. All twelve planned chapters currently have to complete. Recording can resume at a verified camera cut after a failed approach.

Use `--appearance silver --recording-plan fixed-station-v1` with `film_cylinder_farm.py --dark-replay ...`. The fixed plan is the default for new replay commands. Old records without plan metadata continue to use `dark-tour-v1` during verification, and reproducing the historical tour requires that flag explicitly.

Each chapter records 36 motion frames and 480 release/landing frames. Physics advances at 240 Hz, and release frames sample every two physics steps. Playing those frames at 30 fps shows the release phase at quarter physical speed. The twelve untrimmed chapters therefore total 206.4 seconds at 30 fps. The 10× file samples that complete recording at every tenth source frame, giving approximately 20.7 seconds after frame-count rounding. Its speed reference is the recorded source. The release phase in that fast file is approximately 2.5× physical speed. The exporter also produces normal-speed station and impact edits for closer inspection.

The exporter audits the actual frame indices retained in the 10× file. At least ten trials must retain two descending-ball frames and a subsequent rebound frame before landing. Camera projection checks and recorded contacts support that audit; final pixel inspection also checks for occlusion.

The [completed silver recording](../../baselines/isaac-silver-station-20261102/README.md) passed the native contact, camera, visibility and video checks. All twelve foreground releases rebounded off the board; three captured and six reached the floor. Every trial retained the required descent/rebound samples in the 10× file, and native pixels were inspected. The readable edit is 57.33 seconds; the fast edit is 20.67 seconds. This is a replay of previously recorded PPO shot actions with scripted arm transitions and initialized ball loading.

Inside the existing pinned 4090 container, use an immutable source snapshot and a fresh output directory:

```bash
python.sh film_cylinder_farm.py --headless --enable_cameras --device cuda:0 \
  --dark-replay /workspace/output/cylinder-film-35-v5 \
  --appearance silver --recording-plan fixed-station-v1 \
  --output /workspace/output/silver-station-new
```

On the host, with NumPy, Pillow and FFmpeg/ffprobe available:

```bash
python3 dark_export.py output/silver-station-new output/silver-station-new-export \
  --source output/cylinder-film-35-v5
```

The exported filenames remain `dark-farm.mp4`, `dark-drops.mp4`, `station.mp4` and `station-10x.mp4` for compatibility. The source metadata identifies the silver appearance. Geometry, provisional measurements and older dark-tour results remain unchanged.
