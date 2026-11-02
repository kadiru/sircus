# Compact dark-floor Isaac replay — November 2, 2026

The native Isaac Sim recording has 35 tables and 70 articulated UF850 arms. Its full camera tour is 60 seconds at 1920 × 1080 and 30 fps. A 19-second edit emphasizes the drops, and an 11.87-second edit stays with one table through four different configurations. A 1.2-second version samples that station edit at 10× playback speed. [Capture, controller and reproduction](../../sircus/isaac/DARK_FARM_FILM.md)

On the configured Ubuntu host, the media are in `$HOME/sircus-isaac/output/dark-farm-20261102-v3-export` and the native run is `output/dark-farm-20261102-v3`. The source prefix remains in `output/dark-farm-20261102-v2` for resume verification.

The visible floor is matte black with 9 mm charcoal seams. Column spacing is 2500 mm and row spacing is 2100 mm, giving 700 mm and 850 mm table-edge gaps. Isolated floor colliders retain their original geometry. All 35 cells passed bucket, rim, table, floor and robot contact checks. Neighboring-cell collision isolation also passed.

## Recorded outcomes

The replay reuses 350 actions sampled during the original PPO study. Of these, 162 passed the new approach/settling checks and released the ball. All 162 rebounded off the board, and 28 achieved the board-to-bucket capture definition. This recording selected source batches for camera coverage, including repeats. These counts describe this replay and do not replace the original training results or establish performance on unseen bucket positions.

Seven focused shots show seven board rebounds, two qualified captures and three measured arrivals at the physical floor. The other two shots hit the bucket/table and missed. The floor positions are visible in the camera and agree with measured contact force at the modeled ground height. The first two focused captures were visually checked against the native RGB frames.

The repeated station is row 1, column 2. Its bucket remains at local XY (155.714, 250) mm. The board stays at a 400 mm horizontal radius about it. Heights are measured above tabletop Z=0, and tilt is measured from horizontal.

| Trial | Board XY, mm | Board H, mm | Ball-bottom release Z, mm | Drop gap, mm | Tilt | Result |
|---|---|---:|---:|---:|---:|---|
| Short gap | 481.262, 17.581 | 320 | 465 | 145 | 24° | Bucket/table miss |
| Wide bearing | 540.673, 358.660 | 320 | 560 | 240 | 29° | Floor |
| Raised board | 464.276, −4.538 | 380 | 645 | 265 | 26° | Clean capture |
| Side release | 554.621, 220.444 | 320 | 560 | 240 | 29° | Floor |

Measured joint excursions in these four motion sections span 21.6–44.6° on the dropper and 28.0–37.5° on the board arm. Motion uses live joint drives with a cubic target ramp and URDF effort limits during transit. Original hold properties are restored before settling and release. Motion time is compressed, and waiting for settlement is omitted. Ball loading is initialized at the release location. This is a replay of recorded shot actions with scripted transitions; it does not validate grasping, collision-free path planning or physical deployment.

The 175 mm bucket measurement retains its reported-height status and unresolved overall-height/interior-depth interpretation. The same initial straight-sided taper and provisional 183 mm modeled rim are used.

## Verification and provenance

- [film.json](film.json) contains chapter timing, focused outcomes, native frame intervals and all four MP4 SHA-256 digests. Each H.264 file decoded at its declared frame count and cadence, and its digest matched after transfer to the Mac.
- [result.json](result.json) records the source scene fingerprint, policy provenance, current stage, capture-source manifest and camera plan. The source study remains [the original cylinder farm](../isaac-cylinder-film-20261101/README.md).
- [probes.json](probes.json) records the new contact and isolation checks.
- [resume.json](resume.json) identifies the seven completed chapters reused byte-for-byte at a camera cut after a neighboring table's transition failed. The remaining front-row shot uses column 4. The exporter verified the copied RGB and records against their original output and checksums, then checked the fresh chapters independently.

The new code passed the local regression suite, including 128 Isaac tests with one Torch-dependent skip on Mac. The capture and export passed their GPU/PhysX and video checks. The run used the existing 4090 environment. Native frames, video files and full joint/camera logs are stored outside Git.
