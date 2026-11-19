# A confirmed historical table-assisted capture

One saved PPO training attempt did bounce **board → table → bucket/rim → inside
the bucket**, then remain fully inside for one continuous second. This is an
observed simulation trajectory, not proof that the final policy learned to
reproduce it reliably.

The shot is `ppo-20261411-train-00221`, from the calibrated benchmark with seed
`20261411` at **64 substeps**. Its settings were:

| Setting | Value |
|---|---:|
| Board center, X/Y/Z | 638.692 / −229.331 / 401.251 mm |
| Board tilt from horizontal | 34.614° |
| Configured drop gap | 103.138 mm |
| Ball-center release height | 541.889 mm |
| Bucket center, X/Y | 1050 / −650 mm |
| Horizontal board-to-bucket distance | 588.334 mm |

Times below are relative to ball release:

| Event | Time |
|---|---:|
| Board impact begins | 0.136979 s |
| Table impact begins | 0.452539 s |
| First bucket/rim contact | 0.654622 s |
| Whole-ball entry | 1.145638 s |
| One continuous second retained | 2.145638 s |

The table contact reversed vertical velocity from **−2.851 to +1.851 m/s**.
There was one board-contact run and one table-contact run before capture, with
bucket/rim assistance before entry. No robot, floor or unattributed contact
exceeded 0.01 N before capture.

Both existing calibrated scorers reproduced the saved result from the dense
native trace. Archive checksum, record identity and original scene fingerprint
were checked. Historical labels remain unchanged: `success=true` for raw
capture, `route_success=false`, `complete_success=false`,
`invalid_physics=false`, and reward **0.025282**. The existing
`sircus-calibrated-board-route-v2` definition excludes table contact; its reward
therefore did not give this trajectory the task-success bonus.

The metadata scan covered 102 files and 82,953 rows, removing 625 known cache
duplicates. It found 123 retained-ball/table-contact candidates explicitly
marked valid physics. Those are **candidates, not 123 verified routes**: this
one example received full contact-order and independent retention verification.
Older entries without cache identities are not claimed to be unique trials.
No Isaac evidence was scanned and no new GPU simulation was run.

The [sanitized evidence summary](../baselines/mjlab-clearance-horizon-20261108/table-assisted-example.json)
records source checksums, relative evidence paths, dimensions and timings.
Raw trajectories and host paths remain outside Git. Whether table-assisted
routes should earn task-success credit is a separate, undecided objective
change; this audit changes neither scoring nor the matched horizon experiment.
