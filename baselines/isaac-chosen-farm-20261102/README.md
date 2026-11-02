# User-selected farm camera — November 2, 2026

Native Isaac Sim 5.1 / Isaac Lab 2.3.2 replay on RTX 4090: 35 tables, 70 arms, silver-gray stage. A 14-second orbit and 2-second approach lead to 44 seconds at the user's exact fixed camera. Eight foreground board rebounds include two retained captures, four floor arrivals and two bucket/table misses. Other tables remain active behind the featured table.

This is replay of selected training actions with scripted joint transitions and initialized balls, not new PPO training, held-out evaluation, learned pickup or physical validation. Across the selected/repeated batches: 420 attempted actions, 198 accepted releases, 29 retained captures. These counts must not be used as a generalization benchmark. Historical baselines and provisional geometry remain unchanged.

## Compact evidence

- [selected-camera.json](selected-camera.json): exact original exported bytes, SHA256 `a7967c072b8f949318b43a510235d503c81c21e05f433670a0ce92a7af385ede`.
- [result.json](result.json): source hashes, scene/policy lineage, camera, stage and native capture plan.
- [film.json](film.json): encoded movie digests, chapter timing, joint excursions and contact/outcome frame references. Every fixed chapter has at least 15 projected background rebounds.
- [probes.json](probes.json) and [rim-rays.json](rim-rays.json): bucket, rim, table, floor, robot and cross-cell isolation passed across all 35 cells.
- [verification.json](verification.json): compact test, visual/media and transfer checks. Both encoded movies decoded at 1080p/30 fps, with their declared frame/chapter counts; hashes matched after Mac transfer. Native release/contact/outcome samples from all eight foreground trials and three full-size orbit/front-row samples were inspected.

Movie outputs, kept outside Git:

| File | Duration | Frames | SHA256 |
|---|---:|---:|---|
| `farm-one-minute.mp4` | 60 s | 1800 | `e616c426e9ccf489eeae449694c7bdb5431381763473bacb852ce7e8808833c9` |
| `front-row.mp4` | 44 s | 1320 | `9b3641c7cc991dc085ab121262e51768fead2f23ec0198d4716759967c902f90` |

The source case `r1c2` is displayed at `r1c7`, exchanging the complete cell origins. No objects are detached from their source cases, and no target is relabeled solved. The selected camera is fixed independently of balls, actions and outcomes. [Full timing, scope and reproduction](../../sircus/isaac/CHOSEN_FARM_FILM.md).

The finite 4090 process exited successfully. Final read-only GPU check found the 4090 idle and the existing 4070 workload active. The camera-scout server and existing services were left running. No hardware, Viam or audio actions occurred. Raw run: `output/chosen-camera-20261102-v1`; export: `output/chosen-camera-20261102-v1-export`; immutable source: `studies/chosen-camera-20261102-v1`.
