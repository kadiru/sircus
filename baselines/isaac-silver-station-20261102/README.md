# Silver-gray fixed-camera Isaac replay — November 2, 2026

The selected silver-gray tabletops use slightly brighter neutral fill and a softer key light over a medium-gray floor. Subtle shadows beneath the slabs show their height. The camera stays fixed through twelve foreground drops at row 1, column 2, with other tables dropping behind it. All images come from the articulated Isaac Sim 5.1 / Isaac Lab 2.3.2 scene on RTX 4090.

The 57.33-second readable edit removes settlement waits and long stationary tails. The 20.67-second 10× version samples the complete 206.4-second recording, retaining all twelve foreground falls and rebounds. The source release phase plays at quarter physical speed at 30 fps; its fast version is approximately 2.5× physical speed. This is replay of previously sampled PPO actions with scripted joint transitions and initialized ball loading. There were no optimizer updates in this recording.

## Outcomes and scope

Across the 35-table batches, 420 attempted actions produced 197 accepted releases and 23 qualified captures. These selected and repeated batches are filming evidence, not a new policy benchmark or held-out evaluation. All twelve foreground attempts rebounded off the board: three captured, six reached the physical floor, and three missed after bucket/table contact. Four configurations were each simulated three times, with the same observed result within each configuration.

The featured bucket stays at local XY (155.714, 250) mm, and the board center stays at a 400 mm horizontal radius about it. Heights are above tabletop Z=0; tilt is from horizontal.

| Configuration | Board XY, mm | Board H, mm | Ball-bottom release Z, mm | Gap, mm | Tilt | Three observed results |
|---|---|---:|---:|---:|---:|---|
| Short gap | 481.262, 17.581 | 320 | 465 | 145 | 24° | 3 bucket/table misses |
| Wide bearing | 540.673, 358.660 | 320 | 560 | 240 | 29° | 3 floor arrivals |
| Raised board | 464.276, −4.538 | 380 | 645 | 265 | 26° | 3 captures |
| Side release | 554.621, 220.444 | 320 | 560 | 240 | 29° | 3 floor arrivals |

The farm remains 7 columns × 5 rows, 35 tables and 70 arms, at 2.50 × 2.10 m spacing. Only appearance, camera plan and recording/controller timing changed. Original collision geometry, contact scoring and prior baselines remain intact. The reported bucket height remains 175 mm with unresolved overall-height/interior-depth interpretation; the provisional model retains its 183 mm rim.

The fixed-plan intermediate joint-ready tolerance is 0.001 rad, avoiding an unnecessary wait at the previous 0.00065 rad threshold. Final measured pose, speed and drift release guards remain in force. Paths, pickup and release initialization are scripted; physical deployment and learned transit are outside this film's validation.

## Verification and files

- [film.json](film.json): actual source-frame intervals, outcomes and four encoded MP4 digests. Each H.264 file decoded at its declared 1080p/30 fps frame count and chapter timing, and matched its digest after transfer to the Mac.
- [result.json](result.json): unchanged source scene/policy fingerprint, 77 capture-source hashes, selected materials/lighting and fixed camera plan.
- [probes.json](probes.json) and [rim-rays.json](rim-rays.json): all-cell bucket, rim, table, floor, robot and cross-cell isolation checks passed.
- [resume.json](resume.json): the first two completed trials were reused byte-for-byte from v1 after an intermediate wait was corrected. The final verifier checked the original RGB/clip hashes and fresh v2 probes. No partial trial was counted.
- [verification.json](verification.json): evidence digests, postprocessor hash, local test results and media checks. All twelve fast trials retained two descent samples and a subsequent rebound. At least seven background rebounds projected into the view in every chapter. Native retained frames for all twelve foreground drops were inspected for visibility.

The full mock/numerical regression suite passed, including 137 Isaac tests with one Torch-dependent skip on the Mac. The finite capture and exporter exited successfully; the 4070 workload and existing services were left running. No physical/Viam/audio actions were performed.

On the configured Ubuntu host, raw runs are under `$HOME/sircus-isaac/output/silver-station-20261102-v1` and `silver-station-20261102-v2`; encoded media are in `silver-station-20261102-v2-export`. The immutable capture source is `studies/silver-station-20261102-v2`, and the separately frozen exporter is `studies/silver-station-post-20261102`. Its only source difference is the native-bool serialization fix in `dark_export.py`, recorded by hash. Media and full joint/camera logs remain outside Git. [Launch and timing details](../../sircus/isaac/STATION_RECORDING.md).
