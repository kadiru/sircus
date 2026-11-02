# Ten silver-gray camera previews — November 2, 2026

Run `camera-options-20261102-v3` rendered ten actual 1920 × 1080 Isaac RTX stills on the 4090. The same frozen 35-table / 70-arm scene appears from ten camera positions, with the demonstrated workcell placed at the right-front corner. No new rollout, policy update or video was produced in this preview task.

Whole source workcells `r1c2` and `r1c7` exchange display origins. Their arms, tools, bucket, board and ball stay together; all 35 distinct target assignments are preserved. Source batch 6 provides the saved poses. This framing arrangement does not establish success at the old `r1c7` target, whose saved source actions failed dropper endpoint IK.

- [camera-options.json](camera-options.json) records camera positions/targets, lens, appearance, source identity, origin permutation, frozen-state digest, native readback, four executed-source hashes and ten RGB digests.
- [verification.json](verification.json) records transfer/source matching, static-state and visual checks, gallery hashes and local tests. All ten images decoded, matched their digests on the Mac and retained the same physics step/state. All four source files match the executed snapshot. All ten final native images were inspected.
- The selected silver lighting remains unchanged. The RTX background source uses neutral gray for rays above the floor. Early v1/v2 background experiments remain local and are not the selected images.

141 Isaac regression tests passed, with one Torch-dependent skip on the Mac. Four new pure checks cover whole-cell permutation, frustum coverage, deterministic distinct poses and independence from Isaac imports. Gallery fixture layout and keyboard navigation checks passed. The finite capture exited; the 4070 workload and existing services were untouched. No physical, Viam or audio operations occurred.

The immutable execution source is `$HOME/sircus-isaac/studies/camera-options-20261102-v3`; native output is `output/camera-options-20261102-v3` on the configured Ubuntu host. The Mac comparison gallery is retained under the ignored runtime media area. Images, gallery files and full logs remain outside Git; this directory contains compact synthetic metadata only. [Commands, choices and next video brief](../../sircus/isaac/CAMERA_OPTIONS.md).
