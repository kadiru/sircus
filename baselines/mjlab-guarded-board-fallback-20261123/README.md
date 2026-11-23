# Guarded board-fallback comparison

Twenty fresh bucket positions, one frozen choice and one native drop per
planner/position at 64 substeps. Both planners captured at the same **15/20**;
each also captured **3/3** separate repeats at the previously regressed target.
The guarded fallback activated once, changed no shots and increased mean
planning request time from **12.936 to 15.052 seconds**. Keep the current planner.

`summary.json` contains sanitized aggregates, all 20 paired outcomes, the
separate diagnostic and source/receipt hashes. It contains no raw trajectories,
host calibration, credentials or model checkpoints. Its 471 computational
source hashes were checked against Git producer
`f8206420f23bd7f3fb498b683538284d7fb1fb53`.

All 64 releases completed (40 fresh, six diagnostic, 18 controls); both control
blocks passed. The full independent audit passed 71,175 checks. Native GPU
lease usage was 490.827 seconds on the 4090. The 4070 and Isaac were unchanged.

See [protocol, interpretation and reproduction](../../sircus/mjlab/GUARDED_BOARD_FALLBACK_STUDY.md).
This is simulation with arms held at checked endpoints, not driven-arm or
physical validation, camera perception, or new policy training. The sample is
stratified, not a uniform table-wide reliability estimate. Historical results
and the live planner are unchanged.
