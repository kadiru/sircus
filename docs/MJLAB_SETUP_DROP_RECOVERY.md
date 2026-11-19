# Drop tests at the five previously blocked bucket positions

The earlier comparison could find no admissible shot at five bucket
positions. A wider CPU search certified 43 of 160 proposed arm poses and
approach paths at those same positions. [Pose search results](MJLAB_FLIGHT_RANKING.md).

The drop study freezes all 43 archived joint pairs, board orientations,
release poses, bucket positions, and path proofs. Each command gets one
first-pass drop at 64 physics substeps. Every first-pass capture gets
three fresh drops with the same command. A shot is confirmed when all three
repeats capture. Capture uses the existing score, which requires a board
rebound and one continuous second with the whole ball inside the bucket.
Table and rim assistance are allowed. A ball that touches an arm or the
floor during the scored route earns no capture.

The run checks the same nine native controls before and after the test.
The CPU audit verifies all request and proof hashes, rescores the saved
trajectories, and reconstructs which shots qualified for repeats.
The resulting coverage applies to these five development positions.

## Results at 64 substeps

| Bucket X, Y (mm) | First-pass captures | Repeat captures | Confirmed shots |
|---|---:|---:|---:|
| 603.8, −168.2 | 7/9 | 21/21 | 7 |
| 725.5, −88.2 | 7/12 | 21/21 | 7 |
| 1028.9, −19.0 | 0/4 | — | 0 |
| 775.2, −27.8 | 11/13 | 33/33 | 11 |
| 618.4, 138.6 | 2/5 | 6/6 | 2 |
| Total | 27/43 | 81/81 | 27 |

Four positions have repeatable captures. All 27 confirmed shots went
from the board to the bucket without a table bounce. At (1028.9, −19.0)
mm, each of the four shots hit a robot after bouncing off the board.
None entered the bucket fully.

Both sets of nine controls passed on the 4090. The independent CPU audit
passed 23,673 checks. The Mac repository test suite passed and the pinned
Ubuntu mjlab suite ran 2,287 tests with seven skipped and zero failures.

- Native producer commit `2d65706`
- Native report SHA-256 `4bfbe28870c0abcbbe854b1a11abdfc215415bc30e8677a50aceceb1874d37aa`
- Repeat plan SHA-256 `554ec252186a3da0bd8cdceda3d45440ad9a277fad45d7f9bf6453a6e05d3f0b`
- Evidence manifest SHA-256 `aa5dc5c9950d7b9cf5c244988682eace6165716eeb97f9d3d4d2fdf8968fc409`
- CPU audit SHA-256 `64de6a57c730c40249875a4245db3b52549f5071974b3abdb39b8b461965974c`

## Reproduction

Use the pinned mjlab environment and the audited `fixed35-sixhour-v2`,
`intergrid-route-fallback-v1`, `intergrid-setup-recovery-v1`, and
`corrected-grid-v1` archives. Run
`python -m sircus.mjlab.intergrid_setup_drop_study --help` for the two audit
receipts, runtime reference, assets, lease, and fresh output directory
arguments. Expose only the 4090 using its UUID and use the existing
shared lease.

After both control batches pass, run
`python -m sircus.mjlab.intergrid_setup_drop_audit --help` with CUDA hidden.
The audit uses the same parent, fallback, setup, assets, and output
arguments, plus a fresh sibling path for the audit receipt. Raw observations
and simulator configuration remain in the Ubuntu archives.
