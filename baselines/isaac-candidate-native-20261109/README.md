# Frozen seed 22 + candidate bank: native TRAIN result

Both cohorts completed and passed the separate independent audit: **70 new
requests, zero optimizer updates**, on the RTX4090. Each method requested the
same 35 original TRAIN bucket positions once, parent first and bank second.
The original six-run benchmark, seed 22 checkpoint, bank and 18797 reader stayed
unchanged. [Method and guards](../../sircus/isaac/CANDIDATE_NATIVE.md).

| Measured over all 35 requests | Original seed 22 | Candidate selector |
|---|---:|---:|
| Captures | 13/35 (37.1%) | 16/35 (45.7%) |
| Clean captures | 4/35 (11.4%) | 5/35 (14.3%) |
| Rejected before release | 7/35 (20.0%) | 0/35 (0.0%) |
| Released | 28/35 | 35/35 |
| Native wall | 55.136 s | 67.472 s |
| Process CPU | 123.442 s | 148.661 s |
| Driven setup wall | 44.295 s | 56.152 s |
| Scoring interval wall | 4.870 s | 5.116 s |
| Process lifetime peak RSS | 6,069.73 MiB | 6,141.05 MiB |
| Sampled occupied GPU memory | 6,189.44 MiB | 6,189.44 MiB |

The bank preserved 28 exact parent actions and their native IK seed protocol.
Their observed capture/clean/rejection outcomes all agreed. Seven actions
changed; three gained a capture (one clean), four released without capture.
There were **three paired capture gains and zero losses**. The original seven
rejections were three unresolved Arm14 IK, two unresolved Arm10 IK, and two
necessary outer-link-length bound failures. Numerical IK failure is unresolved
feasibility, not a proof of impossibility.

Native wall includes canonical home, geometry/setup, flight and serialization;
the setup/scoring intervals are included, not additional costs. Shared native
initialization/control probes cost 14.543 s. Current CPU bank loading/verification
was 0.846 s and selection/verification 7.265 s. Inherited bank geometry preparation
was 45.700 s; the prior 117.003 s preparation-plus-audit timing is inclusive. The
new independent CPU native audit took 4.804 s. Process CPU can exceed wall because
it sums CPU threads. RSS is a process lifetime high-water, not isolated phase
memory. Torch peaks were 0.279 MiB allocated/2 MiB reserved; these exclude the
PhysX/renderer allocations included in the sampled whole-device occupancy.
These are measured resource times, not monetary prices.

This is a small **TRAIN-informed, single-seed** comparison. The bank includes
historical successful TRAIN actions; it does not measure unseen-position
improvement. Selecting the nearest learned preference is a changed finite
selection rule, not new PPO learning or an unchanged continuous policy law.
Geometry admission does not guarantee a useful bounce, capture or physical
safety. The controller remains scripted with original measured release guards.

## Evidence and operational record

- [Summary](summary.json): counts, costs, paired outcomes and immutable input
  identities; raw trial observations are excluded.
- [Exact independent audit](audit.json): 63 released requests rescored from 1,920
  full-rate 240 Hz time steps, plus source/action/geometry/startup/control checks.
- [Publication verification](verification.json): bank 4 files, original 13 metadata/
  source files and the frozen 18797 report retained their original hashes.
- Producer commit 5edcc89ed9c5f6f7b645fe9f070cd35722a92bf3; source archive SHA
  d510dbd443597911c408af3c8ecaf503a32dd0d9e617b6ca4c1aa83260802bb6.
- Parent checkpoint SHA 57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f.
- Signed comparison-plan SHA 701105db369716c17bb4f2d530dce27967db61bdf0f68be64a98dcdd9311838e.
- Native audit SHA c07b22ace0d8347a304868e191cab0d999531ceeb43196b2854dda6ed12366cc.

The first launcher snapshot failed before any native process, output directory
or PID receipt existed: its lightweight process helper imported NumPy. Its
source/logs remain preserved. A standard-library-only helper fix was tested
with Python `-S`, then archived in a new snapshot. The same 70 actions, choices
and seeds ran once in a new output directory. No partial or paid run repeated.

The finite native job exited and released the 4090. The 4070 and foreign workloads
were never signaled. Storage remained 38.69 GiB free, above the 6 GiB floor. Native
physics/trials, plans, logs and browser QA remain in ignored runtime storage.
The GET-only comparison viewer uses localhost 18802; the original benchmark
viewer on 18797 remains separate. No training or model promotion followed.
