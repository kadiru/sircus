# Five remaining 30° recipes — November 5, 2026

The [fixed verification protocol](../../sircus/mjlab/TILT650.md) completed all
54 requests as native releases, with zero geometry rejects or invalid results.
No recipe passed the fresh capture gate or the clean gate at both resolutions.
The independent CPU audit passed all **14,410 checks**. Historical results,
materials, scoring and provisional bucket geometry remain unchanged.

| Phase | Requests/native | Captures | Clean | Rim-assisted |
|---|---:|---:|---:|---:|
| Initial controls | 12/12 | 6 | 6 | 0 |
| Five recipes × three repeats × 64/128 | 30/30 | 15 | 0 | 15 |
| Final controls | 12/12 | 6 | 6 | 0 |
| Total | 54/54 | 27 | 12 | 15 |

All 12 clean captures belong to positive controls. Candidate results are
15/15 rim-assisted captures at 64 and 0/15 at 128, with no whole-ball entry at
128. Both before/after control gates passed. Three identical repeats per
resolution are nominal replay checks, not a noise success-rate estimate.

## Exact selected targets

Board center height is 380 mm and tilt is 30° from horizontal for all five.
The same XY translation is applied to board and bucket, preserving the 650 mm
vector and bearing. Gap means ball-bottom clearance above the board-center
plane; release height is the ball center above tabletop.

| Parent grid suffix | ΔX mm | ΔY mm | Gap mm | Release H mm | Capture64 | Capture128 | Paired350ms drift mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0043 | -10 | 60 | 410 | 827.5 | 3/3 rim | 0/3 | 1.577 |
| 0070 | -10 | 80 | 410 | 827.5 | 3/3 rim | 0/3 | 1.572 |
| 0096 | 0 | 40 | 390 | 807.5 | 3/3 rim | 0/3 | 1.699 |
| 0151 | 0 | 80 | 410 | 827.5 | 3/3 rim | 0/3 | 1.582 |
| 0177 | 10 | 40 | 390 | 807.5 | 3/3 rim | 0/3 | 1.674 |

All five pass physics/oracle validity and the 5 mm early-position-drift bound,
but fail cross-resolution capture agreement. These were the exact five formerly
unverified parent captures; they are now verified failures of this nominal gate,
not a proof that every possible 650 mm shot is impossible.

## Posthoc contact diagnosis

The separately labeled [scalar contact analysis](contact-diagnostic.json)
contains its self-contained CPU analysis code, code hashes, source record/cache
hashes and scalar diagnostics. It does not add releases, change capture labels
or enter the qualification gate. All five matched first-repeat pairs have
bitwise-identical native initial joints.

- The 64-step shots first touch the bucket's near upper rim: their ball centers
  are 112–114 mm short of bucket center, about 5 mm sideways, and ball bottoms
  are about 2–3 mm below the rim at the saved contact-onset state.
- With 410 mm gap, the outgoing forward/up speeds are about 1.794/0.772 m/s at
  64 versus 1.809/0.794 m/s at 128. At 20 ms after the first rim contact,
  forward speeds differ by about 0.94 m/s. Later descending ball-center
  rim-plane crossings are 346–351 mm long at 128; the 64 crossings are inside.
- With 390 mm gap, forward/up speeds are about 1.773/0.788 versus 1.759/0.769 m/s.
  At 128 the ball reverses after the near-rim contact and crosses 371–375 mm
  short. The 64 counterparts enter the bucket.

These observations locate contact/timestep sensitivity under the same commands;
they do not prove a complete physical cause or real-world material calibration.
Post-rim crossing is descriptive only and is never substituted for the original
first-flight ranking, whole-ball entry or one-second retention criterion.

## Exploration boundary and provenance

Five recent calibrated 650 mm studies now total **1,312 requests / 663 native
releases**, including 120 controls and 150 verification requests. Discovery
remains 1,042 requested / 393 native releases, 32 rim captures and zero clean;
the new study adds verification, not newly discovered configurations.
The four earlier worker elapsed times total 48.6 minutes, and this worker took
447.3 seconds. These include setup/scoring/cache work and exclude queueing,
software development and independent audits. A foreign 4090 workload appeared
after Sircus acquired its cooperative lock, so timing was not isolated.

Stop repeated fine height/tilt/gap grids. The recommended next experiment is a
new, prespecified impact-point/clearance study, capped at one worker-hour and
at most 200 requested releases including controls and verification. That budget
is a recommendation, not a launched job or hidden extension to this protocol.
After it, either preserve timestep-stable recipes for learning or explicitly
reconsider geometry/contact modeling. Noise and further PPO remain deferred.

- Frozen producer/checker/UI source: `8f5ab77954efd1f1487958039b87e63ec1582207`.
- Contract: `b6a56fc0e1ca045ce7fb5067d4b5e74502404780dee433162b740f6342ab8fd2`.
- Parent: `7db7440c12005c39808ee4dfbf831d8b81000cda048a8d4eddc896d9b1d31137`.
- All 24 evidence artifacts, 140 producer source files, 26 imported audit
  dependencies, ten collision assets and exact parent/proof bytes were checked.
- The read-only viewer was restored after reboot and reused for publication.
  No 4070 workload or Isaac service was started, stopped or reconfigured.
  The finite native/audit workers have finished. No policy was activated.

`summary.json`, all 54 rows of `requested-shots.csv`, and checksums retain
configured geometry, scalar outcomes and provenance. Raw trajectories, joints,
observed poses, logs, credentials and host paths remain outside Git. This is
privileged-geometry held-endpoint verification, not new RL, generalization,
grasping, collision-free transit, actuator tracking, perception or deployment.
The reported 175 mm bucket-height interpretation remains unresolved.

[Completed results and actual saved traces](http://127.0.0.1:18781/#tilt650Study).
