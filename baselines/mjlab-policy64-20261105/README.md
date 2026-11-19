# Fixed-64 success bootstrap and PPO — November 5, 2026

The [prespecified study](../../sircus/mjlab/POLICY64.md) completed all **1,284 fresh
requested decisions**, including 1,024 TRAIN shots, 48 validation shots, 200 TEST
shots and 12 controls. The independent CPU audit passed **346,485 checks**.
Both fresh positive/native-miss control blocks passed. No 128 or noise tests
were added, and the earlier 128 failures retain their original labels.

## Frozen comparison on 50 new bucket positions

| Method | Requested | Native | Captures | Clean | Invalid |
|---|---:|---:|---:|---:|---:|
| Geometry prior | 50 | 48 | 19/50 | 2 | 2 |
| Nearest TRAIN success | 50 | 31 | 18/50 | 3 | 19 |
| Behavior cloning | 50 | 43 | 20/50 | 3 | 7 |
| Selected BC→PPO checkpoint | 50 | 39 | 18/50 | 4 | 11 |

PPO did not improve capture rate: 36% versus BC's 40% and the prior's 38%.
On paired layouts, both BC and PPO captured 13; PPO gained 5 captures but lost
7 BC captures. Of PPO's 32 failures, 21 were native flight misses, 5 finite-IK
rejections, 4 endpoint-clearance/pose rejections and 2 fixed-base-clearance
rejections. The extra clean capture does not establish better control from one
seed and one nominal shot per position. Diagnose these paired failures before
another training run; this policy is not ready for reliable all-table placement.

Each method made one nominal request at each of the same 50 fresh coordinates.
All rejected/invalid requests remain in the denominator. The target box is
X100–1240 mm, Y-650–250 mm, with at least 5 mm separation from all old benchmark
coordinates and the other new layouts. These are 50 evaluated positions, not
continuous all-table coverage or noise robustness. Method differences are
descriptive for this single seed/cohort; this is not an equal-acquisition-budget
algorithm contest.

Selection retained the checkpoint after 1024 fresh requested PPO shots; later or earlier checkpoints were not chosen using TEST outcomes.

## What was learned

The historical source contains 6,144 audited TRAIN requests / 4,380 native
releases / 2,439 captures / 842 clean captures across 35 layouts. One best
successful TRAIN teacher per 22 successful layouts fits the actor; all TRAIN
outcomes fit the critic. Historical VAL/TEST labels never train it. Source
acquisition is not free, and prior geometry acquisition cost is not fully known.

The nine-input actor predicts five continuous shot parameters through a bounded
residual: board X/Y/height, tilt and drop gap. Board yaw faces the bucket by the
existing geometric rule, and dropper XY is above board center. A shared
geometry-only source/IK/setup resolver runs before prediction. No ball-flight
search, retries or learning occur at TEST. This is shot-configuration learning
with prescribed held endpoints, not a joint-torque, grasp or driven-transit policy.

BC used 800 full-batch actor steps and 400 critic steps. PPO then completed 32
updates from 1,024 fresh requested shots. Fresh TRAIN observed
615/1024 captures, including 374 clean;
these training counts are not TEST performance.

| Fresh TRAIN prefix | VAL captures | Clean | Native | Invalid | Mean reward |
|---:|---:|---:|---:|---:|---:|
| 0 | 6/12 | 2 | 11 | 1 | 1.9858 |
| 256 | 7/12 | 3 | 11 | 1 | 2.3389 |
| 512 | 7/12 | 3 | 10 | 2 | 2.2965 |
| 1024 | 7/12 | 4 | 11 | 1 | 2.3774 |

Checkpoint selection used only 12 validation layouts: captures, mean calibrated
reward, then earliest prefix. All four methods' complete TEST predictions were
frozen before any TEST outcomes. Prior uses 17 historical geometric hints;
nearest/BC share the 6,144 historical TRAIN exposures; PPO adds 1,024 new exposures.

## Concrete saved configuration

One saved successful TEST example, policy64-test-14: bucket X/Y 315.0/214.4 mm; board X/Y/H 571.9/-97.7/319.4 mm; tilt 36.680°; nominal center-plane gap 368.8 mm; ball-center release H 725.7 mm; board-center-to-bucket horizontal distance 404.2 mm. This was a rim-assisted or other non-clean capture in one nominal trial, not a repeated reliability estimate.

All listed geometry is requested nominal simulation geometry. Heights are above
the table; gap is vertical ball-bottom clearance above the board center plane,
not the physical top face. Ball contact, whole entry and uninterrupted one-second
retention come from fresh native traces and independent rescoring. Clean capture
is a separate subset; rim-assisted captures remain valid captures.

## Reuse and evidence

- [Summary and portable inference contract](summary.json).
- [All 200 TEST requests and nominal settings](test-positions.csv).
- [BC checkpoint](bc-policy.pt), SHA256 `1b201dc48101078ed641b614b89115440aa9725cea567fb2327ba71b9679b51d`.
- [Selected checkpoint](selected-policy.pt), SHA256 `d42f683e0fd79dd0ee6e01c042abc9df9d49273622a7d50a1026733dea64dcd0`.
- [Checksums](checksums.json) and [complete publication checksums](SHA256SUMS).

Checkpoint files are exact audited copies. Load with `torch.load(...,
map_location="cpu", weights_only=True)`. The portable contract explains the
actor layers, normalization, state prefix and residual decoder; callers still
need the external geometry-only anchor provider. No raw prior joint bank or
host calibration is included.

- Frozen native source: `f00a29a42cac69221b0896173f9be785f113a8a6`.
- Frozen archive SHA256: `4c450aa30abd41bb0c8a6d89fbc181a56dbec0df61b3b44318ef15a5617d2e8c`.
- Contract: `d30ad91111cb8beb26438f2271357a103daf50d0399ac3646f3b7f163d87f02e`.
- Independent proof: `4dc317e0697e426a4708c3c997f8671d97b1f49de6bd4450c56e6487a6e93ac3`.
- Reported study runtime: 6625.3 s, excluding initial
  input verification and independent CPU audit; not isolated GPU-kernel time.

Historical labels came from older native reset/clock source. Its compatibility
guard remains intact; a new explicit reference bridge binds historical geometry
and assets to the newer audited impact runtime. Every new rollout used the
current pinned64 options and calibrated scorer/oracle. Raw source rows, joints,
poses, contacts, trajectories and logs remain outside Git. The bucket's reported
175 mm height interpretation remains unresolved, with the existing explicit
interior-depth approximation preserved.

The read-only [results and actual native flights](http://127.0.0.1:18781/#policy64Study)
are separate from the existing interactive solver; publishing this checkpoint
does not silently activate it for arbitrary interactive bucket queries. The
4070 and Isaac setup were not changed, and the existing viewer was not restarted.
