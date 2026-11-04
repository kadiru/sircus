# Native impact-pose and short-miss development study

This is a separate **native MuJoCo Warp simulation** diagnostic, not a new RL
policy or held-out benchmark. The completed calibrated-v2 benchmark, selected
policies, contact calibration/scorer and old validation audit remain frozen.
Bucket height/placement and other provisional model assumptions are unchanged.
See [method and limitations](../../sircus/mjlab/IMPACT_POSE_SWEEP.md).

The source-bound plan found three qualifying selected-checkpoint short postures
at two validation bucket positions: 03/sweep, 05/PPO and 05/sweep. No missing
method/layout example was invented. Each source fixes bucket XY and board XYZ,
then requests four gap offsets (0/60/120/180 mm) and three tilt offsets
(-4/0/+4 degrees). All original IK, collision and forward/reverse path checks
remain active. Higher settings are rejected, not clipped or substituted.

| Phase | Requests | Native releases | Captures | Setup rejects |
|---|---:|---:|---:|---:|
| Height/tilt search | 36 | 18 | 0 | 18 |
| Separate selected-setting repeats | 9 | 9 | 0 | 0 |
| Total | 45 | 27 | 0 | 18 |

All 18 setup rejects are `finite_ik_unresolved` at the +120/+180 mm gap levels;
this is a finite solver result, not proof that no arm posture can reach them.
All 27 native throws have a measured board rebound, but no whole-ball entry and
an observed short crossing. The smallest tested shortfalls to the nominal
whole-ball-clearance interval at the diagnostic descending height plane are
159.9 mm (03/sweep), 103.3 mm (05/PPO) and 92.3 mm (05/sweep). These describe
recorded geometry, not a causal model of speed, tilt or future performance.

**None of the selected settings is a successful recipe.** The predeclared rule
selects capture/quality/retention first, then closest scored distance and stable
ID. With no captures, it picks a proximity fallback. The closest-distance metric
can include later external-contact motion; it is not identical to crossing
shortfall. Repeats did not choose a replacement. All three cases scored 0/3.

| Source posture | Horizontal board/bucket distance | Board-centre height | Selected tilt | Selected gap | Ball-centre release height | Repeats |
|---|---:|---:|---:|---:|---:|---:|
| 03 / sweep | 593.0 mm | 409.1 mm | 36.750° | 195.3 mm | 641.9 mm | 0/3 |
| 05 / PPO | 533.3 mm | 417.5 mm | 32.744° | 128.2 mm | 583.2 mm | 0/3 |
| 05 / sweep | 525.7 mm | 409.1 mm | 28.750° | 135.3 mm | 581.9 mm | 0/3 |

The gap means ball bottom above the **board-centre plane**. Ball-centre release
height is board-centre Z + gap + 37.5 mm radius. Exact requested settings,
rejections and observed classifications are in [settings.csv](settings.csv).

## Pose evidence and numerical controls

Both arms' integrated joints/velocities are saved every native state (15,360 Hz).
CPU FK reconstructs board/flange/link poses at 240 Hz throughout the four-second
horizon and every native state within 10 ms of first board impact. Contact
intervals, held pre-step commands and post-integration poses are distinct.
Across the 27 new traces, maximum sampled board displacement was 0.0004578 mm,
normal change 0.0001170°, initial ball-centre release error 0.00003263 mm and
integrated joint excursion 0.000001192 rad. These are numerical observations in
an endpoint-reset simulator, not physical accuracy or natural arm settling.
The ball is spawned; grasping, compliant impact and real release are untested.

The independent data/pose audit passed **42,760 checks**, and a fresh CPU-only
process reproduced its exact result without new physics or evidence writes.
This audit binds every requested action, raw packet, score, FK sample, replay,
selection and repeat. It does not certify physical fidelity or equivalence of
different simulator runs.

The stricter 13-world logger control remains **failed full-path equivalence**,
including baseline versus baseline. All 16,432 FK samples independently checked;
initial flight/first-impact ball positions, clocks and contact geometry matched,
as did capture/entry/rebound/quality decisions. Later paths differed by up to
0.578 mm; impact forces/velocities and later contact counts/peaks also varied.
The [explicit CPU review](control-review.json) is
`reviewed_with_baseline_variability`, not a retroactive pass. Physics/calibration
thresholds were never relaxed. The [strict report](control-report.json) remains
failed. Earlier two-throw strict and three-singleton controls are retained
separately on Ubuntu; the singleton numerical check passed. An unpublished first
sweep hit a logger metadata indexing error; its corrected successor reused the
same grid in a new output directory. No failed run supplied selection outcomes.

## Reproduction and next step

Native producer source: `5e8ba501f528d929a7b181c97d757771b64844d0`.
Report SHA256: `b143eaa2a369f0a7ad4fdd1dcb325cac21cfdceb4f51b133ba4bc1fdd2ae7535`.
Audit SHA256: `13f4c3ccb2cac553058fd06c4e845ae81d01c4185434c2ef3dc448e40bafbff4`.
The report records 119.8 s of setup/native/report construction, excluding later
independent CPU verification and separate engineering controls. Pinned package
versions are in the report. The new output is `output/impact-pose-v2` under the
isolated Ubuntu mjlab root. All earlier studies/services are preserved; 4070,
hardware, Viam and audio were untouched.

Open [the read-only interface](http://127.0.0.1:18793/#impactPoseSweep), choose a
source posture, click a grid square and inspect exact settings, impact poses and
the recorded replay. This view cannot run new shots or command hardware.

This compact bundle contains exact metadata and checksums only. The derived
manifest references unbundled native packets, paths and replay files retained on
Ubuntu. Raw observations/trajectories are excluded from Git. Reproduction needs
those retained inputs and matching assets. This is local development diagnosis,
not all-table coverage or new policy evaluation. Next: audit the rejected high
releases and allow a closer/lower board posture, with direct-flight proximity
kept distinct from later external-contact motion, before another long RL run.
