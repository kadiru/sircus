# Learned shot ordering: completed comparison

Completed **November 17, 2026 at 06:47:17 PST** (`2026-11-17T14:47:17Z`).
The learned selector captured four more balls overall than the current planner
on 200 new bucket positions. Typical planning was slightly faster, but difficult
positions took longer. **The interactive default has not changed.**

## What was tested

A small supervised neural network orders proposed shot configurations before
expensive arm-pose and path checks. Exact admission checks and the existing
physics flight score still choose the final shot. This is not another PPO run
or a policy that directly controls moving joints.

The study collected **3,901 training drops at 160 positions** and **613
validation drops at 40 other positions**. Validation selected the **500-epoch**
checkpoint. The model and all three planners' selected commands were frozen
before testing on **200 further, unseen positions**. Each planner received one
drop per position at **64 substeps**, for **600 test drops** total.

## Results on the same 200 positions

| Planner | Captures | Capture rate | Feasible setups | Median planning | 90th-percentile planning | Robot-contact shots |
|---|---:|---:|---:|---:|---:|---:|
| Current physics planner | 169/200 | 84.5% | 200/200 | 27.6 s | 67.1 s | 16 |
| Conditional wider search | 170/200 | 85.0% | 200/200 | 33.5 s | 222.6 s | 20 |
| Learned shot selector | 173/200 | 86.5% | 200/200 | 24.8 s | 168.3 s | 19 |

The 90th percentile is the time within which 90% of positions finished planning.
Relative to the current planner, the learned selector captured at **ten positions
where the current planner missed**, but missed at **six positions the current
planner captured**. Both captured at 163 positions; both failed at 21. The net
gain is four captures, or two percentage points. This does not establish a
reliable general improvement, and the longer slow-case times preclude an
overall speedup claim.

Capture requires a board rebound, whole-ball entry, and one continuous second
inside the bucket. Rim assistance and table bounces are allowed. The selector's
**one robot-assisted capture** is reported separately and excluded from standard
captures. **One wider-search and three selector invalid-physics attempts** remain
in the 200-position denominators. They are not evidence of valid physical misses.

## Validation and accounting

- **36 matched controls passed**, around collection and testing.
- The independent audit **passed 939,248 checks**, including frozen choices,
  split separation, native execution, scoring, provenance, and accounting.
- Completed native releases: **4,514 collection + 600 test + 36 controls = 5,150**.
- Total native GPU lease: **8,162.613 seconds = 2.27 hours**, within the combined
  seven-hour cap. CPU preparation, fitting, and waiting are separate from this
  GPU lease time.
- The 4090 ran the native drops. The 4070 and Isaac setup were not modified.

## Evidence and reproduction

The [method and ordered runner stages](NEURAL_SHOT_STUDY.md) describe the frozen
input chain, candidate generation, checkpoint selection, scoring, and audit.
The implementation launched from commit
`e2645e57c4f5599fde8fed49807ba1f07467c423`.

Ubuntu evidence remains outside Git:

- Study directory: `/mnt/robot-training/sircus-mjlab/neural-selector-v1`
- Report: `report.json`; inventory: `evidence-manifest.json`
- Model: `neural-checkpoint.json`; fitting record: `fit.json`
- Native measurements: `rows.jsonl` and the study's native evidence directories
- Independent audit: `/home/kadir/sircus-mjlab/output/neural-selector-v1-audit.json`

The report SHA-256 is
`9ba17a95fc7d6acbefa8530fb9176ae73ad532937dfe5755088a83988fb73cf6`.
The evidence-manifest SHA-256 is
`17f011266f788484e78fd8ceebc1f850ba60f5a4b82bfc25ba04a43436ed2e8a`.

The derived UI section is `#neuralShotStudy`; final scores are published only
after the audit passes. No automatic model promotion or service restart occurs.

## Next step and limits

Keep the current default while testing a **bounded planning-time/candidate
budget** on fresh positions. Compare capture rate and slow-case planning time
before deciding whether to enable the learned selector by default. Another
broad training run is not yet justified by these results alone.

Perception is a separate next integration: rendered depth/point clouds should
produce object estimates, with simulator truth retained separately for error
measurement. This study used exact simulator geometry. Native drops held both
arms at their selected endpoints; sampled CPU path checks do not demonstrate
dynamically executed arm motion, physical grasping, or real-world performance.
