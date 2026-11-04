# Impact poses and a controlled short-miss sweep

This is a new validation-only development experiment, separate from the frozen
[calibrated sweep/PPO benchmark](CALIBRATED_BENCHMARK.md) and its
[recorded failure audit](VALIDATION_FAILURE_AUDIT.md). It does not update a policy,
alter the scorer, relabel historical shots, or open native final-TEST packets.
The benchmark report, manifest, calibration and audit remain byte-identical.

## What is measured

The additive `ImpactPoseEnvironment` calls the unchanged calibrated physics loop.
It records both arms' actual integrated joint positions and velocities at every
native state (15,360 Hz). CPU MuJoCo forward kinematics reconstructs board centre,
orientation/normal, the dropper flange and both arms' links after the rollout.
Transforms cover the full four-second horizon at 240 Hz and every native state
within 10 ms before/after the first measured board-contact interval. Invalid
padding is explicitly masked. Ball and arm records share exact native indices.
A separate CPU pass reconstructs every exported transform from the saved joints.

No extra forward call touches the running GPU simulator. A paired engineering
control compares uninstrumented and instrumented ball/contact arrays and scores
before the new sweep. Dense observations and full replay payloads stay outside
Git; compact summaries and checksums are suitable for publication.

These are **prescribed endpoints**, not a compliant arm controller. The engine
resets both arms to their commanded joints before every integration interval.
Contact force row `k` describes interval `[t[k],t[k+1]]` at that held pre-step
pose. Recorded state `k+1` is the integrated response before the next reset.
Both are reported distinctly. Sampled board extrema are not continuous-time
maxima. No natural settling, physical stiffness, grasp or gripper release is
validated; the ball is spawned in free flight. The board contact point/normal
itself is not measured by this pose logger.

## Predeclared experiment

The prior failure audit identifies selected-checkpoint short examples at two
validation bucket coordinates. Deterministic selection takes one saved posture
per method per coordinate (four cases when all are available). Every source row,
native packet and original report is bound to the frozen manifest and audit.

For each case, bucket XY and board XYZ stay fixed. The only varied values are:

- Ball-bottom gap above the board-centre plane: baseline plus 0, 60, 120, 180 mm.
- Bucket-facing board tilt: baseline minus 4°, baseline, baseline plus 4°.

This gives 12 requested grid points per case, including its exact baseline,
followed by three fresh nominal repeats of its selected setting. The actual
source-bound plan has three qualifying postures at two bucket positions: 36
search plus nine confirmation requests. A missing method/layout example is not
invented. Logger-equivalence engineering throws are outside that budget.
The initial two-throw exact comparison is retained separately: positions,
clocks and contact geometry matched, while force/velocity differed by at most
two FP32 representations. A new predeclared three-throw control compares two
uninstrumented runs and one instrumented run. Only force/velocity observations
allow up to eight FP32 representations per element; geometry, clocks, contact
threshold masks and the complete scalar scoring output must match exactly.
This engineering comparison does not relax capture scoring or calibration.
The three singleton controls passed. A later three-batch, 13-world control
(39 engineering throws) exercised the corrected multi-world metadata layout and
verified all 16,432 FK samples. Its full-horizon comparison failed, including
baseline versus baseline: later paths differed by up to 0.578 mm. Ball positions,
clocks and contact geometry matched through first impact and first external
contact; impact force/velocity observations had small numerical differences.
Capture/entry/rebound/retention/quality decisions agreed. The explicit CPU review
is **reviewed with baseline variability**, not a retroactive equivalence pass.
The bounded development study proceeds with independently scored observations
and fresh nominal repeats, without changing physics or capture thresholds.
The first attempted sweep also retained an unpublished batch-metadata indexing
failure; its corrected successor uses the same requested grid and new output.
Selection uses capture, quality,
retention, observed opening proximity, then stable request ID. A case with no
admitted native request remains unresolved; its baseline repeats still count as
requested failures. Three nominal repeats do not demonstrate robust capture.

Absolute ball-centre release height is **board-centre Z + gap + ball radius**.
The gap is not the distance to the board's tilted upper surface. Existing bounds,
strict IK, minimum board/bucket distance, collision and forward/reverse path
checks remain unchanged. Rejected settings are shown and stay in denominators;
there is no clipping, hidden fallback or failed-layout resampling.

The small grid is a local diagnostic, not an exhaustive feasibility map. Its
selection and repeats are not new held-out policy scores. Any resulting method
change needs a fresh final-test cohort before a new performance claim.

## Run and inspect

Use the already pinned Ubuntu mjlab environment and select only the authorized
RTX 4090 UUID. The shared GPU lock waits without interrupting other jobs. The
4070 workload remains untouched. New source snapshots and output directories
must be outside every frozen study. See `python -m
sircus.mjlab.impact_pose_sweep --help` for pinned input hashes, plan-only mode and
the separate CPU verification pass. Dependencies are in `requirements.txt`.

`impact_pose_server` serves a separate read-only interface on port 18793 at
`#impactPoseSweep`. It requires pinned new report/audit hashes and verifies their
bindings to the previous diagnostic and original benchmark on every read. All
POST commands return 405. Recorded replays use opaque, whitelisted, hash-bound
GETs; rejected settings have no fabricated trajectory. Existing services are
preserved. UI playback uses observed FK samples, not invented arm travel.
