# Full bucket-approach development test

The preceding 50-position comparison captured 44/50 with each planner. Three
saved misses hit the bucket wall while rising: expanded case 04, current case 10,
expanded case 19. The forecast checked a later descending opening crossing and
could overlook an earlier collision. These are now known development targets,
not a new generalization test.

The opt-in `BucketApproach` checker follows the predicted release, board flight,
and optional table rebound against compiled bucket collision geometry with the
ball radius. Its continuous-segment lower bounds account for the space between
sampled times. This is a forecast warning/ranking input, never a new success rule.
Board-first, whole-ball entry and one-second retention remain unchanged; later
table, robot and other bounces remain permitted. A prediction cannot yet model
every later rebound, and a collision warning can occur on a successful rim shot.

CPU preparation tries local tilt/gap changes and transfers from the existing
TRAIN shot library. Candidate ordering alternates the two families after a
necessary wrist-reach and nominal opening screen. Each target has at most 40
setup checks, 30 seconds per check and 600 seconds per case, using two CPU workers.
Every admitted proposal retains full arm-pose/path proof. The largest full
approach score wins, with stable ID ties; it must improve over the unchanged
original forecast. Nonpositive scores are permitted and marked risky. No choice
uses new native outcomes. The best candidate from the same menu without the new
bucket check is also recorded, to distinguish candidate expansion from reranking.

Freeze the complete commands and predictions, then run at 64 substeps:

- Three unchanged and three chosen-correction repeats per wall-miss target: 18.
- Four exact previously successful settings, three repeats each: 12.
- Current case 47's exact robot-contact miss, three unchanged diagnostic repeats: 3.
- Standard pre/post controls: 18.

Maximum 51 native releases; a declined correction retains its requested denominator
but launches no ball. All controls/results are reported, including preservation
regressions. Standard control failure stops execution. The 4090 cooperative lease
caps acquired GPU time at one hour. No 4070/Isaac changes, retraining, live planner
promotion, service restart or hardware actions. Arms are held at checked endpoints
during the native flights; this does not validate driven-arm motion.

Run CPU-only `python -m sircus.mjlab.bucket_approach_study --stage prepare`, then
`--stage freeze`, supplying source study/audit/diagnosis, matched control parent,
runtime reference/assets, fresh output and GPU lock paths. The separate
`sircus.mjlab.bucket_approach_audit --stage preparation --audit-output ...` must
pass before `--stage run --preflight-audit ...`. Afterwards use `--stage finalize`
and independent audit `--stage full`. Preserve incomplete studies; never overwrite
or silently rerun an already-started native study. Both audits are read-only and
reconstruct commands, proofs and (after execution) every saved score.
