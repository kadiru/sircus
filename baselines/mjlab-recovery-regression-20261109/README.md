# Recovery regression diagnosis — November 9, 2026

The completed recovery comparison remains **37/70 captures versus parent
imitation 34/70**, with eight gains and five losses. The five losses comprise
**three setup rejections and two released misses**. This inspection makes the
original TEST positions development data; it does not change their historical
score. The focused correction is **completed and independently audited**.

[Correction results](summary.json): all 63 requested slots completed in
221.1 worker seconds, with 48 actual releases and 22,266 independent checks.
Both formal control blocks passed. There were no physics-invalid attempts.

| Selected loss cases, three requests each | Captures / requests | Releases | Setup declines |
| --- | ---: | ---: | ---: |
| Exact parent recipes | 15/15 | 15 | 0 |
| Exact recovery recipes | 0/15 | 6 | 9 |
| Restore parent board height only | 3/15 | 9 | 6 |

**Only `r3-c1-n0` recovered, with 3/3 direct, clean captures.** Its board height
changed from 241.129 to 280.424 mm. The unchanged recovery coordinates were
board X/Y 613.087/−14.842 mm, tilt 26.792°, gap 173.760 mm, and bucket X/Y
339.213/−30.949 mm. Requested release-center Z became 491.685 mm. The two
released regressions still missed 3/3 each; the two remaining IK cases still
declined. All diagnostic flights were free of robot contact. Six deliberately
negative control flights had robot contact and remained misses, as expected.

This recovered **one development case**, not a new policy score of 38/70.
The nine old refused slots are cached decisions; the six corrected refusals
reuse two newly checked actions. They are not independent repeated IK searches.
No weights or active policy changed. Next: test the two flight losses' drop-gap
hypothesis separately, while retaining the verified complete parent recipes
as controls for the unresolved setup cases.

[Compact evidence](diagnosis-summary.json) records all 33 recovery failures,
the five exact paired recipes, saved admission/flight facts and source hashes.
The immutable parent passed 283,251 checks; this CPU diagnosis passed another
**3,103 source-binding, accounting and independent v3 rescoring checks**.
Raw trajectories, joints, checkpoints and full proofs remain outside Git.

| Recovery failure | Count |
| --- | ---: |
| No finite dropper IK branch | 5 |
| No finite board IK branch | 5 |
| Neither arm had a finite IK branch | 1 |
| Strict endpoint clearance/pose rejection | 2 |
| Fixed-base clearance rejection | 2 |
| Action bounds rejection | 1 |
| Released shot without whole-ball entry | 17 |

The 17 released misses first contacted the bucket after board rebound in six
cases, the table in six and a robot in five. Seven had forbidden contact at
some point. These contact observations do not by themselves establish why a
shot failed; table and bucket contacts remain allowed.

The two lost released captures are `r0-c5-n1` and `r0-c6-n1` (all IDs carry the
`recovery-test-` prefix). Both parent shots captured after table contact; both
recovery shots had no forbidden contacts but never entered. Their first bucket
contacts occurred earlier and lower:

| Case | Parent: time / ball-center Z | Recovery: time / ball-center Z |
| --- | --- | --- |
| r0-c5-n1 | 0.702604 s / 219.725 mm | 0.609115 s / 185.570 mm |
| r0-c6-n1 | 0.721549 s / 216.957 mm | 0.632552 s / 192.247 mm |

The configured rim is 183 mm high and ball radius is 37.5 mm. Parent whole-ball
entry occurred at 0.876758/0.966927 s, followed by one continuous second retained.
Recovery gaps were larger by 30.567/37.745 mm, but all five predicted parameters
changed; this comparison does not isolate a causal gap effect.

The three setup losses are `r1-c1-n1` (no dropper branch), `r2-c4-n0` (no board
branch) and `r3-c1-n0` (all six checked board branches rejected on table/link5
contact). Finite search rejection is not proof of unreachable geometry.

One correction is frozen for all five losses: **restore only board-height
parameter index 2 to its matched parent value**, retaining the other four
recovery parameters and bucket position. It tests admission sensitivity and may
leave the two flight misses unresolved. With gap unchanged, absolute release Z
also shifts by the height delta; this is one shot-parameter intervention, not
isolated physical board movement. No gap compensation or second correction is
included. The completed comparison had three nominal requests per case for exact
parent, exact recovery and corrected actions, plus 18 controls: 63 total slots.
Historical refusals remain cached refused requests, with no placeholder flight.

No new training, selector activation, 4070/Isaac change, service restart or
hardware action is part of this diagnosis. Fixed64 held-endpoint simulation
does not validate arm transit, noise robustness or generalization.

The live `#recoveryRegression` panel includes all five paired cases, all 33
failure categories, corrected first-repeat traces and a **60.13-second recorded
3D replay** of the two historical flight regressions plus a representative
robot-contact miss. The movie uses saved ball samples and held joints; it does
not invent arm movement or show a newly trained policy. Its 1,804 frames passed
decoding and browser playback. [Compact media provenance](video-summary.json)
records source/clip hashes and verified CPU rendering; raw media stays local.
