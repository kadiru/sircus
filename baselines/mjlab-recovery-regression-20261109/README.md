# Recovery regression diagnosis — November 9, 2026

The completed recovery comparison remains **37/70 captures versus parent
imitation 34/70**, with eight gains and five losses. The five losses comprise
**three setup rejections and two released misses**. This inspection makes the
original TEST positions development data; it does not change their historical
score. The focused correction is **pending**.

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
included. The planned comparison has three nominal requests per case for exact
parent, exact recovery and corrected actions, plus 18 controls: 63 total slots.
Historical refusals remain cached refused requests, with no placeholder flight.

No new training, selector activation, 4070/Isaac change, service restart or
hardware action is part of this diagnosis. Fixed64 held-endpoint simulation
does not validate arm transit, noise robustness or generalization.
