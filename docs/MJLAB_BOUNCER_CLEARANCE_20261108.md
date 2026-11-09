# Bouncer clearance search — November 8, 2026

Moving case 00's board center **120 mm in +Y** removed observed robot contact in
all three fresh repeats. **It still captured 0/3 and never achieved whole-ball
entry.** Cases 18 and 26 still contacted a robot at every admitted XY setting.
This is a useful clearance improvement for one inspected layout, not a successful
replacement shot or a newly trained policy.

## What was tested

The [frozen protocol](../sircus/mjlab/BOUNCER_CLEARANCE.md) tested a 5×5 XY grid
per case, using offsets −120/−60/0/+60/+120 mm. Bucket XY, board height, board tilt
and drop gap stayed fixed. Heading continued to face the bucket, so shifting XY
also changed heading, horizontal board-to-bucket distance and release XY. Original
joint branches were preserved for matched baselines; no solver or guard was
relaxed and no same-pose alternative branch was substituted.

| Stage | Requested configurations/repeats | Native drops | Captures | Robot-contact drops |
| --- | ---: | ---: | ---: | ---: |
| XY discovery | 75 | 9 | 0 | 7 |
| Exact original baselines | 9 | 9 | 0 | 9 |
| Selected case 00 +120 mm Y, fresh repeats | 3 | 3 | 0 | 0 |
| Direct/table/miss controls, before and after | 18 | 18 | 12 | 6 |
| Total | **105** | **39** | **12, all controls** | **22** |

All 75 proposals were attempted. The 66 declines were `finite_ik_unresolved` and
remain explicit unflown outcomes, not ball-flight misses or proof of geometric
impossibility. Every admitted candidate was flown. All nine discovery flights
bounced from the board; none entered the bucket. There were no invalid physics
rows. Both control gates passed: six direct captures, six table-contact captures
and six expected misses.

## Exact configurations

Coordinates below are the simulator's table frame, in millimetres. Board Z is
center height above the tabletop; tilt is from horizontal. Values are rounded
for reading; the ledger retains full precision. These are tested misses, not
successful recipes.

| Case / setting | Board X, Y, Z (mm) | Bucket X, Y (mm) | Tilt | Drop gap | Horizontal distance |
| --- | --- | --- | ---: | ---: | ---: |
| 00 original | 597.992, −143.557, 460 | 1109.540, −221.464 | 28° | 251.819 mm | 517.446 mm |
| 00 selected +120 Y | 597.992, −23.557, 460 | 1109.540, −221.464 | 28° | 251.819 mm | 548.496 mm |
| 18 original | 593.577, −125.029, 460 | 1101.717, 181.450 | 28° | 240.200 mm | 593.410 mm |
| 26 original | 590.267, −125.631, 480 | 1188.291, 168.864 | 24° | 242.833 mm | 666.603 mm |

The only admitted offsets in each case were `(ΔX, ΔY) = (0, 0), (0, +60),
(0, +120) mm`. Discovery outcomes were:

| Case | Original | +60 mm Y | +120 mm Y |
| --- | --- | --- | --- |
| 00 | Robot contact, no entry | No robot contact, no entry | No robot contact, no entry; selected |
| 18 | Robot contact, no entry | Robot contact, no entry | Robot contact, no entry |
| 26 | Robot contact, no entry | Robot contact, no entry | Robot contact, no entry |

For case 00, the exact original repeats recorded first robot contact at **0.4763 s**.
The selected setting had no robot contact before its scoring terminal in all
three repeats. In its first repeat, the board contact occurred at **0.2230 s**,
bucket contact at **0.5494 s**, then table contact at **1.3520 s**. The ball strikes the near rim, rebounds upward, passes over the opening and
lands outside. It recorded rim-band contact but no whole-ball entry. The release ball center was
`(597.992, −23.557, 749.319) mm`. Mean unchanged v3 reward rose from **−0.194** to
**+0.361** for those matched repeats; the improvement is shaped progress, not a
capture. Table contact is allowed and was not the reason this shot failed.

## Interpretation and next work

The user's point is correct: collision, short flight, rim contact and misses are
expected search outcomes and should teach the system what is feasible. All failed
proposals and native outcomes were retained. We have not installed an avoidance
filter that simply refuses these regions, nor promoted a policy based on this
small search. Selecting a contact-free candidate only governed this diagnostic's
three repeat slots; it did not relabel excluded candidates or change training.

Next, use the clear case 00 family to search a bounded tilt/drop-gap neighborhood
for actual entry and retention. Cases 18/26 still need another admitted posture
family; same-pose alternative IK branches require explicit branch-choice evidence
before native replay. Preserve collision examples alongside successes. This is
posthoc development on three previously inspected layouts; new held-out bucket
positions are needed for a later policy comparison.

## Recorded video

The UI includes a **43.4-second, 1280×720, 30 fps replay** of actual saved native
ball samples: original case 00, its +120 mm Y repeat, unresolved +120 mm Y cases
18/26, and a table-assisted positive control. Case 00 uses identical before/after
cameras. Normal-speed views show each full scored interval; quarter-speed
close-ups emphasize impact, with an overhead view showing control retention.
The two arms remain at their recorded held endpoints. No arm transit, ball spin
or new physics is synthesized. This is a replay of simulation evidence, not a
policy rollout or a recording of learned arm motion.

The panel provides five chapter buttons, playback speed and MP4 download. The
compact summary binds the movie and per-frame provenance to the passed native
audit. CPU llvmpipe rendered all 1,302 frames; full MP4 decoding and byte/hash
checks passed. Chrome verified the published counts, five chapters, playback,
pause, speed selection and 43.4-second media metadata. Movie SHA-256:
`4ed1a411c0f3cc2b2aa88802c2931d18d9cf05a0f7de81a16ef26a9fd41cb52b`.
The existing viewer was updated additively with a backup and compare-and-swap;
other HTML sections and report keys were preserved and no service restarted.

## Evidence and scope

- Independent audit: **17,380 passed checks**, full native replay verified, no failures.
- CPU admission: **34.5 s**; leased native phase: **208.3 s** of wall time, not isolated GPU compute.
- Frozen producer: `a6fe6af7afce0ecfe667bbe7efbfdd9f8f39b437`; draft [PR #98](https://github.com/kadiru/sircus/pull/98), stacked on [PR #97](https://github.com/kadiru/sircus/pull/97).
- [Compact results](../baselines/mjlab-bouncer-clearance-20261108/README.md); interface anchor `#bouncerClearanceStudy`.
- Contract fingerprint: `51d508b341bba79e002c0a953e74d81317e3b4873533f71e2548626bc9475b21`.
- Report SHA-256: `a284d8e785a3d5aec8a45f92877377f63812c38767d6738fe1b6669a69b399a4`.
- Audit SHA-256: `7de9a6222d65aca1f1ac884566459bb30b250348f6c0175abc78177965b5e175`.

Native64 physics, v3 scoring and one-second whole-ball retention were unchanged.
No PPO, noise, timestep comparison, policy activation or physical work was done.
The 4070 workload and Isaac setup were left untouched. Raw trajectories, host
configuration and full media remain outside Git.
