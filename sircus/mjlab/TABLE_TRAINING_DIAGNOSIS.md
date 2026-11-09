# Table-assisted policy diagnosis and one tilt correction

This follow-up diagnoses the audited [table-assisted training comparison](TABLE_ASSISTED_TRAINING.md).
The selected PPO captured 15 of 50 requested bucket positions: 31 released
misses and four setup declines remain. Those TEST outcomes are now inspected
as development evidence; the original frozen comparison is not overwritten.
There is no new model fitting or automatic policy promotion.

## Diagnosis

Bind the parent report, independent audit, row ledger, geometry proofs and dense
native archives by their recorded hashes. Inspect all 35 failures, retaining
setup declines separately from executed ball flights. A first post-board contact
is an observed force event; it is not necessarily the cause of a later miss.
All 31 released misses bounced on the board, but none achieved whole-ball entry.
First subsequent contacts partition them into table (16), bucket (8) and robot (7).
Thirteen misses had forbidden contact at some point; that count differs from
seven robot-first contacts because later collisions also matter.

Cases 06 and 36 had 9 and 6 valid dropper IK branches, respectively, but no valid
board-arm branch within the fixed finite search. Their anchor configurations
were admitted; the residual policy's requested endpoints were not. This does
not prove that every configuration for those bucket positions is unreachable.
Cases 14 and 46 failed fixed-base/bucket clearance checks. Case 14 includes an
unresolved conservative distance bound, not a proven penetration measurement.
The 5 mm clearance requirement and all existing admission checks stay in force.

For representative flights, show actual saved positions and contact timing,
requested board height/tilt/drop gap, closest observed approach and any measured
rim-plane crossing. A first-flight shortfall does not itself imply failure:
successful table-assisted shots can land short before reaching the bucket later.
Keep table contact, whole-ball entry and retained capture as distinct facts.

## Frozen correction experiment

Hypothesis: the shallow far-shot family produces an unfavorable outgoing path;
changing only board tilt toward the roughly 32° observed in successful far
shots may reduce robot encounters or improve subsequent table/bucket entry.
Those successful examples also have different heights/gaps, so they are a
motivation for a controlled test, not proof that tilt alone explains success.

Select the first two context IDs in each observed first-post-board-contact
class (table, bucket, robot) satisfying original tilt <22° and horizontal
board-to-bucket distance ≥450 mm. Selection uses the existing diagnosis only;
no corrected outcomes are inspected. The rule yields:

- Table-first: `table-training-test-00`, `table-training-test-01`.
- Robot-first: `table-training-test-18`, `table-training-test-26`.
- Bucket-first: no eligible cases; do not substitute other cases.

For each of these four development contexts:

1. Replay the unchanged original request three times using its authenticated
   admission proof and exact joint branch.
2. Change only the requested tilt to **32°**, obtain fresh admission under the
   same rules, and request three repeats. Keep bucket XY, board XY/height and
   release gap identical. Refusals count as failed requested attempts; never
   substitute a different action or loosen an IK/collision guard.
3. Run three repeats each of direct-capture, table-contact-capture and expected
   miss controls before and after. Controls are separate from the four cases.

Total **42 requested slots**: 24 paired development requests + 18 controls.
The acquired 4090 slot is capped at **3,600 seconds**, with an 8 GiB storage
reserve and cooperative GPU lock. No further search follows automatically.
The 4070, Isaac setup, existing viewer and physical services remain untouched.

Use unchanged native **64 substeps**, contact physics, v3 scoring/rewards,
geometry, canonical home and source prior. Capture still requires a measured
board bounce, whole-ball entry and one continuous second inside. Table/rim
contacts are allowed; robot, floor and unattributed contacts remain forbidden.
The correction is a test of requested shot geometry, not a learned arm-motion
controller: endpoints remain initialized and held as in the parent study.

## Reporting and verification

Report requested/released/captured counts, setup declines, robot/other forbidden
contacts, entry and invalid physics by case and variant. Retain all failed or
partial evidence. Independently rescore dense native observations and verify
source identities, unchanged parameters, exact baseline branches, fresh
correction admission and both control gates before calling results verified.
Do not add these post-hoc repeats to the original 50-position policy score.
The read-only `#tableTrainingDiagnosis` panel shows diagnosis and correction
results separately. A successful correction here still needs fresh positions
before it becomes a policy-selection rule.

## Completed outcome

[The 42-request test](../../baselines/mjlab-table-training-diagnosis-20261108/README.md)
passed both control gates and 12,557 independent audit checks. Original shots
captured 0/12, with six robot-contact shots. All 12 requests changed to 32° failed
the finite board-arm IK search before release. No corrected flight was observed,
so the experiment does not establish a contact or capture improvement. Retain the
existing policy and find admitted joint height/tilt combinations before more PPO.
