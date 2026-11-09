# Height and tilt admission search for four development cases

This follows the [tilt-only correction](TABLE_TRAINING_DIAGNOSIS.md), whose
32° requests all failed the finite board-arm IK search. Search cases
`table-training-test-00`, `01`, `18` and `26` without fitting another policy.
These are inspected development cases, not fresh held-out evaluation positions.

## Frozen search

For each case, use every combination of:

- Board-center height above the table: **320, 360, 400, 440, 460, 480 mm**.
- Board tilt from horizontal: **16, 20, 24, 28, 32 degrees**.

This gives 30 candidates per case and 120 in total. Preserve each original
bucket XY, board XY and drop gap exactly. Height and tilt are the only changed
shot parameters. The full flange orientation convention, IK start/iteration
budget, 5 mm clearance requirement, checked home path, physics, native 64
substeps and v3 scorer remain unchanged. No heuristic reach estimate becomes
a rejection rule.

The extra 460 mm row covers a narrow potential overlap between the two arms'
reach limits near 28°. That geometric estimate guides candidate placement;
only the existing exact-action admission checks decide whether to release.
Failure of this finite IK search does not prove a configuration impossible.

Write and hash the complete CPU admission matrix before native discovery.
Retain every declined or unattempted candidate with its reason; distinguish
board/dropper IK branch availability from full endpoint/path admission.
Run native drops only for admitted candidates, using their recorded joint
branches. A passed setup check does not establish a successful ball flight.

## Native comparison

Use the unchanged original shots three times per case (12 requests), and
three direct-capture, three table-assisted-capture and three expected-miss
controls before and after the experiment (18 requests). Execute the admitted
grid candidates once for discovery.

For each case, select only a discovery capture, ordered by route quality,
then v3 reward, then lowest grid index. Freeze that selection before three
fresh nominal repeats of each selected configuration (at most 12 requests).
Do not select a miss merely to fill the repeat budget. These repeats contain
no added noise and are not evidence of noise robustness.

The maximum is **162 requested configurations/repeats**, including all 120
grid entries whether admitted or declined. Report geometry requests and
native releases separately. Cap CPU admission at one hour, the acquired
4090 native phase at one hour, and waiting for the shared GPU slot at two
hours. Preserve partial evidence if a cap or the 8 GiB storage reserve is
reached. Leave other GPU workloads, Isaac and the existing viewer untouched.

Capture requires a measured board rebound, whole-ball entry and one continuous
second inside. Table and rim contacts are allowed; robot, floor and unknown
contacts are forbidden. Preserve the original scoring definition and provenance
on historical results. This experiment holds simulated arm endpoints; checked
setup paths do not constitute a dynamic arm-transit demonstration.

## Evidence and interface

Independently reconstruct the grid, exact baseline branches, admission matrix,
selection and request denominators; rescore dense native traces at 64 substeps.
Bind reports to source, scene, assets, parent evidence and admission proofs by
hash. Display provisional progress separately from audited results in the
additive `#tableHeightTiltStudy` panel. Keep the previous diagnosis visible.

Report exact successful height/tilt/gap combinations and repeat outcomes, or
state clearly if no successful setting is found. A positive result here is a
development recipe, not policy generalization. No PPO, active-policy replacement,
noise test, 4070 work, Isaac change or physical operation is part of this task.

## Completed result

[The frozen experiment](../../baselines/mjlab-table-height-tilt-20261108/README.md)
finished all 120 admission checks: 58 admitted and 62 finite IK declines.
Every admitted shot bounced the board, but none achieved whole-ball entry or
capture; 42 had robot contact. The original shots again captured 0/12. No
discovery capture qualified for the conditional three-repeat stage. All 18
controls passed, and the independent audit passed 36,371 checks across 88
native releases and the full 150-request accounting.

The 460 mm row found a reachable 28° posture for case 00 that the adjacent
440/480 mm rows did not admit. Reachable endpoints alone did not solve the
flight. Keep the current policies unchanged and investigate ball clearance
and alternative admitted geometry before more PPO.
