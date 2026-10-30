# Taped pickup stations

The Manual operation show desk has separate **Hand loading** and **Taped station** controls. A station holds one repeatable pickup pose for each operator-selected color. Teaching a station does not change the saved hand-loading pose, drop pose or board reference.

## Teach one color at a time

1. Position the taped pad and select Pink, Purple, Green or Gray.
2. Manually position arm10's gripper at the intended grasp pose using the established Viam controls, then stop jogging.
3. Choose **Teach pickup pose**. This only reads stationary robot state and saves the full pose. Review XYZ and orientation before moving again.
4. The approach waypoint is derived with **world Z +100 mm**, identical X/Y and full orientation. It is a flange pose reported by `Motion.GetPose(arm, world)`, not a measured ball-center position. Pad/ball geometry is not inferred from photos.
5. Repeat for the other colors. The independent poses and their teaching provenance stay in host-local calibration, excluded from Git.

The optional approach-only buttons move to either the selected station's approach or 100mm above the earlier hand-loading target. They do not open, close or descend. The earlier hand-loading target is not assumed to be a tabletop pickup point.

Starting at a taught grasp uses a constrained straight-up retreat. An unrecognized low start, another station or an interrupted descent blocks lateral travel until the operator manually establishes clearance; the controller does not guess a recovery path among the balls.

If station heights differ, generic transfers use a staging endpoint at the highest saved approach Z. Extra vertical constrained segments connect that level to the selected station's own 100 mm approach, including the return after pickup. These endpoints do not prove that every robot link stays above a clearance plane during the generic planned transfer.

## One pickup-and-drop attempt

After confirming the pad, neighboring-ball/jaw clearance, fixed board/bucket and transfer route, press **Pick up → Drop** once:

1. Plan to the selected approach waypoint and settle.
2. Establish recording and open the gripper (skip only when its last panel command acknowledged Open).
3. Request a straight vertical descent to the taught grasp pose.
4. Close and require the driver's held-object acknowledgment.
5. Request a straight vertical lift back to the approach.
6. Plan to the unchanged saved drop pose, settle for two seconds, check recording/readback and release.

Each vertical segment uses Viam `Motion.Move` with a typed `LinearConstraint`: requested line tolerance2mm and orientation tolerance1°. The start must be fresh, stationary and at the expected full pose. A rejected constrained path aborts; it never retries unconstrained. These bounds constrain the arm endpoint, not every link, and do not prove physical finger, neighboring-ball or table clearance. First physical use needs supervised route validation with the actual mounted tool. Motion speed remains the existing hardware/planner configuration.

Stop retains its existing behavior: cancel the pending routine/planner operation and stop only the dropper, without opening either tool or changing board suction. There is no automatic repeat. Practice mode saves completed attempts unlabeled; outcome remains unknown until supplied by an operator or supported sensing.

## Evidence and boundaries

Trial records retain selected ball identity, grasp/approach/drop poses, calibration hash, constrained-move request/result, timestamps and existing recording/release evidence. Ball identity and pad clearance are operator supplied; a gripper acknowledgment is not visual identity verification. Camera and joint RPCs are not hardware synchronized. Board constancy uses the existing joint reference; bucket placement remains operator confirmed.

Audio hooks remain nonblocking and retain trial/ball IDs. Measured ODD impacts play the selected generic bounce voice; they do not establish board/bucket contact or capture. Teaching and source synchronization do not actuate tools or move arms. Existing simulation geometry, scoring, policies and historical results are unchanged.

## Mock checks

Run the physical Python suite and show/physical browser-unit tests with the project interpreter:

```sh
python -m unittest discover -s sircus/physical_trials -p 'test_*.py'
node --test sircus/simulation/tests/test_show_panel.mjs sircus/simulation/tests/test_physical_panel.mjs
```

Coverage includes independent teaching, failed grip, stale/mismatched start, missing recording, moved board, constraint rejection, Stop during descent and a late canceled response. Mock results establish control behavior only. One operator-authorized four-station physical run subsequently completed on October 30; native planning accepted the constrained segments and all four releases were recorded. This is not continuous clearance measurement or verified bucket capture. See the physical STATUS record.

## Explicit camera-off demo

Taped pickup offers an unchecked **No-camera demo · robot-state log only** option.
It requires unlabeled practice mode. The API accepts `recording_mode: operator_log_only`
only on `pickup_drop_cycle` with `practice_mode: true`; omission keeps camera recording
mandatory. This mode never starts camera recording or claims recording readiness.
Trial records retain the mode, command timestamps, discrete pickup/release robot states,
and an unknown outcome. It is not a visual capture dataset.

Fresh two-arm readback, fixed board joints, taught poses, constrained vertical pickup,
held-ball acknowledgment, two-second release settling, cancellation and Stop checks
remain mandatory. Audio remains nonblocking. Each button press runs one ball only.
