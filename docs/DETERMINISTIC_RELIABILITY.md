# Deterministic routine reliability

This checkpoint hardens **Manual operation**: saved loading TCP → open if needed →
two-second loading countdown → close and verify the driver's held-object report →
saved drop TCP → two seconds settled → release → two seconds of recording. Each
repeat needs another operator press. Arm14, board, bucket, release XY and tool
orientation stay fixed. No perception or learned policy is required for this mode.

Software verification is mock-only. A completed mock routine is **not** a bucket
capture, a validated physical path, or proof the real gripper is holding a ball.
The real driver's held-object indication also needs physical commissioning.

The controller now invalidates held-ball belief after connection/readback loss,
checks that fresh recording belongs to the current trial, and keeps recovery
locked while an owned actuator request or planner operation remains unresolved.
Stop still reaches the dropper when journal writes fail. A delayed command from
before Stop is rejected by the action epoch; old replies cannot unlock the UI.
Pose edits are retained until a matching saved readback confirms them.

Recording freshness includes collection and persistence delay. Cancellation cannot
redirect an old observation write into the next trial's directory. If an optional
native capture start is sent but never acknowledged, its ownership remains
**unknown** and requires operator inspection; the software does not claim cleanup
or send a blanket stop to potentially foreign capture activity.

## Verified checkpoint — October 30, 2026

The final source completed 20/20 mock cycles, five per ball color. Each cycle
rejected an overlapping press and retained a unique unlabeled record. All 560
mock frames and paired arm observations were saved; 623 evidence-file checksums
and the runner/controller/mock-adapter source hashes verified. Minimum loading
wait was 2.000s; minimum drop settling was 2.126s; median mock cycle was 8.657s.
No spontaneous repeat occurred. No bucket capture rate was measured.

The full suite passed 368 Python tests and 92 Node tests plus standalone Logic
guards. Of these, 73 exercise the physical controller/adapter with mocked I/O.
Browser QA verified pose editing, completed/repeated routines, Stop during loading,
unlabeled retention of an aborted trial, reconfirmation and a clean-start cycle.
The task's separate mock preview is on `http://127.0.0.1:18771/show.html`; it is
idle after verification. Earlier runtimes on 18766/18769/18770 remain unchanged.

Detailed generated evidence remains outside Git under the task's ignored
`.runtime/deterministic-reliability/` directory. Use the command below to generate
a new complete report on another checkout. The earlier exploratory endurance run
is retained separately from the final source-matched run.

## Reproduce the endurance check

From the repository root with the project Python environment:

```sh
python -m sircus.physical_trials.validate_deterministic \
  --cycles 20 --output .runtime/deterministic-validation-001
```

The script imports only the controller and mock adapter, takes no credentials and
has no hardware option. It retains the normal timing constants, rotates the four
operator ball IDs, rejects an overlapping second press, verifies exact command
order and release pose, checks saved mock images/both-arm observations and verifies
that no repeat occurs without a new press. It refuses to overwrite an existing
output directory. `report.json` records source hashes and timing; `manifest.json`
checksums the report and generated records. Outcomes remain unknown/unlabeled.

Run the fault/race regressions separately:

```sh
python -m unittest discover -s sircus/physical_trials -p 'test_*.py' -v
node --test sircus/simulation/tests/test_*.mjs
```

Unit tests shorten selected timers and inject failures. The endurance script does
not shorten the loading/settling timers. Neither connects to Viam or the audio
service. Generated records belong in ignored `.runtime/`, not Git or a hardware
calibration directory. Existing historical trial and study results stay unchanged.

## Supervised physical acceptance, when separately authorized

1. Confirm one runtime owns the dropper and board arm, including any `sfhack14`
   reference to arm14. Preserve the host's existing calibration and records.
2. Confirm actual camera/both-arm state and fresh local recording. Inspect the
   saved loading/drop TCP targets, tool orientation and route at the real setup.
   Board attachment geometry is provisional and endpoint poses do not validate
   a route. Fix the board and bucket for the entire run.
3. Rehearse one cycle without a ball. Expect the missing-object check to abort
   before carry/release if the real driver reports no object. Confirm the actual
   driver's behavior; a mock `grabbed=True` is not evidence of real sensing.
4. At loading, use Stop during the countdown and verify no subsequent close or
   carry. With no hand in the workspace, check Stop during motion/settling. Confirm
   the dropper stops while its gripper and board suction retain their state.
   A network error cannot establish that physical Stop succeeded.
5. Run 20 supervised presses, five per color. Clear hands before the two-second
   close. Do not change bucket/board/poses mid-run. Select the actual color and
   retain every trial, including aborted or uncertain attempts.
6. Record two independent measures: **routine completion / attempted presses** and
   **observed bucket captures / observed released balls**. Count unknown outcomes
   separately. Practice mode removes the annotation requirement; it does not infer
   a successful capture. For scored evidence, use explicit human labels/video.

A routine acceptance pass needs 20/20 correctly ordered completed cycles, unique
records and no unrequested repeat or new tool command dispatched after Stop.
Stop cannot undo a tool command already dispatched before it was pressed. Any failure is
retained and investigated before another acceptance run. This proposed threshold
is a short demo qualification, not a statistical reliability guarantee. Physical
capture accuracy is a separate result; do not hide misses behind control success.

## Deployment boundary

Pulling this branch does not reload a service. Use a separate credential-free
preview to review it. The existing physical dashboard, Viam configuration, audio
services and arm14 tools are unchanged by this task. Hardware deployment and the
acceptance run above require a separate operator-controlled step.
