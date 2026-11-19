# Two-round operator panel

Open `/show.html` → Manual operation → **Run the complete act**. The backend must
be the explicitly launched hardware dashboard, with cameras disabled. Source sync
alone does not launch or deploy it.

1. Park the empty dropper at either taught round drop pose; the board must also
   be at either taught round pose. Load the four pickup marks. Check both arm
   paths, neighboring balls, board seating and bucket placement.
2. Check the loading/clearance confirmation and choose whether to use sounds.
   Press **Run Round 1 · four balls**. The board moves to Position 1 first;
   Gray → Purple → Pink → Green then use the existing constrained 100 mm
   approach, vertical descent, grip acknowledgment, vertical lift, native transfer,
   two-second settled release and two-second post-release wait.
3. Round 1 ends in `awaiting_reload`. Reload all four marks, confirm again, then
   press **Reloaded · run Round 2**. This is a separate operator command, never
   automatic. The second button can also explicitly run Round 2 on its own.
4. Each completed cycle remains locally saved and unlabeled. Completion is not
   evidence of bucket capture. No camera preview/recording is requested by the
   camera-disabled hardware launch.

Closing the browser does **not** stop an active round. The serialized backend
owns the sequence and persists progress; the browser displays it and sends
explicit operator commands. Keep the page open and use **Stop demo motion**.
Stop cancels the owned planner and stops the dropper plus an owned board
transition; it never opens either tool or changes suction. Ordinary single-arm
operations retain dropper-only Stop. External controllers are not owned by this
panel. Restarting an interrupted round never resumes it automatically.

## Interrupted trials

No automatic retry is performed, and no completed pickup is repeated. Read the
current trial and confirm the real held-ball state before recovery. Existing
**Verify stationary recovery** and **Finish held ball → saved drop** controls
retain the original trial ID and require a measured held-ball sensor, matching
approach/clearance position, unchanged calibration and no prior release command.
For an empty gripper after an interrupted approach, keep the stopped trial
unlabeled, select that ball under Taped pickup station, reconfirm the corridor and
run only that pickup. Full-round buttons always mean all four marks are loaded;
they are not retry-remaining buttons.

## Local configuration and evidence

The existing host-only `round-presets.pending.json` in the physical data root
supplies two complete world-frame arm poses and taught joint solutions. Its
historical `execution_enabled:false` flag does not start anything; an explicit
`run_round` request with fresh action epoch and operator confirmation selects a
snapshot of the preset. The historical name/flags are retained rather than
rewriting old teaching evidence. No poses or credentials are embedded in source.

The controller checks both stationary arms, an empty dropper, quiescent server
motion operations and live vacuum holding before transition. It verifies the
board pose and joint solution, unchanged dropper pose and vacuum afterward.
Each run stores its preset snapshot/hash and unique trial IDs under
`round-runs/<id>/run.json`; trial records also include the round ID and hash.
All these files stay local and ignored by Git.

Audio controls enqueue locally and do not wait for transport/readiness. The
round's sounds option enables FX during ball cycles and requests mute on finish,
failure or Stop. Volume remains an operator setting. Measured ODD impacts retain
their existing identity/provenance; no sound establishes capture or surface type.

## Green/readback diagnosis

Green is the fourth and farthest pickup; that is a correlation, not a diagnosis.
It completed several runs successfully. One interruption followed grip/lift;
the later interruption occurred at approach before any gripper command. The
latter trial contains approach and Stop events, not a failed grip or sound event.

The former chat-side orchestration aborted as soon as cached dashboard state
reported unavailable. The new backend sequence uses the controller's bounded
state reads and native command completion directly instead of that extra polling
abort path. Actual readback failures still stop/lock the routine; deadlines,
settling, board and release guards were not relaxed. This does not establish
that the underlying intermittent transport problem is fixed.

Per-resource RPC elapsed times/status and the latest failing method are now
exposed under Readback timing diagnostics. These contain method/error type,
not SDK exception strings or credentials. Distinguishing far-position effects,
transport latency and fourth-in-sequence duration needs further physical evidence.

## Validation boundary

156 physical-controller/adapter mock tests, 42 UI tests and 11 startup checks
passed. New tests cover four unique trial IDs per round, explicit reload pause,
Stop during board transition, cancellation ownership, failed fourth pickup,
changed board joint solution, missing/invalid confirmation, low/held start,
missing vacuum acknowledgment, unavailable audio and restart without resume.
The two-round browser rehearsal uses only in-memory mock arms. Hardware execution
of the new coordinator remains an operator commissioning step; prior physical
runs used the earlier host-local scripts and are not relabeled as this UI's tests.

Deployment verification: the idle physical dashboard was refreshed to this branch
on port8765 with cameras disabled. Read-only checks showed both arms stationary,
no fault, both round presets available, and FX off with the prior100% volume
restored. Ghost Booth was not restarted. The browser mock completed4/4 in each
round with an explicit reload pause; no physical motion or tool command was sent.
The live UI currently reports3/4 ball inputs connected; wake/reconnect all four
before an audible demonstration. This does not gate robot motion.
