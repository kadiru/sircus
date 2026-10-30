# Saved-pose hardware demo: MacBook control and sound

This checkpoint prepares the smallest physical show: the MacBook runs the
sfhack10 dashboard and Ghost Booth native sound. **Logic Pro and the Mac mini
are optional.** Nothing here starts on pull, reboot or source synchronization.
Hardware startup and commissioning remain explicit operator actions.

## What the routine does

Select Pink, Purple, Green or Gray, confirm the fixed setup, and press
**Load → Drop** once. Arm10 goes to its saved loading TCP pose, opens only when
needed, waits two seconds, closes and checks the grip, moves to the saved drop
TCP pose, settles for two seconds, and releases with recording ready. Another
press is required for another ball. Practice retains completed records with an
unknown outcome; successful execution does not mean the bucket captured it.

Arm14, its suction, the board and bucket stay fixed. No perception or learned
policy is required. Optional height controls remain locked until their separate
height reference and limits are calibrated. The fixed routine uses exact saved
targets rather than inventing a ball-to-board height.

FX starts off. A bounded drum roll follows carrying and settling and stops before
release. After release, verified positive MIDI NoteOn events from each explicitly
bound ball play its generic bounce sound for the configured impact window
(default30seconds). Earlier balls retain their own trial identity while another
color loads. Taps do not identify a board/bucket contact or prove capture/miss.
Unknown outcomes stay silent. Motion, release and Stop never await audio.

## Offline preparation

Copy `sircus/physical_trials/hardware-demo.profile.example.json` into an ignored
host directory, with mode0600. Set explicit absolute paths to the reviewed
Python3.12 venv, existing sfhack10 credentials, existing `data/hardware` directory
and a separate show-state directory. Never copy mock records over hardware data.
The host credentials use `address`, `api_key_id`, and `api_key`; no secret is
included in this repository.

For same-Mac sound, create a separate mode0600 JSON file named in
`audio_env_file`, containing backend environment fields:

```json
{
  "SIRCUS_AUDIO_URL": "http://127.0.0.1:8766",
  "SIRCUS_AUDIO_TOKEN": "",
  "SIRCUS_AUDIO_NODE": "/absolute/path/to/node"
}
```

The Node field is optional if Node is on PATH. These are backend-only values;
never paste a private-LAN token into browser configuration or commit it. The
first demo uses loopback and `audio_mode: "local"`. Mini mode needs an explicit
private IPv4 address and token, plus separate network commissioning.

From this reviewed checkout:

```bash
./start-hardware-demo.sh --check --profile /absolute/private/profile.json
```

Omitting `--check` also only checks. This reads local configuration and installed
package metadata; it neither imports the robot SDK nor connects, moves, plays,
starts a service, rewrites calibration or recovers trial journals. A passing
report means files are prepared, not that live operation is validated.

## Future operator startup

Use these commands only when ready to commission the setup. First ensure there
is one owner of each physical arm, including any older dashboard, another host
or sfhack14 runtime that references the board arm. The new local lease catches
cooperating same-user dashboards across ports/worktrees; it cannot arbitrate
remote machines or older software.

1. Start Ghost Booth on the MacBook, in a separate terminal:

   ```bash
   SIRCUS_PYTHON=/absolute/project/venv/bin/python ./sircus/audio/booth/launch.sh
   ```

   It uses loopback port8766 by default. It compiles its native helper when
   needed; starting it opens the sound/MIDI service. No automatic audition is
   part of this preparation task. Leave FX off until the intended output and
   input identities are checked.

2. Explicitly start the physical dashboard:

   ```bash
   ./start-hardware-demo.sh --launch --profile /absolute/private/profile.json
   ```

   It starts readback on sfhack10 and opens no gripper or suction tool. The
   profile chooses port8765; open `http://127.0.0.1:8765/show.html`. Manual mode is
   selected and numerical modes are disabled on this hardware server. Existing
   mock previews on18770/18772 are separate. `--offline` in the generated argv
   disables the cloud **simulation** bridge, not the explicitly enabled physical
   adapter. HTTP binding and the exclusive local lease precede SDK/client setup.
   Invalid/unavailable audio stays optional; it cannot block robot startup.

3. Check the live camera and both arms' fresh stationary state. Confirm the
   saved TCP targets, board reference and actual clear route against the setup.
   A plausible image overlay or saved pose does not prove clearance. Re-teach
   through the operator UI if the layout has changed. Saving pose values does
   not command motion. Board/suction placement remains an operator responsibility.

4. Pair the ODD balls to this MacBook. Inspect each one separately in the Booth
   Monitor and record its actual endpoint ID, channel and positive NoteOn note.
   Enter these in **Ball identities & sounds**, then save. Do not guess bindings
   from color or assume the Mini's endpoint IDs remain valid on this Mac.
   The header separates saved bindings, connected inputs and native-engine
   readiness. None alone proves audible sound.

5. If Booth restarted or input selection changed, press **Reapply saved audio
   mappings** while idle. This mutes sound and retires old ball associations.
   Enable FX explicitly again. Verify each ball with a listening check using a
   valid release/test association; the show routes measured bounces only after
   that ball's release. A standalone Booth audition is sound-only evidence.
   Check volume, distinct colors, overlapping older-ball impacts, and FX mute.

6. Run a supervised one-ball cycle; check grip, recording before release, the
   two-second waits and Stop. Stop cancels pending actions and stops arm10; it
   does not open the gripper or release board suction. Repeat only by another
   button press after verifying the fixed setup. Verify physical Stop separately
   before shutting down a hardware session: process cleanup is not a safety Stop.

## This checkpoint's evidence and limits

- Existing host calibration and completed trial records were preserved. Both
  fixed targets and six board reference joints passed offline validation;
  variable-height calibration is still absent and unused by the fixed routine.
- A private MacBook profile points at the original host files. Credentials and
  backend audio settings remain ignored. The native Swift sound helper compiled
  without executing it, opening MIDI devices or playing audio.
- Tests use mocks, static UI fixtures and a temporary fake loopback audio service.
  They cover unavailable audio, duplicate/delayed events, stale mappings after
  restart, cancellation, lease contention, startup failure, and fixed-cycle
  recording/Stop behavior. These are software checks, not acoustic or robot tests.
- No physical connection, movement, tool actuation, cloud change, native audio
  service start or existing-service restart was performed. The four physical
  MIDI bindings, live speaker output, route clearance and capture rate remain
  unverified.
- Shared board/bucket geometry, provisional175mm bucket height, historical scene
  fingerprints, numerical results and native-replay guards are unchanged.

See [show runbook](SHOW_RUNBOOK.md), [deterministic acceptance](DETERMINISTIC_RELIABILITY.md)
and [audio integration](../sircus/audio/MACBOOK_INTEGRATION.md).
