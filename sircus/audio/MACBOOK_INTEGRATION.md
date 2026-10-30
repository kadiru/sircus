# MacBook handoff: offline first, explicit observer later

This milestone changes no controller, physical panel, simulation viewer, physics,
scene, calibration or Viam configuration. Fetching code does not deploy anything.
No cloud machine identity, secrets or private Logic project should be transferred.

## Exact offline adoption steps

1. In a separate clean MacBook clone, fetch and inspect this branch. If the active
   checkout has unrelated edits, leave it alone and use a worktree:

   ```sh
   git fetch origin
   git worktree add --detach ../sircus-audio-review origin/codex/mac-mini-oddball-sfx
   cd ../sircus-audio-review
   git config core.hooksPath .githooks
   python3.12 -m unittest discover -s sircus/audio/tests -p 'test_*.py' -v
   python3.12 -m sircus.audio demo
   ```

2. Open `.runtime/sfx-preview.html` manually. Review capture/miss/unknown scenarios;
   verify Stop and mute. No robot or dashboard needs to run. For Logic, build the
   helper and create the separate cue track using [README.md](README.md). Browser
   review requires no Python package installation; full-repository tests do.

3. Use **copies of already saved records**, kept in `.runtime/`. A simulation
   viewer export has its physics result under `result`; a raw physics result is
   also accepted. Assign one stable trial ID for that saved result:

   ```sh
   python3.12 -m sircus.audio convert .runtime/sim-export.json \
     --source simulation --trial-id review-sim-001 --output .runtime/sim-events.jsonl
   python3.12 -m sircus.audio replay .runtime/sim-events.jsonl
   ```

   A physical saved trial with an existing operator label uses its own ID:

   ```sh
   python3.12 -m sircus.audio convert .runtime/saved-trial.json \
     --source operator --output .runtime/label-events.jsonl
   python3.12 -m sircus.audio replay .runtime/label-events.jsonl
   ```

   This only reads records. It does not save/change labels, contact the dashboard
   or move hardware. An unlabeled hardware record produces a silent unknown
   outcome. Do not pass a mock/dry-run record as hardware evidence.

4. When the operator wants sound, append `--play --midi-helper .runtime/sircus-midi`
   to a replay command. This auditions the saved record locally; it does not
   synchronize with a running physical/simulation trial. Ctrl-C stops this audio
   process. Leave the existing MacBook dashboard and simulation processes alone.

## Future live integration seam (not installed by this change)

An optional dedicated audio observer must own the sink, worker, mute control and
playback generation. Keep startup, record conversion, filesystem/network work and
cleanup out of robot callbacks. Producers should deliver bounded immutable events
through a best-effort queue to this observer; dropping audio is acceptable. Never
await audio acknowledgment before arm movement or emergency stop.

| Existing component | Explicit attachment point for a later reviewed change |
|---|---|
| `two-arm/physics.py` | Convert a completed result using `simulation_events`; retain simulation elapsed clock and predicted provenance. Do not edit the model or fabricate omitted bounces. |
| `two-arm/viam_module.py` | Observe the replay/trial lifecycle separately. Arm motion or gripper-open records are not contacts. A failed native replay's nested prediction is not evidence of completed execution. |
| `simulation/web/app.js` | Add an operator audio-enable control only in a later UI change. Start one audio generation per replay; cancel it on stop, pause, scrub, navigation, speed change or replacement. Existing browser and CLI clocks are not synchronized. |
| `physical_trials/controller.py` | A completed `save_trial` annotation can feed `operator_outcome`; `unsure`/unlabeled stays unknown. `keep_unlabeled` must not trigger miss. No new capture detector is assumed. |
| `simulation/web/physical.js` | A later operator opt-in can request audio observation, but must never gate physical Stop or any control request on audio. Cancel by trial ID to avoid stopping another generation. |
| ODD CoreMIDI listener | Not implemented here. Isolate a verified physical port, filter positive ch1/note0, supply host arrival clock and stable per-observation IDs to `odd_tap`. Keep object/outcome unknown. |

If Logic remains on the Mac Mini while the dashboard runs on the MacBook, use
offline exported JSONL for this milestone. A live cross-host link would need its
own explicit transport, permissions, source authentication, clock mapping,
bounded queues and stop semantics. No OSC/MIDI network port or cloud identity is
silently reused here.

Before claiming physical integration: verify port identity after reconnect, test
lost notes and cancellation at low level, measure BLE/host/audio timing and missed
impacts, validate board/contact detection separately, and establish capture/miss
evidence. These remain experiments, not guarantees from a successful replay.
