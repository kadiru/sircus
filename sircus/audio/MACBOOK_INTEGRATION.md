# MacBook handoff — shared Sircus audio

Ghost Booth now lives in **this repository's `sircus/audio/booth/`**. The Mac Mini
owns audio development/playback; the MacBook dashboard and Ubuntu simulation
remain independent. Pulling code does not deploy a dashboard or start a robot.

## Review without hardware

In a clean checkout/worktree, fetch `codex/mac-mini-oddball-sfx` and run:

```sh
python3.12 -m unittest discover -s sircus/audio/tests -p 'test_*.py' -q
python3.12 -m unittest discover -s sircus/audio/booth/tests -p 'test_*.py' -q
python3.12 -m sircus.audio booth
```

On a Mac with Command Line Tools, open http://127.0.0.1:8766/. FX starts off.
Pick a profile, edit/save sound choices, then explicitly enable FX to review.
The audio engine runs on the Mac hosting the service, not on the browser device.
No private Logic project or credentials need copying.

Saved record conversion remains pure/read-only. Keep records and generated traces
in ignored `.runtime/`:

```sh
python3.12 -m sircus.audio convert .runtime/sim-export.json \
  --source simulation --trial-id review-sim-001 --output .runtime/sim-events.jsonl
python3.12 -m sircus.audio convert .runtime/saved-hardware-trial.json \
  --source operator --output .runtime/label-events.jsonl
```

After explicitly enabling FX, choose the JSONL file in Ghost Booth's replay
control. Known simulated outcomes are predictions; confirmed saved operator
labels retain their provenance; unsure/unlabelled records end silently. Conversion
does not add contacts omitted by the model or invent pickup/flight timings.
The built-in whole-sequence demonstration is separately labelled synthetic.

## Future live attachment (not installed)

Use the optional server-side client and private-LAN instructions in
[booth/integration/HANDOFF.md](booth/integration/HANDOFF.md). Localhost is the
default; cross-computer transport has not been enabled or tested. Producers
should enqueue immutable `sircus.audio.v1` events without awaiting audio. Never
wait for sound before moving/stopping a robot. Use stable trial/event IDs, keep
simulation and physical evidence distinct, and explicitly cancel on stop,
replacement, navigation, pause, speed change or scrub.

| Producer | Reviewed attachment point for later work |
|---|---|
| `two-arm/physics.py` | `simulation_events` converts a completed saved result with its simulation clock; no fabricated bounces. |
| `two-arm/viam_module.py` | Observe lifecycle separately. Gripper-open/motion commands do not prove impact, successful pickup or capture. Failed native execution's nested prediction is not physical evidence. |
| `simulation/web/app.js` | Later opt-in audio/replay controls; the current browser and Mac Mini clocks are not synchronized. |
| `physical_trials/controller.py` | Saved hardware `operator_outcome`; unsure/unlabelled must stay unknown. |
| `simulation/web/physical.js` | Later operator audio proxy; audio failures must never block physical controls. |
| ODD CoreMIDI | Ghost Booth already selects one physical input and filters configured impact notes. This does not identify contact objects or capture/miss. |

No robot/controller/viewer hooks, cloud identity reuse, Viam configuration change,
physical sensing validation or dashboard deployment is included in this change.
The six-cue procedural Logic route remains available in [LEGACY_MIDI.md](LEGACY_MIDI.md)
for advanced diagnostics; use Ghost Booth for the unified event sound library.
