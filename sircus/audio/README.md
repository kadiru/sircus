# Sircus audio — Ghost Booth

**Ghost Booth is the main sound-design and rehearsal interface for Sircus.**
It is an independently runnable subsystem of this repository, owned by the Mac
Mini audio work. It requires no robot, dashboard, Logic project or cloud account.
The earlier ODD drum/synth Logic sessions remain private musical experiments.

## Start on the Mac

Python 3.12 and Apple Command Line Tools are required. No Python audio packages,
purchased plugins or sound libraries are needed.

```sh
python3.12 -m sircus.audio booth
# Equivalent: sircus/audio/booth/launch.sh
```

Open **http://127.0.0.1:8766/**. It starts with **FX off**, no selected MIDI input,
and the last saved custom mix (or Ghost Booth defaults). Sound comes from this
Mac's default audio output. The browser is the controller, not the audio engine.
Run only one instance. Ctrl-C stops the local audio service.

1. Load **Ghost Booth**, **Cartoon Circus**, **Minimal**, or a saved mix.
2. Select an event in the left column. Assign any of the sixteen sounds, a pool
   that cycles/randomizes, or none. Set event volume and pitch. Fixed uses the
   first selected sound; deselect/reselect to change the selection order.
3. **Apply changes** updates live assignments. **Save mix** writes a named local
   profile; saving also applies it. Built-ins cannot be overwritten. Applying or
   loading without saving does not change which custom mix restores on restart.
4. When ready to listen, enable FX and use **Preview event**, a sound tile, or
   **Play synthetic sequence**. Preview applies pending edits first. The synthetic
   sequence is hand-authored audio rehearsal, not a physics or hardware result.
5. For a real ball, choose its unique MIDI input, explicitly enable FX again, and
   tap it. Free play uses **Bounce / impact** for each tap. In trial mode, use
   **Release / arm live trial** before tapping. After a replay, click **Return to
   live ball**. Input selection/reconnection forces FX off.

| Event slot | Trigger / interpretation |
|---|---|
| Picked up | Explicit pickup action/observation |
| Carried by arm | Carry start; loop stops on carry stop, release, impact or cancellation |
| Released / dropped | Explicit release; arms the configured live impact window |
| Travelling through air | Flight start; loop stops on flight stop, impact, outcome or cancellation |
| First impact after release | First armed tap is a sound heuristic; a trace's first explicit board contact also selects this slot |
| Bounce / impact | Later trial impacts; all free-play taps |
| Captured | Known simulated result or explicitly confirmed physical outcome |
| Missed | Known simulated result or explicitly confirmed physical outcome |

Any catalog sound can be assigned to any slot. Only carry/flight allow looping,
with an adjustable hard limit of 0.2–20 seconds. Event pitch is ±12 semitones;
native varispeed also changes sample duration. Random selection can avoid
immediate repeats. Volume zero or an empty sound pool is silent.

**Stop playback** cancels scheduled playback/loops and retires the current event
trial; free-play ball taps may still play afterward. **Mute all audio** disables
FX, silences voices and clears queued playback. These are audio controls only.
No robot control, Viam connection or automatic capture detector is installed.
ODD MIDI currently provides impact gestures; pickup/flight/outcomes need explicit
operators, saved traces, or a separately validated event producer.

## Check all four balls before a round

Open **Ghost Booth → Four-ball tap test**, choose the duration (default **15 s**,
1–120 whole seconds), and click **Start tap test**. It explicitly enables audio
and listens to the existing four endpoint/channel/note bindings. Tap each ball:
the cards show connection, accepted tap count, last-tap age and its assigned
unknown-contact bounce sound. Listen to confirm output; a MIDI count alone does
not prove audible sound. Disconnected and connected-but-no-tap are separate.

The server mutes when the timer expires, including if this tab closes. **End test
& mute**, FX off, reconnect, mapping changes and native errors cancel the test.
Finish the check before starting the dashboard round; round FX is enabled there
separately. The tester does not connect/disconnect BLE devices: reconnect missing
balls in macOS Bluetooth MIDI, then explicitly start another test. Existing live
trial windows are not extended and no synthetic release is fabricated. Incoming
round events replace test playback; an old timer cannot mute the newer round.
No arm, tool, camera, Viam configuration or robot controller is accessed.

## Shared implementation

- `contract.py` and `sources.py`: versioned evidence/clock-aware events; pure
  converters for saved simulation results and operator labels.
- `booth/`: local HTTP controller, selected CoreMIDI input, eight native playback
  voices, original WAV catalog, event profiles and the browser interface.
- [Event contract](EVENT_CONTRACT.md), [ODD evidence](ODD_TRANSPORT.md),
  [MacBook handoff](MACBOOK_INTEGRATION.md), [HTTP/profile API](booth/README.md).
- [Legacy procedural MIDI/Logic adapter](LEGACY_MIDI.md): retained for offline
  diagnostics and the private muted Logic review project. Its six-cue HTML export
  is not the primary design interface and does not read Ghost Booth profiles.

Saved profiles, recordings, compiler caches and tokens stay in ignored
`booth/runtime/`. Generated redistributable catalog assets are checked in;
recorded performance audio and Logic projects are not. No audio/network service
starts merely by importing the package or pulling the branch.

## Checks and remaining review

```sh
python3.12 -m unittest discover -s sircus/audio/tests -p 'test_*.py' -q
python3.12 -m unittest discover -s sircus/audio/booth/tests -p 'test_*.py' -q
node sircus/audio/booth/tests/test_client.mjs
node sircus/audio/tests/logic-sfx.test.cjs
SIRCUS_PYTHON="$PWD/.venv/bin/python" scripts/test.sh
```

Consolidation checks on October 29–30, 2026 cover event/profile behavior, persistence,
invalid-input rejection, cancellation, domain isolation, generated sample levels,
HTTP policy and a nonblocking fake-network client. Native Swift compilation and
silent CoreAudio readiness passed. Browser editing/saving was tested with FX off.
The original Logic files were preserved. The user found the earlier rising whoosh
too noisy; its new tonal hum replacement and the added energy sweep await listening
review. See [the synthesis recipe and Logic options](booth/logic/SOUND_DESIGN.md).
Wireless latency/loss,
BLE interruption and MacBook-to-Mac-Mini delivery remain unverified. No new
physical robot or simulation run was performed.
