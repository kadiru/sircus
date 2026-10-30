> Historical milestone-one procedural MIDI route. Use [Ghost Booth](README.md) for current sound design, profiles and live ODD input.

# Sircus sound-effects adapter — milestone 1

Optional, isolated audio driven by explicit evidence. No robot SDK, controller hook,
network listener, dashboard deployment or live event bus is introduced here.
Python imports make no sound. Playback is an operator action.

The first observed board contact plays **boing**; subsequent contact records select
**squeak/plop/spring** deterministically by event ID. Known simulated outcomes play
celebration or three descending **wongs**, always labeled predictions. Physical
outcome cues require explicit confirmation with provenance. Unknown/unlabeled
outcomes have no terminal sound. An ODD tap has unknown contact object, so it plays
a generic bounce, never claims a board impact or capture.

## Try the local preview (no Logic or dependencies)

From the repository root, using Python 3.12:

```sh
python3.12 -m sircus.audio demo
```

Open `.runtime/sfx-preview.html` in a browser and click a cue or synthetic scenario.
It is a self-contained file: no server, network calls, samples, autoplay or robot
connection. Stop cancels scheduled tones; mute discards playback. The level slider
controls the browser preview. It uses the same MIDI score plan with a simple
triangle oscillator, so its timbre differs from Logic's patch.

The command also writes `captured.jsonl`, `missed.jsonl`, and `unknown.jsonl` in
`.runtime/`. These are **hand-authored synthetic demonstrations**, not measured
trials or output from the physics engine. Regenerating the demo overwrites these
generated files only. A dry run shows the exact MIDI plan without opening a port:

```sh
python3.12 -m sircus.audio replay .runtime/captured.jsonl
```

## Play through Logic on a Mac

Requirements: Python 3.12, installed Apple Command Line Tools/CoreMIDI, Logic Pro
with its stock **Scripter** MIDI effect and **Retro Synth** instrument. No external
plugins, sound packs, samples, Python MIDI packages, ODD phone app or Viam credentials
are needed for replay. Browser preview also works without Logic.

```sh
mkdir -p .runtime
xcrun swiftc -module-cache-path .runtime/swift-cache sircus/audio/coremidi.swift -o .runtime/sircus-midi
.runtime/sircus-midi --check
```

`--check` creates then disposes a virtual source; it sends no MIDI. Run one helper
at a time. The source has a stable CoreMIDI ID, and collisions fail closed.

1. Create a separate, empty software-instrument project. Keep Stereo Out muted
   while configuring; start at −18 dB with a limiter. Leave Play stopped.
2. Put **Scripter → Retro Synth** on one track. Paste `logic-sfx.js` into Scripter's
   editor and click **Run Script**. Confirm **Input channel = 16**, **Output = 1**,
   **Mute and release = Off**. Logic can retain parameter values from a previously
   loaded script; do not assume the new defaults replaced them.
3. Use Retro Synth's **Dynamic Soft Pad** as a starting timbre. In Controls view,
   set **Pitch Bend Range = 1 Oct** (±12 semitones), Amp Env Attack about **8 ms**,
   Amp Env Release **80 ms**, and Glide/Autobend **Off**. Keep resonance low and
   distortion/ChromaGlow bypassed. These are initial review settings, not an
   acoustically approved final patch.
4. Start the operator command below. During its 30-second silent setup interval,
   select **MIDI Input = Sircus SFX** on this track, with input monitoring **I on**
   and record enable **R off**. Other instrument tracks should not monitor this
   port. Never choose All or either ODD input for the cue receiver.
5. Unmute Stereo Out when ready to listen. Logic's transport stays stopped;
   the CLI schedules notes. **Ctrl-C in the CLI** cancels replay. Logic's Stop
   button alone does not stop the external CLI's clock.

```sh
python3.12 -m sircus.audio replay .runtime/captured.jsonl \
  --play --midi-helper .runtime/sircus-midi --setup-seconds 30
# Repeat explicitly with missed.jsonl or unknown.jsonl.
```

The port disappears when the process exits. Recheck its selection on reconnect;
Logic's port restoration across host versions has not been validated. For more
setup time use `--setup-seconds 120`. Ctrl-C also cancels the setup interval.

The receiver reuses the Mac Mini's tested logging-free note ownership and Reset
approach. It releases on retrigger/parameter change, filters input channels and
unrelated CCs, caps velocity at 90, passes bounded pitch bend, and releases missing
note-offs after 2.5 seconds via ProcessMIDI. It has no Trace calls and no delayed
Scripter note queues. The adapter owns timing and sends release/panic messages.
The stock synth's short release may still ring briefly after cancellation.

## Configuration and API

Copy `mapping.example.json` into `.runtime/mapping.json`, edit it, then pass
`--config .runtime/mapping.json` to `demo` or `replay`. Supported cue names are
`boing`, `squeak`, `plop`, `spring`, `celebration`, `wong`. Gain is 0–1; mute is a
boolean. Default MIDI channel is 16; match the Scripter input if you change it.
`max_events` bounds per-trial deduplication; `max_pending_messages` bounds the
combined event/message queue. Overflow drops sound, never delays robot work.

```python
from sircus.audio import AudioAdapter

# Construct a sink/adapter in a separate audio observer, outside robot callbacks.
# sink.send([status, data1, data2]) is the only required output interface.
audio = AudioAdapter(sink)
audio.start_trial(trial_id, mode="physical")  # explicit operator opt-in
audio.submit(event)                         # best-effort bool, no output I/O here
audio.set_gain(0.4)
audio.set_muted(True)                       # clears pending cues; no unmute backlog
audio.cancel(trial_id)                      # call on trial/replay stop or navigation
# On observer shutdown, away from the control path:
audio.close()
```

One active trial/domain per adapter. Simulated input is rejected in physical mode
and vice versa. Deduplication keys are `(source, event_id)` for the whole active
trial; IDs are not evicted to make room. `start_trial` is an intentional new replay
generation and permits replaying the same trace again. Polling must not call it on
every refresh. An outcome ends cue acceptance for that trial; later impacts are
silent. IDs are immutable observations; corrected labels need new IDs and an
explicit new audition if the previous outcome already played.

`submit` takes an optional **host-relative** `delay_s`; live events default to now.
It never subtracts producer timestamps across clocks. The output worker performs
all sink calls outside the state lock. Failure increments `errors/last_error`,
drops audio and does not retry ambiguous sends. A hung sink cannot hold the
submission/cancellation lock; cleanup waits at most 100 ms for the worker.
Initialize/dispose sinks in the audio process, never in the robot control path.
The supplied MIDI sink writes to a nonblocking local pipe. This is best-effort
Python scheduling, not a hard-real-time guarantee or a robot safety interlock.

Each new cue releases and preempts the previous monophonic gesture, preventing
older pitch bends from corrupting a later bounce. Stop clears pending generations
and asks the worker to send CC120/123 and bend center. An already-in-flight send
may finish before panic. If MIDI/Logic itself is unavailable, immediate silence
cannot be guaranteed; the receiver watchdog and short envelope limit lost offs.

## Evidence and boundaries

See [EVENT_CONTRACT.md](EVENT_CONTRACT.md) for fields and clock semantics,
[ODD_TRANSPORT.md](ODD_TRANSPORT.md) for measured versus unknown ODD behavior, and
[MACBOOK_INTEGRATION.md](MACBOOK_INTEGRATION.md) for the exact offline handoff and
future explicit observer attachment points.

Validated on the Mac Mini, October 29, 2026:

- Python adapter tests and fake-output deterministic traces; complete repository
  mock/numerical suite. Tests do not command physical hardware.
- Node mocked Scripter checks and generated-preview checks with fake Web Audio.
- Native helper compiled with the installed Swift toolchain; `--check` created
  the CoreMIDI source, and the silent source appeared as **Sircus SFX** in Logic.
- Logic Pro 12.3 build 6674 on macOS 26.5.2 evaluated the new receiver successfully.
  Input 16/output 1, isolated port, I on/R off, octave bend, 8.10 ms attack and
  80 ms release were verified in a private review copy. Stereo Out stayed muted
  at −18 dB. The copy was saved and Logic was closed.
- **No new audio was auditioned, no physical ODD trace was captured, and no
  end-to-end cue timing/acoustic quality was measured in this milestone.** The
  generated preview could not be opened automatically because the browser tool
  disallowed local file URLs; its script was tested with fake Web Audio instead.
  Last night's audible two-ball success is historical evidence, not a test of
  this new cue adapter. Original Logic project/script checksums were unchanged.

The review `.logicx` is host-local and excluded from Git. Rebuild with the source
instructions above on another Mac. Older Logic Pro X versions are unverified.

```sh
python3.12 -m unittest discover -s sircus/audio/tests -p 'test_*.py' -v
node sircus/audio/tests/logic-sfx.test.cjs
python3.12 -m sircus.audio demo
node sircus/audio/tests/preview.test.cjs .runtime/sfx-preview.html
# Full suite requires the repository's pinned simulation dependencies:
SIRCUS_PYTHON="$PWD/.venv/bin/python" scripts/test.sh
```
