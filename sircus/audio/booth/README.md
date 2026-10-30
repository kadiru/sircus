# Ghost Booth implementation and API

User workflow: [audio README](../README.md). Shared event fields and provenance:
[EVENT_CONTRACT.md](../EVENT_CONTRACT.md). Cross-host opt-in:
[integration/HANDOFF.md](integration/HANDOFF.md).

`launch.sh` builds `src/Native.swift` when needed, then starts `src/bridge.py`.
Python 3.12 uses only its standard library. Swift uses Apple AVAudioEngine and
CoreMIDI. Native mode loads sixteen original mono PCM16 48 kHz WAVs into memory.
Seven voices play one-shots; one is reserved for a carry/flight loop. Each voice
has bounded varispeed. Sources peak at approximately −14 dBFS and the master has
an additional 0.5 gain factor. A three-second missing-parent-heartbeat watchdog
mutes the native helper. No diagnostic MIDI is forwarded to an instrument.

## Local JSON API

All mutation bodies are JSON objects, at most 1 MiB. Default binding is loopback.
FX starts off and is not persisted. Host/Origin checks reject cross-site browser
requests. No CORS is enabled. Errors return 400/503; a valid but dropped cue may
return `accepted:false` with its reason. Acknowledgment is not proof of audible
sound or physical event truth.

| Endpoint | Body / purpose |
|---|---|
| `GET /api/status` | Native readiness/meter, selected input, FX, trial/domain, profile and eight event descriptions |
| `GET /api/sounds` | Catalog with sound IDs, note mapping and WAV metadata |
| `POST /api/fx` | `{"enabled":true}`; false cancels/mutes |
| `POST /api/panic` | `{}`; disable FX and mute all voices |
| `POST /api/stop` | `{}` stops current playback; optional `trial_id` retires only that trial and does not stop a different active trial |
| `POST /api/volume` | `{"volume":0.35}`, finite 0–1 |
| `POST /api/input` | `{"id":<listed ID>}` or null; forces FX off |
| `POST /api/mode` | `{"mode":"free"}` or `trial`; cancels replay/arming and returns to live input |
| `POST /api/profile` | `action`: `apply` plus `assignments`, `load` plus `name`, or `save` plus both |
| `POST /api/preview` | `{"event":"flight"}`; explicit sound preview, not a physical trial event |
| `POST /api/event` | A full `sircus.audio.v1` event; arrival-time playback without cross-host clock subtraction |
| `POST /api/replay-events` | `{"events":[<AudioEvent>,…]}`; one trial/domain/clock, 1–4096 events, maximum one hour |
| `POST /api/record` | `{"enabled":true}` / false; host-local raw selected MIDI recording |
| `POST /api/replay` | `{"events":[{"t":0,"status":144,"data":[0,90]}]}`; raw MIDI rehearsal, at most 10000 rows / 600 seconds |
| `POST /api/mapping` | Subset of impact_note, channel, min_velocity, exact_duplicate_ms, trial_window_s, velocity_gamma |
| `GET /api/monitor?after=123` | Bounded 600-row recent monitor; sequence cursor, not a complete evidence archive |

Legacy explicit operator calls to `/api/event` accept `type`: pickup,
transport_start/carry_start, transport_stop/carry_stop/arrive, release,
flight_start/flight_stop, impact, capture, miss, stop. Include stable `trial_id`
and `event_id`. These calls are labelled operator actions/labels, never sensors.
`type:test` with `sound` and `type:audition` are manual listening controls.
Do not use compatibility capture/miss calls for an unconfirmed robot prediction;
use a full versioned event with its actual source and evidence instead.

Event IDs are deduplicated by trial/source/ID; retired trial starts cannot revive
old sound. FX-off events are consumed, not saved for unmute. Explicit replay
permits reauditioning the selected trace and resets only that trace's dedup state.
A replay blocks live impacts until the operator returns to live input. Unknown
outcomes end the trial silently. Profile changes cancel playback. Loop stop,
panic and queued generations are separate from robot stop/safety mechanisms.
The native pipe is nonblocking; a full/broken pipe fails audio instead of waiting
in a producer callback. The server should remain outside all robot control paths.

## Profile format

Eight keys: `pickup`, `carry`, `release`, `flight`, `first_impact`, `impact`,
`capture`, `miss`. Each assignment has exactly these fields:

```json
{"sounds":["cartoon_plop","bone_low"],"selection":"random","gain":0.6,"loop":false,"avoid_repeat":true,"pitch":0,"max_seconds":8}
```

Selection is fixed/cycle/random. Sound IDs must exist in `/api/sounds`; unique
lists, gain 0–1, pitch −12..12, max_seconds 0.2–20, booleans for loop/avoid_repeat.
Only carry and flight may loop. Empty/zero-gain assignments are silent. Fixed
uses the first selected ID. Cycle has one counter per event, reset on profile
apply/load/save. Random avoids its immediately previous ID when requested.

`runtime/profiles.json` uses `sircus.sound-profiles.v1`, max 32 custom profiles,
names 1–64 characters. Saving uses a same-directory temporary file and atomic
replacement; a failed write preserves the old store. Corruption falls back to
Ghost Booth with a visible error and leaves the original file intact. Last saved
custom mix restores on restart; FX, selected input and ongoing playback do not.
Built-in profiles remain read-only. Loading/applying alone is transient.

Regenerate catalog assets, without playing them:

```sh
python3.12 src/generate_pack.py
python3.12 src/render_scores.py
```

The second command combines the eight haunted sounds and two tonal sci-fi gestures
with the six rendered MIDI scores and rebuilds `audition.wav`/`.mid`.
See `assets/LICENSE.txt` for provenance.
The optional uncommissioned Logic Sampler route is described in `logic/SETUP.md`;
full event loops/pitch are native-only. `--mock-backend` logs commands without
CoreMIDI/audio and is for explicit API tests, not physical validation.
