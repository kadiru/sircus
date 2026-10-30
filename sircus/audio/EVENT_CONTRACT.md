# Proposed/implemented audio boundary: sircus.audio.v1

This is a new local contract, not a Viam command. Ghost Booth accepts it at `/api/event` and `/api/replay-events`.
`AudioEvent` in `contract.py` is the strict validator; `to_dict`/`from_dict` provide
JSON round trips. Unknown/missing fields and unsupported versions are rejected.
The optional data fields below may be omitted and default to null/unknown/false.

| Field | Meaning |
|---|---|
| `schema` | `sircus.audio.v1` (default); incompatible changes require another version |
| `event_id`, `trial_id` | Nonempty, at most 256 characters; producer-assigned immutable observation ID and stable trial ID |
| `source` | `simulation`, `sensor`, `operator` |
| `event_type` | `pickup`, `carry_start`, `carry_stop`, `release`, `flight_start`, `flight_stop`, `impact`, `outcome`, `stop` |
| `contact_object` | Explicit object name such as `board`, or `unknown`; never infer board/bucket from ODD note number |
| `timestamp` | Finite, nonnegative seconds in `time_basis` |
| `time_basis` | `simulation_elapsed_s`, `host_monotonic_s`, `unix_s` |
| `clock_id` | Producer clock domain/session identity; host monotonic values across hosts/boots cannot be compared |
| `bounce_index` | Optional reliable 1-based ordinal; null for current physics contacts and ODD gestures |
| `intensity` | Optional normalized 0–1 proxy with units/interpretation in provenance; not assumed physical force |
| `outcome` | `captured`, `missed`, `unknown`; only outcome events may carry a known outcome |
| `confirmed` | Default false; physical known outcomes sound only when true; predictions may not set this true |
| `confidence` | Optional numeric 0–1; null means unavailable, not zero; never substitute a made-up probability |
| `evidence` | `model_prediction` for simulation; `midi_gesture` or future `sensor_detection` for sensor; `operator_label` or `operator_action` for operator |
| `provenance` | Required concise explanation of producer and limitations; no credentials or host secrets |

Simulation must use its elapsed clock; physical observations cannot use it. A
`midi_gesture` must have unknown object/outcome and `confirmed=false`. `operator_action` is an explicit request, not confirmed physical evidence; it
cannot set a known outcome or `confirmed=true`. No automatic capture detector exists. A future `sensor_detection` producer must supply its own
documented validation and confidence policy before confirming an outcome.

Example of a **synthetic predicted** first board contact:

```json
{"schema":"sircus.audio.v1","event_id":"demo-board-1","trial_id":"demo-1","source":"simulation","event_type":"impact","contact_object":"board","timestamp":0.3,"time_basis":"simulation_elapsed_s","clock_id":"demo-1","bounce_index":null,"intensity":null,"outcome":"unknown","confirmed":false,"confidence":null,"evidence":"model_prediction","provenance":"hand-authored synthetic example; no robot or physics run"}
```

`simulation_events(result, trial_id)` accepts a complete saved two-arm physics
result. It converts `contacts[].t/object` and the terminal result. It intentionally
does not turn contact-array indices into bounce indices: the current physics
model collapses some contacts and does not guarantee every bounce is represented.
First board means **first observed record in this replay**, not proof no earlier
physical contact occurred. Intensity/confidence remain null.

- `retained` plus `retained=true` → predicted captured (current model uses 0.2 s).
- Explicit `miss`, `board_miss`, `floor_contact_not_retained` → predicted missed.
- `model_limit`, absent/unknown status, or `retained=false` alone → unknown.
- Gripper-open, native motion-command events and an unfinished/failed native task
  are not impact evidence. Do not silently extract a failed task's nested
  `predicted_physics_result` as if the physical/native trial completed.

`operator_outcome(trial)` accepts the current `sircus-physical-trial-v1` saved
record only in `mode=hardware`. A saved, annotated `captured`/`missed` label becomes
confirmed operator evidence, not automatic sensor evidence. `unsure`, unsaved or
unlabeled records stay unknown; `dry_run` is rejected. Timestamps require timezone
information. Existing operator labels are assertions, not calibrated probabilities.

`odd_tap(...)` converts an already-filtered positive ch1/note0 gesture into an
unknown-object impact. IDs must come from a host/session observation counter,
not velocity or timestamp alone: two legitimate hits may share those values.
Ghost Booth selects a CoreMIDI input and maps its positive configured impact notes. Port names alone do not identify physical balls.

Offline JSONL replay allows one trial and one clock domain. It sorts by source
time, removes exact duplicate IDs, and rejects an ID reused for different data.
Simulation starts at t=0. Host/unix traces are rebased to their earliest event.
No cross-host time synchronization is implied. Live `submit(event, delay_s=0)`
plays on arrival; callers must provide a separately established clock conversion
if they need scheduled future events. Playback speed/pause changes require cancel
and an explicit fresh replay plan; no browser/Viam clock is auto-followed.

Ghost Booth adds pickup/carry/release/flight types to this pre-merge v1 contract.
Those timings must come from explicit actions or a documented producer; the saved
physics converter does not invent them. The legacy procedural MIDI adapter ignores
those extra lifecycle cues; the configurable native booth handles all eight slots.
The booth previews profiles independently of trial evidence, and its synthetic
whole-sequence example is labelled hand-authored rather than a physics result.


## Four-ball show extension (source integration; commissioning outstanding)

The optional fields `ball_id`, `other_ball_id`, `sound_id`, `source_trial_id` and
`source_event_id` extend the pre-release v1 wire contract. Omitted fields are
omitted on serialization too, preserving the legacy event shape. A client must
verify the Mini's `show_audio.schema == "sircus.show-audio.v1"` capability before
sending extended events; older services reject them. The show worker also
requires acknowledgment of its current four-ball configuration before FX enable.

Four stable operator slots are `ball-pink`, `ball-purple`, `ball-green`, and
`ball-gray`. Labels and displayed colors are editable; this is not color sensing.
Every optional identifier obeys the existing 256-character limit. An explicit
ball/ball collision needs two different identities, but the current single-ball
simulator does not produce multi-ball collisions. The physical UI offers no
manual contact classification buttons.

Physical ODD input requires a unique, operator-verified CoreMIDI endpoint ID,
channel and note tuple for each slot. All bindings start null. A positive measured
NoteOn may produce `sensor / midi_gesture / impact / unknown` only, with that
ball's configurable generic bounce sound. It never proves a board/bucket/collision
contact or an outcome. Missing bindings stay silent. Released trials for different
balls coexist for a configured 1–30 second impact window (default 12 seconds);
starting the same ball again replaces its association. Cancellation retires that
trial permanently within the process. Source timestamps are checked within each
clock domain; they are never compared across MacBook and Mini clocks. Identical
MIDI packets are suppressed only within the existing short duplicate window.

`versioned_simulation_events` is separate from the historical `simulation_events`
adapter. It requires the exact one-second scoring/physics versions, a completed
executed complete-shot result, and explicit contacts. Predicted capture requires
board bounce, whole-ball entry, and one continuous second inside. Invalid actions
are rejected and unknown/model-limit outcomes remain silent. Contact indices are
not bounce counts. Saved rectangular/0.2-second results and their definitions are
unchanged.

Each repeat of a saved numerical replay gets a unique playback `trial_id`;
`source_trial_id`, `source_event_id`, source timestamps and clock IDs retain the
original numerical evidence. This permits repeated listening without reviving a
retired playback. Its separate carry and two-second settle segment is explicitly
synthetic choreography, never inferred from contact timing or physical motion.
Cancel/pause/navigation/scrub, FX off, transport failure and remote mute cancel
pending playback. A disabled replay does not leave a future queue to sound when
FX is later enabled. Native trial-tagged voices permit scoped cancellation while
other balls' voices/transport continue; global FX off/panic mutes all voices.

The configured carry sound is a bounded percussion loop (default catalog
`drum_roll`, maximum 20 seconds). For physical cycles it begins at the recorded
carry command and continues through the stationary two-second settle, stopping
immediately before the open command is issued. These hooks enqueue optional audio
and never await acknowledgment or change motion, recording, release or Stop guards.

The native route can cancel individual trial-tagged voices. Optional Logic MIDI
notes cannot independently cancel same-pitch overlapping sampler tails; the
source preserves the newer note owner instead of cutting it. Logic tail behavior
requires separate listening/commissioning and is not claimed equivalent to native.
