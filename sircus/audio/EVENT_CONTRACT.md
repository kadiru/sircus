# Proposed/implemented audio boundary: sircus.audio.v1

This is a new local contract, not an existing Viam command or network API.
`AudioEvent` in `contract.py` is the strict validator; `to_dict`/`from_dict` provide
JSON round trips. Unknown/missing fields and unsupported versions are rejected.
The optional data fields below may be omitted and default to null/unknown/false.

| Field | Meaning |
|---|---|
| `schema` | `sircus.audio.v1` (default); incompatible changes require another version |
| `event_id`, `trial_id` | Nonempty, at most 256 characters; producer-assigned immutable observation ID and stable trial ID |
| `source` | `simulation`, `sensor`, `operator` |
| `event_type` | `impact`, `outcome`, `stop` |
| `contact_object` | Explicit object name such as `board`, or `unknown`; never infer board/bucket from ODD note number |
| `timestamp` | Finite, nonnegative seconds in `time_basis` |
| `time_basis` | `simulation_elapsed_s`, `host_monotonic_s`, `unix_s` |
| `clock_id` | Producer clock domain/session identity; host monotonic values across hosts/boots cannot be compared |
| `bounce_index` | Optional reliable 1-based ordinal; null for current physics contacts and ODD gestures |
| `intensity` | Optional normalized 0–1 proxy with units/interpretation in provenance; not assumed physical force |
| `outcome` | `captured`, `missed`, `unknown`; only outcome events may carry a known outcome |
| `confirmed` | Default false; physical known outcomes sound only when true; predictions may not set this true |
| `confidence` | Optional numeric 0–1; null means unavailable, not zero; never substitute a made-up probability |
| `evidence` | `model_prediction` for simulation; `midi_gesture` or future `sensor_detection` for sensor; `operator_label` for operator |
| `provenance` | Required concise explanation of producer and limitations; no credentials or host secrets |

Simulation must use its elapsed clock; physical observations cannot use it. A
`midi_gesture` must have unknown object/outcome and `confirmed=false`. No automatic
capture detector exists. A future `sensor_detection` producer must supply its own
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
This milestone does not install a BLE listener or infer physical identity.

Offline JSONL replay allows one trial and one clock domain. It sorts by source
time, removes exact duplicate IDs, and rejects an ID reused for different data.
Simulation starts at t=0. Host/unix traces are rebased to their earliest event.
No cross-host time synchronization is implied. Live `submit(event, delay_s=0)`
plays on arrival; callers must provide a separately established clock conversion
if they need scheduled future events. Playback speed/pause changes require cancel
and an explicit fresh replay plan; no browser/Viam clock is auto-followed.
