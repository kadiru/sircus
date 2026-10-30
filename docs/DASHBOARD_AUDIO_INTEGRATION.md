# Dashboard → Ghost Booth integration

This task branch adds optional physical-trial audio to the Mac dashboard using
the client from draft PR #1. It includes main's reviewed scoring merge
`6f00ac3faa4e34d508b005891e62d9911ead81cf`. It does not deploy the dashboard,
restart Ghost Booth, commission private-LAN transport, or command a robot.

The dashboard shows an initially off FX toggle, volume and audio status. With no
backend configuration it shows **Audio not configured** and does not start the
Node worker. Its module is independent of simulation and physical-control UI
code. Robot controls never consult audio connection, pending or mute status.

## Backend configuration

The Mac Mini service still defaults to loopback. Cross-computer use requires an
explicitly coordinated private-LAN session; a URL alone does not make its current
localhost listener reachable. [The Mini handoff](../sircus/audio/booth/integration/HANDOFF.md)
describes interface binding and the audio-only bearer credential. Do not start
another instance or restart the existing one as part of pulling this branch.

Use [the empty environment template](../sircus/audio/.env.example) to document
`SIRCUS_AUDIO_URL`, `SIRCUS_AUDIO_TOKEN`, and optional `SIRCUS_AUDIO_NODE`. Provision
actual values into the backend environment outside Git when deployment is
authorized. The URL must be an explicit HTTP loopback/RFC1918 IPv4 address with
a port and no credentials, path, query or fragment. Private-LAN URLs require a
token. Do not put either value into a frontend bundle, browser input, URL, log,
committed config or PR. Never reuse Viam/robot credentials for audio.

Only the Python observer and its Node child receive audio configuration. The
worker environment is restricted to PATH and the two audio values. Browser
controls use a separate, ephemeral local action token; it is not the Mini token.
HTTP status exposes only whitelisted operational fields and sanitized errors.

## Control and event paths

`GET /api/audio/status` reads the observer snapshot. `POST /api/audio/control`
accepts FX and volume controls only, behind loopback Host, same-origin, JSON,
size and local action-token checks. The browser supplies a session identity and
monotonic intent sequence so a delayed ON cannot undo that session's later OFF.
No browser endpoint accepts arbitrary audio events, service URLs, credentials or
robot commands. Controls return queued status; connection/acknowledgment is an
audio concern only.

Physical lifecycle hooks snapshot a strict `AudioEvent` and synchronously enqueue
it. The observer does no network/process/pipe I/O on that path. Its background
Node worker calls the supplied `SircusAudio.emit` immediately; that client owns
bounded cue delivery, expiry and request timeouts. Audio errors are isolated
from motion, release and Stop. Robot Stop never awaits mute acknowledgment and
still affects only the dropper's motion, not either tool.

| Controller evidence | Audio cue | Meaning |
|---|---|---|
| Gripper-close command sent | pickup | Operator-requested action; no verified grasp |
| Drop move requested / returned | carry start / stop | Command timing; no ball-motion sensing |
| Release-open command sent | release | Command timing; no verified ball separation |
| Saved hardware Captured / Missed label | outcome | Confirmed human label, not sensor detection |
| Unsure, unlabeled, dry-run, ODD MIDI alone | no outcome cue | Outcome stays unknown |

Persisted lifecycle event IDs, trial IDs and timestamps are reused. Human labels
use the existing deterministic converter. Cancellation, errors and replacement
retire the old trial. No contact/flight timing or automatic capture detector is
invented. Stop and FX-off clear pending cues; FX-off requests remote mute of
active playback. Transport failure or an ambiguous late enable can prevent
confirmation, so the UI retains **remote mute unconfirmed** and directs the
operator to Ghost Booth's local mute. An HTTP acknowledgment is not acoustic
verification, and audio mute is not a robot emergency stop.

This checkpoint attaches only physical-controller cues. Simulation playback is
not attached. The saved-trace converter still describes legacy 0.2-second
results; it must not relabel PR #2's one-second v2 results. V2 native replay is
still blocked, historical results remain unchanged, and the learning draft is
separate. Explicit synchronization is needed before combining both dashboard
branches; pulling source does not update a running process.

## Verification and commissioning boundary

Run `scripts/test.sh` using the pinned Python 3.12 environment and Node 22+.
Tests use in-memory adapters, mocked audio, and isolated test processes. No
native audio backend, physical ODD/BLE, Logic, Viam service or robot is required.

The checkpoint passed 192 Python tests, three Node suites, Scripter assertions
and the orientation contract checks. A disposable loopback fake verifies the
real Python→Node→provided-client transport, event evidence, duplicates, retirement
and mute. Browser QA verified default OFF, status, enable, volume and OFF with
that fake; its final state was muted. No actual service or robot was contacted.

Before cross-computer testing, coordinate the Mini's actual private interface,
audio-only token provisioning, listener ownership, and a deliberate FX-off
starting state. Then separately validate authenticated reachability, enable and
volume controls, unavailable/delayed/duplicate cues, trial cancellation and
remote/local mute. Physical listening, ODD/BLE interruption/reconnection and
optional Logic commissioning remain outstanding. Keep PR #1 draft.
