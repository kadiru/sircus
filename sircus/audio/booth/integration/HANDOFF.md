# Optional MacBook backend → Mac Mini audio

The implemented service defaults to **127.0.0.1:8766, FX off**. The MacBook observer
now exists on its integration task branch; [its guide](../../../../docs/DASHBOARD_AUDIO_INTEGRATION.md)
describes the source-only attachment. No physical-dashboard or cross-host
deployment has been performed. Use saved-trace rehearsal first;
[the local API](../README.md) and [event schema](../../EVENT_CONTRACT.md) are shared.

## Explicit private-LAN opt-in

Stop the localhost instance before starting another. Bind the Mac Mini's actual
private interface address; the address below is an example, not a discovered host:

```sh
sircus/audio/booth/launch.sh --host 192.168.1.50 --allow-private-lan
```

Only explicit RFC1918 IPv4 interfaces are accepted. There is no wildcard/public
binding or tunnel. This is plain HTTP for a trusted private LAN; Internet access
is not configured. Cross-host reachability, latency and cancellation are untested.

A random token is stored at `booth/runtime/lan-token` with mode 0600. Alternatively
set `SIRCUS_TOKEN` privately on the service. Supply the same token to the MacBook
**backend** via `SIRCUS_AUDIO_TOKEN`, with its service URL in `SIRCUS_AUDIO_URL`.
Never put credentials in a browser bundle, URL, log or Git. Remote requests need
`Authorization: Bearer <token>`, including reads. Local loopback needs no token.
Host/Origin checks, JSON-only mutations and no CORS require a backend proxy for a
remote dashboard. Do not copy robot/cloud credentials for audio.

## Nonblocking client

```js
import {SircusAudio} from './sircus-audio-client.mjs';
const audio = new SircusAudio({
  baseURL: process.env.SIRCUS_AUDIO_URL,
  token: process.env.SIRCUS_AUDIO_TOKEN,
  onError: error => console.warn('Optional audio:', error.message)
});
// Explicit operator controls may await an audio-only response:
await audio.setFX(true);
await audio.setVolume(0.35);
// A producer never awaits audio. Preserve the complete source/evidence/clock record:
audio.emit(versionedAudioEvent);
// Operator stop; scoped cancellation cannot stop a different active trial:
await audio.stop(trialId);
await audio.setFX(false);
```

`emit` snapshots its event, returns immediately and uses a bounded queue. Requests
time out after 350 ms; queued cues expire after one second. Failures/rejections
report audio status only. There are no automatic retries. Stable IDs deduplicate
an ambiguous send. `stop`/FX off clear the client queue and abort active requests;
the service retires a stopped trial so delayed packets cannot revive it. Transport
failure can prevent a remote mute request arriving: the local audio mute remains
a separate operator control. This is not a real-time guarantee or robot interlock.

Compatibility `event(type,{trialId,eventId,velocity})` is only for explicit
operator actions/labels. Prefer `emit` for all simulation and sensor producers.
An automatic command-success callback must not be relabelled as operator-confirmed
pickup/capture. ODD gestures cannot identify the board/bucket or capture/miss.
Keep one active domain; replay blocks live input until Return to live ball.

Status polling and pending/error indicators are audio UI concerns. No audio
request, readiness check or callback should gate arm motion, release or robot
emergency stop. Add any actual dashboard observer in a separately reviewed change.


## Ordered FX controls (source checkpoint)

The panel and provided `SircusAudio` client share `fx-controls.mjs`. Each panel/client
session has a fresh identity; its FX/Panic intents carry increasing `control_sequence`.
ON first reads `control_epoch` from status and includes it with the command; any
mute, input reconnect, native failure or service replacement invalidates earlier
epochs. This read is audio-control-only: event producers still call `emit` and
return immediately. OFF/Panic never wait for that read. No failed or stale ON is
automatically retried.

The service checks intent order and epoch atomically. Duplicate/stale commands
are rejected with authoritative status, and a reused sequence with changed
content is invalid. Histories are bounded to 128 clients without eviction; at
capacity enables fail closed while OFF/Panic still mute. Older clients may send
unsequenced OFF/Panic, but must upgrade before enabling; direct unsequenced ON
is rejected. This deliberate compatibility restriction prevents old controls
from bypassing mute ordering. Restarting to clear history is a separately
authorized operation, never part of source synchronization.

Use a fresh explicit ON after inspecting status following a mute or failure.
A server acknowledgment still does not establish acoustic silence or replace
local mute. Private-LAN, listening, ODD/BLE and optional Logic acceptance remain
outstanding; no running service was updated by this source fix.
