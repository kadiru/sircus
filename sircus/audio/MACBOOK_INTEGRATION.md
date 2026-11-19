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

## Optional dashboard attachment (source only)

The MacBook integration task branch now provides a backend observer and FX/volume/
status controls; see [the dashboard integration guide](../../docs/DASHBOARD_AUDIO_INTEGRATION.md).
The running physical dashboard has not been deployed or restarted. Ghost Booth's
base draft PR remains separate, with listening and cross-computer checks pending.

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
| ODD CoreMIDI | Legacy mode selects one input; show mode supports up to four explicitly bound endpoint IDs and filters per-ball notes. Neither identifies contact objects or capture/miss. |

The standalone audio contribution introduced no robot/controller/viewer hooks.
The separate dashboard observer observes operator-requested command timing and
saved human labels only; it adds no cloud identity reuse, Viam configuration
change, physical sensing validation or automatic deployment.
The six-cue procedural Logic route remains available in [LEGACY_MIDI.md](LEGACY_MIDI.md)
for advanced diagnostics; use Ghost Booth for the unified event sound library.

## Integration ownership and outstanding review

MacBook Codex owns dashboard/controller integration. The Mac Mini audio PR stays
draft until its listening and integration reviews are satisfied. These checks are
still outstanding as of October 30, 2026; they are not implied by passing mock
tests or by an HTTP acknowledgment:

- Listen to Energy hum, Energy sweep and the saved mix; review pitch changes,
  bounded loops, clipping and FX-off/panic during active voices and tails in the
  current native build. Earlier native mute measurements predate varispeed.
- Observe the selected physical ODD Ball's actual tap note/channel/velocity,
  verify audible single/rapid impacts, then test BLE loss and reconnection with
  FX remaining off. Port enumeration or CC traffic is not a bounce test.
- Commission the optional Logic kit and exclusive MIDI route separately,
  including raw-input bypass/feedback rejection and actual tail muting. The
  private Retro Synth project is not a verified sixteen-sound Sampler kit.
- In a separately authorized LAN session, provision the audio-only token outside
  Git, verify authenticated status/controls and event delivery from the MacBook,
  and test loss, late/duplicate events, cancellation and local mute. Show audio
  pending/error state without gating robot execution; do not claim synchronized
  clocks or guaranteed delivery.
- Review lifecycle producers and provenance before attaching them. Keep unknown
  outcomes silent and require explicit physical outcome evidence. Audio panic is
  not a robot emergency stop.

Integration may review or fetch the draft branch without deploying it. Keep
profiles, recordings, Logic projects and generated runtime files host-local;
catalog assets and source are available through Git. Pulling this PR does not
restore a private saved mix or commission an audio route.


### Ordered controls compatibility update

The supplied client now uses shared per-client intent sequences and status mute
epochs for FX/Panic. `setFX(true)` may await a status read on its audio-only
control path; `emit` remains immediate and nonblocking. OFF/Panic never wait for
status. Older services without `control_epoch` fail closed for enable, and newer
services reject legacy unsequenced enables. Fake endpoints should expose a
`control_epoch` string in status and enforce/control-record the sequence fields.
See [the detailed protocol](booth/integration/HANDOFF.md#ordered-fx-controls-source-checkpoint).


## Three-mode dashboard source integration

The show dashboard reuses `AudioObserver` → backend Node `SircusAudio` → Ghost
Booth. Browser code receives sanitized status and a local action token, never
`SIRCUS_AUDIO_URL` or `SIRCUS_AUDIO_TOKEN`. Its `ShowAudio` coordinator persists only
the four-slot mapping in ignored host-local runtime storage. `configure(config)`
mutes, retires previous ball associations and queues `/api/show-config`; FX starts
off. Failed or unsupported configuration prevents show FX enable. A successful
HTTP response verifies a software configuration, not BLE or audible playback.

`ShowAudio.status()` exposes config/catalog, bounded replay state, and capability
limitations. `start_replay(result, trial_id, ball_id)` accepts only the server's
completed versioned numerical result; it never commands an arm. `cancel_replay()`
and `close()` cancel owned audio work. `AudioObserver.emit` remains immediate,
queue-bounded, expiring and optional. No public show endpoint accepts arbitrary
AudioEvents or manual physical contact labels. See [EVENT_CONTRACT.md](EVENT_CONTRACT.md)
for exact evidence and repeat-replay identity rules.

The physical controller adds idle-only `set_ball {ball_id}` and
`configure_poses {loading_pose, drop_pose, confirmed:true}`. Each pose has the exact
TCP fields `x,y,z,o_x,o_y,o_z,theta` in the existing Viam mm/orientation-vector
convention. Configuration records an operator assertion and fresh board joints;
it performs no motion and does not fabricate measured calibration. Taught poses
remain supported. The existing fixed-cycle confirmation, stationary checks,
board-lock checks, recording readiness, collision/planning behavior and Stop
remain authoritative. Defaults never silently use historical screenshot poses.
Height-calibrated operation still requires its separate measured reference/limits.

### Mac Mini coordination, without automatic deployment

First synchronize the reviewed task branch/commit into a clean separate checkout;
fetching code does not restart the existing booth. Then run the mock suites below
and compile the native source silently. Actual startup or restart is an explicit
operator step, with FX off:

```sh
python3.12 -m unittest discover -s sircus/audio/tests -p 'test_*.py' -q
python3.12 -m unittest discover -s sircus/audio/booth/tests -p 'test_*.py' -q
node sircus/audio/booth/tests/test_client.mjs
node sircus/audio/booth/tests/test_dashboard_worker.mjs
# Operator, only when ready to commission the intended new audio service:
python3.12 -m sircus.audio booth
```

Use [the private LAN handoff](booth/integration/HANDOFF.md) for an explicitly
chosen private interface and environment-only token. Do not assume localhost on
the MacBook reaches the Mini. Cross-computer delivery, four actual BLE balls,
endpoint identity after reconnect, per-ball tap mapping, listening/clipping,
scoped tail cancellation and physical two-second carry timing remain unverified.
The included generated catalog is available; private Logic projects are not
required or silently copied. Optional Logic commissioning remains separate.

The show carry default is the original synthesized `drum_roll.wav` (eight snare-like
strokes/second, seamless one-second loop, -15.9 dBFS peak before playback volume).
Regenerate it with `python3 sircus/audio/booth/src/generate_drum_roll.py`; this does
not rewrite older WAVs or play audio. Waveform/level tests are not listening tests.


## Hardware-demo audio host and readiness checkpoint

The first physical demo uses **Ghost Booth native audio on the MacBook Pro**, the
same computer as the dashboard. Logic and a Mac Mini connection are not required.
Set `SIRCUS_AUDIO_URL=http://127.0.0.1:8766` in the backend environment and leave
`SIRCUS_AUDIO_TOKEN` unset for loopback. These values never belong in browser
configuration. Starting the service is an operator action; pulling this code
starts no service, opens no MIDI device and plays nothing.

Pair the ODD balls with the audio-host Mac. Verify each ball separately in the
Booth Monitor and save its actual endpoint ID, channel and positive NoteOn note
in the Pink/Purple/Green/Gray slots. No source default binds a physical input.
Select the ball color before its robot cycle. The saved release event opens that
ball's measured-impact window; later hits from previous, differently colored
released balls keep their original trial/ball identity. Impacts are generic
measured bounce cues, never inferred board/bucket contact or success/failure.

The Booth now reports `sircus.show-readiness.v1` separately from its event/show
schemas. A configuration ACK returns a SHA-256 digest of the accepted config;
only a later status can confirm that the native helper selected the matching
endpoint set. Dashboard status separates native engine availability, matching
saved mappings, and connected counts/per-ball booleans. HTTP connectivity alone
means neither four connected balls nor audible playback. Endpoint IDs, names,
config digests and private host metadata are not forwarded in the dashboard
readiness status.

On every explicit FX enable, the worker checks fresh status and the client binds
the enable to the accepted digest. The Booth checks digest, native selection,
engine and control epoch together under its lock. A restart, another client's
mapping change, legacy single-input selection, or service loss withdraws stale
readiness. Reapply the saved mappings explicitly, wait for native input selection,
then enable FX. Reapplying mutes audio and retires old audio trial associations;
it never moves an arm. A poll never reapplies mappings or re-enables FX. Partial
connectivity is displayed truthfully; absent inputs stay silent and never block
robot motion or Stop.

Silent readiness regressions cover service restart, mapping replacement,
configuration/enable races, partial input connection, legacy protocol refusal,
remote-mute ordering and sanitized status. Real BLE endpoint identity, listening,
reconnection behavior and overlapping physical ball impacts still require
supervised commissioning. No test result here establishes acoustic verification.
