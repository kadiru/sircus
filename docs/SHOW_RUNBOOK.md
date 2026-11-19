# Sircus show desk

`/show.html` is the four-mode entry point. It combines the existing operator,
training and optional Ghost Booth clients; it does not create a second physical
controller or silently deploy a cloud part.

## Start an isolated rehearsal

After installing the pinned project dependencies (`./setup-ubuntu.sh` on Ubuntu):

```sh
./start-show.sh
# Open http://127.0.0.1:18769/show.html
```

Set `SIRCUS_PYTHON` if using an existing project Python 3.12 environment. The default
launch uses mock arms/camera/recording and local numerical simulation. It needs no
Viam credentials. Generated configuration, trial records and workflow jobs live
under ignored `.runtime/show/`; do not commit or copy them over host calibration.
The interface marks dry-run evidence explicitly. Mock routines are silent; only
explicit numerical audio replay uses synthetic contacts.

## Four acts

| Mode | Implemented behavior | Boundary |
|---|---|---|
| Manual operation | Configure or teach TCP loading/drop targets; one button moves to load, opens if needed, waits 2 seconds, closes, carries to drop, settles for 2 seconds and releases | Real execution requires an explicitly started hardware backend and verified fixed setup; no bucket detector needed |
| Trained operation | Enter or randomize board center, place bucket, then make one frozen scene-conditioned prediction with an explicit geometric heading constraint; acknowledge loading and press Go | Constrained variant is separate from the published 33/48 benchmark; real bucket perception, transforms, native paths and physical transfer remain uncommissioned |
| Simulation | General numerical scene editor, original fixed-board/frozen planners, local training jobs and published study results | Synthetic physics and provisional geometry; endpoint IK does not prove a safe arm path |
| Mock simulation | Same placement/load/Go steps as Trained, with bounded numerical tilt/U/V/height search aimed at the bucket | Per-scene search is explicitly not learned inference; no hardware connection and no guaranteed capture |

See [the bucket-facing workflow](BUCKET_FACING_WORKFLOW.md) for the new angle
convention, randomization and experiment reproduction. Internal API mode key
`deterministic` remains unchanged for the renamed Manual operation panel.

Practice mode saves each completed cycle locally without requiring a capture/miss
label. It requires another button press for the next cycle. Stop cancels pending
panel actions and stops arm10; it never opens either tool or releases board suction.

The pose editor stores full Viam TCP coordinates in millimeters plus orientation
vector and theta (not Euler angles). It requires a stationary snapshot and explicit
operator confirmation, and does not itself move an arm. Editing targets does not
measure ball-to-board height: calibrated variable-height controls remain disabled
until their separate calibration is complete. Camera and both arm states remain
visible through the reused physical panel.

## Four balls and measured sound

The stable slots are **Pink, Purple, Green, Gray**. Select the ball being loaded;
this records an operator identity, not visual color recognition. Each ball needs a
verified physical MIDI endpoint/channel/note binding before measured impacts can
be associated with it. Bindings start empty. The sensor reports an impact, not the
contacted object: physical bounces always use the selected generic-impact voice.
No board/bucket classification, capture/miss inference or manual-contact buttons
are provided in the show desk. Unknown outcomes stay silent.

A completed ball retains its trial identity while a different ball loads, so its
later measured impacts can sound independently. The configured post-release
window defaults to 30 seconds (1–30 configurable). Impacts outside it are ignored.
Reusing the same ball replaces its previous association. Cancellation retires the
trial, and changing mappings mutes audio and clears old associations. Four native
MIDI inputs have independent stream parsers; duplicate and stale events are
rejected. Physical BLE identity and simultaneous-device behavior require on-device
commissioning.

Carrying starts the selected percussion loop. It continues through the fixed
2-second settle and stops immediately before release. It is bounded to 20 seconds;
an unexpectedly long move cannot leave an endless loop. Disabling FX mutes active
playback. Lifecycle hooks enqueue optional audio only: sound availability,
acknowledgments and errors never authorize or block motion/release/Stop.

The numerical replay button schedules audio from the completed current simulator
trial, preserving source trial/event IDs and model provenance in a fresh playback
namespace. It uses a labeled 3-second choreography prelude; it does not claim
synchronization with the independent viewer animation or physical sensors.

## Private Mac Mini connection

Read [the audio integration guide](DASHBOARD_AUDIO_INTEGRATION.md) and
[MacBook client instructions](../sircus/audio/MACBOOK_INTEGRATION.md). Both computers
must run the same reviewed show protocol revision before cross-computer testing.
Keep `SIRCUS_AUDIO_URL` and `SIRCUS_AUDIO_TOKEN` only in the MacBook backend process
environment; never expose them through browser JavaScript, URLs in screenshots,
Git, or handoff text. The Mini still defaults to loopback. Explicit private-LAN
binding, authentication, firewall scope and operator listening are commissioning
steps, not consequences of pulling this branch. No real Ghost Booth service is
started or restarted by tests. FX defaults off and incompatible services fail closed.

Native playback is the intended first demo route. Optional Logic routing needs
separate commissioning; its sampler note/channel model cannot isolate every
same-note overlapping tail. Do not claim equivalent per-voice behavior without
listening and cancellation checks.

## Hardware deployment boundary

`./start-show.sh --hardware` is an explicit deployment command, **not a sync step**.
It uses `SIRCUS_PHYSICAL_DATA_ROOT` or the existing ignored hardware data directory,
requires backend Viam credentials, and disables all three numerical modes. Do not launch
it alongside another dashboard/controller that owns the same arm. First verify
current runtime ownership, real camera/state, taught targets, fixed board/bucket,
recording readiness and operator-controlled routes. Existing host calibration must
be preserved. The browser does not auto-run a robot sequence after connecting.

For the trained stretch goal, the bucket detector, reference-verified camera/world
and board-tool/base transforms, held-ball offset, scan/load poses, collision-aware
motion paths and native v2 adapter remain required. Existing compatibility guards
are retained. The deployment module is a blocked stub, not a commissioned robot.

## Shared-source synchronization

Agents should fetch and inspect their clean persistent checkout, then fast-forward
`main` only when it is clean. Use their own `codex/<owner>-<task>` branch for work.
The show integration is on `codex/macbook-three-mode-show`; audio PRs and main remain
separate until review. Use a separate worktree to evaluate this draft. Do not reset
unrelated work, copy older archives over source, or restart services to sync Git.

The baseline front-bucket study and video remain in
`baselines/front-bucket-20261030/` and `/study-results.html`. Default frozen inference
scored 33/48 on held-out layouts; fixed action scored 25/48 and separately labeled
bounded search 37/48. These synthetic results do not establish real-robot success.
Historical positive-Y and rectangular-container results retain their original
fingerprints. Board200×300×5mm, base spacing1337.31mm, bucket inner177/150mm and
outer200/158mm are preserved; reported175mm height interpretation and placement
remain provisional.
