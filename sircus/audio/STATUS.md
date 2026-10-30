# Audio status — October 30, 2026 (America/Los_Angeles)

## MacBook pre-round tap tester — October 30, 2026

Task branch `codex/macbook-ball-tap-test` adds a 15-second-default (1–120 s)
audio-only four-ball test with per-ball MIDI counts, sound names, connection and
last-tap age. Backend expiry survives tab closure; explicit mute, reconnects and
replacement events cancel the test. No fake release or capture evidence is made.
The normal four-ball release-window router remains unchanged outside a test.

Verification: 95 Ghost Booth Python tests, 56 shared audio tests, and nine Node
tests/suites passed. The sandbox initially blocked compiler/CoreAudio and fake
localhost IPC checks; all passed when rerun with the required local permissions.
Native DSP checks render offline, never through speakers. An isolated fake UI
verified 15/30-second starts, three responding balls versus one no-tap ball,
expiry and manual Stop. Live acoustic acceptance of this new tester is pending.
Live deployment: only the previously idle Ghost Booth audio process on8766 was
refreshed. The four private MIDI bindings and selected voices were preserved;
readback shows4/4 connected, native audio ready,100% volume and3× peak-limited
boost, no error, and FX off. The browser displays the new tester at
`http://127.0.0.1:8766/#ball-tester`, ready for an explicit operator Start. No real
tap test or automatic sound preview was triggered during development. Local
runtime PID/path and before/after evidence stay in ignored commissioning records.
No robot/dashboard restart, hardware motion or Viam change occurred.

Ghost Booth is now the primary Sircus audio UI, in `sircus/audio/booth/` in this
shared repository. The Mac Mini checkout moved from `odd-balls-music/sircus` to
`viam-ball-hackathon/sircus`, retaining its branch, Git history and local runtime.
Existing draft PR: https://github.com/kadiru/sircus/pull/1 on
`codex/mac-mini-oddball-sfx`; no merge or deployment is implied.

Latest listening feedback: the user found Rising whoosh too white-noisy. Replaced
its stable `transport` ID/note 41 with **Energy hum**, a periodic four-second tonal
loop, and added **Energy sweep** on note 45. Both use original oscillators without
a random-noise layer and peak at 0.16 full scale. Existing saved carry assignments
resolve to the new hum. These two assets await user listening review.

Alchemy preset files are already present on this Mac. Its oscillator/filter route
can create further sci-fi variants; no extra library was installed and no Logic
preset was auditioned or exported. See `booth/logic/SOUND_DESIGN.md` for the recipe,
inventory and official Apple references. The new WAVs are native generated assets.

Implemented: eight event slots, sixteen original generated sounds, arbitrary
per-event pools or silence, fixed/cycle/random selection, no immediate random
repeat, event gain/pitch, bounded carry/flight loops, three built-in profiles and
atomic named profile saving/loading. Added explicit event preview and synthetic
whole-trial rehearsal; saved simulation/operator traces use the shared event
contract. Loop/voice stop, retired-trial handling, dedup, live/replay separation,
FX-off startup and selected-input isolation remain explicit.

The new default entry point is `python3.12 -m sircus.audio booth` (macOS), serving
localhost8766. No selected input and FX off on startup. The custom **Hackathon
mix** was saved through the browser and restored after a service restart. Its
four bounce choices mix Rubber squeak, Soft plop, Tiny spring and Ghost pinball.
It is host-local under ignored `booth/runtime/profiles.json`. Built-ins ship in
source. Runtime availability must be checked again before future work.

Verification: 23 legacy audio tests and 41 booth tests pass; 80 existing
repository unittest cases and orientation/height synthetic contract smoke also
pass. Node Scripter/client checks pass. Native Swift helper compiled successfully;
CoreAudio reports ready with FX off. Browser load/edit/apply/save and restart
persistence were observed without playing sound. Original musical Logic project
and script hashes matched their preservation baseline. No robot command ran.

Latest checks: the 144 Python unittest cases and synthetic contract smoke pass.
Node syntax checks pass; new WAVs have bounded peaks, negligible DC and no random
noise layer. The hum's loop seam has a normal adjacent sample step. CoreAudio is
ready after restart with FX off; browser shows all sixteen sounds and the restored
Hackathon mix with Energy hum assigned to carry. No automatic playback was used.

Limits: broader native kit/pitch/loop acceptance remains unverified beyond the
user's whoosh feedback. No fresh ball-hit success, BLE interruption, cross-host timing, acoustic
quality, MacBook deployment, pickup/flight detection or capture detector is claimed.
Earlier Ghost Booth native output/watchdog measurements are historical evidence
in the preserved legacy folder; they are not retests of this varispeed build.
The optional Logic sampler route remains uncommissioned for all sixteen sounds;
the older private Retro Synth review project remains separate and muted.

Next: user reviews **Hackathon mix**, then tunes the eight assignments. Verify a
selected ball's tap mapping and audible output at a comfortable level. Only after
that should a separately reviewed dashboard observer publish real lifecycle events.
See README.md, EVENT_CONTRACT.md and MACBOOK_INTEGRATION.md. Preserve the earlier
ODD duo and all existing robot/simulation/physical-trial work.

## Publication checkpoint — October 30, 2026

Fetched origin and inspected the clean checkout and existing draft PR #1. The
active branch remains `codex/mac-mini-oddball-sfx`, previously published at
`a1bfe3537536ec65f959ef0e7add7ab10761220c`. Fetched main is
`26ca998f3e93b3dbfa795e5db354d4c07985d703`, already an ancestor of the audio branch;
there are no newer main changes to merge and no unrelated uncommitted source to
stash or move. Ignored local profiles/runtime and archived personal projects are
preserved. The latest shareable audio implementation was already pushed; this
checkpoint records the standing synchronization policy and integration review.

Fresh, silent checks: all 144 Python unittests and the orientation/height
synthetic contract smoke passed using the repository Python 3.12 environment.
The nonblocking client, Scripter mocks, generated preview with fake Web Audio,
and JavaScript syntax checks passed. Native Swift compiled into a temporary
directory without executing it or replacing the runtime binary. The tracked
file/credential scan passed for all 187 files. The preview test requires a generated HTML
fixture; an initial invocation omitted it, then the documented fixture-based
check passed. No audio service was started/restarted, no playback was enabled,
and no hardware, BLE, Logic or cross-host acceptance test was performed.

MacBook Codex remains the integration owner. The PR must stay draft for the
listening, real ODD/BLE, current native/Logic route and cross-computer checks
listed in MACBOOK_INTEGRATION.md. No dashboard/controller observer, Viam change,
physical deployment or robot command is part of this synchronization.

## MacBook observer checkpoint — October 30, 2026

The separate `codex/macbook-dashboard-audio` branch now contains optional FX,
volume and connection/mute status controls plus physical-trial lifecycle
observers. Main's reviewed PR #2 merge `6f00ac3faa4e34d508b005891e62d9911ead81cf`
was merged into this task branch without replacing existing work. The Mini's
published `df16dbe609adf3242af9300bc88725b85adfd455` audio branch is preserved;
PR #1 remains draft. See [integration guide](../../docs/DASHBOARD_AUDIO_INTEGRATION.md).

Fresh verification: 192 Python tests passed, including 42 physical-controller
mock tests, 7 audio HTTP contract tests and a Python→Node→disposable fake HTTP
round trip using the supplied client. Three Node suites and Scripter assertions
passed; scoring/contact/orientation guards passed. Independent review checked
that no audio readiness/acknowledgment gates motion or Stop, and that IDs,
operator-action provenance and unknown outcomes are preserved.

Browser QA against an isolated offline dashboard and silent fake endpoint
verified initial OFF, connected status, enable, volume and mute; the final fake
state was FX off at 30% volume. It found a WebKit timer receiver bug, now fixed
and regression-tested. No sound played. The temporary dashboard, worker and fake
endpoint were stopped. This is not deployment, listening, real-Mini reachability,
ODD/BLE, Logic, acoustic mute or hardware acceptance.

Fixes include ordered per-browser control intents, preservation of pending mute
across Stop/volume, late-event retirement, bounded queues and conservative mute
uncertainty after ambiguous transport. Credentials stay backend-only, with an
empty template in source. Actual private-LAN listener and credential provisioning
remain a coordinated follow-up. Simulation audio is not attached; the legacy
trace converter has not been promoted to the new one-second scorer. The learning
draft remains separate and v2 native replay stays blocked.

## Control-ordering correction — October 30, 2026

After merging reviewed learning main `a376ad67968f6445d39a51c2a29325a204bb582a`
into the audio branch, fixed a reproduced race in which delayed FX ON could undo
a newer OFF/Panic. The local panel and provided client now share ordered intents;
the service checks bounded client history and mute epochs atomically. Legacy
unsequenced ON is intentionally rejected; OFF/Panic remain available, including
at history capacity. Native failure/reconnect invalidates earlier enable epochs.
Pending enable acquisition is cancelled by newer controls; no automatic enable
retry exists. This changes audio control transport only, not robot behavior or
trial evidence.

Fresh silent checks: 48 booth Python tests (including seven ordering regressions),
provided Node client tests, new shared FX-control tests, and frontend JavaScript
syntax. Tests cover reversed ON/OFF and ON/Panic arrival, stale cross-client
observations, duplicate/reused sequences, bounded capacity, native error/reconnect,
service-instance epochs, old-service rejection, and pending-enable cancellation.
No real service, audio playback, ODD/BLE, Logic, Viam or robot was used. PR #1
remains draft pending the documented listening and integration acceptance work.


## Combined dashboard correction — October 30, 2026

PR #3 is now merged on main at `a376ad67968f6445d39a51c2a29325a204bb582a`.
The integration branch retains both study and audio routes/styles/tests. A Mini
FX-off observation clears dashboard opt-in and queued cues; external Mini FX-on
withdraws a previous mute acknowledgement while local opt-in stays off. The
requested state is propagated through worker, backend and UI. Stale/in-flight
polls cannot undo newer controls, and settled failures no longer show pending.
The new ordered Mini controls and supplied client are included in this branch.
Audio PRs #1 and #4 remain drafts and neither runtime has been deployed.

Final combined verification passed 220 Python tests, all audio/study Node suites,
Scripter and orientation-contract assertions, including the unchanged published
checkpoint's numerical reproduction and the real Python-to-Node-to-fake-HTTP
remote-status regression. Independent reviews and repository scans passed.
These are silent mock/numerical checks; listening, ODD/BLE and private-LAN
commissioning are still outstanding. No existing service was restarted.


## Three-mode show integration (source only)

The show draft adds four explicitly bound Pink/Purple/Green/Gray slots, retained
per-ball trial identity, generic measured-impact voices and an original drum-roll
loop for carry plus the two-second settle. Native MIDI streams and active voice
cancellation are scoped by input/trial. See `docs/SHOW_RUNBOOK.md` from repository
root. The existing catalog/audition baselines remain; drum_roll is an additional
original asset. No real listening, BLE or cross-computer commissioning is claimed.


## MacBook BLE hot-plug discovery correction — October 30, 2026

Live commissioning found that Ghost Booth's native audio and one-second meter
were running, with successful CoreMIDI client/port creation, while newly paired
ODD Balls remained absent from its input list. Fresh read-only CoreMIDI enumeration
saw the online hardware endpoints with nonzero entities; neither endpoint
matched the Booth's exclusions. This ruled out the filtering code and OS pairing.

Two otherwise identical, bounded read-only diagnostic clients ran concurrently,
with no MIDI ports, source connections, audio engine or playback. On a subsequent
ball connection, the client using `RunLoop.main.run()` received 21 topology
notifications and updated from two to three sources. The `dispatchMain()` client
received none and retained its old two-source snapshot despite continued polling.
The running Booth also retained an empty input list while its meter advanced.

Native.swift now services the main run loop instead of entering `dispatchMain()`.
The existing main dispatch queue still serializes controls, input decoding,
reconciliation, playback and watchdog logic. No input is automatically selected
and FX still starts off. The diagnostic endpoint identities/logs remain local,
outside Git. This A/B evidence establishes a topology-discovery defect on this
Mac; subsequent live commissioning is recorded separately below.


The patched full native source compiled successfully without executing the audio
helper. Twenty targeted tests passed: pure Swift MIDI decoding and voice ownership,
mocked multi-ball routing/readiness, and ordered FX controls. The pure Swift tests
used an explicit writable compiler module cache; they initialize no MIDI or audio
backend. No source-text assertion was added as a substitute for the observed
runtime A/B experiment.

With the user's authorization, only Ghost Booth was restarted with the patched
helper. Pink and Purple appeared, then Green connected while Booth was running
and appeared without another restart. Gray was subsequently discovered and all
four endpoints were simultaneously connected. Gray's initial discovery delay
does not establish a device-count limit or a known Bluetooth root cause.

Each color was identified through isolated activation and positive hardware
NoteOn input, then individually confirmed audible by the user. The ignored host
show configuration saves the measured endpoint/channel/note bindings and these
generic `sounds.unknown` assignments:

| Ball | Verified voice |
| --- | --- |
| Pink | `cartoon_boing` |
| Purple | `cartoon_squeak` |
| Green | `cartoon_spring` |
| Gray | `cartoon_plop` |

A separate 30-second manual test armed all four ball/trial routes together.
Recorded sound requests retained the correct ball/trial identities: Pink 37,
Purple 28, Green 27 and Gray 46, with the matching voices above. These are sound
request counts, not independent physical-bounce counts. The user confirmed all
four sounded correct. The arming events were explicitly labeled operator audio
test requests using the release routing gate, never observed robot releases;
the impacts came from live MIDI. Raw events and endpoint IDs remain host-local.

The test ended with all commissioning trials canceled and FX muted. Booth remains
running on loopback; the bounded observation collector was stopped. Full
regression passed 422 Python tests and 106 Node tests, plus Scripter and fixed-scene
orientation-contract smoke assertions. No robot SDK connection, robot/tool action,
Viam configuration, physical-dashboard deployment, or simulator restart occurred.
Robot-cycle timing, surface classification, capture/miss sensing and private-LAN
operation remain unverified. The service's full physical-commissioning flag is
not promoted by this audio-only test.

## Native peak-limited loudness — October 30, 2026

Manual hand drops produced the four expected measured MIDI sound routes at
maximum master and Mac speaker volume. The operator reported balls reaching the
bucket but insufficient loudness. This aggregate report is kept separate from
automatic scoring; no per-ball capture counts or contact classifications were
inferred from MIDI.

The native path now offers an explicit 0–9.5424250944 dB boost (up to 3× signal
amplitude), with Apple's peak limiter after the summed voices. The original
master-volume curve remains. Logic routing is unchanged. Boost starts off,
requires a ready engine and limiter, and reports requested versus acknowledged
native gain. No boost request enables FX or arms a trial. Global mute clears
the limiter tail and preserves the selected gain for later explicit playback.

Silent macOS offline renders verify 3× gain for quiet tones and the original
boing at representative/full velocity, bounded eight-voice and transient peaks,
mute/re-enable silence, retained gain across reset, and rejection of invalid
gain. Mock backend/UI checks cover missing readiness, pending acknowledgments,
failures, unsupported routes and unchanged FX/master controls. See
[loudness instructions](../../docs/AUDIO_LOUDNESS.md).

Full regression passed 440 Python tests and 107 Node tests, plus the Scripter and
fixed-scene orientation assertions. Only Ghost Booth was restarted to deploy this
audio change, initially muted; the saved four-ball bindings and sound assignments
were reapplied. Native readiness, limiter readiness and the +9.54 dB applied
acknowledgment were verified before the manual listening window. The operator
confirmed the boosted hand-drop sounds were loud enough and clear. This is
listening acceptance, not measured sound-pressure level or robot validation.
The UI now reports the configured four-ball connection counts and bounded impact
windows instead of displaying an unselected single MIDI input. No physical
dashboard, Viam configuration, simulator process, arm or tool was changed.
