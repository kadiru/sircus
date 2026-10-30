# Audio status — October 30, 2026 (America/Los_Angeles)

Ghost Booth is now the primary Sircus audio UI, in `sircus/audio/booth/` in this
shared repository. The Mac Mini checkout moved from `odd-balls-music/sircus` to
`viam-ball-hackathon/sircus`, retaining its branch, Git history and local runtime.
Existing draft PR: https://github.com/kadiru/sircus/pull/1 on
`codex/mac-mini-oddball-sfx`; no merge or deployment is implied.

Implemented: eight event slots, fifteen original generated sounds, arbitrary
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

Limits: newly combined native kit/pitch/loops have not been auditioned by the
user. No fresh ball-hit success, BLE interruption, cross-host timing, acoustic
quality, MacBook deployment, pickup/flight detection or capture detector is claimed.
Earlier Ghost Booth native output/watchdog measurements are historical evidence
in the preserved legacy folder; they are not retests of this varispeed build.
The optional Logic sampler route remains uncommissioned for all fifteen sounds;
the older private Retro Synth review project remains separate and muted.

Next: user reviews **Hackathon mix**, then tunes the eight assignments. Verify a
selected ball's tap mapping and audible output at a comfortable level. Only after
that should a separately reviewed dashboard observer publish real lifecycle events.
See README.md, EVENT_CONTRACT.md and MACBOOK_INTEGRATION.md. Preserve the earlier
ODD duo and all existing robot/simulation/physical-trial work.
