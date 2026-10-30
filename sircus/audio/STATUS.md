# Audio status — October 30, 2026 (America/Los_Angeles)

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
