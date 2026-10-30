# Optional Logic Sampler route

Ghost Booth's primary route is its native Mac sampler. It needs no Logic project.
This optional Sampler route is **not commissioned or acoustically verified** for
the consolidated fifteen-sound kit. Pitch shifts and event loops require native
playback; the bridge rejects those features when the optional Logic route is selected.
The separate musical ODD duo project remains unrelated and unchanged.

To commission manually, create a separate empty software-instrument project in a
private, ignored directory. Keep Stereo Out muted while configuring, starting at
−18 dB. Use one full Sampler instrument with the fifteen WAVs in `../assets/map.json`;
exclude `audition.wav`. Each zone's root/low/high key is its numeric MIDI note in
that map, full velocity range. Turn loops and 1Shot off; use a short amplitude
release. Use no downstream reverb/delay/tail effects while checking mute behavior.

Select only **Sircus FX**, channel 16, on this track. Input monitoring on, record
enable off. Never monitor raw ODD or All Inputs on the cue receiver. Insert the
included `sircus-input-guard.js` Scripter before Sampler; Run Script and confirm
successful evaluation. It starts muted until expression CC11 enables it. Configure
CC11 as instrument expression and keep an accessible Logic track mute.

The native helper's generated port is **Sircus FX**. The older procedural MIDI
adapter uses a different port, **Sircus SFX**, and different note meanings; do not
mix their receivers or run both routes for the same input.

For supervised commissioning only, acknowledge the optional route through
`POST /api/route` with `{"route":"logic","commissioning_acknowledged":true}`.
Route changes force FX off. Use a profile with all loop flags false and pitch 0.
Explicitly enable FX and audition at low volume. Check every sound; test FX off,
Stop and Panic during sustained samples and rapid repeats. Confirm no raw-bypass
sound when the ball is hit with FX off. CC panic is not a guarantee that every
Logic instrument/effect becomes immediately silent; listening and meter checks
are still required. If any sound leaks, mute Logic and switch back with
`{"route":"native"}`. Save/reopen the kit only after those checks pass.

This change includes no `.logicx` or saved Sampler instrument and makes no claim
of current Logic route, note overlap, reconnect, or acoustic validation.
