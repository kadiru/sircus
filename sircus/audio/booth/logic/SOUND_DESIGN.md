# Tonal sci-fi movement sounds

The user's listening feedback on October 30 was that Rising whoosh sounded too
much like white noise. Its old synthesis included a large filtered-noise layer.
The native catalog now replaces that asset, keeping its `transport` ID/note 41
so saved assignments work unchanged, with **Energy hum**. **Energy sweep** is a
new note-45 alternative. Catalog total: sixteen sounds.

The hum uses periodic 150/151.5 Hz oscillators, shaped harmonics and slow pitch/
amplitude movement. Its four-second cycle is designed to join continuously.
The 0.9-second sweep eases from 140 to 560 Hz and back with a shaped amplitude
envelope. Both peak at 0.16 full scale before the existing master gain. Neither
uses a random noise source, factory sample or film recording. Source is
`../src/generate_pack.py`; these are original native assets, **not Logic exports**.
They remain candidates for user listening; numerical checks do not establish
perceptual quality or an exact movie-sound match.

## Logic option, already present on this Mac

Apple's **Alchemy** provides virtual analog/additive synthesis, filters,
modulation and a preset library. A suitable next Logic patch would use two
slightly detuned triangle/soft-saw sources around D3, a low-pass filter with low
resonance, subtle pitch movement, and noise sources disabled. A performance knob
could control pitch/filter brightness for the passing sweep. This is a proposed
patch recipe; no Alchemy project or audition is claimed by this change.

Read-only local inventory found these installed Alchemy preset files:

- `Basic/Soundscapes/Vintage Sci-Fi Ambience.acp`
- `Basic/Soundscapes/Infinite Drone.acp`
- `Basic/Sound Effects/Sci-Fi Communications.acp`

They are under `/Library/Application Support/Logic/Plug-In Settings/Alchemy/`.
Logic's patch inventory also includes Laser Brain and Underwater Laser. File
presence is not a listening test or proof every referenced sample is installed.
There is no need to install a third-party synth to build the oscillator patch.
Extra packs can be selected in Logic's Sound Library if a particular preset needs
content; no bulk download or purchase was started.

Official references checked October 30, 2026:

- [Alchemy synthesis overview](https://support.apple.com/en-euro/guide/logicpro/lgsi2618652a/12.3/mac/15.6)
- [Download additional sound packs](https://support.apple.com/en-mn/guide/logicpro/lgcp8b3abc1c/mac)
- [Create and transfer an Alchemy library](https://support.apple.com/en-us/101827)

For Ghost Booth's standalone playback, render an original designed cue to WAV
and add it to the catalog; an Alchemy preset itself runs inside Logic. Keep
factory sample content separate from the redistributable original catalog.
