# Learning demonstration film

`learning-demonstration.mp4` is a standalone H.264/AAC film, 1280×720 at 20fps,
65.85 seconds. `learning-demonstration-poster.jpg` is its cover and
`learning-demonstration-manifest.json` records the source bundle, renderer,
media hashes, chapter offsets, outcomes and soundtrack event provenance.

The film preserves all four N8/24/64/128 predictions on the same predeclared first
held-out layout. **All four miss.** A captured shot on a different held-out layout
is shown separately. The population chart reports mean performance over three
training seeds on the same 48 held-out layouts: 59.0%, 67.4%, 67.4%, 69.4%.
The predeclared N128/seed20261030 model captured 33/48; its 95% interval is 54.7–80.1%.
The fixed 25/48 and bounded-search 37/48 comparisons are separate methods.

The model predicts scene-conditioned board roll/pitch, board-local U/V and
vertical release gap. It does **not** learn a joint policy. Arm positioning uses
smooth interpolation of the saved endpoint joints through the existing forward
kinematics model. This visibly illustrates angle/height changes; the large arc
between distinct IK branches is not a planned, collision-checked or deployable
motion. The ball then follows the original saved numerical frames, selected by
their timestamps. Neither the film nor its renderer runs optimization or refits
a checkpoint. Release gap is world-Z ball-center clearance, not flange height.

The soundtrack uses original bundled synthesized drum/effect WAVs. Carry/settle
drums are illustrative; contact cues use saved simulation contact timestamps,
spaced at least 80ms. Outcome cues use known saved outcomes; unknown stays silent.
These sounds are not microphone recordings, physical ODD/BLE evidence or proof
of contact classification. No audio device is opened to render the film.
Asset provenance and redistribution terms are in `sircus/audio/booth/assets/LICENSE.txt`.

Board/mount/contact dynamics and bucket placement remain provisional. The 175mm
bucket reported height still has an unresolved overall-height/interior-depth
interpretation. Native replay and real-world deployment are not validated.
Existing historical media and study baselines are untouched.

## Reproduce offline

From the repository root, use the existing numerical Python environment and a
separate media environment. Neither command launches a server or uses Viam:

```sh
python3 -m venv .runtime/media-venv
.runtime/media-venv/bin/pip install -r sircus/studies/requirements-media.txt
sircus/simulation/.venv/bin/python -m sircus.studies.render_learning_demonstration prepare \
  --bundle sircus/simulation/web/media/demo-reel/showcase.json \
  --output .runtime/video/learning-geometry.json
.runtime/media-venv/bin/python -m sircus.studies.render_learning_demonstration render \
  --bundle sircus/simulation/web/media/demo-reel/showcase.json \
  --geometry .runtime/video/learning-geometry.json \
  --output sircus/simulation/web/media/demo-reel/learning-demonstration.mp4
```

The two environments may use different Python versions. Intermediate FK geometry
and QA frames stay in ignored `.runtime/`. Matching bundle hashes are required;
changing the bundle requires preparing geometry again. Font rasterization can
vary across macOS/Linux, so regenerated media may differ byte-for-byte while
retaining identical evidence. The numerical tests do not require Pillow/ffmpeg:

```sh
sircus/simulation/.venv/bin/python -m unittest discover \
  -s sircus/studies/tests -p test_learning_demonstration.py -v
```
