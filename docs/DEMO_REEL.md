# Two simple demonstrations

`/demo.html` is a static numerical replay page linked from the four-mode show desk.
It offers a fixed four-ball routine and a recorded learning-study sequence. It
does not send robot commands, fit a model, run a search, or contact Ghost Booth.

## Start on Mac or Ubuntu

Use the project Python environment (`setup-ubuntu.sh` creates it on Ubuntu):

```sh
./start-demo.sh
# Open http://127.0.0.1:18772/demo.html
```

`SIRCUS_PYTHON` can select an existing environment and `SIRCUS_DEMO_PORT` a free
port. The launcher binds loopback, strips audio-service/Viam environment variables,
and fixes physical mode to disabled. It starts an independent process; it never
stops another server or starts a Viam cloud part. Source synchronization does not
run the launcher. Ctrl-C stops only this foreground preview.

Choose **Four-ball routine**, enable **Local bounce sounds** if wanted, and press
Play. Sound defaults off. Local original WAVs provide drums while the illustrated
dropper carries and settles, and color-specific generic voices on saved numerical
contacts. Stop, switching reels and disabling sound cancel active voices and stale
cues. Browser sound needs a user gesture; its failure never blocks the animation.
These are model-timed sounds, not measured ODD/BLE events. The physical audio
workflow remains separate and still requires identified, measured inputs.

## Fixed preset

The nominal front-facing preset is copied unchanged from the published facing
rehearsal: board RPY `[0,-30,90]` degrees, U/V `[0,0]` mm, release gap160mm.
Four fresh seeds `2026103001` through `2026103004` are retained without retries or
action reselection: **4/4 numerical captures**. This is one presentation layout,
not a generalization benchmark. Pink/Purple/Green/Gray label identical mechanics.
Every trial starts with an empty bucket; accumulated-ball dynamics and collisions
between balls are not modeled.

The release gap is world-Z clearance between nominal contact and release ball
centers. It is not a calibrated flange height. Board200×300×5mm and tapered bucket
inner diameters177/150mm, outer200/158mm remain unchanged. The measured reported
height175mm still has an unconfirmed overall-height/interior-depth interpretation.
Placement, mounts and contact parameters remain provisional.

## Learning evidence

The page reuses the completed `front-bucket-20261030` study and actual frozen
checkpoints trained on8,24,64,128 layouts. Mean capture over three training seeds
on the same48 held-out layouts is59.0%,67.4%,67.4%,69.4%. The prespecified N128
checkpoint captures33/48; fixed action25/48 and separately labeled per-scene search
37/48 are comparison methods, not additional model inference.

The paired animation shows the first predeclared held-out layout at each training
size. **All four misses remain visible.** A separately labeled saved successful
example follows. Population improvement is not a promise that each layout improves.
No model is retrained or optimized during playback; the builder verifies each
action against its saved checkpoint and original result. Historical rectangular
and earlier bucket results retain their original definitions and fingerprints.

The model predicts board roll/pitch, impact U/V and release height. Displayed arm
movement interpolates endpoint geometry to explain those choices; it is not a
learned joint trajectory or a collision-checked native path. Ball trajectories and
contacts are saved numerical physics. Capture uses the versioned board-bounce,
whole-ball-entry and continuous-one-second retention evaluator. Physical transfer
and native-path validation remain unverified.

## Rebuild and inspect

```sh
python -m sircus.studies.build_showcase --output .runtime/demo-rebuild
python -m unittest discover -s sircus/studies/tests -p 'test_showcase.py' -v
SIRCUS_PYTHON=path/to/project/python scripts/test.sh
```

The builder requires a new output directory, hashes its source/evidence and keeps
every predetermined trial. The portable bundle is
`sircus/simulation/web/media/demo-reel/showcase.json`, with `SHA256SUMS`. Film
rendering instructions and media checksums are in that directory's `LEARNING_VIDEO.md`.
The historical film and study have not been overwritten.

Next: use the existing mock workflow for new board/bucket layouts, then commission
native paths and physical contact/height calibration before any real deployment.
More simulated successes alone cannot establish that transfer works.
