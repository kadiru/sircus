> Superseded on October 29, 2026: the active dashboard now connects to the two-arm cloud-bound `sircus-sim` runtime on port8082. Robot10 drops; robot14 holds the measured200×300×5mm board. See [current two-arm setup](../two-arm/README.md). The notes below describe the preserved earlier single-arm baseline; use `--offline` to run the old local dashboard engine.

# Sircus simulation

One simulated UFACTORY 850 picks a 75 mm / 200 g ball, releases it over a floating
board, and attempts a bounce into an open container. Robot **10** is the drop-arm
role. Robot **14** is represented by the floating board; a second arm is not yet
modeled. These numbers are role labels, not Viam cloud machine identifiers.

## Open and run

The fast local viewer runs at <http://127.0.0.1:8765>. To restart it:

```sh
cd /Users/kadirumacbook/Documents/ChatGPT/hackathon/sircus/simulation
.venv/bin/python server.py --port 8765
```

Change board XYZ / roll / pitch / yaw, container XY, or board-local U/V and release
gap. **Run drop** replays the calculated sequence. **Learn release** evaluates 24
release proposals, freezes the selected action, then runs ten fresh noisy tests.
**Measurements & support dynamics** exposes the complete editable configuration.
Applying valid JSON saves `scene.json`. Changes to board pose through the main
controls affect the current experiment; apply the JSON to persist them.

The local viewer renders the official 850's box geometries and joint kinematics.
Its interpolated joint path is a visual preview; it does not claim collision-free
motion. The learner rejects actions with unreachable nominal arm waypoints.
Physical grasps, visual detection, automatic ball retrieval, and real robot
control are not implemented. Each trial starts with a modeled ball at the pickup.

## Native Viam adapter

`viam_module.py` adapts the transferred project's WorldStateStore and motion
sequence to the 850 and the new contact model. It streams the table, board, five
container solids and ball, and uses Viam's motion service with obstacles and TCP
readback. Its standalone configuration has only fake resources and binds to
127.0.0.1. It rejects physical drivers, cloud configuration and remotes.

Generate machine configuration after moving this directory:

```sh
.venv/bin/python build_viam_config.py
viam-server --config "$PWD/viam-config.local.json"
```

On this Mac the verified portable runtime can instead be started with
`./start-viam-server.sh`. `data/viam-native-validation.md` records the completed
native trial; `.venv/bin/python check_viam_native.py` repeats it.

`run-viam.sh` is the module launcher, not the server launcher. The generated
`viam-config.local.json` contains absolute local paths. `viam-scene.json` is the
native Viam scene configuration; it is separate from the fast viewer's `scene.json`.
Use the `sircus` generic component's commands:

```json
{"probe":true}
{"run":true,"action":{"u_mm":0,"v_mm":0,"gap_mm":200}}
{"status":true}
{"stop":true}
```

`{"set_scene": <complete validated configuration>}` updates that native scene
while idle. The adapter logs all runs, including aborted motions, to
`data/viam-runs/`. No physical arm has been contacted.

## Dynamics and learning limits

The board has one spring-supported normal displacement and two tilt degrees of
freedom around the supplied suction support point. Off-center contact generates
torque, changing the board motion and the ball's rebound. Contact uses a positive
compliant normal force with dissipative friction. Board pose uses
`Rz(yaw) Ry(pitch) Rx(roll)`. Physics runs in SI; the external configuration uses
millimeters, degrees, kilograms and seconds. The default step is 0.5 ms.

This is a provisional reduced-order model. It omits board bending, ball spin,
tangential/yaw mount compliance and full coupled support inertia. Support and
material parameters have not been measured. The nominal pose includes preload
and sag. Board deflection resets for each trial. Large support motion and board
collisions with other solids produce `model_limit`, not a fabricated successful
trajectory. Invalid initial overlaps are rejected.

Success requires the ball wholly within the bin, near its floor, slower than
60 mm/s for 0.2 s. First floor contact is recorded separately and is not success.
The bin contact damping is a provisional material assumption. Even a 10/10 test
result describes only this model and these release perturbations.

The optimizer searches U/V/gap for the current supplied board pose. It is a small
observation-driven search, not a general learned policy across unseen poses or a
joint-level reinforcement learning controller. Test release noise is ±2 mm XY
and ±3 mm height; it does not test uncertain stiffness, camera error or hardware.
`runs/<timestamp>/policy.json` is written before held-out outcomes are generated.
`result.json` preserves every proposal, test outcome, exact action and geometry
fingerprint. Test outcomes never change that saved policy.

## Measurements still needed

- Board length, width, thickness and mass.
- Suction support position relative to the board center, including height offset.
- Container inside length/width, wall height and floor height; table dimensions.
- Arm-base locations, pickup location and flange-to-ball-center tool offset.
- Center and edge drop recordings at known heights to fit support/contact dynamics.

The initial board is 280 × 260 × 12 mm, mass 0.4 kg; these are placeholders.
The 100 mm tool offset and gripper box are placeholders too. The user's tested
board attachment is accepted; this model focuses on bounce behavior.

## Source and verification

Started from the October 29 transfer at
`../../imports/viam-transfer-20261029/viam-ball-hackathon` from this directory.
Its 613 manifest files matched
and its 34 pure checks passed. That imported example and historical results are
preserved. The new engine does not reuse its old rigid-bounce success rates.

Run the new physics checks:

```sh
.venv/bin/python -m unittest check_physics
```

Fifteen checks cover center/edge differences, board movement, yaw, timestep
convergence, energy sanity, genuine misses, capture vs floor contact, seeded
noise, support offsets and initial overlap rejection. Native Viam execution
evidence is separate in `data/viam-runs/` and the current `STATUS.md`.

The pinned official [Viam UFACTORY 850 model](https://github.com/viam-modules/viam-ufactory-xarm/blob/84e216609a1350b416b4b5ca5818cdc0e64f1d22/arm/xarm850_kinematics.json)
and license are in `models/`. ODD Ball size/mass come from its
[manufacturer product listing](https://oddballism.com/en-us/products/odd-ball).
Contact formulation background: [Drake compliant contact documentation](https://drake.mit.edu/doxygen_cxx/group__compliant__contact.html).
Three.js 0.170.0 is vendored under its MIT license in `web/vendor/`; no CDN is
required at runtime. Python dependencies are in `requirements-viam.txt`.
