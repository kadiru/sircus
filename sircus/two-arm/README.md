# Sircus: two-arm Viam simulation

The active dashboard at http://127.0.0.1:8765 connects through the Python SDK to the cloud-bound Viam runtime on this Mac at127.0.0.1:8082. Both arms use the pinned official UFACTORY850 model with Viam fake drivers. Robot10 picks/drops the modeled ball; robot14 positions and holds its suction-mounted board.

Machine: https://app.viam.com/machine/5db5a9df-2e56-4711-9d39-b42cc86e4d79/3d-scene

Data: https://app.viam.com/data/all?robotId=5db5a9df-2e56-4711-9d39-b42cc86e4d79&view=files&org=6d6c7293-bc67-43e4-9c8b-31c76aac27d3

## Use

1. Set board position/orientation and container location in the dashboard.
2. Learn release places arm14 using Viam motion planning, then evaluates24 release proposals and10 fresh noisy tests. It saves the selected policy before testing.
3. Run drop executes the selected release with Viam motion planning and TCP readbacks for both arms. Replay only replays already computed visualization.
4. Inspect Data > Files, filtered to sircus-sim, for session, trial, policy and summary JSON files.

The learner searches board-local U/V and drop gap for one board pose. Proposal feasibility checks use inverse kinematics; each proposal does not run a native collision-planned trajectory. Use Run drop to validate the selected action's motion path. This is a small simulation search, not a generalized policy across unseen board poses or a hardware-trained controller.

Completed policy and held-out results survive a module restart when the scene and source fingerprint still match. Changing geometry or physics invalidates that recovered policy. See [verified run](STATUS.md).

## Data and physics

The 200×300×5 mm board dimensions are user measurements. The board is mounted 90° around the suction axis for landscape orientation. Its world Euler angles are [0,30,90]°, preserving the earlier −30° roll plane normal and cup orientation. The 0.4 kg board mass, centered suction support, 100 mm tool offsets, support/contact stiffness and damping remain provisional. Base14 is [1337.31,0,0] mm from the user’s 52.65 in spacing measurement, interpreted base-to-base along existing X. The 180° yaw and Y/Z remain from the prior configuration. Board center [600,0,350] mm is a feasible simulated setup, not a measured installation.

The target is an open-top, straight-sided tapered bucket. Measured inner diameters are 177 mm at the opening and 150 mm at the bottom; measured outer diameters are 200 mm at the top and 158 mm at the bottom. Ball contact and capture use the inner frustum; robot clearance uses the outer taper. Viam represents that clearance conservatively with 24 rotated wall boxes plus a floor bounding box, leaving the top open. These faceted collision proxies can slightly overestimate solid volume and are not used for ball contact.

The 175 mm measurement remains **reported height**, with overall height versus interior depth unresolved. Simulation explicitly assumes 175 mm interior depth and an unmeasured 8 mm bottom thickness, producing a provisional 183 mm outer height. Bucket placement is provisional: center XY [680,245] mm and inner floor Z 27 mm.

The contact model includes normal displacement and two tilt degrees of freedom around the support. Center/edge impacts can yield different rebounds. It omits board bending, ball spin and full coupled robot dynamics. Synthetic success requires settling within the container. A10/10 noisy test result only evaluates these assumed dynamics and small release perturbations.

Trial records include scene parameters, seed, action, reward, impact region, compliance telemetry, source hashes and validation scope. All data is labeled synthetic and physically_verified:false. `data/archive/<session>/` retains immutable local evidence; Viam Data Manager uploads disposable copies from `data/sync-outbox/` and removes those copies after success. A zero queue alone is not cloud verification; check Viam Files. Training runs on the local Viam module, while Viam Cloud stores the records and policy files.

## Current bucket validation

Session `20261029T232904069256Z-2a7bbd1f` evaluated 24 proposals and retained the ball in 10/10 fresh synthetic tests, with ±2 mm XY and ±3 mm release-gap perturbations. Selected release: U −16.408 mm, V −44.821 mm, gap 161.715 mm. Native replay `20261029T233247252271Z-67818bd3` passed eight Viam motion plans/readbacks and retained the modeled ball. All 40 new files were visibly verified in Viam Cloud.

The completed policy restores only with the matching current scene/source fingerprint. In-memory changes to either diameter, depth, or board mount correctly rejected it. The 75 earlier rectangular-scene archive files match their pre-change SHA-256 manifest; their results remain historical. See [bucket verification](data/bucket-verification.json), [cloud evidence](data/bucket-cloud-upload-verification.json), [live geometry inventory](data/bucket-wss-verification.json), and [status](STATUS.md).

## Restart

With the cloud runtime stopped, start it using:

```sh
cd /Users/kadirumacbook/Documents/ChatGPT/hackathon/sircus/two-arm
./start-viam-server.sh
```

Start the dashboard in another terminal:

```sh
cd /Users/kadirumacbook/Documents/ChatGPT/hackathon/sircus/simulation
.venv/bin/python server.py --port 8765
```

The default dashboard fails closed if Viam is unavailable. `--offline` explicitly selects the older standalone engine. The module validates the exact cloud part/configuration, loopback binding, fake-only resources and pinned models. Connection credentials remain in the existing private file; never copy them into source, URLs or the dashboard.

## Next measurements

Board mass; suction attachment offset from board center; flange-to-board offset; clarification of the bucket’s reported height and measured bottom thickness; final bucket placement; table and pickup coordinates; verified base transforms; center/edge drop recordings at known heights. Physical hardware deployment and camera-based outcome checking are separate next steps.
