# Paired two-arm engine comparison — November 2, 2026

This is a simulation-only comparison of the same shot states in Isaac PhysX and mjlab/MuJoCo Warp. Twenty-four unique configurations were selected from the earlier cylinder study, eight each from source clean captures, rim-contact misses and other misses. Each was replayed three times. The source outcome strata deliberately bias this diagnostic sample.

Both fresh engines use identical bucket positions, recorded joint configurations, prescribed ball release positions and zero initial ball velocity. All 72 initializations per engine passed the board and dropper pose gates. Both arms are articulated, but joint angles and velocities are reset before each inner physics step to isolate the release/contact comparison. Physics runs at 1920 Hz with eight substeps per 240 Hz scoring tick. All 72 throws in each engine produced a board rebound.

| Metric | Isaac PhysX | mjlab / MuJoCo Warp |
|---|---:|---:|
| Attempts / valid starts | 72 / 72 | 72 / 72 |
| Complete captures | 24 | 41 |
| Clean captures | 14 | 5 |
| Rollout plus shared scoring | 80.11 s | 17.41 s |
| Measured setup and compilation | 10.45 s | 1.08 s |

The canonical comparison is [comparison.json](comparison.json), with a short [generated report](REPORT.md). An earlier matched mjlab batch gave 43 captures in 12.35 seconds; it is preserved in [repeat-batch.json](repeat-batch.json). Thus observed throughput was 4.6–6.5 times faster in these runners. This is headless simulation plus Python scoring, device readback and trajectory collection. It excludes rendering and does not measure physics kernels or RL training speed. Initial cold MuJoCo Warp compilation took much longer than cached startup.

The final engines agreed on capture/miss for 53/72 pairs. Isaac repeated the same binary outcome for all 24 configurations; mjlab did so for 21/24. Most engine disagreement is in rim-contact cases. Higher mjlab capture count is not evidence that the unchanged actions or learned policy improved. Both engines use the same ShotScore, EntryQuality and BoardRoute code. Success requires a board rebound, whole-ball entry and one continuous second inside. Rim assistance qualifies, partial exits reset retention, and intervening table/floor/robot/unknown contact disqualifies a direct route.

## Contact fidelity and checks

Normal contact was calibrated separately from bucket outcomes, on the actual 5 mm flat-board proxy. The frozen mjlab setting is solref(0.004,0.15), solimp(0.95,0.99,0.001,0.5,2), effective ball-contact friction 0.35, and eight physics substeps. The friction mapping starts from averaging source ball 0.2 and other-surface 0.5 coefficients. The runtime uses MuJoCo's default pyramidal cone and 100 solver iterations.

[Calibration summary](calibration-summary.json) preserves the selected candidate and the full calibration report's digest. Train drops at 100/300/600 mm measured restitution 0.616/0.664/0.737. Held-out 200/400 mm drops yielded 0.687/0.604. Maximum train penetration was 1.757 mm. No tested 960 Hz setting met the 2 mm limit. The 0.65 target is the source configured coefficient; it was not measured in a fresh PhysX flat-drop calibration. Contact equivalence remains unestablished.

The port initially exposed a static-collider cache issue. Bucket movement now uses a mocap body, and the runner validates both bucket body poses and all actual bucket collision-geom positions. The final collider error was zero. Native MuJoCo compilation and 32 independent URDF forward-kinematics checks also passed. The bucket retains 64 separate convex walls and an open cavity. Its 177/200 mm top and 150/158 mm bottom measurements, reported 175 mm height and unresolved height interpretation are preserved.

## Reproduction and artifacts

[Source and runnable commands](../../sircus/mjlab/README.md) use an isolated mjlab 1.6.0 environment at ~/sircus-mjlab. [Runtime versions](runtime.json) identify the checked package set. Original Isaac files, configuration and runtime remain in place. The 4070 workload was untouched.

The canonical raw results are ~/sircus-mjlab/output/matched-final-v1/result.json and ~/sircus-isaac/output/mjlab-paired-fine-v1.json. The repeated batch is matched-fine-v2. Full trajectories, model XML, caches and media remain outside Git. The comparison video uses a common native MuJoCo renderer to display the recorded GPU trajectories from both engines at fixed arm poses.

This establishes a two-arm asset/scoring port and a controlled contact benchmark. A new mjlab policy, interactive bucket dragging and constraint-conditioned pose search are subsequent work. The planned interface accepts a bucket location plus optional locks on either arm and returns a feasible shot configuration or a clear failure to find one.
