# Current checkpoint

The latest [randomized perception validation](PERCEPTION_VALIDATION.md) completed
144 observations. The GPU study has exited; observation workers and the preview
remain. The older runtime statements below are historical.

See [RGB-D perception and completed RL status](PERCEPTION.md). The foundation
snapshot below is historical: RL now exists, the4070 is stopped by user request,
and the4090 is replaying a frozen policy with independent CPU perception.
Recheck actual processes before any deployment.

# Isaac foundation checkpoint — 2026-10-30 Pacific

This is synthetic simulation evidence, not physical validation or a trained policy.
Observed runtime state below must be rechecked before future changes.

## Installed and running

- Ubuntu workspace: `~/sircus-isaac`. Dedicated container: `sircus_isaac_4090`.
- Isaac Lab `v2.3.2` / commit `37ddf626871758333d6ed89cf64ad702aef127d0`.
- Actual mounted binary: Isaac Sim `5.1.0-rc.19+release.26219.9c81211b.gl`.
  The reused Docker image tag says 6.0.1; the read-only 5.1 mount determines the actual runtime.
- Private Python3.11 venv; bundled torch2.7.0+cu128. No global Python/driver update.
- Only the RTX4090 is exposed to the Sircus container. Limit: 12 CPU cores, 28GiB RAM.
- Existing `aiworker_pouring_4070` retained its original GPU assignment and start
  time (`2026-10-29T07:16:02Z`). No restart, reconfiguration or workload signal.
- Read-only preview: Ubuntu loopback `http://127.0.0.1:18860`; Mac SSH forward
  `http://127.0.0.1:18780`. Viewer has no robot/simulator command API.
- Continuous scene holds starting joint targets and renders the fixed RGB-D view.
  It does not autonomously pick, drop, bounce or train. Nothing auto-starts at boot.

## Completed verification

Final finite capture: `output/20261031T023110Z/` on Ubuntu (not committed).

| Check | Result |
|---|---|
| Geometry unit tests | 4 passed on Mac and Ubuntu |
| Existing project regression suite | `scripts/test.sh` passed, including physical mocks, numerical/scoring contracts, study UI and the new dependency-light geometry tests |
| Manufacturer URDF → USD import | Both arms imported; 6 arm joints each, plus 2 gripper-proxy joints on arm10 |
| Simulation stepping | 180 rendered frames / 6 simulated seconds at 240Hz physics, 30Hz render |
| Wrist command response | Requested +0.05rad; measured arm10 +0.0499992, arm14 +0.0499654 |
| Attached board | Center (0.670115, -0.099995, 0.396280)m; rigid wrist offset 0.09999996m |
| Board facing | Normal (-0.005913, -0.422787, 0.906210); faces negative-Y bucket, landscape long edge along X |
| Bucket opening/floor | Ball fell through opening; settled center Z=0.0455000m, matching 8mm base + 37.5mm radius |
| RGB-D | 640×480 RGB and axial metre depth; 149,557 valid depth points |
| Camera transform | 35,650 reconstructed points within 2mm of known synthetic tabletop |
| Saved observation | RGB PNG, depth NPY, colored optical-frame PLY, intrinsics/world transform, joint/body states and provenance |

The board's approximately 3.7mm gravity sag reflects this provisional actuator
configuration. Command response is not trajectory collision validation. The
bucket probe is not a complete board-bounce-and-retention success measurement.
Camera points are from rendered visible surfaces; the tabletop test uses known
synthetic geometry and does not claim learned object detection.

The live browser preview was verified with advancing frames and about149,390
valid depth pixels. The final scene holds a ball on the loading area; the finite
smoke placed a separate test ball over the bucket to test its opening/floor.

## Storage and dependencies

At validation, the new workspace used approximately 2.4GB. Root had about 29GB
free; `/data` about 6.3GB free. No existing datasets, caches or logs were deleted.
Reused the large existing simulator/image; no second full simulator download.
Live preview replaces bounded latest files, and timestamped captures are opt-in.
The scene stops without deleting anything if free output storage drops below 3GiB.

`pip check` is **not globally clean**: the existing bundled optional
`nvidia-srl-usd[-to-urdf]` packages advertise missing `usd-core`, an `lxml<5`
constraint versus bundled5.4, and bundled `msal` advertises missing `pyjwt`.
These optional export/cloud paths were not used. Kit's own USD, URDF import,
articulations, camera, Torch and saved-capture inspection were exercised successfully.
We did not install another USD build into Kit to silence dependency metadata.

Startup can take a few minutes on a cold shader cache. Sim5.1's stage/Replicator
cleanup hung on earlier smoke runs; after synchronous capture writes the script
uses the documented `SimulationApp.close(skip_cleanup=True)` exit path. The final
smoke completed and exited cleanly. A lock prevents concurrent Sircus scene runs.

## Still missing

- RL task reset/observations/actions/reward/termination and PPO training.
- Board-bounce + whole-ball entry + continuous-one-second retention scoring in PhysX.
- Collision-checked cooperative trajectories; self-collision currently disabled.
- Grasp/release validation; current gripper is a geometric proxy.
- Compliant suction/board mount, ball parameter calibration and timestep convergence.
- Depth noise, segmentation/localization and held-out perception-driven evaluation.

Bucket diameter measurements and unresolved175mm height interpretation are
preserved. Table, placements, tool offsets, masses and contact parameters remain
provisional. No old simulation score or policy is claimed valid in this backend.

## Dense shot catalogue checkpoint

See [completed settings and robustness results](../../baselines/isaac-shot-catalogue-20261031/README.md). Fixed400 mm distance;579 searched settings,240 initial confirmation trials,100 independent follow-up trials. Recommended starting point H400 mm/gap130 mm/tilt21° passed140/140 clean captures across the tested nominal and narrow perturbation conditions. Other settings were substantially less robust. Simulation search, not new RL or physical calibration. Finite4090 jobs completed; preview read-only;4070 and hardware untouched.
