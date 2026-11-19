# Sircus engine contact comparison

Compared 24 selected configurations with 3 unperturbed repeats. Protocol is fresh_paired_fixed_endpoint_contact_replay.

| Source outcome stratum | Configurations | First captures / valid | Second captures / valid | Capture agreement |
|---|---:|---:|---:|---:|
| clean_capture | 8 | 24 / 24 | 23 / 24 | 23 / 24 |
| other_miss | 8 | 0 / 24 | 0 / 24 | 24 / 24 |
| rim_contact_miss | 8 | 0 / 24 | 18 / 24 | 6 / 24 |

First engine is isaac_physx; second engine is mjlab. Initialization failures are 0 and 0 respectively.

A complete success uses the shared board-rebound, whole-ball entry and continuous one-second retention scorer. Rim-assisted captures qualify. External table, floor, robot or unknown contact disqualifies the complete route.

- Configurations were stratified by original Isaac outcome. Aggregate capture rates describe this diagnostic selection, not arbitrary-table policy performance.
- Unperturbed repeats test numerical repeatability. They do not establish robustness to perception, contact or calibration error.
- Fixed joint endpoints isolate release/contact physics. Grasping, path planning and interactive constrained policy search are separate tasks.
