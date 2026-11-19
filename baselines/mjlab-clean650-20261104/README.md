# Fixed650 coarse height/tilt/gap search — November 4, 2026

Native calibrated MuJoCo/Warp simulation, held articulated endpoints, one fixed
board/bucket XY layout. This is development search, not policy evaluation.
The [protocol](../../sircus/mjlab/CLEAN650.md) preserves scene, materials, scoring,
ball and provisional bucket geometry from the prior calibrated650 replay.

| Phase | Requested | Native | Clean captures | Rim captures |
|---|---:|---:|---:|---:|
| Initial controls |12|12|6|0|
| Coarse plus reference |361|47|0|1|
| Local refinement |81|54|0|1|
| Five settings, matched verification |30|30|0|3|
| Final controls |12|12|6|0|
| Total |496|155|12|5|

**No clean650 setting qualified.** All12 clean captures are positive controls;
none is a discovery result. The two discovery captures repeat the same original
board380mm, tilt28°, gap386.8mm, ball-center release804.3mm, bucket650mm setting.
Its verification was3/3rim-assisted at64 and0/3capture at128. The other four
verified settings missed0/3 at both resolutions. These are nominal repeats, not
noise robustness. The five settings'350ms paired trajectory differences were
1.189–2.364mm; outcome agreement still failed for the rim-sensitive original.

All before/after positive controls captured cleanly3/3 at each resolution and
negative controls missed0/3 with valid physics. Independent CPU audit passed
**64,075 checks**, including dense rescoring, frozen plans, strict5mm endpoint
checks, source/assets/packages, native options and byte-bound artifacts.

There were341 endpoint rejects across the full request budget. In the coarse
grid,305 requests had no resolved board/dropper IK branch and9 failed strict
self-collision clearance; the reference was admitted. These finite solver
outcomes do not prove physical impossibility. The fallback opening-distance
ranking can favor late post-table motion on misses; only actual clean/contact
verification can qualify a recipe. A separately frozen fine neighborhood study
follows without modifying or relabeling this result.

Files:

- `summary.json`: allowlisted configuration/outcomes, control and qualification
  gates, full source/asset/calibration/evidence hashes and audit identity.
- `requested-shots.csv`: all496 requested shots including rejects, configured
  parameters, outcome and provenance IDs.
- `checksums.json`: SHA256 of both exports. Raw joints, trajectories, caches,
  generated runtime state, host paths and credentials are excluded.

Producer commit `e98991ca65d3a8c0dda6d40dc8e937432f5f1935`;
contract `28baa4e28639d0e7581561bf546a42de2a738ef4fd6a1521230d69887424f28b`.
No PPO update, noise study, service restart, physical validation or policy
replacement.4090 only;4070 workload and Isaac setup unchanged.
