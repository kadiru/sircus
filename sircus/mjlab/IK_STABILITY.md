# Stable endpoint IK before more training

This opt-in solver/search protocol fixes query-order instability without editing
historical solver, physics, scorer, policy, or result files. The ten remaining
development targets are original indices 8,9,10,12,21,23,26,35,39,41. The two
already eligible coordinates, 19 and 48, remain outside the targeted search.

## What changed

The old solver assigned random starts by batch slot and stopped all queries when
the entire batch had converged. Adding or reordering unrelated queries could
therefore change a request's solution or lose a previously valid endpoint. It
also selected the smallest mathematical residual before checking that branch
for native collisions.

`stable_ik.py` reuses the original URDF kinematics/Jacobians and strict 1e-4
position/weighted-pose acceptance. Starts are hashed from each target and role;
each branch converges independently within a fixed iteration budget. No global
RNG or another query determines its starts or convergence. Internal polishing
aims for 1e-8; the acceptance threshold is not loosened.

`stable_geometry.py` checks candidate joint pairs through the same native joint
limits, forward kinematics, bucket transforms and endpoint collision gates.
A colliding first branch no longer discards other valid branches. Supplied
source joints are preferred only after these gates. They do not certify arm
transit or that a ball's flight avoids the robot.

Exact nominal reuse requires the original source bucket and action values to
match the current request exactly, plus strict IK/native validity. A changed
request uses those joints as continuation seeds and polishes toward the new
target. Without source-request provenance, only a tightly matching FK result
can remain unchanged. This prevents small jitter from being silently absorbed
inside the looser 2 mm/0.5° native presentation gate.

## Validation boundaries

The native study audits 16 source queries: six captured controls plus one source
for each of the ten targets, including duplicate requests. Original, matched
stable 4-start/90-iteration, stable 8/120 and source-seeded modes are tested in
full, reversed and single-query batches. Raw masks, failure reasons, query/start
identities and joints are retained. Numerical joint tolerance is 1e-9 rad.

The controlled native search keeps 8/120 and source continuation. A separately
labelled CPU-only cold-start diagnostic also tests larger budgets. More starts
are not counted as a same-budget algorithm improvement. Known-source recovery,
cold recovery, endpoint invariance and ball-capture repeatability are distinct
measurements. A deterministic finite solver can still fail to find a valid pose.

## Ten-target native search

`stable_study.py` identifies the exact ten empty-mask rows from the preserved v2
development cache and verifies its checksum. It uses historical measured source
joints only as explicit geometry seeds. Fresh native throws decide outcomes.
The full existing wider action envelope is retained; no release velocity,
off-center impact family, moving board or contact-material change is introduced.

The proposal acquisition prioritizes qualified direct capture, then unassisted
entry/retention progress. A read-only 240 Hz observer measures opening proximity
before the first table/floor/robot/unattributed contact, so a later table-assisted
entry does not dominate the search. The unchanged scorer still requires a board
rebound, whole-ball entry and one continuous second inside without disallowed
external contact. Rim-assisted bucket captures still count.

Each source must capture in the native search, pass at least 8/10 independently
perturbed screening attempts, then pass at least 16/20 separately seeded final
attempts. Geometry rejects remain failures. Noise is Gaussian, clipped to the
declared bounds: 1 mm board X/Y, 2 mm board height/gap and 0.35° tilt. A selected
development confirmation is not a population reliability guarantee.

The original 17 templates remain unchanged. A new template records its source
request, exact native joints, branch IDs, cache location and confirmation
provenance. Translated targets use those joints only as seeds; they cannot
inherit source success or nominal-preservation status. This new catalogue needs
a matching stable-IK adapter before any future policy study or deployment.

There is **no PPO or model training** in this task. Current solve controls retain
the previous policy. The new read-only `#ikStability` panel presents endpoint
validation, native controls, search outcomes and both confirmation stages.

## Reproduction and isolation

Use the existing pinned mjlab environment and a new immutable source/output
directory. Run CPU checks with `CUDA_VISIBLE_DEVICES=""`. For native work expose
only the 4090 UUID and supply the existing shared `scene.lock`; the runner waits
without signalling or changing any other job. Isaac assets are read-only inputs;
its source/setup and the 4070 workload are outside this task.

The CLI requires `--parent-study`, `--prior-study`, `--prior-expansion`, one or
more `--source-study`, `--assets`, `--output` and `--gpu-lock`. The checked search
uses `--batch 128 --native-per-target 1024 --proposals 256 --proposal-rounds 96
--top 8 --control-repeats 3`.
`--audit-only` runs the endpoint audit without creating a GPU context.

`verify_stable.py` audits exact native-source hashes, parent identities, the
stored permutations, original tolerance/FK/collisions, source/hint identities,
nonabsorbed jitter, native-release accounting, both confirmation stages and bank
prefix preservation. It is CPU-only and does not independently rerun dynamics.
Run it from its separate auditor snapshot with `--source` pointing at the exact
native snapshot, plus the parent/prior directories and native assets.

Keep raw trials, force/pose arrays, generated configs and logs outside Git.
The bucket's reported 175 mm height interpretation and all provisional scene and
material assumptions remain unchanged. Historical results are not relabelled.
