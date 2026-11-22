# Moving the board to clear three missed shots

The preceding planner captured at 17 of 20 new bucket positions. At the three
missed positions, the ball bounced from the board into the bouncer arm's middle
segment. Changing joint configuration while keeping the board fixed mainly
flipped the wrist and did not move that obstruction. See
[the recorded diagnosis](HYBRID_MISS_CORRECTION.md).

This development search keeps those three bucket positions fixed and allows
the board center and heading to change. Board height, tilt and release gap
may also vary to preserve or recover the required flight range. The dropper
must remain above the board, and both arms retain the existing full-pose IK
and geometric path checks. This is bounded configuration search, not training
or a new unseen-position evaluation.

## Protocol

1. Generate a deterministic menu around each original configuration. Use the
   existing approximate flight model to prioritize candidates; its predictions
   are not capture evidence.
2. Check arm reachability and paths on CPU, then predict ball-to-arm clearance.
   Preserve proposed, rejected, admitted and selected counts separately.
3. Freeze up to 40 admitted configurations per bucket before native execution.
   Run their ball flights on the RTX 4090 at 64 substeps.
4. Repeat every qualifying captured configuration three additional times.
   Keep those confirmation results separate from the exploratory captures.
5. Retest the three unchanged original configurations three times each and run
   matched simulator controls before and after the experiment. Audit saved
   native trajectories independently before publishing final counts.

A qualifying clear capture requires a board bounce, whole-ball entry and one
continuous second inside the bucket under the current direct/table-assisted
scorer. Robot-assisted retained outcomes remain separately identified. The
experiment does not relabel historical results or replace successful recipes.

## Scope and resources

The native experiment has a one-hour GPU budget and cooperatively acquires
the existing 4090 lease. The 4070 and Isaac workloads, source and setup remain
untouched. The simulation uses privileged geometry and holds the arms at
their checked endpoints during flight; geometric path checks do not establish
native driven-arm motion, camera perception or physical validation.

The new results panel is `#boardPoseRecovery`. Replays use recorded native ball
samples and the actual held arm poses, with normal-speed and quarter-speed
views. The interactive planner stays unchanged until a separately authorized
integration test. Raw observations, generated media and host runtime files
remain outside Git.

Results are pending. A completed result must identify the frozen source,
request and evidence hashes, independent audit, counts for every stage,
confirmation outcomes and exact settings for each recovered target.
