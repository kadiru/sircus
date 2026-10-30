# Headless orientation/height capture contract

This contract applies to `sircus/two-arm/train_orientation_height.py` v2 and
`shot_rollout.rollout`. It is synthetic integration evidence, not hardware
validation, a collision-planned arm path, or a generalized learned policy.
The existing orientation/height search still varies roll, pitch and gap while
fixing board center/yaw, bucket placement, world release XY and tool orientation.

## Versions and compatibility

| Field | Value |
| --- | --- |
| Experiment/policy/row schema | `sircus-orientation-height-v2` |
| Rollout schema | `sircus-shot-result-v1` |
| Scoring | `sircus-board-bounce-capture-1s-v1` |
| Physics | `sircus-compliant-board-bucket-impulse-v1` |
| Fingerprint format | `sircus-shot-contract-v1` |

`physics.simulate(..., capture_mode=SCORING_VERSION)` opts into the new scorer
and bucket solver. Omitting the keyword preserves the legacy penalty contacts,
slow/low 0.2-second `retained` rule, reward and record fields. The existing
dashboard, Viam module and older training loop therefore remain legacy callers.
The historical 10/10 result remains legacy 0.2-second evidence. Neither it nor
the separate Ubuntu Task 1 0/100 is relabeled or rewritten by this integration.

New rollout results use `success`, `outcome` and `shaping_reward`; they omit
legacy `retained` and `reward`. Candidate selection is lexicographic:
`(success is True, shaping_reward)`. Shaping is 2 for any whole-ball entry plus
the best downward opening-crossing clearance in metres; absent crossing uses
-1 for that clearance term. A high shaping value is not a capture label.
Invalid actions, invalid contacts and model-limit results cannot be selected.
Both candidate rollouts and held-out tests call the same `rollout` function.

Policies are frozen before held-out evaluation. Loading checks schema,
scoring/physics versions, normalized scene/source/dependency fingerprint,
the complete manifest and reconstructed parameters/action/coordinates.
Fingerprint normalization treats integer/float and signed zero consistently
only for these new artifacts. Any changed source file covered by the manifest
invalidates the policy conservatively, including comment-only changes. Original
fingerprint algorithms and historical files are unchanged. No automatic migration
or legacy-flag promotion exists.

**Native replay is not yet an adapter for this contract.**
`replay_orientation_height.py` rejects v2 policies before `connect()` because
the runtime's `run` route still uses legacy dynamics/scoring. It also explicitly
rejects v1 policies as incompatible with the new verifier. Use the unchanged
historical source revision to inspect historical policy behavior; do not bypass
the guard. A separately reviewed runtime integration is required for v2 native
motion replay. The UI and Mac physical backend are unchanged.

## Complete shots, fixtures and events

`CaptureTracker` requires these events, in order, for a complete-shot success:

1. An observed finite-board contact followed by outgoing motion along the
   nominal board normal, before the qualifying whole-ball entry.
2. A downward crossing of the bucket opening plane whose center clearance is
   `inner_top_radius - ball_radius - radial_distance >= -1e-6 mm`.
3. Whole-sphere containment inside the actual tapered interior, above the bottom
   and below the rim, continuously for at least one simulated second.

The board bounce flag uses this reduced-order board model's first separation
velocity and nominal normal. It is not a calibrated spin/compliance measurement.
Containment is checked at every physics step, with `1e-6 mm` round-off tolerance
and `1e-9 s` clock tolerance. A center-plane crossing alone never suffices.
The scorer records downward crossings, whole-ball entries, containment losses,
bounce-outs, full exits and board bounce separately from contact telemetry.
Absent crossings have null clearance and signed miss XY, not invented zeros.

Rim contact before the qualifying entry produces `capture_class= rim_assisted`;
otherwise a successful entry is `clean`. Both count identically. This is an
event-based classification, not proof the rim contact was causally necessary.
Any loss of full containment resets the dwell timer. Partial protrusion over
the top counts in `bounce_outs`, but retains opening eligibility so re-entry
does not require a center crossing the trajectory never made. A complete sphere
exit clears eligibility and requires a fresh valid crossing. Dwell never sums
across exits. A subsequent successful re-entry still needs a new full second.

`episode_kind=contact_fixture` allows explicit incoming position/velocity for
isolated contact tests. It may yield `bucket_retained_1s=true` and
`outcome=fixture_retained`, but can never set complete-shot `success=true`.
These overrides are rejected in complete shots. Normal launches have zero
velocity. Full-shot fixtures use the board and nominal release routine.

| Outcome | Meaning |
| --- | --- |
| `captured` | Board bounce, valid full entry and continuous one-second containment |
| `no_entry` | Entry window closed or ball escaped without whole-ball entry |
| `bounce_out` | Whole entry followed by top protrusion/exit, without later capture |
| `containment_lost` | Full containment lost elsewhere, without later capture |
| `timeout` | Horizon cut an unfinished episode or current dwell short; `truncated=true` |
| `no_board_bounce` | One-second containment without a board bounce preceding entry |
| `fixture_retained` | Bucket-only fixture met retention; not shot success |
| `invalid_action` | Invalid/unsupported input or unreachable endpoint; not an executed drop |
| `invalid_contact` | Caller marks the episode contact evidence invalid |
| `model_limit` | Solver overlap/correction, support range/contact or numerical divergence limit |

Model limits override capture. At the hard horizon, an ongoing dwell takes
`timeout` precedence over an earlier bounce-out; event counts retain that history.
Initial/re-entry dwell must start by 4 s. A dwell that starts in time gets its
full second; the wrapper horizon is 5.00025 s. A late partial exit does not grant
a new dwell after the entry deadline. Legacy 0.2-second settling cannot stop this
rollout. A deliberately shortened low-level horizon produces a timeout rather
than a false capture.

## Contact model and assumptions

The finite 200 x 300 x 5 mm board and spring-supported normal displacement/tilt
are preserved. The upright open bucket retains the existing analytic annular
rim, tapered inner/outer walls and bottom. The new mode uses sphere projection
and restitution/friction impulses, iterating up to four active contacts so a
wall/floor corner does not leave an unresolved second surface.

Defaults: timestep 0.00025 s; restitution 0.35; friction from `ball.friction`
(default 0.2). Unresolved overlap above `1e-6 mm`, or any contact projection above
2 mm, terminates as `model_limit`. Contact telemetry includes normals,
corrections, incoming/outgoing velocity and kinetic energies. These checks and
three-timestep regression fixtures detect tested instabilities; they are not
continuous collision detection or a universal tunneling guarantee. Unsupported
bucket orientation fields are rejected, rather than silently ignored.

No new engine or package is introduced. NumPy/SciPy are already pinned. The scene
continues to identify 175 mm interior depth plus an unmeasured 8 mm base as a
provisional construction: reported 175 mm overall height versus interior depth
is unresolved. Ball diameter 75 mm/mass 0.2 kg, bucket placement, tool offset,
support stiffness/damping and contact parameters remain uncalibrated assumptions.

## Coordinates and records

- World Z is up. Positions and board-local U/V/gap are mm; rotations are degrees,
  with `R = Rz(yaw) Ry(pitch) Rx(roll)`; mass is kg and time is simulated seconds.
- U is along the board's 200 mm dimension, V along its 300 mm dimension. U/V
  are not world XY. Rotation is about the unchanged board center, not a tool pivot.
- `release_point` constructs `center + R*[U,V,thickness/2+ball_radius]`, then adds
  gap along world Z. Gap is neither distance along the normal nor TCP Z.
- `h_m = (commanded ball-center world Z - nominal first-contact ball-center
  world Z)/1000`. The reference is the highest intersection of the vertical
  release line with the sphere-dilated finite board box. A miss has no reference
  and is rejected. The trainer verifies its selected top-face construction has
  `h_m*1000 == gap_mm` within `1e-6 mm`; this is checked, not assumed generally.
- `actual_h_m` uses sampled release and first-board-contact ball-center Z, or null
  without contact. Contact-time discretization/compliance/noise can change it.
- The inherited kinematics assume ball center at gripper TCP (zero translation,
  identity relative rotation) and the existing `DOWN` tool orientation. Both are
  recorded and provisional, not measured ball-in-gripper calibration.

Each experiment/policy has a manifest of physics settings, coordinates, scene
and relevant source/model/dependency hashes. Trial rows cite its fingerprint and
store candidate/effective scene, action, commanded/actual release data, event
records, outcome, shaping and separate endpoint-IK results. The CLI keeps its
existing individual JSON-file format; shared TrialStore/replay-envelope support
is deferred. `rollout(..., frames=True)` supplies numerical trajectories for
offline inspection; the small CLI search stores events with `frames=False`.

Held-out tests preserve the existing independent uniform release perturbations:
X and Y each +/-2 mm, gap +/-3 mm, seeds `seed + 104729*(j+1)`. These synthetic
uncertainty assumptions do not vary board/bucket poses or claim calibration.
No native motion or collision-planned path is executed by this trainer.

## Verification

Run `scripts/test.sh` with `SIRCUS_PYTHON` set to the installed project Python.
It includes focused contact/scoring fixtures, legacy regressions, mocked
orientation selection/compatibility checks and Mac controller/adapter mocks.
No server or hardware connection is needed. For a small integration smoke,
use `train_orientation_height.py --budget 4 --tests 2 --seed 42 --output PATH`
with a new output directory. This is not generalization evaluation or a request
to deploy, start native replay, or run a larger experiment.
