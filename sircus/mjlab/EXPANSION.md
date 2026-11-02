# Targeted shot-bank expansion

The twelve empty-mask positions from the original categorical study are now
**development targets**. They can influence proposal search and training. Their
subsequent scores are not new held-out evidence. The original bank, checkpoints,
test definitions and results remain unchanged.

## Procedure

`expand_geometry.py` generates bounded proposals and audits actual URDF joint
limits, native forward kinematics, board normal and endpoint collisions on CPU.
`expand_bank.py` then simulates eligible configurations with the original
1920 Hz integration and 240 Hz scorer. Its first phase retains the original
action bounds; a separately versioned wider phase allows lower boards and
larger drop gaps. Rejected endpoints remain failures in confirmation counts.

The wider envelope is board X300–1040 mm, Y−500–350 mm, board height 120–600 mm,
tilt 8–60° from horizontal and ball-bottom-to-board-center gap 50–650 mm. Release
XY remains the board center and heading faces the bucket. These limits do not
guarantee reachability. The unchanged IK/FK/collision gates remain necessary.

Each promoted source must be an actual route capture and pass at least 8/10
fresh, perturbed native throws. Noise standard deviations are 1 mm in board X/Y,
2 mm in board height/gap and 0.35° tilt, clipped to the declared envelope. This
small development sample is a selection rule, not an 80% population guarantee.

`expand_refine.py` separately retests high-mouth-clearance, diverse captured
settings from a completed search. It keeps inherited throws and fresh
confirmation counts separate. It does not turn table- or robot-assisted entries
into valid routes, weaken the capture definition, or repair old results.

## Retraining and comparison

`expanded_study.py` warm-starts the old9→64→64→1 categorical actor/critic and
performs actual on-policy terminal PPO with native rewards. It samples 175
declared training layouts: the 35 old grid positions,12 converted development
positions and 128 new random coordinates. Empty action masks are excluded from
training sampling but remain failures in complete evaluation denominators.

Thirty fresh validation layouts select a checkpoint by captures, then reward.
Another 100 fresh test layouts, at least 5 mm from every historical/development,
training and validation coordinate, evaluate the frozen checkpoint. Each method
uses three nominal repeats. The four primary comparisons are old bank/uniform,
old bank/learned, expanded bank/uniform and expanded bank/learned. A fifth
old-bank/wider-bounds diagnostic separates envelope changes from template growth.
Uniform baselines share a seeded per-layout quantile. Shared original-template
slots reuse the same audited joint branches and masks across banks; adding a
template cannot change old candidates through IK batch-slot reseeding. No evaluation method
optimizes a shot using that query's outcomes. Bootstrap differences group all
repeats of a bucket position; nominal repeats are not independent layouts.

Original templates stay an unchanged prefix. Expanded checkpoints have their
own policy class and native environment contract. Existing inference guards
deliberately reject the new class until a matching adapter is commissioned.
The live interface adds a read-only study panel; its previous policy stays active.

## Reproduction and isolation

Use the pinned `requirements.txt` environment and a new immutable source/output
directory for each run. Point `--parent-study` at the completed continuous study,
`--old-bank-study` at its completed categorical follow-up and `--assets` at the
matching URDF/mesh assets. Launch only with the 4090 GPU UUID visible and the
existing shared lock supplied through `--gpu-lock`. Jobs wait for that slot
without stopping any other process. Headless checks hide both GPUs.

Run `verify_expansion.py`, `verify_refinement.py` and `verify_expanded.py` on CPU
against the exact source snapshots and evidence. The expanded-study verifier
requires the parent, original-bank, expansion and prior-search directories as
appropriate. It checks provenance, masks, native endpoints, source IDs,
confirmation counts, PPO weight/log-probability history and frozen comparisons.

`prepare_expanded_report.py` appends compact results to a derived display JSON.
Use a backed-up base display, never an experimental report as its output. Static
UI files and the derived JSON can be updated without restarting the preview.

These remain privileged-geometry, held-endpoint simulations. Grasping, arm
transit, moving-board interception, RGB-D perception and physical validation are
outside this experiment. The provisional ball/mount materials and the bucket's
reported 175 mm height uncertainty are preserved. Isaac source, setup and the 4070
workload are outside this task.
