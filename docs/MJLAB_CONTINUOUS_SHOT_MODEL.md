# Predicting shot settings between bucket positions

The model predicts changes to seven shot parameters from a saved successful
shot and a new bucket position. The parameters are board X, Y, height, tilt,
heading, twist, and the release gap. The saved joint positions initialize
inverse kinematics. The existing pose and path checks decide if both arms
can reach the predicted setting.

Start with the audited data at the 33 fixed grid positions. Freeze 24
intermediate collection positions, one per rectangle, and a separate set of
50 test positions before collecting any flights. All new positions are at
least 5 mm from the inventoried prior coordinates and each other.
Use the fixed collider and tabletop geometry to screen bucket placements.

At each collection position, propose up to 24 shots by adapting nearby
successful shots. Vary board translation, heading, tilt, and release gap.
Simulate every proposal that passes setup at 64 substeps. Keep both successes
and misses in the evidence. The model fits standard captures. Each position
with an eligible capture has equal weight in a ridge regression of shot
changes between nearby, compatible successful examples. Freeze the model
before proposing the test settings.

For the test, both methods start from the same ordered list of up to eight
known shots. The nearest feasible baseline uses the first saved setting
that passes setup. The model adjusts each source setting for the new bucket
position and uses its first feasible prediction. Exclude predictions that
match a saved training geometry. Give each method one attempt per position.
A setup rejection counts against that method. The score requires a board
bounce and one second of whole-ball retention. Table and rim bounces may
qualify. Arm-assisted captures have their own outcome.

Run `python -m sircus.mjlab.intergrid_proposer_study --help` for the
producer and `python -m sircus.mjlab.intergrid_proposer_audit --help` to
reconstruct the evidence. The producer uses the authorized 4090 lease.
The audit reads the saved simulation traces on CPU. Keep all raw observations,
proof files, and checkpoints in the experiment output directory outside Git.
