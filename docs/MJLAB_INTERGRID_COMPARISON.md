# Sircus intergrid comparison

Train a binary capture ranker on the 33 usable fixed bucket positions and
evaluate whether it chooses better shots at positions between them.

The dataset combines completed historical native64/v3 observations with the
audited 13,000-drop sweep. Training uses valid releases at the 33 canonical
coordinates. Identical shot geometry and joint commands share one record.
Standard captures are positive. Misses, including arm-assisted captures,
are negative. Conflicting repeat labels are retained in the evidence and
excluded from fitting. Each position contributes 1/33 of the training loss.

The model sees target bucket XY, candidate board and release geometry, and
the joint positions of both arms. Its score ranks candidate shots. The
comparison uses a 32-unit hidden layer and 120 full-batch Adam updates on the
RTX 4090. Save the fitted checkpoint before selecting any test shot.

The test has 50 positions that are at least 5 mm from all inventoried
historical coordinates and from each other. Sample two in each inter-grid
rectangle and two near the fixed arm bases. Freeze the positions from
geometry before the fit. The model's candidate menu contains up to two
recorded shots from each of the four nearest training positions. Pose and
path checks recompute the joints at the target bucket using the saved
joints as initial guesses. The nearest-shot baseline uses the best-supported
recorded shot from the nearest training position. A rejected setup counts
against that method. The model chooses its highest-scoring admitted shot.
Report the paired outcomes both over all placements and where both methods
released, to distinguish setup advantages from flight outcomes.

Freeze all choices before the first test drop. Each method gets one attempt
per position at 64 substeps. Run the original nine controls before and after.
Use the same board bounce and one-second whole-ball retention rule. Table and
rim assistance can qualify. Arm-assisted outcomes have a separate label.

Run `python -m sircus.mjlab.intergrid_study --help` for the input paths.
Select the authorized 4090 UUID and the shared 4090 lease. Run
`python -m sircus.mjlab.intergrid_audit --help` to reconstruct the dataset,
frozen choices and results and to rescore the saved traces on CPU. Keep the
observation ledgers, dense traces and output checkpoint in the study
directory outside Git.
