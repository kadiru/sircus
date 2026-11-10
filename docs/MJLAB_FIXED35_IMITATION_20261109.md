# Fixed 35-position imitation handoff

The user selected **35 fixed bucket locations first**, with positioning
perturbations deferred. The [study protocol](../sircus/mjlab/FIXED35_IMITATION.md)
uses the canonical seven-column, five-row grid, unchanged native64/version3
scoring and bucket-facing gripper rotation shared by both compared policies.

The implementation refits continuous imitation using retained teacher contexts
and two verified local bucket-facing corrections. It preserves prior successful
recipes and model files. Both models and all 70 predictions freeze before
evaluation. The 35 canonical positions are historical development locations,
so the comparison is labeled coverage/regression rather than unseen-position
generalization.

The additive `#fixed35Imitation` panel displays one cell per bucket location,
paired outcomes, exact coordinates and shot commands. Setup refusals are
separate from released misses and remain in the denominator. Completed results
require the independent source, checkpoint, control and trajectory audit.

Implementation status: results are pending; this file does not claim a completed
native run, improved capture rate, model promotion or noise qualification.
The final report must record source commit, compact result location, checkpoint
identities, complete counts and independent audit identity before changing that
status. The 4070 and Isaac setup remain outside this study; no service restart
or physical hardware action is part of publication.
