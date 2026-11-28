# Checked-shot preservation study

This folder contains sanitized summaries only. Raw simulated trajectories,
prepared arm/path proofs, native command receipts and runtime configurations
remain in the Ubuntu experiment archives and are not committed to Git.

`diagnosis.json` summarizes the nine setup declines in the independently
audited `approach-comparison-v3` experiment. It preserves original case IDs,
bucket coordinates, checked-candidate counts, rejection reasons, and the
rank of the current planner's chosen geometry in the expanded candidate menu.
The parent audit and evidence-manifest hashes identify the source evidence.

The experiment protocol is in
[MJLAB_PRESERVED_SHOT.md](../../docs/MJLAB_PRESERVED_SHOT.md). Development
repeats and fresh first-drop evaluation must be reported separately. No model
is trained by this study, and the live planner is not automatically changed.
