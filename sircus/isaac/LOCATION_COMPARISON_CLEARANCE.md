# Saved-data clearance investigation

This diagnostic investigates the separate seed-22 location comparison whose
CPU recovery audit stopped on `minimum_mouth_clearance_mm`. It is not another
audit attempt, training run, simulator replay or replacement for the failed
audit. The original shutdown failure and recovery-audit failure remain recorded.
The completed three-seed sweep/PPO benchmark is a separate, unchanged experiment.

Clearance is the gap between the ball's cross-section and the polygonal bucket
opening during entry, expressed in millimetres. The original clean-entry rule
requires at least 1 mm clearance plus its contact and exit conditions. Historical
capture is calculated separately; the published capture also requires the
original board-route condition. A scalar mismatch does not by itself establish
that either label changed.

The diagnostic reads the sealed recordings and imports the unchanged original
`ShotScore`, `EntryQuality`, `BoardRoute` and comparison helper. It preserves the
240 Hz observations, per-cell capture/fall stopping rule and absolute comparison
tolerance of 1e-7 in each field's units. It records disagreements rather than
turning any disagreement into an audit pass.

Three arithmetic paths isolate the possible cause:

1. The auditor's float64 subtraction of JSON world origins from saved positions.
2. Float32 coordinate subtraction and float32 scorer inputs, corresponding to
   the producer's position representation; library-version differences still
   matter in a cross-version diagnostic.
3. Float32 coordinate subtraction followed by float64 scorer inputs. This is a
   separate arithmetic experiment to distinguish coordinate rounding from
   scalar arithmetic, not an assumed replacement for the producer environment.

The report compares every returned scorer field and records both directions of
catch/clean-label changes for individual paid requests. It excludes inactive
padding from denominators and groups results by model, phase, milestone and
held-out cohort. Opposite label changes cannot cancel unnoticed in totals.
Sealed hashes, source identity, request association and full observation-clock
coverage are prerequisites for reporting complete coverage.

The default native-data mode requires the producer's NumPy version. A separately
requested cross-version diagnostic reports the actual environment and cannot
claim producer-runtime parity or a successful scientific audit. Neither mode
loads policy weights, changes a scoring rule, promotes a model, repairs saved
files, or starts a simulator. Raw recordings and detailed per-request outputs
stay outside Git; only compact evidence belongs in the repository.
