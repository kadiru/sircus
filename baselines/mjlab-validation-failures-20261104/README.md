# Frozen-benchmark validation failure diagnosis

CPU-only analysis of existing calibrated-v2 validation traces, with no new shots,
policy changes or TRAIN/TEST native-packet access. The original benchmark report,
manifest and audit hashes are unchanged; the original scoring is preserved.
See [method and observability limits](../../sircus/mjlab/VALIDATION_FAILURE_AUDIT.md).

All864 validation requests were audited at the four separate checkpoints. The
primary216-request cohort uses each method/seed's already-selected checkpoint;
it is a subset of those milestones, not additional evaluation. These observations
already selected checkpoints and are not a fresh or unbiased performance estimate.

| Selected validation outcome | Guided sweep | PPO |
|---|---:|---:|
| Requested shots |108 |108 |
| Captures |70 |74 |
| Native releases |90 |93 |
| Setup rejects |18 |15 |
| No whole-ball entry |17 |16 |
| Retention disqualified by external contact |3 |3 |

There were no missing accepted board rebounds, invalid-physics outcomes or
entry-without-retention outcomes in the selected cohort. Across both methods,
29 of33 no-entry trajectories crossed the nominal whole-entry-height plane short
of the bucket's sphere-clearance region. Four remain unknown after intervening
contact. This describes observed trajectories; it does not prove a drop-speed,
height or tilt cause. Non-clean captures are assisted **or marginal**; the label
does not prove rim assistance in each case.

The12-layout heatmap has120 separate method/checkpoint/layout cells, each with
nine requests. It does not interpolate coverage across the table.100 deterministic
representatives include74 recorded native trajectories and26 rejected requests
with no trajectory. The live read-only reader verified74 GET replays,26 expected
404s and POST405. All pose comparisons are initial native reset/FK observations.
Among704 native validation requests, maximum board-centre error was0.000285311mm,
normal error0.000025953°, ball-spawn position error0.000041596mm and reported
flange error0.0330524mm. Sampled cached board displacement was0. These numerical
initialization checks do not establish dynamic settling, impact pose, physical
accuracy or compliant response; board/joint pose histories were not recorded.

Exact diagnostic report SHA256:
`5904953c31553affab867ed462c5d75e7302a8b48b8008fb085b6ab280e6e09c`.
Producer source checkpoint: `c33cde16947cfd0ea3b4ce0e22277e98ed840688`;
the report also binds the diagnostic source file, all66 inspected input files,
and the original passed1,204,473-check audit. Two independent metadata reviews
reconciled source bindings, all cells, selected budgets and original validation
curves. Full repository mock/headless checks and focused reader/UI checks passed.

This compact bundle contains exact diagnostic metadata, the original derived-file
manifest and the live-reader verification receipt. Dense observations and all
replay trajectories remain outside Git on Ubuntu. The derived manifest references
those unbundled replay files; SHA256SUMS covers only this compact bundle. Re-running
the trace audit requires the separately retained original raw benchmark evidence.

Open [the separate read-only view](http://127.0.0.1:18792/#validationFailureAudit).
Choose a method/checkpoint, select a bucket point for counts, then use **Replay
recorded miss** on an available example. The viewer holds initial arm geometry
and replays actual ball samples. No existing service was restarted and neither
GPU was used by this diagnostic.
