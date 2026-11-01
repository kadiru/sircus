# Completed IK and distance diagnostic — October 31, 2026

[Machine-readable aggregates](summary.json) · [Method and reproduction](../../sircus/isaac/IK_DISTANCE_AUDIT.md)

The RTX4090 completed **215 PhysX shots in 318 seconds**. No new training or weight
updates occurred. The 4070 remained idle. This is a controlled diagnosis, not a
replacement for the original randomized test results or a generalization claim.

## Findings

**Board IK was frequently a solver-budget problem.** All 62 historical Arm14
failures reproduced under the original solver. Increasing iterations recovered
29; additional starting guesses recovered 25; eight remained unresolved. All
16 valid controls solved with the original solver. Recovered solutions retained
the original strict acceptance threshold, with worst position error 0.0994 mm
and board-normal error 0.0162 degrees. The eight unresolved cases had position
errors 0.136–0.630 mm and normal errors 0.0054–0.0275 degrees. They did not sit
at a joint limit (minimum margins 0.0596–0.1143 rad), but that does not prove
reachability. No tolerance was relaxed and no failed case was relabeled.

**Both policies miss feasible layouts even when IK succeeds.** At each of these
five fixed-board layouts, original and extended board IK produced the same
capture/miss outcomes. Thus this slice isolates shot selection from the board
solver failures seen in the randomized study.

| Bucket distance | 100-layout PPO | 1,000-layout PPO | Fixed warm action | Grid captures / all candidates | Grid-selected tilt / gap | Fresh repeats |
|---|---|---|---|---|---|---|
| 200 mm | Miss | Capture | Capture | 10 / 35 | 16° / 200 mm | 3 / 3 |
| 250 mm | Miss | Capture | Capture | 17 / 35 | 36° / 150 mm | 3 / 3 |
| 300 mm | Capture | Capture | Capture | 18 / 35 | 36° / 150 mm | 3 / 3 |
| 350 mm | Capture | Miss | Miss | 11 / 35 | 36° / 200 mm | 3 / 3 |
| 400 mm | Miss | Capture | Miss | 3 / 35 | 28° / 100 mm | 3 / 3 |

Each policy cell represents one shot under each solver, with matching outcomes;
fixed action is one shot. These small counts do not estimate reliable success
rates. Grid selection is by capture then reward, **not a trained prediction**.
The three fresh repeats use identical geometry and no added noise. All 15
captured, establishing repeatability only in this narrow simulation setup.

**The largest tested gap is unsupported by the current dropper IK.** Seven
candidates at every distance (all tilts at 250 mm gap) failed Arm10 IK, totaling
35 invalid shots. They stay in the 35-candidate denominator. An independent
full-orientation audit solved 60–200 mm gaps with the original solver; the 250 mm
gap still failed after larger budgets and eight other starts (best position
error 4.985 mm, orientation error 0.330°). This is not a proof of physical
unreachability. We did not expand bounds or substitute a different tool angle.

**Farther is less forgiving here, but difficulty is not strictly monotonic.**
At 400 mm only 3/28 IK-valid grid actions captured, compared with 18/28 at 300 mm.
The best tilt/gap does not increase monotonically with distance: wall/rim
interactions and retention also matter. This coarse grid does not prove which
continuous action is optimal or bound all reachable solutions.

**Model targets and actual articulated poses also differ.** After-rollout board
body readback was 5.1–13.2 mm below the requested center. This was not sampled
just before release, and its cause has not been isolated. Before relying on
submillimeter IK residuals as physical accuracy, audit settled pre-release board
pose/normal and actuator tracking separately.

## Next bounded checkpoint

1. Add the tested board multi-start solver behind a new versioned contract;
   preserve failure reporting and validate actual settled poses. Add dropper
   feasibility checks before sampling actions. Do not silently clamp or relax
   tolerances, or claim collision-safe travel from endpoint IK.
2. Re-evaluate frozen policies with the improved solver on the existing diagnostic
   set, clearly labeled retrospective. This measures how much the 54 rescued
   endpoints actually improve capture, without retraining.
3. Run a short, distance-balanced learning sanity study. Inspect policy action
   sensitivity versus bucket distance, captures by distance, and invalid-action
   rates. Existing observations are already normalized; the near-constant output
   in the completed study is evidence of weak conditioning, not proof of a
   specific network/optimizer defect. Compare against the fixed action and the
   separately labeled grid reference. Use fresh disjoint final test layouts and
   multiple seeds for the next confirmatory study.
4. Only then increase training budget or start quasi-perception. More layouts
   alone will not fix solver failures or demonstrate conditioned behavior.

## Provenance and limits

- UTC run: `distance-sweep-20261101T010024Z`; original physics fingerprint and
  selected checkpoint/source hashes are in `summary.json`.
- Mac local evidence: ignored `.runtime/ik-distance-audit/` under the project root.
- Ubuntu: `~/sircus-isaac/output/distance-sweep-20261101T010024Z/` and
  `~/sircus-isaac/output/ik-distance-audit-v1/`; source snapshot
  `~/sircus-isaac/studies/ik-distance-v1/`. `input100` is a local copy of the
  completed `general-rl-20261031T205024Z-n100-seed42` outputs from the 4070 workspace.
- No raw observations, joint records, checkpoints, generated assets or credentials
  are committed. Their evidence digests are retained in the aggregate report.
- The original random-layout evaluation remains unchanged. Training source is
  unchanged; extended IK is diagnostic-only. No new long training job runs.
- Simulation uses rigid board mounting and provisional bounce parameters. Bucket
  reported height remains 175 mm with overall/interior interpretation unresolved.
  No hardware, Viam, audio, physical controller or 4070 workload changed.
- Validation: full project mock/numerical suite passed; final Isaac suite 42 tests,
  including IK/report checks; two diagnostic DOM tests passed. HTTP verified the
  served completed report, all 175 heatmap cells and updated page. Browser visual
  inspection was unavailable in this tool session, so it is not claimed.

The read-only preview was reloaded to publish the panel. The finite diagnostic
process exited; the preview and existing CPU perception workers remain. Stale
sensor frames are not new observations.
