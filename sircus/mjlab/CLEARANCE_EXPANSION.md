# Wider search when predicted flight clearance is poor

Completed: [audited results](../../baselines/mjlab-clearance-expansion-20261116/README.md). Wider search captured 42/50 versus 40/50; two gains, no lost captures. Median planning time increased from 33.7 to 39.5 seconds. The active interactive planner remains unchanged.

This experiment compares two frozen planners on 50 fresh bucket positions. It changes how candidate shots are searched, not the learned model, simulator, or capture definition.

The current planner proposes direct and table-bounce shots from the existing learned model and saved shot libraries. It expands the geometry search only if no arm setup is feasible. The new planner also expands when the best feasible shot has a nonpositive predicted clearance score. That score is the smallest predicted margin to the robot, bucket opening, or board edge.

For a risky position, the new planner:

1. Tries up to 32 additional board-center, height, tilt and release-gap combinations.
2. Takes up to four promising feasible combinations and tries alternate board-arm joint configurations, using 16 deterministic IK starts and admitting at most four branches per combination.
3. Keeps the exact dropper joints and full board/release targets fixed during each joint-configuration search.
4. Chooses the highest-scored feasible candidate before observing any simulated drop.

The current shot remains available. Ties keep it. If every candidate still has a nonpositive score, the selected shot is labeled risky rather than hidden from the results.

## Frozen test

- 50 positions sampled with seed 2026112801, excluding the inventoried historical grid, training and test positions and the 11 saved interactive placements.
- One selected native drop per planner per position, at 64 substeps. Both receive the same bucket positions.
- Nine matched controls before and nine after the comparison.
- The existing capture definition requires a measured board rebound, whole-ball entry and one continuous second inside. Direct and table-assisted captures count. Robot-assisted captures remain separately classified.
- Feasible setups, captures, released misses, robot contacts, paired gains/losses, unchanged-command outcomes and CPU planning time are reported separately.
- No training on these test outcomes. No online trial-and-error drops are used to pick a test shot.

Preparation uses four CPU workers with CUDA hidden. Native drops acquire the shared 4090 lock and refuse a foreign GPU process; the 4070 and Isaac jobs are not changed. Raw poses, traces, generated configurations and logs stay outside Git.

## Reproduction

Use the same audited inputs and pinned environment as the parent `intergrid_system_study`. The new runner is `python -m sircus.mjlab.clearance_expansion_study`; its `--help` lists the required audited source studies. It additionally takes `--system-study`, `--system-audit`, and `--interactive-history`.

Run `--prepare-only` with CUDA hidden and a fresh `--output` directory. Preserve that directory: it binds the input models, layouts, planner source and selected commands. Then run `--run-prepared` with only the 4090 UUID visible. The worker waits cooperatively for the GPU lock. It must not be restarted as a new attempt after a partial native run. Finally run `--audit OUTSIDE_OUTPUT.json` with CUDA hidden. Publish comparison metrics only after that audit passes.

Relevant checks:

```bash
python -m unittest sircus.mjlab.tests.test_clearance_expansion_plan sircus.mjlab.tests.test_clearance_expansion_audit
node --test sircus/mjlab/tests/test_interactive_physics_ui.mjs
```

The UI section is `#clearanceExpansionStudy`. It reads the derived `clearance_expansion_study` report without changing the active interactive planner. A later decision to promote the new planner must consider capture changes and added planning time together.

## Limits

This is an endpoint-held simulated shot comparison with privileged geometry and a fitted flight approximation. CPU path checks are preparation evidence; the native drop holds the selected arm endpoints. The test does not establish physical performance, moving-arm execution, arbitrary positional locks, or success everywhere on the table.
