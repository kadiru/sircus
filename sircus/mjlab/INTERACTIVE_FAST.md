# Fast planning in the bucket-placement UI

The interactive task is: place the bucket, choose a checked pose for each arm,
then explicitly run one simulated drop. This integration uses the frozen
early-return selector from [the planning-speed study](FAST_PLANNING_RESULTS.md).
It does not train another model or change the contact physics or capture rule.

## Operator flow

1. Drag the bucket on the table, or enter its X/Y coordinates.
2. Leave **Fast checked planner** selected and press **Plan shot**.
3. Inspect both arm poses and the release point. Planning uses a 30-second
   decision budget; worker startup and evidence verification add wall time.
4. Press **Run one drop** to simulate the exact previewed configuration at
   64 substeps. A completed result can be replayed without another simulation.
5. If no checked shot is found, select **Original planner** and plan again.
   This is an explicit fallback, with the original planning behavior and no
   30-second decision limit. There is no automatic second drop or retry.

Changing the bucket or planner invalidates the previous preview. Controls are
locked while a plan or drop is running. Duplicate and expired drop requests are
rejected by the backend, not merely by disabling a button.

## What changes

The fast planner reuses initialized CPU workers after clean completion and may
return the first fully checked candidate meeting the frozen predicted-clearance
rule. A timed-out or failed worker is discarded. Each request gets a new evidence
directory, including when the same bucket coordinates are requested again.
Arm and path checks remain required. The archived request and exact preview
are bound to the native drop by hashes.

The original planner remains available under the `classic` API identifier.
An older server invocation without the verified fast-study inputs continues to
serve only that planner. The updated browser also works with such a server.

To enable both planners, retain the existing `interactive_physics_server`
arguments and supply all six verified artifact paths:

```text
--fast-study /path/to/fast-planning-v2
--fast-audit /path/to/fast-planning-v2-audit.json
--neural-study /path/to/neural-selector-v1
--neural-audit /path/to/neural-selector-v1-audit.json
--budget-study /path/to/planning-budget-v1
--budget-audit /path/to/planning-budget-v1-audit.json
```

This defaults to fast; `--default-planner classic` explicitly retains the old
default. The service fails closed if a supplied model, source or audit differs.
Use the pinned Ubuntu environment and preserve the archived source references
listed in the fast study's contract. Deployment is a separate simulation-server
action; fetching the branch alone changes no running service.

## Scope and limits

- Inputs are simulator geometry, not camera or point-cloud estimates.
- Native tests start at the checked held poses. They validate ball dynamics;
  they do not execute or validate a full pickup-to-release arm trajectory.
- Capture requires a board bounce, whole-ball entry and one continuous second
  inside the bucket. Table and rim assistance are accepted; robot-assisted
  outcomes remain separate.
- A feasible preview is not a promise of capture. The final ball simulation
  decides the outcome, including contacts and misses.
- Planning runs on CPU. A drop leases only the authorized 4090 and waits for
  unrelated GPU work. The 4070 and Isaac environments are not modified.

## Fresh-placement acceptance

The acceptance comparison uses ten coordinates fixed before testing: four
interior, four near table edges and two near robot bases. Both planners receive
the same positions through the actual HTTP API, with alternating method order
and at most one drop per planner per position. Previous training, study and
interactive coordinates are excluded. Missing setups and failed attempts stay
in the ten-position denominator.

Report planning time separately from GPU waiting and drop time. Compare the new
fast method with the actual original interactive planner, not with the bounded
`warm_30` control from the earlier study. Ten positions are an integration check,
not a claim of complete table coverage or a statistically established winner.

The results panel uses `interactive_fast_acceptance` and preserves historical
reports. Raw requests, trajectories, checkpoints and local launch records remain
outside Git. See the acceptance checker and its protocol for reproduction and
resume rules.
