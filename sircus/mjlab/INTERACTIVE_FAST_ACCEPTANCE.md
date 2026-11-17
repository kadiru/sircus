# Live fast-planner acceptance

This checks the [interactive integration](INTERACTIVE_FAST.md), not a newly trained model.
The comparison uses **10 new bucket positions**, with the fast planner and the original live
`PhysicsPlanner` each getting one plan and at most one explicitly requested drop per position.
It does not substitute the previous study's 30-second baseline for the original UI planner.

The geometry-only sample contains four interior, four grid-edge and two near-base positions.
All remain at least 5 mm from the audited system and fast-planning inventories, including the
neural/budget study placements inherited by the fast study, its 50 test positions, and the
supplied old/new interactive evidence directories. `positions.json` freezes the full exclusion
inventory and selected coordinates before any test planning. Method order alternates by position.

Run on the loopback service host with its pinned environment. Supply the same arguments for
each stage; `--history-evidence` can be repeated for older interactive evidence directories.

```bash
python -m sircus.mjlab.interactive_fast_check freeze \
  --base-url http://127.0.0.1:18781 \
  --system-study /path/to/intergrid-system-v1 --system-audit /path/to/system-audit.json \
  --fast-study /path/to/fast-planning-v2 --fast-audit /path/to/fast-planning-v2-audit.json \
  --evidence-root /path/to/new/service/evidence \
  --history-evidence /path/to/old/service/evidence \
  --output /path/to/interactive-fast-acceptance-v1 \
  --display-report /path/to/display.json
```

Replace `freeze` with `run` to execute, then `audit` to repeat CPU-only verification.
`--gpu-wait-s 3600` optionally checks for foreign 4090 compute jobs before each drop; it never
stops them or takes their lease. The native worker still acquires the shared GPU lock and checks
occupancy. A GPU wait is separate from planning latency. The 4070 and Isaac are untouched.

The runner writes an intent before each POST. After interrupted drop submission it only polls
the existing job ID; it never submits another drop. An intent that was never accepted is counted
as uncertain/unreleased. An unknown plan-submission outcome requires manual inspection. A
timeout while polling retains the job for resume. A process lock prevents two runner instances.
Keep the service and evidence directory intact when resuming.

Every measured result must match the accepted preview, requested bucket, exact joints and
release request hash. The final CPU audit verifies native trace hashes/IDs, 64-substep time,
independently recalculates the scalar capture score and route labels, and matches the displayed
trajectory to the saved native trace. Duplicate and stale drop requests must be rejected.
The unchanged native engine/scoring retain their existing audited controls; this integration
test does not run a new calibration or retune physics.

All 10 positions remain in each denominator, including setup declines and failed/expired drops.
A capture means the existing standard direct/table route and one continuous second of whole-ball
retention; robot-assisted outcomes stay separate. `releases` counts completed measured results,
so `releases + no_setup + failed = positions` at completion.

The additive display key is `interactive_fast_acceptance`:

- `status`, `audit_status`, `progress: {completed, planned: 20}`.
- `comparison.fast/classic`: `positions: 10`, `captures`, `releases`, `no_setup`, `failed`,
  `planning`, `backend_request`, `drop`, and `worker_reused` (request count).
- Timing summaries have `samples`, and when nonempty `mean_s`, `median_s`, `p90_s`, `max_s`.
  `planning` measures HTTP submission through observed ready/declined state; `backend_request`
  is the service's measured planner call. `drop` includes native startup, GPU wait and scoring.
- `paired`: `both`, `gained` (fast only), `lost` (classic only), `neither`.
- Compact `positions`, `scope`, `next`, evidence hashes and `updated_at_unix`.

Final capture comparisons should display only after all 20 attempts finish and the audit passes.
This small acceptance batch establishes UI behavior and observed timing/captures, not coverage
of every tabletop position or statistically proven superiority.
