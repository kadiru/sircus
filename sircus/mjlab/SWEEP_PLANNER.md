# Simulation sweep-based shot planner

This independent interface accepts a bucket center anywhere inside the tabletop
footprint limits. It searches board XYZ, bucket-facing tilt and release gap,
using the corrected native MuJoCo/mjlab contact model. A finite unresolved search
does not establish impossibility or continuous table coverage.

This is **search, not RL inference**, and the viewer is a single workcell with a
7 x 5 farm-target selector. It is not an Isaac Sim viewport or 35 simultaneously
live renderings. Existing learned-policy readiness and historical studies stay
separate.

## Use

1. Open the separately launched planner. Select one of the 35 farm targets, drag
   the bucket, or enter its X/Y coordinates in millimetres.
2. Leave both positional locks off initially. Optional locks preserve board XYZ
   or the released ball-center XYZ; incompatible constraints are rejected.
3. Choose a native **search** budget of 16-128 candidates (default64), then click
   **Find drop-bounce posture**. The UI checks four times the native budget in
   geometry proposals(default256); geometric rejects are reported separately
   from native releases. An explicit backend request can use up to1,024 proposals.
4. Inspect the result and replay its recorded native ball trajectory. A saved
   exact-target recipe is explicitly labeled recorded evidence; it is not a new
   physics trial or a learned prediction.

The corrected catalogue supplies 17 confirmed source settings at 12 of the 35
grid locations. Only exact bucket/scene/asset/physics identity permits direct
reuse. Translated or rotated source geometry is a proposal and always needs new
admission, native simulation and confirmation. Other table positions are queries,
not promised successful targets.

## New-position search

- Source-relative recipes lead; corrected inverse-ballistic proposals and
  deterministic broad samples cover additional board locations, heights, tilts
  and gaps. Analytic predictions never assign capture labels.
- The stable16-start/180-iteration CPU IK and independent5mm geometry gate check
  self/cross-arm, board, table, bucket, tool and carried-ball clearance.
- The selected exact joint branches execute under the corrected64-substep
  passive-only contact contract. Native contact/position/velocity records are
  independently rescored before qualification.
- Up to three captured candidates receive common-home sampled geometric path
  admission and fresh confirmation. A ready result requires **3/3 nominal and
  at least18/20 fresh** captures, with rejected variations kept in denominators.
  Confirmation requests are additional to the displayed search budget and are
  reported separately.
- The fresh variation standard deviations are board XY1mm each, H2mm, tilt0.35
  degrees and gap2mm. Explicit positional locks are preserved.

Capture requires a board rebound, whole-ball entry and one continuous second
inside, without external assistance. Rim-assisted captures count. Material,
mount and contact properties remain provisional. The bucket's reported175mm
height interpretation remains unresolved; the original assumed175mm interior
plus8mm base is preserved.

Arm joints are prescribed before each native substep. Neither the endpoint shot
nor the sampled geometric setup path demonstrates physical grasping, driven
actuator tracking or realistic impact-time arm compliance. Sound and Viam are
not involved.

## Results and busy states

- `verified_shot`: qualification passed; inspect actual parameters and counts.
- `catalog_lookup`: exact saved native evidence, with original source and path
  checks. No native work occurs for that request.
- `busy`: another process owns the4090. Nothing is interrupted; retry later.
- `common_home_unresolved`: the prescribed staging pose is blocked; another
  staging strategy would need separate planning.
- `no_verified_solution_within_budget`: no captured configuration found within
  the finite proposal/native budget; not a proof of impossibility.
- `capture_found_confirmation_unresolved`: exploratory capture found, but path
  or repeatability qualification failed. The result is not marked ready.

Failures never fall back to the historical legacy-physics search or a policy.
Requests are serialized within this interface, and the shared4090 lock prevents
interference with other native work. The4070 workload is not used.

## Launch separately on Ubuntu

Use the pinned `requirements.txt` environment and an immutable source snapshot.
The existing corrected-grid raw evidence and generated collision assets remain
local; credentials are not required. A clean Git checkout alone does not supply
those generated assets/evidence. `load_reference()` checks the reference against
the committed compact publication, native source hashes, actual assets, packages,
scene and physics before enabling search.

```bash
export PYTHONPATH="$HOME/sircus-mjlab/studies/YOUR_PLANNER_SNAPSHOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$HOME/sircus-mjlab/venv/bin/python" -m sircus.mjlab.sweep_server \
  --reference "$HOME/sircus-mjlab/output/corrected-grid-v1" \
  --assets "$HOME/sircus-isaac/assets" \
  --gpu-lock "$HOME/sircus-isaac/scene.lock" \
  --cache "$HOME/sircus-mjlab/planner-cache-NEW" \
  --state-dir "$HOME/sircus-mjlab/planner-live-NEW" \
  --host 127.0.0.1 --port 18786
```

The server is loopback-only and has no GPU context between queries. Backend
environment settings are `SIRCUS_SWEEP_REFERENCE`, `SIRCUS_SWEEP_ASSETS`,
`SIRCUS_SWEEP_LOCK` and `SIRCUS_SWEEP_CACHE`, set by the launcher. They are not sent
to the browser. Child requests select only the4090 UUID, have a600-second lifetime
and parent-death cleanup, and release their GPU slot on completion/cancellation.

Each fresh query keeps unique attempt directories, compact request/source
identities, geometry rejects, shot records, native arrays and a checksummed result
bundle. Exact successful query caches verify every listed evidence hash on reuse.
They bind exact XY/locks, budgets, scene/assets/code and physics; changing inputs
does not inherit old success. Raw traces, cache, generated configs and logs stay
out of Git. A512MiB free-space floor and256MiB per-attempt budget stop new work
without deleting any existing evidence.

Run the following, the interface JS tests and the repository regression before
a source checkpoint:

```bash
python -m unittest sircus.mjlab.tests.test_sweep_contract \
  sircus.mjlab.tests.test_sweep_worker sircus.mjlab.tests.test_interface
```

Native new-position confirmation and UI playback are separate from those mocks.

[Published verification](../../baselines/mjlab-sweep-planner-20261102/README.md)
records two new development queries, their exact settings and independent audits.
