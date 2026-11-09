# Fresh held-out preparation and native guard adapter

This implements the next preparation step in the
[continuation plan](CANDIDATE_CONTINUATION_PLAN.md). It preserves the original
seed-22 model, completed benchmark, TRAIN bank and existing interfaces.
Training, simulator startup and new task or diagnostic shots remain off.

[The completed CPU preparation evidence](../../baselines/isaac-candidate-heldout-20261109/README.md)
records 847 qualified candidates at 102/105 positions after all 8,505 proposals,
with three empty positions retained. Preparation and independent audit took
146.568 s wall and 456.261 aggregate CPU-seconds. All 39 new CPU cases passed
without skips, including genuine native class resolution without initialization.
This is geometry/code verification, not a new capture result.

## Frozen fresh geometry

[Coordinate discovery](candidate_fresh_history.py) reads only `bucket_xy_m`
positions from unique published baseline JSON blobs at fetched origin refs and
task HEAD, retaining the original historical pool even if an old baseline is no
longer present. Outcome fields never affect inclusion. Null UI goals represent
unset positions and are ignored; malformed recorded coordinates fail discovery.
The inventory binds Git blob identities and raw SHA256 values. This is novelty
relative to discovered recorded coordinates, not all unpublished experiments.

The [new split contract](candidate_fresh_splits.py) draws exactly 35 validation
and 70 TEST positions with seed `20261126`, the original bounds and 7-by-5
stratification. Every point stays at least 25 mm from the discovered pool,
**all 140 original benchmark positions**, and all other fresh points. Every
tile remains required; no reachability/outcome filter, refill or relaxed spacing
is allowed. One million coordinate draws is a hard upper bound. A separate
audit checks distances and exact seeded draw replay.

The coordinate manifest is frozen and audited before any policy inference or
kinematic qualification. [Parent export](candidate_fresh_parent.py) then loads
the exact original checkpoint strictly, with CUDA hidden, and computes direct
mean/scales/actions for the new geometry. It checks saved and in-memory hashes,
CPU RNG and absence of gradients. It creates no optimizer or ball outcome.

## Finite candidate preparation

[Bank generation and qualification](candidate_fresh_bank.py) retain exactly
8,505 raw proposal aliases: 105 positions times the frozen parent's mean action,
16 Gaussian proposals and 64 Sobol proposals. The proposal seed is `20261127`.
There is no historical-success lookup, nearest-success transfer, refreshed
policy, flight ranking or outcome query. Exact physical duplicates may share
one record while all raw aliases remain accounted for.

Each unique proposal receives one original coupled-arm preflight using the
declared frozen canonical home. Original finite robust-IK multistarts, residual
tolerances, endpoint clearance and 16-sample path proxies remain unchanged.
The gap variable still means ball-bottom release Z minus board-center Z, with
the original 80 mm lower bound; it has not become a different tool-clearance
variable. Prepared fits only provide possible warm starts.

Unresolved IK is reported as unresolved. A rejected finite candidate does not
prove its bucket position impossible. A geometry-admitted non-parent action on
a saturated categorical boundary is explicitly excluded without clipping its
action. Empty supports remain in the final requested-position denominator and
are never refilled or removed. A passing preflight is neither a mesh/self-
collision certificate nor evidence of successful dynamics or capture.

[Independent bank audit](candidate_fresh_bank_audit.py) reconstructs every raw
Gaussian/Sobol proposal and source alias, verifies each recorded FK/geometry/
path gate without new IK search, and checks exact support actions/latents and
all empty positions. Proposal identities are audited in the pinned producer
NumPy/SciPy runtime, avoiding cross-runtime byte-derived ID changes.

The [preparation driver](candidate_fresh_prepare.py) uses at most four owned
CPU workers, 30 minutes total wall time, two aggregate CPU-hours and the
original 6 GiB storage floor. Audit work also has a 10-minute wall cap and shares
the remaining preparation CPU/wall allowance. It terminates only its own
workers on failure, preserves completed/partial evidence, and cannot certify
an incomplete or over-budget bank. New outputs live outside the frozen
experiment. Parent and inputs remain read-only. Raw coordinates, fits and
predictions remain outside Git; compact counts and hashes are published.

## Simulator guard path

The [native adapter](candidate_continuation_native.py) accepts an already
initialized guarded environment and a durable TRAIN reservation from the
[bounded driver](CANDIDATE_RUNNER.md). There is no launcher, simulator
construction, optimizer callback or assessment dispatch in this adapter.

Before calling home or a shot, it verifies:

- Exact persisted reservation, request order, candidate, latent, action,
  support, current model and source identities.
- Original scene/materials, geometry, camera, URDFs, effort limits, canonical
  poses file and native default home.
- The genuine inherited controller methods and delegation chain, including
  measured setup, `_measure`, `_joint_status`, all-DOF velocity checks, bounded
  correction and release gates. Substituted methods/readbacks fail before home.
- Independently verified startup controls from that future application and
  the unchanged storage floor. This task runs no startup probes.

A new-only fsynced dispatch claim prevents a consumed reservation from running
again after an interruption or adapter restart. The adapter calls canonical
home and the unchanged native shot method once per active cohort. Inactive
padding stays parked and supplies no optimizer sample. Prepared fits are seeds;
the controller recomputes preflight against actual joint readbacks and keeps
its original driven-motion, settling and release checks.

Released requests require full-rate lossless physics evidence. Native packets
retain exact actions, seeds, preflight records, measured home, transit,
correction and all-DOF settling history, full scoring evidence, original reward
features and hashes. Every active rejection returns its original invalid reward
and remains a requested slot. Failures retain partial raw evidence and prohibit
retries. The caller separately records outcomes; returning a packet never
implicitly updates a model.

Mock tests check reservation tampering, altered sources/home/metrics, exact
padding/seeds, duplicate dispatch, original rewards and failure preservation.
A CPU-only test allocates an uninitialized class with the actual native method
resolution and verifies the unmocked guard inventory; it never constructs or
steps a simulator. This is code-path verification, not a new native validation.

## Remaining execution gates

The full native scoring/guard auditor and one-start operational supervisor,
bounded startup diagnostic receipt integration and final-only three-method
assessment still require completion/review before a real continuation.
The fresh bank is geometry-assisted held-out preparation: geometry was seen,
outcomes were not. No new capture rate, policy improvement, physical safety
certificate or model promotion follows from this work.
