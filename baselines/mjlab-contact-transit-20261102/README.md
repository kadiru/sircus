# Contact correction and transit audit — November 2, 2026

Synthetic MuJoCo Warp/mjlab experiments on RTX 4090. No PPO, checkpoint replacement,
physical robot commands, audio, Viam change or Isaac setup change. The 4070 workload
was not signalled or reconfigured. The shared 4090 lock queued the second job behind
another Isaac job, which completed naturally. Runtime elapsed fields include that
wait and are not GPU benchmarks.

## Findings

The prior 13/20 result was a perturbed local confirmation at `old-test-21`, not 20
identical nominal repeats. All 20 endpoints were valid; seven failed whole-ball
entry. A CPU gravity audit verifies duplicate gravity compensation: removing the
extra bias force eliminates spurious acceleration at rest. At the nominal pose the
old force produced 9.887 m/s² upward board acceleration between pose clamps.

The dominant capture improvement is **finer contact integration**. Gravity removal
alone does not improve the coarse-step result. Geometry, friction, solref, original
scorers and one-second retention rules remain unchanged. Rim-assisted captures count;
external assistance does not. Old results retain their original contract.

| Articulated protocol | Old 20, two resets | First new 20, two resets | Fresh 100, two resets |
|---|---:|---:|---:|
| Original extra bias,8 substeps (1,920 Hz) |26/40|23/40|144/200|
| Passive gravity only,8 substeps |23/40|24/40|Not run|
| Passive gravity only,16 substeps (3,840 Hz) |40/40|40/40|Not run|
| Passive gravity only,32 substeps (7,680 Hz) |40/40|40/40|200/200|
| Passive gravity only,64 substeps (15,360 Hz) |Not run|Not run|200/200|

The fresh 200/200 means **100 unique independently seeded perturbed requests, each
replayed twice**. Both corrected protocols also captured 32/32 nominal clones;
all four corrected single-world checks captured. All fresh corrected captures were
rim-assisted (zero clean captures). This is local robustness, not all-table coverage
or evidence that a learned policy improved. The nominal configuration was frozen
before these tests and never retuned.

Two rates agree on capture outcomes but not bitwise trajectories. At 350 ms,32→64
has maximum position difference 4.873 mm and velocity difference 0.0467 m/s. Singleton
ballistic paths match batch paths before bucket contact; later motion inside the fixed bucket can
diverge slightly without changing capture.64 substeps is the conservative reference
for subsequent contact work;32 is a tested faster diagnostic. Physical material and
mount compliance remain uncalibrated.

## Exact configuration and tested variations

- BucketXY **(162.752554721,−636.535646700)mm**.
- BoardXYZ **(470.875779301,−277.893401451,397.387845885)mm**.
- Tilt **36.154791124° from horizontal**, heading toward bucket.
- Vertical gap to ball bottom **303.293042081 mm**.
- Ball-center release height **738.180887966 mm**; flange **833.180887966 mm**.
- Horizontal board-to-bucket distance **472.825741265 mm**.

[141 exact contact configurations](contact-configurations.csv), including all 12
joint values, contain the unique nominal pose,20 old perturbations,20 diagnostic
jitters and 100 separately seeded confirmation jitters. Standard deviations are 1 mm
XY,2 mm board-height/gap and 0.35° tilt. Paired repetitions do not add new geometry.
[Nominal source](nominal-source.json) retains its historical outcome and identity.

The measured 200×300×5 mm board and straight-sided tapered bucket remain unchanged.
Bucket measured inner 177/150 mm and outer 200/158 mm diameters retain their respective
contact/clearance roles. Reported 175 mm height is unresolved; the same provisional
175 mm interior plus 8 mm bottom assumption remains. Ball/mass/material/mount/table
assumptions remain provisional in [scene.json](scene.json).

## Arm transit

The separate CPU validation model checks both arms, non-adjacent self-collision,
board, carried ball, table, floor and bucket with 5 mm modeled clearance. Mechanical
adjacency and intentional grasp contacts are explicit exclusions. Joint edges are
sampled at≤0.5° per joint; unresolved distance queries fail closed. Values may be
conservative AABB/SAT separation lower bounds, not exact surface distances.

[25 exact transit endpoints](transit-configurations.csv) and the
[full compact route report](transit-summary.json) retain failures:

- Nominal 21 plus all 20 old jitter geometries pass complete serialized cycles:
  **21/25**,11,650 sampled joint configurations, exact source joints restored.
  Minimum sampled clearance lower bound **5.042769 mm**. Nominal alone uses 556
  samples and a 5.367798 mm lower bound.
- Dropper first lifts/retracts, parks with flangeZ 950 mm, then the board takes a
  lower route with centerZ 240 mm. Full board extent is checked. Board retracts
  (+150,+40)mm and transfers (+50,+20)mm before restoring; dropper returns high,
  then descends. The initial straight 950 mm lift was unreachable and stays rejected.
- Three other endpoints cannot complete that prescribed very-low cycle. A separate
  [bounded alternative study](transit-detours.json) finds modest reconfigure-and-return
  routes: target 10 board shift(0,+50,+30)mm toH 306.326519 mm;
  target 23 same shift toH 371.162313 mm; target 26 shift(0,+50,−30)mm toH 318.301922 mm.
  Selected paths are direct sampled joint edges; RRT is implemented but unnecessary
  for these accepted goals. They are not added to the 21 full-cycle count.
- Target 39 is rejected at its initial pose: board-arm link 4/link 6 overlap 0.114609 mm.
  Its historical endpoint check had same-arm collision disabled.

[Sampled native-FK preview](transit-preview.json) is a geometric visualization with
illustrative timing, not a dynamic motion replay. No claim of actuator tracking,
torque sufficiency, grasp stability or formally continuous collision-free motion.

## Verification and provenance

The 1,008-throw diagnostic passed 415 independent checks. The 700-throw fresh confirmation
passed 414. All 1,708 saved throws were independently rescored with original scalar
capture/route/quality rules. Exact sources,17 assets, request/joint identities,
noise seeds, denominators, static endpoint FK and stored-array parity were checked.
The independent auditor does not rerun native ball dynamics.

- [Diagnostic summary](diagnostic-summary.json), [audit 415](diagnostic-audit.json),
  [protocol excerpt](diagnostic-protocol.json).
- [Confirmation summary](confirmation-summary.json), [audit 414](confirmation-audit.json),
  [protocol excerpt](confirmation-protocol.json).
- [Gravity audit](gravity-audit.json), [executed source identities](source-identities.json).

All published evidence JSON is curated to omit local execution paths. Protocol
files are excerpts, with the original contract fingerprint and file hash retained.
Embedded `evidence_sha256` and gravity-report hashes identify original raw Ubuntu
artifacts, not these sanitized copies. `manifest-sha256.json` separately checksums
the exact tracked files in this directory. Original contracts, raw arrays, result records and immutable
source snapshots remain under `~/sircus-mjlab/{output,studies}/contact-repeat-v1`
and `contact-confirm-v1` on Ubuntu, outsideGit. First native snapshot differs from
current `contact_env.py` only in 64-step support and the corrected hinge-only
measurement scope; [this reconstruction patch](reconstruct-first-env.patch) produces
its exact `7e46c5d1…` SHA from the published source. Physics/scoring comparisons use the
actual frozen snapshots, not the latest helper files.

Current selected checkpoint SHA remains
`768d21bdbe104f64c87eeec96bda81395f9ffea9af14f211baa53c70f7a898bd`.
The previous UI inference adapter stays active under its original guards. This study
is displayed read-only at `#contactRepeatability`; it does not silently switch an old
checkpoint to new physics. The same simulation HTTP process remains running; only
static files/derived evidence were refreshed. A stale Mac localhost 18781 SSH forward
was replaced to restore display access.

Next: bind corrected contact and stricter endpoint/transit admission to a new study
contract, check more bucket layouts, then train against its own geometric baseline.
Do not promote an initially self-colliding pose, a failed route, or a legacy checkpoint
as validated under the new contract.
