# Simulation contact calibration and setup controls — November 3, 2026

These simulation-only engineering controls precede the corrected sweep/PPO
benchmark. They are not new benchmark capture rates, learned-policy results,
physical calibration, or evidence that arbitrary shots succeed. Historical
baselines retain their original definitions.

The isolated current-ball bucket controls passed an independent CPU audit:

- 57 native fit cases fixed the contact envelope using the declared
  `1.25 × maximum fit overlap + 8 µm` rule.
- Canonical 64-substep, finer 128-substep and reversed-order 64-substep
  verification each checked 72 separate cases against that frozen envelope.
- Fit passed 3,444 checks; verification passed 15,011. The additional 32-substep
  cases remain diagnostic-only, even though their observed overlaps also fit.
- Wall tolerance is 3.563906 mm; floor tolerance is 2.943836 mm. These account for
  measured simulated compliant-contact overlap and remain below the declared
  wall/floor thickness caps. The independent geometry epsilon is 2 µm, the
  bucket-contact speed domain is at most 6.25 m/s, and retention remains one
  continuous second. No tolerance was chosen from policy/method outcomes.
- CPU compiled-geometry queries checked faces, vertices, floor, rim and bucket
  translations. Dynamic contact distances match the pre-integration state;
  force observations belong to the corresponding native step interval.
  Every zero-frame contact was separately checked as a precision-bounded
  geometric tangency; no direction was inferred from a zero frame.

The versioned complete-shot scorer still requires a measured board rebound,
whole-ball mouth entry and continuous one-second retention. Rim-assisted
captures count; a partial exit resets retention. Component containment controls
alone do not validate that complete-shot route. The separate native complete-shot
engineering gate subsequently passed from immutable `calibrated-native-v2c`: 50
requested decisions, 48 native releases, and captures for all 17 exact shared
source recipes. One deliberate setup rejection remained in the ledger. The
32-decision discarded PPO check used actual simulator scores and changed the
actor; its model/data are not reused for the matched-budget benchmark. The
independent CPU audit passed 9,167 checks. Producer controls took 367.0 seconds;
the audit took 3.4 seconds. These are shared engineering checks, separate from
method-training budgets and held-out evaluation, with no new benchmark rates.
See `native-complete-shot-summary.json` and the exact
`native-complete-shot-audit-v2c.json`; both native traces and models remain out
of this bundle.

`home-control-v5.json` is the exact simulated canonical-home record with identity
`8f89fdb561beeb571ee1975f8677354decb3a8002be3fb95d6e4772bbe2ba62d`.
It defines a new simulator initial park. It does not claim a collision-free path
from the older colliding park or a safe physical homing procedure.

`admission-geometry-summary.json` distinguishes producer-reported geometry from
an independent audit of proof identities and structure. The 71 historical
layouts yield 69 admitted setups and two invariant fixed-base/bucket rejects;
no ball flights were run for those setup checks. The historical layout roles
are diagnostic labels, not the new benchmark's held-out pools. Existing 5 mm
sampled clearance and greater-than-150 mm board/bucket separation constraints
remain in force. Sampled geometry/path proofs do not validate driven arm
motion, grasping or impact compliance.

The contact reports, native complete-shot audit and canonical-home JSON preserve their exact original
bytes and fingerprints. Derived summaries retain original input hashes and
relative source/proof references. `source-and-evidence-manifest.json` binds the
producer, auditor and raw native-cache hashes. No raw arrays, host paths,
credentials, physical calibration, runtime logs or policies are included.
This compact bundle cannot independently rerun the native-data audit without
its separately retained raw simulation controls and exact assets.

Verify these files with `sha256sum --check SHA256SUMS` on Ubuntu or
`shasum -a 256 -c SHA256SUMS` on macOS. The implementation and audit entry points
are `sircus.mjlab.calibrated_score`, `calibrated_score_oracle`,
`verify_capture_calibration` and `calibrated_admission`. Current-ball material
parameters, privileged geometry and provisional bucket dimensions remain
simulation assumptions; the reported 175 mm bucket height interpretation is
still unresolved.
