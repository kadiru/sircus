# Saved bouncer alternatives — completed

**30/35 fixed-grid positions have confirmed recipes; the unchanged policy remains 24/35.**
The preceding six-case study added one recipe. This follow-up added none.

| Phase | Requests | Result |
|---|---:|---|
| Saved bouncer alternatives | 15 | 0 captures, all 15 admitted to native flight |
| Exact preserved recipes | 30 | 30 captures |
| Original controls, before and after | 18 | Both gates passed: 12 expected captures, 6 expected failures |
| Fresh confirmation | 0 | No new capture to confirm |

All 63 requests used native64 and unchanged v3 scoring. Zero setup refusals or
invalid-physics shots. Worker time was 572.2384 s. The three searchable cells
03-04, 03-05 and 04-06 exhausted 3, 5 and 7 saved alternatives respectively.
Two other cells remain actual fixed-base/bucket intersections; those fixed
layout conflicts cannot be repaired by a different arm joint configuration.

Every alternative preserved the exact bucket, board XYZ/tilt/drop gap and
dropper joints while changing bouncer joints. Thirteen contacted a robot before
any bucket contact. Two contacted the bucket first but never entered it, then
contacted a robot later. The latter contacts do not explain the initial miss.
Robot forces are recorded in aggregate; individual robot-link identities are
not available in these traces. No whole-ball entries occurred in the 15 trials.

The comparison uses 15 matched pairs from 11 distinct historical parent shots,
not 15 independent parents. This finite negative result does not prove that
the three remaining layouts have no successful configuration.

## Audit and provenance

- Native producer: `ac872e8e39fa181b2e716ce04633bdd57e6a9018`.
- Final independent auditor: `2b795d64159fc3d9718e05744c08588e9fd15d0c`.
- **35,407 checks passed**, including native trajectories, candidate identities,
  unchanged controls, source bindings, geometry and request limits.
- Report: `499a0353b1e3a0b439582e613ed41da666d11eea25c99f48060c42e98c771c0d`.
- V3 audit: `10b5aa88d44be0f7cf859017eb51cbac849e1a3b35d7ba00760a08d222fa5477`.

The original pipeline remains recorded as `audit_failed`: V1 omitted an
audit-only historical branch field on one preserved record; V2 then encountered
exact-dictionary inequality from near-zero `acos` rounding. Both failure
receipts and all source/native evidence remain intact. The separate V3 audit
recovers metadata from authenticated source proofs and reproduces every saved
endpoint certificate exactly. Only the independent derived near-zero angle
allows 2 microdegrees of rounding; observed maximum was 1.20742 microdegrees.
All other fields, the 5 mm clearance threshold, pose/path gates and scoring
remain unchanged. No physics was rerun to resolve these auditor issues.

All 117 protected files and the service inventory remained unchanged. The owned
4090 worker exited. No UI, 4070, Isaac, hardware or service operation occurred.
No learning or policy promotion occurred; arms were held at admitted endpoints.

[Derived results and exact shot settings](summary.json) · [Isolation](isolation.json)
· [Workflow and next integration boundary](../../docs/MJLAB_FIXED35_BOUNCER_20261110.md)
· [Draft PR #123](https://github.com/kadiru/sircus/pull/123)
