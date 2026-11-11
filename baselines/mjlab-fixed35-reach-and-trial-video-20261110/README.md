# Fixed-grid reach diagnosis and saved-trial inventory

[Report](../../docs/MJLAB_FIXED35_REACH_AND_TRIAL_VIDEO_20261110.md) ·
[Sanitized summary](summary.json)

Read-only diagnosis of 288 numeric settings / 576 branch requests at cells
03-04, 03-05 and 04-06 in the completed `fixed35-six-v1` experiment:

- 263 settings exceed a conservative arm wrist-reach bound.
- One has unresolved IK inside that outer bound.
- Ten have overlap in every tested endpoint branch; two have a branch with
  positive clearance below the 5 mm buffer.
- Twelve settings pass, four per cell; all 24 selected setup paths pass.

The separate video inventory contains 100 archived native drops across four
historical cohorts at these three cells, plus 737 setup declines. The verified
combined film contains all 100 drops in 5,412 frames / 225.5 seconds. Three
individual case films contain 44, 37 and 19 drops. The summary records video
hashes, independent ledger/media checks and preservation results. Videos remain
outside Git. No new simulation or trained model is claimed. Recipe coverage
remains 30/35 and the frozen policy remains 24/35.

The summary includes immutable source identities and numerical diagnostics,
without raw traces, absolute host paths, credentials, calibration or logs.
