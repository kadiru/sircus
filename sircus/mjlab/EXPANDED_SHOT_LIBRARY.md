# Expanded shot library

This library reuses saved simulation evidence to train a shot-ranking network.
It does not run new drops, change old scores, or claim a new capture rate.

1. Read the independently audited version-4 historical rescore. Require the
   ordinary scene, its calibrated bucket, articulated held-endpoint arms,
   passive-only gravity and **64 substeps**. Other scenes and cadences remain
   archived but do not enter this dataset.
2. Recover the original command from its saved request packet and verify the
   packet against the source study's evidence manifest. Require explicit board
   heading and twist, or verify the older nominal defaults against the saved
   full board quaternion within 0.0004 radians and the frozen request/explicit
   bucket-facing behavior contract. Missing source seeds remain missing; solved
   query joints never fill them. Authenticate native metadata and bind its
   archive hash/column. Unsupported old command formats stay excluded.
3. Group repeated settings by bucket position, five task parameters, heading,
   twist, both arm joint vectors and release point. This removes repeat weight,
   not evidence. Sub-micrometer/radian rounding only absorbs numerical encoding
   differences; different elbows, tool orientations and whole-turn joint
   representations remain distinct.
4. Keep every native outcome in each group. Its capture target is accepted
   captures divided by valid native repeats. Invalid physics is excluded from
   that denominator. For example, a setting that captured twice and missed once
   has target **2/3**, not three training rows or an unconditional success label.
5. Select one original pre-IK proposal deterministically per setting, independent
   of its outcome. The ranking features contain that proposal's geometry,
   declared route intent and original source joint seeds. Solved query joints
   are **not** substituted for the source seeds. Both the training row and new
   candidate proposals use this same feature convention. Alternative original
   seed representations are counted and retained in provenance.
6. Assign whole bucket coordinates to TRAIN or validation. The 33 usable fixed
   grid positions stay in TRAIN; other historical positions use a seeded 80/20
   coordinate hash. Historical TEST data is now explicitly development data;
   its published scores remain unchanged. Every historical coordinate,
   including excluded shots and setup failures, excludes nearby fresh TEST
   targets. The subsequent experiment adds its other historical inventories.
7. Supply only TRAIN settings with at least one observed capture as candidate
   source settings. Preserve the full success fraction and actual contact-route
   labels. Robot/table contact can count after the first confirmed board rebound;
   the route **intent** remains the original pre-release intent. Contact presence
   alone does not establish that the surface caused the capture.

The selector fits one weight per deduplicated setting and equal total weight per
bucket coordinate. The original neural corpus adds authenticated rejected and
admitted-but-unreleased proposals for the setup head; these have no capture
label. The training script uses validation to choose an epoch, freezes the
checkpoint, and then runs its separate fresh-position comparison.

`expanded_shot_library.py` writes `train-dataset.jsonl`,
`validation-dataset.jsonl`, `candidate-sources.json`, `all-positions.json`,
`library-summary.json` and `library-manifest.json` to a **new** output directory.
`verify_library()` checks frozen source/artifact hashes, repeat fractions,
TRAIN/validation separation and the TRAIN-only candidate-to-evidence mapping.

Raw trajectories, command packets, host geometry, source evidence and generated
library files stay outside Git. Deleting them merely because a new library
exists would break historical reproduction and future scoring changes.
