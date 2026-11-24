# Add range to the arm-free shot

The preceding two-position study found a board setup at bucket position
**(1192.994, 147.423) mm** that let the ball clear both arms, but the ball still
missed the bucket. The board twist was **135°**. This experiment tests whether
additional release height, followed by modest tilt changes, can turn that clear
flight into a capture.

The original command is `parallel-board-miss-09-clearance-0016-b0` from the
audited `remaining-link-clearance-v3` study. Its board-center height was about
461.9 mm, tilt 25.47° and drop gap 109.44 mm. Drop gap is ball-bottom release Z
minus board-center Z, not absolute release height.

## Fixed experiment

Board-center X/Y/Z, board heading, board twist and bucket position stay fixed.
Use this exact grid, in tilt-outer/gap-inner order:

- Gap increases: 0, 20, 40, 60, 80, 100, 130, 160, 200 and 240 mm.
- Tilt changes: 0°, −3°, +3°, −6° and +6°.

The unchanged combination receives three matched repeats. The other **49
combinations** are new proposals. Each gets fresh full-pose, endpoint-clearance
and setup-path checks. For gap-only proposals, preserve the original board-arm
joint vector and reject a substitution greater than 0.0001 rad. Tilt changes
require a newly checked board pose. The dropper uses the nearest available
checked IK solution initialized from the saved source joints.

The CPU preparation receives 600 seconds, with at most 30 seconds per proposal.
A timeout is recorded as an unresolved check; remaining unattempted cells remain
visible. After a timeout, rebuild the engine and reload verified completed
fallback proofs. No existing proof is overwritten. Every eligible grid cell
receives one native drop; approximate flight ranking does not select the grid.

## Test, preserve and report

Use the unchanged **64-substep** native simulation and versioned scoring.
A capture requires a board bounce followed by whole-ball entry and one continuous
second inside the bucket. Direct and table-assisted captures qualify for three
additional exact repeats. Robot-assisted retained balls stay separately labeled.

Run nine pre-controls, nine preservation drops from the three previously
recovered bucket positions, the original and new grid drops, capture
confirmations, and nine post-controls. The maximum is **226 native releases**
within one leased RTX 4090 hour. Do not run if the independent preparation audit
fails; do not publish final capture claims before the final independent audit.

The results panel includes all 49 cells, including failed or unresolved setup
checks. It separates first captures, fresh repeats, robot contacts and confirmed
settings. Previous successful commands and the live planner remain unchanged.
This is a controlled search on one known position, not retraining or a test of
new-position generalization. Ball dynamics hold the arms at checked endpoints;
there is no physical, perception or driven-motion validation.

## Reproduction

Use the pinned mjlab environment and a fresh output directory outside Git:

```text
python -m sircus.mjlab.clear_shot_range_study
  --source-study <remaining-link-clearance-v3> --source-audit <its audit>
  --parent-study <fixed35-sixhour-v2> --parent-audit <its audit>
  --reference <corrected-grid-v1> --assets <original read-only assets>
  --output <new empty study output> --gpu-lock <shared scene.lock>
  --stage prepare|freeze|run
```

Hide both GPUs for CPU preparation and use one BLAS/Torch thread. Freeze all
requests before physics. Supply the independent preparation receipt through
`--preflight-audit` for `run`. Expose only the RTX 4090 UUID, acquire the shared
lease, and wait if another workload owns it. The 4070 and Isaac are untouched.
Raw evidence, calibration and logs remain outside Git. No service restart or
planner promotion is part of publication.
