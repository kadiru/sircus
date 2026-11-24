# Increasing range at one arm-free missed shot

The previously observed shot at bucket **(1192.994, 147.423) mm** cleared both
arms but fell short. This fixed 64-substep follow-up increased release gap and
varied board tilt while keeping the board center, heading and 135° twist.

**Result: no new capture.** Eleven of 49 new settings passed setup checks and
were dropped; eight hit a robot and three cleared both arms but still missed.
All three unchanged-source repeats also cleared the arms and missed. No
captures qualified for confirmation repeats. The other unresolved bucket,
**(1127.555, −38.348) mm**, was not tested in this pass.

| Tilt from horizontal | Drop gaps tested (mm) | Captures | Robot-contact flights |
|---|---|---:|---:|
| 25.469° | 129.439, 149.439, 169.439 | 0/3 | 2/3 |
| 28.469° | 109.439, 129.439, 149.439, 169.439 | 0/4 | 3/4 |
| 31.469° | 109.439, 129.439, 149.439, 169.439 | 0/4 | 3/4 |
| Unchanged source: 25.469° | 109.439, repeated three times | 0/3 | 0/3 |

The [saved-flight diagnosis](followup-diagnosis.json) found that all eight new
robot-contact flights first struck board-arm **link 3**. Native forces establish
contact; the link name comes from CPU collision-geometry reconstruction.
The three new arm-free misses all used **169.439 mm gap**:

| Tilt | Short of bucket center at rim height | Sideways offset |
|---|---:|---:|
| 25.469° | 193.25 mm | 46.16 mm |
| 28.469° | 179.71 mm | 47.65 mm |
| 31.469° | 175.08 mm | 47.98 mm |

These are interpolated descending ball-center crossings of the rim plane,
not entry measurements. The balls later bounced on the table and hit the
bucket exterior below the rim; none entered. All 11 new drops had one board
contact episode. The original had two: its second increased forward speed
from **0.813 to 1.290 m/s**. Increasing gap therefore did not simply extend
that original flight. This diagnosis used saved trajectories, with no new drops.

## Exactly what was tested

Source command: `parallel-board-miss-09-clearance-0016-b0` from the audited
`remaining-link-clearance-v3` study. Fixed board center was
**(680.343, −11.044, 461.868) mm**, heading offset **6.947744°**, twist **135°**.
Starting tilt was **25.468752°** and gap **109.438641 mm**. Gap means released
ball-bottom Z minus board-center Z; it is not the ball's absolute height.

The grid combined five tilt changes (**0, −3, +3, −6, +6°**) with ten gap
increases (**0,20,40,60,80,100,130,160,200,240 mm**). The unchanged pair was
reserved for matched repeats, leaving **49 new settings**. All were checked:

- **11 admitted** and released once each.
- **30 rejected by the full-pose wrist-reach screen** for these exact requested
  poses. This does not prove the bucket inaccessible with other board/release
  positions or orientations.
- **8 rejected by strict endpoint-clearance or pose checks**.
- **Zero untested cells and zero timeouts**. CPU preparation took **43.594 s**.

Gap-only settings preserved the original board-arm joints. Tilt changes needed
new board poses and setup-path checks. The [summary](summary.json) retains every
grid cell, all 11 exact selected commands and joint vectors, observed labels,
source/evidence hashes and three preservation commands. Setup rejections are
not counted as simulated flight misses.

## Validation and preservation

**41 native releases** = 11 search + three unchanged-source repeats + nine
preservation drops + 18 simulator controls. All **9/9 preservation captures**
passed, using one saved setting from each of the three previously recovered
bucket positions, repeated three times. Both nine-drop control blocks passed.
There were no invalid-physics or arm-assisted retention outcomes in this pass.

The final independent audit passed **10,993 checks**. Leased RTX 4090 time was
**261.253 seconds**. Working shots and historical results remain unchanged.
No model was retrained or promoted; the live planner, 4070 and Isaac setup were
not changed. UI result section: **`#clearShotRange`**.

This is selected development search at one known target, not a new-position
policy evaluation. The simulator held arms at their checked endpoints during
ball flight; these results do not validate driven-arm motion, perception,
physical hardware, noise robustness or another timestep.

## Reproduce and verify

Use the [frozen protocol and CLI](../../sircus/mjlab/CLEAR_SHOT_RANGE.md), pinned
mjlab environment, immutable parent/reference assets and a new output directory.
CPU preparation hides both GPUs; native execution exposes only the authorized
4090 under the shared lease. Calibration, raw trajectories and runtime files
remain outside Git.

- Computational producer: `457c97134367e3176068ea5e09e4ce1ad71443a1`.
- Draft [PR #163](https://github.com/kadiru/sircus/pull/163), stacked on #162.
- Report SHA-256: `fc6a103d4f4a18712a7f02f780730159a6643996d9ef9201c592cd253d9110c7`.
- Final audit SHA-256: `20ca749f368cf720d6123d8a6cce41c4ba872b0b1584d8f3b19775dd8084fdf6`.
- Evidence manifest SHA-256: `7a5649b9f3baf08fd16a59559d8fbbf312aac95c08e66cb5a08e600e90c59b62`.
- Saved-flight diagnosis SHA-256: `a1ebcb14fc5009e03d484b0c5868261a43fda550c7b72e84880bf47b5612f8f5`.
- Sanitized summary SHA-256: `39bfe7a596f82667cdb59a16ebbf39512028378245d8b5aa2cd957455a79fb0b`.

Proposed next test: change board position and height together to shorten the
required flight while keeping the release pose reachable. One option is to
lower the board and increase the gap by the same amount, preserving the clear
shot's **668.807 mm ball-center release height**, with a modest forward board
shift if needed. This is untested; each new arm pose/path still needs checking.
No additional search or training is implied by publishing this result.
