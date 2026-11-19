# Shot planning under a 30-second decision budget

The comparison completed on **100 fresh bucket positions**. With the same
30-second decision budget, the frozen learned selector captured **75/100**
versus **69/100** for the current physics planner. The independent audit passed.
The live planner has **not** been changed.

## What was tested

Both planners received the same new positions, unchanged candidate libraries,
exact arm checks, physics ranking and scoring. The learned selector's weights
were frozen; no retraining or test-drop feedback guided the choices. Each
planner could release at most one ball per position, at **64 substeps**.

A deadline can return the best fully checked shot already found. If no shot
was ready, no ball was released and that position still counted as unsuccessful.
The [protocol](PLANNING_BUDGET_STUDY.md) describes the deadline, preserved
evidence and reproduction stages.

## Results

| Measurement | Current planner | Learned selector |
|---|---:|---:|
| First-drop captures / all positions | **69/100** | **75/100** |
| Direct captures | 51 | 57 |
| Table-assisted captures | 18 | 18 |
| Fully checked shots ready and released | 87 | 85 |
| No checked shot ready | 13 | 15 |
| Released flight misses | 17 | 10 |
| Invalid-physics attempts | 1 | 0 |
| Shots with robot contact | 13 | 9 |
| Deadline reached | 57 | 51 |
| Deadline reached with a checked shot | 44 | 36 |
| Deadline reached without a checked shot | 13 | 15 |

The learned selector captured at **ten positions the current planner missed**,
but missed at **four positions the current planner captured**. Both captured
at 65 positions; neither captured at 21. The net gain is **six captures, or six
percentage points**. Six positions used identical commands and had no outcome
disagreements.

Capture requires a board rebound, whole-ball entry and one continuous second
inside the bucket. Rim and table assistance count; robot-assisted captures are
separate. The current planner's one invalid-physics attempt remains in the
100-position denominator, but is not evidence of a valid physical miss. Robot
contact counts can overlap flight outcomes; they are not extra attempts.

## Timing

| Measurement | Current planner | Learned selector |
|---|---:|---:|
| Mean decision time | 26.20 s | 25.47 s |
| Median / 90th percentile / maximum decision time | 30 / 30 / 30 s | 30 / 30 / 30 s |
| Median worker startup | 1.28 s | 1.26 s |
| Median worker lifecycle | 31.25 s | 31.12 s |

The decision clock includes candidate generation, inference, arm checks,
physics ranking and publication of a completed choice. Worker lifecycle runs
from startup through worker shutdown, excluding parent evidence verification,
hash/readback checks and archive writes. **It is not total UI request latency.**

Native simulation used **383.961 leased GPU seconds** on the 4090, approximately
6.4 minutes. CPU planning and waiting are separate. The 4070 and Isaac setup
were not modified.

## Validation and evidence

- **172 test drops + 18 matched controls = 190 native releases** completed.
- Both control blocks passed; the independent audit passed **87,307 checks**.
- **43 Python tests and 10 targeted UI tests** passed; CI passed.
- No automatic planner promotion or service restart accompanied the result.

The frozen protocol implementation is recorded in commit `d4964daf`; the subsequent
`d1b44178` change only guarded optional Torch imports in test discovery.
The frozen evidence and generated configuration remain outside Git:

- Study: `/mnt/robot-training/sircus-mjlab/planning-budget-v1`
- Independent receipt: `/home/kadir/sircus-mjlab/output/planning-budget-v1-audit.json`
- Report SHA-256: `6bd9318e30d80642aeb66160976434f421e8d55aeb89a1786abf7c341ee844b3`
- Frozen choices SHA-256: `cfff1147be2ed69572aff6a0e583182a3bbf9fa4730624c550131263253e61ba`
- Evidence manifest SHA-256: `02d444f9752ae81496fa273dac91169440839a2ff0d871a48d97e7d7fe667f83`

The results panel is `#planningBudgetStudy`. Final scores require the passed
audit; it preserves the earlier unrestricted study separately.

## What this supports next

The learned selector is the stronger observed option in this one matched
100-position batch, with fewer robot-contact shots and six more captures.
It is suitable for an **explicit UI trial with the same bounded planning
behavior**, while retaining the current planner as a selectable fallback.
This result does not justify an automatic default switch or prove an advantage
at every position.

These positions differ from the earlier 200-position comparison, so a change
in capture rate between studies cannot be attributed to the time limit alone.
Inputs are still simulator geometry. Native drops hold settled arm endpoints;
this is not a test of perception, driven arm transit, physical deployment, or
guaranteed coverage of the whole table.
