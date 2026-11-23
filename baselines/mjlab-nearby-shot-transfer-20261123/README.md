# Nearby successful settings recovered three missed bucket positions

At five bucket positions where the current planner had missed, we adapted
settings that had already captured at other positions. Every adjusted setup
received fresh arm-pose and path checks. We then tested 20 distinct settings
per target at 64 substeps and repeated each qualifying capture three times.
The [protocol](../../sircus/mjlab/NEARBY_SHOT_TRANSFER.md) froze the search,
selection and repeat rules before the drops.

| Bucket X, Y (mm) | Unchanged original | Search captures / 20 | Route | Fresh confirmations | Settings confirmed 3/3 |
|---|---:|---:|---|---:|---:|
| 942.6, −31.7 | 0/3 | 16/20 | Direct | 48/48 | 16 |
| 1193.0, 147.4 | 0/3 | 0/20 | None | No repeats qualified | 0 |
| 966.7, −40.9 | 0/3 | 10/20 | Direct | 30/30 | 10 |
| 1209.6, −160.6 | 0/3 | 16/20 | Board → table → bucket | 48/48 | 16 |
| 1127.6, −38.3 | 0/3 | 0/20 | None | No repeats qualified | 0 |

**Three of five known misses recovered; all 42 qualifying settings passed
3/3 fresh confirmations (126/126).** All 15 unchanged original drops missed.
Capture requires a board rebound and one continuous second fully inside the
bucket. The first search capture is additional to the three confirmations.
These are repeated known configurations, not a new generalization score.

The [sanitized summary](summary.json) includes all 100 selected commands,
the 42 confirmed commands, exact arm joints, source evidence strength, source
distances, original commands and hashes binding the raw evidence. Existing
successful source shots and historical definitions are preserved.

## Remaining failures and unusual outcomes

Both unresolved targets had robot contacts in all 20 tested flights. At
(1193.0, 147.4) mm, the closest repeatedly successful source was only 13.3 mm
away. Its transferred shot retained height, tilt and gap, but the ball hit
the board arm's middle link after rebounding. A different wrist solution was
admitted; that alone is not established as the cause. At the new shot's first
robot-contact time, CPU reconstruction placed the old ball 10.7 mm clear of
that link and the new ball slightly overlapping it. This is a matched-time
measurement of two saved states, not a whole-flight clearance minimum.

One additional setting at (966.7, −40.9) mm retained the ball after robot
contact. It is preserved as an **unconfirmed arm-assisted capture**, separate
from the 42 direct/table settings. The frozen repeat rule did not repeat it.
One search drop at (942.6, −31.7) mm failed the physics-validity guard
(`through_wall_after_entry`); it remains in the 20-attempt denominator and
is not counted as a capture.

## Verification and reproduction

- CPU: 3,758 proposal records including 40 multistart retries; 520 setup
  checks; 358 admitted joint-solution records; 100 selected target commands.
  Five parallel preparations completed without exceeding their limits.
- Native: **259 releases** = 100 search + 15 originals + 126 confirmations
  + 18 controls; **973.070 seconds** of leased RTX 4090 time. Both nine-drop
  control blocks passed, including expected direct/table captures and misses.
- Independent preflight passed **10,221 checks**; final saved-trajectory,
  contact, scoring, proof and repeat audit passed **56,065 checks**.
- Frozen computational producer: `db83569af21c16a1fdef5e4d2fe788972d73bbc3`.
  Draft [PR #161](https://github.com/kadiru/sircus/pull/161), stacked on #160.
  Pinned Ubuntu focused Python tests, UI tests and producer CI passed.
- Raw Ubuntu evidence remains outside Git in
  `/mnt/robot-training/sircus-mjlab/output/nearby-shot-transfer-v1`;
  its sibling `-preflight.json` and `-audit.json` are the audit receipts.
  Frozen source is the matching `studies/nearby-shot-transfer-v1` directory.
  Run the protocol's prepare, freeze, preflight, native, final-audit and
  publish stages in that order, using a fresh output directory. All source
  studies, assets, scorer and pinned package versions must match.

The live UI shows these results at `#nearbyShotTransfer`, including exact
confirmed settings. Its existing planner remains `hybrid`; all historical
videos are preserved. There was no retraining, service restart, hardware,
4070 or Isaac change. Native drops held both arms at checked endpoints:
driven-arm execution, perception, noise robustness and other timesteps were
not validated here.

Next, retain the confirmed commands as candidate sources for a separately
tested planner update. The remaining two targets need board/arm configurations
that clear the rebound path; this bounded search does not establish impossibility.
