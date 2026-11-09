# Corrected imitation — frozen 70-position comparison

November 9, 2026. Simulation only; native 64 substeps, unchanged version 3 scoring.

**The refitted policy captured 38/70 (54.3%); the current policy captured
40/70 (57.1%) on the same new positions.** There were four gains, six losses,
34 shared captures and 26 shared failures. This sample did not improve, so the
current policy remains unchanged. It does not establish statistically
significant general worsening.

| Result on the same 70 positions | Corrected imitation | Current imitation |
| --- | ---: | ---: |
| Captures / all requests | 38 / 70 | 40 / 70 |
| Setup rejections | 23 | 19 |
| Native releases | 47 | 51 |
| Released misses | 9 | 11 |
| Direct captures | 35 | 38 |
| Table-assisted captures | 3 | 2 |
| Invalid physics | 0 | 0 |

Direct includes possible rim assistance. A capture requires a measured board
bounce, whole-ball entry and one continuous second inside. Table rebounds are
allowed; robot, floor and unattributed contacts invalidate the route. All
rejections and misses count in the 70-request denominator.

The candidate retained all 92 existing teachers and added one recipe for each
of five corrected development contexts: one height, two gap and two board-X/Y
corrections, each previously verified with 3/3 nominal captures. It received
800 full-batch actor-only Adam steps at learning rate 0.001 on 97 teachers.
The current checkpoint was not updated. Both models and all 140 TEST commands
were frozen before any TEST outcome. This compares the combined effect of
five extra examples and 800 extra fitting steps, not either effect alone.

The fresh seed `2026110941` places two buckets in each cell of a 7 × 5 table
grid. All 70 positions are at least 5 mm from inventoried historical and
training positions and each other. Novelty is relative to that frozen
inventory; these finite samples are not all-table coverage. Each policy had
one attempt per position, without outcome search, retries or TEST fitting.

All **158 requests** finished: 140 TEST requests and 18 pre/post controls.
Each control block retained three direct captures, three table-assisted
captures and three expected misses. The independent audit passed **55,633
checks**, including native reset, teacher/model identity and trajectory
rescoring. Worker wall time was **1,387.39 s**; the full pipeline was
**1,414.47 s**. These are elapsed times, not isolated GPU compute measurements.

[Exact compact results](summary.json) include both checkpoint hashes, all 70
paired positions and correction lineage. The file is copied byte-for-byte from
the audit-bound publisher. [Protocol](../../sircus/mjlab/CORRECTED_IMITATION.md)
and [handoff](../../docs/MJLAB_CORRECTED_IMITATION_20261109.md) describe provenance
and limitations. Frozen producer/UI: `fecbe4d61bd92c912e0b4d80b1fdd7be3a2bcb48`;
draft [PR #112](https://github.com/kadiru/sircus/pull/112) stacks on #111.

The published, visually checked `#correctedImitation` panel preserves all 57
previous report keys/bytes and historical panels. No service was restarted.
The 4090 worker exited; the 4070 workload and Isaac setup were left unchanged.

**Next:** inspect the six new regressions (three finite IK declines and three
released misses) against their successful current-policy commands before more
training. Preserve the successful recipes and historical 37/70 evaluation;
that older score used a different set of positions. No policy promotion,
physical validation, moving-arm transit or perception validation is claimed.
