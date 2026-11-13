# Matched direct and table-bounce evaluation

The [previous route study](MJLAB_ROUTE_POLICY_STUDY.md) allowed each planner
to pick its route. In this evaluation each planner receives both route
requests at every bucket position. This measures the ability to produce
each kind of shot under the same setup budget.

- Reuse the audited pooled model, route-conditioned model and pair of
  specialists with the same training recipes and geometry prior.
- Sample 50 new test positions from the same tabletop region and exclude
  every inventoried coordinate within 5 mm, including the previous
  training and test layouts. Freeze the positions before any releases.
- For each position and requested route, give each model the same four
  starting recipes. The first prediction that passes the existing pose
  and path checks gets one drop. Freeze all 300 choices before the first
  test release.
- A success requires the requested route and one continuous second of
  whole-ball retention after a board rebound. Count a setup rejection,
  capture by the other route or experimental arm-assisted capture against
  the requested route. The denominator is 50 positions per model and route.
- Use the same MuJoCo scene, 64 substeps and the nine control shots before
  and after the test. The independent CPU audit replays scoring from the
  saved trajectories and verifies positions, candidate budgets, frozen
  decisions and paired results.

`python -m sircus.mjlab.intergrid_matched_route_study --help` shows the
producer inputs. `python -m sircus.mjlab.intergrid_matched_route_audit --help`
shows how to reproduce the audit. Raw trajectories and model files live in
the Ubuntu experiment archive.

## Results on 50 new bucket positions

| Frozen planner | Direct captures | Direct setup failures | Table captures | Table setup failures |
|---|---:|---:|---:|---:|
| Pooled | 34/50 (68%) | 5 | 12/50 (24%) | 15 |
| Route-conditioned | 36/50 (72%) | 8 | 13/50 (26%) | 19 |
| Two specialists | 34/50 (68%) | 10 | 11/50 (22%) | 18 |

225 of 300 requests passed setup and were released. One route-conditioned
table drop produced invalid physics and counts as a failure. There were
no alternate-route or experimental arm-assisted captures. The pooled
model found more feasible setups on both routes. The route-conditioned
model captured more often after release: 36/42 direct and 13/31 table,
compared with 34/45 and 12/35 for the pooled model.

On the same direct positions, route conditioning alone scored three
times against one for pooling alone. On table shots, the split was three
to two. The specialists had no advantage in the overall rate for either
route. The differences of one or two captures are too small to establish
which model generalizes best.

The 4090 run completed both control sets (18 releases). The independent
CPU audit passed 65,121 checks, including every held-out position,
setup decision and native trajectory. The mjlab suite ran 2,269 tests
with seven skipped. The Ubuntu evidence directory is
`intergrid-matched-routes-v1`.

- Frozen checkpoint SHA-256: `847b1119d4321ce362b628f7ef5f2ace3ed202cbb46b597bb9875bf1fc351c52`
- Report SHA-256: `3ac255678b06492cdb5403cebe9fc159f5239959dbdb8e53539116df5ca98e0e`
- Audit SHA-256: `4b1757a7b6dda9d8aa1c63d724f708b3f1d34f70d5a5b1a7463e64b5248ea0ae`
