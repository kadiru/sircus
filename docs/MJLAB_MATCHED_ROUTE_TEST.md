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
