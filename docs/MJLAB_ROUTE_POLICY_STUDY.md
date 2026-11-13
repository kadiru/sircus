# Does teaching the routes separately improve the first shot?

Each planner gets the same bucket position and up to eight proposals. Four
start from verified board-to-bucket shots and four from verified
board-to-table-to-bucket shots. All planners use the same arm pose and path
checks. A shared rule picks the route from collection data at the five
nearest training positions, using the other route when only it has a
feasible proposal. Then the simulator drops the ball once. If neither route
is feasible, that location counts as a setup rejection.

1. Freeze 24 new collection and 50 new test positions between the grid points
   before releasing a ball. Exclude archived layouts and enforce a 5 mm
   minimum separation. Use 16 proposals per intended route at every
   collection position. Record setup rejections as well as every release.
2. Score a standard capture when the ball rebounds from the board, fully
   enters the bucket and stays inside for one continuous second. Rim and
   table contacts can count. An arm, floor or unattributed contact rules out
   a standard capture. Label a successful shot "table" only when the trace
   shows a downward-to-upward table rebound after the board and before the
   ball first fully enters, without simultaneous board or bucket contact
   during that table rebound. Keep the intended route separate from the
   measured route.
3. Combine the previous audited training set with the new collection
   releases. Keep observations labeled test or validation by older runs out
   of the fit and candidate source menus. Train from the same verified successes for all three model
   designs. For each route, match a successful recipe at one position to a
   compatible successful recipe at another position. The target is the
   change in board center (X, Y, height), tilt, ball release gap, board
   heading and twist. Weight routes equally and weight target positions
   equally within each route. Fit three regularized regression designs:
   one model without a route input, one shared model with a correction
   for each route, and one separate model per route. The last two
   specialists share the same eight-candidate budget and one release.
4. Freeze the three models, route rule, and chosen shots before the
   first test release. Report capture per 50 positions, direct/table
   breakdown, misses, setup rejections, and paired wins/losses. The
   comparison tests whether a different way of sharing training data
   improves first-shot performance. The route rule may favor one route;
   per-route outcomes show where that limits the comparison.

Use the audited six-hour native64 parent and the previous audited intergrid
proposer. The two runners print their path arguments with `--help`:

```sh
python -m sircus.mjlab.intergrid_route_study --help
python -m sircus.mjlab.intergrid_route_audit --help
```

Use the pinned mjlab environment and expose only the 4090 by UUID for the
study. It waits for an idle GPU and takes the existing cooperative lease.
Pose/path workers and the separate audit run without CUDA. Native
rollouts use 64 substeps and repeat the same audited controls before and
after. The audit checks spatial separation, the candidate budgets, the
checkpoint fit, the choices frozen before test, and the saved native
physics evidence. Save raw traces and checkpoints in the output tree,
outside Git.
