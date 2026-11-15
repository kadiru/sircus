# Comparing route selection with direct-first fallback

The [previous one-drop study](MJLAB_ONE_DROP_SELECTOR.md) found 40 captures
with the route selector and 33 with direct-only on 50 bucket positions.
Five of the seven gains came at positions where only table bounce was
feasible. This follow-up gives both policies access to table bounce and
tests which should choose it when direct is also feasible.

- Freeze 100 fresh positions with the same tabletop geometry and sampling
  balance as before. Make two passes of 50 with seeds `2026112201` and
  `2026112202`. Require 5 mm separation from archived positions and between
  new positions, including between passes.
- Generate up to four direct and four table candidates from the same
  frozen specialist and the same starting shots for both policies. Use
  the expanded table library. Find the first feasible shot for each
  route with the existing arm checks.
- Direct-first always chooses the feasible direct shot. If direct cannot
  set up, it chooses the feasible table shot. The selector uses its
  original five-neighbor rule to rank feasible routes. Both choose once
  before release and get at most one drop per position. The test retains
  all positions in the denominator.
- In mjlab at 64 substeps, a standard capture requires a board rebound
  and one continuous second with the whole ball inside the bucket.
  Compare paired capture outcomes overall and where both routes were
  feasible. When both methods choose the same route, simulate two drops
  with identical commands and record any scoring differences. Compare
  different-route outcomes separately to measure the effect of the
  selector's decision.
- Run nine controls before and nine after the test. Recreate the split,
  shot selection and scoring in a separate CPU audit.

Use the audited input archives listed in the
[one-drop study](MJLAB_ONE_DROP_SELECTOR.md), plus its output
`intergrid-route-selector-v1` and audit receipt. The new producer is
`python -m sircus.mjlab.intergrid_route_fallback_study --help` and the
verifier is `python -m sircus.mjlab.intergrid_route_fallback_audit --help`.
Native simulation uses the 4090 via the existing shared lock. Keep the
`intergrid-route-fallback-v1` evidence archive outside Git.
