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

## Audited results

| Policy | Feasible setups | Captures across all positions | Captures where both routes were feasible |
|---|---:|---:|---:|
| Direct-first fallback | 95/100 | 76/100 (76%) | 29/42 |
| Route selector | 95/100 | 78/100 (78%) | 31/42 |

The selector chose table over a feasible direct shot at nine positions.
Among these, both captured at four positions, only the selector captured
at three, only direct-first captured at one, and both missed at one. All
86 pairs that used an identical command had matching outcomes. Both
policies failed setup at five positions. Each produced one invalid
trajectory, counted as a failure.

The selector gained two net captures. There were four differing capture
outcomes, so this test leaves the size of the advantage uncertain. The
two-sided exact paired p-value was 0.625.

The 4090 ran 190 test drops and 18 controls. Both control sets and the
independent CPU audit passed, with 54,781 audit checks. The mjlab
regression suite ran 2,279 tests with seven skipped and zero failures.

- Producer commit `bc87ebcb718be61b738d41b6f0ebfd6b228711a0`
- Frozen checkpoint SHA-256 `3b43cb74946ab4dd53f110ffe1d9e07bff5409bd40966f1f128eb72009da6161`
- Report SHA-256 `e6e0b18994ef9aa27da7f5039fe8efb307de455eab83aff46855cc08be26dc76`
- Evidence manifest SHA-256 `ef5aa69daa75e9c5ff8bfcb07d085b912b7f67eb089a3c7b643417bcfa982fee`
- Choices SHA-256 `2fbd85927dcc4254cb72d4164fe6252016f85783a58feba1541b156d4c001a6b`
- Audit SHA-256 `0d2aefa62ff81bb2949c8f8f0fc68e85bf26b6aaa14137e6a9b187d0d64a3528`
