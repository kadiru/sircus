# Choosing a bounce route before one drop

Compare two planners at the same 50 fresh bucket positions. Direct-only
chooses a board-to-bucket shot. The route selector also considers a
board-to-table-to-bucket shot using the
[expanded table library](MJLAB_TABLE_SHOT_LIBRARY.md).

1. Freeze the original direct library, the expanded table library and
   the refitted specialist checkpoint. Use up to four starting shots
   for each route and take the first candidate for that route that passes
   the existing pose and path checks. Both planners share the exact
   direct candidate and its arm configuration.
2. If only one route passes setup, the selector chooses it. If both
   pass, compare capture rates at the five geographically nearest
   positions in the [matched refit study](MJLAB_ROUTE_REFIT.md) where
   both routes passed setup. Add one simulated count to both successes
   and failures for smoothing. Choose the higher rate, favoring direct
   on ties. Those earlier table attempts used the original starting
   library, so their outcomes serve as a proxy for the expanded library.
3. Freeze the test positions using seed `2026112101`, at least 5 mm from
   archived placements and each other. Freeze all choices before any
   test releases. Each planner drops at most once at each position.
4. Use the existing mjlab scene at 64 substeps. Count a standard capture
   when the ball rebounds from the board and stays fully inside the
   bucket for one continuous second. Score both planners on all 50
   positions, including setup failures. Save the chosen and measured
   routes, plus the paired gains and losses.
5. Run the same nine control shots before and after the test. Verify the
   selected routes, one-drop budget and all recorded outcomes in a
   separate CPU audit.

The producer is `python -m sircus.mjlab.intergrid_route_selector_study --help`
and the verifier is `python -m sircus.mjlab.intergrid_route_selector_audit --help`.
Use the audited archives listed in the [library comparison](MJLAB_TABLE_SHOT_LIBRARY.md)
and add `intergrid-table-library-v1` and its audit receipt. The new evidence
lives in `intergrid-route-selector-v1` on Ubuntu. Keep raw evidence outside
Git. Native simulation uses the 4090 and the existing shared lock.
