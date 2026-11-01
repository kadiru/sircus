# Fixed-plane catalogue across bucket distances

A simulation-only development search extending the fixed 400 mm catalogue. It does
not train an RL policy, move hardware, change Viam, or validate collision-safe arm
travel. Prior results and scoring definitions remain unchanged.

## Prespecified geometry and search

Board and release centers remain at XY=(670, −100) mm. The bucket is moved along
−Y to distances 300, 400, 500, 550, 600 and 650 mm from that vertical plane. All
bucket footprints must remain within the unchanged table: at 650 mm, the outer
top radius leaves 25 mm to its front edge. Heights refer to tabletop Z=0.

The 400 mm control rechecks the earlier strongest H400 / gap130 / tilt21° setting.
It is not a new search for a better 400 mm setting. Every other distance uses:

- H: 300, 350, 400, 450, 500, 550 mm.
- Vertical ball-bottom-to-board-center gap: 80, 120, 160, 200, 240 mm.
- Tilt from horizontal facing the bucket: 12, 18, 24, 30, 36, 42, 48 degrees.
- Six prior 400 mm candidates as additional proposals, not assumed successes.
- Local refinement around the six highest-ranked valid rows: gap ±10 mm crossed
  with tilt ±1.5°, plus H ±12.5 mm. Deduplicate existing settings. If captures are
  absent, the existing shaped reward ranks near misses for refinement. This is
  a finite search, not proof of all feasible solutions or physical impossibility.

Ball center release Z = H + gap +37.5 mm. Provisional dropper flange Z adds95 mm.
The ball starts stationary over the board center. Finite ball radius and board
thickness mean its actual first contact need not be exactly at that center.

## Selection and independent confirmation

Freeze up to three geometrically separated captured candidates at each distance.
Each receives five unchanged trials and 20 seeded variations (H ±2 mm, gap ±2 mm,
tilt ±0.5°, independent uniform distributions). Choose the highest perturbed
capture count, then clean captures, nominal captures, and deterministic ID.
Freeze that winner before 50 fresh perturbed trials, with disjoint seeds and no
retuning. Both good and bad outcomes stay in the results. A distance with no
captured candidate has no confirmation set and is labeled "no capture found",
not a measured 0/50 policy rate.

The 400 mm control uses the same five/20/50 protocol for its single candidate.
Seed base20261102, screening indices100000+distanceIndex*1000+candidateIndex*100+i;
confirmation indices200000+distanceIndex*1000+i. This avoids earlier catalogue
seeds and any screening/final overlap.

Scoring is unchanged from `SHOT_CATALOGUE.md`: 2 clean capture, 1 assisted/marginal
capture, 0 released miss; rejected IK/settling separately reported. Both capture
tiers require a measured board bounce, whole-ball entry, and one continuous second
inside. Multiple board contacts remain allowed. "Clean" concerns bucket entry.
Per-distance floor/rim/outside contact probes must pass before searching there.
Measured board pose, normal, speed and drift gates remain unchanged.

A high finite-trial success rate applies only to this narrow simulated variation.
Rigid support, mass, friction, restitution, ball radius, offsets and 175 mm bucket
height interpretation remain provisional. No perception or material noise is added.
The two arms initialize at endpoint IK solutions; travel/grasp/release dynamics
and full two-arm collision avoidance are not learned or certified here.

## Execution and evidence

Use an immutable source snapshot in the existing Sim5.1 / Lab2.3.2 environment:

```bash
export SIRCUS_ISAAC_SOURCE=/workspace/studies/YOUR_SNAPSHOT
flock -n /workspace/scene.lock "$SIRCUS_ISAAC_SOURCE/python.sh" \
  "$SIRCUS_ISAAC_SOURCE/launch_scene.py" \
  --headless --enable_cameras --device cuda:0 --distance-catalogue
```

The launched run is bounded by a three-hour timeout, one-GPU scene lock and3 GiB
free-space guard. Only the4090 is used. The4070 workload remains untouched.

`distance-catalogue.json` updates atomically for read-only `#distance-catalogue`.
The interface distinguishes searched captures, screening and frozen fresh results,
plots clean/assisted confirmation rates, and provides a selected-settings CSV.
Incomplete rows remain marked pending. Historical study panels are unchanged.

The run retains contract/source hashes, per-distance probes, refinements, selected
candidates, frozen winners, unique trial IDs and raw results. Confirmation paths
are saved for diagnostics; raw robot states/trajectories remain outside Git.
`distance_catalogue_report.py` validates the full candidate/seed/selection schedule,
aggregates, settling and capture conditions before exporting compact evidence.
