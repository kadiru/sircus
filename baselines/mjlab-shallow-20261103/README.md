# Completed exact shallow-board candidate — verified November 4, 2026

**No strict capture at 650/700/750/770 mm.** Each target received one exact native
shot; all four endpoints were admitted. All24 exact parent controls reproduced.
The independent audit passed **359 checks**, including scalar rescoring of all
28 native flights. Native execution took 85.90 seconds after the shared4090
allocation became available.

The requested board-center height was **380 mm**, tilt **28° from horizontal**,
and drop gap **386.8 mm**. Ball-center release was 804.3 mm above tabletop.
The exact rounded user input was tested, separately from the earlier CPU-only
386.824338 mm proposal. Board/bucket geometry, physics, scoring and parent
fingerprints are unchanged.

| Distance mm | Native discovery shots | Whole-ball entry | Longest strict retention | Strict captures |
|---:|---:|:---:|---:|---:|
|650|1|Yes|0.5125 s|0|
|700|1|No|0 s|0|
|750|1|No|0 s|0|
|770|1|No|0 s|0|

This is four single DEVELOPMENT discoveries, not a probability estimate,
repeatability study or proof that these settings can never capture. No captured
setting was selected, so no engineering repeats or 32/128-substep flights ran.
Only 64 substeps have native evidence. The 24 successful controls are not far-target
captures. No RL update, policy activation, arm transit or actuator validation
occurred.

## Near-entry diagnosis and next priority

The 650 mm ball fully entered but failed the one-continuous-second rule. Saved
samples show an early partial top exit and later small negative side-plane
margins resetting retention. At the reset of its longest 0.5125 s segment, the
side margin was approximately−0.008415 mm; the worst post-entry side margin was
about−0.192620 mm. No whole-ball-above-rim sample occurred after entry, and no
external assistance was recorded. This should not be described as a confirmed
full escape or a simple range miss.

[Post-hoc diagnostic](completed/far-diagnostics.json) is explanatory CPU analysis
of saved data, not new dynamics or a revised score. A separate [calibrated benchmark branch](https://github.com/kadiru/sircus/blob/9019ff4/sircus/mjlab/CALIBRATED_BENCHMARK.md)
already implements and verifies a current-ball soft-contact envelope plus compiled
cavity geometry, under a new scoring contract. Its completed frozen TEST reports
PPO 97/216 versus sweep 78/216 (three shared seeds/layout cohorts); these are not
comparable before/after percentages for this four-shot study, and that branch
has not been merged or activated here.

The next bounded task is **reuse that existing calibrated contract for a fresh
650 mm matched-control replay and 64/128 timestep validation**. Check actual
wall/contact geometry and valid downward open-mouth entry; do not silently relax
this old scorer, retag old data or assume the new replay will capture. Preserve
this historical miss and bind new results to the calibrated version.

After that check, a small650-only gap window at fixedH380/T28 can test retention
(e.g. 376.8/380.8/384.8/386.8/388.8 mm, three nominal trials each, all strict guards
retained). These are proposals, not tested settings. Freeze any actual capture
before new 3+30 paired perturbation qualification. Expand distance/direction and
return to a disjoint-layout learned-policy comparison only after that evidence.

## Evidence and reproduction

- [Exact discovery outcomes](completed/discovery-shots.csv), [summary](completed/summary.json),
  [settings](completed/settings.json), [provenance hashes](completed/provenance.json),
  [empty selected recipes](completed/recipes.json) and [checksums](completed/SHA256SUMS).
- Frozen execution commit `274d27eeeae0ab313451c5d0ba34f7c17f55f448`;
  contract `c4e59119e71f724b1e1941ae080e40c8aa001d954356715188895803fcd41a84`.
- [Protocol and launch instructions](../../sircus/mjlab/SHALLOW_STUDY.md).
  Immutable source/evidence namespaces: `studies/shallow-v1` /`output/shallow-v1`.
  The finite worker and CPU audit exited normally.
- `http://127.0.0.1:18781/#shallowBoardStudy` serves completed audited results,
  four actual saved native flights and explicit separation of discovery/controls.
  No Ubuntu service was restarted. The local viewer SSH tunnel was restored
  during the November 4 results check; it does not deploy a runtime.
- [CPU-only preflight](cpu-preflight.json) remains separate historical endpoint
  evidence: 4/4 admitted, minimum clearance lower bound5.969446 mm, no ball flights.

The 4070 workload and Isaac installation/deployment were left untouched. Raw
observations, joints, logs, runtime configuration and credentials remain outside
Git. The bucket's reported 175 mm height interpretation is still unconfirmed;
its provisional interior-depth model is unchanged. Endpoint flight success or
a passed evidence audit would not establish physical accuracy or generalization.
