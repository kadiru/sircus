# Six imitation regressions: diagnosis complete; correction not run

November 9, 2026. Simulation only; fixed 64 substeps and unchanged version 3 scoring.

The six losses in the previous 70-position comparison comprise **three finite IK
refusals, two ball–robot contacts and one rim miss without whole-ball entry**.
One robot-contact shot subsequently retained the ball inside for one second,
but its earlier forbidden contact invalidates the route. Table and rim
assistance remain accepted. Current imitation remains **40/70**, corrected
imitation **38/70**; these historical scores and successful recipes are unchanged.

The diagnosis passed 1,230 checks, authenticating the original 158 requests and
independently rescoring the nine released flights among these six pairs.
It uses the parent's passed 55,633-check native audit. See [compact evidence](summary.json)
and the [handoff](../../docs/MJLAB_IMITATION_REGRESSION_20261109.md).

## Frozen correction awaiting a free 4090

For every case, move the corrected action halfway toward its successful current
recipe in all five shot coordinates. This is one coupled correction, not a claim
about the causal effect of a single parameter. Keep both originals intact.

The plan is 54 diagnostic requests (six cases × three variants × three requests)
plus 18 pre/post controls. Original joint branches and finite refusals are kept;
the midpoint receives a fresh strict setup proof. All admitted drops must run
fresh native physics at 64 substeps and pass the independent audit.

**0/72 slots ran.** The 900-second queue expired before acquiring the 4090,
which was occupied by a protected Isaac workload. No correction outcome or
improvement is available; this is a scheduling blocker, not a failed-shot result.
No further training, automatic retry, policy promotion or service restart occurred.
The 4070 workload and Isaac setup were left untouched.

## Historical videos in the interface

`#imitationRegression` shows all six historical success/failure pairs. The three
IK refusals use explicit no-flight cards. The other clips replay actual saved
native ball centers with held endpoint joints: no new physics, interpolation,
or invented arm motion. Six H.264 movies total 2,705 frames (90.17 seconds),
1280×720 at 30 fps; all decoded successfully. CPU llvmpipe rendered them.
Their hashes, source rows, event times and chapter bounds are in
[historical-videos.json](historical-videos.json). Full media and raw traces remain
private runtime artifacts outside Git.

The live UI was visually checked, including playback of the far robot-contact
case. All prior report bytes and HTML/CSS outside the new panel were preserved.
Source tests and CI passed; native correction validation is still pending.

Next: a new bounded launch of the unchanged frozen protocol when the 4090 is
free; audit, then publish actual matched outcomes. Do not retrain on these
inspected positions and call their replay a new generalization evaluation.
