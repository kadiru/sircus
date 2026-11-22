# Three missed shots: diagnose contact, then change one variable

The hybrid planner captured at 17 of 20 new bucket positions in the
[preceding comparison](STALLED_REFERENCE_STUDY.md). This study inspects the
three misses and tests alternative board-arm joint configurations. It does
not retrain the model, change the interactive planner, or produce a new
unseen-position capture rate.

## Completed result — November 22, 2026

The attempted correction **did not help**. All original and alternative
repeats hit a robot and missed the bucket:

| Bucket X, Y (mm) | Original captures | Alternative captures | Robot contact: original / alternative |
|---|---:|---:|---:|
| 948.4, −35.1 | 0/3 | 0/3 | 3/3 · 3/3 |
| 1195.0, 160.5 | 0/3 | 0/3 | 3/3 · 3/3 |
| 1226.0, 183.4 | 0/3 | 0/3 | 3/3 · 3/3 |
| **Total** | **0/9** | **0/9** | **9/9 · 9/9** |

All 18 diagnostic requests released; none was declined at setup. Separately,
all 18 simulator control checks passed: 12 expected captures and six expected
noncaptures. **36 native drops completed**, consuming **202.705 leased GPU
seconds**. The independent audit passed **8,588 checks**. This does not alter
the original planner's 17/20 fresh-position result or establish a new policy
score. No model was retrained and the interactive planner remains unchanged.

The failure is useful: changing the wrist configuration at the same board
pose did not move the middle arm link that blocks the rebound. The next
bounded search should allow the board center or heading to move enough to
clear that link, then verify the flight. The two farther targets also appear
to need greater range in the conditional projections below; this remains a
prediction until tested with native drops.

## What the saved flights show

All three balls rebounded from the board, hit the board arm's **link 3**, and
failed to enter the bucket completely. Contact times come from native force
records; the specific link was identified by reconstructing the same arm
geometry at the recorded ball position on CPU.

| Bucket X, Y (mm) | Board height (mm) | Tilt (°) | Drop gap (mm) | First robot contact |
|---|---:|---:|---:|---:|
| 948.4, −35.1 | 497.6 | 22.33 | 187.0 | 0.478 s |
| 1195.0, 160.5 | 473.2 | 21.41 | 114.7 | 0.414 s |
| 1226.0, 183.4 | 373.7 | 16.02 | 253.5 | 0.533 s |

The first ball then struck the bucket near its rim and bounced away. The other
two reversed direction after hitting the arm, then struck the table.
The original planner already predicted negative ball-to-arm clearance for
all three shots; it returned its best checked arm setup despite poor predicted
flight clearance. Checking an arm's route to its final pose does not ensure
that the later ball flight clears the stationary arm.

A simplified projection from the measured post-board velocity, ignoring robot
contacts, gives rim-band clearance of **+30.7 mm** for the first direct shot,
but **−112.9 mm** and **−127.7 mm** for the other two table-assisted shots.
These are conditional predictions, not additional captures. Removing the
robot collision alone therefore is not predicted to solve the last two.

## One frozen correction

Preserve the bucket, board center, full board orientation, release point and
exact dropper joints. Search for other board-arm joint solutions at that same
board pose, retaining the existing pose and path checks. Rank accepted
alternatives by predicted ball-to-arm clearance on the original predicted
route before executing any new drop. Search uses three deterministic seeds
and a maximum 180 seconds of CPU preparation per case.

Preparation saved **1, 8 and 5** accepted joint-solution records respectively.
These counts include angle wraps and numerically different solutions; they
do not mean 14 geometrically different arm postures. The selected alternatives
primarily flip the wrist configuration. Joints 1–3
remain essentially unchanged, so the obstructing middle link stays in almost
the same place. Predicted clearance changes by **less than 0.001 mm** and is
still negative in all three cases. This is an important limitation of this
fixed-board-pose correction, not evidence that the rebound path is now clear.

For each case, the run executed **three unchanged and three alternative nominal
repeats** with matched physics at **64 substeps**, alternating order. Nine
simulator controls ran before and nine after: **36 drops total**. The frozen
protocol retained setup declines in the requested denominator and prohibited
substituting a shot based on an observed outcome. The native run waited
cooperatively for the 4090; the 4070 workload and Isaac setup were untouched.

The passed external audit binds the frozen preparation, report and native
evidence. The UI withholds final correction capture counts until that audit
passes. The original recipes remain preserved; the failed correction is not
promoted to the live planner.

## Interface and videos

The panel is `#hybridMissCorrection`. Each case has its contact timeline,
exact geometry and joint vectors, then original and corrected video slots.
The movies replay sampled native ball centers with the actual held arm poses;
they include normal speed and a quarter-speed close view. All **six clips**
were rendered and verified: the three historical misses and the first
alternative repeat at each position. They do not generate new physics or
depict driven arm motion. Their byte and native-trace hashes are recorded in
the compact aggregate.

`hybrid_miss_publish.py` composes only the derived display report. Video bytes
are embedded in the runtime HTML's `hybridMissVideoAssets` JSON element, rather
than copied into the five-second polled report. The publisher validates each
movie's byte hash, trace hash, case and recording identity, and replaces only
that owned element, preserving all other embedded assets. Generated movies,
raw traces, runtime HTML and logs stay out of Git.

```bash
python -m sircus.mjlab.hybrid_miss_publish \
  --study /home/kadir/sircus-mjlab/output/hybrid-miss-correction-v1 \
  --diagnosis /home/kadir/sircus-mjlab/output/hybrid-miss-diagnosis-v1.json \
  --audit /home/kadir/sircus-mjlab/output/hybrid-miss-correction-v1-audit.json \
  --media-dir /home/kadir/sircus-mjlab/output/hybrid-miss-media-v1 \
  --display-report /home/kadir/sircus-mjlab/output/clearance-correction-v1-display.json \
  --runtime-index /path/to/live/runtime/sircus/mjlab/web/index.html
```

Install the updated static UI files before embedding; the publisher refuses
an HTML file without exactly one owned placeholder. Omit `--audit` for an
explicitly pending publication. It never starts a service or a simulated drop.

Scope remains privileged simulator geometry and endpoint-held native ball
dynamics, with geometric arm transit checks. Camera perception, dynamic arm
tracking, noise and a changed timestep are outside this experiment.

## Evidence and reproduction

Frozen computational producer:
`a135f3a3379ccade21be84e09eec77b34feabdf3`.
The [compact published aggregate](../../baselines/mjlab-hybrid-miss-20261122/summary.json)
contains exact counts, selected shot parameters, artifact hashes and paths.

```text
/home/kadir/sircus-mjlab/studies/hybrid-miss-correction-v1/.study-args.json
/home/kadir/sircus-mjlab/studies/hybrid-miss-correction-v1/.launch.json
/home/kadir/sircus-mjlab/output/hybrid-miss-correction-v1/preparation.json
/home/kadir/sircus-mjlab/output/hybrid-miss-correction-v1/report.json
/home/kadir/sircus-mjlab/output/hybrid-miss-correction-v1-audit.json
```

| Artifact | SHA-256 |
|---|---|
| Report | `403ef97d59b5e250010c0bb7d9d9ece7dbbdff839056ea4e669ef3510928c87c` |
| Frozen preparation | `cd98e9c09ebfaf7742660d42393390ce26ff2aa57dc0d68c423f410fb879c3f6` |
| Evidence manifest | `ac7dba2a7294436cdc18c703e3cf96ebef6296b44ded3412534e76fce75df6cf` |
| Independent audit | `b1eda9197f18f1bc581c372da45e402ce3670376c53097bb890b60ac810a81ed` |

To repeat the protocol, use the pinned runtime and dependencies, copy the saved
argument list with a **new output directory**, and run `hybrid_miss_study`
stages `prepare`, `freeze`, then `run`. Preparation requires
`CUDA_VISIBLE_DEVICES=''`; native execution must explicitly select the 4090
UUID from the saved launch record and use the same cooperative GPU lock.
Do not reuse or overwrite this completed study directory. New CPU preparation
may select numerically different joint vectors; exact replay requires the
saved frozen requests and proof files. Preserve the original audited evidence.
