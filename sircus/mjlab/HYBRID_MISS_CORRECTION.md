# Three missed shots: diagnose contact, then change one variable

The hybrid planner captured at 17 of 20 new bucket positions in the
[preceding comparison](STALLED_REFERENCE_STUDY.md). This study inspects the
three misses and tests alternative board-arm joint configurations. It does
not retrain the model, change the interactive planner, or produce a new
unseen-position capture rate.

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

Preparation found **1, 8 and 5** distinct accepted alternatives respectively.
The selected alternatives primarily flip the wrist configuration. Joints 1–3
remain essentially unchanged, so the obstructing middle link stays in almost
the same place. Predicted clearance changes by **less than 0.001 mm** and is
still negative in all three cases. This is an important limitation of this
fixed-board-pose correction, not evidence that the rebound path is now clear.

For each case, execute **three unchanged and three corrected nominal repeats**
with matched physics at **64 substeps**, alternating order. Add nine simulator
controls before and nine after: **36 requested drops total**. Keep any setup
decline in its requested denominator and never substitute a different shot
based on an observed outcome. The native run waits cooperatively for the 4090;
the 4070 workload and Isaac setup remain untouched.

Final counts require an external audit binding the frozen preparation,
report and native evidence. Until it passes, the UI shows preparation and the
original recorded videos but withholds final correction capture counts.
The same preserved original recipes remain available regardless of outcome.

## Interface and videos

The panel is `#hybridMissCorrection`. Each case has its contact timeline,
exact geometry and joint vectors, then original and corrected video slots.
The movies replay sampled native ball centers with the actual held arm poses;
they include normal speed and a quarter-speed close view. They do not generate
new physics or depict driven arm motion.

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
