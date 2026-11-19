# RGB-D perception baseline

This estimates visible geometry from registered RGB and **axial depth in metres**.
It unprojects depth with camera intrinsics, then transforms the points using the
supplied camera calibration. The Isaac camera is ideal and its extrinsics are
known synthetically; this does not establish physical camera calibration.

## Data boundary

```text
Isaac rendered RGB + depth + camera calibration
                  │
          sensors/latest.npz (atomic, bounded)
                  │
       independent CPU perception_worker.py
                  │
        perception.json + perception.jpg ──► read-only preview
                  │
                  └─► perception_evaluate.py ◄─ evaluation_truth/latest.json
                           exact frame ID match       simulator object poses
                                      │
                           separate evaluation panel
```

The estimator accepts only RGB, depth, intrinsics and camera transform plus frame
ID/time. The bundle loader rejects extra fields. No simulator object position,
instance-ID mask, contact sensor, scene configuration or policy result enters the
estimator. Truth is written separately and read only by the evaluator. Calibration
is permitted input; shape and appearance priors are documented below.

Outputs report missing objects explicitly, without truth substitution or silently
holding old estimates. The UI marks observations older than three seconds stale.
Quality labels express geometric checks, not calibrated confidence probabilities.
Latest files are overwritten; no unbounded image/depth queue is accumulated.

## Current detectors and limitations

| Object | Observed estimator | Assumptions / limits |
|---|---|---|
| Table | Depth normals, connected support patches, robust plane fit | Broad horizontal table; chooses highest plausible plane. Center is visible patch, not complete table center. |
| Ball | Pink RGB candidate, curved depth support, free-radius sphere fit | Radius allowed 15–80 mm, at least 20 support pixels. Current color prior covers the simulated pink ball only. Occlusion, lighting and small visible caps can cause misses. |
| Board | Neutral-color candidate, robust plane, observed rectangle extents | Elevated 200×300 mm board; partial visibility can bias center. Other board shapes are future work. |
| Bucket | Yellow RGB candidate, highest rim band, plane + circle fit | Upright relative to observed table; enough visible rim arc required. Fitted radius is the rim midpoint, **not** the inner capture boundary. |

This is RGB-assisted geometric perception, not learned general-purpose instance
segmentation. It does not yet estimate temporal bounce/entry/one-second retention.
The visual capture verdict is **unknown**. RL success still comes from the separate
versioned PhysX/contact scorer. Perception estimates do not control either arm yet.
Rendered color silhouettes can lag moving depth edges; dynamic-ball outliers need
further investigation. Depth noise, missing pixels, lighting variation and new
layouts have not been systematically validated.

## Run on the existing isolated Ubuntu environment

Use the pinned Isaac/Lab environment described in README.md. NumPy, SciPy and
Pillow are supplied by the existing Isaac Python environment; use `python.sh`, not
the bare venv interpreter (which misses Kit's dependency paths). Workers use one
BLAS thread and no GPU or robot API.

From the Ubuntu checkout:

```bash
bash sircus/isaac/perception.sh start
bash sircus/isaac/perception.sh status
# If no Sircus scene is running: frozen replay only, no optimization.
bash sircus/isaac/run.sh evaluate /workspace/output/RUN/checkpoint.json 256
# Independent workers can be stopped without stopping the scene:
bash sircus/isaac/perception.sh stop
```

Replace RUN with the locally saved matching run. Checkpoint fingerprint guards
remain enforced. Training and evaluation publish sensor bundles every camera
update; this adds serialization cost but does not change observations, actions,
reward, physics or checkpoint definitions. Do not launch a second scene alongside
training. The scene lock rejects competing launchers.

The separate host `preview.py --output OUTPUT` serves loopback port 18860. The
existing Mac forward is port 18780. Reload the page to see perception overlay,
centers/normals, missing/stale states and the collapsible truth-only error panel.
The preview has no command endpoints. A finite replay ends normally and its final
observation becomes stale; workers wait for another producer. Nothing auto-starts.

Offline replay of a sensor-only bundle:

```bash
# Inside the isolated container, with an output directory of your choice:
/workspace/source/sircus/isaac/python.sh \
  /workspace/source/sircus/isaac/perception_worker.py \
  --input /workspace/output/sensors/latest.npz --output /workspace/output/offline --once
```

## Validation checkpoint — 2026-10-30 Pacific

The earlier 256-episode training run had already completed before perception work.
Its initial policy captured 0/2; midpoint heuristic 2/2; frozen learned policy 8/8.
These are correlated deterministic fixed-scene repeats, not generalization or
proof of an advantage over the heuristic. Its checkpoint/results were preserved.

A fresh frozen replay supplied a 30-second, non-curated observation window. Of 179
matched frames, table, board and bucket were estimated in all 179; ball in 63.
The remaining 116 ball estimates were missing. This is availability, **not recall**:
visibility was not independently annotated, and occlusion is not the only possible
cause of missing estimates. Errors are conditional on an estimate being present.

| Object | Error measure | Mean | 95th percentile | Maximum |
|---|---|---:|---:|---:|
| Table | Plane distance | 1.99 mm | 1.99 mm | 1.99 mm |
| Board | Surface-center distance | 2.18 mm | 2.71 mm | 3.98 mm |
| Bucket | Top-rim-center distance | 3.33 mm | 3.55 mm | 3.64 mm |
| Ball | Center distance (63 frames) | 4.48 mm | 4.46 mm | 21.91 mm |

Mean normal error: table 0.038°, board 0.034°, bucket 0.182°. A detected-ball frame
processed in 107 ms on the CPU. These figures are ideal synthetic single-scene
checks, not physical accuracy claims. No threshold was fitted to truth errors.

20 Isaac tests passed, including 9 new perception tests for projection, missing
input, plane/sphere/rim geometry, insufficient curvature/arc rejection, strict
sensor fields and exact-frame evaluation. The full project regression suite and
repository checks also passed. A flat-patch false positive was caught and fixed
by the curvature regression. Live overlay images and HTTP routes were inspected;
the app browser automation was unavailable, so full browser layout interaction
was not verified automatically.

Raw observations and diagnostic files stay outside Git under the Ubuntu runtime's
`output/perception-validation/`. The existing 256-trial frozen replay and CPU
workers were left running; the replay is bounded. The 4070 workload stays stopped.
No physical dashboard, Viam configuration, hardware or sound service was touched.

## Next step

Create disjoint layouts and visibility labels; vary camera, lighting and depth
noise; measure false positives, availability and conditional pose error. Then add
a temporal visual capture observer and a policy-observation adapter with explicit
missing-data handling. Compare the **same frozen policy** with privileged versus
perception-derived observations before training on estimated state.
