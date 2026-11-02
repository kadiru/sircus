# Expanded-bank repeatability diagnostic

Completed November 2, 2026: 132 new native mjlab shots on RTX 4090, 146.6 seconds including first-use compilation. No teacher, policy, scorer, original result, simulator service or other GPU workload was changed.

The six route-captured development targets each use the measured search source with greatest recorded mouth clearance. Each receives five exact nominal replays with its saved joint branches, five nominal replays using fresh CPU IK, and ten fresh-IK perturbations (Gaussian standard deviations: board X/Y1mm, H2mm, tilt 0.35°, gap 2 mm; clipped to that source's original bounds). Five independent simulator resets contain one cached, one fresh and two perturbed cases per target. All geometry rejections remain failures in the denominator.

| Original test index, now DEV | Nominal saved branch | Nominal fresh IK | Perturbed fresh IK |
|---|---:|---:|---:|
|10|2/5|0/5|0/10|
|19|5/5|4/5|7/10|
|21|5/5|5/5|4/10|
|26|5/5|1/5|0/10|
|39|5/5|5/5|1/10|
|48|5/5|5/5|3/10|
|Total|27/30|20/30|15/60|

Fresh nominal failures include eight unresolved IK cases and one endpoint collision; perturbed cases include fifteen unresolved IK cases. These are finite solver failures, not evidence that the requested poses are physically unreachable. The solver's randomly initialized fourth start depends on the batch slot, so identical nominal requests can return different branches or fail to recover a known accepted branch. Accepted alternate branches can also create a ball-flight obstruction despite passing the endpoint collision check. Saved-branch target 10 only captured 2/5 across resets, identifying an additional numerical/retention repeatability issue beyond IK recovery.

The other six formerly unresolved targets each receive two exact saved-branch controls. All twelve entered; eleven retained the ball for one continuous second, and one retained it for 0.9708s. All failed the unchanged direct-route criterion. At indices 8,9,23,35,41 the ball hits the **table before the bucket**; index 12 hits a **robot before the bucket**. For example, index 8 first contacts the board at 0.2458s, table at 0.7458s, and bucket at 1.1292s. Index 12 contacts board at 0.2833s, robot at 0.5333s, bucket at 0.8708s. These are assisted routes, not missing-range proofs or direct captures.

The diagnostic attaches a read-only last-sensor observer: maximum force across the eight native substeps and ball XYZ at 240 Hz. It changes no control, force, physics or scoring calculation. Its saved force streams independently reconstruct all 108 released shots' original route classifications. All 108 saved-joint endpoint FK/limit checks passed; maximum saved board-coordinate difference was 0.234µm. The original scalar/GPU scorer comparison passed on 52 recorded cases. All 31 executed module hashes and 12 evidence artifact hashes matched. These checks do not independently rerun dynamics or establish real-world robustness.

Recommended next correction: preserve a successful source branch as a local IK continuation seed, generate any additional starts deterministically per query rather than per batch slot, and retain the existing full-pose, joint-limit, endpoint collision and in-flight contact checks. Reconfirm perturbed shots under a separately fingerprinted protocol. More PPO updates alone cannot repair missing endpoint candidates. Never promote table/robot-assisted entry by removing the direct-route guard.

Runtime evidence remains outside Git:

- Source: `~/sircus-mjlab/studies/repeat-diagnostic-v1`
- Evidence: `~/sircus-mjlab/output/repeat-diagnostic-v1`
- `contract.json`, selected `sources.json`, exact132-row `plan.json`, CPU `geometry.npz`, native `shots.jsonl`, seven240Hz `contact-batch-*.npz`, `report.json`, independent `diagnostic-audit.json`.
- Report SHA256: `1f1d03ce23b362c0a3e956b85fd804444222c16966a93f5122c3c05d18d45629`.

Reproduce into a **new** output directory from a copied source snapshot, using the pinned mjlab environment and shared 4090 lock:

```bash
CUDA_VISIBLE_DEVICES=GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4 \
~/sircus-mjlab/venv/bin/python -m sircus.mjlab.repeat_diagnostic \
  --source-study ~/sircus-mjlab/output/expanded-search-v2 \
  --parent-study ~/sircus-mjlab/output/learning-v2 \
  --assets ~/sircus-isaac/assets \
  --output ~/sircus-mjlab/output/repeat-diagnostic-NEW \
  --gpu-lock ~/sircus-isaac/scene.lock
```

This is development diagnosis, with rigid held endpoints and privileged geometry. It is neither fresh held-out policy performance nor learned arm transit or exhaustive tabletop coverage.
