# Fixed-angle actual PPO training — November 2, 2026

Native Isaac Sim 5.1 / Isaac Lab 2.3.2 on RTX 4090, using the user's exact camera and silver-gray appearance. All ten fresh training batches are recorded; their 350 sampled slot actions feed five PPO updates. This is a bounded filming pass, not a held-out evaluation or a promoted general policy. Joint travel and ball loading remain scripted.

The geometry-only menu has qualified options at 17 of the 35 distinct grid sites; the other 18 remain parked and release-disabled. All 350 attempted slots remain in PPO, including invalid penalties. Of 170 enabled slots, 156 actually released and 14 failed current endpoint/settling checks. There were 30 qualified captures, 28 clean. The foreground table made ten board rebounds: two captures, five table outcomes and three measured floor arrivals.

## Video and timing

| File, outside Git | Frames | Duration | SHA256 |
|---|---:|---:|---|
| `action-training.mp4` | 700 | 23.33 s | `2cf1503f9f5517d5d2374d9b3c188a571d229547482b32c7d35a7d2a08dcb702` |
| `action-training-untrimmed.mp4` | 1570 | 52.33 s | `24ab1899687f08cc6352256ab4f6e3a4e418c411074bad7171821d9846378af0` |

Both are 1080p/30 fps native RGB from one fixed view, with no ball following. The action edit retains every batch's 36 measured setup frames, release, rebound and immediate outcome. Cuts end 0.20–0.225 seconds after table/floor contact or the final successful entry interval. Rim contact is not treated as a terminal miss. Full one-second retention scoring is reconstructed independently from 240 Hz telemetry, even when its stationary tail is removed visually.

The complete captured sequence preserves every recorded frame, not all 271.37 seconds of wall time. Settling, IK, optimizer waits and camera capture overhead were not filmed. Ball-flight sampling is 1× physical time; the 36 setup samples compress joint travel. The repeated foreground configuration in batch eight has little motion there while background cells change. It is not represented as learned joint control.

## Evidence and checkpoint

- [result.json](result.json): native counts, five update links, source hashes, stage and camera.
- [film.json](film.json): every raw/edit interval, terminal reason, measured foreground joint excursion, background contact coverage and movie hashes.
- [checkpoint-audit.json](checkpoint-audit.json): CPU reload of all sampling policies and recomputation of all 350 latent log probabilities; maximum absolute error `9.5367431640625e-7`. Eleven checkpoint and ten clip digests are bound. This does not independently reconstruct optimizer updates or establish held-out performance.
- [policy-00.pt](policy-00.pt) and [policy-final.pt](policy-final.pt): initial and experimental final checkpoints. Load with `torch.load(..., weights_only=True, map_location="cpu")`; the final file includes the unchanged source task contract. Do not promote it on the basis of this video.
- [menu-provenance.json](menu-provenance.json), [probes.json](probes.json), [rim-rays.json](rim-rays.json), [selected-camera.json](selected-camera.json), [verification.json](verification.json): geometry-only menu lineage, fresh native contact/isolation checks, original camera bytes and test/transfer evidence. All 88 captured Python source hashes match the final source. All ten batches' native setup/release/contact/outcome samples were visually inspected.

Local suite: 182 tests, seven Torch-dependent skips; all six new checkpoint tests passed on Ubuntu CPU. The actual run then passed CPU policy/likelihood, native contact/camera/scoring, media decode and transferred-hash checks. The finite capture and exporter exited successfully. The 4090 was idle afterward; the 4070 workload, existing services and camera scout were untouched. No hardware, Viam or audio work occurred.

The incomplete v2 capture stopped before its eighth release because a tiny joint residual stalled relative camera sampling. It remains preserved outside Git. The completed v3 run starts fresh; it reuses no v2 frames or selected outcomes. Its training-only camera schedule follows interpolation time while joint readiness and final release guards remain unchanged. Historical replay schedules, geometry/provisional measurements and prior studies are preserved.

Raw run: `output/action-training-20261102-v3`; media: `output/action-training-20261102-v3-export`; immutable source: `studies/action-training-20261102-v3`. All are under the configured Ubuntu workspace. [Method, limitations and commands](../../sircus/isaac/ACTION_TRAINING_FILM.md).
