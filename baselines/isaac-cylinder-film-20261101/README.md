# 35-table Isaac Sim training film

This simulation used five rows of seven two-arm workcells on an RTX 4090. Each bucket occupied a different tile of the shared tabletop grid. Each board center lay on a 400 mm horizontal radius around its own bucket. Isaac Sim 5.1.0 advanced PhysX at 240 Hz while the perspective RTX camera recorded ten PPO rollout batches. [Scene, camera path and reproduction](../../sircus/isaac/CYLINDER_FILM.md)

## Training evidence

| Check | Result |
|---|---|
| Rim contact attribution | Passed in 35/35 cells. Bucket, table, floor, robot and neighboring-cell isolation checks also passed. |
| Two-arm endpoint screening | 154 of 2,100 candidates passed IK and settling across 17/35 cells. Screening used no ball outcomes. |
| Training | 350 attempts, 170 releases and 32 qualified captures across 13 cells. Five PPO updates. |
| Endpoint rejections | 126 dropper and 54 board-arm IK failures. All counted in the 350 attempts. |
| Below-rim shots | Zero releases with settled board center below the modeled 183 mm rim. All 35 cells screened 150 mm board height at 15 azimuths. None of those postures passed both arms' checks. |

All 32 captures met the clean, single-board-bounce criterion and retained the whole ball for one continuous second. These are on-policy training results. A frozen-checkpoint test on new bucket positions remains open.

The front-row shots were selected from settled release poses before the balls fell. In the first shot (row 1, column 3) the ball bounced and stayed in the bucket. In the second (row 1, column 5) it bounced and missed. These are the two quarter-speed sequences at 22.5–32.1 s of the full camera flight.

## Checkpoint and provenance

- [cylinder-policy.pt](cylinder-policy.pt) contains the final policy state and scene contract. CPU PyTorch loaded it with `weights_only=True` and strict parameter matching. Its restored weights matched the final PPO hash.
- [menu.json](menu.json) is the exact per-cell geometry menu required to decode policy proposals. Its digest matched both the checkpoint and run record. The serialized checkpoint contract matched the recorded contract after JSON normalization.
- [result.json](result.json) contains frame counts, per-batch outcomes, the five weight updates and the source manifest. All 67 recorded Python source hashes matched the capture checkout. Export runs separately.
- [film.json](film.json) records chapter times, the two front-row outcomes and SHA-256 digests for the full film and the close-up cut. Both 1080p H.264 exports decoded at 30 fps and matched the digests after transfer to the Mac.

| SHA-256 identity | Digest |
|---|---|
| Checkpoint file | `eee598474bac29b4dcc3ad17426665e4aace0d7073c6aada447eff118d14e459` |
| Reloaded final weights | `5a510b0c991c1233d3fc9c528d208c1b01fbb458959bc212eed76149d3fe675e` |
| Geometry menu file | `bb4c4349089b50a791ddb9c6ab1cd479494bd63699bba97ccc93e357d67a808d` |
| Scene contract | `98ebe47b48a01b9028747ab03bc913a033d90611e44f03b912e58f41c87ff79a` |
