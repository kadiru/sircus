# Native Isaac viewport capture — November 1, 2026

[Reproduction and scope](../../sircus/isaac/NATIVE_VIEWPORT.md). A direct X11 recording of the Isaac Sim 5.1 application shows all 32 workcells and 64 arms in one live PhysX world. Four 32-shot batches and two PPO updates completed in a separate demonstration run (seed 20261104). There were 27/128 training captures, seven clean entries, and zero endpoint rejections. These are on-policy training observations, not held-out performance or evidence of improvement.

The MP4 is 58.1 seconds, 2400 × 1000, H.264 at 30 fps, 1,743 frames. No image compositing, added graphics, frame interpolation or speed conversion was applied. All contact/foreign-cell probes, batch membership/count checks, the policy-weight update chain, full video decode and copied-file checksum passed. Four frames across the recording were visually inspected. [Verification metadata](verification.json) includes source/artifact hashes. Raw footage, logs, generated assets and the short demonstration checkpoint remain outside Git.

This is distinct from the earlier presentation-style composite of real Isaac camera renders and does not replace its 2,048-throw pilot results. Board/release XY and bucket height are fixed; heading is prescribed. The policy chooses board height, drop gap and tilt. Arms reset to solved endpoints; pickup/transit/general 3D bucket-placement learning remains unimplemented. Provisional physical properties and bucket height interpretation are unchanged.

The temporary recording container was removed after completion. The existing simulation container, preview and unrelated 4070 workload were preserved.

## Elevated perspective recording

A second direct recording uses a true perspective camera looking diagonally across the farm, with side panels hidden for a larger viewport. The 58.634-second, 2400 × 1000 clip shows all 32 arm pairs. It repeats the same bounded seed/contract: 128 throws, two PPO updates, 27 training captures, seven clean, no endpoint rejection. This is a camera/framing follow-up, not an additional independent learning study. Original orthographic media/evidence is preserved.

[Perspective verification](perspective-verification.json) records the camera parameters, executed-source hashes, four batch checks, passed contact/isolation probes, policy-update chain, complete video decode, copied checksum and four-frame visual review. The temporary GUI container was removed; the existing preview and 4070 workload remained active.

## Low close-up recording

The 58.634-second close-up focuses on cell 2 (the centered near target), with neighboring tables behind it. Camera height is 1.25 m above the table, a 35 mm perspective lens aims at Z 0.45 m, and the physics playback cap is 0.5× (rendering may be slower). Native frames show the ball leaving the release point, the board and the bucket mouth at readable scale. The fixed foreground cell had one miss followed by three captures; it was chosen for framing before recording rather than selected by outcome. This repeats the bounded demonstration and is not an independent test of learning.

[Close-up verification](closeup-verification.json) records all 128 throws, two policy updates, contact/isolation probes, a compact four-shot foreground summary, camera/playback settings, source/media hashes, full decode and six-frame visual inspection. Original recordings remain intact. The temporary GUI container was removed; the existing preview and 4070 workload remained active.
