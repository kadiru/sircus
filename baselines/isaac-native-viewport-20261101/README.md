# Native Isaac viewport capture — November 1, 2026

[Reproduction and scope](../../sircus/isaac/NATIVE_VIEWPORT.md). A direct X11 recording of the Isaac Sim 5.1 application shows all 32 workcells and 64 arms in one live PhysX world. Four 32-shot batches and two PPO updates completed in a separate demonstration run (seed 20261104). There were 27/128 training captures, seven clean entries, and zero endpoint rejections. These are on-policy training observations, not held-out performance or evidence of improvement.

The MP4 is 58.1 seconds, 2400 × 1000, H.264 at 30 fps, 1,743 frames. No image compositing, added graphics, frame interpolation or speed conversion was applied. All contact/foreign-cell probes, batch membership/count checks, the policy-weight update chain, full video decode and copied-file checksum passed. Four frames across the recording were visually inspected. [Verification metadata](verification.json) includes source/artifact hashes. Raw footage, logs, generated assets and the short demonstration checkpoint remain outside Git.

This is distinct from the earlier presentation-style composite of real Isaac camera renders and does not replace its 2,048-throw pilot results. Board/release XY and bucket height are fixed; heading is prescribed. The policy chooses board height, drop gap and tilt. Arms reset to solved endpoints; pickup/transit/general 3D bucket-placement learning remains unimplemented. Provisional physical properties and bucket height interpretation are unchanged.

The temporary recording container was removed after completion. The existing simulation container, preview and unrelated 4070 workload were preserved.
