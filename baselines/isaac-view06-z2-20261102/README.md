# Selected view 06 revision — November 2, 2026

Native Isaac run `camera06-z2-20261102-v1` rendered one 1920 × 1080 still using the selected silver-gray look. View 06 moves 0.30 m along its original horizontal camera-right direction and rises to world Z=2.0 m above tabletop Z=0. Its original look-at point stays fixed. The original ten choices are preserved.

World eye: (11.278317, −6.458747, 2.0) m. World target: (7.65, −4.15, 0.18) m. The featured demonstrated source workcell remains at the right-front display corner using the documented whole-cell permutation. No geometry, source shot or lighting change accompanies this camera revision.

[camera-options.json](camera-options.json) records the pose, adjustment, readback, four capture-source hashes, frozen scene state and native JPEG digest. [verification.json](verification.json) records matching source/image hashes on the Mac, native readback within 0.2 mm and visual inspection. All 143 Isaac tests passed with one Torch-dependent skip on the Mac. The new adjustment checks preserve the aim, original definitions and foreground/background framing.

The finite job waited for the existing learning run's GPU lock, then completed on the 4090. Neither GPU workload was interrupted; existing services, hardware, Viam and audio were untouched. This is a static composition preview, with no new rollout, training or physical-validation claim.

Execution snapshot: `$HOME/sircus-isaac/studies/camera06-z2-20261102-v1`. Native output: `output/camera06-z2-20261102-v1`. The JPEG and full logs remain outside Git. Reproduce with the [camera preview command](../../sircus/isaac/CAMERA_OPTIONS.md), adding `--camera-option 6 --camera-right-m 0.30 --camera-z-m 2.0` and a fresh output directory. The one-minute orbit/close-up recording remains separate, pending review of this revised still.
