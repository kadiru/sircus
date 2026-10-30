# Project state at repository initialization — October29,2026

- Mac: physical sfhack10 hosts arm10/dropper and arm14/board holder. Operator dashboard on localhost8765. Practice cycles use taught poses,2s loading,2s settled release,2s post-release recording. Completed practice records are retained locally unlabeled. Successful panel Open commands can skip redundant loading Open; unknown/failed states do not. Saved hardware poses/credentials are deliberately not in Git.
- Ubuntu: isolated fake-only server8083 and local dashboard8765. New orientation/height scripts from the tested Varun transfer are now canonical in sircus/two-arm. Original transfer archives were not changed.
- Geometry: board200x300x5mm, landscape90° mount; base spacing1337.31mm. Tapered bucket inner177/150mm, outer200/158mm.175mm reported height remains unresolved; provisional simulation depth175mm and8mm base. Mount offsets, masses, support compliance and bucket placement remain provisional.
- Learning: old U/V/gap search has one fixed board pose. New CLI searches roll/pitch/gap at fixed release XY and fixed board/bucket centers. This is not a generalized policy. New schema/fingerprint protects replay; selected candidates require native path validation.
- Perception: camera on arm10 tool bracket; stationary capture works. Camera-to-world calibration and reliable ball/board/bucket localization remain incomplete. No current scene overlay is proof of calibration.
- Tests before repo initialization:34 physical-controller/adapter mock tests;46 two-arm baseline regressions plus orientation contract checks. New training smoke4 candidates/2 synthetic tests and8 successful native fake-arm motion checks on Mac. Linux CI is the next portability check; native Linux Viam runtime remains independently unverified.
- Native cloud simulation runtime previously on the Mac is no longer assumed running. Do not reuse its cloud identity without coordinating ownership. Local8083 needs no credentials.

Current detailed physical/perception STATUS files include historical PIDs and timestamps. Verify live state before acting. Read docs/UBUNTU_SETUP.md for repository startup; old transfer/local absolute paths are historical.

## Mac Mini audio — October 30, 2026

Ghost Booth is the unified local event-sound/rehearsal interface under
`sircus/audio/booth/`, with eight configurable event slots and sixteen original
sounds. See [audio status](../sircus/audio/STATUS.md). It runs independently with
FX off and no robot dependency. The main audio branch/draft PR carries this code;
no physical dashboard hook, Viam configuration or robot deployment is included.
