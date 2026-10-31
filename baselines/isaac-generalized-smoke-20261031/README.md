# Development smoke: scene-conditioned PhysX shots

These are small **development-only** execution checks, not the final randomized
learning study and not hardware evidence. Each GPU trained16 throws; eight disjoint
development layouts were evaluated after checkpoint selection. Both GPU runs gave
the same capture counts. Three moved-bucket floor probes passed on each device.
Eight repeated development inputs had zero capture disagreements.

The two main4,096-throw studies use separate train/validation/test seeds and were
still running when this evidence was published. Early v1/v2 diagnostics with stale
static bucket colliders are excluded and must not be presented as valid learning
results. See [methodology](../../sircus/isaac/LEARNING_STUDY.md).
