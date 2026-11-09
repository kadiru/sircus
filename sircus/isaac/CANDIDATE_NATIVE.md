# Frozen seed-22 candidate selector: native TRAIN comparison

[Completed audited results](../../baselines/isaac-candidate-native-20261109/README.md): captures 13→16, clean 4→5, rejections 7→0 across 35 known TRAIN positions; no training.

This separate experiment compares the unchanged seed-20261122 preferred shot
against a finite, TRAIN-informed candidate selector. It makes exactly 70 new
requests: the same 35 original TRAIN bucket positions once per method, original
parent first and candidate-bank selector second. All declines/rejections stay
in the requested-position denominator. It performs no optimizer updates and
never writes to the completed sweep/PPO experiment or its checkpoints.

The [audited candidate bank](CANDIDATE_BANK.md) contains 1,814 qualified exact
settings, including every historical successful paid TRAIN shot. The selector
scores candidates with the frozen parent's Gaussian latent density, normalizes
a new finite categorical distribution, and takes its deterministic argmax.
It uses no ball-flight prediction, outcome ranking, projection, inference-time
shot search, validation or TEST inputs. This changes the decision rule; it is
not an unchanged continuous PPO action distribution.

Twenty-eight qualified parent choices are preserved exactly, with identical
native IK seeding in both cohorts. Seven infeasible original choices are
replaced by qualified candidates. Changed bank choices use their prepared IK
warm starts, while native endpoint, path, measured settling and release guards
remain mandatory. Prepared qualification never guarantees release or capture.

The native runner uses the original 35-table/70-arm scene, assets, materials,
240 Hz physics, camera, scorer, reward calculations and driven setup controller.
Actions and seeds are signed before any new outcomes. The source is archived
from an exact producer commit. The supervisor verifies source/input bytes,
waits for the exclusive scene lock and an idle RTX4090, enforces finite queue
and execution limits, and signals only its own native process if required.
The RTX4070 and other workloads/services remain untouched. A 6 GiB storage
floor applies; evidence is never deleted to continue.

A separate CPU-only audit verifies the fixed request ledger, actions, seeds,
source/model/scene identities, original startup controls, geometry/FK/path
checks, live transit/settling history and full-rate release/flight scoring.
Native completion is distinguished from a passed audit. The separate GET-only
viewer on localhost 18802 exposes final outcomes only after this audit passes;
the frozen benchmark reader on localhost 18797 stays unchanged.

Report captures, clean captures, rejections and paired gains/losses over all
35 requests per cohort. Native wall/CPU include canonical home, setup, flight
and serialization; shared initialization/control probes are separate. RSS is
a lifetime high-water value. Torch CUDA peaks include persistent scene tensors;
sampled whole-device occupancy also includes PhysX/renderer. CPU candidate-bank
preparation and current selection timings are reported separately. The prior
117.003 s preparation-plus-audit measurement is inclusive, not pure audit time.

This is a small, single-seed, familiar-TRAIN-position comparison. Bank inclusion
uses historical successful TRAIN actions, so improvement cannot establish
unseen-position generalization. Scripted transit is not learned motor control,
collision certification, physical validation or a guarantee of every good shot.
No checkpoint promotion or new training is part of this task.
