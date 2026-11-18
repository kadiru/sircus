# Saved-data CPU audit failed: bucket-entry clearance mismatch

The separately approved CPU audit **did not pass**. It exited on November 18,
2026 at 19:40:00 UTC with:

> Native rescored outcome differs: minimum_mouth_clearance_mm

The [original simulator shutdown failure](../isaac-seed22-comparison-shutdown-20261118/README.md)
also remains a failure. There is no successful original final audit, no successful
recovery certificate and no model promotion. Captures, clean captures, rejection
rates, learning curves and trial videos have not been published as verified.

## What ran and what was preserved

The source at `b07d46f` passed CI, 59 mocks inside the original numerical
environment, 35 outer-watchdog mocks and 12 private-launcher mocks. The actual
CPU preflight verified all 25,059 sealed files (4,330,228,832 bytes), 1,764 journal
entries, 101 pinned inputs and 217 immutable producer-source files. These
successful preparation checks do not establish scientific correctness.

One scientific audit child then ran with the original scientific checks and a
separate proof for the failed shutdown. It exited with code 1, without OOM or
forced stop/kill. Both independent supervisors observed safe retirement. No
simulator, optimizer step or new shot ran. No audit retry was started.

| Measured recovery cost | Value |
|---|---:|
| Entire CPU container watchdog wall | 943.763 s |
| Inner audit watchdog wall | 942.735 s |
| Exact audit-child user CPU | 1,216.189 s |
| Exact audit-child system CPU | 892.980 s |
| Exact audit-child peak RSS | 1,215,520,768 bytes (about 1.13 GiB) |
| Visible GPU devices | 0 |

CPU/RSS use `wait4` for the exact child including reaped descendants. Wall and
CPU are different measures and are not added. Limits remained two hours, four
CPU cores, 8 GiB memory and 6 GiB free on the real evidence filesystem.

A separate **hash-only** check after failure confirmed every sealed native file,
all 101 inputs and all 217 producer sources unchanged, and the original native
container still retired. It did not rescore samples, load a model, start a
container or retry the audit. Original evidence, failed receipts and earlier
setup attempts remain preserved. The historical frozen benchmark is unchanged.

User-approved cleanup removed only 911,401,090 bytes of downloaded APT package
files to restore storage headroom. Installed software and environments were
preserved. CPU setup failures and their diagnostics remain separate from this
single scientific attempt.

## What the error establishes

The unchanged reader compares a saved outcome field with a reconstruction from
the recorded full-rate position and force samples. The error identifies the
bucket-mouth clearance field, but omits the trial identity and the two values.
It therefore does not establish whether the difference is numerical, whether a
field is missing, or whether capture/clean labels change.

Static source review found a diagnostic lead: the live producer and independent
reader can subtract cell origins with different arithmetic precision. The
comparison tolerance for this field is 0.0000001 mm, distinct from the 1 mm
clean-entry criterion. This is **not a demonstrated cause** and does not justify
relaxing the check or accepting the results. Original scoring and verification
helpers were not changed.

The next investigation should locate the discrepant saved sample, compare exact
values and dtypes, and determine any impact on capture and clean classifications.
That can use saved data; no new training or shot is needed. No such diagnostic
rescore, changed tolerance, scientific-audit retry or publication was performed
as part of this failure record.

Compact identities and exact measurements are in [summary.json](summary.json).
Raw observations, checkpoint binaries, frames, runtime controls and logs remain
outside Git. The separately prepared publisher and ten-clip encoder remain
blocked because their successful-audit pins are unavailable.
