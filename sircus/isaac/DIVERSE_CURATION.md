# Curating the completed diverse-region pilot

`diverse_pilot_curate.py` is an offline publication step. It starts no simulator,
policy inference, optimizer, service or network connection. Run it only after
the native pilot, direct final-producer setup check, independent audits and
fixed-camera film export have completed and their hashes have been reviewed.

```bash
PYTHONPATH=sircus/isaac python3 sircus/isaac/diverse_pilot_curate.py \
  RUN EXPORT baselines/isaac-diverse-settled-pilot-YYYYMMDD \
  --setup-run DIRECT_FINAL_PRODUCER_SETUP_RUN \
  --setup-audit PASSED_SETUP_AUDIT.json \
  --support-cost ADDITIVE_DISPLAY_COST_RECEIPT.json \
  --report-sha256 REVIEWED_REPORT_SHA256 \
  --native-audit-sha256 REVIEWED_NATIVE_AUDIT_SHA256 \
  --checkpoint-audit-sha256 REVIEWED_CHECKPOINT_AUDIT_SHA256 \
  --setup-audit-sha256 REVIEWED_SETUP_AUDIT_SHA256 \
  --auditor-commit FULL_AUDITOR_COMMIT --exporter-commit FULL_EXPORTER_COMMIT
```

The destination must be new and outside immutable source evidence. The curator
requires 1,600 attempted training shots, 16 audited updates, the three-method
comparison, original camera and unchanged scoring identities. It verifies the
preflight and pilot both actually ran the selected final producer/archive; an
older producer's measurements cannot be relabeled as a new direct preflight.
The default producer is `92c09f4`, with its separately pinned source archive.

The output retains byte-identical initial/final weights and full signed
`support.json`. This geometry artifact contains offline anchor configurations,
IK warm starts and path proxies, and keeps its original fingerprint. Local
anchors and unqualified global exploration remain distinct. No compact summary
is substituted for the loader's original support identity.

Raw physics, paired setup/transit histories, direct native pose readbacks,
images/depth, media, logs, credentials and host runtime configuration are
excluded. Contact/rim diagnostics, omitted frame lists and native observations
are represented by aggregate checks and hashes. The film receipt verifies that
all scheduled chapters and release frames remain; it copies no video.

`manifest.json` checksums every selected output file and binds reviewed source
receipts, native/audit/export source identities and external movie digests. The
generated README includes frozen capture counts, limitations and a CPU prediction
example. Prediction is distinct from native-qualified execution.

Earlier aborted/queued runs may be attached using `--diagnostic-record FILE`;
only their source hash, status, source revision, zero-work counts and scope are
retained. They never contribute PPO samples, evaluation or capture totals.
