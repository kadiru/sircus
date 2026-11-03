# Fixed-bucket gap-window/SAC study

Status at launch: native anchor controls passed3/3; finite comparison running.
No final learning/test result is claimed in this launch record.

- [Protocol](../../sircus/mjlab/WINDOW_STUDY.md), opt-in version
  `sircus-fixed-bucket-gap-study-v1`.
- Immutable execution source on Ubuntu: `~/sircus-mjlab/studies/window-sac-v2`.
  Evidence: `~/sircus-mjlab/output/window-sac-v1`.
- Source v1 was a CPU-only packaging preflight missing the audit scene fixture;
  v2 includes it and passed all59 new tests on pinned Ubuntu CPU before launch.
  Source v1 never launched native dynamics. No prior experiment was overwritten.
- [CPU preflight](cpu-preflight.json):23 isolated geometry/learning tests passed,
  exact anchor and ±10 mm gaps admitted; both full flange poses verified, fixed
  board branch preserved. This checks geometry, not flight or robot tracking.
- Existing viewer PID186578 remains unchanged. New static panel `#windowStudy`
  and derived-report updates do not change active inference. Supervisor PID928174
  has a bounded child job and independent CPU audit; identifiers are historical,
  not evidence that a process remains live.
- Finite pipeline uses the shared4090/Isaac lock, waits rather than interrupting
  work, leaves4070 untouched, and preserves raw arrays/logs outside Git.

Final reports and selected checkpoints will be curated only after completion and
independent rescoring. Historical corrected-grid/rectangular/bucket baselines
retain their original definitions and evidence.
