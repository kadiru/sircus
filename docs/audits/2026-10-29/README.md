# Mac implementation audit — October 29, 2026

Read these before proposing or implementing the next learning experiment:

1. [System architecture and implementation inventory](MAC_SIRCUS_SYSTEM_AUDIT.md) — services, shared/duplicated code, geometry, sensing, physics and historical results.
2. [UI and API contracts](MAC_UI_API_CONTRACTS.md) — controls, routes, payloads, state machines, environment boundaries and side effects.
3. [Learning integration plan](MAC_LEARNING_INTEGRATION_PLAN.md) — proposed randomized-scene study, reusable components, missing interfaces and acceptance criteria.

The system report includes the sensor/assets and detailed learning audit appendices. These are historical findings at source commit `6e1fbf1`, not live health checks. In particular, the audit's empty-remote/permission-blocker observations predate the successful repository publication and Ubuntu CI run linked in each document.

This repository edition excludes credential files, tokens, raw recordings/trials, cloud configuration dumps, screenshots and exact taught calibration vectors. References to local evidence identify provenance only. The original reports and sanitized snapshot remain under the Mac's ignored `handoff/` directory.

No application behavior, robot configuration, calibration, or runtime is changed by publishing these documents. The learning plan is a proposal, not an implemented capability.
