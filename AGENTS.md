# Sircus agent workflow

This repository is shared by Mac hardware and Ubuntu simulation contributors.

- Start by reading README.md, CONTRIBUTING.md and docs/PROJECT_STATE.md. Fetch origin and inspect status before editing. Do not overwrite unrelated changes.
- Use a separate checkout/worktree and `codex/<owner>-<task>` branch for concurrent work. The GitHub repo is the source of truth for pushed code; frozen transfer archives are historical.
- At a completed task boundary, run relevant tests, review and commit only your changes, then push the task branch and report the commit. Do not force-push or merge someone else's work without authorization. If push is blocked, state what remains local.
- Keep credentials, raw observations, host calibration, generated runtime configs, queues, virtualenvs and logs out of Git. Use explicit staging, never blind git add -A. Run scripts/check-repository.py before committing.
- Default to mocks/fake-only local simulation. sfhack10 physical commands and deployment are separate from code synchronization. Never move arms, actuate tools, restart the physical dashboard, change Viam configs, or start the same cloud part on another host merely to test/pull code. Require task authorization for hardware work.
- Preserve measured vs provisional geometry and original scene fingerprints. Synthetic results remain labeled synthetic. Do not replace or relabel old rectangular/bucket baselines.
- The orientation/height CLI is fixed-scene optimization with endpoint IK; selected native paths need separate replay. Do not claim full dynamics, generalization or physical validation.
- Read scoped README/STATUS files as historical evidence, not proof a PID or calibration is current. Check actual runtime state before any authorized deployment.
- Do not send messages/invite collaborators unless the user explicitly asks. Repository access is per account and is not implied by an agent's existence.
