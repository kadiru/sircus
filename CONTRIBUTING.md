# Shared work across Mac and Ubuntu

Use one clone per computer and separate branches/checkouts for concurrent agents. Do not point Ubuntu at sfhack10 or start multiple hosts under the same Viam cloud part.

## Before work

```bash
git fetch origin
git status --short --branch
# On a clean main checkout:
git switch main
git pull --ff-only origin main
git switch -c codex/yourname-task
```

If the working tree has another task's changes, use another checkout/worktree; do not stash/reset them automatically. Read AGENTS.md and docs/PROJECT_STATE.md. Pulling code does not update a running process; deployment must be explicit, with the hardware idle.

## Before handoff

```bash
scripts/test.sh
python3 scripts/check-repository.py --staged
git diff --check
git diff --cached
# Stage only this task's files; inspect before committing.
git commit -m "Describe the behavior changed"
git push -u origin HEAD
```

Create a PR for integration to main. Share the branch, commit, tests and open issues so other agents can fetch your work. They use main for reviewed work or explicitly fetch your branch when collaborating on an unfinished task. Never force-push a shared branch or silently overwrite another agent's work. Commit/push a coherent checkpoint before ending a completed task; disclose if authentication or checks block synchronization.

## Helper commands

`scripts/sync.sh status` fetches and reports local/remote differences. `scripts/sync.sh pull` performs a fast-forward-only pull on a clean tree with an upstream. `scripts/sync.sh push` refuses a dirty tree and pushes the current commit normally. None stage files, create commits, merge, restart services, or move hardware.

Install local pre-commit/pre-push file checks after cloning:

```bash
git config core.hooksPath .githooks
```

These checks reject obvious credentials and host/runtime artifacts. They supplement review and cannot prove all secrets absent. GitHub CI runs the same tracked-file scan and the mock/numerical checks without robot credentials.

## Sharing access

Owner: kadiru. Each human contributor needs their own GitHub account granted access, and each agent's GitHub/CLI connection must be able to access this private repository. Never copy another user's token. Keep credentials in ignored .private directories and provision physical credentials only on the authorized Mac.
