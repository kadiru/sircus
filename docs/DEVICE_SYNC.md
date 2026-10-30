# Device source synchronization

GitHub is the shared source of truth. Fetching does not merge drafts or deploy a
runtime. Main includes reviewed PR3 at a376ad67968f6445d39a51c2a29325a204bb582a;
fetch actual latest main instead of assuming that commit stays latest.

## Send to both agents

> Read README.md, CONTRIBUTING.md and docs/PROJECT_STATE.md in the persistent
> kadiru/sircus clone. Inspect status/worktrees and fetch origin. Fast-forward an
> unused clean main checkout to actual origin/main; if it has local changes or
> serves a running process, use a separate persistent worktree instead. Never
> auto-stash, reset, clean or overwrite local evidence/runtime data. Keep audio
> PR1 and PR4 draft and separate. Source synchronization is not deployment.
> Do not restart services, enable audio, change Viam/cloud configuration, or
> command hardware. Report checkout, branch, HEAD, origin/main and cleanliness.

## Ubuntu: new operator/training work

Create a separate persistent task worktree from
`origin/codex/macbook-guided-training` (use a unique branch name if this exists):

```bash
git fetch origin
git worktree add -b codex/ubuntu-guided-trials ../sircus-operator origin/codex/macbook-guided-training
cd ../sircus-operator
./setup-ubuntu.sh
./start-operator-sim.sh
```

Open http://127.0.0.1:18766/operations.html. This is a new local-only process with
no cloud identity. Preserve sircus-ubuntu-baseline and all of its services/data.
For code synchronization alone, stop before the setup/start commands. Run project
checks before new work. Do not copy older baseline source over the shared checkout.

## Mac mini: audio source

Fetch `origin/codex/mac-mini-oddball-sfx` for the Ghost Booth fix and
`origin/codex/macbook-dashboard-audio` for the stacked dashboard integration. Use
an unused clean task checkout or a new worktree; do not change the source beneath
the currently running Ghost Booth. Read `sircus/audio/MACBOOK_INTEGRATION.md` and
`sircus/audio/booth/integration/HANDOFF.md`. The service and client control protocol
must be deployed together in a separately coordinated session; new ON commands
fail closed against older services. Keep the running audio process untouched.

The new operator workspace branches from reviewed main and deliberately does not
merge the audio drafts. All branches are available to agents through GitHub, but
separate workstreams do not need identical working directories or runtime versions.
